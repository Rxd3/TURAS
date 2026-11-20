"""Publish aggregate reports and prepare a private, gold-free model/run handoff."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tarfile

from turkish_asr_rdl.training.acoustic import sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--evaluation',required=True,type=Path)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('handoff output must be fresh')
    p=json.loads(args.protocol.read_text());summary=json.loads((args.evaluation/'summary.json').read_text())
    if summary['status']!='final_frozen_working_benchmark_complete' or summary['protocol_sha256']!=sha256_file(args.protocol):
        raise ValueError('completed evaluation does not match the frozen protocol')
    args.output.mkdir(parents=True,exist_ok=False)
    public=Path('results/final');public.mkdir(parents=True,exist_ok=True)
    if (public/'m45_test_summary.json').exists():raise ValueError('final public report already exists; never overwrite silently')
    write_json(public/'m45_test_summary.json',summary)
    systems={r['system_id']:r for r in summary['systems']}
    timings={r['system_id']:r for r in summary['headline_timing']}
    fields=['system','raw_wer_pct','normalized_wer_pct','raw_cer_pct','normalized_cer_pct',
            'term_precision_pct','term_recall_pct','term_f1_pct','quantity_diagnostic_f1_pct','end_to_end_rtf','rtfx']
    with (public/'m45_metrics.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for system,r in systems.items():
            raw=r['metrics']['raw-v1'];norm=r['metrics']['tr-normalized-v1'];term=r['domain']['micro'];timing=timings.get(system,{})
            writer.writerow(dict(zip(fields,[system,100*raw['wer'],100*norm['wer'],100*raw['cer'],100*norm['cer'],
                100*term['precision'],100*term['recall'],100*term['f1'],100*r['domain']['quantity_diagnostics']['micro']['f1'],
                timing.get('end_to_end_rtf'),timing.get('rtfx')])))
    lines=['# TURAS M4/M5 final working benchmark','',
        f"Frozen protocol SHA-256: `{summary['protocol_sha256']}`.",
        f"Test: {summary['test_utterances']} clips, {summary['recording_groups']} recording groups; reference `{summary['reference_variant']}`.",
        'The model choices and all conditions were frozen before test transcript access. All results are one training seed (17).','',
        '## Main acoustic comparisons','',
        '| System | Raw WER | Normalized WER | Normalized CER | Term micro F1 | End-to-end RTF |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for spec in p['systems']:
        if spec['kind']=='relational':continue
        name=spec['system_id'];r=systems[name];n=r['metrics']['tr-normalized-v1'];raw=r['metrics']['raw-v1'];t=timings[name]
        lines.append(f"| {name} | {100*raw['wer']:.3f}% | {100*n['wer']:.3f}% | {100*n['cer']:.3f}% | {100*r['domain']['micro']['f1']:.3f}% | {t['end_to_end_rtf']:.4f} |")
    lines+=['','CTC adaptation includes an explicit train-only vocabulary extension (38 to 52 tokens). The gain is not attributed to fine-tuning alone.','',
        '## Fixed-candidate RDL and ablations','',
        '| Adapted ASR | Original one-best WER | Acoustic reordering | Learned full graph | Without quantities | Without co-mentions | Untrained graph |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for native in p['adapted_system_ids']:
        values=[systems[native]['metrics']['tr-normalized-v1']['wer']]
        for suffix in ['acoustic-control','rdl-full','rdl-no-quantities','rdl-no-comention','rdl-random']:
            values.append(systems[native+'--'+suffix]['metrics']['tr-normalized-v1']['wer'])
        lines.append('| '+native+' | '+' | '.join(f'{100*v:.3f}%' for v in values)+' |')
    lines+=['',
        'All graph/control rows use the same native N-best file per acoustic model. Positive RDL weights were selected on validation, with the same positive weight used in its relation-removal and untrained controls. Gamma zero is a separate acoustic reordering control. The primary model choices remain the validation-selected original decoder outputs; no final model was chosen from test scores.',
        'Cached candidate selections produce contract-valid predictions and component scores. Their measured cache-only latency is excluded from headline ASR RTF. Joint learned-RDL timing is measured separately with fresh acoustic inference.','',
        '## Repeated timing','',
        f"Batch 1, three repeats, {p['timing']['utterances']} frozen timing clips totaling {p['timing']['duration_s']:.3f} seconds per repeat, across {p['timing']['recording_groups']} recording groups. Three train warm-up clips are excluded. OS caches were not flushed; inputs and weights were local. Accuracy runs may be concurrent and are excluded from headline speed claims.",
        '', '| System | End-to-end RTF | RTFx | Per-repeat RTF | p50 / p95 request seconds | Peak reserved GiB | Thermal flags |',
        '| --- | ---: | ---: | --- | --- | ---: | ---: |']
    for t in summary['headline_timing']:
        values=', '.join(f'{v:.4f}' for v in t['per_repeat_end_to_end_rtf'])
        lines.append(f"| {t['system_id']} | {t['end_to_end_rtf']:.4f} | {t['rtfx']:.2f} | {values} | {t['latency_p50_s']:.3f} / {t['latency_p95_s']:.3f} | {t['peak_reserved_gib']:.3f} | {t['thermal_flags']} |")
    lines+=['','Any spill/thermal flags are retained in the JSON; timings are hardware-specific. Cold-load time and secondary core-model RTF are stored per system.','',
        '## Paired uncertainty','',
        f"Recording-group percentile bootstrap: {summary['bootstrap']['replicates']} paired replicates, seed {summary['bootstrap']['seed']}; {summary['recording_groups']} groups. Differences below are normalized WER percentage points, system minus control.",
        '', '| System | Control | Difference | 95% interval |','| --- | --- | ---: | --- |']
    for b in summary['bootstrap']['comparisons']:
        lo,hi=b['paired_recording_bootstrap_95_interval_percentage_points']
        lines.append(f"| {b['system']} | {b['control']} | {b['normalized_wer_difference_percentage_points']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines+=['','## Reference quality and limitations','',
        f"M1-reported status counts: `{json.dumps(summary['verification_status_counts'],sort_keys=True)}`. Every system is also scored on the M1-reported listening-verified subset in the JSON, with its actual support. That small subset is not a substitute for the full test set.",'']
    lines+=['- '+text for text in summary['limitations']]
    lines+=['','## Reproduction and handoff','',
        'The exact checkpoints, parents, graph/linker hashes, timing IDs, reference hashes and comparison conditions are in `configs/evaluation/m45_final_protocol.json`. The run index links every prediction/N-best/timing and score artifact. Environment versions are pinned in `requirements/owner-m45-windows-cu130.txt`; install that lock and the editable package in a fresh compatible environment. Run the included commands from the repository root and use fresh output folders.',
        'Private model/run artifacts are kept outside Git. The private handoff includes selected weights/processors, corrected graphs/control, all prediction/timing artifacts and an exact source Git bundle. It excludes reference transcripts and audio. Obtain the separately hashed M1 audio bundle and evaluator-only references through their proper private channel.',
        'No independent reviewer approval, consumer acknowledgment or completion of other members\' baseline tracks is asserted.']
    report='\n'.join(lines)+'\n'
    (public/'M4_M5_FINAL_REPORT.md').write_text(report,encoding='utf-8')
    bundle=args.output/'TURAS-M4-M5-source.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'feat/m4-m5-integration'],check=True)
    sourcepaths=[]
    for spec in p['systems']:
        if spec['kind']=='adapted':sourcepaths.append(Path(spec['checkpoint']))
    sourcepaths+=list({Path(c['graph']) for c in p['rescore_conditions']})
    sourcepaths+=[args.run_root/'accuracy',args.run_root/'rescore',args.run_root/'timing',args.evaluation]
    files={}
    for directory in sourcepaths:
        for path in sorted(directory.rglob('*')):
            if path.is_file():files[path.as_posix()]={'bytes':path.stat().st_size,'sha256':sha256_file(path)}
    write_json(args.output/'artifact-index.json',{'protocol_sha256':sha256_file(args.protocol),'files':files,
        'source_bundle_sha256':sha256_file(bundle),'reference_transcripts_included':False,'audio_included':False})
    readme='TURAS M4/M5 private model/run handoff\n\nIncludes actual selected checkpoints, graph/control artifacts, predictions, N-best, timing and evaluation reports.\nReference transcripts and audio are deliberately separate. Do not upload this archive to the public repository.\n\nUse the source Git bundle or GitHub integration branch, the pinned runtime lock, and frozen protocol.\nLoRA needs the exact openai/whisper-small parent listed in its checkpoint manifest; baseline parents are pinned in the protocol.\nThe full report and JSON explicitly disclose edited minutes, unknown speakers, single seed, and independent-review status.\n'
    (args.output/'README.txt').write_text(readme,encoding='utf-8')
    archive=args.output/'TURAS-M4-M5-private-handoff.tar'
    with tarfile.open(archive,'w') as tar:
        for name in sorted(files):tar.add(name,arcname=name,recursive=False)
        for path in [bundle,args.output/'artifact-index.json',args.output/'README.txt',args.protocol]:
            tar.add(path,arcname=path.name if path.parent==args.output else path.as_posix(),recursive=False)
        for path in public.iterdir():
            if path.is_file():tar.add(path,arcname=path.as_posix(),recursive=False)
    write_json(args.output/'handoff.json',{'status':'private_handoff_ready','archive':archive.as_posix(),
        'bytes':archive.stat().st_size,'sha256':sha256_file(archive),'protocol_sha256':sha256_file(args.protocol),
        'recipient_acknowledgment':'not recorded; ready for owner delivery',
        'public_report':(public/'M4_M5_FINAL_REPORT.md').as_posix()})
    print(json.dumps({'status':'private_handoff_ready','archive':archive.as_posix(),'sha256':sha256_file(archive)}))


if __name__=='__main__':main()
