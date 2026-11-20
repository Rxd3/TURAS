"""Score the predeclared RDL fusion grid on fixed validation candidates, untimed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from turkish_asr_rdl.data.recover_validation import RELEASE, SPLIT_HASH, VARIANT
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1
from turkish_asr_rdl.rdl.rescore import run
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nbest', 'graph', 'reference-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-unreviewed-minutes', action='store_true')
    args = parser.parse_args()
    if not args.allow_unreviewed_minutes:
        parser.error('explicit recovered-minutes protocol flag is required (D35)')
    if args.output.exists():
        parser.error('output must be new')
    if not args.output.resolve().is_relative_to((Path.cwd() / 'artifacts').resolve()):
        parser.error('private N-best outputs must stay under ignored artifacts/')
    reference_file = args.reference_dir / 'references.validation.jsonl'
    reference_manifest = args.reference_dir / 'reference-manifest.json'
    evidence = json.loads(reference_manifest.read_text(encoding='utf-8'))
    if (evidence['reference_variant'] != VARIANT or evidence['split_manifest_sha256'] != SPLIT_HASH
            or evidence['status'] != 'recovered_upstream_minutes_not_m1_corrected_bundle'
            or sha256_file(reference_file) != evidence['references_sha256']):
        raise ValueError('recovered-reference identity/hash mismatch')
    references = read_jsonl(reference_file)
    ids_path = Path('data/splits') / RELEASE / 'validation.txt'
    ids = ids_path.read_text(encoding='utf-8').splitlines()
    if (sha256_file(ids_path) != evidence['validation_ids_sha256']
            or [r['utterance_id'] for r in references] != ids
            or any(r['reference_variant'] != VARIANT or r['verification_status'] != 'unreviewed' for r in references)):
        raise ValueError('recovered-reference coverage/status mismatch')
    protocol = Path('configs/decoding/rdl_rescoring.json')
    base = json.loads(protocol.read_text(encoding='utf-8'))
    if base['dataset_release'] != RELEASE or base['split_manifest_sha256'] != SPLIT_HASH or base['split'] != 'validation':
        raise ValueError('rescoring protocol release/split mismatch')
    if base['fusion']['lm_weight'] != 0 or 0.0 not in base['validation_search']['rdl_weight']:
        raise ValueError('the declared grid needs an acoustic-only control and disabled LM')
    base['inputs'].update(nbest=args.nbest.as_posix(), nbest_sha256=sha256_file(args.nbest),
                          graph_artifact=args.graph.as_posix(),
                          graph_manifest_sha256=sha256_file(args.graph / 'artifact-manifest.json'))
    args.output.mkdir(parents=True, exist_ok=False)
    trials, control_ids = [], None
    for gamma in sorted(base['validation_search']['rdl_weight']):
        config = json.loads(json.dumps(base))
        config['fusion']['rdl_weight'] = gamma
        config['status'] = 'working recovered-minutes validation; no final test or timing claim'
        prefix = args.output / ('gamma-' + str(gamma))
        write_json(Path(str(prefix) + '.config.json'), config)
        target = Path(str(prefix) + '.nbest.jsonl')
        sidecar = run(config, target)
        rows = read_jsonl(target)
        selected = []
        selected_ids = []
        for row in rows:
            candidate = next(c for c in row['candidates'] if c['candidate_id'] == row['selected_candidate_id'])
            selected_ids.append(candidate['candidate_id'])
            # Evaluator-only in-memory join rows; no timed prediction artifact
            # or false combined RTF is created by offline candidate selection.
            selected.append({k: row[k] for k in ('utterance_id', 'audio_sha256', 'dataset_release', 'split_manifest_sha256')}
                            | {'status': 'ok' if candidate['text'] else 'empty', 'hypothesis_text': candidate['text']})
        joined = join(selected, references, split='validation', split_manifest_sha256=SPLIT_HASH)
        rates = {view: asdict(corpus_rates([score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status)
                                           for r in joined])) for view in (RAW_V1, TR_NORMALIZED_V1)}
        if gamma == 0:
            control_ids = selected_ids
        item = {'rdl_weight': gamma, 'fusion': config['fusion'], 'metrics': rates,
                'changed_selections_from_acoustic_control': sum(a != b for a, b in zip(selected_ids, control_ids)),
                'nbest_file': target.name, 'nbest_sha256': sidecar['output_sha256'],
                'run_sidecar_sha256': sha256_file(Path(str(target) + '.run.json'))}
        trials.append(item)
        write_json(args.output / 'grid-progress.json', {'status': 'running', 'trials': trials})
        print(json.dumps({'gamma': gamma, 'normalized_wer': rates[TR_NORMALIZED_V1]['wer'],
                          'changed_selections': item['changed_selections_from_acoustic_control']}), flush=True)
    best = min(trials, key=lambda t: (t['metrics'][TR_NORMALIZED_V1]['wer'],
                                     t['metrics'][RAW_V1]['wer'], t['rdl_weight']))
    report = {'status': 'working_validation_fusion_selected', 'reference_variant': VARIANT,
        'dataset_release': RELEASE, 'split_manifest_sha256': SPLIT_HASH,
        'reference_sha256': sha256_file(reference_file), 'reference_manifest_sha256': sha256_file(reference_manifest),
        'source_nbest_sha256': base['inputs']['nbest_sha256'],
        'graph_manifest_sha256': base['inputs']['graph_manifest_sha256'],
        'protocol_sha256': sha256_file(protocol), 'source_script_sha256': sha256_file(Path(__file__)),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'trials': trials, 'selected_rdl_weight': best['rdl_weight'],
        'criterion': 'lowest normalized WER, then raw WER, then smaller gamma', 'fusion_frozen_for_test': False,
        'limitations': ['Working validation on recovered edited minutes, not original M1 corrections',
                       'Same candidates for every gamma; acoustic control is original-log-score/length rescoring',
                       'Untimed offline selection; no end-to-end ASR RTF measured',
                       'No final test evaluation performed']}
    write_json(args.output / 'selection.json', report)


if __name__ == '__main__':
    main()
