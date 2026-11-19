"""Audit effective decoder widths from frozen code and all emitted N-best files."""
import argparse
from collections import Counter
import json
from pathlib import Path

from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/owner-m45/final'))
    parser.add_argument('--output', type=Path, default=Path('results/final/m45_decoding_provenance_correction.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('correction output must be fresh')
    protocol = Path('configs/evaluation/m45_final_protocol.json')
    p = json.loads(protocol.read_text())
    index = json.loads((args.root/'inference-index.json').read_text())
    if sha256_file(protocol) != index['protocol_sha256']:
        raise ValueError('original frozen protocol changed')
    source_paths = ['src/turkish_asr_rdl/decoding/whisper.py', 'src/turkish_asr_rdl/decoding/adapted.py',
                    'src/turkish_asr_rdl/decoding/ctc.py']
    for name in source_paths:
        if sha256_file(Path(name)) != p['critical_code_sha256'][name]:
            raise ValueError('decoder code changed from the actual frozen implementation')
    systems = {s['system_id']:s for s in p['systems']}
    conditions = {s['system_id']:s for s in p['rescore_conditions']}
    reports = []
    totals = {'whisper':Counter(), 'ctc':Counter()}
    for entry in index['runs']:
        spec = systems.get(entry['system_id']) or conditions[entry['system_id']]
        base = systems[spec['base_system']] if 'base_system' in spec else spec
        family = base['family']
        key = 'num_beams' if family == 'whisper' else 'beam_size'
        expected = p['decoder']['requested_n'] if family == 'whisper' else p['decoder']['beam_size']
        directory = Path(entry['directory'])
        done = directory/'completed.json'
        if sha256_file(done) != entry['completed_sha256']:
            raise ValueError('original completed run record changed')
        nbest = directory/'nbest.jsonl'
        if sha256_file(nbest) != json.loads(done.read_text())['files']['nbest.jsonl']:
            raise ValueError('original N-best bytes changed')
        rows = read_jsonl(nbest)
        observed = Counter(c['score_details'][key] for r in rows for c in r['candidates'])
        if not observed or set(observed) != {expected}:
            raise ValueError('effective beam evidence disagrees with frozen implementation')
        totals[family].update(observed)
        reports.append({'system_id':entry['system_id'], 'mode':entry['mode'], 'family':family,
            'utterances':len(rows), 'candidate_evidence_field':'score_details.'+key,
            'observed_effective_beam_counts':dict(observed), 'nbest_sha256':sha256_file(nbest)})
    result = {'schema_version':'0.1', 'correction_version':'m45-effective-decoding-v1',
        'status':'verified_all_frozen_exports', 'authority':'owner-authorized factual provenance correction, D41',
        'original_protocol_sha256':sha256_file(protocol), 'original_generic_beam_size_field':p['decoder']['beam_size'],
        'effective_decoding':{'whisper':{'num_beams':5, 'requested_n':5},
                              'ctc':{'prefix_beam_size':10, 'requested_n':5}},
        'cause':'The frozen adapter forwards beam_size only to CTC. Whisper uses num_beams=requested_n; candidate score_details already recorded this correctly.',
        'consequence':'Prior prose describing every system as beam 10 is corrected. M2 beam-5 Whisper settings match the measured Whisper width. Parent/adapted comparisons remain matched.',
        'frozen_model_code_sha256':{name:p['critical_code_sha256'][name] for name in source_paths},
        'audit_code_sha256':sha256_file(Path(__file__)), 'runs_checked':len(reports),
        'candidate_evidence_totals':{k:dict(v) for k,v in totals.items()}, 'runs':reports,
        'models_predictions_protocol_and_metrics_changed':False, 'references_read':False}
    write_json(args.output,result)
    print(json.dumps({'status':result['status'],'runs':len(reports),'effective_decoding':result['effective_decoding']}))


if __name__ == '__main__':
    main()
