"""Score validation predictions with M6's evaluator and an explicit reference view."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from turkish_asr_rdl.data.recover_validation import RELEASE, SPLIT_HASH, VARIANT
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import corpus_rates, score_pair
from turkish_asr_rdl.evaluation.normalize import RAW_V1, TR_NORMALIZED_V1
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-unreviewed-minutes', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    if not args.allow_unreviewed_minutes:
        parser.error('this preliminary reference view requires --allow-unreviewed-minutes (owner D35)')
    reference_path = args.reference_dir / 'references.validation.jsonl'
    reference_manifest = args.reference_dir / 'reference-manifest.json'
    evidence = json.loads(reference_manifest.read_text(encoding='utf-8'))
    if (evidence.get('status') != 'recovered_upstream_minutes_not_m1_corrected_bundle'
            or evidence.get('reference_variant') != VARIANT
            or evidence.get('split_manifest_sha256') != SPLIT_HASH
            or sha256_file(reference_path) != evidence.get('references_sha256')):
        raise ValueError('reference manifest/hash/version mismatch')
    ids_path = Path('data/splits') / RELEASE / 'validation.txt'
    if sha256_file(ids_path) != evidence['validation_ids_sha256']:
        raise ValueError('frozen validation ID hash mismatch')
    ids = ids_path.read_text(encoding='utf-8').splitlines()
    predictions, references = read_jsonl(args.predictions), read_jsonl(reference_path)
    if not validate_predictions(predictions, ids).ok:
        raise ValueError('prediction schema/ID validation failed')
    if ([r['utterance_id'] for r in references] != ids
            or any(r['reference_variant'] != VARIANT or r['verification_status'] != 'unreviewed' for r in references)):
        raise ValueError('unexpected recovered-reference content')
    joined = join(predictions, references, split='validation', split_manifest_sha256=SPLIT_HASH)
    metrics = {}
    for view in (RAW_V1, TR_NORMALIZED_V1):
        rates = corpus_rates([score_pair(r.utterance_id, r.reference, r.hypothesis, view, r.status) for r in joined])
        metrics[view] = {**asdict(rates), 'reference_words': rates.words.ref_len, 'reference_characters': rates.chars.ref_len}
    from turkish_asr_rdl.evaluation import align, normalize, metrics as metrics_module
    report = {'status': 'scored_preliminary_validation_minutes', 'dataset_release': RELEASE,
        'split': 'validation', 'split_manifest_sha256': SPLIT_HASH, 'reference_variant': VARIANT,
        'reference_sha256': sha256_file(reference_path), 'reference_manifest_sha256': sha256_file(reference_manifest),
        'predictions_sha256': sha256_file(args.predictions), 'system_id': predictions[0]['system_id'],
        'model_artifact_sha256': predictions[0]['model_artifact_sha256'],
        'evaluator_source_sha256': {p.name: sha256_file(p) for p in [Path(align.__file__), Path(normalize.__file__), Path(metrics_module.__file__)]},
        'metrics': metrics,
        'limitations': ['Upstream edited minutes, every recovered row marked unreviewed',
                       'Missing three private validation corrections and original listening records',
                       'Not an official M1-corrected-reference or final-test result',
                       'Training/checkpoint/fusion selection must identify this reference view explicitly']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'system_id': report['system_id'],
                      'metrics': {view: {'wer': values['wer'], 'cer': values['cer'], 'utterances': values['utterances']}
                                  for view, values in metrics.items()}}))


if __name__ == '__main__':
    main()
