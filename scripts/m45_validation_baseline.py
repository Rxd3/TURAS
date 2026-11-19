"""Preserve unmodified-parent validation inference before fixed-budget adaptation.

No reference file is accepted; this is not a WER/CER evaluation or final RTF run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turkish_asr_rdl.decoding.adapted import PretrainedAsrAdapter
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.timing import MIN_WARMUP_BATCHES, run_timed, write_run_outputs
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--family', choices=['whisper', 'ctc'], required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--warmup-inference', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new directory')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    split_root = Path(config['inputs']['train_ids']).parent
    manifest = json.loads(Path(config['inputs']['split_manifest']).read_text(encoding='utf-8'))
    assert sha256_file(Path(config['inputs']['split_manifest'])) == config['split_manifest_sha256']
    ids_file = split_root / 'validation.txt'
    assert sha256_file(ids_file) == manifest['splits']['validation']['sha256']
    ids = ids_file.read_text(encoding='utf-8').splitlines()
    rows = read_jsonl(args.inference)
    assert [r['utterance_id'] for r in rows] == ids
    timed = [InferenceInput.from_row(r) for r in rows]
    train_ids = set((split_root / 'train.txt').read_text(encoding='utf-8').splitlines())
    warmup = [InferenceInput.from_row(r) for r in read_jsonl(args.warmup_inference)[:MIN_WARMUP_BATCHES]]
    assert len(warmup) == MIN_WARMUP_BATCHES and all(r.utterance_id in train_ids for r in warmup)
    for row in timed + warmup:
        assert row.dataset_release == config['dataset_release'] and row.split_manifest_sha256 == config['split_manifest_sha256']

    class ProgressAdapter(PretrainedAsrAdapter):
        completed = 0
        def transcribe(self, batch):
            output = super().transcribe(batch)
            self.completed += len(batch)
            if self.completed == 1 or self.completed % 25 == 0:
                print(json.dumps({'event': 'baseline_inference', 'items_including_warmup': self.completed}), flush=True)
            return output

    run_id = 'm45-' + args.family + '-pretrained-validation-seed17'
    adapter = ProgressAdapter(config['model'], family=args.family, system_id=run_id)
    # Weights/processor must already be cached: no download is included in timing.
    run = run_timed(adapter, timed, warmup, audio_root=args.audio_root, run_id=run_id, repeats=1, batch_size=1)
    ctx = RunContext(run_id, adapter.system_id, config['dataset_release'], 'validation',
                     config['split_manifest_sha256'], adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    files = write_run_outputs(args.output, run, timed, ctx, adapter, command='scripts/m45_validation_baseline.py')
    predictions = read_jsonl(args.output / 'predictions.jsonl')
    nbest = read_jsonl(args.output / 'nbest.jsonl')
    for report in (validate_predictions(predictions, ids), validate_nbest(nbest, ids)):
        if not report.ok:
            raise ValueError('baseline export failed the shared contract: ' + '; '.join(report.errors[:5]))
    if any(row['status'] == 'error' for row in predictions + nbest):
        raise ValueError('baseline contains error rows; training gate remains closed')
    write_json(args.output / 'baseline-evidence.json', {
        'status': 'completed', 'scope': 'validation_inference_only',
        'metrics_status': 'pending_validation_references', 'training_mode': 'pretrained',
        'split': 'validation', 'model_id': config['model']['id'], 'revision': config['model']['revision'],
        'weight_sha256': config['model']['weight_sha256'], 'dataset_release': config['dataset_release'],
        'split_manifest_sha256': config['split_manifest_sha256'], 'utterances': len(predictions),
        'files': files, 'inference_sha256': sha256_file(args.inference),
        'limitations': ['No validation references read; WER/CER not measured',
                        'One timed pass, not the final three-repeat RTF protocol',
                        'Checkpoint and fusion selection remain pending validation scoring']})
    print(json.dumps({'status': 'baseline_inference_completed_metrics_pending', 'utterances': len(predictions)}), flush=True)


if __name__ == '__main__':
    main()
