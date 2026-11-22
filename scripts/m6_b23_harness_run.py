"""B2/B3 test inference or timing through M2's adapters and the shared run_timed harness.

NEW, UNREVIEWED helper; not on origin/main. It mirrors scripts/m45_final_infer.py:
the frozen protocol supplies the 636 test IDs (accuracy) or the 107 timing IDs
with 3 repeats (timing), the 3 train warm-ups, batch 1 and the input hashes.
Only the model comes from an M2 baseline config. No reference file is read.
Run from the repository root so the protocol's relative paths resolve.
"""
import argparse
import json
from pathlib import Path
import uuid

from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def make_adapter(config, checkpoint, **adapter_kwargs):
    model_id = config['model']['id']
    if model_id.startswith('openai/whisper'):
        from turkish_asr_rdl.baselines.whisper import WhisperBaselineAdapter as base
    elif model_id.startswith('Qwen/Qwen3-ASR'):
        from turkish_asr_rdl.baselines.qwen import QwenBaselineAdapter as base
    else:
        raise ValueError('unsupported baseline model: ' + model_id)

    class Progress(base):
        completed = 0

        def transcribe(self, batch):
            results = super().transcribe(batch)
            self.completed += len(batch)
            if self.completed == 1 or self.completed % 25 == 0:
                print(json.dumps({'items_including_warmup': self.completed}), flush=True)
            return results

    return Progress(config, checkpoint_path=checkpoint, **adapter_kwargs)


def frozen_inputs(protocol, mode):
    inputs = protocol['inputs']
    inference = Path(inputs['test_inference'])
    if sha256_file(inference) != inputs['test_inference_sha256']:
        raise ValueError('inference hash mismatch')
    ids_path = Path(inputs['test_ids'] if mode == 'accuracy' else inputs['timing_ids'])
    if sha256_file(ids_path) != inputs['test_ids_sha256' if mode == 'accuracy' else 'timing_ids_sha256']:
        raise ValueError('frozen IDs hash mismatch')
    ids = ids_path.read_text().splitlines()
    wanted = set(ids)
    rows = [r for r in read_jsonl(inference) if r['utterance_id'] in wanted]
    if [r['utterance_id'] for r in rows] != ids:
        raise ValueError('inference coverage/order mismatch')
    warmup_path = Path(inputs['warmup_inference'])
    if sha256_file(warmup_path) != inputs['warmup_inference_sha256']:
        raise ValueError('warm-up hash mismatch')
    timed = [InferenceInput.from_row(r) for r in rows]
    warmup = [InferenceInput.from_row(r) for r in read_jsonl(warmup_path)[:3]]
    if any(r.dataset_release != protocol['dataset_release']
           or r.split_manifest_sha256 != protocol['split_manifest_sha256'] for r in timed + warmup):
        raise ValueError('inference release/split mismatch')
    return ids, timed, warmup


def error_rows_split(run, repeats, predictions):
    """Error rows of the warm-up batches, of each timed repeat, and in predictions.jsonl (repeat 1)."""
    return {'warmup': sum(r.error_rows for r in run.records if r.warmup),
            'per_repeat': [sum(r.error_rows for r in run.records if not r.warmup and r.repeat == k)
                           for k in range(1, repeats + 1)],
            'predictions': sum(r['status'] == 'error' for r in predictions)}


def main(argv=None, adapter_factory=make_adapter):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=['accuracy', 'timing'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output must be fresh')
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError('frozen protocol hash mismatch')
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.config.read_text(encoding='utf-8'))
    ids, timed, warmup = frozen_inputs(protocol, args.mode)
    adapter = adapter_factory(config, args.checkpoint)
    run_id = adapter.system_id + '-' + args.mode + '-' + uuid.uuid4().hex[:8]
    repeats = 1 if args.mode == 'accuracy' else protocol['timing']['repeats']
    run = run_timed(adapter, timed, warmup, audio_root=Path(protocol['inputs']['audio_root']),
                    run_id=run_id, repeats=repeats, batch_size=protocol['timing']['batch_size'])
    ctx = RunContext(run_id, adapter.system_id, protocol['dataset_release'], 'test',
                     protocol['split_manifest_sha256'], adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    files = write_run_outputs(args.output, run, timed, ctx, adapter, command='b23_harness_run.py --mode ' + args.mode)
    predictions = read_jsonl(args.output / 'predictions.jsonl')
    if not validate_predictions(predictions, ids).ok or not validate_nbest(read_jsonl(args.output / 'nbest.jsonl'), ids).ok:
        raise ValueError('output contract violation')
    split = error_rows_split(run, repeats, predictions)
    # predictions.jsonl holds the repeat-1 hypotheses, so its error rows are already in run.records.
    errors = sum(r.error_rows for r in run.records)
    status = 'completed' if not errors else 'failed_error_rows'
    write_json(args.output / 'completed.json', {
        'status': status, 'mode': args.mode,
        'protocol_sha256': args.protocol_sha256, 'system_id': adapter.system_id, 'rows': len(ids),
        'repeats': repeats, 'error_rows': errors, 'error_rows_split': split, 'files': files,
        'baseline_config_sha256': sha256_file(args.config), 'tool_sha256': sha256_file(Path(__file__)),
        'tool_status': 'unreviewed helper outside origin/main',
        'timing_claim': 'same-session serial timing' if args.mode == 'timing' else 'accuracy run; not a timing claim'})
    print(json.dumps({'status': status, 'system': adapter.system_id, 'mode': args.mode,
                      'rows': len(ids), 'error_rows': errors}), flush=True)
    if errors:
        raise ValueError('inference error rows; evaluation gate remains closed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
