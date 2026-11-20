"""Run one saved M4/M5 epoch on gold-free frozen validation audio."""
import argparse
import json
from pathlib import Path
import uuid

from turkish_asr_rdl.decoding.adapted import AdaptedAsrAdapter
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.timing import MIN_WARMUP_BATCHES, run_timed, write_run_outputs
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'inference', 'warmup-inference', 'audio-root', 'split-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint-manifest-sha256', required=True)
    parser.add_argument('--system-id', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')

    class ProgressAdapter(AdaptedAsrAdapter):
        completed = 0
        errors = 0
        def transcribe(self, batch):
            result = super().transcribe(batch)
            self.completed += len(batch)
            self.errors += sum(h.status == 'error' for h in result)
            if self.completed == 1 or self.completed % 25 == 0:
                print(json.dumps({'items_including_warmup': self.completed, 'error_rows': self.errors}), flush=True)
            return result

    adapter = ProgressAdapter(args.checkpoint, args.checkpoint_manifest_sha256,
                              system_id=args.system_id, device='cuda', precision='bf16')
    checkpoint = adapter.manifest
    split_sha = sha256_file(args.split_manifest)
    manifest = json.loads(args.split_manifest.read_text(encoding='utf-8'))
    if checkpoint['split_manifest_sha256'] != split_sha or checkpoint['dataset_release'] != manifest['dataset_release']:
        raise ValueError('checkpoint/frozen release mismatch')
    ids_path = args.split_manifest.parent / 'validation.txt'
    train_path = args.split_manifest.parent / 'train.txt'
    for path, split in ((ids_path, 'validation'), (train_path, 'train')):
        if sha256_file(path) != manifest['splits'][split]['sha256']:
            raise ValueError('frozen ID hash mismatch')
    ids = ids_path.read_text(encoding='utf-8').splitlines()
    rows = read_jsonl(args.inference)
    if [r['utterance_id'] for r in rows] != ids:
        raise ValueError('validation inference ID order mismatch')
    timed = [InferenceInput.from_row(r) for r in rows]
    warmup = [InferenceInput.from_row(r) for r in read_jsonl(args.warmup_inference)[:MIN_WARMUP_BATCHES]]
    train_ids = set(train_path.read_text(encoding='utf-8').splitlines())
    if len(warmup) != MIN_WARMUP_BATCHES or any(r.utterance_id not in train_ids for r in warmup):
        raise ValueError('warm-up must use the shared minimum number of train clips')
    if any(r.dataset_release != checkpoint['dataset_release'] or r.split_manifest_sha256 != split_sha for r in timed + warmup):
        raise ValueError('inference release/split mismatch')
    run_id = args.system_id + '-' + uuid.uuid4().hex[:8]
    run = run_timed(adapter, timed, warmup, audio_root=args.audio_root, run_id=run_id, repeats=1, batch_size=1)
    ctx = RunContext(run_id, args.system_id, checkpoint['dataset_release'], 'validation', split_sha,
                     adapter.model_artifact_sha256, adapter.decoder_config_sha256)
    write_run_outputs(args.output, run, timed, ctx, adapter, command='scripts/m45_validation_adapted.py')
    predictions = read_jsonl(args.output / 'predictions.jsonl')
    for report in (validate_predictions(predictions, ids), validate_nbest(read_jsonl(args.output / 'nbest.jsonl'), ids)):
        if not report.ok:
            raise ValueError('adapted validation export failed shared validators')
    if adapter.errors or any(row['status'] == 'error' for row in predictions):
        raise ValueError('adapted validation contains error rows; selection remains pending')
    print(json.dumps({'status': 'validation_inference_completed', 'utterances': len(predictions)}), flush=True)


if __name__ == '__main__':
    main()
