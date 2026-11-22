"""M6 F2 reproduction helper: preflight, configs, derived protocols, epoch selection, status, comparison.

Run with the GPU Python from the main clone (cwd) with PYTHONPATH=<main>\\src.
Prints aggregates only; never reference or hypothesis text.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

RELEASE = 'turas-tr-econ-legis-v0.1'
MAIN_COMMIT = 'ee1f3e4ab215dd6449365b6012684d57a27e3cb9'
PROTOCOL = Path('configs/evaluation/m45_final_protocol.json')
PROTOCOL_SHA256 = '288aebd71d8a3574e13d436748d4161bc258a487e65066f1c3508bc8e835bec1'
CONFIG = Path('configs/fine-tuning/whisper_small_full.json')
SPLIT = Path('data/splits') / RELEASE
SPLIT_MANIFEST_SHA256 = 'a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8'
DATA_SHA256 = {
    'inference.train': '1e4bfd3df1bf10bb39bde558c50939631d71bdc003c56f19ff134a8edb36dc8d',
    'inference.validation': '1c6b12d4a7de3ed5b717d193f07d8194dc7105bd6785305029afef6d9e3ac3f2',
    'inference.test': '299dc6ae1f50772eb85081293c9bbe746c746a9aefc94870deff530a641059ef',
    'references.train': 'f4992d2951389ac243c281d2875511cfa0669d026d8698993db48b30b09a4508',
    'references.validation': 'd0c8519ca6aed053f89786bc733541a70faee09cf8d8bdb6cc12065d97dbd958',
}
TEST_REFERENCES_SHA256 = '31e372f0e35f504de59b49f1e0e1f3ed6fcc2772138c591a2cf2b054a4e7ef3f'
AUDIO_FILES = 5450
EPOCHS = (1, 2, 3)
RAW, NORM = 'raw-v1', 'tr-normalized-v1'
SMOKE_ROWS = 16
SMOKE_ACCUMULATION = 4
KEEP_AWAKE = 0x80000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
RELEASE_AWAKE = 0x80000000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """ASCII JSON (no BOM), so the repo's locale-default read_text() decodes it the same everywhere."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='ascii')


def data_file(data, kind):
    return Path(data) / f'{RELEASE}.{kind}.jsonl'


def checkpoint_sha(checkpoint):
    return sha256(Path(checkpoint) / 'checkpoint-manifest.json')


# ---------------------------------------------------------------- preflight

def _check(results, name, ok, detail=''):
    results.append(ok)
    print(json.dumps({'check': name, 'ok': bool(ok), 'detail': detail}), flush=True)


def _repo_checks(results):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    _check(results, 'main clone at ee1f3e4', head == MAIN_COMMIT, head[:12])
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.splitlines()
    print(json.dumps({'info': 'main clone modified/untracked entries', 'count': len(dirty)}), flush=True)
    _check(results, 'frozen protocol sha256', sha256(PROTOCOL) == PROTOCOL_SHA256)
    _check(results, 'split manifest sha256', sha256(SPLIT / 'split_manifest.json') == SPLIT_MANIFEST_SHA256)
    manifest = read_json(SPLIT / 'split_manifest.json')
    for split in ('train', 'validation', 'test'):
        _check(results, f'{split}.txt sha256', sha256(SPLIT / f'{split}.txt') == manifest['splits'][split]['sha256'])
    protocol = read_json(PROTOCOL)
    _check(results, 'lexicon sha256', sha256(Path(protocol['lexicon']['path'])) == protocol['lexicon']['sha256'])
    bad = [p for p, d in protocol['critical_code_sha256'].items() if not Path(p).is_file() or sha256(p) != d]
    _check(results, 'frozen critical code (56 files) unchanged', not bad, f'{len(bad)} mismatched: {bad[:3]}')
    config = read_json(CONFIG)
    _check(results, 'F2 config model/budget', config['mode'] == 'full' and config['training'] == {
        'epochs': 3, 'accumulation': 16, 'learning_rate': 1e-05, 'seed': 17, 'device': 'cuda', 'precision': 'bf16'})


def _data_checks(results, data, test_refs):
    for kind, expected in DATA_SHA256.items():
        _check(results, f'{kind} sha256', sha256(data_file(data, kind)) == expected)
    _check(results, 'test references sha256 (M1 payload)', sha256(test_refs) == TEST_REFERENCES_SHA256, str(test_refs))
    count = sum(1 for _ in (Path(data) / 'audio').glob('*.flac'))
    _check(results, 'audio FLAC count', count == AUDIO_FILES, str(count))


def _gpu_checks(results):
    import importlib.metadata as md
    import os
    versions = {}
    for name in ('torch', 'transformers', 'huggingface_hub', 'safetensors', 'soundfile', 'soxr', 'pandas', 'jiwer', 'numpy'):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    _check(results, 'packages importable', all(versions.values()), json.dumps(versions))
    _check(results, 'offline env set', os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1')
    import torch
    cuda = torch.cuda.is_available()
    _check(results, 'CUDA + bf16', cuda and torch.cuda.is_bf16_supported(), torch.cuda.get_device_name(0) if cuda else 'no CUDA')
    if cuda:
        free, total = torch.cuda.mem_get_info()
        print(json.dumps({'info': 'GPU memory GiB', 'free': round(free / 2**30, 2), 'total': round(total / 2**30, 2)}), flush=True)
    from huggingface_hub import hf_hub_download
    model = read_json(CONFIG)['model']
    weight = hf_hub_download(model['id'], model['weight_file'], revision=model['revision'])
    _check(results, 'cached whisper-small weight sha256', sha256(weight) == model['weight_sha256'])
    from transformers import WhisperProcessor
    WhisperProcessor.from_pretrained(model['id'], revision=model['revision'], language='turkish', task='transcribe')
    _check(results, 'cached processor loads offline', True)


def cmd_preflight(args):
    import turkish_asr_rdl
    results = []
    source = Path(turkish_asr_rdl.__file__).resolve()
    _check(results, 'turkish_asr_rdl imported from main clone', source.is_relative_to(Path.cwd().resolve() / 'src'), str(source))
    _repo_checks(results)
    if args.data:
        _data_checks(results, args.data, args.test_refs)
    if args.gpu:
        _gpu_checks(results)
    print(json.dumps({'preflight': 'PASS' if all(results) else 'FAIL', 'checks': len(results), 'failed': results.count(False)}))
    return 0 if all(results) else 1


# ---------------------------------------------------------------- configs and protocols

def cmd_config(args):
    config = read_json(CONFIG)
    data = Path(args.data)
    config['inputs'] = {**config['inputs'],
        'inference': str(data_file(data, 'inference.train')), 'references': str(data_file(data, 'references.train')),
        'audio_root': str(data), 'split_manifest': str((SPLIT / 'split_manifest.json').resolve()),
        'train_ids': str((SPLIT / 'train.txt').resolve()), 'baseline_report': str(Path(args.baseline)),
        'references_sha256': DATA_SHA256['references.train'], 'inference_sha256': DATA_SHA256['inference.train']}
    config['training'] = {**config['training'], 'seed': args.seed}
    if args.smoke:
        config['training'] = {**config['training'], 'epochs': 1, 'accumulation': SMOKE_ACCUMULATION}
        config['m6_note'] = f'SMOKE ONLY: first {SMOKE_ROWS} train rows, 1 epoch; never scored'
    write_json(args.output, config)
    print(json.dumps({'config': str(args.output), 'seed': args.seed, 'smoke': args.smoke}))
    return 0


def cmd_smoke_train(args):
    """Unchanged run_training, with the validated train rows truncated (smoke only)."""
    from turkish_asr_rdl.training import acoustic, whisper
    full = acoustic.load_train_rows
    acoustic.load_train_rows = lambda *a, **k: full(*a, **k)[:SMOKE_ROWS]
    result = acoustic.run_training(read_json(args.config), Path(args.output), whisper.build_model, whisper.build_collator)
    print(json.dumps({'status': result['status'], 'checkpoints': len(result['checkpoints'])}))
    return 0


def cmd_protocol(args):
    """Frozen protocol with input paths repointed to the public data and one adapted system."""
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('frozen protocol checksum mismatch')
    protocol = read_json(PROTOCOL)
    data = Path(args.data)
    inputs = {**protocol['inputs'], 'audio_root': str(data),
              'test_inference': str(data_file(data, 'inference.test')),
              'warmup_inference': str(data_file(data, 'inference.train'))}
    if args.test_ids_subset:
        subset = Path(args.output).with_name('smoke-test-ids.txt')
        ids = (SPLIT / 'test.txt').read_text(encoding='utf-8').splitlines()[:args.test_ids_subset]
        subset.write_bytes(('\n'.join(ids) + '\n').encode('ascii'))
        inputs = {**inputs, 'test_ids': str(subset), 'test_ids_sha256': sha256(subset)}
    checkpoint = Path(args.checkpoint).resolve()
    system = {'system_id': args.system_id, 'kind': 'adapted', 'family': 'whisper', 'checkpoint': str(checkpoint),
              'checkpoint_manifest_sha256': checkpoint_sha(checkpoint), 'selected_epoch': args.epoch}
    derived = {**protocol, 'inputs': inputs, 'systems': [system],
               'm6_derived_from': {'protocol_sha256': PROTOCOL_SHA256, 'changed': [
                   'inputs.audio_root', 'inputs.test_inference', 'inputs.warmup_inference', 'systems']
                   + (['inputs.test_ids (smoke subset)'] if args.test_ids_subset else []),
                   'note': 'M6 post-freeze reproduction; decoder block unchanged'}}
    write_json(args.output, derived)
    print(sha256(args.output))
    return 0


def cmd_sha(args):
    print(sha256(args.path))
    return 0


# ---------------------------------------------------------------- selection

def _metrics(report, view):
    return report['metrics'][view]


def select_epoch(summary, prefix):
    """Lowest normalized WER; ties by raw WER, then earlier epoch (configs/fine-tuning/whisper_small_full.json)."""
    reports = {r['system_id']: r for r in summary['systems']}
    candidates = []
    for epoch in EPOCHS:
        report = reports[f'{prefix}-e{epoch}']
        if report['error_rows']:
            raise ValueError(f'epoch {epoch} has error rows')
        candidates.append({'epoch': epoch, 'normalized_wer': _metrics(report, NORM)['wer'],
                           'raw_wer': _metrics(report, RAW)['wer'], 'normalized_cer': _metrics(report, NORM)['cer'],
                           'raw_cer': _metrics(report, RAW)['cer']})
    best = min(candidates, key=lambda c: (c['normalized_wer'], c['raw_wer'], c['epoch']))
    return candidates, best


def cmd_select(args):
    seed_dir = Path(args.seed_dir)
    summary = read_json(seed_dir / 'score-val' / 'summary.json')
    candidates, best = select_epoch(summary, f'F2-m6repro-s{args.seed}')
    checkpoint = seed_dir / 'train' / f"epoch-{best['epoch']}"
    write_json(seed_dir / 'selection.json', {
        'seed': args.seed, 'criterion': 'lowest validation normalized WER; raw WER, then earlier epoch',
        'reference_sha256': summary['reference_sha256'], 'candidates': candidates, 'selected_epoch': best['epoch'],
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_manifest_sha256': checkpoint_sha(checkpoint),
        'selected_before_test_decode': True})
    print(json.dumps({'seed': args.seed, 'selected_epoch': best['epoch'],
                      'validation_normalized_wer_pct': round(100 * best['normalized_wer'], 4)}))
    return 0


# ---------------------------------------------------------------- comparison and status

def committed_values():
    """Committed F2 numbers, read from the repo (percent)."""
    import csv
    with Path('results/final/m45_metrics.csv').open(encoding='utf-8', newline='') as stream:
        row = next(r for r in csv.DictReader(stream) if r['system'] == 'F2-whisper-full')
    test = {k: float(row[f'{k}_pct']) for k in ('raw_wer', 'normalized_wer', 'raw_cer', 'normalized_cer')}
    sel = next(s for s in read_json('results/evaluation/m45_working_checkpoint_selection.json')['selections']
               if s['model'] == 'whisper-full-seed17')['selected']
    val = {'raw_wer': 100 * sel['raw_wer'], 'normalized_wer': 100 * sel['normalized_wer'], 'raw_cer': None,
           'normalized_cer': 3.447}  # docs/m45_validation_complete.md table (rounded there)
    return {'test': test, 'validation': val, 'selected_epoch': sel['epoch']}


def _pct(report):
    return {'raw_wer': 100 * report['metrics'][RAW]['wer'], 'normalized_wer': 100 * report['metrics'][NORM]['wer'],
            'raw_cer': 100 * report['metrics'][RAW]['cer'], 'normalized_cer': 100 * report['metrics'][NORM]['cer']}


def _spread(values):
    if not values:
        return {}
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
    return {'n': len(values), 'mean': mean, 'sd': sd, 'min': min(values), 'max': max(values)}


def cmd_compare(args):
    root = Path(args.root)
    committed = committed_values()
    rows = {}
    for seed in args.seeds:
        seed_dir = root / f'seed{seed}'
        test = seed_dir / 'score-test' / f'F2-m6repro-s{seed}.json'
        if not test.is_file() or not (seed_dir / 'selection.json').is_file():
            print(json.dumps({'seed': seed, 'status': 'not finished; left out'}))
            continue
        selection = read_json(seed_dir / 'selection.json')
        val = next(c for c in selection['candidates'] if c['epoch'] == selection['selected_epoch'])
        rows[seed] = {'selected_epoch': selection['selected_epoch'], 'test': _pct(read_json(test)),
                      'validation': {k: 100 * val[k] for k in ('raw_wer', 'normalized_wer', 'raw_cer', 'normalized_cer')}}
    keys = ('raw_wer', 'normalized_wer', 'raw_cer', 'normalized_cer')
    fmt = lambda v: 'n/a' if v is None else f'{v:8.4f}'
    for split in ('test', 'validation'):
        print(f'\n{split.upper()} (percent)      ' + ''.join(f'{k:>16}' for k in keys))
        print(f"{'committed F2 (seed 17, e' + str(committed['selected_epoch']) + ')':<26}" + ''.join(f'{fmt(committed[split][k]):>16}' for k in keys))
        for seed, row in rows.items():
            print(f"{'M6 repro seed ' + str(seed) + ' (e' + str(row['selected_epoch']) + ')':<26}" + ''.join(f'{fmt(row[split][k]):>16}' for k in keys))
        stats = {k: _spread([row[split][k] for row in rows.values()]) for k in keys}
        if rows:
            print(f"{'mean over ' + str(len(rows)) + ' seed(s)':<26}" + ''.join(f"{fmt(stats[k]['mean']):>16}" for k in keys))
            print(f"{'sample SD':<26}" + ''.join(f"{fmt(stats[k]['sd']):>16}" for k in keys))
            print(f"{'range (max-min)':<26}" + ''.join(f"{fmt(stats[k]['max'] - stats[k]['min']):>16}" for k in keys))
        if 17 in rows:
            print(f"{'seed 17 - committed':<26}" + ''.join(
                f"{fmt(None if committed[split][k] is None else rows[17][split][k] - committed[split][k]):>16}" for k in keys))
    write_json(root / 'compare.json', {'committed_f2': committed, 'reproduction': {str(k): v for k, v in rows.items()},
        'units': 'percent', 'note': 'M6 scorer (reproduction) vs frozen scorer (committed test); cross-scorer'})
    print(json.dumps({'written': str(root / 'compare.json')}))
    return 0


def _seed_status(seed_dir):
    state = {'seed': seed_dir.name}
    run = seed_dir / 'train' / 'run.json'
    progress = seed_dir / 'train' / 'progress.json'
    if run.is_file():
        try:
            data = read_json(run)
            state.update(train=data['status'], epochs_saved=len(data['checkpoints']))
        except (ValueError, KeyError):
            state['train'] = 'run.json being written'
    if progress.is_file():
        try:
            p = read_json(progress)
            state.update(epoch=p['epoch'], step=p['optimizer_steps'], loss=round(p['mean_train_loss_so_far'], 4),
                         minutes=round(p['elapsed_optimization_s'] / 60, 1),
                         peak_reserved_gib=round(p.get('peak_cuda_reserved_bytes', 0) / 2**30, 2))
        except (ValueError, KeyError):
            pass
    state['val_decoded'] = [e for e in EPOCHS if (seed_dir / f'val-e{e}.ok.json').is_file()]
    for name in ('score-val/summary.json', 'selection.json', 'test/completed.json', 'score-test/summary.json'):
        state[name.split('/')[0]] = (seed_dir / name).is_file()
    return state


def cmd_status(args):
    root = Path(args.root)
    for seed_dir in sorted(root.glob('seed*')) + [root / 'smoke']:
        if seed_dir.is_dir():
            print(json.dumps(_seed_status(seed_dir)))
    for name in ('pipeline.log', 'trainloop.done', 'evalloop.done'):
        path = root / name
        if path.is_file():
            print(json.dumps({name: path.read_text(encoding='utf-8').strip().splitlines()[-1:]}))
    return 0


def cmd_peak(args):
    """Peak reserved CUDA GiB from a training progress.json (0 when absent)."""
    try:
        print(round(read_json(args.progress).get('peak_cuda_reserved_bytes', 0) / 2**30, 2))
    except (OSError, ValueError):
        print(0)
    return 0


def cmd_keepawake(args):
    """Hold a system-awake request until the stop file appears or the hour limit passes."""
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(KEEP_AWAKE)
    try:
        ends = time.monotonic() + args.hours * 3600
        while time.monotonic() < ends and not Path(args.until_file).exists():
            time.sleep(30)
    finally:
        kernel.SetThreadExecutionState(RELEASE_AWAKE)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight'); p.add_argument('--data'); p.add_argument('--test-refs'); p.add_argument('--gpu', action='store_true')
    p = sub.add_parser('config'); p.add_argument('--data', required=True); p.add_argument('--baseline', required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    p = sub.add_parser('smoke-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('protocol'); p.add_argument('--data', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--system-id', required=True); p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--output', required=True); p.add_argument('--test-ids-subset', type=int, default=0)
    p = sub.add_parser('sha'); p.add_argument('path')
    p = sub.add_parser('select'); p.add_argument('--seed-dir', required=True); p.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('compare'); p.add_argument('--root', required=True); p.add_argument('--seeds', type=int, nargs='+', default=[17, 18, 19])
    p = sub.add_parser('status'); p.add_argument('--root', required=True)
    p = sub.add_parser('peak'); p.add_argument('progress')
    p = sub.add_parser('keepawake'); p.add_argument('--until-file', required=True); p.add_argument('--hours', type=float, default=5.5)
    args = parser.parse_args(argv)
    handler = {'preflight': cmd_preflight, 'config': cmd_config, 'smoke-train': cmd_smoke_train, 'protocol': cmd_protocol,
               'sha': cmd_sha, 'select': cmd_select, 'compare': cmd_compare, 'status': cmd_status, 'peak': cmd_peak,
               'keepawake': cmd_keepawake}[args.command]
    try:
        return handler(args)
    except (OSError, ValueError, KeyError, StopIteration, ImportError) as exc:
        sys.stderr.write(f'f2repro {args.command}: {type(exc).__name__}: {exc}\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
