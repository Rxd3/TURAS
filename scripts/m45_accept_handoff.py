"""Restore and exercise the delivered models from a fresh, offline source checkout.

Uses two frozen validation inputs and train warm-ups; never reads references.
This owner-assisted check does not impersonate an independent peer review.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file, write_json


def extract_delivery(archive_path, destination):
    """Reject links/traversal and conflicting files before extracting anything."""
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive member')
        pending = []
        for member in members:
            name = member.name
            target = (destination / name).resolve()
            if (not member.isfile() or '\\' in name or ':' in name or
                    PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
                    not target.is_relative_to(destination)):
                raise ValueError('unsafe archive member')
            if target.exists():
                digest = hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest()
                if not target.is_file() or sha256_file(target) != digest:
                    raise ValueError('archive would overwrite different checkout bytes')
            else:
                pending.append(member)
        archive.extractall(destination, members=pending, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('results/final/m45_handoff_manifest.json'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference-root', type=Path, required=True)
    args = parser.parse_args()
    original_root = Path.cwd().resolve()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(original_root / 'artifacts'):
        parser.error('output must be fresh under ignored artifacts/')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    archive_path = (args.archive or Path(manifest['private_archive'])).resolve()
    if sha256_file(archive_path) != manifest['archive_sha256']:
        raise ValueError('delivered archive checksum mismatch')
    output.mkdir(parents=True)
    bundle = output / 'runtime-source.bundle'
    with tarfile.open(archive_path) as archive:
        source = archive.getmember('TURAS-M4-M5-source.bundle')
        if not source.isfile():
            raise ValueError('runtime source bundle is not a regular file')
        bundle.write_bytes(archive.extractfile(source).read())
    if sha256_file(bundle) != manifest['source_git_bundle_sha256']:
        raise ValueError('runtime source bundle checksum mismatch')
    repo = output / 'restored-repository'
    subprocess.run(['git', 'clone', '--branch', 'feat/m4-m5-integration', str(bundle), str(repo)], check=True)
    extract_delivery(archive_path, repo)
    index = repo / 'artifact-index.json'
    if sha256_file(index) != manifest['artifact_index_sha256']:
        raise ValueError('restored artifact index mismatch')
    inventory = json.loads(index.read_text(encoding='utf-8'))
    for name, record in inventory['files'].items():
        target = (repo / name).resolve()
        if not target.is_relative_to(repo) or sha256_file(target) != record['sha256']:
            raise ValueError('restored artifact byte verification failed')
    frozen_path = repo / 'configs/evaluation/m45_final_protocol.json'
    if sha256_file(frozen_path) != manifest['protocol_sha256']:
        raise ValueError('restored final protocol mismatch')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    for name, digest in frozen['critical_code_sha256'].items():
        if sha256_file(repo / name) != digest:
            raise ValueError('restored runtime source mismatch')
    inference_root = args.inference_root.resolve()
    release = frozen['dataset_release']
    export_hashes = json.loads((repo / f'data/manifests/{release}.json').read_text())['inference_exports_sha256']
    val = inference_root / f'{release}.inference.validation.jsonl'
    warm = inference_root / f'{release}.inference.train.jsonl'
    if sha256_file(val) != export_hashes['validation'] or sha256_file(warm) != export_hashes['train']:
        raise ValueError('external validation/train inference bundle mismatch')
    frozen_validation_ids = (repo / f'data/splits/{release}/validation.txt').read_text().splitlines()
    ids = frozen_validation_ids[:2]
    inputs = read_jsonl(val)
    # The strict inference contract deliberately has no split/reference field.
    # Role comes from the pinned export hash and exact frozen ID membership.
    if ([r['utterance_id'] for r in inputs] != frozen_validation_ids or
            any(r['dataset_release'] != release or r['split_manifest_sha256'] != frozen['split_manifest_sha256'] for r in inputs)):
        raise ValueError('acceptance inputs must be frozen validation audio')
    proof_root = repo / 'artifacts/delivery-acceptance'
    proof_root.mkdir(parents=True)
    ids_path = proof_root / 'validation-ids.txt'
    ids_path.write_bytes(('\n'.join(ids) + '\n').encode())
    proof = deepcopy(frozen)
    proof.update(status='frozen_validation_preflight', evaluation_split='validation',
                 systems=[s for s in frozen['systems'] if s['kind'] != 'pretrained'])
    proof.pop('test_reference')
    proof['inputs'] = {'audio_root':str(inference_root), 'test_inference':str(val),
        'test_inference_sha256':sha256_file(val), 'warmup_inference':str(warm),
        'warmup_inference_sha256':sha256_file(warm), 'test_ids':str(ids_path),
        'test_ids_sha256':sha256_file(ids_path), 'timing_ids':str(ids_path),
        'timing_ids_sha256':sha256_file(ids_path)}
    proof['timing']['repeats'] = 1
    proof_path = proof_root / 'validation-only-protocol.json'
    write_json(proof_path, proof)
    digest = sha256_file(proof_path)
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONIOENCODING='utf-8')
    state = {'status':'running', 'verification':'owner-assisted fresh-source restore; existing pinned runtime reused', 'jobs':[]}
    write_json(output / 'status.json', state)
    def run(name, command):
        item = {'name':name, 'status':'running'}
        state['jobs'].append(item)
        write_json(output / 'status.json', state)
        with (output / (name + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u'] + command, cwd=repo, env=env,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        item.update(status='passed' if result.returncode == 0 else 'failed', exit_code=result.returncode)
        write_json(output / 'status.json', state)
        if result.returncode:
            raise RuntimeError('restored delivery check failed: ' + name)
    run('verify-import-origin', ['-c', "import pathlib,turkish_asr_rdl; p=pathlib.Path(turkish_asr_rdl.__file__).resolve(); assert p.is_relative_to(pathlib.Path.cwd()/'src'); print(p.relative_to(pathlib.Path.cwd()))"])
    prefix = ['--protocol',str(proof_path),'--protocol-sha256',digest]
    for spec in proof['systems']:
        # All are functional validation runs, including fresh joint decoding.
        # Accuracy mode labels its timing as excluded from headline benchmarks.
        mode = 'accuracy'
        target = proof_root / mode / spec['system_id']
        target.parent.mkdir(exist_ok=True)
        run(spec['system_id'], ['scripts/m45_final_infer.py'] + prefix +
            ['--system',spec['system_id'],'--mode',mode,'--output',str(target)])
    for spec in proof['rescore_conditions']:
        target = proof_root / 'rescore' / spec['system_id']
        target.parent.mkdir(exist_ok=True)
        run(spec['system_id'] + '-cached', ['scripts/m45_final_rescore.py'] + prefix +
            ['--condition',spec['system_id'],'--source',str(proof_root/'accuracy'/spec['base_system']),'--output',str(target)])
    for name in frozen['adapted_system_ids']:
        cached = read_jsonl(proof_root/'rescore'/(name+'--rdl-full')/'predictions.jsonl')
        joint = read_jsonl(proof_root/'accuracy'/(name+'--rdl-full')/'predictions.jsonl')
        if [(r['utterance_id'],r['hypothesis_text']) for r in cached] != [(r['utterance_id'],r['hypothesis_text']) for r in joint]:
            raise ValueError('restored joint/cached transcript parity failed')
    state.update(status='passed', archive_sha256=manifest['archive_sha256'],
        original_final_protocol_sha256=manifest['protocol_sha256'], validation_preflight_protocol_sha256=digest,
        restored_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        validation_utterance_ids=ids, restored_files_verified=len(inventory['files']),
        selected_acoustic_models_executed=3, online_joint_models_executed=3, cached_controls_executed=15,
        joint_cached_transcript_parity='passed all three models', references_read=False,
        test_audio_decoded=False, final_protocol_and_results_modified=False,
        independent_human_review='not claimed', runtime_lock_sha256=sha256_file(repo/'requirements/owner-m45-windows-cu130.txt'))
    write_json(output / 'status.json', state)
    write_json(output / 'acceptance.json', state)
    print(json.dumps({'status':'passed','restored_files':len(inventory['files']),'model_runs':21,'references_read':False}))


if __name__ == '__main__':
    main()
