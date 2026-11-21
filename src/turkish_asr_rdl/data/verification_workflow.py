"""Private review kits and access-separated exports; never mutate frozen inputs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import uuid

from turkish_asr_rdl.data import verification as V
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.evaluation.normalize import normalize, TR_NORMALIZED_V1
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file


def _archive(path, members):
    if path.exists():
        raise ValueError('archive output exists')
    hashes = {}
    with tarfile.open(path,'w') as archive:
        for name, source in sorted(members.items()):
            if name.startswith('/') or '..' in Path(name).parts or Path(name).name.startswith('._'):
                raise ValueError('unsafe archive name')
            payload = source if isinstance(source,bytes) else None
            info = tarfile.TarInfo(name)
            info.size = len(payload) if payload is not None else source.stat().st_size
            info.mode = 0o600
            if payload is not None:
                archive.addfile(info,io.BytesIO(payload))
                hashes[name] = hashlib.sha256(payload).hexdigest()
            else:
                with source.open('rb') as stream:
                    archive.addfile(info,stream)
                hashes[name] = sha256_file(source)
        sums = ''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())).encode()
        info = tarfile.TarInfo('SHA256SUMS'); info.size = len(sums); info.mode = 0o600
        archive.addfile(info,io.BytesIO(sums))
    return {'file':path.name,'sha256':sha256_file(path),'bytes':path.stat().st_size,
            'members':len(hashes)+1,'macos_appledouble_members':0}


def _registry(repo):
    with (repo/'data/metadata/turas-tr-econ-legis-v0.1-source-registry.csv').open(encoding='utf8',newline='') as stream:
        return {r['source_id']:r['session_date'] for r in csv.DictReader(stream)}


def load_workspace(root):
    meta = json.loads((root/'workspace.json').read_text(encoding='utf8'))
    refs_path = root/'sources/references.full.jsonl'
    if sha256_file(refs_path) != meta['reference_snapshot_sha256']:
        raise ValueError('prepared reference snapshot was modified')
    refs = {r['utterance_id']:r for r in read_jsonl(refs_path)}
    plan_path = root/'verification-plan.json'
    if sha256_file(plan_path) != meta['verification_plan_sha256']:
        raise ValueError('pinned verification plan changed')
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    if plan['reference_snapshot_sha256']!=meta['reference_snapshot_sha256']:
        raise ValueError('reference snapshot differs from the pinned public plan')
    return meta,refs,plan


def audit_splits(rows, ids_by_split):
    """Exact checks only: no claim of near-duplicate or speaker auditing."""
    overlaps={}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        def values(split,key):
            return {normalize(rows[u]['reference_text'],TR_NORMALIZED_V1) if key=='normalized_text' else rows[u][key] for u in ids_by_split[split]}
        overlaps[left+'--'+right]={key:len(values(left,key)&values(right,key)) for key in ['leakage_group_id','audio_sha256','normalized_text']}
    if any(value for pair in overlaps.values() for value in pair.values()):
        raise ValueError('cross-split exact overlap detected; frozen inputs are not modified')
    return overlaps


def plan_committed(repo, meta):
    try:
        payload = subprocess.check_output(['git','show','HEAD:'+meta['public_plan_path']],cwd=repo,stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return hashlib.sha256(payload).hexdigest()==meta['verification_plan_sha256']


def prepare(repo, inputs_root, profile_path, output, render):
    profile = json.loads(profile_path.read_text(encoding='utf8'))
    release = profile['dataset_release']
    split_dir = repo/'data/splits'/release
    if sha256_file(split_dir/'split_manifest.json') != profile['split_manifest_sha256']:
        raise ValueError('frozen split manifest mismatch')
    audio = (inputs_root/profile['inference_root']).resolve()
    output = output.resolve()
    if output.exists() or output.is_relative_to(audio) or audio.is_relative_to(output):
        raise ValueError('private output must be fresh and separate from the audio input root')
    if output.is_relative_to(repo.resolve()) and not any(output.is_relative_to(repo.resolve()/p) for p in ['data/private','artifacts']):
        raise ValueError('private output cannot be a public repository path')
    rows, reference_sources, inference_files = {}, {}, {}
    ids_by_split, inference = {}, {}
    for split in ('train','validation','test'):
        expected_ids = (split_dir/(split+'.txt')).read_text().splitlines()
        ids_by_split[split] = expected_ids
        source = profile['references'][split]
        path = (inputs_root/source['path']).resolve()
        if sha256_file(path) != source['sha256']:
            raise ValueError('reference source mismatch: '+split)
        refs = read_jsonl(path)
        if [r['utterance_id'] for r in refs] != expected_ids or any(r['split']!=split for r in refs):
            raise ValueError('reference IDs/order/split mismatch: '+split)
        reference_sources[split] = path
        export = audio/f'{release}.inference.{split}.jsonl'
        if sha256_file(export) != profile['inference_sha256'][split]:
            raise ValueError('inference source mismatch: '+split)
        items = read_jsonl(export)
        if [r['utterance_id'] for r in items] != expected_ids:
            raise ValueError('inference coverage mismatch')
        for item in items:
            InferenceInput.from_row(item)
        inference_files[split] = export
        inference.update({r['utterance_id']:r for r in items})
        for r in refs:
            uid = r['utterance_id']
            if uid in rows or r['dataset_release']!=release or r['audio_sha256']!=inference[uid]['audio_sha256']:
                raise ValueError('reference identity mismatch')
            rows[uid] = r
    import soundfile as sf
    audio_members = {}
    for uid,item in sorted(inference.items()):
        path = (audio/item['audio_uri']).resolve()
        if not path.is_relative_to(audio) or sha256_file(path)!=item['audio_sha256']:
            raise ValueError('audio path/hash mismatch')
        info = sf.info(path)
        if info.samplerate!=16000 or info.channels!=1 or info.subtype!='PCM_16' or abs(info.frames/16000-item['duration_s'])>0.001:
            raise ValueError('audio format/duration mismatch')
        audio_members[f'{release}/{item["audio_uri"]}'] = path
    overlaps=audit_splits(rows,ids_by_split)
    pilot = set((repo/f'data/verification/{release}-pilot-ids.txt').read_text().splitlines())
    if not pilot.issubset(rows):
        raise ValueError('pilot outside frozen release')
    groups = {uid:rows[uid]['leakage_group_id'] for uid in ids_by_split['test']}
    mandatory = {uid for uid,r in rows.items() if uid in pilot and r['reference_origin']=='local_human'}
    if profile['original_validation_corrections_missing']:
        mandatory.update(uid for uid in pilot if rows[uid]['split']=='validation')
    plan = V.stratified_plan(groups,pilot,mandatory)
    plan.update(dataset_release=release,base_split_manifest_sha256=profile['split_manifest_sha256'],
        original_validation_corrections_missing=profile['original_validation_corrections_missing'],
        source_profile_sha256=sha256_file(profile_path))
    snapshot=b''.join((json.dumps(rows[uid],ensure_ascii=False,allow_nan=False)+'\n').encode('utf8') for uid in sorted(rows))
    plan['reference_snapshot_sha256']=hashlib.sha256(snapshot).hexdigest()
    name=f'{release}-stratified-test-v1'
    public_plan=repo/f'data/verification/{name}.json'
    V.dump_json(public_plan,plan)
    V.write_new(repo/f'data/verification/{name}.txt',('\n'.join(plan['test_ids'])+'\n').encode())
    output.mkdir(parents=True)
    V.dump_json(output/'verification-plan.json',plan)
    V.write_new(output/'sources/references.full.jsonl',snapshot)
    meta = {'dataset_release':release,'reference_snapshot_sha256':sha256_file(output/'sources/references.full.jsonl'),
        'verification_plan_sha256':sha256_file(public_plan),'public_plan_path':public_plan.relative_to(repo).as_posix(),
        'audio_root':str(audio),'source_profile_sha256':sha256_file(profile_path),
        'reference_variants':{k:v['variant'] for k,v in profile['references'].items()},
        'missing_components':profile['missing_components'],'M6_approval':None,'human_review_records_supplied':0}
    V.dump_json(output/'workspace.json',meta)
    registry = _registry(repo)
    kit_files = []
    batches = [('test-new',sorted(set(plan['test_ids'])-pilot)),('pilot-evidence',sorted(pilot))]
    for label, chosen in batches:
        for offset in range(0,len(chosen),32):
            chunk=chosen[offset:offset+32]
            path=output/f'review-kits/{label}-{offset//32+1:02d}.html'
            render(audio.parent,[rows[u] for u in chunk],
                f'{label}-{offset//32+1}-{meta["reference_snapshot_sha256"][:10]}',registry,path,role='primary')
            kit_files.append({'file':path.relative_to(output).as_posix(),'items':len(chunk),'sha256':sha256_file(path)})
    split_stats = {}
    for split, ids in ids_by_split.items():
        split_stats[split] = {'utterances':len(ids),'hours':sum(inference[u]['duration_s'] for u in ids)/3600,
            'recordings':len({rows[u]['leakage_group_id'] for u in ids}),
            'source_reported_verification_tags':dict(Counter(rows[u]['verification_status'] for u in ids))}
    erratum=(repo/f'data/metadata/{release}-status-erratum-v1.json').read_bytes()
    common={f'{release}/split_manifest.json':split_dir/'split_manifest.json',f'{release}/STATUS-ERRATUM.json':erratum}
    for split in ids_by_split:
        common[f'{release}/{split}.txt']=split_dir/(split+'.txt')
    prefix=output/'working-exports';prefix.mkdir()
    bundles=[]
    members={**common,**audio_members,**{f'{release}/{p.name}':p for p in inference_files.values()}}
    members[f'{release}/README.txt']=b'WORKING inference/audio only. No reference text. Legacy manifest approval claim is corrected by STATUS-ERRATUM.json.\n'
    bundles.append({**_archive(prefix/(release+'-INFERENCE-CLEAN.tar'),members),'recipients':['M2','M3','M4','M5','M6'],'scope':'reference-free inference and audio'})
    for split,role in [('train','TRAIN-REFERENCES'),('validation','VALIDATION-WORKING')]:
        members={**common,f'{release}/{release}.references.{split}.jsonl':reference_sources[split]}
        members[f'{release}/REFERENCE-VARIANT.json']=V.json_bytes(profile['references'][split])
        if split=='train':
            train_dir=inputs_root/profile['train_relational_dir']
            allowed={'provenance.csv','quantifiers.csv','utterances.csv','relationships.csv'}
            members.update({f'{release}/relational/train/{p.name}':p for p in train_dir.glob('*.csv') if p.name in allowed})
        bundles.append({**_archive(prefix/(release+'-'+role+'.tar'),members),
            'recipients':['M3','M4','M5'] if split=='train' else ['validation evaluators/model selectors'],
            'scope':split+' references; never inference input'})
    members={**common,**{f'{release}/{release}.references.{s}.jsonl':p for s,p in reference_sources.items()}}
    members[f'{release}/MISSING-COMPONENTS.json']=V.json_bytes(profile['missing_components'])
    with tarfile.open(inputs_root/profile['test_archive']) as archive:
        for m in archive.getmembers():
            if m.isfile() and '/relational/test/' in m.name and Path(m.name).name in {'provenance.csv','quantifiers.csv','utterances.csv','relationships.csv'}:
                members[f'{release}/relational/test/{Path(m.name).name}']=archive.extractfile(m).read()
    bundles.append({**_archive(prefix/(release+'-EVALUATOR-WORKING.tar'),members),'recipients':['M6/evaluator only'],
        'scope':'available train/test snapshots plus explicitly recovered validation; not the missing original M6 archive'})
    manifest={'status':'working_exports_ready_listening_and_release_approval_pending',
        'dataset_release':release,'base_split_manifest_sha256':profile['split_manifest_sha256'],
        'reference_variants':meta['reference_variants'],'bundles':bundles,
        'missing_components':profile['missing_components'],'human_acknowledgments':{},
        'cloud_sharing_permissions_changed':False,'distribution':'owner must deliver through restricted recipient channels'}
    V.dump_json(output/'working-handoff.json',manifest)
    corrected=sum(rows[u]['reference_origin']=='local_human' for u in pilot if rows[u]['split']=='test')
    n=sum(rows[u]['split']=='test' for u in pilot)
    audit={'status':'preparation_complete_human_evidence_pending','dataset_release':release,'splits':split_stats,
        'audio_files_hashed_and_format_checked':len(inference),'cross_split_exact_intersections':overlaps,
        'near_duplicate_audio_audit':'not performed; singleton duplicate_cluster_id values are not evidence of clustering',
        'speakers':'unknown; recording fallback, not speaker-disjointness',
        'verification_test_target':len(plan['test_ids']),'pilot_test_ids_in_target':len(set(plan['test_ids'])&pilot),
        'new_test_clips':len(set(plan['test_ids'])-pilot),'required_primary_evidence':len(plan['required_primary_ids']),
        'actual_review_records_received':0,'review_pages':kit_files,
        'M1_reported_test_pilot':{'corrected':corrected,'clips':n,'fraction':corrected/n,'wilson_95_interval':V.wilson_interval(corrected,n),
            'evidence':'derived from supplied M1 reference-origin tags; original decision payloads not available'},
        'chronology':plan['chronology'],'reference_status_tags_are_not_new_listening_evidence':True,
        'frozen_audio_or_split_files_modified':False,'M6_approval_claimed':False}
    V.dump_json(output/'preparation-audit.json',audit)
    (output/'README.txt').write_text('TURAS M1 private verification workspace\n\nOpen review-kits/*.html locally. Test-new pages contain the 199 additional test clips; pilot-evidence pages recover the original 50-record evidence if exports remain unavailable. No decisions are prefilled. Enter your own reviewer ID, listen to every assigned clip, then export decisions. Validation pilot text is recovered upstream minutes, NOT M1\'s missing corrected bundle.\n\nImport genuine exports using scripts/m1_verification.py ingest. Generate a pinned adjudication kit after primary coverage is complete, then have a different listener export adjudication decisions. See the repository\'s M1 closeout instructions. Do not publish this directory or share evaluator material with inference inputs.\n\nWorking exports are newly hashed, clean role-separated archives. They do not assert M6 approval or reproduce unavailable review records, original validation corrections or the upstream exclusion ledger. The source freeze bytes and existing experiments are unchanged. The owner must restrict cloud sharing; this program does not change Drive ACLs.\n',encoding='utf8',newline='\n')
    return audit,manifest


def build_adjudication_kit(repo, root, records, output, render):
    meta,refs,plan=load_workspace(root)
    if output.exists():
        raise ValueError('adjudication output must be fresh')
    pinned=V.adjudication_plan(refs,plan,records)
    output.mkdir(parents=True)
    V.dump_json(output/'adjudication-plan.json',pinned)
    primary=V.primary_state(records)
    rows=[]
    for uid in pinned['required_ids']:
        item=dict(refs[uid]);current=primary[uid]
        item['reference_text']=V.record_text(current[-1],refs[uid])
        item['_adjudicates_record_ids']=pinned['primary_bindings'][uid]
        item['_blocked_reviewers']=sorted({r['reviewer_id'] for r in current})
        rows.append(item)
    registry=_registry(repo)
    for offset in range(0,len(rows),32):
        render(Path(meta['audio_root']).parent,rows[offset:offset+32],
            f'adjudication-{pinned["primary_state_sha256"][:12]}-{offset//32+1}',registry,
            output/f'adjudication-{offset//32+1:02d}.html',role='adjudication')
    return pinned


def verify_bundles(workspace):
    """Read every emitted byte and check role boundaries without extracting TARs."""
    workspace = Path(workspace)
    manifest = json.loads((workspace/'working-handoff.json').read_text(encoding='utf8'))
    checked = []
    for bundle in manifest['bundles']:
        path = workspace/'working-exports'/bundle['file']
        if path.name != bundle['file'] or sha256_file(path)!=bundle['sha256'] or path.stat().st_size!=bundle['bytes']:
            raise ValueError('working bundle identity mismatch')
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            if len(set(names))!=len(names) or len(names)!=bundle['members']:
                raise ValueError('duplicate or missing archive members')
            if any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts or
                   any(p.startswith('._') or p in {'.DS_Store','__MACOSX'} for p in Path(m.name).parts) for m in members):
                raise ValueError('unsafe or macOS metadata archive member')
            sums = dict(line.split('  ',1)[::-1] for line in archive.extractfile('SHA256SUMS').read().decode().splitlines())
            if set(sums)!=set(names)-{'SHA256SUMS'}:
                raise ValueError('checksum coverage mismatch')
            reference_splits, inference_count, audio_count = set(), 0, 0
            actual_hashes, audio_expected = {}, {}
            for name in sorted(sums):
                stream = archive.extractfile(name)
                digest = hashlib.sha256()
                while chunk := stream.read(1<<20):
                    digest.update(chunk)
                actual_hashes[name] = digest.hexdigest()
                if actual_hashes[name]!=sums[name]:
                    raise ValueError('member checksum mismatch')
                if name.endswith('.flac'):
                    audio_count += 1
                elif '.inference.' in name or '.references.' in name:
                    rows=[json.loads(line) for line in archive.extractfile(name).read().decode('utf8').splitlines()]
                    split = Path(name).name.split('.')[-2]
                    ids=archive.extractfile(manifest['dataset_release']+'/'+split+'.txt').read().decode().splitlines()
                    if [r['utterance_id'] for r in rows]!=ids or ('.references.' in name and any(r['split']!=split for r in rows)):
                        raise ValueError('bundle IDs/order/split mismatch')
                    if '.inference.' in name:
                        for row in rows:
                            InferenceInput.from_row(row)
                            audio_expected[manifest['dataset_release']+'/'+row['audio_uri']] = row['audio_sha256']
                        inference_count += len(rows)
                    else:
                        reference_splits.add(split)
            if any(actual_hashes.get(name)!=digest for name,digest in audio_expected.items()):
                raise ValueError('bundled audio does not match inference hashes')
            filename = bundle['file']
            expected = set() if 'INFERENCE-CLEAN' in filename else {'train'} if 'TRAIN-REFERENCES' in filename else {'validation'} if 'VALIDATION-WORKING' in filename else {'train','validation','test'}
            if reference_splits!=expected:
                raise ValueError('reference access-role mismatch')
            if 'INFERENCE-CLEAN' in filename and (inference_count!=5450 or audio_count!=5450 or any('/relational/' in n for n in names)):
                raise ValueError('inference bundle boundary/coverage mismatch')
            if 'TRAIN-REFERENCES' in filename and any('/relational/' in n and '/relational/train/' not in n for n in names):
                raise ValueError('held-out evidence in training bundle')
        checked.append({'file':bundle['file'],'sha256':bundle['sha256'],'members_verified':len(sums),
            'audio_members':audio_count,'inference_rows':inference_count,'reference_splits':sorted(reference_splits)})
    return {'status':'all_working_bundle_bytes_and_roles_verified','bundles':checked,
        'human_listening_or_delivery_proved':False,'cloud_acl_checked_or_changed':False}


def validation_tables(repo, inputs_root, profile_path, output):
    """Rebuild only evaluator validation evidence from the pinned working minutes."""
    import pandas as pd
    from turkish_asr_rdl.data.relational import write_split_tables
    from turkish_asr_rdl.data.screening import Lexicon

    profile=json.loads(Path(profile_path).read_text(encoding='utf8'))
    release=profile['dataset_release'];source=profile['references']['validation']
    reference=(Path(inputs_root)/source['path']).resolve()
    if sha256_file(reference)!=source['sha256']:
        raise ValueError('validation reference source hash mismatch')
    rows=read_jsonl(reference)
    ids=(Path(repo)/'data/splits'/release/'validation.txt').read_text(encoding='utf8').splitlines()
    if [r['utterance_id'] for r in rows]!=ids or any(r['split']!='validation' or r['dataset_release']!=release for r in rows):
        raise ValueError('validation IDs/order/split mismatch')
    output=Path(output).resolve();repo=Path(repo).resolve()
    audio=(Path(inputs_root)/profile['inference_root']).resolve()
    if output.exists() or output.is_relative_to(audio) or audio.is_relative_to(output):
        raise ValueError('supplement output must be fresh and separate from inference audio')
    if output.is_relative_to(repo) and not any(output.is_relative_to(repo/p) for p in ['data/private','artifacts']):
        raise ValueError('reference supplement cannot target a public repository path')
    stage=output.with_name(output.name+'.staging-'+uuid.uuid4().hex)
    stage.mkdir(parents=True)
    V.write_new(stage/'references.validation.jsonl',reference.read_bytes())
    tables=stage/'relational/validation'
    counts=write_split_tables(pd.DataFrame(rows),'validation',
        Lexicon.load(repo/'configs/datasets/turas-domain-lexicon-v0.1.json'),release,source['sha256'],tables)
    provenance=tables/'provenance.csv'
    with provenance.open(encoding='utf8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;records=list(reader)
    records[0].update(source_uri='references.validation.jsonl',source_revision=source['variant'],
        available_at_utc=datetime.now(timezone.utc).isoformat())
    with provenance.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(records)
    def table(path):
        with path.open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))
    utterances=table(tables/'utterances.csv')
    if [r['utterance_id'] for r in utterances]!=ids or any(r['split']!='validation' for r in utterances):
        raise ValueError('generated validation utterance scope mismatch')
    relations=table(tables/'relationships.csv')
    if any(r['access_split']!='validation' or r['evidence_utterance_id'] not in set(ids) for r in relations):
        raise ValueError('generated validation evidence scope mismatch')
    metadata={'status':'working_validation_evidence_only_not_human_verified',
        'dataset_release':release,'reference_variant':source['variant'],'reference_sha256':source['sha256'],
        'source_reported_verification_tags':dict(Counter(r['verification_status'] for r in rows)),
        'counts':counts,'csv_sha256':{p.name:sha256_file(p) for p in sorted(tables.glob('*.csv'))},
        'access':'evaluator/validation selection only; never training graph or inference inputs',
        'original_M1_corrected_validation_bundle_recovered':False,'new_human_review_records':0}
    V.dump_json(stage/'REFERENCE-VARIANT.json',metadata)
    members={release+'/references.validation.jsonl':stage/'references.validation.jsonl',
        release+'/REFERENCE-VARIANT.json':stage/'REFERENCE-VARIANT.json'}
    members.update({release+'/relational/validation/'+p.name:p for p in sorted(tables.glob('*.csv'))})
    bundle=_archive(stage/(release+'-VALIDATION-RELATIONAL-WORKING.tar'),members)
    metadata['bundle']=bundle
    V.dump_json(stage/'supplement-manifest.json',metadata)
    if output.exists():raise ValueError('supplement destination changed')
    os.replace(stage,output)
    return metadata
