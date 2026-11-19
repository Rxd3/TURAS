"""Freeze final systems and timing IDs using metadata only, before test gold access."""
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import tarfile

from turkish_asr_rdl.decoding.adapted import verify_checkpoint
from turkish_asr_rdl.rdl.train import load_model
from turkish_asr_rdl.training.acoustic import sha256_file,read_jsonl,write_json

ROOT=Path('artifacts/owner-m45');VALIDATION=ROOT/'validation-recovery';FINAL=ROOT/'final'
RELEASE='turas-tr-econ-legis-v0.1'


def main():
    target=Path('configs/evaluation/m45_final_protocol.json')
    if target.exists():raise ValueError('final protocol already exists; never silently refreeze')
    FINAL.mkdir(parents=True,exist_ok=True)
    split=Path('data/splits')/RELEASE
    split_sha=sha256_file(split/'split_manifest.json')
    if split_sha!='a8fff65b99fcf11d9452685e438e250207032ecfe11152741562147126d095b8':raise ValueError('release mismatch')
    selection=json.loads((VALIDATION/'working-checkpoint-selection.json').read_text())
    archive=Path('data/private')/RELEASE/'test-archive'/f'{RELEASE}-TEST-split-EVALUATOR-ONLY.tar'
    archive_sha='01ad58784cc27489db6048b573d2412ea193ec2370f2d447eddb0e841cf06d97'
    if sha256_file(archive)!=archive_sha:raise ValueError('test archive hash mismatch')
    with tarfile.open(archive) as tar:
        def member(suffix):
            items=[m for m in tar.getmembers() if m.isfile() and not Path(m.name).name.startswith('._') and m.name.endswith(suffix)]
            if len(items)!=1:raise ValueError('ambiguous/missing archive metadata '+suffix)
            return items[0]
        refmember=member('.references.test.jsonl')
        sums=tar.extractfile(member('/SHA256SUMS')).read().decode('utf-8')
        checks={line.split(maxsplit=1)[1].lstrip('*').removeprefix('./'):line.split(maxsplit=1)[0] for line in sums.splitlines()}
        reference_sha=checks[refmember.name.removeprefix('./')]
        if tar.extractfile(member('/split_manifest.json')).read()!=(split/'split_manifest.json').read_bytes():raise ValueError('archive manifest mismatch')
        if tar.extractfile(member('/test.txt')).read()!=(split/'test.txt').read_bytes():raise ValueError('archive IDs mismatch')
        group_member=member('/relational/test/utterances.csv')
        stream=io.TextIOWrapper(tar.extractfile(group_member),encoding='utf-8',newline='')
        reader=csv.DictReader(stream)
        if set(reader.fieldnames)!=set(['utterance_id','dataset_release','split','leakage_group_id','provenance_id']):
            raise ValueError('group metadata must not contain protected reference fields')
        groups=list(reader)
    ids=(split/'test.txt').read_text().splitlines()
    if sorted(r['utterance_id'] for r in groups)!=ids or any(r['split']!='test' or r['dataset_release']!=RELEASE for r in groups):
        raise ValueError('metadata group coverage mismatch')
    metadata=Path('data/metadata/m45-test-recording-groups.csv')
    with metadata.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['utterance_id','leakage_group_id'],lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in sorted(groups,key=lambda r:r['utterance_id']))
    audio=Path('data/private')/RELEASE/'inference'/RELEASE
    testfile=audio/f'{RELEASE}.inference.test.jsonl';warmfile=audio/f'{RELEASE}.inference.train.jsonl'
    testrows=read_jsonl(testfile)
    if [r['utterance_id'] for r in testrows]!=ids:raise ValueError('test inference ID mismatch')
    duration={r['utterance_id']:r['duration_s'] for r in testrows}
    grouped=defaultdict(list)
    for r in groups:grouped[r['leakage_group_id']].append(r['utterance_id'])
    rng=random.Random(2026);order=sorted(grouped);rng.shuffle(order)
    for key in order:rng.shuffle(grouped[key])
    chosen=[];seconds=0.0;position=0
    while seconds<1800:
        for key in order:
            if position<len(grouped[key]):
                uid=grouped[key][position];chosen.append(uid);seconds+=duration[uid]
                if seconds>=1800:break
        position+=1
    chosen.sort();timing=split/'m45-timing-30min.txt';timing.write_bytes(('\n'.join(chosen)+'\n').encode())
    model_specs=[]
    for system,family,config in [('B1-whisper-small','whisper','whisper_small_lora'),('B4-xlsr-tr-cv7','ctc','ctc_xlsr_turkish_vocab_extension')]:
        m=json.loads((Path('configs/fine-tuning')/(config+'.json')).read_text())['model']
        model_specs.append({'system_id':system,'kind':'pretrained','family':family,'model':m})
    mapping={'whisper-lora-seed17':'F1-whisper-lora','whisper-full-seed17':'F2-whisper-full','ctc-extended-seed17':'F3-ctc-vocab'}
    chosen_models={}
    for s in selection['selections']:
        best=s['selected'];verify_checkpoint(Path(best['checkpoint']),best['checkpoint_manifest_sha256'])
        if sha256_file(Path(best['score_report']))!=best['score_report_sha256']:raise ValueError('validation evidence changed')
        name=mapping[s['model']];chosen_models[name]=(s,best)
        model_specs.append({'system_id':name,'kind':'adapted','family':'ctc' if name.startswith('F3') else 'whisper',
            'checkpoint':best['checkpoint'],'checkpoint_manifest_sha256':best['checkpoint_manifest_sha256'],
            'selected_epoch':best['epoch'],'validation_score_report_sha256':best['score_report_sha256']})
    graph_root=VALIDATION/'corrected-linker-v2'
    graph_paths={'full':graph_root/'graph-full','no-quantities':graph_root/'graph-no-quantities',
                 'no-comention':graph_root/'graph-no-comention','random':FINAL/'random-graph-control'}
    for path in graph_paths.values():load_model(path)
    rescores=[];positive_evidence={};pairs=[{'system':'F1-whisper-lora','control':'B1-whisper-small'},
        {'system':'F2-whisper-full','control':'B1-whisper-small'},{'system':'F3-ctc-vocab','control':'B4-xlsr-tr-cv7'},
        {'system':'F2-whisper-full','control':'F1-whisper-lora'}]
    for name,(s,best) in chosen_models.items():
        source=graph_root/f"full--{s['model']}-epoch-{best['epoch']}"/'selection.json'
        v=json.loads(source.read_text());positive=min((t for t in v['trials'] if t['rdl_weight']>0),
            key=lambda t:(t['metrics']['tr-normalized-v1']['wer'],t['metrics']['raw-v1']['wer'],t['rdl_weight']))
        gamma=positive['rdl_weight'];positive_evidence[name]={'report_sha256':sha256_file(source),'best_positive_gamma':gamma,
            'criterion':'secondary RDL-enabled condition: best positive validation gamma; original one-best remains primary'}
        for suffix,graph,weight in [('acoustic-control','full',0.0),('rdl-full','full',gamma),
            ('rdl-no-quantities','no-quantities',gamma),('rdl-no-comention','no-comention',gamma),('rdl-random','random',gamma)]:
            gp=graph_paths[graph]
            condition={'system_id':name+'--'+suffix,'base_system':name,'graph':gp.as_posix(),
                'graph_manifest_sha256':sha256_file(gp/'artifact-manifest.json'),
                'fusion':{'acoustic_weight':1.0,'lm_weight':0.0,'rdl_weight':weight,'acoustic_exponent':1.0,'lm_exponent':1.0}}
            rescores.append(condition)
            if suffix=='rdl-full':
                model_specs.append({**condition,'kind':'relational'})
                pairs.append({'system':condition['system_id'],'control':name+'--acoustic-control'})
                pairs.append({'system':condition['system_id'],'control':name})
            elif suffix!='acoustic-control':pairs.append({'system':condition['system_id'],'control':name+'--rdl-full'})
    critical=[q for q in Path('src/turkish_asr_rdl').rglob('*.py')]
    critical += [Path('scripts')/name for name in ['m45_final_infer.py','m45_final_rescore.py','m45_final_evaluate.py',
        'm45_stage_final_references.py','m45_final_handoff.py','m45_prepare_final.py','m45_random_graph_control.py']]
    protocol={'schema_version':'0.1','status':'frozen_for_final_evaluation','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'authority':'project owner instruction to finish M4/M5; docs/owner_decisions.md D38-D40',
        'dataset_release':RELEASE,'split_manifest_sha256':split_sha,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'critical_code_sha256':{q.as_posix():sha256_file(q) for q in sorted(critical)},
        'inputs':{'audio_root':audio.as_posix(),'test_inference':testfile.as_posix(),'test_inference_sha256':sha256_file(testfile),
         'test_ids':(split/'test.txt').as_posix(),'test_ids_sha256':sha256_file(split/'test.txt'),
         'warmup_inference':warmfile.as_posix(),'warmup_inference_sha256':sha256_file(warmfile),
         'timing_ids':timing.as_posix(),'timing_ids_sha256':sha256_file(timing),
         'test_groups':metadata.as_posix(),'test_groups_sha256':sha256_file(metadata)},
        'test_reference':{'variant':'m1-test-minutes-archive-01ad5878-v1','archive_sha256':archive_sha,
         'archive_member':refmember.name,'payload_sha256':reference_sha,'gold_bodies_unopened_at_freeze':True,
         'access_rule':'only after all frozen accuracy, cached-selection and timing outputs have completed'},
        'validation_reference':{'variant':selection['reference_variant'],'manifest_sha256':selection['reference_manifest_sha256']},
        'selection_evidence':positive_evidence,'systems':model_specs,'adapted_system_ids':list(chosen_models),
        'rescore_conditions':rescores,'paired_comparisons':pairs,
        'decoder':{'device':'cuda','precision':'bf16','requested_n':5,'beam_size':10,'max_new_tokens':440},
        'missing_probability':0.1,'lexicon':{'path':'configs/datasets/turas-domain-lexicon-v0.1.json',
         'sha256':'8ef903fe74135897a9d3781966c5e484c0711a6679d72346efb1ec89eff6df94','term_metric_version':'canonical-occurrence-m1-v1'},
        'timing':{'repeats':3,'batch_size':1,'warmup_batches':3,'utterances':len(chosen),'duration_s':seconds,
         'recording_groups':len({r['leakage_group_id'] for r in groups if r['utterance_id'] in set(chosen)}),
         'algorithm':'seed2026 shuffled recordings and within-recording IDs; round-robin until >=1800s; then sorted IDs',
         'concurrency':1,'cache':'local files, OS caches not flushed; no downloads in measurement','cooldown_seconds':30},
        'bootstrap':{'seed':2026,'replicates':2000,'method':'paired recording-group percentile bootstrap'},
        'limitations':['Working benchmark on edited parliamentary minutes, not fully verified verbatim speech',
         'M1 reported pilot listening corrections; no independent M6 release approval is claimed',
         'Unknown speakers: recording-disjoint splits do not prove speaker disjointness',
         'Single training seed17; no training-seed uncertainty or robustness claim',
         'Validation used recovered upstream minutes without three private M1 corrections',
         'Quantities are automatic digit-based amount/percentage diagnostics, not human quantity annotations',
         'This completes M4/M5 matched-parent work; other baseline-family tracks remain separate M2/M3 responsibilities',
         'Diagnostic relation-removal studies were chosen after initial validation results; final test conditions were frozen before gold access']}
    timing_order=[s['system_id'] for s in model_specs];random.Random(2026).shuffle(timing_order)
    protocol['timing']['system_order']=timing_order
    target.parent.mkdir(exist_ok=True,parents=True);write_json(target,protocol)
    write_json(FINAL/'protocol-lock.json',{'protocol':target.as_posix(),'sha256':sha256_file(target),
        'frozen_at_utc':protocol['frozen_at_utc'],'test_gold_opened':False})
    print(json.dumps({'status':'frozen','protocol_sha256':sha256_file(target),'test_gold_opened':False,
        'native_systems':5,'rescore_conditions':len(rescores),'timing_systems':len(model_specs),
        'timing_clips':len(chosen),'timing_seconds':seconds}))


if __name__=='__main__':main()
