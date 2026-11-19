"""Supplementary post-freeze artifact, candidate, timing and exact-overlap audit.

No selection, metric-policy changes or new model runs. Requires the entire
frozen benchmark/handoff to have completed before reading test references.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from turkish_asr_rdl.evaluation.normalize import normalize,word_tokens,RAW_V1,TR_NORMALIZED_V1
from turkish_asr_rdl.training.acoustic import read_jsonl,sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/owner-m45/final'))
    parser.add_argument('--output',type=Path,default=Path('results/final/m45_final_integrity_audit.json'))
    args=parser.parse_args()
    if args.output.exists():parser.error('audit output must be fresh')
    state=json.loads((args.root/'status.json').read_text())
    if state['status']!='completed_m4_m5_final_benchmark_and_handoff' or not state['gold_opened']:
        raise ValueError('audit waits for the complete frozen benchmark and reference gate')
    protocol=Path('configs/evaluation/m45_final_protocol.json');p=json.loads(protocol.read_text())
    if sha256_file(protocol)!=state['protocol_sha256']:raise ValueError('protocol mismatch')
    for path,digest in p['critical_code_sha256'].items():
        if sha256_file(Path(path))!=digest:raise ValueError('frozen code changed')
    index=json.loads((args.root/'inference-index.json').read_text());inventory=[]
    entries={}
    for entry in index['runs']:
        directory=Path(entry['directory']);done=directory/'completed.json'
        if sha256_file(done)!=entry['completed_sha256']:raise ValueError('completed record changed')
        for name,digest in json.loads(done.read_text())['files'].items():
            if sha256_file(directory/name)!=digest:raise ValueError('delivered output changed')
        inventory.append({'system_id':entry['system_id'],'mode':entry['mode'],'completed_sha256':entry['completed_sha256']})
        entries[(entry['system_id'],entry['mode'])]=directory
    candidate_checks=[]
    fields=['candidate_id','rank','text','token_ids','token_count','acoustic_log_score','acoustic_score_kind','decoder_sequence_score']
    for spec in p['rescore_conditions']:
        native=read_jsonl(entries[(spec['base_system'],'accuracy')]/'nbest.jsonl')
        rescored=read_jsonl(entries[(spec['system_id'],'cached_nbest_selection')]/'nbest.jsonl')
        preserved=len(native)==len(rescored) and all(
            a['utterance_id']==b['utterance_id'] and
            [{k:c[k] for k in fields} for c in a['candidates']]==[{k:c[k] for k in fields} for c in b['candidates']]
            for a,b in zip(native,rescored))
        if not preserved:raise ValueError('fixed-candidate preservation failed: '+spec['system_id'])
        candidate_checks.append({'system_id':spec['system_id'],'utterances':len(native),'all_native_candidates_and_acoustic_scores_preserved':True})
    timing_checks=[]
    for spec in p['systems']:
        mode='cached_nbest_selection' if spec['kind']=='relational' else 'accuracy'
        full={r['utterance_id']:r for r in read_jsonl(entries[(spec['system_id'],mode)]/'predictions.jsonl')}
        timing=read_jsonl(entries[(spec['system_id'],'timing')]/'predictions.jsonl')
        changed=sum((r['hypothesis_text'],r['status'])!=(full[r['utterance_id']]['hypothesis_text'],full[r['utterance_id']]['status']) for r in timing)
        timing_checks.append({'system_id':spec['system_id'],'timing_utterances':len(timing),'changed_first_repeat_transcripts':changed,
            'scope':'first timed repeat versus full-test accuracy/cached selection; later-repeat hypotheses are not retained by the shared harness'})
    release=p['dataset_release'];private=Path('data/private')/release
    refpath=private/'evaluator-final/references.test.jsonl'
    if sha256_file(refpath)!=p['test_reference']['payload_sha256']:raise ValueError('reference mismatch')
    refs=read_jsonl(refpath);trainpath=private/'train'/release/(release+'.references.train.jsonl')
    if sha256_file(trainpath)!='f4992d2951389ac243c281d2875511cfa0669d026d8698993db48b30b09a4508':raise ValueError('train reference mismatch')
    train=read_jsonl(trainpath)
    overlap={}
    for name,transform in [('literal_reference_string',lambda x:x),
            ('raw_word_token_sequence',lambda x:tuple(word_tokens(normalize(x,RAW_V1)))),
            ('turkish_normalized_reference',lambda x:normalize(x,TR_NORMALIZED_V1))]:
        known={transform(r['reference_text']) for r in train}
        matching=[r for r in refs if transform(r['reference_text']) in known]
        overlap[name]={'test_utterances_also_in_train':len(matching),
            'distinct_matching_test_sentences':len({transform(r['reference_text']) for r in matching})}
    def common(key):return len({r[key] for r in train}&{r[key] for r in refs})
    handoff=json.loads((args.root/'handoff/handoff.json').read_text())
    if sha256_file(Path(handoff['archive']))!=handoff['sha256']:raise ValueError('handoff archive hash mismatch')
    delivered=json.loads((args.root/'handoff/artifact-index.json').read_text())
    for path,record in delivered['files'].items():
        if sha256_file(Path(path))!=record['sha256']:raise ValueError('handoff member hash mismatch')
    result={'status':'supplementary_final_integrity_audit_complete','protocol_sha256':state['protocol_sha256'],
        'audit_code_sha256':sha256_file(Path(__file__)),'audit_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'post-freeze descriptive integrity/overlap audit; no model selection or metric-policy changes',
        'run_inventory':inventory,'fixed_candidate_checks':candidate_checks,'timing_prediction_consistency':timing_checks,
        'train_test_overlap':{'train_reference_sha256':sha256_file(trainpath),'test_reference_sha256':sha256_file(refpath),
            'utterance_ids':common('utterance_id'),'recording_groups':common('leakage_group_id'),'exact_audio_hashes':common('audio_sha256'),
            'sentence_checks':overlap,'speakers':'unknown; recording disjointness does not prove speaker disjointness'},
        'private_handoff':handoff,'delivered_files_verified':len(delivered['files']),
        'test_reference_status_counts':dict(Counter(r.get('verification_status','missing') for r in refs))}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)
    print(json.dumps({'status':result['status'],'runs':len(inventory),'candidate_conditions':len(candidate_checks),
        'overlap':result['train_test_overlap'],'changed_timing_transcripts':sum(r['changed_first_repeat_transcripts'] for r in timing_checks)}))


if __name__=='__main__':main()
