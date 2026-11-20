"""One frozen all-system test evaluation using shared scoring and domain metrics."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from turkish_asr_rdl.data.screening import Lexicon
from turkish_asr_rdl.evaluation.domain import evaluate_domain
from turkish_asr_rdl.evaluation.join import join
from turkish_asr_rdl.evaluation.metrics import score_pair,corpus_rates,UtteranceScore
from turkish_asr_rdl.evaluation.normalize import RAW_V1,TR_NORMALIZED_V1
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.training.acoustic import read_jsonl,sha256_file,write_json


def summarize(scores):
    if not scores:return {'status':'unavailable','utterances':0}
    result=corpus_rates(scores)
    return {**asdict(result),'reference_words':result.words.ref_len,'reference_characters':result.chars.ref_len}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--inference-index',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must be fresh')
    if sha256_file(args.protocol)!=args.protocol_sha256:raise ValueError('protocol checksum mismatch')
    p=json.loads(args.protocol.read_text());index=json.loads(args.inference_index.read_text())
    refpath=args.reference_dir/'references.test.jsonl'
    if sha256_file(refpath)!=p['test_reference']['payload_sha256']:raise ValueError('test reference checksum mismatch')
    if index['protocol_sha256']!=args.protocol_sha256 or index['status']!='inference_complete_gold_still_closed':
        raise ValueError('incomplete frozen inference')
    ids=Path(p['inputs']['test_ids']).read_text().splitlines();refs=read_jsonl(refpath)
    if [r['utterance_id'] for r in refs]!=ids:raise ValueError('frozen test identity mismatch')
    if sha256_file(Path(p['lexicon']['path']))!=p['lexicon']['sha256']:raise ValueError('lexicon checksum mismatch')
    lexicon=Lexicon.load(p['lexicon']['path'])
    verified={r['utterance_id'] for r in refs if r.get('verification_status') in ['verified','adjudicated']}
    groups=sorted({r['leakage_group_id'] for r in refs})
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[];group_scores={}
    accuracy=[r for r in index['runs'] if r['mode'] in ['accuracy','cached_nbest_selection']]
    for entry in accuracy:
        directory=Path(entry['directory']);done=json.loads((directory/'completed.json').read_text())
        if sha256_file(directory/'completed.json')!=entry['completed_sha256']:raise ValueError('completed run changed')
        for name,digest in done['files'].items():
            if sha256_file(directory/name)!=digest:raise ValueError('output changed after frozen inference')
        predictions=read_jsonl(directory/'predictions.jsonl');nbest=read_jsonl(directory/'nbest.jsonl')
        if not validate_predictions(predictions,ids).ok or not validate_nbest(nbest,ids).ok:raise ValueError('export contract failed')
        joined=join(predictions,refs,split='test',split_manifest_sha256=p['split_manifest_sha256'])
        domain=evaluate_domain(joined,lexicon);containing=set(domain.pop('term_containing_utterance_ids'))
        scores={view:[score_pair(r.utterance_id,r.reference,r.hypothesis,view,r.status) for r in joined]
                for view in [RAW_V1,TR_NORMALIZED_V1]}
        subset=lambda view,wanted:[s for s in scores[view] if s.utterance_id in wanted]
        domain['term_containing_utterances']={'count':len(containing),
            'metrics':{view:summarize(subset(view,containing)) for view in scores}}
        system=done['system_id'];group_scores[system]=[]
        for group in groups:
            gs=[s for s,r in zip(scores[TR_NORMALIZED_V1],joined) if r.leakage_group_id==group]
            rates=corpus_rates(gs)
            group_scores[system].append(UtteranceScore(group,TR_NORMALIZED_V1,'ok',rates.words,rates.chars))
        coverage=Counter(row['actual_n'] for row in nbest)
        relational=[c['score_details']['rdl'] for row in nbest for c in row['candidates'] if 'rdl' in c['score_details']]
        report={'system_id':system,'protocol_sha256':args.protocol_sha256,'reference_variant':p['test_reference']['variant'],
            'reference_sha256':sha256_file(refpath),'predictions_sha256':sha256_file(directory/'predictions.jsonl'),
            'nbest_sha256':sha256_file(directory/'nbest.jsonl'),'metrics':{view:summarize(values) for view,values in scores.items()},
            'm1_reported_verified_subset':{'utterances':len(verified),'metrics':{view:summarize(subset(view,verified)) for view in scores}},
            'domain':domain,'nbest_coverage':{'requested_n':p['decoder']['requested_n'],'actual_n_distribution':dict(sorted(coverage.items())),
             'hypotheses':sum(row['actual_n'] for row in nbest)},
            'relation_coverage':{'hypotheses_with_rdl_scores':len(relational),
             'hypotheses_with_linked_entities':sum(d['linked_entities']>0 for d in relational),
             'linked_entity_counts':sum(d['linked_entities'] for d in relational),
             'relation_queries':sum(d['relation_count'] for d in relational),
             'quantifier_queries':sum(d['quantifier_count'] for d in relational),
             'unknown_or_contextless_quantifiers':sum(d['unknown_or_contextless_quantifiers'] for d in relational)},
            'error_rows':sum(r['status']=='error' for r in predictions),'mode':done['mode']}
        # Audit the main ASR systems with an independent edit-distance library.
        if done['mode']=='accuracy':
            import jiwer
            from turkish_asr_rdl.evaluation.normalize import normalize,word_tokens
            for view in scores:
                rtext=[normalize(r.reference,view) for r in joined];htext=[normalize(r.hypothesis,view) for r in joined]
                w=jiwer.process_words([' '.join(word_tokens(s)) for s in rtext],[' '.join(word_tokens(s)) for s in htext])
                tr=jiwer.ReduceToListOfListOfChars();c=jiwer.process_characters(rtext,htext,reference_transform=tr,hypothesis_transform=tr)
                if abs(w.wer-report['metrics'][view]['wer'])>1e-12 or abs(c.cer-report['metrics'][view]['cer'])>1e-12:
                    raise ValueError('independent final WER/CER cross-check failed')
            report['independent_jiwer_crosscheck']='passed both views'
        write_json(args.output/(system+'.json'),report);reports.append(report)
        print(json.dumps({'scored_system':system,'rows':len(ids),'normalized_wer':report['metrics'][TR_NORMALIZED_V1]['wer']}),flush=True)
    timings=[]
    for entry in [r for r in index['runs'] if r['mode']=='timing']:
        directory=Path(entry['directory']);m=json.loads((directory/'manifest.json').read_text())
        records=[r for r in read_jsonl(directory/'timing.jsonl') if r.get('record_type')=='batch' and not r['warmup']]
        latency=[r['end_to_end_s'] for r in records]
        t={'system_id':entry['system_id'],**m['evaluation_timing'],'environment':m['environment'],
            'request_measurements':len(records),'latency_p50_s':float(np.percentile(latency,50)),
            'latency_p95_s':float(np.percentile(latency,95)),
            'peak_reserved_gib':max((r['peak_reserved_gib'] or 0) for r in records),
            'vram_spill_suspected':any(r['vram_spill_suspected'] for r in records),
            'timing_sha256':sha256_file(directory/'timing.jsonl'),'manifest_sha256':sha256_file(directory/'manifest.json')}
        t['per_repeat_rtf_std']=float(np.std(t['per_repeat_end_to_end_rtf']))
        t['rtfx']=1/t['end_to_end_rtf']
        timings.append(t)
    rng=np.random.default_rng(p['bootstrap']['seed']);samples=rng.integers(0,len(groups),size=(p['bootstrap']['replicates'],len(groups)))
    pairs=list(p['paired_comparisons']);intervals=[]
    for pair in pairs:
        a,b=group_scores[pair['system']],group_scores[pair['control']]
        diffs=[corpus_rates([a[i] for i in sample]).wer-corpus_rates([b[i] for i in sample]).wer for sample in samples]
        lo,hi=np.percentile(diffs,[2.5,97.5]).tolist()
        intervals.append({**pair,'normalized_wer_difference_percentage_points':100*(corpus_rates(a).wer-corpus_rates(b).wer),
            'paired_recording_bootstrap_95_interval_percentage_points':[100*lo,100*hi]})
    result={'status':'final_frozen_working_benchmark_complete','protocol_sha256':args.protocol_sha256,
        'dataset_release':p['dataset_release'],'reference_variant':p['test_reference']['variant'],
        'reference_sha256':sha256_file(refpath),'test_utterances':len(ids),'recording_groups':len(groups),
        'verification_status_counts':dict(Counter(r.get('verification_status','missing') for r in refs)),
        'reference_origin_counts':dict(Counter(r.get('reference_origin','missing') for r in refs)),
        'systems':reports,'headline_timing':timings,'bootstrap':{**p['bootstrap'],'groups':len(groups),'comparisons':intervals},
        'limitations':p['limitations']}
    write_json(args.output/'summary.json',result)


if __name__=='__main__':main()
