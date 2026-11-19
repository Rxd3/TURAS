"""Contract-valid cached candidate selections; timings explicitly exclude ASR."""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

from turkish_asr_rdl.decoding.interface import Candidate,Hypothesis,InferenceInput
from turkish_asr_rdl.decoding.relational import RelationalAsrAdapter
from turkish_asr_rdl.decoding.export import RunContext
from turkish_asr_rdl.decoding.memory import NoVram
from turkish_asr_rdl.decoding.timing import run_timed,write_run_outputs
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.training.acoustic import read_jsonl,sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must be new')
    if sha256_file(args.protocol)!=args.protocol_sha256:raise ValueError('protocol hash mismatch')
    p=json.loads(args.protocol.read_text());spec=next(s for s in p['rescore_conditions'] if s['system_id']==args.condition)
    if p['status'] not in ['frozen_for_final_evaluation','frozen_validation_preflight']:raise ValueError('protocol not frozen')
    split=p.get('evaluation_split','test')
    if p['status']=='frozen_validation_preflight' and split!='validation':raise ValueError('preflight uses validation only')
    completed=json.loads((args.source/'completed.json').read_text())
    if completed['system_id']!=spec['base_system'] or completed['protocol_sha256']!=args.protocol_sha256:
        raise ValueError('source acoustic run does not match the frozen condition')
    for name,digest in completed['files'].items():
        if sha256_file(args.source/name)!=digest:raise ValueError('source artifact checksum mismatch')
    ids=Path(p['inputs']['test_ids']).read_text().splitlines()
    testpath=Path(p['inputs']['test_inference']);warmpath=Path(p['inputs']['warmup_inference'])
    if sha256_file(testpath)!=p['inputs']['test_inference_sha256'] or sha256_file(warmpath)!=p['inputs']['warmup_inference_sha256']:
        raise ValueError('frozen inference metadata hash mismatch')
    warmrows=read_jsonl(warmpath)[:3]
    testrows=[r for r in read_jsonl(testpath) if r['utterance_id'] in set(ids)]
    if [r['utterance_id'] for r in testrows]!=ids:raise ValueError('test ID coverage mismatch')
    candidates=read_jsonl(args.source/'nbest.jsonl');warms=read_jsonl(args.source/'warmup-nbest.jsonl')
    if not validate_nbest(candidates,ids).ok or not validate_nbest(warms,[r['utterance_id'] for r in warmrows]).ok:
        raise ValueError('cached source candidates violate the shared contract')
    if (any(r['split']!=split for r in candidates) or any(r['split']!='train' for r in warms)
            or any(r['dataset_release']!=p['dataset_release'] or r['split_manifest_sha256']!=p['split_manifest_sha256'] for r in candidates+warms)):
        raise ValueError('cached source role/release mismatch')
    class CachedCandidates:
        requested_n=p['decoder']['requested_n']
        system_id=spec['base_system']
        model_artifact_sha256=candidates[0]['model_artifact_sha256']
        score_convention_id=candidates[0]['score_convention_id']
        config={'kind':'cached_model_derived_nbest','measurement_scope':'candidate lookup/selection only; acoustic inference excluded',
                'device':'cpu','source_nbest_sha256':sha256_file(args.source/'nbest.jsonl'),
                'source_decoder_config_sha256':candidates[0]['decoder_config_sha256']}
        decoder_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        def load(self):self.rows={r['utterance_id']:r for r in candidates+warms}
        def transcribe(self,batch):
            result=[]
            for item in batch:
                row=self.rows[item.utterance_id]
                cs=tuple(Candidate(c['candidate_id'],c['rank'],c['text'],tuple(c['token_ids']),c['acoustic_log_score'],
                    c['acoustic_score_kind'],c['decoder_sequence_score'],c['score_details']) for c in row['candidates'])
                text=cs[0].text if cs else ''
                result.append(Hypothesis(item.utterance_id,'ok' if text else 'empty',text,cs))
            return result
    adapter=RelationalAsrAdapter(CachedCandidates(),Path(spec['graph']),spec['graph_manifest_sha256'],
        dataset_release=p['dataset_release'],split_manifest_sha256=p['split_manifest_sha256'],
        fusion=spec['fusion'],missing_probability=p['missing_probability'])
    adapter.system_id=spec['system_id']
    timed=[InferenceInput.from_row(r) for r in testrows];warmup=[InferenceInput.from_row(r) for r in warmrows]
    def cpu_only_sync():return None
    runid=spec['system_id']+'-'+uuid.uuid4().hex[:8]
    run=run_timed(adapter,timed,warmup,audio_root=Path(p['inputs']['audio_root']),run_id=runid,repeats=1,batch_size=1,
                   sync=cpu_only_sync,memory=NoVram())
    ctx=RunContext(runid,spec['system_id'],p['dataset_release'],split,p['split_manifest_sha256'],
        adapter.model_artifact_sha256,adapter.decoder_config_sha256)
    files=write_run_outputs(args.output,run,timed,ctx,adapter,command='scripts/m45_final_rescore.py')
    predictions=read_jsonl(args.output/'predictions.jsonl')
    if not validate_predictions(predictions,ids).ok or not validate_nbest(read_jsonl(args.output/'nbest.jsonl'),ids).ok:
        raise ValueError('cached-selection output contract failed')
    if any(r['status']=='error' for r in predictions):raise ValueError('selection error rows; no scoring allowed')
    write_json(args.output/'completed.json',{'status':'completed','mode':'cached_nbest_selection',
        'system_id':spec['system_id'],'protocol_sha256':args.protocol_sha256,'rows':len(ids),'files':files,
        'scope':'cached model-derived candidate rescoring; not end-to-end ASR RTF',
        'source_acoustic_completed_sha256':sha256_file(args.source/'completed.json')})
    print(json.dumps({'status':'completed','condition':args.condition,'rows':len(ids)}))


if __name__=='__main__':main()
