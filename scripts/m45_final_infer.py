"""Frozen gold-free final inference/timing; no reference file is read here."""
import argparse
import json
from pathlib import Path
import uuid

from turkish_asr_rdl.decoding.adapted import AdaptedAsrAdapter, PretrainedAsrAdapter
from turkish_asr_rdl.decoding.relational import RelationalAsrAdapter
from turkish_asr_rdl.decoding.export import RunContext,nbest_row
from turkish_asr_rdl.decoding.interface import InferenceInput
from turkish_asr_rdl.decoding.timing import run_timed, write_run_outputs
from turkish_asr_rdl.schemas.predictions import validate_predictions
from turkish_asr_rdl.schemas.nbest import validate_nbest
from turkish_asr_rdl.schemas.jsonl import write_jsonl
from turkish_asr_rdl.training.acoustic import read_jsonl,sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--system',required=True)
    parser.add_argument('--mode',choices=['accuracy','timing'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must be fresh')
    if sha256_file(args.protocol)!=args.protocol_sha256:raise ValueError('frozen protocol hash mismatch')
    protocol=json.loads(args.protocol.read_text())
    if protocol['status'] not in ['frozen_for_final_evaluation','frozen_validation_preflight']:raise ValueError('protocol is not frozen')
    split=protocol.get('evaluation_split','test')
    if protocol['status']=='frozen_validation_preflight' and split!='validation':raise ValueError('preflight uses validation only')
    specs={s['system_id']:s for s in protocol['systems']}
    spec=specs[args.system]
    base=specs[spec['base_system']] if spec['kind']=='relational' else spec
    class ProgressMixin:
        completed=0;errors=0
        def transcribe(self,batch):
            results=super().transcribe(batch)
            if not hasattr(self,'warmup_results'):self.warmup_results=[]
            if len(self.warmup_results)<3:self.warmup_results.extend(results[:3-len(self.warmup_results)])
            self.completed+=len(batch);self.errors+=sum(h.status=='error' for h in results)
            if self.completed==1 or self.completed%25==0:
                print(json.dumps({'items_including_warmup':self.completed,'errors':self.errors}),flush=True)
            return results
    class Pinned(ProgressMixin,PretrainedAsrAdapter):pass
    class Adapted(ProgressMixin,AdaptedAsrAdapter):pass
    class Combined(ProgressMixin,RelationalAsrAdapter):pass
    if base['kind']=='pretrained':
        adapter=Pinned(base['model'],family=base['family'],system_id=base['system_id'],**protocol['decoder'])
    else:
        adapter=Adapted(Path(base['checkpoint']),base['checkpoint_manifest_sha256'],system_id=base['system_id'],**protocol['decoder'])
    if spec['kind']=='relational':
        adapter=Combined(adapter,Path(spec['graph']),spec['graph_manifest_sha256'],dataset_release=protocol['dataset_release'],
            split_manifest_sha256=protocol['split_manifest_sha256'],fusion=spec['fusion'],missing_probability=protocol['missing_probability'])
        adapter.system_id=spec['system_id']
    inputs=protocol['inputs']
    inference=Path(inputs['test_inference'])
    if sha256_file(inference)!=inputs['test_inference_sha256']:raise ValueError('inference hash mismatch')
    ids_path=Path(inputs['test_ids'] if args.mode=='accuracy' else inputs['timing_ids'])
    expected=inputs['test_ids_sha256'] if args.mode=='accuracy' else inputs['timing_ids_sha256']
    if sha256_file(ids_path)!=expected:raise ValueError('frozen IDs hash mismatch')
    ids=ids_path.read_text().splitlines();wanted=set(ids)
    rows=[r for r in read_jsonl(inference) if r['utterance_id'] in wanted]
    if [r['utterance_id'] for r in rows]!=ids:raise ValueError('inference coverage/order mismatch')
    timed=[InferenceInput.from_row(r) for r in rows]
    warmup_path=Path(inputs['warmup_inference'])
    if sha256_file(warmup_path)!=inputs['warmup_inference_sha256']:raise ValueError('warm-up hash mismatch')
    warmup=[InferenceInput.from_row(r) for r in read_jsonl(warmup_path)[:3]]
    if any(r.dataset_release!=protocol['dataset_release'] or r.split_manifest_sha256!=protocol['split_manifest_sha256'] for r in timed+warmup):
        raise ValueError('inference release/split mismatch')
    run_id=spec['system_id']+'-'+args.mode+'-'+uuid.uuid4().hex[:8]
    repeats=1 if args.mode=='accuracy' else protocol['timing']['repeats']
    run=run_timed(adapter,timed,warmup,audio_root=Path(inputs['audio_root']),run_id=run_id,repeats=repeats,batch_size=1)
    ctx=RunContext(run_id,spec['system_id'],protocol['dataset_release'],split,protocol['split_manifest_sha256'],
        adapter.model_artifact_sha256,adapter.decoder_config_sha256)
    files=write_run_outputs(args.output,run,timed,ctx,adapter,command='scripts/m45_final_infer.py')
    warm_ctx=RunContext(run_id+'-warmup',spec['system_id'],protocol['dataset_release'],'train',protocol['split_manifest_sha256'],
        adapter.model_artifact_sha256,adapter.decoder_config_sha256)
    warm_rows=[nbest_row(h,i,warm_ctx,adapter.requested_n,adapter.score_convention_id) for h,i in zip(adapter.warmup_results,warmup)]
    if not validate_nbest(warm_rows,[i.utterance_id for i in warmup]).ok:raise ValueError('warm-up candidate export failed')
    files['warmup-nbest.jsonl']=write_jsonl(args.output/'warmup-nbest.jsonl',warm_rows)
    predictions=read_jsonl(args.output/'predictions.jsonl')
    if not validate_predictions(predictions,ids).ok or not validate_nbest(read_jsonl(args.output/'nbest.jsonl'),ids).ok:
        raise ValueError('final output contract violation')
    if adapter.errors or any(r['status']=='error' for r in predictions):raise ValueError('inference error rows; evaluation gate remains closed')
    write_json(args.output/'completed.json',{'status':'completed','mode':args.mode,'protocol_sha256':args.protocol_sha256,
        'system_id':spec['system_id'],'rows':len(ids),'repeats':repeats,'files':files,
        'timing_claim':'headline serial benchmark' if args.mode=='timing' else 'accuracy run; its timings are not headline benchmarks'})
    print(json.dumps({'status':'completed','system':spec['system_id'],'mode':args.mode,'rows':len(ids)}),flush=True)


if __name__=='__main__':main()
