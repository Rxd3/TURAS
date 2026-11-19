"""Execute a frozen M4/M5 benchmark, then open gold once and build the handoff."""
import argparse
import concurrent.futures
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from turkish_asr_rdl.training.acoustic import sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('configs/evaluation/m45_final_protocol.json'))
    parser.add_argument('--root',type=Path,default=Path('artifacts/owner-m45/final'))
    args=parser.parse_args()
    root=args.root;lock_record=json.loads((root/'protocol-lock.json').read_text())
    digest=lock_record['sha256']
    if sha256_file(args.protocol)!=digest:raise ValueError('protocol checksum mismatch')
    p=json.loads(args.protocol.read_text());guard=threading.Lock()
    state={'status':'starting','protocol_sha256':digest,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'queue_pid':os.getpid(),'gold_opened':False,'jobs':[],'freeze_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    def save():
        state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        tmp=root/'status.tmp';write_json(tmp,state);tmp.replace(root/'status.json')
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_SYMLINKS_WARNING='1',TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    def verify_code():
        if sha256_file(args.protocol)!=digest:raise ValueError('protocol changed after freeze')
        for path,expected in p['critical_code_sha256'].items():
            if sha256_file(Path(path))!=expected:raise ValueError('critical code changed after freeze: '+path)
    def run(name,command,*,directory=None,mode=None,system=None):
        verify_code()
        if (root/'STOP').exists():raise RuntimeError('STOP requested')
        log=root/(name+'.log')
        item={'name':name,'status':'running','command':[sys.executable,'-u']+command,'log':log.as_posix(),
              'started_at_utc':datetime.now(timezone.utc).isoformat()}
        if directory:item.update(directory=directory.as_posix(),mode=mode,system_id=system)
        with guard:state['jobs'].append(item);save()
        with log.open('x',encoding='utf-8') as output:
            process=subprocess.Popen([sys.executable,'-u']+command,stdout=output,stderr=subprocess.STDOUT,
                env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            with guard:item['pid']=process.pid;save()
            while process.poll() is None:
                if (root/'STOP').exists():
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=True,stdout=subprocess.DEVNULL)
                    process.wait(timeout=30)
                time.sleep(3)
        with guard:
            item.update(status='completed' if process.returncode==0 else 'failed',exit_code=process.returncode,
                        ended_at_utc=datetime.now(timezone.utc).isoformat())
            if directory and process.returncode==0:item['completed_sha256']=sha256_file(directory/'completed.json')
            save()
        if process.returncode:raise RuntimeError(name+' failed; evaluation stops before further gold access')
        return item
    prefix=['--protocol',str(args.protocol),'--protocol-sha256',digest]
    def infer(spec,mode):
        directory=root/mode/spec['system_id'];directory.parent.mkdir(exist_ok=True)
        return run(mode+'--'+spec['system_id'],['scripts/m45_final_infer.py']+prefix+
            ['--system',spec['system_id'],'--mode',mode,'--output',str(directory)],
            directory=directory,mode=mode,system=spec['system_id'])
    def select(spec):
        directory=root/'rescore'/spec['system_id'];directory.parent.mkdir(exist_ok=True)
        return run('rescore--'+spec['system_id'],['scripts/m45_final_rescore.py']+prefix+
            ['--condition',spec['system_id'],'--source',str(root/'accuracy'/spec['base_system']),'--output',str(directory)],
            directory=directory,mode='cached_nbest_selection',system=spec['system_id'])
    keep_awake=0
    try:
        keep_awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        state['temporary_system_awake_request']=bool(keep_awake)
        state['status']='accuracy_inference_gold_closed';save()
        native=[s for s in p['systems'] if s['kind']!='relational']
        # Put the slower CPU-heavy CTC jobs alongside GPU-heavy Whisper jobs.
        order=sorted(native,key=lambda s: ['B4-xlsr-tr-cv7','B1-whisper-small','F1-whisper-lora','F3-ctc-vocab','F2-whisper-full'].index(s['system_id']))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda s:infer(s,'accuracy'),order))
        state['status']='cached_candidate_controls_gold_closed';save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(select,p['rescore_conditions']))
        state['status']='serial_headline_timing_gold_closed';save()
        specs={s['system_id']:s for s in p['systems']}
        for name in p['timing']['system_order']:
            state['cooldown_before']=name;save()
            time.sleep(p['timing']['cooldown_seconds'])
            if (root/'STOP').exists():raise RuntimeError('STOP requested')
            infer(specs[name],'timing')
        index={'status':'inference_complete_gold_still_closed','protocol_sha256':digest,
            'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'runs':[{key:j[key] for key in ['system_id','mode','directory','completed_sha256']} for j in state['jobs'] if 'directory' in j]}
        write_json(root/'inference-index.json',index)
        state['status']='opening_test_references_after_frozen_inference';save()
        references=Path('data/private')/p['dataset_release']/'evaluator-final'
        archive=Path('data/private')/p['dataset_release']/'test-archive'/(p['dataset_release']+'-TEST-split-EVALUATOR-ONLY.tar')
        run('stage-evaluator-only-references',['scripts/m45_stage_final_references.py']+prefix+
            ['--inference-index',str(root/'inference-index.json'),'--archive',str(archive),'--output',str(references)])
        state.update(gold_opened=True,status='scoring_frozen_all_system_test_comparison');save()
        run('score-final-test',['scripts/m45_final_evaluate.py']+prefix+
            ['--inference-index',str(root/'inference-index.json'),'--reference-dir',str(references),'--output',str(root/'evaluation')])
        state['status']='building_reports_and_private_handoff';save()
        run('build-handoff',['scripts/m45_final_handoff.py','--protocol',str(args.protocol),'--evaluation',str(root/'evaluation'),
            '--run-root',str(root),'--output',str(root/'handoff')])
        state.update(status='completed_m4_m5_final_benchmark_and_handoff',ended_at_utc=datetime.now(timezone.utc).isoformat());save()
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
