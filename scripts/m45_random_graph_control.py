"""Create an explicitly untrained parameter control on the same train-only graph."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file
from turkish_asr_rdl.rdl.model import HeterogeneousRelationModel
from turkish_asr_rdl.rdl.train import load_model
from turkish_asr_rdl.training.acoustic import sha256_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((Path.cwd()/'artifacts').resolve()):
        parser.error('output must be fresh under ignored artifacts/')
    original,_,manifest=load_model(args.source)
    config=json.loads((args.source/'resolved-config.json').read_text())
    seed=config['seed'];torch.manual_seed(seed)
    control=HeterogeneousRelationModel(original.snapshot,**config['model']).eval()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['snapshot.json','linker.json']:
        shutil.copy2(args.source/name,args.output/name)
    config['control_kind']='untrained_random_parameters'
    config['training']={'epochs':0,'learning_rate':None}
    write_json(args.output/'resolved-config.json',config)
    write_json(args.output/'training.json',{'status':'not_trained_control','history':[],
        'initialization_seed':seed,'purpose':'learned-versus-untrained graph ablation; never an ASR baseline'})
    write_json(args.output/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}})
    save_file({k:v.detach().cpu().contiguous() for k,v in control.state_dict().items()},str(args.output/'model.safetensors'))
    write_json(args.output/'artifact-manifest.json',{'schema_version':'0.1','status':'untrained_random_parameter_control',
        'control_kind':'untrained_random_parameters','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'dataset_release':manifest['dataset_release'],'split_manifest_sha256':manifest['split_manifest_sha256'],
        'parent_graph_manifest_sha256':sha256_file(args.source/'artifact-manifest.json'),
        'files':{p.name:sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        'limitations':['No optimization performed; this is a control, not a trained contribution',
                       'Same train-only graph and frozen linker as the learned model']})
    load_model(args.output)
    print(json.dumps({'status':'untrained_control_created','manifest_sha256':sha256_file(args.output/'artifact-manifest.json')}))


if __name__=='__main__':main()
