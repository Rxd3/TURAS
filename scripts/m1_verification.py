"""Prepare review/role bundles, ingest real decisions, and publish a new reference variant."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
sys.path.insert(0,str(REPO/'scripts'))

from turkish_asr_rdl.data import verification as V
from turkish_asr_rdl.data import verification_workflow as W
from turkish_asr_rdl.training.acoustic import read_jsonl
from m1_review_kit import _write_page


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--profile',type=Path,default=REPO/'configs/datasets/m1-verification-working-v1.json')
    prepare.add_argument('--inputs-root',required=True,type=Path)
    prepare.add_argument('--private-root',required=True,type=Path)
    verify=sub.add_parser('verify-bundles')
    verify.add_argument('--workspace',required=True,type=Path)
    verify.add_argument('--out',type=Path)
    tables=sub.add_parser('validation-tables')
    tables.add_argument('--profile',type=Path,default=REPO/'configs/datasets/m1-verification-working-v1.json')
    tables.add_argument('--inputs-root',required=True,type=Path)
    tables.add_argument('--out',required=True,type=Path)
    for name in ['ingest','adjudication-kit','check','publish']:
        command=sub.add_parser(name)
        command.add_argument('--workspace',required=True,type=Path)
        command.add_argument('--records',type=Path)
        if name=='ingest':
            command.add_argument('--exports',nargs='+',required=True,type=Path)
            command.add_argument('--out',required=True,type=Path)
        if name in ['adjudication-kit','publish']:
            command.add_argument('--out',required=True,type=Path)
        if name in ['check','publish']:
            command.add_argument('--adjudication-plan',type=Path)
        if name=='publish':
            command.add_argument('--variant',required=True)
            command.add_argument('--sampled-verification',action='store_true')
        if name=='check':
            command.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        audit,_=W.prepare(REPO,args.inputs_root.resolve(),args.profile,args.private_root,_write_page)
        print(json.dumps({k:audit[k] for k in ['status','audio_files_hashed_and_format_checked','verification_test_target','new_test_clips','required_primary_evidence']}))
        return 0
    if args.command=='verify-bundles':
        report=W.verify_bundles(args.workspace)
        if args.out:V.dump_json(args.out,report)
        print(json.dumps({'status':report['status'],'bundles':len(report['bundles'])}))
        return 0
    if args.command=='validation-tables':
        report=W.validation_tables(REPO,args.inputs_root.resolve(),args.profile,args.out)
        print(json.dumps({'status':report['status'],'counts':report['counts'],'bundle_sha256':report['bundle']['sha256']}))
        return 0
    root=args.workspace.resolve()
    meta,refs,plan=W.load_workspace(root)
    records=read_jsonl(args.records) if args.records else []
    if args.command=='ingest':
        records=V.import_exports(refs,[json.loads(p.read_text(encoding='utf8')) for p in args.exports],records)
        V.dump_jsonl(args.out,records)
        print(json.dumps({'records':len(records),'path':str(args.out)}))
        return 0
    if args.command=='adjudication-kit':
        pin=W.build_adjudication_kit(REPO,root,records,args.out,_write_page)
        print(json.dumps({'adjudication_items':len(pin['required_ids']),'primary_state_sha256':pin['primary_state_sha256']}))
        return 0
    adj=json.loads(args.adjudication_plan.read_text(encoding='utf8')) if args.adjudication_plan else None
    committed=W.plan_committed(REPO,meta)
    if args.command=='check':
        report,_,_=V.verification_gate(refs,plan,records,adj,plan_committed=committed)
        if args.out:V.dump_json(args.out,report)
        print(json.dumps({'status':report['status'],'plan_committed':committed,'required_primary':report['required_primary'],
            'primary_coverage':report['primary_coverage'],'missing_primary':len(report['missing_primary_ids']),
            'missing_adjudication':len(report['missing_independent_adjudication_ids'])}))
        return 0 if report['status']=='review_gate_passed' else 2
    output=args.out.resolve()
    if output.is_relative_to(Path(meta['audio_root'])) or output.is_relative_to(REPO/'data/splits'):
        raise ValueError('publication must be separate from frozen audio/split roots')
    if output.is_relative_to(REPO) and not any(output.is_relative_to(REPO/p) for p in ['data/private','artifacts']):
        raise ValueError('reference publication cannot target a public repository path')
    manifest=V.publish_variant(refs,plan,records,adj,output,args.variant,
        sampled=args.sampled_verification,plan_committed=committed)
    print(json.dumps({'status':manifest['status'],'variant':args.variant,'M6_approval':None}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
