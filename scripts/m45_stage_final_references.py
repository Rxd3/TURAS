"""Open test gold only after frozen, hash-verified inference is complete."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import tarfile

from turkish_asr_rdl.training.acoustic import sha256_file,read_jsonl,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--inference-index',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((Path.cwd()/'data/private').resolve()):
        parser.error('output must be fresh under data/private/')
    if sha256_file(args.protocol)!=args.protocol_sha256:raise ValueError('protocol checksum mismatch')
    p=json.loads(args.protocol.read_text());index=json.loads(args.inference_index.read_text())
    if p['status']!='frozen_for_final_evaluation' or index['status']!='inference_complete_gold_still_closed' or index['protocol_sha256']!=args.protocol_sha256:
        raise ValueError('final inference/benchmark prerequisites are not complete')
    expected={(s['system_id'],'accuracy') for s in p['systems'] if s['kind']!='relational'}
    expected|={(s['system_id'],'timing') for s in p['systems']}
    expected|={(s['system_id'],'cached_nbest_selection') for s in p['rescore_conditions']}
    actual=[(r['system_id'],r['mode']) for r in index['runs']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('incomplete system/mode inventory; test gold remains closed')
    for entry in index['runs']:
        directory=Path(entry['directory'])
        if sha256_file(directory/'completed.json')!=entry['completed_sha256']:raise ValueError('completed run inventory mismatch')
        report=json.loads((directory/'completed.json').read_text())
        for name,digest in report['files'].items():
            if sha256_file(directory/name)!=digest:raise ValueError('inference output changed before test access')
    if sha256_file(args.archive)!=p['test_reference']['archive_sha256']:raise ValueError('test archive checksum mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    audit={'status':'opening_evaluator_only_test_references','opened_at_utc':datetime.now(timezone.utc).isoformat(),
           'protocol_sha256':args.protocol_sha256,'inference_index_sha256':sha256_file(args.inference_index),
           'reference_variant':p['test_reference']['variant'],'archive_sha256':p['test_reference']['archive_sha256']}
    write_json(args.output/'reference-access.json',audit)
    with tarfile.open(args.archive) as archive:
        members=[m for m in archive.getmembers() if m.name==p['test_reference']['archive_member'] and m.isfile()]
        if len(members)!=1:raise ValueError('expected unique reference member missing')
        payload=archive.extractfile(members[0]).read()
    import hashlib
    if hashlib.sha256(payload).hexdigest()!=p['test_reference']['payload_sha256']:raise ValueError('test reference payload checksum mismatch')
    target=args.output/'references.test.jsonl';target.write_bytes(payload)
    rows=read_jsonl(target);ids=Path(p['inputs']['test_ids']).read_text().splitlines()
    if [r['utterance_id'] for r in rows]!=ids or any(r['split']!='test' or r['dataset_release']!=p['dataset_release'] for r in rows):
        raise ValueError('test reference identity/coverage mismatch')
    inference={r['utterance_id']:r for r in read_jsonl(Path(p['inputs']['test_inference']))}
    if any(r['audio_sha256']!=inference[r['utterance_id']]['audio_sha256'] for r in rows):raise ValueError('reference/audio identity mismatch')
    import csv
    group_path=Path(p['inputs']['test_groups'])
    if sha256_file(group_path)!=p['inputs']['test_groups_sha256']:raise ValueError('group metadata hash mismatch')
    with group_path.open(encoding='utf-8',newline='') as f:groups={r['utterance_id']:r['leakage_group_id'] for r in csv.DictReader(f)}
    if any(r['leakage_group_id']!=groups[r['utterance_id']] for r in rows):raise ValueError('reference/group metadata mismatch')
    audit.update(status='test_references_staged_for_frozen_evaluation',rows=len(rows),reference_sha256=sha256_file(target),
        limitations=['M1 minutes with reported pilot corrections; not fully human-verified verbatim speech',
                     'M6 release-review disagreement remains disclosed; owner authorizes this working benchmark'])
    write_json(args.output/'reference-access.json',audit)
    print(json.dumps({'status':audit['status'],'rows':len(rows),'reference_sha256':audit['reference_sha256']}))


if __name__=='__main__':main()
