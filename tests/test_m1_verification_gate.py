"""Synthetic policy fixtures, never listening evidence or benchmark transcripts."""
import json
from pathlib import Path

import pytest

from turkish_asr_rdl.data import verification as V
from turkish_asr_rdl.data.verification_workflow import audit_splits


def refs():
    return {uid:{'utterance_id':uid,'audio_sha256':str(i)*64,'reference_text':'SYNTHETIC_BASE_'+uid,
        'reference_origin':'upstream_human','verification_status':'verified','split':split,
        'dataset_release':'synthetic-fixture','leakage_group_id':'group_'+uid}
        for i,(uid,split) in enumerate([('test1','test'),('test2','test'),('test3','test'),('train1','train')],1)}


def plan():
    return V.stratified_plan({'test1':'g1','test2':'g2'},{'train1'}, {'test1'},per_recording=1)


def decision(reference,uid,*,reviewer='rev_a',stamp='2026-01-01T00:00:00Z',action='accept',text=None,targets=None,shown=None):
    return {'utterance_id':uid,'audio_sha256':reference[uid]['audio_sha256'],'reviewer_id':reviewer,
        'review_utc':stamp,'annotation_policy_version':'turas-verbatim-v0.1','decision':action,
        'corrected_text':text,'text_changed':action=='correct','issue_codes':['other'] if action=='reject' else [],
        'reviewed_reference_text':shown if shown is not None else reference[uid]['reference_text'],
        'adjudicates_record_ids':targets or []}


def primary(reference,p,correction=False):
    return V.import_exports(reference,[{'kit':'synthetic-unit-fixture','role':'primary','reviews':[
        decision(reference,u,action='correct' if correction and u=='test1' else 'accept',
                 text='SYNTHETIC_CORRECTION' if correction and u=='test1' else None)
        for u in p['required_primary_ids']]}])


def complete(reference,p,records,*,reviewer='rev_b',reject=False):
    adj=V.adjudication_plan(reference,p,records)
    current=V.primary_state(records)
    exports=[decision(reference,u,reviewer=reviewer,stamp='2026-01-02T00:00:00Z',
        targets=adj['primary_bindings'][u],shown=V.record_text(current[u][-1],reference[u]),
        action='reject' if reject and u=='test1' else 'accept') for u in adj['required_ids']]
    return V.import_exports(reference,[{'role':'adjudication','reviews':exports}],records),adj


def test_stratified_sample_is_order_invariant_and_keeps_pilot():
    groups={f'id{i}':'g'+str(i//5) for i in range(10)}
    a=V.stratified_plan(groups,{'id1','id7'},per_recording=3)
    b=V.stratified_plan(dict(reversed(list(groups.items()))),{'id7','id1'},per_recording=3)
    assert a==b and len(a['test_ids'])==6
    assert {'id1','id7'}.issubset(a['test_ids'])
    assert a['selected_per_recording']=={'g0':3,'g1':3}


def test_invalid_mandatory_scope_or_pilot_quota_refused():
    with pytest.raises(ValueError,match='pilot exceeds'):
        V.stratified_plan({'a':'g','b':'g'},{'a','b'},per_recording=1)
    with pytest.raises(ValueError,match='mandatory adjudication'):
        V.stratified_plan({'a':'g'},set(),{'unknown'})


def test_reference_status_tags_never_substitute_for_actual_records():
    report,_,_=V.verification_gate(refs(),plan(),[],plan_committed=True)
    assert report['status']=='blocked_pending_evidence'
    assert report['primary_coverage']==0 and len(report['missing_primary_ids'])==3


def test_real_primary_coverage_still_requires_pinned_independent_adjudication():
    r=refs();p=plan();records=primary(r,p)
    report,_,_=V.verification_gate(r,p,records,plan_committed=True)
    assert report['invalid_adjudication_plan'] and report['missing_independent_adjudication_ids']
    records,adj=complete(r,p,records,reviewer='rev_a')
    report,_,_=V.verification_gate(r,p,records,adj,plan_committed=True)
    assert report['status']=='blocked_pending_evidence'
    assert 'test1' in report['missing_independent_adjudication_ids']


def test_new_primary_record_invalidates_the_old_adjudication():
    r=refs();p=plan();records,adj=complete(r,p,primary(r,p))
    changed=decision(r,'test1',stamp='2026-01-03T00:00:00Z',action='correct',text='SYNTHETIC_NEW_CORRECTION')
    records=V.import_exports(r,[{'role':'primary','reviews':[changed]}],records)
    report,_,_=V.verification_gate(r,p,records,adj,plan_committed=True)
    assert report['invalid_adjudication_plan'] and report['status']=='blocked_pending_evidence'


def test_adjudicated_accept_keeps_the_proposed_correction_and_all_ids(tmp_path):
    r=refs();p=plan();records,adj=complete(r,p,primary(r,p,correction=True))
    original=json.dumps(r,sort_keys=True)
    with pytest.raises(ValueError,match='sampled-verification'):
        V.publish_variant(r,p,records,adj,tmp_path/'no-flag','synthetic-reviewed-v1',plan_committed=True)
    assert not (tmp_path/'no-flag').exists()
    out=tmp_path/'new-variant'
    manifest=V.publish_variant(r,p,records,adj,out,'synthetic-reviewed-v1',sampled=True,plan_committed=True)
    rows=[json.loads(line) for path in out.rglob('references.*.jsonl') for line in path.read_text().splitlines()]
    by_id={row['utterance_id']:row for row in rows}
    assert set(by_id)==set(r) and json.dumps(r,sort_keys=True)==original
    assert by_id['test1']['reference_text']=='SYNTHETIC_CORRECTION'
    assert by_id['test1']['reference_origin']=='local_human'
    assert by_id['test3']['verification_status']=='unreviewed'
    assert manifest['M6_approval'] is None
    assert all(b'\r\n' not in path.read_bytes() for path in out.rglob('*.json*'))


def test_rejection_is_a_blocker_never_silently_removed_from_frozen_ids(tmp_path):
    r=refs();p=plan();records,adj=complete(r,p,primary(r,p),reject=True)
    report,_,_=V.verification_gate(r,p,records,adj,plan_committed=True)
    assert report['adjudicated_rejection_ids']==['test1']
    assert report['rejection_ledger'][0]['reason']=='rejected_in_listening_review'
    assert report['rejection_ledger'][0]['frozen_membership_unchanged'] is True
    with pytest.raises(ValueError,match='incomplete'):
        V.publish_variant(r,p,records,adj,tmp_path/'out','synthetic-reviewed-v1',sampled=True,plan_committed=True)
    assert not (tmp_path/'out').exists() and 'test1' in r


def test_uncommitted_sample_cannot_publish_even_with_complete_records(tmp_path):
    r=refs();p=plan();records,adj=complete(r,p,primary(r,p))
    with pytest.raises(ValueError,match='incomplete'):
        V.publish_variant(r,p,records,adj,tmp_path/'out','synthetic-reviewed-v1',sampled=True)
    assert not (tmp_path/'out').exists()


def test_timestamp_requires_actual_utc_not_lexical_sorting():
    for value in ['2026-01-01T00:00:00','2026-01-01T03:00:00+03:00']:
        with pytest.raises(ValueError,match='explicit UTC'):
            V.utc(value)
    assert V.utc('2026-01-01T00:00:00Z')==V.utc('2026-01-01T00:00:00+00:00')


def test_record_tampering_and_stale_reference_snapshots_refused():
    r=refs();p=plan();records=primary(r,p)
    bad=[dict(x) for x in records];bad[0]['reviewer_id']='rev_other'
    with pytest.raises(ValueError,match='tampered'):
        V.validate_records(r,bad)
    r[records[0]['utterance_id']]['reference_text']='SYNTHETIC_DIFFERENT_SNAPSHOT'
    with pytest.raises(ValueError,match='stale reference'):
        V.validate_records(r,records)


def test_exact_leaking_text_or_recording_is_rejected():
    r=refs();r['test1']['split']='validation'
    ids={'train':['train1'],'validation':['test1'],'test':['test2','test3']}
    assert audit_splits(r,ids)
    r['test2']['leakage_group_id']=r['train1']['leakage_group_id']
    with pytest.raises(ValueError,match='cross-split'):
        audit_splits(r,ids)
    r['test2']['leakage_group_id']='different'
    r['test1']['reference_text']=r['train1']['reference_text']
    with pytest.raises(ValueError,match='cross-split'):
        audit_splits(r,ids)


def test_new_file_transport_is_lf_atomic_and_non_overwriting(tmp_path):
    p=tmp_path/'metadata.json';V.dump_json(p,{'label':'SYNTHETIC_FIXTURE'})
    before=p.read_bytes();assert b'\r\n' not in before
    V.dump_json(p,{'label':'SYNTHETIC_FIXTURE'})
    with pytest.raises(ValueError,match='different bytes'):
        V.dump_json(p,{'label':'SYNTHETIC_OTHER'})
    assert p.read_bytes()==before


def test_working_bundle_verifier_checks_checksums_and_reference_roles(tmp_path):
    from turkish_asr_rdl.data.verification_workflow import _archive, verify_bundles
    rid='synthetic-fixture';prefix=tmp_path/'working-exports';prefix.mkdir()
    refs_payload=(json.dumps({'utterance_id':'id1','split':'train'})+'\n').encode()
    members={rid+'/train.txt':b'id1\n',rid+'/'+rid+'.references.train.jsonl':refs_payload}
    info=_archive(prefix/'synthetic-TRAIN-REFERENCES.tar',members)
    V.dump_json(tmp_path/'working-handoff.json',{'dataset_release':rid,'bundles':[info]})
    assert verify_bundles(tmp_path)['bundles'][0]['reference_splits']==['train']
    with (prefix/info['file']).open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError,match='identity mismatch'):
        verify_bundles(tmp_path)
