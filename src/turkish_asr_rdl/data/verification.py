"""Pinned, evidence-based verification of an immutable TURAS audio/split release.

No review decisions are inferred from reference status tags or model output.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import uuid

from turkish_asr_rdl.data import review as RV
from turkish_asr_rdl.training.acoustic import read_jsonl, sha256_file

GATE_VERSION = 'turas-stratified-review-v1'


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def write_new(path, payload):
    """LF bytes, atomic publication, and no silent replacement of existing data."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ValueError('output already exists with different bytes: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_bytes(payload)
    if temporary.resolve().parent != path.resolve().parent or path.exists():
        raise ValueError('atomic publication target changed')
    os.replace(temporary, path)


def dump_json(path, value):
    write_new(path, json_bytes(value))


def dump_jsonl(path, rows):
    write_new(path, b''.join((json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8') for r in rows))


def stratified_plan(groups, pilot_ids, mandatory_adjudication_ids=(), *, per_recording=8, seed=2026):
    if per_recording < 1 or not groups:
        raise ValueError('nonempty groups and positive recording quota required')
    by_group = defaultdict(list)
    for uid, group in sorted(groups.items()):
        by_group[group].append(uid)
    rng = random.Random(seed)
    chosen = []
    counts = {}
    for group, ids in sorted(by_group.items()):
        prior = sorted(set(ids) & set(pilot_ids))
        quota = min(per_recording, len(ids))
        if len(prior) > quota:
            raise ValueError('pilot exceeds a recording quota')
        remaining = sorted(set(ids)-set(prior))
        selected = prior+rng.sample(remaining, quota-len(prior))
        chosen.extend(selected)
        counts[group] = len(selected)
    if not set(mandatory_adjudication_ids).issubset(set(chosen)|set(pilot_ids)):
        raise ValueError('mandatory adjudication IDs must belong to required primary scope')
    return {'gate_version':GATE_VERSION, 'seed':seed, 'per_recording':per_recording,
        'test_ids':sorted(chosen), 'pilot_ids':sorted(pilot_ids),
        'required_primary_ids':sorted(set(chosen)|set(pilot_ids)),
        'mandatory_adjudication_ids':sorted(set(mandatory_adjudication_ids)),
        'selected_per_recording':counts,
        'selection_inputs':'frozen utterance/recording IDs and pre-existing pilot IDs only; no predictions/errors',
        'chronology':'Prepared 2026-11-21 after working test evaluation; no blind-before-all-evaluation claim'}


def utc(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z','+00:00'))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('review_utc must be a UTC ISO timestamp') from exc
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('review_utc must have an explicit UTC offset')
    return stamp.astimezone(timezone.utc)


def signed_record(body):
    body = {k:v for k,v in body.items() if k != 'verification_record_id'}
    digest = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {'verification_record_id':('m1rev_'+digest[:32] if body.get('review_gate_version') else 'rev_'+digest[:20]), **body}


def import_exports(refs, exports, existing=()):
    result = {r['verification_record_id']:r for r in existing}
    for export in exports:
        role = export.get('role','primary')
        if role not in RV.ROLES:
            raise ValueError('invalid review role')
        for decision in export.get('reviews',[]):
            if decision.get('decision') is None:
                continue
            ref = refs.get(decision.get('utterance_id'))
            if ref is None:
                raise ValueError('unknown review utterance')
            shown = decision.get('reviewed_reference_text', ref['reference_text'])
            if role == 'primary' and shown != ref['reference_text']:
                raise ValueError('primary review used a different reference snapshot')
            RV.validate_decision(decision, {**ref,'reference_text':shown})
            utc(decision['review_utc'])
            targets = sorted(decision.get('adjudicates_record_ids',[]))
            if (role == 'primary' and targets) or (role == 'adjudication' and not targets):
                raise ValueError('adjudication must explicitly bind its primary record IDs')
            record = RV.make_record(decision, export.get('kit'), role)
            record.update(review_gate_version=GATE_VERSION, reviewed_reference_text=shown,
                          adjudicates_record_ids=targets)
            record = signed_record(record)
            result[record['verification_record_id']] = record
    records = sorted(result.values(), key=lambda r:(utc(r['review_utc']),r['verification_record_id']))
    validate_records(refs, records)
    return records


def validate_records(refs, records):
    seen = set()
    for r in records:
        uid = r.get('utterance_id')
        if uid not in refs or r.get('role') not in RV.ROLES:
            raise ValueError('unknown utterance or review role')
        if r.get('verification_record_id') in seen or signed_record(r)['verification_record_id'] != r.get('verification_record_id'):
            raise ValueError('duplicate/tampered review record')
        seen.add(r['verification_record_id'])
        if (r.get('audio_sha256') != refs[uid]['audio_sha256'] or
                r.get('reviewed_audio_sha256') != refs[uid]['audio_sha256']):
            raise ValueError('reviewed audio differs; supply measured equivalence evidence or re-review current audio')
        if r.get('annotation_policy_version') != RV.POLICY_VERSION or not re.fullmatch(r'rev_[a-z0-9_]{1,32}',str(r.get('reviewer_id',''))) or r['reviewer_id']=='rev_unknown':
            raise ValueError('invalid reviewer or transcription policy')
        utc(r.get('review_utc'))
        if r['role']=='primary' and r.get('reviewed_reference_text',refs[uid]['reference_text'])!=refs[uid]['reference_text']:
            raise ValueError('primary review used a stale reference snapshot')
        if r.get('decision') not in RV.DECISIONS or any(c not in RV.ISSUE_CODES for c in r.get('issue_codes',[])):
            raise ValueError('invalid review decision/issues')
        if r['decision']=='correct' and not str(r.get('corrected_text') or '').strip():
            raise ValueError('correction text missing')
        if r['decision']=='reject' and not r.get('issue_codes'):
            raise ValueError('rejection issue missing')
        if r['role']=='adjudication' and (not r.get('adjudicates_record_ids') or 'reviewed_reference_text' not in r):
            raise ValueError('legacy adjudication lacks a target/text binding; use the pinned adjudication kit')


def primary_state(records):
    latest = {}
    for r in records:
        if r['role']=='primary':
            key = (r['utterance_id'],r['reviewer_id'])
            if key not in latest or (utc(r['review_utc']),r['verification_record_id']) > (utc(latest[key]['review_utc']),latest[key]['verification_record_id']):
                latest[key] = r
    by_uid = defaultdict(list)
    for r in latest.values():
        by_uid[r['utterance_id']].append(r)
    return {uid:sorted(rows,key=lambda r:(utc(r['review_utc']),r['verification_record_id'])) for uid,rows in sorted(by_uid.items())}


def record_text(record, reference):
    if record['decision']=='correct':
        return record['corrected_text']
    return record.get('reviewed_reference_text',reference['reference_text'])


def adjudication_plan(refs, plan, records):
    validate_records(refs, records)
    primary = primary_state(records)
    if set(plan['required_primary_ids'])-primary.keys():
        raise ValueError('primary review coverage incomplete')
    required = set(plan['mandatory_adjudication_ids'])
    accepts = []
    for uid, rows in primary.items():
        outcomes = {(r['decision'],record_text(r,refs[uid])) for r in rows}
        if len(outcomes)>1 or any(r['decision'] in ('correct','reject') for r in rows):
            required.add(uid)
        elif rows[-1]['decision']=='accept':
            accepts.append(uid)
    count = math.ceil(len(accepts)*0.1)
    sample = sorted(random.Random(plan['seed']).sample(sorted(accepts),count)) if count else []
    required.update(sample)
    bindings = {uid:sorted(r['verification_record_id'] for r in rows) for uid,rows in primary.items()}
    return {'gate_version':GATE_VERSION, 'verification_plan_sha256':hashlib.sha256(json_bytes(plan)).hexdigest(),
        'seed':plan['seed'], 'accept_sample_ids':sample, 'required_ids':sorted(required),
        'primary_bindings':bindings,
        'primary_state_sha256':hashlib.sha256(json_bytes(bindings)).hexdigest()}


def verification_gate(refs, plan, records, pinned_adjudication=None, *, plan_committed=False):
    validate_records(refs,records)
    primary = primary_state(records)
    missing_primary = sorted(set(plan['required_primary_ids'])-primary.keys())
    missing_adj, invalid_adj, rejected = [], [], []
    final = {}
    expected = None
    if not missing_primary:
        expected = adjudication_plan(refs,plan,records)
        if pinned_adjudication != expected:
            invalid_adj.append('missing_or_stale_pinned_adjudication_plan')
        for uid, rows in primary.items():
            chosen = rows[-1]
            if uid in expected['required_ids']:
                targets = expected['primary_bindings'][uid]
                proposals = {record_text(r,refs[uid]) for r in rows}
                reviewers = {r['reviewer_id'] for r in rows}
                candidates = [r for r in records if r['utterance_id']==uid and r['role']=='adjudication'
                    and sorted(r.get('adjudicates_record_ids',[]))==targets
                    and r['reviewer_id'] not in reviewers
                    and utc(r['review_utc'])>=max(utc(p['review_utc']) for p in rows)
                    and r.get('reviewed_reference_text') in proposals]
                if not candidates:
                    missing_adj.append(uid)
                    continue
                chosen = max(candidates,key=lambda r:(utc(r['review_utc']),r['verification_record_id']))
            final[uid] = chosen
            if chosen['decision']=='reject':
                rejected.append(uid)
    ready = bool(plan_committed and not missing_primary and not missing_adj and not invalid_adj and not rejected)
    report = {'status':'review_gate_passed' if ready else 'blocked_pending_evidence',
        'gate_version':GATE_VERSION, 'plan_committed':plan_committed,
        'required_primary':len(plan['required_primary_ids']), 'primary_coverage':len(set(plan['required_primary_ids'])&primary.keys()),
        'missing_primary_ids':missing_primary, 'missing_independent_adjudication_ids':missing_adj,
        'invalid_adjudication_plan':invalid_adj, 'adjudicated_rejection_ids':sorted(rejected),
        'rejection_ledger':[{'utterance_id':uid, 'reason':'rejected_in_listening_review',
            'verification_record_id':final[uid]['verification_record_id'],
            'issue_codes':final[uid]['issue_codes'], 'split':refs[uid]['split'],
            'frozen_membership_unchanged':True} for uid in sorted(rejected)],
        'rejection_policy':'Frozen split membership is never changed here; unresolved rejected IDs block publication.',
        'reference_status_tags_alone_count_as_review_evidence':False,
        'M6_release_approval_claimed':False}
    return report, final, expected


def publish_variant(refs, plan, records, pinned_adjudication, output, variant, *, sampled=False, plan_committed=False):
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,95}',variant):
        raise ValueError('explicit safe reference variant required')
    report, final, _ = verification_gate(refs,plan,records,pinned_adjudication,plan_committed=plan_committed)
    if report['status']!='review_gate_passed':
        raise ValueError('listening/adjudication gate is incomplete; nothing published')
    heldout = {uid for uid,r in refs.items() if r['split'] in ('validation','test')}
    if not sampled and heldout-final.keys():
        raise ValueError('unreviewed held-out rows require the pinned sampled-verification policy')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('publication output must be fresh')
    stage = output.with_name(output.name+'.staging-'+uuid.uuid4().hex)
    stage.mkdir(parents=True)
    by_split = defaultdict(list)
    for uid, base in sorted(refs.items()):
        row = dict(base)
        row['reference_variant'] = variant
        if uid in final:
            record = final[uid]
            text = record_text(record,base)
            row.update(reference_text=text,
                reference_origin='local_human' if text!=base['reference_text'] else base['reference_origin'],
                verification_status='adjudicated' if record['role']=='adjudication' else 'verified',
                verification_record_id=record['verification_record_id'])
        else:
            # Reported legacy tags without supplied records are not new verified evidence.
            row.update(verification_status='unreviewed',verification_record_id=None)
        by_split[row['split']].append(row)
    files = {}
    for split, rows in sorted(by_split.items()):
        path = stage / ('evaluator-only' if split=='test' else split) / f'references.{split}.jsonl'
        dump_jsonl(path,rows)
        files[path.relative_to(stage).as_posix()] = sha256_file(path)
    dump_json(stage/'verification-plan.json',plan)
    dump_json(stage/'adjudication-plan.json',pinned_adjudication)
    dump_jsonl(stage/'evaluator-only/review-records.jsonl',records)
    manifest = {'status':'review_gate_passed_pending_independent_release_review', 'reference_variant':variant,
        'base_split_manifest_sha256':plan.get('base_split_manifest_sha256'),
        'verification_scope':'pinned recording-stratified test sample and pilot evidence; remaining minutes unreviewed' if sampled else 'complete held-out listening',
        'files_sha256':files, 'gate':report, 'split_membership_changed':False,
        'rejected_ids_silently_dropped':False, 'M6_approval':None}
    dump_json(stage/'reference-manifest.json',manifest)
    if stage.parent != output.parent or output.exists():
        raise ValueError('publication destination changed')
    os.replace(stage,output)
    return manifest


def wilson_interval(corrected, total):
    if not total:
        return None
    z = 1.959963984540054
    p = corrected/total
    denominator = 1+z*z/total
    midpoint = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [midpoint-radius,midpoint+radius]
