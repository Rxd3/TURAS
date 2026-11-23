"""Self-check of f2repro selection, comparison and status on the committed M6 summary schema.

Run from the main clone: <python> f2repro_selfcheck.py <folder holding f2repro.py>   (PYTHONPATH=<main>\\src)
"""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import f2repro as f  # noqa: E402

real = json.loads(Path('results/m6/validation/summary.json').read_text(encoding='utf-8'))
base = real['systems'][0]


def system(sid, norm, raw):
    s = copy.deepcopy(base)
    s['system_id'] = sid
    s['metrics']['tr-normalized-v1']['wer'] = norm
    s['metrics']['raw-v1']['wer'] = raw
    return s


P = 'F2-m6repro-s17'
summ = {**real, 'systems': [system(f'{P}-e1', .11, .19), system(f'{P}-e2', .10, .185), system(f'{P}-e3', .104, .18)]}
assert f.select_epoch(summ, P)[1]['epoch'] == 2                      # lowest normalized WER wins
summ['systems'][2] = system(f'{P}-e3', .10, .18)
assert f.select_epoch(summ, P)[1]['epoch'] == 3                      # tie on normalized -> lower raw WER
summ['systems'][2] = system(f'{P}-e3', .10, .185)
assert f.select_epoch(summ, P)[1]['epoch'] == 2                      # tie on both -> earlier epoch
bad = copy.deepcopy(summ)
bad['systems'][0]['error_rows'] = 1
try:
    f.select_epoch(bad, P)
    raise SystemExit('error rows not refused')
except ValueError:
    pass
st = f._stats([1.0, 2.0, 3.0])
assert abs(st['sd'] - 1.0) < 1e-12 and st['range'] == 2.0, st       # sample SD uses n-1

root = Path(tempfile.mkdtemp(prefix='f2repro-check-'))
try:
    test_real = json.loads(Path('results/m6/test/summary.json').read_text(encoding='utf-8'))['systems'][0]
    for seed in (17, 18):
        sd = f.job_dir(root, 'F2', seed)
        (sd / 'score-val').mkdir(parents=True)
        (sd / 'score-test').mkdir()
        for e in (1, 2, 3):
            ck = sd / 'train' / f'epoch-{e}'
            ck.mkdir(parents=True)
            (ck / 'checkpoint-manifest.json').write_text('{}', encoding='ascii')
        s = {**summ, 'systems': [dict(x, system_id=x['system_id'].replace('s17', f's{seed}')) for x in summ['systems']]}
        (sd / 'score-val' / 'summary.json').write_text(json.dumps(s), encoding='utf-8')
        (sd / 'score-test' / f'F2-m6repro-s{seed}.json').write_text(json.dumps(test_real), encoding='utf-8')
        assert f.main(['select', '--job-dir', str(sd), '--prefix', f'F2-m6repro-s{seed}']) == 0
        sel = json.loads((sd / 'selection.json').read_text())
        assert sel['selected_epoch'] == 2 and sel['checkpoint'].endswith('epoch-2'), sel
    assert f.main(['compare', '--root', str(root), '--jobs', 'F2:17,F3:17,F2:18,F1:17,F2:19']) == 0
    out = json.loads((root / 'compare.json').read_text())['systems']
    assert abs(out['F2']['reproduction']['17']['test']['normalized_wer'] - 100 * test_real['metrics']['tr-normalized-v1']['wer']) < 1e-9
    assert abs(out['F2']['committed']['test']['normalized_wer'] - 10.370695114228653) < 1e-9
    assert abs(out['F2']['committed']['validation']['normalized_wer'] - 10.193558030042597) < 1e-9
    assert abs(out['F3']['committed']['test']['normalized_wer'] - 25.14978810463247) < 1e-9
    assert out['F1']['committed']['selected_epoch'] == 3 and out['F3']['committed']['selected_epoch'] == 3
    assert f.job_dir(root, 'F3', 17).name == 'f3-seed17' and f.job_dir(root, 'F1', 17).name == 'f1-seed17'
    assert f.main(['status', '--root', str(root)]) == 0
    print('F2REPRO SELF-CHECK PASS')
finally:
    shutil.rmtree(root)
