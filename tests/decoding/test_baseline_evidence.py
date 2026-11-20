"""Synthetic contract fixtures: baseline recovery accepts N-best without status."""
from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from turkish_asr_rdl.decoding.export import nbest_row, prediction_row
from turkish_asr_rdl.decoding.interface import Candidate, Hypothesis
from turkish_asr_rdl.schemas.jsonl import write_jsonl
from turkish_asr_rdl.training.acoustic import sha256_file, write_json


def test_finalize_real_export_schema_and_reject_changed_payload(tmp_path, make_input, run_context):
    script = Path(__file__).resolve().parents[2] / 'scripts/m45_validation_baseline.py'
    finalize = runpy.run_path(str(script))['finalize_outputs']
    ctx = replace(run_context, split='validation')
    inp = make_input('synthetic-only')
    candidate = Candidate('c1', 1, 'SYNTHETIC fixture', (1,), -1.0, 'synthetic-test-only')
    hyp = Hypothesis(inp.utterance_id, 'ok', candidate.text, (candidate,))
    output = tmp_path / 'out'
    output.mkdir()
    nbest = nbest_row(hyp, inp, ctx, 1, 'synthetic-test-only')
    assert 'status' not in nbest  # actual N-best contract; previous code assumed it
    nbest_hash = write_jsonl(output / 'nbest.jsonl', [nbest])
    prediction_hash = write_jsonl(output / 'predictions.jsonl', [prediction_row(hyp, inp, ctx, 'synthetic-timing', nbest_hash)])
    files = {'nbest.jsonl': nbest_hash, 'predictions.jsonl': prediction_hash}
    write_json(output / 'manifest.json', {'outputs': dict(files)})
    files['manifest.json'] = sha256_file(output / 'manifest.json')
    inference = tmp_path / 'synthetic-input.jsonl'
    inference.write_text('{}\n')
    config = {'model': {'id': 'synthetic-only', 'revision': 'c' * 40, 'weight_sha256': ctx.model_artifact_sha256},
              'dataset_release': ctx.dataset_release, 'split_manifest_sha256': ctx.split_manifest_sha256}
    finalize(config, output, files, [inp.utterance_id], inference)
    evidence = json.loads((output / 'baseline-evidence.json').read_text())
    assert evidence['metrics_status'] == 'pending_validation_references'
    assert evidence['utterances'] == 1
    (output / 'nbest.jsonl').write_text('changed synthetic fixture')
    with pytest.raises(ValueError, match='hash mismatch'):
        finalize(config, output, files, [inp.utterance_id], inference)
