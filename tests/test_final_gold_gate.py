"""A claimed-complete index must not open gold while systems/timings are missing."""
import json
from pathlib import Path
import runpy
import sys

import pytest
from turkish_asr_rdl.training.acoustic import sha256_file


def test_missing_systems_refuse_before_touching_nonexistent_gold(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/m45_stage_final_references.py'
    main=runpy.run_path(str(script))['main']
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'status':'frozen_for_final_evaluation',
        'systems':[{'system_id':'synthetic-native','kind':'adapted'},{'system_id':'synthetic-joint','kind':'relational'}],
        'rescore_conditions':[{'system_id':'synthetic-joint'}]}))
    digest=sha256_file(protocol)
    index=tmp_path/'index.json'
    index.write_text(json.dumps({'status':'inference_complete_gold_still_closed','protocol_sha256':digest,'runs':[]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'argv',['stage','--protocol',str(protocol),'--protocol-sha256',digest,
        '--inference-index',str(index),'--archive',str(tmp_path/'NONEXISTENT-GOLD.tar'),
        '--output','data/private/evaluator'])
    with pytest.raises(ValueError,match='incomplete system/mode inventory'):
        main()
    assert not (tmp_path/'data/private/evaluator').exists()
