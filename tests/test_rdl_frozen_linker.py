"""Public frozen lexicon regression cases; these are synthetic phrase fixtures."""
import csv
import json
from pathlib import Path

import pytest

from turkish_asr_rdl.rdl.scoring import linker_from_tables, link_entities
from turkish_asr_rdl.training.acoustic import sha256_file


def frozen_linker():
    root = Path(__file__).resolve().parents[1]
    static = root / 'data/relational/turas-tr-econ-legis-v0.1'
    with (static / 'entities.csv').open(encoding='utf-8', newline='') as stream:
        entities = list(csv.DictReader(stream))
    with (static / 'entity_aliases.csv').open(encoding='utf-8', newline='') as stream:
        aliases = list(csv.DictReader(stream))
    lexicon = root / 'configs/datasets/turas-domain-lexicon-v0.1.json'
    return linker_from_tables(entities, aliases, frozen_lexicon=json.loads(lexicon.read_text(encoding='utf-8')),
                              lexicon_sha256=sha256_file(lexicon)), entities, aliases


def test_frozen_prefix_exclusion_and_longest_span_rules():
    linker, _, _ = frozen_linker()
    assert link_entities('bankan\u0131n', linker) == ['BANK_BANKA']
    assert link_entities('\u00fccretlerin', linker) == ['LAB_UCRET']
    assert link_entities('\u00fccretsiz', linker) == []
    assert link_entities('PLAN VE B\u00dcT\u00c7E KOM\u0130SYONUNUN', linker) == ['PUBFIN_PLAN_BUTCE_KOMISYONU']
    assert link_entities("KDV'ye", linker) == ['TAX_KDV']


def test_wildcard_tables_need_full_exclusions_and_must_match_frozen_ids():
    linker, entities, aliases = frozen_linker()
    with pytest.raises(ValueError, match='wildcard aliases require'):
        linker_from_tables(entities, aliases)
    with pytest.raises(ValueError, match='source checksum'):
        linker_from_tables(entities, aliases, frozen_lexicon=linker['lexicon'])
    with pytest.raises(ValueError, match='differ from the frozen lexicon'):
        linker_from_tables(entities, aliases[:-1], frozen_lexicon=linker['lexicon'],
                            lexicon_sha256=linker['lexicon_source_sha256'])
    # Existing exact-token artifacts remain reproducible, not silently changed.
    legacy = {'aliases': [{'tokens': ['banka'], 'entity_id': 'BANK_BANKA'}]}
    assert link_entities('banka', legacy) == ['BANK_BANKA']
    assert link_entities('bankan\u0131n', legacy) == []
