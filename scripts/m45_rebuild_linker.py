"""Create a versioned linker-corrected graph bundle without changing learned weights."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from turkish_asr_rdl.rdl.graph import read_table
from turkish_asr_rdl.rdl.scoring import linker_from_tables
from turkish_asr_rdl.rdl.train import load_model
from turkish_asr_rdl.training.acoustic import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static-dir', type=Path, default=Path('data/relational/turas-tr-econ-legis-v0.1'))
    parser.add_argument('--lexicon', type=Path, default=Path('configs/datasets/turas-domain-lexicon-v0.1.json'))
    parser.add_argument('--lexicon-sha256', default='8ef903fe74135897a9d3781966c5e484c0711a6679d72346efb1ec89eff6df94')
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((Path.cwd() / 'artifacts').resolve()):
        parser.error('output must be a fresh directory under ignored artifacts/')
    _, _, original = load_model(args.source)
    if sha256_file(args.lexicon) != args.lexicon_sha256:
        raise ValueError('frozen lexicon checksum mismatch')
    config = json.loads((args.source / 'resolved-config.json').read_text(encoding='utf-8'))
    hashes = config['inputs']['tables_sha256']
    entities = read_table(args.static_dir / 'entities.csv', hashes['static/entities.csv'])
    aliases = read_table(args.static_dir / 'entity_aliases.csv', hashes['static/entity_aliases.csv'])
    relations = read_table(args.static_dir / 'relationships_static.csv', hashes['static/relationships_static.csv'])
    linker = linker_from_tables(entities, aliases, frozen_lexicon=json.loads(args.lexicon.read_text(encoding='utf-8')),
                                lexicon_sha256=args.lexicon_sha256)
    linker['entity_topics'] = {r['head_id']: r['tail_id'] for r in relations if r['relation_type_id'] == 'rel_belongs_to'}
    args.output.mkdir(parents=True, exist_ok=False)
    for name in original['files']:
        if name != 'linker.json':
            shutil.copy2(args.source / name, args.output / name)
    write_json(args.output / 'linker.json', linker)
    revision = {'policy': linker['policy'], 'parent_graph_manifest_sha256': sha256_file(args.source / 'artifact-manifest.json'),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'lexicon_sha256': args.lexicon_sha256, 'learned_weights_changed': False,
        'snapshot_changed': False, 'reason': 'preserve frozen prefix/exclusion/raw-token matching in inference'}
    write_json(args.output / 'linker-revision.json', revision)
    updated = {**original, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'parent_graph_manifest_sha256': revision['parent_graph_manifest_sha256'], 'linker_policy': linker['policy'],
        'files': {p.name: sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    write_json(args.output / 'artifact-manifest.json', updated)
    assert sha256_file(args.output / 'model.safetensors') == original['files']['model.safetensors']
    assert sha256_file(args.output / 'snapshot.json') == original['files']['snapshot.json']
    load_model(args.output)
    print(json.dumps({'status': 'linker_corrected_weights_unchanged', 'output': str(args.output),
                      'manifest_sha256': sha256_file(args.output / 'artifact-manifest.json')}))


if __name__ == '__main__':
    main()
