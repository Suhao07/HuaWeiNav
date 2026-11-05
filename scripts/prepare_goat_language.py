"""Freeze an auditable single-goal GOAT language subset for the project acceptance set."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from benchmark.goat_language import freeze_subset, language_tasks


def main():
    """Read official split shards and write tasks plus a reproducible selection receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    episodes, sources = [], {}
    for file in sorted((args.split_dir/'content').glob('*.json.gz')):
        sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        with gzip.open(file, 'rt') as stream:
            episodes.extend(language_tasks(json.load(stream), split=args.split_dir.name, source_file=str(file)))
    selected, unique_count = freeze_subset(episodes, args.count, args.seed)
    categories = {name: index for index, name in enumerate(sorted({ep['object_category'] for ep in selected}))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(dict(episodes=selected, category_to_task_category_id=categories,
                       category_to_scene_annotation_category_id=categories), stream)
    receipt = dict(protocol='goat_language_single_parent_start_v1', source_split=args.split_dir.name,
        source_hashes=sources, eligible_description_tasks=len(episodes), unique_tasks=unique_count,
        selected=len(selected), seed=args.seed, task_ids=[e['episode_id'] for e in selected],
        task_file_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_hashes','task_ids')},indent=2))


if __name__ == '__main__':
    main()
