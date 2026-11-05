"""Freeze separate IntentionNav and LangMap instruction evaluation manifests."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from benchmark.langmap import convert_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intention-metadata', type=Path, required=True)
    parser.add_argument('--langmap-annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    intents = [json.loads(line) for line in args.intention_metadata.read_text().splitlines() if line.strip()]
    if len(intents) != 500 or len({i['selection_id'] for i in intents}) != 500:
        raise ValueError('Expected the frozen 500 unique IntentionNav tasks')
    # 两种表达共享同一起点与目标，condition_id 用于逐次评测，task_id 用于配对分析。
    intention = [dict(condition_id=f"{item['selection_id']}:{style}", task_id=item['selection_id'],
                      scene_id=item['scene_id'], style=style, instruction=item[style])
                 for item in intents for style in ('natural_en', 'emotional_en')]
    tasks, source_hashes = [], {}
    for file in sorted(args.langmap_annotations.glob('*.json.gz')):
        source_hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        with gzip.open(file, 'rt') as stream:
            rows, _ = convert_scene(json.load(stream), file.name.removesuffix('.json.gz'))
        # 单纯类别导航属于简单组；需求组保留房间、区域与实例描述。
        tasks.extend(row for row in rows if row['info']['navigation_type'] != 'object')
    unique = {}
    for row in tasks:
        identity = json.dumps([row['scene_id'], row['start_position'], row['start_rotation'],
                               sorted(g['object_id'] for g in row['goals']), row['info']['instruction']], sort_keys=True)
        unique.setdefault(identity, row)
    if len(unique) < 1000:
        raise ValueError(f'Only {len(unique)} unique LangMap description tasks available')
    selected = sorted(unique.values(), key=lambda row: hashlib.sha256(
        f"{args.seed}:{row['episode_id']}".encode()).hexdigest())[:1000]
    langmap = [dict(condition_id=row['episode_id'], task_id=row['episode_id'],
                   scene_id=row['scene_id'], instruction=row['info']['instruction'],
                   navigation_type=row['info']['navigation_type']) for row in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in [('intentionnav_1000', intention), ('langmap_1000', langmap)]:
        file = args.output / f'{name}.jsonl'
        file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    runtime_file = args.output / 'langmap_runtime.json'
    runtime_file.write_text(json.dumps({'runtime_episodes': [
        {key: row[key] for key in ('episode_id', 'scene_id', 'start_position', 'start_rotation', 'object_category')}
        for row in selected]}, indent=2) + '\n')
    hashes[runtime_file.name] = hashlib.sha256(runtime_file.read_bytes()).hexdigest()
    receipt = dict(seed=args.seed, counts={'intentionnav_conditions':1000, 'intentionnav_geometry':500,
        'langmap_tasks':1000}, intention_styles=['natural_en','emotional_en'],
        langmap_types=dict(Counter(row['navigation_type'] for row in langmap)),
        source_hashes={'intention': hashlib.sha256(args.intention_metadata.read_bytes()).hexdigest(),
                       'langmap':source_hashes}, manifest_sha256=hashes,
        reporting='Separate dataset metrics; IntentionNav also grouped by style and paired geometric task')
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
