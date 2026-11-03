#!/usr/bin/env python3
"""Find strong raw-observation support groups within frozen predicted instances.

No GT is consumed and no assignments change. An undirected edge requires both
surface coverages > 0.85 (the existing strict target-fusion coverage threshold).
Components are evidence candidates, not accepted physical object identities.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mapping.object_fusion import score_object_candidates
from prior_map.bundle import load_bundle
from prior_map.observations import load_fusion_observation, write_json


def main():
    """Audit every predicted instance in the trace using identical frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    bundle = load_bundle(args.bundle)
    records = {o['observation_id']: o for o in bundle['observations']}
    receipt = json.loads((args.bundle/'build_receipt.json').read_text())
    trace = json.loads(args.trace.read_text())
    groups = defaultdict(list)
    for row in trace['observations']:
        if row['assigned_uid'] is not None:
            groups[row['assigned_uid']].append(row['observation_id'])

    @lru_cache(maxsize=64)
    def load(oid):
        return load_fusion_observation(args.bundle, records[oid], voxel_m=receipt['voxel_m'],
            max_depth_m=receipt['max_depth_m'], profile=receipt['capture_manifest'].get('capture_profile'))[0]

    edges, components = [], []
    pair_count = 0
    for uid, ids in groups.items():
        adjacency = {oid: set() for oid in ids}
        for first, second in combinations(ids, 2):
            pair_count += 1
            _, forward, reverse = score_object_candidates(load(first), [load(second)])
            if min(float(forward[0]), float(reverse[0])) > .85:
                adjacency[first].add(second)
                adjacency[second].add(first)
                edges.append(dict(first=first, second=second, uid=uid,
                    forward=float(forward[0]), reverse=float(reverse[0]),
                    different_frame=records[first]['frame_id'] != records[second]['frame_id']))
        remaining = set(ids)
        for seed in ids:
            if seed not in remaining:
                continue
            pending, members = [seed], []
            remaining.remove(seed)
            while pending:
                oid = pending.pop()
                members.append(oid)
                for neighbor in sorted(adjacency[oid] & remaining):
                    remaining.remove(neighbor)
                    pending.append(neighbor)
            positions = {tuple(np.round(np.asarray(records[i]['camera_to_map'])[:3, 3], 4)) for i in members}
            components.append(dict(uid=uid, observations=sorted(members),
                frame_count=len({records[i]['frame_id'] for i in members}),
                camera_position_count=len(positions)))
    sources = [args.trace, args.bundle/'observations.jsonl', args.bundle/'build_receipt.json',
               Path(__file__), Path('mapping/object_fusion.py'), Path('prior_map/observations.py')]
    result = dict(scope='raw_support_components_diagnostic_not_refusion', coverage_threshold=.85,
        position_rounding_m=.0001, pair_count=pair_count, edges=edges, components=components,
        elapsed_s=time.monotonic()-started,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    write_json(args.output, result)
    print(json.dumps(dict(pairs=pair_count,edges=len(edges),components=len(components),elapsed_s=result['elapsed_s'])))


if __name__ == '__main__':
    main()
