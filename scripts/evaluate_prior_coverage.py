#!/usr/bin/env python3
"""Evaluate scan positions on a frozen, independent geodesic probe set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.observations import read_records, write_json


def main():
    """Keep evaluation probes fixed when candidate sampling budgets change.

    Coverage is mean_q[ min_p d_nav(q,p) <= radius ]. This measures sampled
    navigation-position coverage, not visible surfaces or semantic rooms.
    """
    import habitat_sim
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('navmesh', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--probe-seed', type=int,
                        help='Independent audit RNG seed; avoids reuse of collection RNG samples')
    parser.add_argument('--probe-count', type=int, default=1024)
    args = parser.parse_args()
    current = json.loads((args.capture / 'manifest.json').read_text())
    reference = json.loads((args.reference / 'manifest.json').read_text())
    for key in ('scene_sha256', 'navmesh_sha256', 'coordinates'):
        if current[key] != reference[key]:
            raise ValueError(f'Capture/reference differ: {key}')
    if hashlib.sha256(args.navmesh.read_bytes()).hexdigest() != reference['navmesh_sha256']:
        raise ValueError('Evaluation NavMesh differs from recorded NavMesh')
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(args.navmesh)):
        raise ValueError('Could not load evaluation NavMesh')
    poses = {r['frame_id']: r['agent_position_habitat']
             for r in read_records(args.capture / 'poses.jsonl')}
    positions = {}
    for frame in read_records(args.capture / 'frames.jsonl'):
        positions.setdefault(frame['island_id'], {})[frame['position_id']] = poses[frame['frame_id']]
    frozen = json.loads((args.reference / 'coverage.json').read_text())
    if args.probe_count < 1:
        parser.error('Probe count must be positive')
    regions = []
    for region in frozen['regions']:
        island = region['island_id']
        if args.probe_seed is None:
            probes = region['check_points_habitat']
        else:
            # 采集器增大采样预算会消费原随机流中的旧检查点；审计使用独立随机流。
            pathfinder.seed(args.probe_seed + island)
            probes = [pathfinder.get_random_navigable_point(island_index=island).tolist()
                      for _ in range(args.probe_count)]
        distances = []
        for probe in probes:
            nearest = math.inf
            for position in positions.get(island, {}).values():
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = probe, position
                if pathfinder.find_path(path):
                    nearest = min(nearest, float(path.geodesic_distance))
            distances.append(nearest)
        covered = sum(d <= region['radius_m'] for d in distances)
        regions.append({'island_id': island, 'check_count': len(probes),
            'position_count': len(positions.get(island, {})),
            'covered_count': covered, 'fraction': covered / len(probes),
            'radius_m': region['radius_m'], 'check_points_habitat': probes,
            'nearest_path_distance_m': [d if math.isfinite(d) else None for d in distances]})
    files = [args.capture / name for name in ('manifest.json', 'poses.jsonl', 'frames.jsonl')]
    files += [args.reference / 'manifest.json', args.reference / 'coverage.json', args.navmesh]
    write_json(args.output, {'measurement': 'frozen_sampled_geodesic_position_coverage',
        'probe_seed': args.probe_seed,
        'surface_coverage': 'not_measured', 'regions': regions,
        'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    print(json.dumps([{k: r[k] for k in ('island_id', 'position_count', 'fraction')}
                      for r in regions]))


if __name__ == '__main__':
    main()
