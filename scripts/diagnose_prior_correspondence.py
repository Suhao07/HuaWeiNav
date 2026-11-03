#!/usr/bin/env python3
"""Measure cross-map evidence overlap in a shared Habitat coordinate frame.

This offline diagnostic reconstructs mask-supported surfaces, not the builder's
internal filtered clouds. It reports every pair and makes no identity decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from prior_map.bundle import load_bundle
from prior_map.observations import backproject, load_depth, metric_depth_validity, write_json


def evidence_clouds(root, bundle, capture, voxel_m, max_depth_m, *, by_observation=False):
    """Reconstruct surfaces per predicted instance or per original observation.

    Per-observation surfaces expose disagreements hidden by an instance union;
    they retain the original mask identity and never split or relabel the map.
    """
    cells = ({obs['observation_id']: [] for obs in bundle['observations']} if by_observation
             else {obj.uid: [] for obj in bundle['prior_map'].objects})
    for obs in bundle['observations']:
        depth = load_depth(root / obs['depth_ref'])
        mask = np.asarray(Image.open(root / obs['mask_ref'])) > 0
        valid = mask & metric_depth_validity(depth, capture.get('capture_profile'))
        valid &= np.isfinite(depth) & (depth <= max_depth_m)
        points = backproject(np.where(valid, depth, 0), np.asarray(obs['intrinsics']),
                             np.asarray(obs['camera_to_map']))
        voxels = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0)
        for uid in ([obs['observation_id']] if by_observation else obs['object_uids']):
            cells[uid].append(voxels)
    return {uid: (np.unique(np.concatenate(parts), axis=0) + .5) * voxel_m
            for uid, parts in cells.items() if parts}


def overlap(current, historical, radius_m):
    """Return directional and pooled nearest-surface overlap at radius r.

    f(C,H)=sum_{p in C} 1[min_{q in H} ||p-q|| < r]/|C|.
    The pooled statistic weights the two fractions by their point counts, as
    in STRIVE fusion. Partial views can have high f(C,H) and low f(H,C).
    """
    dc = cKDTree(historical).query(current, eps=0)[0]
    dh = cKDTree(current).query(historical, eps=0)[0]
    nc, nh = int((dc < radius_m).sum()), int((dh < radius_m).sum())
    return {'current_support': nc / len(current), 'historical_support': nh / len(historical),
            'pooled_support': (nc + nh) / (len(current) + len(historical)),
            'current_median_distance_m': float(np.median(dc))}


def main():
    """Save all-pair spatial diagnostics without changing either source map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('historical', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--voxel-m', type=float, required=True)
    parser.add_argument('--radius-m', type=float, required=True)
    parser.add_argument('--max-depth-m', type=float, required=True)
    parser.add_argument('--current-observations', action='store_true',
                        help='Compare each current mask separately to historical instances')
    args = parser.parse_args()
    if min(args.voxel_m, args.radius_m, args.max_depth_m) <= 0:
        parser.error('Geometry scales must be positive')
    roots = [args.historical, args.current]
    bundles = [load_bundle(p) for p in roots]
    captures = [json.loads((p / 'build_receipt.json').read_text())['capture_manifest'] for p in roots]
    if any(c['pose_source'] != 'simulator_gt' for c in captures):
        raise ValueError('This diagnostic requires a shared simulator world; real maps need alignment')
    for field in ('scene_sha256', 'coordinates', 'pose_source'):
        if captures[0][field] != captures[1][field]:
            raise ValueError(f'Shared Habitat coordinates required: {field} differs')
    clouds = [evidence_clouds(p, b, c, args.voxel_m, args.max_depth_m,
                             by_observation=args.current_observations and i == 1)
              for i, (p, b, c) in enumerate(zip(roots, bundles, captures))]
    labels = [{o.uid: o.label for o in b['prior_map'].objects} for b in bundles]
    if args.current_observations:
        labels[1] = {o['observation_id']: o['detector_label'] for o in bundles[1]['observations']}
    pairs = []
    for cuid, current in clouds[1].items():
        for huid, historical in clouds[0].items():
            if len(current) and len(historical):
                pairs.append(dict(current_uid=cuid, historical_uid=huid,
                    current_label=labels[1][cuid], historical_label=labels[0][huid],
                    current_points=len(current), historical_points=len(historical),
                    **overlap(current, historical, args.radius_m)))
    sources = []
    for root, bundle in zip(roots, bundles):
        evidence = {obs[key] for obs in bundle['observations'] for key in ('mask_ref', 'depth_ref')}
        files = sorted(evidence | {'prior_map.json', 'observations.jsonl', 'build_receipt.json'})
        sources.append({'bundle': str(root), 'file_sha256': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}})
    write_json(args.output, {'scope': 'offline_surface_correspondence_diagnostic',
        'identity_decisions_made': False, 'semantic_gt_used': False,
        'current_unit': 'observation' if args.current_observations else 'instance',
        'current_observation_objects': ({o['observation_id']: o['object_uids']
            for o in bundles[1]['observations']} if args.current_observations else {}),
        'geometry': {'voxel_m': args.voxel_m, 'radius_m': args.radius_m,
                     'max_depth_m': args.max_depth_m, 'surface': 'mask_evidence_voxel_centres'},
        'sources': sources,
        'pairs': pairs})
    print(f'{len(pairs)} spatial pairs; no map changes or identity updates')


if __name__ == '__main__':
    main()
