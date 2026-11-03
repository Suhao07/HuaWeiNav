#!/usr/bin/env python3
"""Check whether the existing largest-cluster rule retains DBSCAN noise."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import open3d as o3d
from prior_map.observations import load_depth
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json


def main():
    """Replay saved masks with frozen depth/voxel settings; preserve original map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.bundle)
    bundle = load_bundle(root)
    receipt = json.loads((root / 'build_receipt.json').read_text())
    rows = []
    object_points = defaultdict(list)
    for obs in bundle['observations']:
        depth = load_depth(root / obs['depth_ref'])
        mask = np.asarray(Image.open(root / obs['mask_ref'])) > 0
        points = backproject(np.where(mask & (depth <= receipt['max_depth_m']), depth, 0),
                             np.array(obs['intrinsics']), np.array(obs['camera_to_map']))
        cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points.astype(np.float32)))
        cloud = cloud.voxel_down_sample(receipt['voxel_m'])
        labels = cloud.cluster_dbscan(eps=.5, min_points=10, print_progress=False).numpy()
        ids, counts = np.unique(labels, return_counts=True)
        selected = int(ids[np.argmax(counts)])
        retained = cloud.point.positions.numpy()[labels == selected]
        for uid in obs['object_uids']:
            object_points[uid].append(retained)
        rows.append({'observation_id': obs['observation_id'], 'selected_label': selected,
            'point_count': len(labels), 'noise_count': int((labels == -1).sum()),
            'cluster_sizes': {str(int(i)): int(n) for i, n in zip(ids, counts)}})
    geometry = []
    for obj in bundle['prior_map'].objects:
        points = np.concatenate(object_points[obj.uid])
        cells = np.floor(points / receipt['voxel_m']).astype(np.int64)
        unique = len(np.unique(cells, axis=0))
        geometry.append({'object_uid': obj.uid, 'replayed_point_count': len(points),
                         'exported_point_count': obj.metadata['point_count'],
                         'occupied_voxels': unique, 'point_to_voxel_ratio': len(points) / unique})
    report = {'scope': 'replay_existing_cluster_rule_on_accepted_masks',
        'observation_count': len(rows),
        'selected_noise_count': sum(r['selected_label'] == -1 for r in rows),
        'observations': rows, 'object_geometry': geometry,
        'replayed_points': sum(r['replayed_point_count'] for r in geometry),
        'occupied_voxels_sum': sum(r['occupied_voxels'] for r in geometry)}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('observations', 'object_geometry')}))


if __name__ == '__main__':
    main()
