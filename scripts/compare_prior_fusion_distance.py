#!/usr/bin/env python3
"""Replay identical accepted observations with dense and indexed nearest queries."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import open3d as o3d
import mapping.object_fusion as fusion
from mapping_utils.geometry import pointcloud_distance, gpu_pointcloud_from_array, gpu_cluster_filter
from mapping_utils.representation import ObjectNode
from prior_map.observations import load_depth
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json


def main():
    """Freeze masks, geometry, observation order and labels; vary nearest solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.bundle)
    bundle = load_bundle(root)
    receipt = json.loads((root / 'build_receipt.json').read_text())
    inputs = []
    for obs in bundle['observations']:
        depth = load_depth(root / obs['depth_ref'])
        mask = np.asarray(Image.open(root / obs['mask_ref'])) > 0
        points = backproject(np.where(mask & (depth <= receipt['max_depth_m']), depth, 0),
                             np.array(obs['intrinsics']), np.array(obs['camera_to_map']))
        cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 160), o3d.core.Device('CPU:0'))
        cloud = gpu_cluster_filter(cloud.voxel_down_sample(receipt['voxel_m']))
        inputs.append((obs, cloud))
    indexed = fusion.nearest_distances
    bound = fusion.clouds_may_overlap
    results = {}
    try:
        for name, distance in [('dense', pointcloud_distance), ('indexed', indexed)]:
            fusion.nearest_distances = distance
            fusion.clouds_may_overlap = (lambda a, b: True) if name == "dense" else bound
            objects, assignments = [], []
            started = time.monotonic()
            for obs, cloud in inputs:
                current = cloud.clone()
                points = current.point.positions.numpy()
                entity = ObjectNode(current, current, obs['detector_label'], obs['detector_score'],
                                    points.mean(axis=0), len(points), None, obs['bbox_xyxy'])
                entity.uid = obs['observation_id']
                objects, indices = fusion.associate_object_entities(objects, [entity], target='',
                                                                     is_target_tag=lambda tag: False)
                assignments.append(objects[indices[0]].uid)
            results[name] = {'elapsed_s': time.monotonic() - started, 'assignments': assignments,
                'objects': [{'uid': o.uid, 'label': o.tag, 'position': o.position.tolist(),
                             'point_count': int(o.pcd.point.positions.shape[0])} for o in objects]}
    finally:
        fusion.nearest_distances = indexed
        fusion.clouds_may_overlap = bound
    same = results['dense']['assignments'] == results['indexed']['assignments']
    original_assignments = [obs['object_uids'][0].split(':', 1)[1] for obs, _ in inputs]
    report = {'scope': 'fixed_accepted_observations_fusion_solver_contrast',
        'indexed_uses_aabb_bound': True, 'observation_count': len(inputs), 'assignment_parity': same,
        'dense_reproduces_saved_assignments': results['dense']['assignments'] == original_assignments,
        'object_records_equal': results['dense']['objects'] == results['indexed']['objects'],
        'runs': results}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}))
    print(json.dumps({name: {'elapsed_s': row['elapsed_s'], 'objects': len(row['objects'])}
                      for name, row in results.items()}))


if __name__ == '__main__':
    main()
