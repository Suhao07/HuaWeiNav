#!/usr/bin/env python3
"""Bind a current RGB-D candidate to STRIVE's existing better-view policy."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.observations import load_depth
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json
from backends.habitat.capture import WORLD_TO_MAP
from backends.habitat.navmesh_build import verify_build_receipt


def main():
    """Resolve fresh candidate geometry without accepting its semantic label."""
    import hashlib
    import habitat_sim
    import open3d as o3d
    from mapping_utils.geometry import gpu_pointcloud_from_array
    from planning.viewpoint_policy import build_check_again_viewpoints
    from navigation.view_verification_controller import _log_and_pick_best_proposal
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'motion', 'navmesh', 'build_receipt', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--candidate-uid', required=True)
    parser.add_argument('--required-stop-distance', type=float, required=True)
    parser.add_argument('--arrival-tolerance', type=float, default=.2)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    build = json.loads((args.bundle / 'build_receipt.json').read_text())
    capture = build['capture_manifest']
    verify_build_receipt(args.build_receipt, hashlib.sha256(args.navmesh.read_bytes()).hexdigest(),
                         capture['scene_sha256'])
    if hashlib.sha256((args.motion / 'receipt.json').read_bytes()).hexdigest() != capture['source_motion_receipt_sha256']:
        raise ValueError('Current candidate is bound to another motion')
    obs, = [r for r in bundle['observations'] if args.candidate_uid in r['object_uids']]
    current_frame = json.loads((args.motion / 'frames.jsonl').read_text().splitlines()[-1])
    start = np.asarray(current_frame['agent_position_habitat'])
    pose = np.asarray(obs['camera_to_map'])
    mask = np.asarray(Image.open(args.bundle / obs['mask_ref'])) > 0
    depth = load_depth(args.bundle / obs['depth_ref'])
    points = backproject(np.where(mask & (depth <= build['max_depth_m']), depth, 0),
                         np.asarray(obs['intrinsics']), pose)
    # Canonical map (x,-z,y) -> baseline absolute mapper axes (x,z,y).
    mapper_points = points * [1., -1., 1.]
    cloud = gpu_pointcloud_from_array(mapper_points.astype(np.float32),
        np.full(mapper_points.shape, 160, dtype=np.uint8), o3d.core.Device('CPU:0'))
    obj = SimpleNamespace(pcd=cloud, bbox=obs['bbox_xyxy'], position=mapper_points.mean(axis=0))
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(args.navmesh)):
        raise ValueError('Cannot load checked NavMesh')
    current_xy = (WORLD_TO_MAP @ start)[:2]
    closest = points[np.linalg.norm(points[:, :2] - current_xy, axis=1).argmin()]
    endpoint = pf.snap_point([closest[0], start[1], -closest[1]])
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = start, endpoint
    if not pf.find_path(path):
        raise ValueError('No path to snapped current candidate approach point')
    proposals, _ = build_check_again_viewpoints(object_node=obj,
        camera_intrinsic=np.asarray(obs['intrinsics']), mapper_initial_position=np.zeros(3),
        habitat_path_points=path.points, target_height=pose[2, 3],
        success_distance=args.required_stop_distance, stop_criterion=.7,
        arrival_tolerance=args.arrival_tolerance, image_width=depth.shape[1], image_height=depth.shape[0])
    reachable = []
    for proposal in proposals:
        position = proposal['position']
        request = habitat_sim.ShortestPath()
        request.requested_start, request.requested_end = start, [position[0], start[1], position[1]]
        if pf.find_path(request) and request.geodesic_distance > args.arrival_tolerance:
            proposal['travel_distance'] = float(request.geodesic_distance)
            proposal['position'] = position.tolist()
            reachable.append(proposal)
    selected = _log_and_pick_best_proposal(reachable)
    base_pose = None
    if selected is not None:
        p = selected['position']
        position = np.array([p[0], -p[1], start[1]])
        forward = points.mean(axis=0) - position
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        base_pose = np.eye(4)
        base_pose[:3, :3] = np.column_stack((forward, np.cross([0, 0, 1], forward), [0, 0, 1]))
        base_pose[:3, 3] = position
    write_json(args.output, dict(scope='current_candidate_baseline_view_proposal_not_execution',
        candidate_uid=args.candidate_uid, required_stop_distance=args.required_stop_distance,
        look_at_point_map=points.mean(axis=0).tolist(),
        arrival_tolerance=args.arrival_tolerance, start_habitat=start.tolist(),
        selected=selected, pose_in_map=None if base_pose is None else base_pose.tolist(), proposals=reachable))
    print(f'{len(reachable)} reachable proposals; selected={selected is not None}')


if __name__ == '__main__':
    main()
