"""Resolve simulated historical camera poses into checked NavMesh candidates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from prior_map.bundle import load_bundle
from .capture import WORLD_TO_MAP
from .navmesh_build import verify_build_receipt


def camera_to_base_pose(camera_to_map, sensor_height):
    """Remove the known vertical camera offset and orient an x-forward base.

    Applies to the collector's [0, sensor_height, 0] Habitat sensor mounting.
    Handheld camera trajectories require a separate ground/robot pose resolver.
    """
    camera = np.asarray(camera_to_map, dtype=float)
    forward = camera[:3, 2].copy()
    forward[2] = 0
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError('Camera has no horizontal viewing direction')
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.])
    base = np.eye(4)
    base[:3, :3] = np.column_stack([forward, np.cross(up, forward), up])
    base[:3, 3] = camera[:3, 3] - [0., 0., sensor_height]
    return base


def resolve_habitat_viewpoints(bundle_directory, navmesh_path, start_habitat, *, build_receipt=None,
                              object_uid=None):
    """Return current-path-checked candidates while keeping the map read-only.

    Requires the known simulation capture mounting. A rebuilt NavMesh must
    provide a receipt binding its hash and settings to the captured scene.
    Candidate records carry reported settings and the current start. Legacy
    meshes can omit build settings; Habitat then guesses defaults. These values
    remain unverified and cannot establish robot compatibility for admission.
    object_uid restricts candidates to evidence of that historical instance.
    This function never submits a goal, changes task progress or verifies an object.
    """
    import habitat_sim
    root, navmesh_path = Path(bundle_directory), Path(navmesh_path)
    bundle = load_bundle(root)
    if object_uid is not None and object_uid not in {obj.uid for obj in bundle['prior_map'].objects}:
        raise ValueError(f'Unknown historical object: {object_uid}')
    evidence_ids = {obs['observation_id'] for obs in bundle['observations']
                    if object_uid is None or object_uid in obs['object_uids']}
    capture = json.loads((root / 'build_receipt.json').read_text())['capture_manifest']
    # 两种采集器使用相同的刚性相机安装；采集运动方式不改变外参。
    if capture['pose_source'] != 'simulator_gt' or capture['mode'] not in ('coverage_scan', 'continuous_walk'):
        raise ValueError('This resolver requires the known Habitat capture sensor mounting')
    digest = hashlib.sha256(navmesh_path.read_bytes()).hexdigest()
    verified_settings = None
    if build_receipt is not None:
        verified_settings = verify_build_receipt(build_receipt, digest, capture['scene_sha256'])
    elif digest != capture['navmesh_sha256']:
        raise ValueError('NavMesh differs from the recorded scene geometry')
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(navmesh_path)):
        raise ValueError(f'Cannot load NavMesh: {navmesh_path}')
    start = np.asarray(start_habitat, dtype=float)
    if not pf.is_navigable(start, max_y_delta=.05):
        raise ValueError('Current navigation start is outside NavMesh')
    settings = pf.nav_mesh_settings
    profile = {'source': 'pathfinder_reported_settings', 'build_profile_verified': False,
               'radius_m': float(settings.agent_radius),
               'height_m': float(settings.agent_height), 'max_climb_m': float(settings.agent_max_climb),
               'max_slope_deg': float(settings.agent_max_slope)}
    if verified_settings is not None:
        # 以哈希绑定的构建记录为准；旧格式加载器可能补出默认参数。
        profile = {'source': 'scene_bound_build_receipt', 'build_profile_verified': True,
                   'radius_m': verified_settings['agent_radius'],
                   'height_m': verified_settings['agent_height'],
                   'max_climb_m': verified_settings['agent_max_climb'],
                   'max_slope_deg': verified_settings['agent_max_slope']}
    viewpoints, rejected = [], []
    for historical in bundle['viewpoints']:
        if historical['kind'] != 'observation_pose':
            continue
        observation_ids = [uid for uid in historical['observation_ids'] if uid in evidence_ids]
        if not observation_ids:
            continue
        base = camera_to_base_pose(historical['pose_in_map'], capture['config']['sensor_height'])
        destination = WORLD_TO_MAP.T @ base[:3, 3]
        if not pf.is_navigable(destination, max_y_delta=.05):
            rejected.append({'source_viewpoint': historical['uid'], 'reason': 'base_outside_navmesh'})
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = start, destination
        if not pf.find_path(path):
            rejected.append({'source_viewpoint': historical['uid'], 'reason': 'unreachable_from_current_start'})
            continue
        viewpoints.append({'uid': f"navigation:{historical['uid']}", 'kind': 'navigation_viewpoint',
            'pose_in_map': base.tolist(), 'pose_axes': 'x_forward_y_left_z_up',
            'observation_ids': observation_ids, 'historical_object_uid': object_uid,
            'robot_profile': profile,
            'reachability_status': ('path_reachable_build_profile_verified' if verified_settings is not None
                                    else 'path_reachable_profile_unverified'), 'checked_map_revision': digest,
            'source_map_revision': bundle['manifest']['map_revision'],
            'checked_start_habitat': start.tolist(), 'path_distance_m': float(path.geodesic_distance),
            'motion_admission': ('requires_runtime_robot_profile_match_and_current_recheck' if verified_settings is not None
                                 else 'requires_verified_navmesh_build_profile_and_current_runtime_recheck')})
    return {'scope': 'historical_viewpoint_candidates_not_motion_execution',
        'map_id': bundle['manifest']['map_id'], 'historical_object_uid': object_uid,
        'navmesh_sha256': digest,
        'robot_profile': profile, 'viewpoints': viewpoints, 'rejected': rejected}
