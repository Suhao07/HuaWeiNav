"""Action-executed Habitat capture with the same portable RGB-D protocol."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from .capture import WORLD_TO_MAP, OPTICAL_TO_GL
from .navmesh_build import verify_build_receipt
from prior_map.observations import read_records, write_frame, write_json


def collect_walk(reference, scene, navmesh, build_receipt, output, *, max_actions=160, max_positions=3,
                 capture_profile=None):
    """Follow reachable reference positions and scan by actual turn actions.

    The reference supplies task-independent coverage positions and sensor setup.
    Initialize once; every later pose comes from sim.step, never interpolation or
    teleportation. Geometry-assisted greedy following is a collection baseline,
    separate from STRIVE exploration and from instruction navigation acceptance.
    """
    import habitat_sim
    import quaternion
    reference, scene, navmesh, output = map(Path, (reference, scene, navmesh, output))
    source = json.loads((reference / 'manifest.json').read_text())
    scene_digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    mesh_digest = hashlib.sha256(navmesh.read_bytes()).hexdigest()
    if scene_digest != source['scene_sha256']:
        raise ValueError('Reference and walking scene differ')
    settings = verify_build_receipt(build_receipt, mesh_digest, scene_digest)
    profile = json.loads(Path(capture_profile).read_text()) if capture_profile is not None else None
    if profile is not None:
        if profile['scene_sha256'] != scene_digest or profile['navmesh_sha256'] != mesh_digest:
            raise ValueError('Matched capture profile requires the same scene and active NavMesh')
        if max_actions != profile['max_actions']:
            raise ValueError('Matched capture must use the profile action budget')
    if max_actions < 1 or max_positions < 1:
        raise ValueError('Use positive action and position budgets')
    output.mkdir(parents=True, exist_ok=False)
    config = source['config'] if profile is None else profile
    turn = 10. if profile is None else float(profile['turn_degrees'])
    forward = .25 if profile is None else float(profile['move_forward_m'])
    arrival_radius = .2 if profile is None else float(profile['arrival_radius_m'])
    scan_steps = panorama_action_count(turn)
    manifest = {**source, 'status': 'running', 'mode': 'continuous_walk',
        'navmesh_sha256': mesh_digest, 'navmesh_path': str(navmesh),
        'collector': 'habitat_greedy_follower_actual_actions',
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_capture_sha256': hashlib.sha256((reference / 'manifest.json').read_bytes()).hexdigest(),
        'robot_build_settings': settings, 'action_config': {'move_forward_m': forward, 'turn_degrees': turn},
        'config': config, 'capture_profile': profile,
        'max_actions': max_actions, 'max_positions': max_positions,
        'frame_count': 0, 'position_count': 0, 'action_count': 0,
        'termination': None, 'elapsed_seconds': None,
        'coverage_requirement_met': None}
    write_json(output / 'manifest.json', manifest)
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id, cfg.enable_physics = str(scene), False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.radius, agent_cfg.height = settings['agent_radius'], settings['agent_height']
    sensors = []
    for name, kind in [('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = name, kind
        sensor.resolution = [config['height'], config['width']]
        sensor.position, sensor.hfov = [0., config['sensor_height'], 0.], config['hfov']
        sensors.append(sensor)
    agent_cfg.sensor_specifications = sensors
    agent_cfg.action_space = {name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
        for name, amount in [('move_forward', forward), ('turn_left', turn), ('turn_right', turn)]}
    sim = None
    started = time.monotonic()
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        pf = sim.pathfinder
        if not pf.load_nav_mesh(str(navmesh)):
            raise ValueError('Failed loading built NavMesh')
        positions = []
        for row in read_records(reference / 'poses.jsonl'):
            p = np.asarray(row['agent_position_habitat'])
            if not any(np.array_equal(p, q) for q in positions):
                positions.append(p)
        valid = [p for p in positions if pf.is_navigable(p, max_y_delta=.05)]
        if not valid:
            raise ValueError('No reference position lies on the current NavMesh')
        start = valid[0] if profile is None else np.asarray(profile['start_position_habitat'])
        if not pf.is_navigable(start, max_y_delta=.05):
            raise ValueError('Capture start is outside the loaded NavMesh')
        def distance(a, b):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = a, b
            return float(path.geodesic_distance) if pf.find_path(path) else float('inf')
        candidates = [p for p in (valid[1:] if profile is None else valid)
                      if (profile is None or distance(start, p) > .2) and math.isfinite(distance(start, p))]
        route = [start]
        while candidates and len(route) < max_positions:
            index = min(range(len(candidates)), key=lambda j: distance(route[-1], candidates[j]))
            route.append(candidates.pop(index))
        write_json(output / 'route.json', {'positions_habitat': [p.tolist() for p in route],
            'reference_positions': len(positions), 'navigable_positions': len(valid),
            'reachable_positions': len(route) + len(candidates), 'selection': 'nearest_geodesic_within_start_island'})
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quaternion.quaternion(*(profile['start_rotation_wxyz'] if profile else [1, 0, 0, 0]))
        agent = sim.get_agent(0)
        agent.set_state(state)
        follower = sim.make_greedy_follower(0, goal_radius=arrival_radius, forward_key='move_forward',
                                          left_key='turn_left', right_key='turn_right')
        focal = config['width'] / (2 * math.tan(math.radians(config['hfov']) / 2))
        intrinsic = np.array([[focal, 0, (config['width']-1)/2], [0, focal, (config['height']-1)/2], [0, 0, 1.]])
        if profile is not None:
            intrinsic = np.asarray(profile['intrinsics'])
        step, poses, groups, path_length = 0, [], [], 0.
        island = int(pf.get_island(start))
        def record(obs, phase, waypoint):
            camera = agent.get_state().sensor_states['rgb']
            pose = np.eye(4)
            pose[:3, :3] = WORLD_TO_MAP @ quaternion.as_rotation_matrix(camera.rotation) @ OPTICAL_TO_GL
            pose[:3, 3] = WORLD_TO_MAP @ camera.position
            depth = obs['depth'] if profile is None else np.clip(
                obs['depth'], profile['depth_min_m'], profile['depth_max_m'])
            write_frame(output, step, obs['rgb'], depth, intrinsic, pose,
                        sim_step=step, phase=phase, position_id=waypoint, island_id=island,
                        valid_depth_fraction=float((obs['depth'] > 0).mean()))
            poses.append({'frame_id': step, 'agent_position_habitat': agent.get_state().position.tolist(),
                          'camera_to_map': pose.tolist()})
        record(sim.get_sensor_observations(), 'initial', 0)
        def act(action, phase, waypoint):
            nonlocal step, path_length
            before = agent.get_state().position.copy()
            obs = sim.step(action)
            step += 1
            travelled = float(np.linalg.norm(agent.get_state().position - before))
            path_length += travelled
            record(obs, phase, waypoint)
            with (output / 'actions.jsonl').open('a') as stream:
                stream.write(json.dumps({'sim_step': step, 'action': action, 'phase': phase,
                    'waypoint': waypoint, 'from_frame': step-1, 'to_frame': step,
                    'translation_m': travelled, 'collided': bool(obs['collided'])}) + '\n')
        reached = 0
        for waypoint, goal in enumerate(route):
            while step < max_actions:
                action = follower.next_action_along(goal)
                if action is None:
                    break
                act(action, 'transit', waypoint)
            arrived = distance(agent.get_state().position, goal) <= arrival_radius
            if not arrived:
                break
            reached += 1
            # 与巡视相同：预留完整全景预算；到达事实与完整观察分别记账。
            if profile is not None and step + scan_steps > max_actions:
                break
            scan_frames = [step]
            for _ in range(scan_steps):
                if step == max_actions:
                    break
                act('turn_left' if profile is None else profile['scan_action'], 'scan', waypoint)
                scan_frames.append(step)
            groups.append({'group_id': waypoint, 'kind': 'action_executed_panorama',
                'frame_ids': scan_frames, 'complete': len(scan_frames) == scan_steps + 1})
            if step == max_actions:
                break
        for name, rows in [('poses', poses), ('observation_groups', groups)]:
            (output / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        finished = reached == len(route) and len(groups) == len(route) and all(g['complete'] for g in groups)
        manifest.update(status='complete', frame_count=step+1, action_count=step,
            position_count=reached, planned_position_count=len(route), actual_path_length_m=path_length,
            elapsed_seconds=time.monotonic()-started,
            termination='planned_route_finished' if finished else 'budget_exhausted')
        write_json(output / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        if sim is not None:
            sim.close()


def panorama_action_count(turn_degrees):
    """Require n * delta = 360 degrees for an action-executed full panorama."""
    if not 0 < turn_degrees <= 360:
        raise ValueError('Turn increment must be in (0, 360]')
    count = round(360 / turn_degrees)
    if not math.isclose(count * turn_degrees, 360., abs_tol=1e-6):
        raise ValueError('Turn increment must divide a full panorama')
    return count
