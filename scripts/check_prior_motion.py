#!/usr/bin/env python3
"""Execute a historical base viewpoint through the shared motion owner.

This is a Habitat-Sim motion integration check, not an ObjectNav episode. It
uses Habitat's greedy follower through the existing controller interface and
records fresh RGB-D. Task parsing and semantic acceptance remain separate
acceptance steps; heading alignment can be explicitly enabled.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.capture import WORLD_TO_MAP, OPTICAL_TO_GL
from backends.habitat.prior_viewpoints import resolve_habitat_viewpoints
from backends.habitat.motion import HabitatExecution
from motion.contracts import NavigationIntent, MotionGoalMode
from scene.contracts import Pose3D
from prior_map.observations import write_frame, write_json
from prior_map.query_binding import queried_object_uid


def main():
    """Run an explicit historical-instance motion diagnostic with fresh evidence."""
    import habitat_sim
    import quaternion
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'navmesh', 'build_receipt', 'output'):
        parser.add_argument(name, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--object-uid')
    selection.add_argument('--query-receipt', type=Path,
                           help='Bind map retrieval to nearest reachable historical evidence')
    parser.add_argument('--start', nargs=3, type=float, required=True)
    parser.add_argument('--max-actions', type=int, default=200)
    parser.add_argument('--align-heading', action='store_true')
    parser.add_argument('--approach-proposal', type=Path,
                        help='Execute an explicit current-candidate proposal instead of a historical pose')
    parser.add_argument('--look-at', nargs=3, type=float, help='Current object point in map coordinates')
    args = parser.parse_args()
    if args.max_actions < 1:
        raise ValueError('Use a positive action budget')
    if args.query_receipt is not None:
        args.object_uid = queried_object_uid(json.loads(args.query_receipt.read_text()), args.bundle)
    candidates = resolve_habitat_viewpoints(args.bundle, args.navmesh, args.start,
        build_receipt=args.build_receipt, object_uid=args.object_uid)
    if not candidates['viewpoints']:
        raise ValueError('Historical instance has no reachable evidence viewpoint')
    # 查询接入选择最短路径；直接指定 UID 的旧诊断保留最远点以检查实际平移。
    choose = min if args.query_receipt is not None else max
    goal = choose(candidates['viewpoints'], key=lambda v: v['path_distance_m'])
    if args.approach_proposal is not None:
        approach = json.loads(args.approach_proposal.read_text())
        if approach['pose_in_map'] is None:
            raise ValueError('Current candidate has no executable approach proposal')
        if not np.allclose(approach['start_habitat'], args.start, atol=1e-6, rtol=0):
            raise ValueError('Approach start differs from this diagnostic start')
        goal = dict(uid='current-candidate-approach', pose_in_map=approach['pose_in_map'],
                    path_distance_m=approach['selected']['travel_distance'],
                    current_candidate_uid=approach['candidate_uid'])
    capture = json.loads((args.bundle / 'build_receipt.json').read_text())['capture_manifest']
    sensor = capture['config']
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', scope='motion_integration_not_task_success',
        historical_object_uid=args.object_uid, selected_viewpoint=goal,
        selection=('current_candidate_baseline_proposal' if args.approach_proposal else
                   'nearest_reachable_from_bound_query' if args.query_receipt is not None else
                   'farthest_reachable_for_motion_test'), start_habitat=args.start,
        query_receipt_sha256=(hashlib.sha256(args.query_receipt.read_bytes()).hexdigest()
                              if args.query_receipt is not None else None),
        max_actions=args.max_actions, heading_alignment=False, semantic_verification=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output / 'receipt.json', receipt)
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id, config.enable_physics = capture['scene_path'], False
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.radius = candidates['robot_profile']['radius_m']
    agent_config.height = candidates['robot_profile']['height_m']
    specs = []
    for name, kind in [('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = [sensor['height'], sensor['width']]
        spec.position, spec.hfov = [0., sensor['sensor_height'], 0.], sensor['hfov']
        specs.append(spec)
    agent_config.sensor_specifications = specs
    agent_config.action_space = {name: habitat_sim.agent.ActionSpec(name,
        habitat_sim.agent.ActuationSpec(amount=amount)) for name, amount in
        [('move_forward', .25), ('turn_left', 10.), ('turn_right', 10.),
         ('look_up', 10.), ('look_down', 10.)]}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent_config]))
    try:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError('Cannot load checked NavMesh')
        state = habitat_sim.AgentState()
        state.position = np.asarray(args.start)
        sim.get_agent(0).set_state(state)
        destination = WORLD_TO_MAP.T @ np.asarray(goal['pose_in_map'])[:3, 3]
        def distance(start, end):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = start, end
            return float(path.geodesic_distance) if sim.pathfinder.find_path(path) else float('inf')
        get_state = lambda: sim.get_agent(0).get_state()
        follower = sim.make_greedy_follower(0, goal_radius=.2, forward_key=1, left_key=2, right_key=3)
        environment = SimpleNamespace(episode_over=False,
            sim=SimpleNamespace(get_agent_state=get_state, geodesic_distance=distance))
        actions = []
        focal = sensor['width'] / (2 * math.tan(math.radians(sensor['hfov']) / 2))
        intrinsic = np.array([[focal, 0, (sensor['width']-1)/2],
                              [0, focal, (sensor['height']-1)/2], [0, 0, 1.]])
        def record(observation):
            camera = get_state().sensor_states['rgb']
            pose = np.eye(4)
            pose[:3, :3] = WORLD_TO_MAP @ quaternion.as_rotation_matrix(camera.rotation) @ OPTICAL_TO_GL
            pose[:3, 3] = WORLD_TO_MAP @ camera.position
            write_frame(args.output, len(actions), observation['rgb'], observation['depth'],
                        intrinsic, pose, agent_position_habitat=get_state().position.tolist())
        record(sim.get_sensor_observations())
        def step(action):
            before = get_state().position.copy()
            observation = sim.step({1: 'move_forward', 2: 'turn_left', 3: 'turn_right',
                                    4: 'look_up', 5: 'look_down'}[action])
            actions.append(dict(action=action, position_habitat=get_state().position.tolist(),
                translation_m=float(np.linalg.norm(get_state().position-before)),
                collided=bool(observation['collided'])))
            environment.episode_over = len(actions) >= args.max_actions
            record(observation)
            return observation
        environment.step = step
        agent = SimpleNamespace(env=environment, episode_steps=0, on_node_flag=False,
            planner=SimpleNamespace(get_next_action=lambda p: follower.next_action_along(p) or 0))
        def update(_on_node):
            agent.episode_steps = len(actions)
        agent.update_trajectory = update
        execution = HabitatExecution(environment)
        execution.controller.agent = agent
        intent = NavigationIntent(MotionGoalMode.IMPROVE_VIEW,
            goal_pose=Pose3D(tuple(np.asarray(goal['pose_in_map'])[:3, 3]), frame_id='map_z_up'),
            metadata={'historical_object_uid': args.object_uid})
        execution.motion.submit(intent, intent.to_motion_goal())
        while execution.motion.active_goal_id is not None:
            execution.controller.advance()
            execution.motion.poll_active()
        motion_status = execution.motion.last_terminal_status.status.value
        forward = np.asarray(goal['pose_in_map'])[:3, 0]
        desired_yaw = math.atan2(forward[1], forward[0])
        def heading_error():
            optical = WORLD_TO_MAP @ quaternion.as_rotation_matrix(get_state().sensor_states['rgb'].rotation) @ OPTICAL_TO_GL
            yaw = math.atan2(optical[1, 2], optical[0, 2])
            return math.atan2(math.sin(desired_yaw-yaw), math.cos(desired_yaw-yaw))
        # 位置终态释放槽位后，转向动作继续通过同一 owner；误差限为离散转角的一半。
        tolerance = math.radians(5.) + 1e-6
        if args.align_heading and motion_status == 'reached':
            while abs(heading_error()) > tolerance and len(actions) < args.max_actions:
                execution.step(2 if heading_error() > 0 else 3)
        pitch_error = None
        if args.look_at is not None and motion_status == 'reached':
            def pitch_error_now():
                camera = get_state().sensor_states['rgb']
                direction = np.asarray(args.look_at) - WORLD_TO_MAP @ camera.position
                desired = math.atan2(direction[2], np.linalg.norm(direction[:2]))
                rotation = WORLD_TO_MAP @ quaternion.as_rotation_matrix(camera.rotation) @ OPTICAL_TO_GL
                current = math.atan2(rotation[2, 2], np.linalg.norm(rotation[:2, 2]))
                return desired - current
            while abs(pitch_error_now()) > tolerance and len(actions) < args.max_actions:
                execution.step(4 if pitch_error_now() > 0 else 5)
            pitch_error = math.degrees(pitch_error_now())
        receipt.update(status='complete', motion_status=motion_status,
            action_count=len(actions), final_distance_m=distance(get_state().position, destination),
            path_length_m=sum(row['translation_m'] for row in actions),
            collisions=sum(row['collided'] for row in actions),
            heading_alignment=args.align_heading and abs(heading_error()) <= tolerance,
            look_at_point_map=args.look_at, final_pitch_error_degrees=pitch_error,
            final_heading_error_degrees=math.degrees(heading_error()))
        write_json(args.output / 'actions.json', actions)
        write_json(args.output / 'receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('motion_status', 'action_count', 'final_distance_m', 'path_length_m')}))
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(args.output / 'receipt.json', receipt)
        raise
    finally:
        sim.close()


if __name__ == '__main__':
    main()
