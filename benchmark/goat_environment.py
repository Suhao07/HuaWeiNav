"""Private continuous GOAT action boundary over one legacy Habitat carrier."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmark.goat import GOATEvaluator


class SegmentBoundary(RuntimeError):
    """A skill attempted an action after the current segment's budget boundary."""


class GOATEnvironment:
    """Count all public step calls; reserve one budget slot for a terminal event."""

    ACTIONS = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT',
               4: 'LOOK_UP', 5: 'LOOK_DOWN'}

    def __init__(self, base, episode, *, max_steps_per_subtask, global_budget,
                 record, after_reset=None):
        if min(max_steps_per_subtask, global_budget) < 1:
            raise ValueError('Budgets must include at least one terminal event')
        self._base, self._episode = base, episode
        self._segment_limit, self._global_limit = max_steps_per_subtask, global_budget
        self._record, self._after_reset = record, after_reset
        self.total_steps = self.segment_steps = self.index = self.parent_reset_count = 0
        self.evaluator = None
        self._reset_called = self._ended = False
        self._observation = None
        self.last_event = None

    @property
    def sim(self):
        """Expose the existing simulator for observed geometry and local planning."""
        return self._base.sim

    @property
    def current_episode(self):
        """Expose only the neutral carrier, never parent tasks or evaluator goals."""
        return self._base.current_episode

    @property
    def boundary_reason(self):
        """Segment episode_over is reversible; the underlying episode is not reset."""
        if self.total_steps >= self._global_limit - 1:
            return 'global_budget'
        if self.segment_steps >= self._segment_limit - 1:
            return 'subtask_budget'
        return None

    @property
    def episode_over(self):
        """Stop nested scan/verify loops at the current segment's action boundary."""
        return self._ended or self._base.episode_over or self.boundary_reason is not None

    def pose(self):
        """Read actual base position and XYZW rotation, not sensor or mapper pose."""
        state = self.sim.get_agent_state()
        rotation = state.rotation
        if hasattr(rotation, 'imag') and hasattr(rotation, 'real') and np.ndim(rotation) == 0:
            rotation = [*rotation.imag, rotation.real]
        return {'position': np.asarray(state.position, dtype=float).tolist(),
                'rotation_xyzw': np.asarray(rotation, dtype=float).tolist()}

    def _distance(self, index):
        endpoints = self._episode.subtasks[index].view_positions
        if not endpoints:
            raise ValueError('GOAT subtask has no view_positions')
        return float(self.sim.geodesic_distance(self.pose()['position'], list(endpoints)))

    def reset(self):
        """Initialize the carrier, navmesh and scorer exactly once at the parent start."""
        if self._reset_called:
            raise RuntimeError('Continuous GOAT permits only one parent reset')
        self._reset_called = True
        self.parent_reset_count += 1
        self._observation = self._base.reset()
        if self._after_reset is not None:
            self._after_reset(self.sim)
        pose = self.pose()
        if not np.allclose(pose['position'], self._episode.start_position, atol=1e-5):
            raise RuntimeError('Carrier reset did not select the parent start position')
        q, expected = np.asarray(pose['rotation_xyzw']), np.asarray(self._episode.start_rotation)
        if not (np.allclose(q, expected, atol=1e-5) or np.allclose(q, -expected, atol=1e-5)):
            raise RuntimeError('Carrier reset did not select the parent start rotation')
        self.evaluator = GOATEvaluator(self._episode, self._distance(0))
        return self._observation

    def get_metrics(self):
        """Legacy display placeholders only; GT distances never enter agent/model state."""
        return {'success': 0.0, 'spl': 0.0, 'distance_to_goal': float('nan')}

    def _event(self, action, before, after, *, reason=None, error=None):
        delta = float(np.linalg.norm(np.asarray(after['position']) - before['position']))
        self.total_steps += 1
        self.segment_steps += 1
        event = dict(step=self.total_steps, subtask_index=self.index, action=action,
                     segment_step=self.segment_steps, before=before, after=after,
                     path_delta=delta, reason=reason)
        if error is not None:
            event['error'] = f'{type(error).__name__}: {error}'
        self.last_event = event
        self._record(event)
        return delta

    def step(self, action):
        """Execute every physical action once, including actions inside nested skills."""
        if isinstance(action, dict):
            if action.get('action_args'):
                raise ValueError('GOAT forbids parameterized/teleport navigation actions')
            action = action['action']
        if isinstance(action, str):
            action = next((key for key, name in self.ACTIONS.items() if name == action.upper()), -1)
        if action not in self.ACTIONS:
            raise ValueError(f'Unsupported GOAT navigation action: {action}')
        if self._ended or self.evaluator is None:
            raise RuntimeError('GOAT action outside an active parent')
        if self._base.episode_over:
            raise RuntimeError('Carrier ended before the GOAT driver')
        if action != 0 and self.boundary_reason:
            raise SegmentBoundary(self.boundary_reason)
        if self.total_steps >= self._global_limit:
            raise SegmentBoundary('global_budget')
        before = self.pose()
        error = None
        try:
            self._observation = self._base.step(action)
        except Exception as exc:
            error = exc
            raise
        finally:
            after = self.pose()
            delta = self._event(self.ACTIONS[action], before, after, error=error)
        if action == 0 and delta != 0:
            raise RuntimeError('Global STOP moved the robot')
        self.evaluator.step(distance_to_goal=self._distance(self.index), path_delta=delta,
                            stop_called=action == 0)
        self._ended = action == 0
        if self._base.episode_over and not self._ended:
            raise RuntimeError('Unexpected underlying Habitat timeout')
        return self._observation

    def subtask_stop(self, reason):
        """Legacy equivalent of SubtaskStopAction: same-pose observation and advance."""
        if self._ended or self.evaluator is None or self._base.episode_over:
            raise RuntimeError('Subtask STOP outside an active parent')
        if self.total_steps >= self._global_limit - 1:
            raise SegmentBoundary('global_budget')
        before = self.pose()
        self._observation = self.sim.get_observations_at()
        after = self.pose()
        delta = self._event('SUBTASK_STOP', before, after, reason=reason)
        if before != after:
            raise RuntimeError('Subtask STOP changed the base pose')
        next_index = self.index + 1
        distance = self._distance(next_index) if next_index < len(self._episode.subtasks) else None
        self.evaluator.step(distance_to_goal=distance, path_delta=delta, subtask_stop=True)
        self.index = next_index
        self.segment_steps = 0  # Release only the wrapper's segment boundary.
        self._ended = self.evaluator.ended
        return self._observation

    def close(self):
        """Close the single owned carrier environment."""
        self._base.close()


def make_legacy_environment(episode, *, scene, scene_dataset, scenes_root,
                            profile, global_budget, output):
    """Build a legacy carrier and a post-reset Stretch navmesh receipt callback."""
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from benchmark.instance_goal_dataset import InstanceGoalDataset

    cfg = get_config()
    if not hasattr(cfg, 'defrost') or not hasattr(cfg, 'SIMULATOR'):
        raise RuntimeError('GOAT carrier requires legacy YACS Habitat (105 environment)')
    cfg.defrost()
    cfg.DATASET.TYPE = 'InstanceGoalNav-v1'
    cfg.DATASET.SCENES_DIR = str(scenes_root)
    cfg.DATASET.SPLIT = episode.split
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = global_budget + 1
    cfg.ENVIRONMENT.MAX_EPISODE_SECONDS = 10**9
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    cfg.TASK.TYPE = 'ObjectNav-v1'
    cfg.TASK.SENSORS = []
    cfg.TASK.MEASUREMENTS = []
    cfg.TASK.POSSIBLE_ACTIONS = list(GOATEnvironment.ACTIONS.values())
    sim = cfg.SIMULATOR
    sim.SCENE = str(scene)
    if not hasattr(sim, 'SCENE_DATASET'):
        raise RuntimeError('Legacy simulator lacks SCENE_DATASET support')
    sim.SCENE_DATASET = str(scene_dataset)
    sim.FORWARD_STEP_SIZE = profile['forward_step_size']
    sim.TURN_ANGLE = profile['turn_angle']
    sim.TILT_ANGLE = profile['tilt_angle']
    sim.ACTION_SPACE_CONFIG = 'v1'
    sim.HABITAT_SIM_V0.ALLOW_SLIDING = False
    sim.AGENT_0.HEIGHT, sim.AGENT_0.RADIUS = profile['height'], profile['radius']
    sim.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
    for sensor in (sim.RGB_SENSOR, sim.DEPTH_SENSOR):
        sensor.HEIGHT, sensor.WIDTH = profile['image_dimensions']
        sensor.HFOV = profile['hfov']
        sensor.POSITION = profile['camera_position']
        sensor.ORIENTATION = [0., 0., 0.]
    sim.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    sim.DEPTH_SENSOR.MAX_DEPTH = 5.0
    for key, name in GOATEnvironment.ACTIONS.items():
        if int(getattr(HabitatSimActions, name)) != key:
            raise RuntimeError('Legacy action IDs differ from GOAT primitive IDs')
    cfg.freeze()
    carrier = dict(episode_id='goat-carrier', scene_id=str(scene),
                   scene_dataset_config=str(scene_dataset),
                   start_position=list(episode.start_position), start_rotation=list(episode.start_rotation),
                   object_category='object', info={}, goals=[dict(
                       object_id='0', object_category='object', position=list(episode.start_position),
                       view_points=[])])
    dataset = InstanceGoalDataset()
    dataset.from_json(json.dumps(dict(
        category_to_task_category_id={'object': 0}, category_to_scene_annotation_category_id={'object': 0},
        episodes=[carrier])))
    base = habitat.Env(config=cfg, dataset=dataset)

    def after_reset(actual_sim):
        if actual_sim.sim_config.sim_cfg.scene_dataset_config_file != str(scene_dataset):
            raise RuntimeError('Carrier reset replaced the explicit scene dataset config')
        # 旧版 Lab 的 v1 action_space 使用整数动作键。
        actions = actual_sim.sim_config.agents[0].action_space
        effective_actions = []
        for key, amount in ((1, profile['forward_step_size']), (2, profile['turn_angle']),
                            (3, profile['turn_angle']), (4, profile['tilt_angle']), (5, profile['tilt_angle'])):
            action = actions[key]
            if (action.name != GOATEnvironment.ACTIONS[key].lower()
                    or not math.isclose(action.actuation.amount, amount)):
                raise RuntimeError('Effective simulator action differs from Stretch profile')
            effective_actions.append(dict(id=key, name=action.name, amount=action.actuation.amount))

        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height, settings.agent_radius = 1.41, 0.17
        settings.agent_max_climb, settings.cell_height = 0.20, 0.20
        actual_sim.pathfinder = habitat_sim.PathFinder()
        recomputed = bool(actual_sim.recompute_navmesh(
            actual_sim.pathfinder, settings, include_static_objects=False))
        loaded = bool(actual_sim.pathfinder.is_loaded)
        if not recomputed or not loaded:
            raise RuntimeError('Cannot recompute Stretch navmesh on the owned simulator')
        navigable = bool(actual_sim.pathfinder.is_navigable(episode.start_position))
        if not navigable:
            raise RuntimeError('Parent start is not navigable on recomputed Stretch navmesh')
        path = Path(output).resolve() / 'stretch.navmesh'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not actual_sim.pathfinder.save_nav_mesh(str(path)):
            raise RuntimeError('Cannot save recomputed Stretch navmesh')
        # Include all actual Recast defaults, not just the four overrides.
        values = {key: getattr(settings, key) for key in dir(settings)
                  if not key.startswith('_') and isinstance(getattr(settings, key), (bool, int, float))}
        return dict(habitat_sim_version=habitat_sim.__version__, habitat_lab_version=habitat.__version__,
                    scope='legacy_habitat_compat_not_official_version_equivalence',
                    settings=values, include_static_objects=False, recomputed=recomputed,
                    is_loaded=loaded, start_navigable=navigable, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), actions=effective_actions)
    return base, cfg, after_reset


def _pose_record(state):
    """Serialize a base or sensor pose, preserving the complete XYZW orientation."""
    rotation = state.rotation
    if hasattr(rotation, 'imag') and hasattr(rotation, 'real') and np.ndim(rotation) == 0:
        rotation = [*rotation.imag, rotation.real]
    return dict(position=np.asarray(state.position, dtype=float).tolist(),
                rotation=np.asarray(rotation, dtype=float).tolist())


def _same_pose(before, after):
    """Allow floating-point roundoff and the two equivalent quaternion signs."""
    left, right = _pose_record(before), _pose_record(after)
    q, r = np.asarray(left['rotation']), np.asarray(right['rotation'])
    return (np.allclose(left['position'], right['position'], rtol=0, atol=1e-5)
            and (np.allclose(q, r, rtol=0, atol=1e-5)
                 or np.allclose(q, -r, rtol=0, atol=1e-5)))


def render_reference_image(actual_sim, parameters, path):
    """Render the world-coordinate goal camera and restore all agent/sensor poses."""
    import habitat_sim
    from habitat_sim._ext import habitat_sim_bindings as hsim
    from habitat_sim.agent.agent import SixDOFPose
    from habitat_sim.utils.common import quat_from_coeffs
    from PIL import Image

    uuid = 'goat_reference'
    agent = actual_sim.get_agent(0)
    original = deepcopy(agent.get_state())
    if (uuid in original.sensor_states or uuid in actual_sim._sensors
            or any(s.uuid == uuid for s in agent.agent_config.sensor_specifications)):
        raise RuntimeError('Temporary goat_reference sensor UUID is already occupied')
    camera = {key: deepcopy(parameters[key])
              for key in ('position', 'rotation', 'hfov', 'image_dimensions')}
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = camera['image_dimensions']
    spec.hfov = camera['hfov']
    try:
        actual_sim.add_sensor(spec)
        state = deepcopy(original)
        state.sensor_states[uuid] = SixDOFPose(
            position=np.asarray(camera['position'], dtype=float),
            rotation=quat_from_coeffs(camera['rotation']))
        agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
        # Rounded parent quaternions require the actual scene transform for world translation.
        node = agent._sensors[uuid].node
        node.translation = node.parent.absolute_transformation().inverted().transform_point(camera['position'])
        sensor = actual_sim._sensors[uuid]
        actual_spec = sensor._spec
        rendered_pose = agent.get_state().sensor_states[uuid]
        if not _same_pose(state.sensor_states[uuid], rendered_pose):
            raise RuntimeError('Reference sensor did not select the world camera pose')
        sensor_parameters = dict(uuid=actual_spec.uuid, sensor_type=str(actual_spec.sensor_type),
                                 sensor_subtype=str(actual_spec.sensor_subtype),
                                 resolution=[int(v) for v in actual_spec.resolution],
                                 hfov=float(actual_spec.hfov), **_pose_record(rendered_pose))
        sensor.draw_observation()
        image = np.asarray(sensor.get_observation())[:, :, :3].copy()
    finally:
        try:
            # Official 0.2.3 removal sequence, also supported by Sim 0.2.4.
            if uuid in actual_sim._sensors:
                actual_sim._sensors[uuid].close()
                del actual_sim._sensors[uuid]
            if uuid in agent._sensors:
                hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, uuid)
                del agent._sensors[uuid]
            agent.agent_config.sensor_specifications = [
                s for s in agent.agent_config.sensor_specifications if s.uuid != uuid]
        finally:
            agent.set_state(deepcopy(original), reset_sensors=False, infer_sensor_states=False)
            restored = agent.get_state()
            if (not _same_pose(original, restored)
                    or original.sensor_states.keys() != restored.sensor_states.keys()
                    or any(not _same_pose(pose, restored.sensor_states[key])
                           for key, pose in original.sensor_states.items())):
                raise RuntimeError('Reference rendering failed full agent/sensor pose restoration')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format='PNG')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                image_parameters=camera, actual_sensor=sensor_parameters, pose_restored=True)
