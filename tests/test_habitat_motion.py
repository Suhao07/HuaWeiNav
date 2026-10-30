"""Validate simulator stepping at the real backend boundary, without Habitat imports."""
import pytest
from backends.habitat.motion import HabitatExecution


class Environment:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return {"action": action, "step": len(self.actions)}


def test_habitat_actions_are_executed_once_and_receipts_are_primitive_scoped():
    env = Environment()
    execution = HabitatExecution(env)
    first = execution.step(1, request_id="forward-1")
    assert execution.step(1, request_id="forward-1") == first
    assert env.actions == [1]
    assert execution.motion.last_terminal_status.metadata["completion_scope"] == "primitive"
    with pytest.raises(ValueError, match="another goal"):
        execution.step(2, request_id="forward-1")
    execution.step(2)
    execution.step(0)
    assert env.actions == [1, 2, 0]


def test_habitat_hold_preserves_simulation_state_until_resume():
    env = Environment()
    execution = HabitatExecution(env)
    execution.motion.hold()
    with pytest.raises(RuntimeError, match="held"):
        execution.step(1)
    assert env.actions == []
    execution.motion.resume()
    assert execution.step(1) == {"action": 1, "step": 1}
    assert execution.controller.observations == {}


def test_waypoint_execution_pumps_existing_planner_and_polling_is_read_only():
    from types import SimpleNamespace
    import numpy as np
    from motion.contracts import NavigationIntent, MotionGoalMode, NavigationStatusCode
    from scene.contracts import Pose3D

    env = Environment()
    env.episode_over = False
    env.sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=(0, 1, 0)))
    execution = HabitatExecution(env)
    actions = iter([1, 2, 0])
    planner_points = []

    def get_next_action(point):
        planner_points.append(point.tolist())
        return next(actions)

    agent = SimpleNamespace(
        env=env, mapper=SimpleNamespace(initial_position=np.array([10, 20, 0])),
        planner=SimpleNamespace(get_next_action=get_next_action), episode_steps=0, on_node_flag=True,
    )
    def update(on_node_flag):
        agent.episode_steps += 1
    agent.update_trajectory = update
    execution.controller.agent = agent
    intent = NavigationIntent(MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D((1, 2, 0), frame_id="mapper_local"))
    execution.motion.submit(intent, intent.to_motion_goal())
    assert execution.motion.poll_active().status.status == NavigationStatusCode.QUEUED
    assert env.actions == []
    for _ in range(2):
        execution.controller.advance()
        assert execution.motion.poll_active().tracking_active
    execution.controller.advance()
    assert execution.motion.poll_active().status.succeeded()
    assert env.actions == [1, 2]
    assert planner_points == [[11, 1, 22]] * 3
    assert agent.episode_steps == 2
    execution.step(0)
    assert env.actions[-1] == 0


def test_historical_base_goal_preserves_floor_height_through_motion_owner():
    """A historical upstairs base pose must reach the planner without flattening."""
    from types import SimpleNamespace
    from motion.contracts import NavigationIntent, MotionGoalMode
    from scene.contracts import Pose3D
    env = Environment()
    env.episode_over = False
    queried_paths, planner_points = [], []
    def distance(start, end):
        queried_paths.append((list(start), end.tolist()))
        return 8.0
    env.sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=(2., 1., -3.)),
                              geodesic_distance=distance)
    def action(point):
        planner_points.append(point.tolist())
        return 0
    agent = SimpleNamespace(env=env, episode_steps=0,
                            planner=SimpleNamespace(get_next_action=action))
    execution = HabitatExecution(env)
    execution.controller.agent = agent
    intent = NavigationIntent(MotionGoalMode.IMPROVE_VIEW,
        goal_pose=Pose3D((4., 7., 5.), frame_id="map_z_up"),
        metadata={"arrival_radius_m": .2, "historical_object_uid": "prior:desk"})
    execution.motion.submit(intent, intent.to_motion_goal())
    execution.controller.advance()
    assert queried_paths == [([2., 1., -3.], [4., 5., -7.])]
    assert planner_points == [[4., 5., -7.]]
    assert execution.motion.poll_active().status.succeeded()
    assert env.actions == []  # Waypoint arrival does not issue task STOP.
