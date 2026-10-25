"""Verify baseline ordering and purpose-specific execution without a simulator."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest

from planning.exploration_policy import select_exploration_node, find_closest_nodes
from backends.habitat.guided_runtime import GuidedNavigation


def policy_agent():
    node = NS(idx=1, room_idx=0)
    room = NS(state=0)
    mapper = NS(nodes=[node], room_nodes=[room],
                explore_in_room=Mock(return_value=node),
                explore_in_room_relocate=Mock(return_value=node),
                explore_after_fully_explored=Mock(return_value=node),
                get_candidate_room_fully_explored=Mock(return_value=room),
                get_candidate_room_fully_explored_by_distance=Mock(return_value=room),
                find_closet_viewpoint_in_room=Mock(return_value=node))
    return NS(mapper=mapper, current_node_idx=0, instruct_goal="Find chair",
              relocate=False, gpt_relocate=True, task=NS(plan=object()))


def test_local_room_policy_avoids_semantic_model():
    agent = policy_agent()
    node, source = select_exploration_node(agent, episode_idx=0, step=13)
    assert node.idx == 1 and source == "current_room"
    agent.mapper.get_candidate_room_fully_explored.assert_not_called()


@pytest.mark.parametrize("semantic", [True, False])
def test_room_selection_only_after_local_exhaustion(semantic):
    agent = policy_agent()
    agent.gpt_relocate = semantic
    agent.mapper.explore_in_room.return_value = None
    _, source = select_exploration_node(agent, episode_idx=0, step=32)
    assert source == ("room_semantic_policy" if semantic else "room_distance_policy")
    assert agent.mapper.get_candidate_room_fully_explored.call_count == int(semantic)
    assert agent.mapper.get_candidate_room_fully_explored_by_distance.call_count == int(not semantic)


def test_fully_explored_uses_original_remaining_nodes():
    agent = policy_agent()
    agent.mapper.explore_in_room.return_value = None
    agent.mapper.room_nodes[0].state = 1
    _, source = select_exploration_node(agent, episode_idx=0, step=32)
    assert source == "remaining_nodes"
    agent.mapper.get_candidate_room_fully_explored.assert_not_called()


def test_distance_ranking_keeps_stable_tie_break(monkeypatch):
    import numpy as np
    import planning.exploration_policy as policy
    nodes = [NS(idx=i) for i in range(3)]
    monkeypatch.setattr(policy, "_geodesic_distances_to_nodes", lambda *a: np.array([float('inf'), 2., 2.]))
    monkeypatch.setattr(policy, "_frontier_prior_score", lambda *a: 0.)
    monkeypatch.setattr(policy, "_prior_frontier_distance_bias_m", lambda *a: 0.)
    monkeypatch.setattr(policy, "_prior_frontier_bias_enabled", lambda *a: False)
    monkeypatch.setattr(policy, "_record_frontier_choice", lambda **kw: None)
    assert find_closest_nodes(NS(), nodes) is nodes[1]


def test_directed_verification_reuses_feedback_without_panorama(monkeypatch):
    import backends.habitat.guided_runtime as runtime
    import sys
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller",
                        NS(rotate_toward_object_for_recheck=lambda agent: True))
    agent = NS(episode_steps=20, object_final=NS(uid="obj"), need_check_again=False,
               last_check_again_evidence={"frame": 20}, final_stop_decision="need_better_view",
               check_again=Mock(), check_again_postion=(1, 2, 0))
    def verify(**kwargs):
        assert kwargs['evidence_override'] == {"frame": 20}
        agent.need_check_again = True
        return False
    agent.final_instruction_check = verify
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record, guided.submit = agent, Mock(), Mock()
    assert guided.verify() == "moving"
    guided.submit.assert_called_once_with((1, 2, 0), candidate_uid="obj", source="baseline_view_feedback")


@pytest.mark.parametrize("kind", ["frontier", "object", "blocked"])
def test_terminal_event_dispatches_observation_or_verification_once(monkeypatch, tmp_path, kind):
    from pathlib import Path
    from motion.contracts import NavigationStatus, NavigationStatusCode
    import backends.habitat.guided_runtime as runtime
    events = []
    agent = NS(save_dir=str(tmp_path), episode_steps=1, scan_complete=True, env=NS(episode_over=False),
               task=NS(progress=NS(completed=False)),
               mapper=NS(nodes=[NS(idx=0)], traj=[], change_state=lambda n: events.append("visited")))
    def step(action):
        events.append("stop")
        agent.env.episode_over = True
    agent.execution = NS(controller=NS(advance=lambda: events.append("advance")), step=step)
    agent.update_trajectory = lambda: None
    monkeypatch.setattr(runtime, "run_observation_mapping_cycle", lambda *a, **kw: events.append("observe"))
    metadata = {"node_index": 0} if kind != "object" else {}
    status = NavigationStatusCode.BLOCKED if kind == "blocked" else NavigationStatusCode.REACHED
    feedback = NS(tracking_active=False, status=NavigationStatus(status),
                  intent=NS(metadata=metadata, target_object_uid="obj"))
    motion = NS(active_request_id="goal")
    def poll():
        motion.active_request_id = None
        return feedback
    motion.poll_active = poll
    def construct(agent, directory, max_calls):
        Path(directory).mkdir(parents=True)
        def verify():
            events.append("verify")
            agent.task.progress.completed = True
            return "verified_finish"
        def recover(reason):
            events.append("recover")
            agent.env.episode_over = True
            return {"name": "observe"}
        return NS(directory=directory, tools=NS(motion=motion, objects={"obj": object()}),
                  runtime=NS(model_calls=0), record=lambda *a, **kw: None,
                  verify=verify, select=lambda idx: False, recover=recover)
    monkeypatch.setattr(runtime, "GuidedNavigation", construct)
    runtime.run_guided_episode(agent, episode_idx=0, max_steps=500)
    if kind == "frontier":
        assert events == ["observe", "advance", "observe", "visited"]
    elif kind == "object":
        assert events == ["observe", "advance", "verify", "stop"]
    else:
        assert events == ["observe", "advance", "recover"]


def test_initial_and_feedback_views_share_planner_arrival(monkeypatch):
    from backends.habitat.motion import HabitatMotionController
    from motion.supervisor import MotionSupervisor
    from motion.contracts import NavigationStatusCode
    import navigation.action_controller as actions
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint", lambda *args: 0.3)
    monkeypatch.setattr(actions, "next_action_to_waypoint", lambda *args: 1)
    env = NS(episode_over=False, step=Mock(return_value={}))
    agent = NS(task=NS(spec=NS(revision=1)), episode_steps=20, update_trajectory=Mock(), on_node_flag=True)
    controller = HabitatMotionController(env, agent)
    motion = MotionSupervisor(controller)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.sequence, guided.record = agent, 0, Mock()
    guided.tools = NS(motion=motion)

    guided.submit((1, 2, 0), candidate_uid="obj", source="baseline_viewpoint")
    controller.advance()
    assert motion.poll_active().status.status == NavigationStatusCode.RUNNING
    env.step.assert_called_once_with(1)
    motion.cancel()
    motion.poll_active()
    env.step.reset_mock()

    # 同样距离为 0.3m，反馈提案必须执行 planner 的向前动作。
    guided.submit((1, 2, 0), candidate_uid="obj", source="baseline_view_feedback")
    controller.advance()
    assert motion.poll_active().status.status == NavigationStatusCode.RUNNING
    env.step.assert_called_once_with(1)


def test_exploration_preserves_node_turns_then_accumulates_translation(monkeypatch):
    from backends.habitat.motion import HabitatMotionController
    from motion.contracts import MotionGoal, MotionGoalMode
    from scene.contracts import Pose3D
    import navigation.action_controller as actions
    commands = iter([3, 1, 3])
    monkeypatch.setattr(actions, "next_action_to_waypoint", lambda *args: next(commands))
    agent = NS(episode_steps=13, on_node_flag=True, update_trajectory=Mock())
    controller = HabitatMotionController(NS(episode_over=False, step=lambda action: {}), agent)
    controller.send_goal(MotionGoal(mode=MotionGoalMode.GO_TO_FRONTIER,
                                    goal_pose=Pose3D((1, 0, 0), frame_id="mapper_local")))
    for _ in range(3):
        controller.advance()
    assert [call.args[0] for call in agent.update_trajectory.call_args_list] == [True, False, False]
