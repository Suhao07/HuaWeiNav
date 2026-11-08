"""Verify baseline ordering and purpose-specific execution without a simulator."""
from contextlib import contextmanager, nullcontext
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


def test_task_free_patrol_reuses_room_distance_policy():
    agent = policy_agent()
    agent.task, agent.instruct_goal, agent.gpt_relocate = None, "", False
    agent.mapper.explore_in_room.return_value = None
    node, source = select_exploration_node(agent, episode_idx=0, step=32)
    assert source == "room_distance_policy" and node.idx == 1
    agent.mapper.get_candidate_room_fully_explored_by_distance.assert_called_once_with(
        "", idx=0, step=32, plan=None)
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
    import numpy as np
    tilts = []
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller",
                        NS(rotate_toward_object_for_recheck=lambda agent: True,
                           tilt_camera=lambda agent, pitch: tilts.append(pitch) or True))
    # 光轴 f=-R[:,2] 指向 +x；目标中心在 (2,2)，残余偏角 45°。
    camera = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    agent = NS(episode_steps=20, task=NS(spec=NS(revision=1)), need_check_again=False,
               selected_candidate=NS(role="primary"),
               object_final=NS(uid="obj", position=np.array([2., 2., 1.])),
               mapper=NS(current_position=np.zeros(3), current_rotation=camera),
               _observation_identity=lambda uid: {"observation_id": "0:20"},
               last_check_again_evidence={"frame": 20}, final_stop_decision="need_better_view",
               check_again=Mock(), check_again_postion=(1, 2, 0))
    def verify(**kwargs):
        assert kwargs['evidence_override'] == {"frame": 20}
        agent.need_check_again = True
        return False
    agent.final_instruction_check = verify
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record, guided.submit = agent, Mock(), Mock()
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.candidate_motion_goals = {}
    guided.view_attempted = lambda *args: False
    guided.tools = NS(publish_object_viewpoints=lambda uid: [NS(
        pose=NS(position=(1, 2, 0)), metadata={"recommended": True})])
    assert guided.verify((1, 2, 0)) == "moving"
    # 未完成的复核在下一次移动前恢复水平相机。
    assert tilts == [0.0]
    guided.submit.assert_called_once_with((1, 2, 0), candidate_uid="obj", source="baseline_view_feedback")
    gap = guided.record.call_args.kwargs["pose_gap"]
    assert gap["observation_id"] == "0:20" and gap["scored_position"] == [1, 2, 0]
    assert gap["position_offset_m"] == pytest.approx(np.sqrt(5))
    assert gap["heading_offset_deg"] == pytest.approx(45)


@pytest.mark.parametrize("stop_on_success", [True, False])
@pytest.mark.parametrize("kind", ["frontier", "object", "blocked"])
def test_terminal_event_dispatches_observation_or_verification_once(monkeypatch, tmp_path, kind, stop_on_success):
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
    metadata = {"task_revision": 1, **({"node_index": 0} if kind != "object" else {})}
    status = NavigationStatusCode.BLOCKED if kind == "blocked" else NavigationStatusCode.REACHED
    feedback = NS(tracking_active=False, status=NavigationStatus(status),
                  intent=NS(metadata=metadata, target_object_uid="obj", goal_pose=NS(position=(1, 2, 0))))
    motion = NS(active_request_id="goal")
    def poll():
        motion.active_request_id = None
        return feedback
    motion.poll_active = poll
    def construct(agent, directory, max_calls):
        Path(directory).mkdir(parents=True)
        def verify(scored_position):
            events.append(("verify", scored_position))
            agent.task.progress.completed = True
            return "verified_finish"
        def recover(reason, episode_idx):
            events.append("recover")
            agent.env.episode_over = True
            return "select"
        return NS(directory=directory, tools=NS(motion=motion, objects={"obj": object()}),
                  runtime=NS(model_calls=0), record=lambda *a, **kw: None,
                  verify=verify, select=lambda idx: False, recover=recover, observation_updated=lambda: None,
                  motion_finished=lambda feedback: None)
    monkeypatch.setattr(runtime, "GuidedNavigation", construct)
    if stop_on_success:
        result = runtime.run_guided_episode(agent, episode_idx=0, max_steps=500)
        directory = tmp_path / 'episode-0' / 'harness'
    else:
        directory = tmp_path / 'subtask-1' / 'harness'
        result = runtime.run_guided_episode(agent, episode_idx=0, max_steps=500,
                                            stop_on_success=False, output_dir=directory)
        assert not (tmp_path / 'episode-0' / 'harness').exists()
    import json
    assert json.loads((directory / 'termination.json').read_text()) == result
    if kind == "frontier":
        assert events == ["observe", "advance", "observe", "visited"]
    elif kind == "object":
        assert events == ["observe", "advance", ("verify", (1, 2, 0))] + (["stop"] if stop_on_success else [])
        assert result['reason'] == 'verified_finish' and result['task_complete']
        assert agent.env.episode_over is stop_on_success
    else:
        assert events == ["observe", "advance", "recover"]


@pytest.mark.parametrize('camera_height', [0.88, 1.31])
def test_waypoint_distance_starts_at_actual_body_pose(camera_height):
    import numpy as np
    from navigation.action_controller import current_habitat_position, geodesic_distance_to_waypoint

    body = np.array([3., 2.65, -4.])
    sim = NS(get_agent_state=lambda: NS(position=body), geodesic_distance=Mock(return_value=2.))
    agent = NS(env=NS(sim=sim), mapper=NS(initial_position=np.array([1., -2., 2.5 + camera_height]),
                                       current_position=np.array([2., -2., 0.])))
    assert np.array_equal(current_habitat_position(agent), body)
    assert geodesic_distance_to_waypoint(agent, np.array([5., 6., -camera_height])) == 2.
    start, end = sim.geodesic_distance.call_args.args
    assert np.array_equal(start, body)
    assert np.array_equal(end, [6., 2.65, 4.])
    from planning.exploration_policy import _geodesic_distances_to_nodes
    agent.mapper.env = agent.env
    assert _geodesic_distances_to_nodes(agent.mapper, [NS(position=np.array([5., 6., -camera_height]))]).tolist() == [2.]
    start, end = sim.geodesic_distance.call_args.args
    assert np.array_equal(start, body)
    assert np.array_equal(end, [6., 2.65, 4.])


def test_initial_and_feedback_views_share_planner_arrival(monkeypatch):
    from backends.habitat.motion import HabitatMotionController
    from motion.supervisor import MotionSupervisor
    from motion.contracts import NavigationStatusCode
    import navigation.action_controller as actions
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint", lambda *args, **kwargs: 0.3)
    monkeypatch.setattr(actions, "next_action_to_waypoint", lambda *args, **kwargs: 1)
    env = NS(episode_over=False, step=Mock(return_value={}))
    agent = NS(task=NS(spec=NS(revision=1)), episode_steps=20, update_trajectory=Mock(), on_node_flag=True)
    controller = HabitatMotionController(env, agent)
    motion = MotionSupervisor(controller)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.sequence, guided.record = agent, 0, Mock()
    guided.tools = NS(motion=motion, objects={"obj": NS(rgb=None, bbox=None)})
    guided.candidate_motion_goals = {}

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
    monkeypatch.setattr(actions, "next_action_to_waypoint", lambda *args, **kwargs: next(commands))
    agent = NS(episode_steps=13, on_node_flag=True, update_trajectory=Mock())
    controller = HabitatMotionController(NS(episode_over=False, step=lambda action: {}), agent)
    controller.send_goal(MotionGoal(mode=MotionGoalMode.GO_TO_FRONTIER,
                                    goal_pose=Pose3D((1, 0, 0), frame_id="mapper_local")))
    for _ in range(3):
        controller.advance()
    assert [call.args[0] for call in agent.update_trajectory.call_args_list] == [True, False, False]


@pytest.mark.parametrize("frontiers", [0, 2])
def test_recovery_selects_untried_approach_once_and_returns_to_driver(monkeypatch, frontiers):
    import backends.habitat.guided_runtime as module
    nodes = [NS(idx=i, position=(i+5,0,0)) for i in range(frontiers)]
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent = NS(episode_steps=30, planner=NS(_goal_radius=.25), task=NS(spec=NS(revision=1)), mapper=NS(
        nodes=nodes, prior_map_last_chosen_frontier={"candidates": [
            {"node_idx": i, "raw_distance_m": 5+i} for i in range(frontiers)]}))
    views = [{"uid": purpose, "pose": {"position": pos}, "metadata": {"purpose": purpose}}
             for purpose, pos in [("approach", (2,0,0)), ("recheck", (1,0,0))]]
    guided.tools = NS(refresh_scene=Mock(), query_viewpoints=Mock(return_value=views),
                      objects={"u": NS(rgb=None, bbox=None)})
    guided.candidate_motion_goals = {(1,"u",None): {(1,0,0): "REACHED"}}
    guided.runtime = NS(model_calls=0, max_model_calls=3, choice_guard=nullcontext,
                        choose_recovery=Mock(return_value="approach"), step=Mock())
    guided.record, guided.submit = Mock(), Mock()
    monkeypatch.setattr(module, "select_baseline_exploration_node", lambda *a, **kw:
                        (nodes[0] if nodes else None, "current_room"))
    monkeypatch.setattr(module, "select_prior_region_node", lambda agent: None)
    assert guided.recover({"candidate_uid": "u", "reason": "view_options_exhausted"}, 0) == "moving"
    guided.submit.assert_called_once_with((2,0,0), candidate_uid="u", source="agent_recovery")
    assert guided.runtime.choose_recovery.call_count == 1
    assert guided.search_review_step == guided.agent.episode_steps
    options = guided.runtime.choose_recovery.call_args.args[0]
    assert {o["action_uid"] for o in options} == ({"approach", "frontier:0", "frontier:1"}
                                                if frontiers else {"approach"})
    guided.runtime.step.assert_not_called()


def test_candidate_discovery_does_not_postpone_search_review(monkeypatch):
    import numpy as np
    import backends.habitat.guided_runtime as module
    node = NS(idx=0, position=(5., 0., 0.))
    obj = NS(uid="candidate")
    agent = NS(episode_steps=80, task=NS(spec=NS(revision=1)), scan_complete=True,
               mapper=NS(current_position=np.zeros(3), nodes=[node],
                   prior_map_last_chosen_frontier={"candidates": [
                       {"node_idx": 0, "raw_distance_m": 5.}]}),
               find_final_waypoint=Mock(), whether_to_check_again=Mock())
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record, guided.submit = agent, Mock(), Mock()
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.deferred_candidates, guided.candidate_view_states = set(), {}
    guided.view_attempted = lambda *args: False
    guided.tools = NS(refresh_scene=Mock(), publish_object_viewpoints=lambda uid: [NS(
        pose=NS(position=(1., 0., 0.)), metadata={"recommended": True})])
    guided.runtime = NS(model_calls=0, max_model_calls=5, choice_guard=nullcontext,
                        choose_frontier=Mock(return_value="node:0"))
    monkeypatch.setattr(module, "select_observed_candidate", Mock(side_effect=[
        (True, obj), (True, obj), (False, None)]))
    monkeypatch.setattr(module, "select_exploration_node", lambda *a, **kw: (node, "current_room"))
    assert guided.select(66)
    agent.episode_steps = 99
    assert guided.select(66)
    assert guided.search_review_step == 0
    guided.runtime.choose_frontier.assert_not_called()
    agent.episode_steps = 100
    assert guided.select(66)
    assert guided.search_review_step == 100
    guided.runtime.choose_frontier.assert_called_once()
    guided.submit.assert_called_with(node.position, node=node, source="agent_search_review")


@pytest.mark.parametrize("step,calls,complete,expected", [
    (99, 0, False, "select"), (100, 0, False, "recovery"),
    (100, 5, False, "select"), (100, 0, True, "verified_finish"),
])
def test_unverified_candidate_enters_periodic_review_without_authorizing_stop(
        monkeypatch, step, calls, complete, expected):
    import sys
    import numpy as np
    from instruction_adapter.view_control import ViewControlState
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller", NS(
        rotate_toward_object_for_recheck=lambda agent: True, tilt_camera=lambda agent, pitch: True))
    agent = NS(episode_steps=step, task=NS(spec=NS(revision=1)),
               selected_candidate=NS(role="primary"),
               object_final=NS(uid="candidate", position=np.ones(3)),
               mapper=NS(current_position=np.zeros(3), current_rotation=np.eye(3)[:, ::-1]),
               _observation_identity=lambda uid: {"observation_id": f"66:{step}"},
               check_again=Mock(), last_check_again_evidence={},
               final_stop_decision="need_relation_check", view_control_state=ViewControlState(),
               final_instruction_check=Mock(return_value=complete),
               _continue_better_view_control=lambda: False)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record, guided.defer_candidate = agent, Mock(), Mock()
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.runtime = NS(model_calls=calls, max_model_calls=5)
    assert guided.verify((0., 0., 0.)) == expected
    assert guided.runtime.model_calls == calls and guided.search_review_step == 0
    if expected == "recovery":
        assert guided.recovery_context == {
            "reason": "prolonged_unverified_candidate", "candidate_uid": "candidate",
            "decision": "need_relation_check", "steps_since_review": 100}
        guided.defer_candidate.assert_not_called()
    elif expected == "select":
        guided.defer_candidate.assert_called_once_with("need_relation_check")


def test_prior_region_adds_its_branch_without_pruning_baseline_options(monkeypatch):
    """A one-node prior region must not reduce recovery to returning to that node."""
    import backends.habitat.guided_runtime as module
    nodes = [NS(idx=i, position=(i, 0, 0)) for i in range(3)]
    mapper = NS(nodes=nodes, prior_map_last_chosen_frontier=None)
    def ranking(*indices):
        mapper.prior_map_last_chosen_frontier = {"candidates": [
            {"node_idx": i, "raw_distance_m": float(i + 1)} for i in indices]}
    def baseline(agent, **kw):
        ranking(1, 2)
        return nodes[1], "current_room"
    def prior(agent):
        ranking(0)
        return nodes[0]
    monkeypatch.setattr(module, "select_baseline_exploration_node", baseline)
    monkeypatch.setattr(module, "select_prior_region_node", prior)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent = NS(episode_steps=36, mapper=mapper, task=NS(spec=NS(revision=1)))
    guided.tools = NS(refresh_scene=Mock())
    guided.runtime = NS(model_calls=0, max_model_calls=3, choice_guard=nullcontext,
                        choose_recovery=Mock(return_value="frontier:1"))
    guided.record, guided.submit = Mock(), Mock()
    assert guided.recover({"reason": "blocked"}, 0) == "moving"
    options = guided.runtime.choose_recovery.call_args.args[0]
    assert [o["action_uid"] for o in options] == ["frontier:1", "frontier:2", "frontier:0"]
    assert [o.get("prior_region_recommended", False) for o in options] == [False, False, True]
    guided.submit.assert_called_once_with(nodes[1].position, node=nodes[1], source="agent_recovery")


@pytest.mark.parametrize("frontiers", [False, True])
def test_completed_anchor_review_excludes_anchor_views_and_reselects_when_empty(frontiers):
    node = NS(idx=2, position=(2., 0., 0.))
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent = NS(episode_steps=104, task=NS(spec=NS(revision=1)), mapper=NS(nodes=[None, None, node]))
    guided.tools = NS(refresh_scene=Mock(), query_viewpoints=Mock())
    guided.runtime = NS(model_calls=0, max_model_calls=5, choice_guard=nullcontext,
                        choose_recovery=Mock(return_value="frontier:2"))
    guided.record, guided.submit = Mock(), Mock()
    rows = [{"node_idx": 2, "raw_distance_m": 2.}] if frontiers else []
    guided.exploration_options = lambda index: (None, "baseline", rows)
    reason = {"reason": "prolonged_target_search", "anchor_uid": "anchor",
              "concept_id": "reference", "anchor_status": "searched_no_terminal_found"}
    assert guided.recover(reason, 0) == ("moving" if frontiers else "select")
    guided.tools.query_viewpoints.assert_not_called()
    if frontiers:
        options = guided.runtime.choose_recovery.call_args.args[0]
        assert [row["action_uid"] for row in options] == ["frontier:2"]
        guided.submit.assert_called_once_with(node.position, node=node, source="agent_recovery")
    else:
        guided.runtime.choose_recovery.assert_not_called()
        guided.submit.assert_not_called()


def test_feedback_retains_initial_motion_attempt_in_view_history(monkeypatch):
    import sys
    import numpy as np
    from instruction_adapter.view_control import ViewControlState
    tilts = []
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller",
                        NS(rotate_toward_object_for_recheck=lambda agent: True,
                           tilt_camera=lambda agent, pitch: tilts.append(pitch) or True))
    state = ViewControlState()
    state.start("u", {}, {})
    agent = NS(episode_steps=20, task=NS(spec=NS(revision=1)), object_final=NS(uid="u", position=np.ones(3)),
               selected_candidate=NS(role="primary"),
               mapper=NS(current_position=np.zeros(3), current_rotation=np.eye(3)[:, ::-1]),
               _observation_identity=lambda uid: {"observation_id": "0:20"},
               check_again=Mock(), last_check_again_evidence={}, final_stop_decision="need_better_view",
               view_control_state=state, _continue_better_view_control=lambda: True,
               final_instruction_check=Mock(return_value=False))
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record, guided.submit, guided.defer_candidate = agent, Mock(), Mock(), Mock()
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.candidate_motion_goals = {(1,"u",None): {(1,0,0): "REACHED"}}
    agent.planner = NS(_goal_radius=.25)
    guided.tools = NS(objects={"u": NS(rgb=None,bbox=None)}, publish_object_viewpoints=lambda uid: [NS(
        pose=NS(position=(1,0,0)), metadata={"recommended": True})])
    assert guided.verify((1, 0, 0)) == "recovery"
    guided.submit.assert_not_called()
    guided.defer_candidate.assert_called_once_with("view_options_exhausted")


def test_motion_terminal_history_is_scoped_to_evidence_and_arrival_tolerance():
    import numpy as np
    obj = NS(rgb=np.zeros((2,2,3), dtype=np.uint8), bbox=[0,0,2,2])
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent = NS(task=NS(spec=NS(revision=1)), planner=NS(_goal_radius=.25))
    guided.tools = NS(objects={"u": obj})
    guided.candidate_motion_goals = {}
    fingerprint = guided.candidate_evidence(obj)
    feedback = NS(intent=NS(target_object_uid="u", goal_pose=NS(position=(1,0,0)),
                           metadata={"task_revision": 1, "candidate_evidence_fingerprint": fingerprint}),
                  status=NS(status=NS(value="BLOCKED")))
    assert not guided.view_attempted("u", (1,0,0))
    guided.motion_finished(feedback)
    assert guided.view_attempted("u", (1.01,0,0))
    assert guided.view_attempted("u", (1,0,.88))
    assert not guided.view_attempted("u", (1.5,0,0))
    obj.rgb[0,0] = 255
    assert not guided.view_attempted("u", (1,0,0))
    assert guided.candidate_motion_goals[(1,"u",fingerprint)][(1,0,0)] == "BLOCKED"


def test_selected_region_drives_existing_navigation(monkeypatch):
    """A selected region uses its bound nodes across online room fragments."""
    import planning.exploration_policy as policy
    node = NS(idx=7, has_frontier=True, state=0)
    region = NS(uid='prior:kitchen', metadata={'node_ids': (7,)})
    choose = Mock(return_value=node)
    monkeypatch.setattr(policy, 'find_closest_nodes', choose)
    mapper = NS(prior_map_runtime=NS(region_candidates=(region,)),
                prior_map_high_level_selection={'decision': 'select', 'selected_uid': region.uid},
                nodes=[node, NS(idx=8, has_frontier=True, state=0)])
    agent = NS(mapper=mapper, relocate=False, gpt_relocate=False)
    actual, source = select_exploration_node(agent, episode_idx=0, step=10)
    assert actual is node and source == 'prior_region_policy'
    choose.assert_called_once_with(mapper, [node])


@pytest.mark.parametrize("entry", ["search_review", "recovery_frontier", "recovery_viewpoint"])
@pytest.mark.parametrize("outcome", ["installed", "unchanged", "invalid", "guard_refused"])
def test_interpretation_handoff_reselects_without_stale_motion(monkeypatch, entry, outcome):
    """A semantic repair is a decision turn, not an old-revision motion attempt."""
    from dataclasses import replace
    import numpy as np
    import backends.habitat.guided_runtime as module
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from instruction_adapter.view_control import ViewControlState
    from motion.supervisor import MotionSupervisor
    from scene.contracts import Pose3D, SemanticMapSnapshot, ViewpointSnapshot
    from task.session import TaskSession

    spec = TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),))
    session = TaskSession(SemanticMapSnapshot(12., Pose3D((0., 0., 0.))), runtime_plan(spec))
    revised = runtime_plan(replace(spec, revision=2, targets=(TargetSpec("goal", "sofa"),)))
    node = NS(idx=0, position=(5., 0., 0.), state=0)
    obj = NS(uid="object", rgb=None, bbox=None)
    view = ViewpointSnapshot("view", Pose3D((1., 0., 0.)), visible_objects=(obj.uid,),
                             metadata={"recommended": True})
    mapper = NS(nodes=[node], current_position=np.zeros(3), change_state=Mock(),
                prior_map_last_chosen_frontier={"candidates": [{"node_idx": 0, "raw_distance_m": 5.}]})
    agent = NS(task=session, mapper=mapper, episode_steps=125, travel_distance=8., scan_complete=True,
               planner=NS(_goal_radius=.25), find_final_waypoint=Mock(), whether_to_check_again=Mock())
    controller = NS(send_goal=Mock(return_value="goal"), moves=7, advance=Mock())
    motion = MotionSupervisor(controller)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.sequence, guided.record = agent, 4, Mock()
    guided.tools = NS(motion=motion, objects={obj.uid: obj}, refresh_scene=Mock(),
                      publish_object_viewpoints=Mock(return_value=(view,)),
                      query_viewpoints=Mock(return_value=[{
                          "uid": "approach", "pose": {"position": (2., 0., 0.)}}]))
    guided.deferred_candidates, guided.deferred_evidence = set(), {(1, obj.uid): {None}}
    old_state = ViewControlState()
    old_state.start(obj.uid, {}, {"observation_identity": {"task_revision": 1}})
    old_state.exhausted = True
    guided.candidate_view_states = {(1, obj.uid): old_state}
    guided.candidate_motion_goals = {(1, obj.uid, None): {view.pose.position: "REACHED"}}
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.runtime = NS(model_calls=2, max_model_calls=5)
    queried = []

    def query(current, **kwargs):
        queried.append((session.spec.revision, session.progress.active_target().name))
        return (True, obj) if session.spec.revision == 2 else (False, None)

    no_choice = outcome == "invalid"
    selected_uid = {"search_review": "node:0", "recovery_frontier": "frontier:0",
                    "recovery_viewpoint": "approach"}[entry]

    def choose(options, *args):
        guided.runtime.model_calls += 1
        assert motion.active_request_id is None
        if outcome == "installed":
            session.install_plan(revised)
            return None
        if no_choice:
            return None
        key = "frontier_uid" if entry == "search_review" else "action_uid"
        assert selected_uid in {option[key] for option in options}
        return selected_uid

    guard_events = []

    @contextmanager
    def choice_guard():
        guard_events.append("enter")
        try:
            if outcome == "guard_refused":
                raise ValueError("choice refused")
            yield
        finally:
            guard_events.append("exit")

    original_submit = guided.submit

    def submit(*args, **kwargs):
        if kwargs["source"] in {"agent_search_review", "agent_recovery"}:
            assert guard_events == ["enter"]
        return original_submit(*args, **kwargs)

    guided.submit = Mock(side_effect=submit)
    guided.runtime.choice_guard = Mock(side_effect=choice_guard)
    guided.runtime.choose_frontier = guided.runtime.choose_recovery = Mock(side_effect=choose)
    monkeypatch.setattr(module, "select_observed_candidate", query)
    monkeypatch.setattr(module, "select_exploration_node", lambda *a, **kw: (node, "current_room"))
    guided.exploration_options = lambda idx: (node, "current_room", mapper.prior_map_last_chosen_frontier["candidates"])
    observe = Mock(side_effect=AssertionError("Decision handoff must not move or observe"))
    monkeypatch.setattr(module, "run_observation_mapping_cycle", observe)

    def decide():
        if entry == "search_review":
            return guided.select(0)
        reason = {"reason": "blocked"}
        if entry == "recovery_viewpoint":
            reason["candidate_uid"] = obj.uid
        return guided.recover(reason, 0)

    if outcome == "guard_refused":
        with pytest.raises(ValueError, match="choice refused"):
            decide()
    else:
        result = decide()
        assert result is True if entry == "search_review" else result == (
            "select" if outcome == "installed" or no_choice else "moving")
    if outcome == "installed" or no_choice:
        assert guided.sequence == 4 and motion.active_request_id is None
        guided.submit.assert_not_called()
        controller.send_goal.assert_not_called()
        guided.record.assert_called_with("interpretation_changed" if outcome == "installed" else "decision_reselect",
            decision_source="agent_search_review" if entry == "search_review" else "agent_recovery",
            task_revision=2 if outcome == "installed" else 1)
    if outcome == "installed":
        assert session.spec.revision == 2
        assert guided.select(0)
        assert queried[-1] == (2, "sofa")
        assert motion.active_intent.target_object_uid == obj.uid
        assert motion.active_intent.metadata["task_revision"] == 2
        assert agent.view_control_state is not old_state
        assert not agent.view_control_state.active
    else:
        assert session.spec == spec
        assert all(call.args[0] != "interpretation_changed" for call in guided.record.call_args_list)
        if outcome == "unchanged":
            assert motion.active_intent.metadata["task_revision"] == 1
            if entry == "recovery_viewpoint":
                assert motion.active_intent.target_object_uid == obj.uid
                assert motion.active_intent.goal_pose.position == (2., 0., 0.)
            else:
                assert motion.active_intent.metadata["node_index"] == node.idx
        else:
            assert motion.active_request_id is None
            guided.submit.assert_not_called()
            controller.send_goal.assert_not_called()
    submitted = int(outcome in {"installed", "unchanged"})
    assert guided.sequence == 4 + submitted and controller.send_goal.call_count == submitted
    if outcome in {"unchanged", "guard_refused"}:
        guided.runtime.choice_guard.assert_called_once_with()
        assert guard_events == ["enter", "exit"]
    else:
        guided.runtime.choice_guard.assert_not_called()
        assert guard_events == []
    assert agent.episode_steps == 125 and agent.travel_distance == 8. and controller.moves == 7
    assert guided.runtime.model_calls == 3 and guided.runtime.max_model_calls == 5
    guided.runtime.choose_frontier.assert_called_once()
    assert guided.candidate_view_states[(1, obj.uid)] is old_state and old_state.exhausted
    assert guided.candidate_motion_goals == {(1, obj.uid, None): {view.pose.position: "REACHED"}}
    assert guided.deferred_evidence == {(1, obj.uid): {None}}
    mapper.change_state.assert_not_called()
    controller.advance.assert_not_called()
    observe.assert_not_called()
