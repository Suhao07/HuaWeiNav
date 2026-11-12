"""固定候选、路径成本和视点的决策回放及执行生命周期回归，不测量建图或导航成功率。"""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from instruction_adapter.view_control import ViewControlState
from motion.contracts import MotionGoalMode, NavigationIntent, NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from navigation.guided import GuidedPolicy, GuidedVerification, run_guided_loop
from scene.contracts import Pose3D, ViewpointSnapshot


class ReplayPolicy(GuidedPolicy):
    """Supply deterministic scene hooks without a simulator or backend runtime."""

    def __init__(self, agent, tools, runtime):
        self.agent, self.tools, self.runtime = agent, tools, runtime
        self.initialize_state(deferred_candidates=set(), review_interval=100)

    @property
    def step(self):
        return self.agent.episode_steps

    @property
    def revision(self):
        return self.agent.task.spec.revision

    @property
    def arrival_tolerance(self):
        return self.agent.planner._goal_radius

    @property
    def candidate_uid(self):
        return self.agent.object_final.uid

    @property
    def view_state(self):
        return self.agent.view_control_state

    @property
    def scan_complete(self):
        return self.agent.scan_complete

    def evidence_fingerprint(self, uid):
        obj = self.tools.objects.get(uid)
        return self.candidate_evidence(obj) if obj is not None else None

    def refresh_scene(self):
        self.tools.refresh_scene()

    def select_candidate(self, episode_idx):
        self.refresh_scene()
        uid = next((uid for uid in self.tools.objects if uid not in self.deferred_candidates), None)
        if uid is not None:
            self.agent.object_final = self.tools.objects[uid]
        return uid

    def initial_views(self, uid):
        self.agent.view_control_state = self.candidate_view_states.setdefault((self.revision, uid), ViewControlState())
        return self.feedback_views(uid)

    def feedback_views(self, uid):
        return self.tools.publish_object_viewpoints(uid)

    def observe_pending(self, episode_idx):
        self.agent.observe()

    def select_exploration(self, episode_idx):
        return self.agent.mapper.nodes[0], "current_room"

    def node_for_index(self, index):
        return self.agent.mapper.nodes[index]

    def exploration_options(self, episode_idx):
        node, source = self.select_exploration(episode_idx)
        return node, source, self.agent.mapper.prior_map_last_chosen_frontier["candidates"]

    def review_options(self, node, source, episode_idx):
        return self.exploration_options(episode_idx)

    def capture_verification(self, scored_position):
        candidate = self.agent.selected_candidate
        decision = "anchor_searched" if candidate.role == "anchor" else self.agent.final_stop_decision
        return GuidedVerification(self.candidate_uid, candidate.role, candidate.target_id,
                                  decision, self.agent.final_instruction_check())

    def submit(self, position, *, candidate_uid=None, node=None, source):
        """Record reference-policy goals through the same real motion supervisor."""
        self.sequence += 1
        metadata = {"task_revision": self.revision, "decision_source": source}
        if node is not None:
            metadata["node_index"] = node.idx
        else:
            metadata["candidate_evidence_fingerprint"] = self.evidence_fingerprint(candidate_uid)
            metadata["candidate_action_epoch"] = self.action_epoch(candidate_uid)
        intent = NavigationIntent(MotionGoalMode.GO_TO_FRONTIER if node is not None else MotionGoalMode.IMPROVE_VIEW,
                                  goal_pose=Pose3D(tuple(position)), target_object_uid=candidate_uid,
                                  metadata=metadata)
        self.tools.motion.submit(intent, intent.to_motion_goal(), request_id=f"replay:{self.sequence}",
                                 control_epoch=self.tools.motion.control_epoch)
        self.record("submitted", decision_source=source, node_index=metadata.get("node_index"),
                    candidate_uid=candidate_uid, position=tuple(position))

    def restore_camera(self):
        return True

    def continue_view_control(self):
        return self.agent._continue_better_view_control()


def replay_policy(backend, monkeypatch, tmp_path):
    """Freeze sensor/model inputs, retaining adapter hooks and shared decisions."""
    import sys
    import backends.habitat.guided_runtime as habitat
    import backends.habitat.query as habitat_query
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.tools import NavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from scene.contracts import ObjectNodeSnapshot, SemanticMapSnapshot
    from task.session import TaskSession
    from verification.contracts import VerificationResult
    from test_isaac_runtime import FakeClient, FakeExplorationMap, _tools

    obj = NS(uid="u", rgb=np.zeros((2, 2, 3), dtype=np.uint8), bbox=[0, 0, 2, 2],
             position=np.array([2., 2., 1.]))
    nodes = [NS(idx=i, room_idx=0, position=(5. + i, 0., 0.), state=0) for i in range(2)]
    rows = [{"node_idx": n.idx, "raw_distance_m": n.position[0]} for n in nodes]
    # Model-bound matching facts, not a replacement for query_candidates/ranking.
    match = NS(object_uid="u", candidate_eligible=True, confidence=1., as_dict=lambda: {})
    session = TaskSession(SemanticMapSnapshot(0., Pose3D((0., 0., 0.))),
        runtime_plan(TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),))),
        concept_matcher=NS(match_many=Mock(return_value=[match]), records={}))
    agent = NS(episode_steps=0, task=session, planner=NS(_goal_radius=0.25),
               object_final=obj, view_control_state=ViewControlState(), scan_complete=True,
               current_node_idx=0, selected_candidate=NS(role="primary", target_id="goal"),
               save_dir=str(tmp_path), relocate=False, gpt_relocate=False,
               mapper=NS(nodes=nodes, current_position=np.zeros(3), current_node_idx=0,
                         room_nodes=[NS(state=0)], explore_in_room=Mock(return_value=nodes[0]),
                         prior_map_policy_adapter=None, search_prior_result=None,
                         current_rotation=np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                         prior_map_last_chosen_frontier={"candidates": rows}),
               find_final_waypoint=Mock(), whether_to_check_again=Mock(), observe=Mock(),
               check_again=Mock(), last_check_again_evidence={}, final_stop_decision="uncertain",
               _observation_identity=lambda uid: {"observation_id": "replay"},
               final_instruction_check=Mock(return_value=False), _continue_better_view_control=Mock(return_value=False))
    views = [ViewpointSnapshot(name, Pose3D(position), visible_objects=("u",),
                              metadata={"recommended": index == 0})
             for index, (name, position) in enumerate([("initial", (1., 0., 0.)), ("alternate", (2., 0., 0.))])]
    runtime = NS(model_calls=0, max_model_calls=5, choice_guard=nullcontext,
                 choose_frontier=Mock(return_value="node:1"), choose_recovery=Mock(return_value="alternate"))
    if backend == "isaac":
        client = FakeClient()
        client.pose = {"position": [0., 0., 0.], "yaw": 0.}
        class ReplayExplorationMap(FakeExplorationMap):
            """Publish identical frontier costs for ordinary selection and review."""

            @property
            def candidates(self):
                return rows

        exploration = ReplayExplorationMap()
        exploration.nodes = {n.idx: n for n in nodes}
        exploration.select = Mock(return_value=(nodes[0], "current_room", rows))
        tools = _tools(session, client, tmp_path, exploration_map=exploration)
    else:
        controller = NS(send_goal=Mock(side_effect=[f"goal:{i}" for i in range(10)]),
                        poll_status=Mock(return_value=NavigationStatus(NavigationStatusCode.QUEUED)),
                        cancel=Mock())
        tools = NavigationTools(session, MotionSupervisor(controller), None, None)
    tools.objects = {obj.uid: obj}
    tools.publish_object_viewpoints = Mock(return_value=views)

    def refresh():
        # Image transport and mapped observations are controlled inputs. Isaac's
        # evidence_fingerprint still opens RGB and invokes the real crop hash.
        session.snapshot = replace(session.snapshot, timestamp=float(agent.episode_steps),
            robot_pose=Pose3D(tuple(agent.mapper.current_position)),
            objects=tuple(ObjectNodeSnapshot(uid, "chair", tuple(getattr(item, "position", (8., 0., 0.))),
                bbox2d_xyxy=tuple(item.bbox), image_ref="replay-rgb" if item.rgb is not None else None)
                for uid, item in tools.objects.items()),
            viewpoints=tuple(tools.publish_object_viewpoints.return_value))

    tools.refresh_scene = Mock(side_effect=refresh)
    # Freeze published views at the mapping boundary. Do not substitute the
    # policy's selection/attempt filtering or IsaacGuidedNavigation.feedback_views.
    tools.query_viewpoints = Mock(side_effect=lambda uid, **kwargs: NavigationTools.query_viewpoints(tools, uid))
    refresh()
    if backend == "core":
        guided = ReplayPolicy(agent, tools, runtime)
    elif backend == "habitat":
        guided = habitat.GuidedNavigation.__new__(habitat.GuidedNavigation)
        guided.agent, guided.tools, guided.runtime = agent, tools, runtime
        guided.initialize_state(deferred_candidates=set(), review_interval=100)
        # Replace observation/prior IO only; actual candidate and room-first
        # selectors run through the adapter rather than being monkeypatched.
        monkeypatch.setattr(habitat_query, "refresh_scene", lambda a: (refresh(), tools.objects)[1])
        monkeypatch.setattr(habitat_query, "refresh_mapper_prior_map_query", lambda *a, **kw: None)
        monkeypatch.setattr(habitat, "run_observation_mapping_cycle", lambda *a, **kw: agent.observe())
        monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller", NS(
            rotate_toward_object_for_recheck=lambda a: True, tilt_camera=lambda a, pitch: True))
    else:
        exploration.refresh = Mock(side_effect=lambda *args: refresh())
        guided = IsaacGuidedNavigation(tools, runtime, tmp_path, search_review_moves=100)
        guided.scan_complete = True
        guided.selected = {"candidate_uid": "u", "target_id": "goal", "role": "primary"}
        image_open = Image.open
        monkeypatch.setattr(Image, "open", lambda path, *a, **kw:
                            Image.fromarray(obj.rgb) if path == "replay-rgb" else image_open(path, *a, **kw))
        evidence = {"observation_identity": {"observation_id": "replay"}, "geometry": {}}
        tools.last_evidence = evidence
        tools.evidence_provider = Mock(return_value=evidence)
        tools.verifier = NS(verify=Mock(return_value=VerificationResult(
            True, "accept", semantic_satisfied=True, task_revision=1, target_uid="u")))
    guided.candidate_view_states[(1, "u")] = agent.view_control_state
    guided.record = Mock()  # Journal IO only; capture every semantic event.
    guided.submit = Mock(wraps=guided.submit)
    guided.captured_verifications = []
    capture = guided.capture_verification

    def capture_once(position):
        result = capture(position)
        guided.captured_verifications.append(result)
        return result

    guided.capture_verification = Mock(side_effect=capture_once)
    tools.motion.submit = Mock(wraps=tools.motion.submit)
    return guided, agent, obj


def finished(guided, code=NavigationStatusCode.BLOCKED, position=(1., 0., 0.)):
    """Consume actual supervisor feedback; Isaac advances its real controller."""
    from backends.isaac.runtime import IsaacGuidedNavigation

    motion = guided.tools.motion
    assert motion.active_intent.goal_pose.position == position
    if isinstance(guided, IsaacGuidedNavigation):
        guided.tools.client.status = code.value
        motion.controller.advance(motion.active_goal_id)
    else:
        motion.controller.poll_status.return_value = NavigationStatus(code)
        guided.agent.episode_steps += 1
    feedback = motion.poll_active()
    assert feedback.status.status is code and not feedback.tracking_active
    assert motion.active_request_id is None
    guided.motion_finished(feedback)


def replay_step(guided, agent, step):
    """Align decision clocks, not claims of equal physical movement distances."""
    from backends.isaac.runtime import IsaacGuidedNavigation

    agent.episode_steps = step
    if isinstance(guided, IsaacGuidedNavigation):
        guided.tools.motion.controller.moves = step


def normalize(guided):
    """Drop only query/pose diagnostics; retain order, decisions, hashes and versions."""
    events = [(call.args[0], {key: value for key, value in call.kwargs.items() if key != "pose_gap"})
              for call in guided.record.call_args_list if call.args[0] != "query_scene"]
    goals = []
    for call in guided.tools.motion.submit.call_args_list:
        intent = call.args[0]
        goals.append((intent.metadata["decision_source"], intent.target_object_uid,
                      intent.metadata.get("node_index"), intent.goal_pose.position, intent.mode.value,
                      intent.metadata["task_revision"], call.kwargs["control_epoch"]))
    choices = {name: [(call.args, call.kwargs) for call in getattr(guided.runtime, name).call_args_list]
               for name in ("choose_frontier", "choose_recovery")}
    state = {"revision": guided.revision, "epoch": guided.tools.motion.control_epoch,
             "deferred": set(guided.deferred_candidates), "deferred_evidence": guided.deferred_evidence,
             "attempts": guided.candidate_motion_goals, "view_scopes": set(guided.candidate_view_states),
             "recovery": guided.recovery_context, "review_step": guided.search_review_step,
             "sequence": guided.sequence, "motion_phase": guided.tools.motion.phase}
    return events, goals, choices, state


@pytest.fixture
def offline_replay(monkeypatch):
    """Reject accidental provider, network or GPU construction in the replay."""
    from test_isaac_runtime import offline_only

    # Reuse the existing boundary guard without importing its autouse fixture
    # into this module's unrelated shared-loop lifecycle checks.
    offline_only.__wrapped__(monkeypatch)


@pytest.mark.parametrize("backend", ["core", "habitat", "isaac"])
@pytest.mark.parametrize("second_has_view", [True, False])
def test_candidate_exhaustion_checks_remaining_candidates_before_frontiers(
        monkeypatch, tmp_path, offline_replay, backend, second_has_view):
    guided, agent, obj = replay_policy(backend, monkeypatch, tmp_path)
    other = NS(uid="v", rgb=obj.rgb.copy(), bbox=list(obj.bbox), position=np.array([4., 0., 1.]))
    guided.tools.objects[other.uid] = other
    agent.task.concept_matcher.match_many.return_value.append(
        NS(object_uid="v", candidate_eligible=True, confidence=1., as_dict=lambda: {}))
    view = ViewpointSnapshot("second-view", Pose3D((3., 0., 0.)), visible_objects=("v",),
                            metadata={"recommended": True})
    views = [view] if second_has_view else []
    guided.tools.publish_object_viewpoints.return_value = views
    guided.tools.publish_object_viewpoints.side_effect = lambda uid: views if uid == "v" else []
    guided.tools.refresh_scene()
    guided.select_exploration = Mock(return_value=(None, "exhausted"))

    assert guided.select(0) is second_has_view
    if second_has_view:
        guided.submit.assert_called_once_with((3., 0., 0.), candidate_uid="v", source="baseline_viewpoint")
        guided.select_exploration.assert_not_called()
        assert guided.deferred_candidates == {"u"}
        assert guided.tools.motion.active_intent.target_object_uid == "v"
    else:
        guided.submit.assert_not_called()
        guided.select_exploration.assert_called_once_with(0)
        assert guided.deferred_candidates == {"u", "v"}
        assert guided.record.call_args.args == ("search_exhausted",)
    assert not agent.task.progress.completed


@pytest.mark.parametrize("scenario", ["blocked", "no_views", "anchor", "stop", "revision", "evidence"])
def test_normalized_shared_and_habitat_decision_replay(monkeypatch, tmp_path, offline_replay, scenario):
    traces = {}
    for backend in ("core", "habitat", "isaac"):
        with monkeypatch.context() as patch:
            guided, agent, obj = replay_policy(backend, patch, tmp_path)
            if scenario == "blocked":
                assert guided.select(0)
                finished(guided)
                assert guided.recover({"candidate_uid": "u", "goal": {"node_index": 0}}, 0) == "moving"
                options = guided.runtime.choose_recovery.call_args.args[0]
                assert [row["action_uid"] for row in options] == ["frontier:1", "alternate"]
                guided.submit.assert_called_with((2., 0., 0.), candidate_uid="u", source="agent_recovery")
            elif scenario == "no_views":
                guided.tools.publish_object_viewpoints.return_value = []
                guided.tools.refresh_scene()
                assert guided.select(0)
                assert guided.deferred_candidates == {"u"}
                assert normalize(guided)[0][0][1]["reason"] == "available_views_already_attempted"
                guided.submit.assert_called_once_with(agent.mapper.nodes[0].position, node=agent.mapper.nodes[0],
                                                      source="current_room")
            elif scenario == "anchor":
                replay_step(guided, agent, 100)
                agent.selected_candidate.role = "anchor"
                if backend == "isaac":
                    guided.selected["role"] = "anchor"
                    agent.task.concept_matcher.match_many.return_value = []
                    agent.task.snapshot = replace(agent.task.snapshot,
                        metadata={"covered_headings": list(range(12))})
                else:
                    agent.task.anchor_search_ledger.mark(raw_instruction=agent.task.spec.raw_instruction,
                        concept_id="goal", anchor_uid="u", status="searched_no_terminal_found")
                # An anchor can never authorize task STOP, even with a malformed complete flag.
                agent.final_instruction_check.return_value = True
                assert guided.verify((1., 0., 0.)) == "recovery"
                # Assert the real capture hook's decision too: the shared anchor
                # branch intentionally ignores this field and cannot detect drift.
                guided.capture_verification.assert_called_once_with((1., 0., 0.))
                captured, = guided.captured_verifications
                assert (captured.role, captured.decision) == ("anchor", "anchor_searched")
                assert captured.complete is (backend != "isaac")  # Malformed input never authorizes STOP.
                guided.runtime.choose_recovery.return_value = "frontier:1"
                assert guided.recover(guided.recovery_context, 0) == "moving"
                guided.tools.query_viewpoints.assert_not_called()
                assert guided.record.call_args_list[0].args == ("anchor_searched",)
                if backend == "isaac":
                    guided.tools.verifier.verify.assert_not_called()
                    guided.tools.evidence_provider.assert_not_called()
                    assert agent.task.anchor_search_ledger.is_blocked(agent.task.spec.raw_instruction, "goal", "u")
                    assert not agent.task.progress.completed
            elif scenario == "stop":
                agent.final_instruction_check.return_value = True
                agent.final_stop_decision = "accept"
                assert guided.verify((1., 0., 0.)) == "verified_finish"
                guided.submit.assert_not_called()
                assert guided.record.call_args.kwargs["task_complete"]
                assert guided.record.call_args.kwargs["decision"] == "accept"
                if backend == "isaac":
                    guided.tools.verifier.verify.assert_called_once()
                    guided.tools.evidence_provider.assert_called_once_with("u")
                    assert agent.task.progress.target_progress["goal"].accepted_candidate_uids == ["u"]
                    assert guided.tools.propose_finish()["accepted"]
            elif scenario == "revision":
                assert guided.select(0)
                old_state = guided.view_state
                finished(guided)
                guided.defer_candidate("uncertain")
                old_fingerprint = guided.evidence_fingerprint("u")
                replay_step(guided, agent, 100)
                # A real idle cancellation advances the control epoch. Neither
                # interpretation handoff nor the next proposal may reset it.
                guided.tools.motion.cancel()
                assert guided.tools.motion.control_epoch == 1

                def revise(*args):
                    from instruction.contracts import TargetSpec
                    from instruction_adapter.task_compat import runtime_plan

                    session = agent.task
                    assert guided.tools.motion.active_request_id is None
                    revised = replace(session.spec, revision=2, targets=(TargetSpec("goal", "sofa"),))
                    assert guided.tools.install_interpretation(runtime_plan(revised), task_revision=1,
                        search_revision=session.search_revision, snapshot=session.snapshot,
                        record={"control_epoch": guided.tools.motion.control_epoch, "reason": "replay"})
                    if backend != "isaac":
                        guided.deferred_candidates.clear()  # Controlled Habitat/core task-install boundary.
                    return None

                guided.runtime.choose_frontier.side_effect = revise
                assert guided.select(0)
                assert guided.record.call_args.args == ("interpretation_changed",)
                assert guided.submit.call_count == 1
                assert guided.deferred_candidates == set()
                assert guided.tools.motion.active_request_id is None
                assert guided.tools.motion.control_epoch == 1
                assert agent.task.spec.revision == 2 and agent.task.spec.targets[0].name == "sofa"
                assert len(agent.task.interpretation_history) == 1
                assert guided.select(0)
                assert guided.view_state is not old_state
                assert guided.candidate_view_states[(1, "u")] is old_state
                assert guided.candidate_motion_goals[(1, "u", 0)][(1., 0., 0.)] == "blocked"
                assert not guided.view_attempted("u", (1., 0., 0.))
                assert guided.tools.motion.active_intent.metadata["task_revision"] == 2
                assert guided.tools.motion.submit.call_args.kwargs["control_epoch"] == 1
            else:
                assert guided.select(0)
                # Submission alone is not a completed view attempt.
                assert not guided.view_attempted("u", (1., 0., 0.))
                finished(guided)
                fingerprint = guided.evidence_fingerprint("u")
                guided.defer_candidate("blocked")
                replay_step(guided, agent, guided.step + 50)
                agent.mapper.current_position += 1
                guided.tools.objects["other"] = NS(rgb=np.ones((2, 2, 3)), bbox=[0, 0, 2, 2])
                guided.observation_updated()
                assert guided.deferred_candidates == {"u"}
                rgb = obj.rgb
                obj.rgb = None
                guided.observation_updated()
                assert guided.deferred_candidates == {"u"}
                obj.rgb = rgb
                guided.observation_updated()
                assert guided.deferred_candidates == {"u"}
                obj.rgb[0, 0] = 255
                guided.observation_updated()
                assert guided.deferred_candidates == {"u"}
                assert guided.view_attempted("u", (1., 0., 0.))
                assert guided.candidate_motion_goals[(1, "u", 0)][(1., 0., 0.)] == "blocked"
                del guided.tools.objects["other"]
                assert guided.select(0)
            assert guided.tools.motion.submit.call_count == guided.submit.call_count
            traces[backend] = normalize(guided)
    assert traces["core"] == traces["habitat"]
    assert traces["core"] == traces["isaac"]


def _relation_feedback(guided, agent, status, anchor_uid=""):
    from instruction_adapter.contracts import TargetQuery

    agent.task.plan.targets.append(TargetQuery("reference", "table", terminal=False, role="anchor"))
    outcomes = [{"predicate_id": "c0", "candidate_uid": "u", "subject_id": "goal",
                 "object_id": "reference", "anchor_concept_id": "reference", "anchor_uid": anchor_uid,
                 "relation": "near", "status": status, "evidence_view_ids": [], "reason": status}]
    guided.relation_outcomes[(guided.revision, "u")] = outcomes
    return outcomes


def _observed_view(session, position=(0., 0., 0.), rotation=None, objects=("u",)):
    session.snapshot = replace(session.snapshot, metadata={"observation_images": [{
        "id": "fresh-frame", "position": list(position),
        "rotation": (np.eye(3) if rotation is None else rotation).tolist(),
        "observed_object_uids": list(objects), "object_boxes": {uid: [0, 0, 2, 2] for uid in objects}}]})


@pytest.mark.parametrize("backend", ["core", "habitat", "isaac"])
@pytest.mark.parametrize("novelty", ["translation", "yaw", "pitch", "pixel", "bbox", "frame", "pose_noise"])
def test_retry_requires_new_observed_camera_coverage(monkeypatch, tmp_path, offline_replay, backend, novelty):
    from navigation_core.view_geometry import pitch_rotation

    guided, agent, obj = replay_policy(backend, monkeypatch, tmp_path)
    assert guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    _observed_view(agent.task)
    guided.defer_candidate("need_better_view")
    if novelty == "translation":
        _observed_view(agent.task, position=(1., 0., 0.))
    elif novelty in {"yaw", "pitch"}:
        rotation = pitch_rotation(np.pi / 3)
        if novelty == "yaw":
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        _observed_view(agent.task, rotation=rotation)
    elif novelty == "pose_noise":
        _observed_view(agent.task, position=(.01, 0., 0.), rotation=pitch_rotation(.01))
    elif novelty == "pixel":
        obj.rgb[0, 0] = 255
    elif novelty == "bbox":
        obj.bbox = [0, 0, 1, 2]
    else:
        agent.task.snapshot.metadata["observation_images"][0]["id"] = "another-frame"
    guided.observation_updated()
    reopened = novelty in {"translation", "yaw", "pitch"}
    assert ("u" not in guided.deferred_candidates) is reopened
    assert guided.view_attempted("u", (1., 0., 0.)) is not reopened
    assert guided.action_epoch("u") == int(reopened)
    assert guided.candidate_motion_goals[(1, "u", 0)][(1., 0., 0.)] == "reached"
    guided.defer_candidate("still_unknown")
    _observed_view(agent.task)
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"}
    assert guided.action_epoch("u") == int(reopened)
    assert not agent.task.progress.completed


@pytest.mark.parametrize("backend", ["core", "habitat", "isaac"])
def test_exhausted_view_budget_ignores_passive_coverage_until_new_anchor(
        monkeypatch, tmp_path, offline_replay, backend):
    from scene.contracts import ObjectNodeSnapshot

    guided, agent, _ = replay_policy(backend, monkeypatch, tmp_path)
    assert guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    state = guided.candidate_view_states[(1, "u")]
    state.start("u", {}, {"observation_identity": {"task_revision": 1}})
    state.exhausted = True
    _observed_view(agent.task)
    guided.defer_candidate("view_options_exhausted")
    _observed_view(agent.task, position=(1., 0., 0.))
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"} and guided.action_epoch("u") == 0
    _relation_feedback(guided, agent, "missing_anchor")
    guided.tools.refresh_scene = Mock()
    guided.refresh_scene = Mock()
    agent.task.snapshot = replace(agent.task.snapshot, objects=(*agent.task.snapshot.objects,
        ObjectNodeSnapshot("new-anchor", "table", (2., 0., 0.))))
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("u") == 1
    assert not agent.task.progress.completed


@pytest.mark.parametrize("status", ["missing_anchor", "rejected_relation", "insufficient_evidence"])
def test_relation_retry_uses_relevant_anchor_not_target_pixels(monkeypatch, tmp_path, offline_replay, status):
    from scene.contracts import ObjectNodeSnapshot

    guided, agent, obj = replay_policy("core", monkeypatch, tmp_path)
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    _relation_feedback(guided, agent, status, "old-anchor" if status != "missing_anchor" else "")
    guided.defer_candidate("need_relation_check")
    obj.rgb[0, 0] = 255
    _observed_view(agent.task, position=(2., 0., 0.))
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"}
    agent.task.snapshot = replace(agent.task.snapshot, objects=(*agent.task.snapshot.objects,
        ObjectNodeSnapshot("unrelated", "wall", (3., 0., 0.))))
    guided.tools.refresh_scene = Mock()
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"} and guided.action_epoch("u") == 0
    agent.task.snapshot = replace(agent.task.snapshot, objects=(*agent.task.snapshot.objects,
        ObjectNodeSnapshot("new-anchor", "table", (2., 0., 0.))))
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("u") == 1
    assert not guided.view_attempted("u", (1., 0., 0.))
    guided.defer_candidate("anchor_not_sufficient")
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"} and guided.action_epoch("u") == 1
    assert not agent.task.progress.completed


def test_relation_retry_ignores_satisfied_predicate_and_reuses_semantic_anchor_match(
        monkeypatch, tmp_path, offline_replay):
    from instruction_adapter.concept_matcher import ConceptMatchRecord, _concept_signature
    from instruction_adapter.contracts import TargetQuery
    from instruction_adapter.verifier import instruction_hash
    from scene.contracts import ObjectNodeSnapshot

    guided, agent, _ = replay_policy("core", monkeypatch, tmp_path)
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    outcomes = _relation_feedback(guided, agent, "rejected_relation", "table-1")
    agent.task.plan.targets.append(TargetQuery("light", "lamp", terminal=False, role="anchor"))
    outcomes.extend([{**outcomes[0], "anchor_uid": "table-2", "status": "verified"},
                     {**outcomes[0], "predicate_id": "c1", "anchor_concept_id": "light",
                      "anchor_uid": "", "status": "missing_anchor"}])
    guided.defer_candidate("need_relation_check")
    guided.tools.refresh_scene = Mock()
    agent.task.snapshot = replace(agent.task.snapshot, objects=(*agent.task.snapshot.objects,
        ObjectNodeSnapshot("table-3", "table", (2., 0., 0.)),
        ObjectNodeSnapshot("light-1", "pendant", (3., 0., 0.))))
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"}
    concept = agent.task.plan.targets[-1].concept_query()
    key = (instruction_hash(agent.task.spec.raw_instruction), _concept_signature(concept), "light-1", "pendant")
    agent.task.concept_matcher.records[key] = ConceptMatchRecord(key[0], concept.id, "light-1", "pendant",
        candidate_eligible=True, matches_concept=True)
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("u") == 1
    assert not agent.task.progress.completed


@pytest.mark.parametrize("status", ["missing_anchor", "rejected_relation", "insufficient_evidence"])
@pytest.mark.parametrize("review_due", [False, True])
def test_relation_gap_changes_executable_action(monkeypatch, tmp_path, offline_replay, status, review_due):
    guided, agent, _ = replay_policy("core", monkeypatch, tmp_path)
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    outcomes = _relation_feedback(guided, agent, status, "anchor" if status != "missing_anchor" else "")
    guided.capture_verification = Mock(return_value=GuidedVerification("u", "target", "goal",
        "reject_relation" if status == "rejected_relation" else "need_relation_check", False,
        relation_outcomes=outcomes))
    if review_due:
        agent.episode_steps = 100
    state = guided.verify((1., 0., 0.))
    if status == "insufficient_evidence":
        assert state == ("recovery" if review_due else "moving")
        if review_due:
            guided.recover(guided.recovery_context, 0)
        assert guided.tools.motion.active_intent.goal_pose.position == (2., 0., 0.)
        assert guided.tools.motion.active_intent.target_object_uid == "u"
    else:
        assert state == "select" and guided.deferred_candidates == {"u"}
        guided.select(0)
        assert guided.tools.motion.active_intent.target_object_uid is None
        assert guided.tools.motion.active_intent.goal_pose.position == (6. if review_due else 5., 0., 0.)
    assert not agent.task.progress.completed


@pytest.mark.parametrize("backend", ["core", "habitat", "isaac"])
@pytest.mark.parametrize("status", ["missing_anchor", "rejected_relation", "insufficient_evidence"])
def test_invisible_target_preserves_relation_retry_context(
        monkeypatch, tmp_path, offline_replay, backend, status):
    from backends.isaac.runtime import IsaacEvidenceVerifier
    from scene.contracts import ObjectNodeSnapshot

    guided, agent, obj = replay_policy(backend, monkeypatch, tmp_path)
    agent.need_check_again = False
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    outcomes = _relation_feedback(guided, agent, status, "anchor" if status != "missing_anchor" else "")
    verifier = IsaacEvidenceVerifier.__new__(IsaacEvidenceVerifier)
    verifier.tools = guided.tools
    result = verifier.verify(raw_instruction=agent.task.spec.raw_instruction,
        instruction_plan=agent.task.plan, candidate=obj, evidence={"observation_status": "need_observation"})
    assert result.decision == "need_better_view" and not result.view_objective.get("relation_outcomes")
    guided.capture_verification = Mock(return_value=GuidedVerification(
        "u", "target", "goal", result.decision, False,
        relation_outcomes=result.view_objective.get("relation_outcomes", [])))
    guided.feedback_views = Mock(return_value=[])
    assert guided.verify((1., 0., 0.)) == "select"
    assert guided.relation_outcomes[(1, "u")] == outcomes
    _observed_view(agent.task, position=(2., 0., 0.))
    guided.observation_updated()
    assert guided.deferred_candidates == {"u"} and guided.action_epoch("u") == 0
    assert guided.view_attempted("u", (1., 0., 0.))
    guided.refresh_scene = Mock()
    agent.task.snapshot = replace(agent.task.snapshot, objects=(*agent.task.snapshot.objects,
        ObjectNodeSnapshot("new-anchor", "table", (3., 0., 0.))))
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("u") == 1
    assert not guided.view_attempted("u", (1., 0., 0.))
    verified = [{**outcomes[0], "anchor_uid": "new-anchor", "status": "verified"}]
    guided.capture_verification.return_value = GuidedVerification(
        "u", "target", "goal", "need_better_view", False, relation_outcomes=verified)
    assert guided.verify((1., 0., 0.)) == "select"
    assert guided.relation_outcomes[(1, "u")] == verified
    _observed_view(agent.task, position=(3., 0., 0.))
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("u") == 2
    assert not agent.task.progress.completed


def test_verified_alternative_does_not_bypass_exhausted_view_control(monkeypatch, tmp_path, offline_replay):
    guided, agent, _ = replay_policy("core", monkeypatch, tmp_path)
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    outcomes = _relation_feedback(guided, agent, "insufficient_evidence", "anchor")
    outcomes.append({**outcomes[0], "anchor_uid": "other", "status": "verified"})
    guided.capture_verification = Mock(return_value=GuidedVerification(
        "u", "target", "goal", "need_better_view", False, relation_outcomes=outcomes))
    guided.continue_view_control = Mock(return_value=False)
    guided.feedback_views = Mock(wraps=guided.feedback_views)
    assert guided.verify((1., 0., 0.)) == "select"
    guided.feedback_views.assert_not_called()
    assert guided.deferred_candidates == {"u"}


def test_empty_relation_recovery_reselects_other_candidates(monkeypatch, tmp_path, offline_replay):
    guided, agent, _ = replay_policy("core", monkeypatch, tmp_path)
    guided.select(0)
    finished(guided, NavigationStatusCode.REACHED)
    _relation_feedback(guided, agent, "insufficient_evidence", "anchor")
    guided.feedback_views = Mock(return_value=[])
    guided.exploration_options = Mock(return_value=(None, "exhausted", []))
    assert guided.recover({"candidate_uid": "u", "reason": "need_relation_check"}) == "select"
    assert guided.deferred_candidates == {"u"}
    guided.runtime.choose_recovery.assert_not_called()
    assert not agent.task.progress.completed


@pytest.mark.parametrize("backend", ["habitat", "isaac"])
def test_refresh_before_terminal_record_cannot_repeat_zero_travel_candidates(
        monkeypatch, tmp_path, offline_replay, backend):
    from backends.habitat.guided_runtime import HabitatGuidedExecution
    from backends.isaac.runtime import IsaacGuidedExecution

    guided, agent, obj = replay_policy(backend, monkeypatch, tmp_path)
    guided.tools.objects["v"] = NS(uid="v", rgb=obj.rgb, bbox=list(obj.bbox), position=np.array([3., 0., 1.]))
    agent.task.concept_matcher.match_many.return_value.append(
        NS(object_uid="v", candidate_eligible=True, confidence=1., as_dict=lambda: {}))
    guided.tools.publish_object_viewpoints.return_value = [
        ViewpointSnapshot(uid, Pose3D((0., 0., 0.)), visible_objects=(uid,)) for uid in ("u", "v")]
    _relation_feedback(guided, agent, "missing_anchor")
    guided.capture_verification = Mock(side_effect=lambda position: GuidedVerification(
        guided.candidate_uid, "target", "goal", "need_relation_check", False,
        relation_outcomes=[{**guided.relation_outcomes[(1, "u")][0], "candidate_uid": guided.candidate_uid}]))
    if backend == "isaac":
        execution = IsaacGuidedExecution(guided, max_moves=20, panorama_views=4)
        execution.observe = Mock()
    else:
        execution = HabitatGuidedExecution(guided, agent, episode_idx=0, max_steps=20, stop_on_success=True)
    visited = []
    for _ in range(3):
        guided.select(0)
        motion = guided.tools.motion
        uid = motion.active_intent.target_object_uid
        visited.append(uid)
        if uid is None:
            break
        if backend == "isaac":
            guided.tools.client.status = "reached"
            motion.controller.advance(motion.active_goal_id)
        else:
            motion.controller.poll_status.return_value = NavigationStatus(NavigationStatusCode.REACHED)
        feedback = motion.poll_active()
        obj.rgb[0, 0] += 1
        execution.after_motion(feedback)
        guided.motion_finished(feedback)
        assert guided.verify((0., 0., 0.)) == "select"
    assert visited == ["u", "v", None]
    assert not agent.task.progress.completed


@pytest.mark.parametrize("backend", ["habitat", "isaac"])
def test_habitat_does_not_override_shared_decision_algorithms(backend):
    from backends.habitat.guided_runtime import GuidedNavigation
    from backends.isaac.runtime import IsaacGuidedNavigation

    adapter = GuidedNavigation if backend == "habitat" else IsaacGuidedNavigation
    for method in ("select", "verify", "recover", "review_search_branch", "candidate_evidence",
                   "view_attempted", "attempted_positions", "motion_finished", "defer_candidate", "observation_updated"):
        assert getattr(adapter, method) is getattr(GuidedPolicy, method)


def loop_fixture(*, code=NavigationStatusCode.REACHED, frontier=False):
    events = []
    controller = NS(send_goal=Mock(return_value="goal"),
                    poll_status=Mock(return_value=NavigationStatus(code)), cancel=Mock())
    motion = MotionSupervisor(controller)
    metadata = {"task_revision": 1, "candidate_evidence_fingerprint": "e1"}
    if frontier:
        metadata["node_index"] = 0
    intent = NavigationIntent(MotionGoalMode.GO_TO_FRONTIER if frontier else MotionGoalMode.IMPROVE_VIEW,
                              goal_pose=Pose3D((1., 0., 0.)), target_object_uid=None if frontier else "u",
                              metadata=metadata)
    motion.submit(intent, intent.to_motion_goal())
    guided = NS(tools=NS(motion=motion), motion_finished=Mock(), record=Mock(),
                select=Mock(return_value=False), verify=Mock(return_value="verified_finish"),
                recover=Mock(return_value="search_exhausted"),
                observation_updated=Mock(side_effect=lambda: events.append("evidence")))
    execution = NS(episode_idx=42, initialize=Mock(), limit_reason=Mock(return_value=None),
                   advance=Mock(), after_motion=Mock(),
                   frontier_arrived=Mock(side_effect=lambda f: events.append("observe")),
                   frontier_observed=Mock(side_effect=lambda f: events.append("mark")),
                   object_arrived=Mock(), finish_success=Mock())
    return guided, execution, controller, events


def test_loop_frontier_observe_refresh_then_mark():
    guided, execution, controller, events = loop_fixture(frontier=True)
    assert run_guided_loop(guided, execution) == "search_exhausted"
    assert events == ["observe", "evidence", "mark"]
    execution.advance.assert_called_once_with()
    guided.select.assert_called_once_with(42)
    guided.verify.assert_not_called()
    execution.finish_success.assert_not_called()
    controller.cancel.assert_not_called()


def test_loop_blocked_recovers_without_observation_or_verification():
    guided, execution, _, _ = loop_fixture(code=NavigationStatusCode.BLOCKED)
    assert run_guided_loop(guided, execution) == "search_exhausted"
    assert guided.recover.call_args.args == (guided.recovery_context, 42)
    assert guided.recovery_context["status"] == "blocked"
    guided.motion_finished.assert_called_once()
    guided.verify.assert_not_called()
    execution.frontier_arrived.assert_not_called()
    execution.finish_success.assert_not_called()


@pytest.mark.parametrize("code", [NavigationStatusCode.REACHED, NavigationStatusCode.RUNNING])
def test_last_budget_accepted_motion_runs_before_decision_limit(code):
    guided, execution, controller, _ = loop_fixture(code=code)
    controller.poll_status.side_effect = [NavigationStatus(code), NavigationStatus(NavigationStatusCode.REACHED)]
    execution.limit_reason.return_value = "decision_budget_exhausted"
    assert run_guided_loop(guided, execution) == "verified_finish"
    assert execution.advance.call_count == (2 if code is NavigationStatusCode.RUNNING else 1)
    guided.verify.assert_called_once_with((1., 0., 0.))
    execution.finish_success.assert_called_once_with()
    controller.cancel.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["initialize", "advance", "after_motion", "limit_reason", "select"])
def test_loop_errors_always_cancel_owned_motion_without_masking_original(failure_stage):
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.RUNNING)
    error = ValueError("original execution error")
    if failure_stage == "select":
        motion = guided.tools.motion
        intent, goal = motion.active_intent, motion.active_motion_goal
        controller.poll_status.return_value = NavigationStatus(NavigationStatusCode.PREEMPTED)
        motion.poll_active()

        def submit_then_fail(index):
            motion.submit(intent, goal)
            raise error

        guided.select.side_effect = submit_then_fail
    else:
        getattr(execution, failure_stage).side_effect = error
    controller.cancel.side_effect = lambda goal_id: setattr(
        controller.poll_status, "return_value", NavigationStatus(NavigationStatusCode.PREEMPTED))
    with pytest.raises(ValueError) as raised:
        run_guided_loop(guided, execution)
    assert raised.value is error
    controller.cancel.assert_called_once_with("goal")
    assert guided.tools.motion.active_request_id is None
    execution.finish_success.assert_not_called()


def test_loop_limit_cancels_without_successful_stop():
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.PREEMPTED)
    execution.limit_reason.return_value = "episode_limit"
    assert run_guided_loop(guided, execution) == "episode_limit"
    controller.cancel.assert_called_once_with("goal")
    execution.advance.assert_not_called()
    execution.finish_success.assert_not_called()
    assert guided.tools.motion.last_terminal_status.status is NavigationStatusCode.PREEMPTED


@pytest.mark.parametrize("original_failure", [False, True])
def test_cleanup_timeout_is_visible_and_original_error_remains_primary(monkeypatch, original_failure):
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.RUNNING)
    # Use the real bounded cancellation path, but a zero deadline to keep the test offline and immediate.
    cancel = guided.tools.motion.cancel_and_wait
    monkeypatch.setattr(guided.tools.motion, "cancel_and_wait", lambda: cancel(timeout=0))
    original = ValueError("original")
    if original_failure:
        execution.initialize.side_effect = original
    else:
        execution.limit_reason.return_value = "episode_limit"
    with pytest.raises(ValueError if original_failure else TimeoutError) as raised:
        run_guided_loop(guided, execution)
    if original_failure:
        assert raised.value is original
        assert isinstance(raised.value.__cause__, TimeoutError)
    assert guided.tools.motion.active_request_id is not None
    assert guided.tools.motion.last_terminal_status is None
    controller.cancel.assert_called_once_with("goal")
    execution.finish_success.assert_not_called()
