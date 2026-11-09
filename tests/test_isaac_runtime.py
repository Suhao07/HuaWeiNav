import base64
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from backends.isaac.perception import IsaacObservationMap, visible_projection
from backends.isaac.runtime import (IsaacClient, IsaacMotionController, IsaacNavigationTools,
                                    pose_value, refresh_observations)
from motion.contracts import MotionGoal, MotionGoalMode, NavigationIntent, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from scene.contracts import (FrontierSnapshot, ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot,
                             ViewpointSnapshot)


class FakeClient:
    def __init__(self):
        self.commands = []
        self.pose = {"position": [0., 0., 1.5], "yaw": 0.}
        self.status = "running"

    def command(self, action, **arguments):
        self.commands.append((action, arguments))
        if action == "move":
            self.pose = {"position": [1., 0., 1.5], "yaw": arguments["yaw"]}
        return {"pose": self.pose, "status": self.status if action == "move" else "reached",
                "path_length_remaining": 1., "zero_velocity": True}


def test_motion_receipts_and_supervisor_ownership():
    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    supervisor = MotionSupervisor(controller)
    goal = MotionGoal(MotionGoalMode.GO_TO_FRONTIER, Pose3D((2., 0., 1.5)))
    uid = supervisor.submit(NavigationIntent(goal.mode), goal, request_id="request", control_epoch=0)
    assert client.commands == []
    assert supervisor.poll_active().status.status == NavigationStatusCode.QUEUED
    controller.advance(uid)
    assert client.commands[-1] == ("move", {"position": [2., 0., 1.5], "yaw": 0., "max_distance": 1.7})
    assert supervisor.poll_active().tracking_active
    client.status = "reached"
    controller.advance(uid)
    assert not supervisor.poll_active().tracking_active
    assert supervisor.active_request_id is None
    assert controller.moves == 2


def test_cancel_requires_stop_receipt():
    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    supervisor = MotionSupervisor(controller)
    goal = MotionGoal(MotionGoalMode.GO_TO_OBJECT, Pose3D((2., 0., 1.5)))
    supervisor.submit(NavigationIntent(goal.mode), goal, request_id="request")
    supervisor.cancel()
    assert client.commands == [("STOP", {})]
    assert supervisor.active_request_id is not None
    assert supervisor.poll_active().status.status == NavigationStatusCode.PREEMPTED
    assert supervisor.active_request_id is None


def test_execution_failure_is_not_arrival():
    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    uid = controller.send_goal(MotionGoal(MotionGoalMode.GO_TO_OBJECT, Pose3D((2., 0., 1.5))))
    def fail(*args, **kwargs):
        raise ConnectionError("simulator unavailable")
    client.command = fail
    with pytest.raises(ConnectionError):
        controller.advance(uid)
    assert controller.poll_status(uid).status == NavigationStatusCode.QUEUED


def projection_fixture():
    yy, xx = np.mgrid[3:7, 2:7]
    points = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    return points, np.ones((10, 10)), np.eye(3), np.eye(4)


def test_current_projection_and_camera_transform():
    points, depth, intrinsic, pose = projection_fixture()
    pose[:3, 3] = (10., -20., 30.)
    points += pose[:3, 3]
    bbox, count = visible_projection(points, depth, intrinsic, pose)
    assert bbox == [2, 3, 7, 7]
    assert count == 20


@pytest.mark.parametrize("value", [0., float("nan"), float("inf"), 2.])
def test_occluded_or_invalid_depth_cannot_verify(value):
    points, depth, intrinsic, pose = projection_fixture()
    depth[:] = value
    assert visible_projection(points, depth, intrinsic, pose) == (None, 0)


def test_behind_camera_and_sparse_surfaces_cannot_verify():
    points, depth, intrinsic, pose = projection_fixture()
    points[:, 2] = -1
    assert visible_projection(points, depth, intrinsic, pose) == (None, 0)
    points[:, 2] = 1
    assert visible_projection(points[:9], depth, intrinsic, pose) == (None, 9)


def test_public_pose_uses_world_z_yaw():
    pose = pose_value({"position": [1., 2., 1.5], "yaw": np.pi})
    assert pose.frame_id == "map"
    assert pose.orientation_xyzw == pytest.approx((0., 0., 1., 0.))


def test_client_exact_json_protocol(monkeypatch):
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"status":"reached"}'
    def open_request(request, timeout):
        assert request.full_url == "http://127.0.0.1:9466/command"
        assert json.loads(request.data) == {"action": "STOP"}
        assert timeout == 240
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", open_request)
    assert IsaacClient(9466).command("STOP") == {"status": "reached"}


def _action(name, **arguments):
    return {"action": {"name": name, "arguments": arguments, "reason": "Offline agent decision"}}


def _replay_episode(monkeypatch, tmp_path, actions, *, initial_calls=1, client=None, spec=None):
    """Use the real agent/tool/session contracts with deterministic local model responses."""
    from backends.isaac import runtime
    from harness.runtime import AgentRuntime
    from instruction.contracts import CompileResult, TargetSpec, TaskSpec
    from task.session import TaskSession
    from verification.contracts import VerificationResult

    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    spec = spec or TaskSpec(public["instruction"], (TargetSpec("t1", "chair"),))
    public["instruction"] = spec.raw_instruction
    initial = SimpleNamespace(model_calls=initial_calls, compile=Mock(return_value=CompileResult("success", spec)))
    client = client or FakeClient()
    state = SimpleNamespace(public=public, client=client, initial=initial, refreshes=[], contexts=[])
    commands = client.command

    def command(action, **arguments):
        if action == "viewpoints":
            client.commands.append((action, arguments))
            return [{"position": [float(i), 0., 1.5], "yaw": float(i),
                     "source": "observed_freemap", "path_length": float(i)} for i in (1, 2)]
        return commands(action, **arguments)

    client.command = command
    objects = tuple(ObjectNodeSnapshot(uid, "chair", (float(i), 0., 1.5))
                    for i, uid in enumerate(("first", "second"), 1))
    frontiers = tuple(FrontierSnapshot(f"f:{i}", (float(i), 2., 1.5), score=float(i),
        metadata={"frame_id": "map", "explored_density": 0.}) for i in (1, 2))

    def session_factory(snapshot, plan, **kwargs):
        assert kwargs["constraint_evaluator"].relation_service.vlm_callback.vlm == "self_hosted"
        state.session = TaskSession(replace(snapshot, objects=objects, frontiers=frontiers), plan, **kwargs)
        return state.session

    def tools_factory(session, motion, *args):
        tools = IsaacNavigationTools(session, motion, *args)
        tools.query_scene = Mock(return_value={"candidates": [
            {"candidate_uid": obj.uid, "target_id": "t1", "role": "primary"} for obj in objects],
            "reason": "observed"})
        tools.evidence_provider = Mock(return_value={})
        tools.verifier = SimpleNamespace(verify=Mock(side_effect=lambda **kwargs: VerificationResult(
            True, "accept", task_revision=session.spec.revision, target_uid=kwargs["candidate"].uid,
            semantic_satisfied=True, view_sufficient_for_stop=True,
            diagnostics={"original_instruction_review": {
                "coverage_complete": True, "interpretation_consistent": True, "task_satisfied": True}})))
        tools.dispatch = Mock(wraps=tools.dispatch)
        state.tools, state.motion, state.controller = tools, motion, motion.controller
        return tools

    def refresh(session, observation_map, controller, client, output, views):
        state.refreshes.append((dict(controller.pose), views, list(session.detector_terms)))
        session.snapshot = replace(session.snapshot, robot_pose=pose_value(controller.pose))

    responses = iter(actions)

    def parse(**kwargs):
        context = json.loads(kwargs["messages"][1]["content"][0]["text"])
        state.contexts.append(context)
        response = next(responses)
        if callable(response):
            response = response(state, context)
        parsed = kwargs["response_format"](**response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    state.parse = Mock(side_effect=parse)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=state.parse))))

    def agent_factory(tools, trace_path, **kwargs):
        state.agent = AgentRuntime(tools, trace_path, client_model=(model, "offline-replay"), **kwargs)
        return state.agent

    state.observation_map_factory = Mock(return_value=object())
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    monkeypatch.setattr(runtime, "InitialInstructionCompiler", Mock(return_value=initial))
    monkeypatch.setattr(runtime, "IsaacObservationMap", state.observation_map_factory)
    monkeypatch.setattr(runtime, "TaskSession", session_factory)
    monkeypatch.setattr(runtime, "IsaacNavigationTools", tools_factory)
    monkeypatch.setattr(runtime, "AgentRuntime", agent_factory)
    monkeypatch.setattr(runtime, "refresh_observations", refresh)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "model_requests"))
    state.run = lambda **kwargs: runtime.run_episode(public, tmp_path, **kwargs)
    return state


@pytest.mark.parametrize("settings,views", [
    ({}, 4),
    ({"panorama_views": 2}, 2),
    ({"panorama_views": 3}, 3),
    ({"panorama_views": 7}, 7),
])
def test_last_model_call_motion_executes_before_budget_stop(
        monkeypatch, tmp_path, settings, views):
    from backends.isaac import runtime

    client = FakeClient()
    command = client.command
    move_statuses = iter(["running", "reached"])

    def advance_command(action, **arguments):
        if action == "move":
            client.status = next(move_statuses)
        return command(action, **arguments)

    client.command = advance_command
    state = _replay_episode(monkeypatch, tmp_path, [_action("submit_motion", task_revision=1,
        control_epoch=0, request_id="last-call", frontier_uid="f:2")], initial_calls=0, client=client)
    original_public = dict(state.public)
    result = state.run(max_model_calls=1, max_moves=3, **settings)
    assert state.public == original_public and len(state.public) == 6
    assert result["observation_mode"] == "multi"
    assert result["panorama_views"] == result["effective_views"] == views
    assert json.loads((tmp_path / "policy_result.json").read_text()) == result
    state.observation_map_factory.assert_called_once_with(runtime.BACKGROUND_CLASSES, device="cuda:0")
    assert "espresso machine" not in state.observation_map_factory.call_args.args[0]
    assert result["status"] == "completed"
    assert result["reason"] == "decision_budget_exhausted"
    assert not result["task_complete"]
    assert result["moves"] == 2
    assert sum(action == "move" for action, _ in client.commands) == 2
    assert client.commands[-2][0] == "finish"
    assert len(state.refreshes) == 3 and state.refreshes[0][0] != state.refreshes[1][0]
    assert [count for _, count, _ in state.refreshes] == [views, 0, views]
    assert result["policy"] == "strive_isaac_full_instruction_agent"
    assert result["semantic_authority"] == "raw_instruction"
    assert result["structured_interpretation"] == "revisable_execution_aid"
    assert "each_agent_tool_decision" in result["model_calls_scope"]
    assert {"prompting/templates.py", "prompting/schemas.py", "instruction/compiler.py",
            "instruction_adapter/constraints.py", "instruction_adapter/relation_verifier.py"} <= set(result["source_sha256"])
    assert result["final_frame"]["pose"] == client.pose == state.controller.pose
    assert state.motion.active_request_id is None
    assert state.tools.movement_budget_remaining == 1
    feedback = [event for event in state.agent.history if event.get("event") == "motion_feedback"]
    assert [event["status"]["status"] for event in feedback] == ["running", "reached"]
    assert [event["tracking_active"] for event in feedback] == [True, False]
    assert [event["active_request_id"] for event in feedback] == ["last-call", None]
    assert [event["moves"] for event in feedback] == [1, 2]
    for index, event in enumerate(feedback, 1):
        assert event["request_id"] == "last-call" and event["goal_id"] == "motion:0"
        assert event["control_epoch"] == 0 and not event["held"]
        assert event["goal"]["mode"] == "go_to_frontier"
        assert event["intent"]["metadata"]["frontier_uid"] == "f:2"
        assert event["robot_pose"] == json.loads(json.dumps(event["status"]["current_pose"]))
        assert event["status"]["metadata"]["receipt"]["pose"] == state.refreshes[index][0]
    assert not any(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list)
    state.tools.evidence_provider.assert_not_called()
    state.tools.query_scene.assert_not_called()


@pytest.mark.parametrize("settings,message", [
    ({"panorama_views": 0}, "panorama_views"),
    ({"panorama_views": 1}, "panorama_views"),
    ({"panorama_views": -1}, "panorama_views"),
    ({"panorama_views": 2.5}, "panorama_views"),
    ({"panorama_views": 2.0}, "panorama_views"),
    ({"panorama_views": "4"}, "panorama_views"),
    ({"panorama_views": None}, "panorama_views"),
    ({"panorama_views": True}, "panorama_views"),
    ({"panorama_views": False}, "panorama_views"),
    ({"max_moves": 0}, "budgets"),
])
def test_episode_rejects_invalid_observation_settings_before_resources(monkeypatch, tmp_path, settings, message):
    from backends.isaac import runtime

    resources = ["IsaacClient", "IsaacObservationMap", "InitialInstructionCompiler", "set_trace_dir"]
    factories = [Mock(side_effect=AssertionError("must not acquire resources")) for _ in resources]
    for name, factory in zip(resources, factories):
        monkeypatch.setattr(runtime, name, factory)
    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    with pytest.raises(ValueError, match=message):
        runtime.run_episode(public, tmp_path, **settings)
    for factory in factories:
        factory.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_observation_mode_is_not_a_runtime_parameter(tmp_path, mode):
    """Removed mode keywords must fail at binding, even for the former multi default."""
    from inspect import signature
    from backends.isaac import runtime

    for function, args in ((runtime.run_episode, (None, tmp_path)),
                           (refresh_observations, (None, None, None, None, tmp_path, 4))):
        assert "observation_mode" not in signature(function).parameters
        with pytest.raises(TypeError, match="observation_mode"):
            function(*args, observation_mode=mode)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("destination", ["second_candidate", "frontier"])
@pytest.mark.parametrize("status", ["reached", "blocked"])
def test_agent_selects_candidate_and_view_or_ignores_candidates_without_auto_verify(
        monkeypatch, tmp_path, destination, status):
    client = FakeClient()
    client.status = status
    actions = [_action("query_scene")]
    if destination == "second_candidate":
        actions += [_action("query_viewpoints", candidate_uid="second"),
                    _action("submit_motion", task_revision=1, control_epoch=0, request_id="chosen",
                            candidate_uid="second", viewpoint_uid="second:view:1:1")]
    else:
        actions += [_action("submit_motion", task_revision=1, control_epoch=0,
                            request_id="chosen", frontier_uid="f:1")]

    def observe_arrival(state, context):
        state.tools.evidence_provider.assert_not_called()
        assert state.motion.active_request_id is None
        assert not state.session.progress.completed
        feedback = context["recent_tools"][-1]
        assert feedback["event"] == "motion_feedback" and feedback["status"]["status"] == status
        assert feedback["candidate_uid"] == ("second" if destination == "second_candidate" else None)
        return _action("observe")

    actions.append(observe_arrival)
    state = _replay_episode(monkeypatch, tmp_path, actions, client=client)
    result = state.run(max_model_calls=len(actions) + 1, max_moves=3)
    assert result["reason"] == "decision_budget_exhausted" and result["moves"] == 1
    assert not result["task_complete"] and result["model_calls"] == len(actions) + 1
    state.tools.query_scene.assert_called_once_with()
    state.tools.evidence_provider.assert_not_called()
    assert not any(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list)
    assert len(state.refreshes) == 2
    goal, = state.controller.goals.values()
    if destination == "second_candidate":
        assert goal.target_object_uid == "second" and goal.goal_pose.position == (2., 0., 1.5)
        assert goal.metadata["viewpoint_uid"] == "second:view:1:1"
        assert [args for action, args in client.commands if action == "viewpoints"] == [{"position": [2., 0., 1.5]}]
    else:
        assert goal.mode == MotionGoalMode.GO_TO_FRONTIER
        assert goal.metadata["frontier_uid"] == "f:1"
        assert not any(action == "viewpoints" for action, _ in client.commands)
    assert client.commands[-2][0] == "finish" and client.commands[-1][0] == "observe"


@pytest.mark.parametrize("finish", [False, True])
def test_verify_requires_explicit_agent_finish_even_after_last_move(monkeypatch, tmp_path, finish):
    client = FakeClient()
    client.status = "reached"

    def verify(state, context):
        assert state.tools.movement_budget_remaining == 0 and state.motion.active_request_id is None
        assert not context["observation"]["progress"]["completed"]
        return _action("verify_target", candidate_uid="second", target_id="t1", task_revision=1)

    actions = [_action("submit_motion", task_revision=1, control_epoch=0, request_id="last-move", frontier_uid="f:1"), verify]
    if finish:
        def propose(state, context):
            assert context["observation"]["progress"]["completed"]
            assert not any(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list)
            return _action("propose_finish")
        actions.append(propose)
    state = _replay_episode(monkeypatch, tmp_path, actions, client=client)
    result = state.run(max_model_calls=len(actions) + 1, max_moves=1)
    assert result["task_complete"] == finish and result["moves"] == 1
    assert result["structured_task_complete"]
    assert result["prediction_task_revision"] == 1 and result["predicted_target_id"] == "t1"
    assert result["prediction_source"] == "current_full_instruction_verification"
    assert result["reason"] == ("verified_finish" if finish else "movement_budget_exhausted")
    assert sum(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list) == int(finish)
    state.tools.evidence_provider.assert_called_once_with("second")
    assert len(state.controller.goals) == 1 and state.motion.active_request_id is None
    assert len(state.refreshes) == 2
    assert all(args["max_distance"] == 1.7 for action, args in client.commands if action == "move")
    assert result["stop_receipt"]["zero_velocity"]
    assert result["final_frame"]["pose"] == state.controller.pose


def test_rejected_agent_finish_does_not_stop_episode(monkeypatch, tmp_path):
    state = _replay_episode(monkeypatch, tmp_path, [_action("propose_finish"), _action("observe")])
    result = state.run(max_model_calls=3)
    assert result["reason"] == "decision_budget_exhausted" and not result["task_complete"]
    assert state.parse.call_count == 2 and result["moves"] == 0
    assert [call.args for call in state.tools.dispatch.call_args_list if call.args[0] == "propose_finish"] == [
        ("propose_finish", {})]
    assert len(state.refreshes) == 1


@pytest.mark.parametrize("last_call", [False, True])
def test_running_motion_cancels_at_move_budget_with_terminal_feedback(monkeypatch, tmp_path, last_call):
    state = _replay_episode(monkeypatch, tmp_path, [_action("submit_motion", task_revision=1,
        control_epoch=0, request_id="budget", frontier_uid="f:1")])
    result = state.run(max_model_calls=2 if last_call else 8, max_moves=2)
    assert result["reason"] == "movement_budget_exhausted" and result["moves"] == 2
    assert state.parse.call_count == 1 and state.tools.movement_budget_remaining == 0
    assert state.motion.active_request_id is None and state.motion.control_epoch == 1
    assert [event["status"]["status"] for event in state.agent.history if event.get("event") == "motion_feedback"] == ["running", "running", "preempted"]
    terminal = state.agent.history[-1]
    assert not terminal["tracking_active"] and terminal["request_id"] == "budget"
    assert terminal["control_epoch"] == 1 and terminal["phase"] == "terminated"
    assert terminal["status"]["metadata"]["receipt"]["zero_velocity"]
    assert [action for action, _ in state.client.commands] == ["observe", "move", "move", "STOP", "finish", "observe"]
    state.tools.evidence_provider.assert_not_called()
    assert not any(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list)


def test_idle_exhausted_budget_rejects_new_motion_but_still_allows_finish(monkeypatch, tmp_path):
    client = FakeClient()
    client.status = "reached"
    actions = [_action("submit_motion", task_revision=1, control_epoch=0, request_id="last", frontier_uid="f:1"),
               _action("submit_motion", task_revision=1, control_epoch=0, request_id="over-budget", frontier_uid="f:2"),
               _action("verify_target", candidate_uid="second", target_id="t1", task_revision=1),
               _action("propose_finish")]
    state = _replay_episode(monkeypatch, tmp_path, actions, client=client)
    result = state.run(max_model_calls=5, max_moves=1)
    assert result["reason"] == "verified_finish" and result["moves"] == 1
    assert len(state.controller.goals) == 1 and state.motion.active_request_id is None
    rows = [json.loads(line) for line in (tmp_path / "tools.jsonl").read_text().splitlines()]
    rejected = next(row["result"] for row in rows if row.get("event") == "returned"
                    and row.get("arguments", {}).get("request_id") == "over-budget")
    assert rejected["accepted"] is False and rejected["reason"] == "movement_budget_exhausted"
    assert state.parse.call_count == 4


@pytest.mark.parametrize("failure", ["model", "move"])
def test_episode_propagates_original_failure_and_quiesces_owner(monkeypatch, tmp_path, failure):
    error = ConnectionError("offline injected failure")

    def fail(*args, **kwargs):
        raise error

    state = _replay_episode(monkeypatch, tmp_path, [fail] if failure == "model" else [
        _action("submit_motion", task_revision=1, control_epoch=0, request_id="failed", frontier_uid="f:1")])
    if failure == "move":
        command = state.client.command
        state.client.command = lambda action, **kwargs: fail() if action == "move" else command(action, **kwargs)
    with pytest.raises(ConnectionError) as raised:
        state.run(max_model_calls=2)
    assert raised.value is error
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["status"] == "error" and result["reason"] == "ConnectionError"
    assert result["model_calls"] == 2 and result["moves"] == 0
    assert state.motion.active_request_id is None and state.client.commands[-1][0] == "STOP"
    assert not any(action == "finish" for action, _ in state.client.commands)


@pytest.mark.parametrize("revision,epoch", [(1, 1), (0, 0)])
def test_version_rejection_preserves_guard_and_allows_corrected_proposal(revision, epoch):
    from backends.isaac.runtime import IsaacNavigationTools
    from scene.contracts import FrontierSnapshot, SemanticMapSnapshot

    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    tools = object.__new__(IsaacNavigationTools)
    tools.movement_budget_remaining = None
    tools.motion = MotionSupervisor(controller)
    tools.session = SimpleNamespace(spec=SimpleNamespace(revision=1), snapshot=SemanticMapSnapshot(
        0., Pose3D((0., 0., 1.5)), frontiers=(FrontierSnapshot(
            "frontier", (1., 0., 1.5), metadata={"frame_id": "map"}),)))
    rejected = tools.submit_motion(revision, epoch, "request", frontier_uid="frontier")
    assert rejected == {"accepted": False, "phase": "rejected", "reason": "proposal_version_mismatch",
                        "task_revision": 1, "control_epoch": 0}
    assert tools.motion.active_request_id is None
    assert controller.goals == {} and client.commands == []
    admitted = tools.submit_motion(1, 0, "request", frontier_uid="frontier")
    assert admitted["phase"] == "accepted"
    assert tools.motion.active_request_id == "request"
    assert len(controller.goals) == 1


def _frame_packet(root, frame_id, depth):
    rgb_path, depth_path = f"{frame_id}-rgb.png", f"{frame_id}-depth.npy"
    Image.new("RGB", (depth.shape[1], depth.shape[0]), (17, 33, 65)).save(root / rgb_path)
    np.save(root / depth_path, depth, allow_pickle=False)
    return {"capture_root": os.path.relpath(root),
            "frame": {"frame_id": frame_id, "rgb_path": rgb_path, "depth_path": depth_path,
                      "intrinsics": np.eye(3).tolist(), "camera_to_map": np.eye(4).tolist()},
            "pose": {"position": [0., 0., 1.5], "yaw": 0.}, "frontiers": []}


class FrameClient(FakeClient):
    def __init__(self, packets):
        super().__init__()
        self.packets = iter(packets)

    def command(self, action, **arguments):
        if action == "observe":
            self.commands.append((action, arguments))
            return next(self.packets)
        return super().command(action, **arguments)


def _surface_map():
    points, _, _, _ = projection_fixture()
    surfaces = {"candidate": points, "anchor": points + [2., 0., 0.],
                "occluded": points * 2, "behind": -points,
                "outside": points + [20., 0., 0.], "sparse": points[:9]}
    objects = tuple(ObjectNodeSnapshot(uid, "chair", (2., 0., 1.5)) for uid in surfaces)
    return SimpleNamespace(objects=lambda: objects, points=Mock(side_effect=surfaces.__getitem__),
                           add=Mock(side_effect=AssertionError("unexpected detector call")))


@pytest.mark.parametrize("target_list", [None, ["espresso machine", "chair"]])
def test_add_reports_only_current_fusion_bindings(monkeypatch, tmp_path, target_list):
    from mapping_utils.representation import ObjectNode

    def tensor(value):
        value = np.asarray(value)
        result = SimpleNamespace(shape=value.shape, numpy=lambda: value)
        result.cpu = lambda: result
        return result

    def cloud(points, *args):
        result = SimpleNamespace(point=SimpleNamespace(positions=tensor(points)))
        result.voxel_down_sample = lambda voxel: result
        return result

    packet = _frame_packet(tmp_path, 41, np.ones((20, 20)))
    boxes = [[0, 0, 5, 5], [1, 1, 13, 13], [3, 3, 15, 15], [2, 2, 14, 14]]
    masks = [np.zeros((20, 20), dtype=bool)]
    for box in boxes[1:]:
        mask = np.zeros((20, 20), dtype=bool)
        mask[box[1]:box[3], box[0]:box[2]] = True
        masks.append(mask)
    observation_map = object.__new__(IsaacObservationMap)
    observation_map.device, observation_map.voxel_m, observation_map.max_depth_m = "CPU:0", .05, 6.
    observation_map.detector = SimpleNamespace(segment=Mock(return_value=(
        list(map(tensor, boxes)), [.9] * 4, ["chair"] * 4, list(map(tensor, masks)), [], [], [])))
    points = np.array([[0., 0., 1.], [1., 1., 1.]])
    observation_map.entities = [ObjectNode(cloud(points), cloud(points), "chair", .9,
        points.mean(axis=0), len(points), None, [0, 0, 1, 1]) for _ in range(2)]
    for entity, uid in zip(observation_map.entities, ("unseen", "fused")):
        entity.uid = uid
    old = {"image_ref": str(tmp_path / "old.png"), "frame_id": 5, "bbox": [0, 0, 1, 1]}
    observation_map.latest = {"unseen": dict(old), "fused": dict(old)}

    def associate(refs, observed, *, target, is_target_tag):
        assert target == "" and not is_target_tag("chair")
        if not observed:
            return refs, []
        assert [entity.uid for entity in observed] == ["observed:41:1", "observed:41:2", "observed:41:3"]
        return refs + [observed[1]], [1, 2, 1]

    monkeypatch.setitem(sys.modules, "mapping.object_fusion", SimpleNamespace(
        associate_object_entities=associate, label_probabilities=lambda entity: {"chair": 1.}))
    monkeypatch.setitem(sys.modules, "mapping_utils.geometry", SimpleNamespace(
        gpu_pointcloud_from_array=cloud, gpu_cluster_filter=lambda value: value))
    report = (observation_map.add(packet) if target_list is None
              else observation_map.add(packet, target_list=target_list))
    assert observation_map.detector.segment.call_args.kwargs == {
        "target": None, "target_list": target_list}
    image = report["observation_image"]
    assert (report["detected"], report["admitted"], report["map_objects"]) == (4, 3, 3)
    assert image == {"id": "41", "image_ref": str(tmp_path / "41-rgb.png"),
        "observed_object_uids": ["fused", "observed:41:2"],
        "object_boxes": {"fused": boxes[3], "observed:41:2": boxes[2]}}
    objects = {obj.uid: obj for obj in observation_map.objects()}
    assert objects["fused"].metadata["frame_id"] == 41
    assert objects["fused"].image_ref == image["image_ref"]
    assert objects["fused"].bbox2d_xyxy == tuple(boxes[3])
    assert observation_map.latest["unseen"] == old
    observation_map.detector.segment.return_value = ([], [], [], [], [], [], [])
    empty = observation_map.add(_frame_packet(tmp_path, 57, np.ones((20, 20))),
                                target_list=["bookshelf"])
    assert observation_map.detector.segment.call_args.kwargs == {
        "target": None, "target_list": ["bookshelf"]}
    assert empty["observation_image"] == {"id": "57", "image_ref": str(tmp_path / "57-rgb.png"),
                                         "observed_object_uids": [], "object_boxes": {}}
    assert observation_map.objects() == tuple(objects.values())
    assert observation_map.detector.segment.call_count == 2
    json.dumps([report, empty], allow_nan=False)


@pytest.mark.parametrize("panorama_views", [0, 2, 3, 4])
def test_refresh_publishes_only_this_batch_and_depth_visible_final_frame(
        tmp_path, capsys, panorama_views):
    observation_map = _surface_map()
    image_count = max(1, panorama_views)
    packets = [_frame_packet(tmp_path, 40 + i * 7, np.ones((10, 10)))
               for i in range(image_count)]
    reports = []
    mapped_packets = packets[:panorama_views]
    for i, packet in enumerate(mapped_packets):
        boxes = {} if i % 2 else {"anchor": [4, 3, 9, 7]}
        reports.append({"frame_id": packet["frame"]["frame_id"], "observation_image": {
            "id": str(packet["frame"]["frame_id"]),
            "image_ref": str(tmp_path / packet["frame"]["rgb_path"]),
            "observed_object_uids": list(boxes), "object_boxes": boxes}})
    observation_map.add = Mock(side_effect=reports)
    old_images = [{"id": "old", "image_ref": str(tmp_path / "old.png"),
                   "observed_object_uids": ["occluded"], "object_boxes": {"occluded": [0, 0, 1, 1]}}]
    session = SimpleNamespace(detector_terms=["espresso machine", "table"],
        snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 1.5)),
                                     metadata={"observation_images": old_images}))
    client = FrameClient(packets)
    client.pose["yaw"] = start_yaw = 1.25
    packets[-1]["pose"] = {"position": [0.5, 0.25, 1.5], "yaw": start_yaw}
    packets[-1]["frontiers"] = [{"uid": "fresh", "position": [1., 2., 1.5]}]
    controller = IsaacMotionController(client, client.pose)
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views)
    images = session.snapshot.metadata["observation_images"]
    assert len(images) == image_count
    assert images[:-1] == ([report["observation_image"] for report in reports[:-1]] if panorama_views >= 2 else [])
    assert controller.pose is packets[-1]["pose"]
    assert session.snapshot.robot_pose == pose_value(packets[-1]["pose"])
    assert session.snapshot.frontiers == (FrontierSnapshot(**packets[-1]["frontiers"][0]),)
    assert session.snapshot.objects is observation_map.objects()
    assert session.snapshot.metadata["observation_mode"] == "multi"
    assert session.snapshot.metadata["effective_views"] == image_count
    assert session.snapshot.metadata["panorama_views"] == panorama_views
    assert images[-1] == {"id": str(packets[-1]["frame"]["frame_id"]),
        "image_ref": str(tmp_path / packets[-1]["frame"]["rgb_path"]),
        "observed_object_uids": ["candidate", "anchor"],
        "object_boxes": {"candidate": [2, 3, 7, 7], "anchor": [4, 3, 9, 7]}}
    assert old_images[0]["id"] == "old"
    assert [call.args[0] for call in observation_map.add.call_args_list] == mapped_packets
    assert [call.kwargs for call in observation_map.add.call_args_list] == [
        {"target_list": session.detector_terms}] * panorama_views
    assert [call.args[0] for call in observation_map.points.call_args_list] == [
        obj.uid for obj in observation_map.objects()]
    expected_commands = []
    if panorama_views >= 2:
        for i in range(panorama_views):
            yaw = start_yaw if i == panorama_views - 1 else start_yaw + 2 * np.pi * (i + 1) / panorama_views
            expected_commands.extend([("look", {"yaw": yaw}), ("observe", {})])
    else:
        expected_commands.extend([("look", {"yaw": start_yaw}), ("observe", {})])
    assert client.commands == expected_commands
    logged = [json.loads(line) for line in (tmp_path / "perception.jsonl").read_text().splitlines()] if panorama_views else []
    assert logged == [{**report, "observation_mode": "multi", "effective_views": image_count}
                      for report in reports]
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"event": "perception", **report} for report in logged]
    assert json.loads((tmp_path / "scene_snapshots.jsonl").read_text())["metadata"] == session.snapshot.metadata

    observation_map.add = Mock(side_effect=AssertionError("unexpected detector call"))
    client.packets = iter([_frame_packet(tmp_path, 99, np.zeros((10, 10)))])
    refresh_observations(session, observation_map, controller, client, tmp_path, 0)
    assert session.snapshot.metadata["observation_images"] == [{
        "id": "99", "image_ref": str(tmp_path / "99-rgb.png"),
        "observed_object_uids": [], "object_boxes": {}}]
    observation_map.add.assert_not_called()


@pytest.mark.parametrize("panorama_views", [1, -1, 2.0, 2.5, "4", None, True, False])
def test_refresh_rejects_single_view_or_invalid_panorama_before_io(tmp_path, panorama_views):
    """Only a panorama or the explicit internal zero-view reprojection is admissible."""
    observation_map, client = Mock(), Mock()
    with pytest.raises(ValueError, match="panorama_views"):
        refresh_observations(None, observation_map, None, client, tmp_path, panorama_views)
    assert not observation_map.mock_calls and not client.mock_calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("panorama_views", [2, 4])
def test_refresh_forwards_replaced_terms_to_actual_observation_map(monkeypatch, tmp_path, panorama_views):
    """Exercise refresh -> add -> segment, retaining historical labels across term revisions."""
    from instruction_adapter.contracts import InstructionPlan, TargetQuery
    from task.session import TaskSession

    monkeypatch.setitem(sys.modules, "mapping.object_fusion", SimpleNamespace(
        associate_object_entities=lambda refs, observed, **kwargs: (refs, [])))
    geometry = Mock(side_effect=AssertionError("Empty detections must not construct point clouds"))
    monkeypatch.setitem(sys.modules, "mapping_utils.geometry", SimpleNamespace(
        gpu_pointcloud_from_array=geometry, gpu_cluster_filter=geometry))
    points, _, _, _ = projection_fixture()
    objects = (ObjectNodeSnapshot("historical", "chair", (2., 0., 1.5)),)
    observation_map = object.__new__(IsaacObservationMap)
    observation_map.max_depth_m = 6.
    observation_map.entities = [SimpleNamespace(tag="chair")]
    historical = {"historical": {"frame_id": 0, "image_ref": str(tmp_path / "old.png")}}
    observation_map.latest = dict(historical)
    observation_map.objects = lambda: objects
    observation_map.points = lambda uid: points
    observation_map.detector = SimpleNamespace(segment=Mock(return_value=([],) * 7))
    session = TaskSession(SemanticMapSnapshot(0., Pose3D((0., 0., 1.5))),
        InstructionPlan("Find a reading spot.", targets=[TargetQuery(
            "goal", "reading spot", detector_terms=["espresso machine", "table"])]))
    spec, ledger = session.spec, session.verification_ledger
    images_per_refresh = max(1, panorama_views)
    packets = [_frame_packet(tmp_path, i + 1, np.ones((10, 10))) for i in range(2 * images_per_refresh)]
    client = FrameClient(packets)
    controller = IsaacMotionController(client, client.pose)
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views)
    first_snapshot = session.snapshot
    session.revise_search([{"concept_id": "goal", "detector_terms": ["bookshelf"]}],
        task_revision=session.spec.revision, search_revision=session.search_revision,
        snapshot=session.snapshot)
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views)
    assert session.spec is spec and session.verification_ledger is ledger
    calls = observation_map.detector.segment.call_args_list
    assert [call.kwargs for call in calls] == [
        {"target": None, "target_list": ["espresso machine", "table"]}] * panorama_views + [
        {"target": None, "target_list": ["bookshelf"]}] * panorama_views
    assert all(np.array_equal(call.args[0][0, 0], [17, 33, 65]) for call in calls)
    assert first_snapshot.objects is objects and session.snapshot.objects is objects
    assert observation_map.entities[0].tag == "chair"
    assert observation_map.latest == historical
    assert observation_map.detector.segment.call_count == 2 * panorama_views
    first_images = first_snapshot.metadata["observation_images"]
    current_images = session.snapshot.metadata["observation_images"]
    assert len(first_images) == len(current_images) == images_per_refresh
    assert first_images[-1]["id"] == str(images_per_refresh)
    assert current_images[-1]["id"] == str(2 * images_per_refresh)
    assert first_images[-1]["observed_object_uids"] == current_images[-1]["observed_object_uids"] == ["historical"]
    assert first_snapshot.objects[0].uid == "historical"
    reports = [json.loads(line) for line in (tmp_path / "perception.jsonl").read_text().splitlines()]
    assert all(report["observation_image"]["observed_object_uids"] == [] for report in reports)
    assert [action for action, _ in client.commands] == ["look", "observe"] * (2 * panorama_views)
    assert len((tmp_path / "scene_snapshots.jsonl").read_text().splitlines()) == 2
    geometry.assert_not_called()


@pytest.mark.parametrize("candidate_visible,anchor_visible", [(True, True), (False, True), (False, False)])
def test_capture_evidence_publishes_depth_filtered_covisibility_and_raw_image(
        monkeypatch, tmp_path, candidate_visible, anchor_visible):
    from backends.isaac import runtime

    observation_map = _surface_map()
    if not candidate_visible:
        original_points = observation_map.points.side_effect
        observation_map.points.side_effect = lambda uid: (
            original_points(uid) * 2 if uid == "candidate" else original_points(uid))
    packet = _frame_packet(tmp_path, 73, np.full((10, 10), 1. if anchor_visible else 0.))
    packet["pose"] = {"position": [0.5, 0.25, 1.5], "yaw": 1.2}
    packet["frontiers"] = ([{"uid": "current", "position": [1., 2., 1.5], "score": 0.75,
                             "metadata": {"frame_id": "map", "explored_density": 0.}}]
                           if anchor_visible else [])
    client = FrameClient([packet])
    tools = object.__new__(IsaacNavigationTools)
    tools.client, tools.output, tools.observation_map = client, tmp_path, observation_map
    tools.motion = MotionSupervisor(IsaacMotionController(client, client.pose))
    old_images = [{"id": "panorama", "image_ref": str(tmp_path / "old.png"),
                   "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [0, 0, 1, 1]}}]
    previous = SemanticMapSnapshot(11., Pose3D((0., 0., 1.5)), objects=observation_map.objects(),
        viewpoints=(ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("candidate",)),),
        frontiers=(FrontierSnapshot("stale", (4., 0., 1.5)),), source="isaac_rgbd_strive_fusion",
        metadata={"observation_images": old_images, "frame_revision": 7, "panorama_views": 3,
                  "observation_mode": "multi", "effective_views": 3,
                  "semantic_gt_used": False, "frontier_source": "official_freemap_action_proposer"})
    tools.session = SimpleNamespace(spec=SimpleNamespace(revision=4), snapshot=previous)
    journal = tmp_path / "scene_snapshots.jsonl"
    previous_row = json.dumps(asdict(previous), allow_nan=False)
    journal.write_text(previous_row + "\n")
    monkeypatch.setattr(runtime.time, "time", lambda: 74.)
    project = Mock(wraps=visible_projection)
    monkeypatch.setattr(runtime, "visible_projection", project)
    raw_path = tmp_path / packet["frame"]["rgb_path"]
    original_rgb = raw_path.read_bytes()
    evidence = tools.capture_evidence("candidate")
    boxes = {}
    if candidate_visible:
        boxes["candidate"] = [2, 3, 7, 7]
    if anchor_visible:
        boxes["anchor"] = [4, 3, 9, 7]
    assert evidence["observation_images"] == [{"id": "73", "image_ref": str(raw_path),
        "observed_object_uids": list(boxes), "object_boxes": boxes}]
    snapshot = tools.session.snapshot
    assert snapshot == replace(previous, timestamp=74., robot_pose=pose_value(packet["pose"]),
        frontiers=tuple(FrontierSnapshot(**item) for item in packet["frontiers"]),
        metadata={**previous.metadata, "panorama_views": 1, "effective_views": 1,
                  "observation_images": evidence["observation_images"]})
    assert snapshot.metadata["observation_mode"] == "multi"
    assert snapshot.objects is previous.objects and snapshot.viewpoints is previous.viewpoints
    assert snapshot.metadata is not previous.metadata
    assert previous.metadata["observation_images"] == old_images
    assert previous.metadata["observation_images"][0]["id"] == "panorama"
    assert tools.motion.controller.pose == packet["pose"]
    assert journal.read_text().splitlines() == [previous_row, json.dumps(asdict(snapshot), allow_nan=False)]
    assert evidence["observation_identity"] == {
        "candidate_uid": "candidate", "task_revision": 4, "observation_id": "73"}
    assert evidence["observation_status"] == ("current" if candidate_visible else "need_observation")
    assert evidence["geometry"]["bbox_xyxy"] == boxes.get("candidate")
    assert evidence["geometry"]["required_stop_distance"] == 2.
    assert evidence["geometry"]["projection_failed_in_final_view"] == (not candidate_visible)
    assert project.call_count == len(observation_map.objects())
    assert [call.args[0] for call in observation_map.points.call_args_list] == [
        obj.uid for obj in observation_map.objects()]
    observation_map.add.assert_not_called()
    assert client.commands == [("look", {"yaw": 0.}), ("observe", {})]
    assert raw_path.read_bytes() == original_rgb
    assert json.loads((tmp_path / "verification" / "000073-evidence.json").read_text()) == evidence
    if candidate_visible:
        assert evidence["current_rgb_with_bbox_path"] != str(raw_path)
        assert Path(evidence["current_rgb_with_bbox_path"]).is_file()
        with Image.open(evidence["object_crop_path"]) as crop:
            assert crop.size == (5, 4)
    else:
        assert "current_rgb_with_bbox_path" not in evidence
        assert "object_crop_path" not in evidence


@pytest.mark.parametrize("frontier_count", [1, 2])
@pytest.mark.parametrize("candidate_visible", [True, False], ids=["golden", "missing_projection"])
def test_agent_frontier_uses_same_current_packet_after_verification(tmp_path, candidate_visible, frontier_count):
    from harness.runtime import AgentRuntime
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from task.session import TaskSession
    from verification.contracts import VerificationResult

    observation_map = _surface_map()
    side_view = _frame_packet(tmp_path, 40, np.ones((10, 10)))
    refreshed = _frame_packet(tmp_path, 47, np.ones((10, 10)))
    current = _frame_packet(tmp_path, 73, np.full((10, 10), 1. if candidate_visible else 0.))
    refreshed["pose"] = {"position": [1., 0., 1.5], "yaw": 0.}
    current["pose"] = {"position": [1., 0., 1.5], "yaw": 1.2}
    for packet, prefix in ((refreshed, "stale"), (current, "current")):
        packet["frontiers"] = [{"uid": f"{prefix}:{i}", "position": [float(i), 2., 1.5],
            "score": float(i), "metadata": {"frame_id": "map", "explored_density": 0.}}
            for i in range(1, frontier_count + 1)]
    raw_path = tmp_path / current["frame"]["rgb_path"]
    Image.new("RGB", (10, 10), (120, 80, 40)).save(raw_path)
    observation_map.add = Mock(side_effect=[{"frame_id": packet["frame"]["frame_id"], "observation_image": {
        "id": str(packet["frame"]["frame_id"]), "image_ref": str(tmp_path / packet["frame"]["rgb_path"]),
        "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [2, 3, 7, 7]}}}
        for packet in (side_view, refreshed)])
    client = FrameClient([side_view, refreshed, current])
    client.status = "reached"
    controller = IsaacMotionController(client, client.pose)
    motion = MotionSupervisor(controller)
    view = ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("candidate",))
    session = TaskSession(SemanticMapSnapshot(0., pose_value(client.pose),
        objects=observation_map.objects(), viewpoints=(view,)),
        runtime_plan(TaskSpec("Find a chair.", (TargetSpec("t1", "chair"),))))
    tools = IsaacNavigationTools(session, motion, observation_map, client, tmp_path, "cognav")
    tools.evidence_provider = Mock(wraps=tools.capture_evidence)
    # Even a visible target can need further search; never invoke a real verifier model.
    tools.verifier.final.verify = Mock(return_value=VerificationResult(False, "need_better_view",
        task_revision=session.spec.revision, target_uid="candidate", view_sufficient_for_stop=False))
    tools.query_scene = Mock(return_value={"candidates": [], "reason": "no_untried_candidate_view"})

    actions = iter([
        _action("submit_motion", task_revision=1, control_epoch=0, request_id="candidate",
                candidate_uid="candidate", viewpoint_uid="view"),
        _action("verify_target", task_revision=1, candidate_uid="candidate", target_id="t1"),
        _action("submit_motion", task_revision=1, control_epoch=0, request_id="frontier",
                frontier_uid=f"current:{frontier_count}")])

    def parse(**kwargs):
        choice = kwargs["response_format"](**next(actions))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=choice))])

    parse_call = Mock(side_effect=parse)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_call))))
    agent = AgentRuntime(tools, tmp_path / "tools.jsonl", client_model=(model, "offline-replay"))
    agent.step()
    controller.advance(motion.active_goal_id)
    feedback = motion.poll_active()
    assert not feedback.tracking_active
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views=2)
    assert [image["id"] for image in session.snapshot.metadata["observation_images"]] == ["40", "47"]
    tools.evidence_provider.assert_not_called()
    agent.step()
    assert not session.progress.completed
    tools.evidence_provider.assert_called_once_with("candidate")
    assert tools.verifier.final.verify.call_count == int(candidate_visible)
    agent.step()

    assert parse_call.call_count == 3
    content = parse_call.call_args.kwargs["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    observation = context["observation"]
    evidence = json.loads((tmp_path / "verification" / "000073-evidence.json").read_text())
    assert evidence["observation_status"] == ("current" if candidate_visible else "need_observation")
    assert observation["robot_pose"] == pose_value(current["pose"]).as_dict()
    assert observation["frontiers"] == [asdict(FrontierSnapshot(**item)) for item in current["frontiers"]]
    assert [item["uid"] for item in observation["frontiers"]] == [
        f"current:{i}" for i in range(1, frontier_count + 1)]
    assert "baseline_recommendation" not in context
    assert len(observation["observation_images"]) == 1
    image = observation["observation_images"][0]
    assert {key: image[key] for key in evidence["observation_images"][0]} == evidence["observation_images"][0]
    assert image["image_status"] == "available"
    assert len(content) == 3 and content[2]["type"] == "image_url"
    assert base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1]) == raw_path.read_bytes()
    assert session.snapshot.metadata["observation_images"] == evidence["observation_images"]
    rows = [json.loads(line) for line in (tmp_path / "scene_snapshots.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["robot_pose"] == observation["robot_pose"]
    assert rows[-1]["timestamp"] == observation["timestamp"] > rows[0]["timestamp"]
    assert rows[-1]["frontiers"] == observation["frontiers"]
    assert rows[-1]["metadata"]["observation_images"] == evidence["observation_images"]
    assert [call.args[0] for call in observation_map.add.call_args_list] == [side_view, refreshed]
    assert [call.kwargs for call in observation_map.add.call_args_list] == [
        {"target_list": session.detector_terms}] * 2
    assert observation_map.points.call_count == 2 * len(observation_map.objects())
    assert [action for action, _ in client.commands] == ["move", "look", "observe", "look", "observe", "look", "observe"]
    assert agent.model_calls == 3 and controller.moves == 1
    goal = controller.goals[motion.active_goal_id]
    assert goal.mode == MotionGoalMode.GO_TO_FRONTIER
    assert list(goal.goal_pose.position) == current["frontiers"][-1]["position"]


@pytest.mark.parametrize("outcome", ["installed", "unchanged", "invalid", "guard_refused"])
def test_agent_interpretation_refresh_preserves_motion_history(monkeypatch, tmp_path, outcome):
    from instruction.compiler import compile_parsed_instruction
    from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES
    from prompting.schemas import ParsedInstruction

    client = FakeClient()
    client.status = "reached"
    spec = compile_parsed_instruction("Find somewhere to sit.",
        ParsedInstruction(targets=[{"name": "chair"}]), RUNTIME_CONSTRAINT_TYPES).task

    def reinterpret(state, context):
        assert state.controller.moves == 1 and state.motion.active_request_id is None
        state.previous_snapshot = state.session.snapshot
        state.previous_ledger = state.session.verification_ledger
        if outcome == "guard_refused":
            state.motion.hold()
        targets = [] if outcome == "invalid" else [{"name": "sofa" if outcome == "installed" else "chair"}]
        return _action("revise_interpretation", interpretation_update={
            "interpretation": {"targets": targets}, "basis_ids": ["instruction"],
            "reason": "Reconsider the user's requested sitting resource"})

    actions = [_action("submit_motion", task_revision=1, control_epoch=0, request_id="past", frontier_uid="f:1"),
               reinterpret, _action("observe")]
    state = _replay_episode(monkeypatch, tmp_path, actions, client=client, spec=spec)
    result = state.run(max_model_calls=4)
    assert result["reason"] == "decision_budget_exhausted"
    assert state.agent.model_calls == 4
    if outcome == "guard_refused":
        rejected = state.agent.history[-2]["result"]
        assert rejected["proposal_rejected"] and not rejected["accepted"]
        assert "motion state or control epoch changed" in rejected["reason"]
    changed = outcome == "installed"
    assert result["initial_predicted_target"] == "chair"
    assert result["predicted_target"] == ("sofa" if changed else "chair")
    assert result["prediction_task_revision"] == (2 if changed else 1)
    assert result["prediction_source"] == "final_active_interpretation"
    assert state.session.spec.revision == (2 if changed else 1)
    assert state.session.spec.raw_instruction == spec.raw_instruction
    assert state.session.spec.targets[0].name == ("sofa" if changed else "chair")
    assert len(state.refreshes) == 2 + int(changed)
    assert state.session.snapshot.objects is state.previous_snapshot.objects
    assert state.session.snapshot.robot_pose == pose_value(client.pose)
    assert (state.session.verification_ledger is state.previous_ledger) == (not changed)
    assert state.controller.moves == len(state.controller.goals) == 1
    assert state.motion.active_request_id is None
    assert state.controller.goals["motion:0"].metadata["task_revision"] == 1
    assert sum(action == "move" for action, _ in client.commands) == 1
    if changed:
        assert state.refreshes[-1][-1] == ["sofa"]
        assert state.contexts[-1]["observation"]["task"]["revision"] == 2
        assert len(state.session.interpretation_history) == 1
    state.tools.evidence_provider.assert_not_called()
    assert not any(call.args[0] == "propose_finish" for call in state.tools.dispatch.call_args_list)


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("action", ["observe", "query_scene"])
def test_agent_search_revision_refreshes_once_only_for_actual_changes(monkeypatch, tmp_path, changed, action):
    def revise(state, context):
        concept = context["observation"]["search_context"]["concepts"][0]
        state.original_spec = state.session.spec
        state.original_ledger = state.session.verification_ledger
        return _action("revise_search", search_updates=[{"concept_id": concept["id"],
            "detector_terms": ["sofa"] if changed else concept["detector_terms"],
            "basis_ids": ["instruction"], "reason": "Search for another sitting resource"}])

    def after_revision(state, context):
        state.tools.dispatch.assert_not_called()
        receipt = context["recent_tools"][-1]
        assert receipt["name"] == "revise_search"
        assert receipt["result"]["search_changed"] is changed
        assert receipt["result"]["search_revision_before"] == 0
        assert receipt["result"]["search_revision"] == int(changed)
        return _action(action)

    state = _replay_episode(monkeypatch, tmp_path, [revise, after_revision, _action("observe")])
    result = state.run(max_model_calls=4)
    assert result["moves"] == 0 and result["model_calls"] == 4
    assert len(state.refreshes) == 1 + int(changed)
    assert state.session.search_revision == int(changed)
    assert state.session.spec is state.original_spec and state.session.verification_ledger is state.original_ledger
    assert state.refreshes[-1][-1] == (["sofa"] if changed else ["chair"])
    assert state.contexts[-1]["observation"]["search_context"]["revision"] == int(changed)
    assert [action for action, _ in state.client.commands] == ["observe", "finish", "observe"]


@pytest.mark.parametrize("revision", ["search", "interpretation"])
def test_explicit_revision_replay_corrects_then_queries_and_moves(monkeypatch, tmp_path, revision):
    from instruction.contracts import TaskSpec, TargetSpec

    client = FakeClient()
    client.status = "reached"
    interpretation = revision == "interpretation"
    expected_task_revision = 2 if interpretation else 1
    expected_search_revision = 0 if interpretation else 1

    def revision_action(state, *, invalid=False):
        if interpretation:
            return _action("revise_interpretation", interpretation_update={
                "interpretation": {"targets": [] if invalid else [{"name": "sofa"}]},
                "basis_ids": ["instruction"], "reason": "Reconsider sitting resource"})
        return _action("revise_search", search_updates=[{
            "concept_id": state.session.spec.targets[0].id,
            "detector_terms": [" " if invalid else "sofa"],
            "basis_ids": ["instruction"], "reason": "Search for sitting resource"}])

    def invalid(state, context):
        state.original_spec = state.session.spec
        state.original_ledger = state.session.verification_ledger
        return revision_action(state, invalid=True)

    def correct(state, context):
        receipt = context["recent_tools"][-1]
        assert receipt["name"] == f"revise_{revision}" and not receipt["result"]["accepted"]
        assert state.session.spec is state.original_spec
        assert state.session.verification_ledger is state.original_ledger
        assert state.session.search_revision == 0 and len(state.refreshes) == 1
        state.tools.dispatch.assert_not_called()
        return revision_action(state)

    def noop(state, context):
        receipt = context["recent_tools"][-1]
        assert receipt["result"]["accepted"]
        assert receipt["result"]["interpretation_changed"] is interpretation
        assert receipt["result"]["task_revision_before"] == 1
        assert receipt["result"]["task_revision"] == expected_task_revision
        assert receipt["result"]["search_revision"] == expected_search_revision
        assert len(state.refreshes) == 2 and state.refreshes[-1][-1] == ["sofa"]
        assert context["observation"]["task"]["revision"] == expected_task_revision
        state.tools.dispatch.assert_not_called()
        return revision_action(state)

    def query(state, context):
        receipt = context["recent_tools"][-1]
        assert receipt["name"] == f"revise_{revision}" and receipt["result"]["accepted"]
        assert not receipt["result"]["interpretation_changed"] and not receipt["result"]["search_changed"]
        assert len(state.refreshes) == 2 and state.controller.moves == 0
        state.tools.dispatch.assert_not_called()
        state.tools.evidence_provider.assert_not_called()
        return _action("query_viewpoints", candidate_uid="second")

    def submit(state, context):
        receipt = context["recent_tools"][-1]
        assert receipt["name"] == "query_viewpoints" and len(receipt["result"]) == 2
        return _action("submit_motion", task_revision=expected_task_revision,
            control_epoch=context["observation"]["motion"]["control_epoch"], request_id="after-revision",
            candidate_uid="second", viewpoint_uid=receipt["result"][1]["uid"])

    def arrived(state, context):
        assert context["recent_tools"][-1]["event"] == "motion_feedback"
        assert state.controller.moves == 1 and state.motion.active_request_id is None
        assert len(state.refreshes) == 3
        state.tools.evidence_provider.assert_not_called()
        return _action("observe")

    actions = [invalid, correct, noop, query, submit, arrived]
    state = _replay_episode(monkeypatch, tmp_path, actions, client=client,
        spec=TaskSpec("Find somewhere to sit.", (TargetSpec("t1", "chair"),)))
    result = state.run(max_model_calls=len(actions) + 1, max_moves=2)
    assert result["reason"] == "decision_budget_exhausted" and result["moves"] == 1
    assert result["model_calls"] == len(actions) + 1 and not result["task_complete"]
    assert state.session.spec.raw_instruction == state.original_spec.raw_instruction
    assert state.session.search_revision == expected_search_revision
    assert state.session.spec.revision == expected_task_revision
    assert state.motion.control_epoch == 0 and state.motion.active_request_id is None
    assert state.tools.movement_budget_remaining == 1
    assert [call.args[0] for call in state.tools.dispatch.call_args_list] == [
        "query_viewpoints", "submit_motion", "observe"]
    goal, = state.controller.goals.values()
    assert goal.target_object_uid == "second" and goal.goal_pose.position == (2., 0., 1.5)
    assert goal.metadata["task_revision"] == expected_task_revision
    assert state.tools.view_generation == 1
    assert goal.metadata["viewpoint_uid"] == "second:view:1:1"
    assert sum(action == "move" for action, _ in client.commands) == 1
    state.tools.evidence_provider.assert_not_called()
    state.tools.query_scene.assert_not_called()


def test_final_prediction_is_bound_to_revised_verified_target(monkeypatch, tmp_path):
    from instruction.contracts import TaskSpec, TargetSpec

    def revise(state, context):
        return _action("revise_interpretation", interpretation_update={
            "interpretation": {"targets": [{"name": "sofa"}]}, "basis_ids": ["instruction"],
            "reason": "Use the original sitting requirement, not the initial category guess"})

    def verify(state, context):
        target = state.session.spec.targets[0]
        return _action("verify_target", candidate_uid="second", target_id=target.id,
                       task_revision=state.session.spec.revision)

    state = _replay_episode(monkeypatch, tmp_path, [revise, verify, _action("propose_finish")],
        spec=TaskSpec("Find somewhere to sit.", (TargetSpec("t1", "chair"),)))
    result = state.run(max_model_calls=4)
    assert result["reason"] == "verified_finish" and result["task_complete"]
    assert result["initial_predicted_target"] == "chair" and result["predicted_target"] == "sofa"
    assert result["prediction_task_revision"] == result["final_task"]["revision"] == 2
    assert result["predicted_target_id"] == state.session.spec.targets[0].id
    assert result["completion_review"]["verified_target_id"] == result["predicted_target_id"]
    assert result["prediction_source"] == "current_full_instruction_verification"
    assert result["moves"] == 0


@pytest.mark.parametrize("precheck_satisfied", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_constraint_precheck_is_forwarded_to_fulltext_verifier(tmp_path, precheck_satisfied, accepted):
    from backends.isaac.runtime import IsaacEvidenceVerifier
    from instruction_adapter.constraints import ConstraintEvaluation
    from verification.contracts import VerificationResult

    obj = ObjectNodeSnapshot("candidate", "chair", (1., 0., 1.5))
    evaluator = SimpleNamespace(evaluate_before_final_verifier=Mock())
    session = SimpleNamespace(spec=SimpleNamespace(revision=2), constraint_evaluator=evaluator,
        snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 1.5)), objects=(obj,)))
    plan = SimpleNamespace(terminal_targets=[SimpleNamespace(id="t1")])
    evidence = {"target_id": "t1", "observation_status": "current",
                "geometry": {"distance_to_object": 1., "required_stop_distance": 2.,
                             "projection_failed_in_final_view": False}}
    # The real evaluator mutates and returns the same evidence object.
    def precheck(**kwargs):
        kwargs["evidence"]["relation_edges"] = [{"state": "unknown", "reason": "anchor not observed"}]
        return ConstraintEvaluation(precheck_satisfied, failed_constraints=[] if precheck_satisfied else ["near desk"],
                                    reason="compiled relation precheck", evidence=kwargs["evidence"])
    evaluator.evaluate_before_final_verifier.side_effect = precheck
    verifier = IsaacEvidenceVerifier(SimpleNamespace(session=session), "offline", tmp_path)
    expected = VerificationResult(accepted, "accept" if accepted else "need_relation_check",
        task_revision=2, target_uid="candidate", semantic_satisfied=accepted, view_sufficient_for_stop=accepted)
    verifier.final.verify = Mock(return_value=expected)
    result = verifier.verify(raw_instruction="Find somewhere to sit.", instruction_plan=plan,
                             candidate=obj, evidence=evidence)
    assert result is expected
    evaluator.evaluate_before_final_verifier.assert_called_once()
    verifier.final.verify.assert_called_once_with("Find somewhere to sit.", plan, obj, evidence)
    checked = evidence["constraint_precheck"]
    assert checked["satisfied"] is precheck_satisfied
    assert checked["failed_constraints"] == ([] if precheck_satisfied else ["near desk"])
    assert checked["reason"] == "compiled relation precheck"
    assert checked["relation_edges"] == []
    assert evidence["relation_edges"] == [{"state": "unknown", "reason": "anchor not observed"}]
    assert "evidence" not in checked and "geometry" not in checked
    json.dumps(evidence, allow_nan=False)


def test_missing_fresh_projection_still_blocks_precheck_and_final(tmp_path):
    from backends.isaac.runtime import IsaacEvidenceVerifier

    evaluator = SimpleNamespace(evaluate_before_final_verifier=Mock())
    tools = SimpleNamespace(session=SimpleNamespace(spec=SimpleNamespace(revision=3), constraint_evaluator=evaluator))
    verifier = IsaacEvidenceVerifier(tools, "offline", tmp_path)
    verifier.final.verify = Mock(side_effect=AssertionError("must retain physical projection gate"))
    result = verifier.verify(raw_instruction="Find a chair.", instruction_plan=None,
        candidate=SimpleNamespace(uid="candidate"), evidence={"observation_status": "need_observation"})
    assert not result.accepted and result.decision == "need_better_view"
    assert result.task_revision == 3 and result.target_uid == "candidate"
    evaluator.evaluate_before_final_verifier.assert_not_called()
    verifier.final.verify.assert_not_called()


@pytest.mark.parametrize("failure", ["invalid", "compile_exception", "setup_exception"])
def test_initial_admission_failures_retain_calls_without_observation_or_motion(monkeypatch, tmp_path, failure):
    from backends.isaac import runtime
    from instruction.contracts import CompileResult, TargetSpec, TaskSpec
    from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES

    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    initial = SimpleNamespace(model_calls=0)

    def compile_initial(text, capabilities):
        assert text == public["instruction"] and capabilities == RUNTIME_CONSTRAINT_TYPES
        initial.model_calls = 2
        if failure == "compile_exception":
            raise ConnectionError("initial provider failed")
        if failure == "invalid":
            return CompileResult("invalid", diagnostics=("invalid target structure",))
        return CompileResult("success", TaskSpec(text, (TargetSpec("goal", "chair"),)))

    initial.compile = Mock(side_effect=compile_initial)
    factory = Mock(return_value=initial)
    client = FakeClient()
    observation_map = Mock(side_effect=RuntimeError("setup failed"))
    task_factory = Mock(side_effect=AssertionError("must not construct a task session"))
    motion_factory = Mock(side_effect=AssertionError("must not construct a motion controller"))
    monkeypatch.setattr(runtime, "InitialInstructionCompiler", factory)
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    monkeypatch.setattr(runtime, "IsaacObservationMap", observation_map)
    monkeypatch.setattr(runtime, "TaskSession", task_factory)
    monkeypatch.setattr(runtime, "IsaacMotionController", motion_factory)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "model_requests"))
    exception, message = {"invalid": (ValueError, "invalid: invalid target structure"),
                          "compile_exception": (ConnectionError, "initial provider failed"),
                          "setup_exception": (RuntimeError, "setup failed")}[failure]
    with pytest.raises(exception, match=message):
        runtime.run_episode(public, tmp_path, max_model_calls=5)
    factory.assert_called_once_with(max_model_calls=5,
        trace_path=tmp_path / "initial_interpretation.jsonl", vlm="self_hosted")
    initial.compile.assert_called_once()
    assert observation_map.call_count == int(failure == "setup_exception")
    task_factory.assert_not_called()
    motion_factory.assert_not_called()
    assert client.commands == [("STOP", {})]
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["status"] == "error" and result["reason"] == exception.__name__
    assert result["model_calls"] == 2 and result["moves"] == 0
    assert not result["task_complete"]
    assert "initial_interpretation" in result["model_calls_scope"]
    assert (tmp_path / "instruction_plan.json").exists() == (failure == "setup_exception")


@pytest.mark.parametrize("initial_calls", [1, 2, 3])
def test_episode_startup_calls_share_agent_tool_budget(monkeypatch, tmp_path, initial_calls):
    client = FakeClient()
    client.status = "reached"
    actions = [_action("submit_motion", task_revision=1, control_epoch=0,
                       request_id=f"request:{i}", frontier_uid="f:1") for i in range(3 - initial_calls)]
    state = _replay_episode(monkeypatch, tmp_path, actions, initial_calls=initial_calls, client=client)
    result = state.run(max_model_calls=3, max_moves=3)
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert result["model_calls"] == state.agent.model_calls == 3
    assert state.parse.call_count == result["moves"] == 3 - initial_calls
    assert sum(action == "move" for action, _ in client.commands) == 3 - initial_calls
    assert not result["task_complete"]
    assert client.commands[-2][0] == "finish"
    plan = json.loads((tmp_path / "instruction_plan.json").read_text())
    assert plan["diagnostics"]["initial_model_calls"] == initial_calls
