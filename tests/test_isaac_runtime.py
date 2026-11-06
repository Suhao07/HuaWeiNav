import json
from types import SimpleNamespace

import numpy as np
import pytest

from backends.isaac.perception import visible_projection
from backends.isaac.runtime import IsaacClient, IsaacMotionController, pose_value
from motion.contracts import MotionGoal, MotionGoalMode, NavigationIntent, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from scene.contracts import Pose3D


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


def test_last_model_call_motion_executes_before_budget_stop(monkeypatch, tmp_path):
    from backends.isaac import runtime

    client = FakeClient()
    client.status = "reached"
    from harness.tools import NavigationTools
    from scene.contracts import FrontierSnapshot, SemanticMapSnapshot

    plan = SimpleNamespace(as_dict=lambda: {}, concept_queries=[], terminal_targets=[SimpleNamespace(name="chair")])
    progress = SimpleNamespace(completed=False, as_dict=lambda: {"completed": False})
    session = SimpleNamespace(progress=progress, spec=SimpleNamespace(revision=1),
        snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 1.5)), frontiers=tuple(
            FrontierSnapshot(f"f:{i}", (float(i), 0., 1.5), score=float(i),
                             metadata={"frame_id": "map", "explored_density": 0.}) for i in (1, 2))))
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    monkeypatch.setattr(runtime, "compile_instruction_plan", lambda *a, **k: plan)
    monkeypatch.setattr(runtime, "IsaacObservationMap", lambda *a, **k: object())
    def task_session(*args, **kwargs):
        assert kwargs["constraint_evaluator"].relation_service.vlm_callback.vlm == "self_hosted"
        return session
    monkeypatch.setattr(runtime, "TaskSession", task_session)
    def make_tools(s, m, *args):
        tools = NavigationTools(s, m, None, None)
        tools.query_scene = lambda: {"candidates": [], "reason": "no_observed_candidate"}
        return tools
    monkeypatch.setattr(runtime, "IsaacNavigationTools", make_tools)
    refreshes = []
    monkeypatch.setattr(runtime, "refresh_observations", lambda *a: refreshes.append(client.pose.copy()))
    class LastCallAgent:
        def __init__(self, tools, *a, **k):
            self.tools, self.model_calls = tools, 0
        def choose_frontier(self, options, recommended_uid, reason):
            self.model_calls += 1
            return recommended_uid
    monkeypatch.setattr(runtime, "AgentRuntime", LastCallAgent)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "model_requests"))
    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    result = runtime.run_episode(public, tmp_path, max_model_calls=1, max_moves=3)
    assert result["status"] == "completed"
    assert result["reason"] == "decision_budget_exhausted"
    assert not result["task_complete"]
    assert result["moves"] == 1
    assert any(action == "move" for action, _ in client.commands)
    assert client.commands[-2][0] == "finish"
    assert len(refreshes) == 2 and refreshes[0] != refreshes[1]


@pytest.mark.parametrize("role,status", [("primary", "reached"), ("primary", "blocked"), ("anchor", "reached")])
def test_guided_candidate_verification_and_recovery(tmp_path, role, status):
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.tools import NavigationTools
    from instruction_adapter.concept_matcher import AnchorSearchLedger
    from scene.contracts import ObjectNodeSnapshot, SemanticMapSnapshot, ViewpointSnapshot

    client = FakeClient()
    client.status = status
    controller = IsaacMotionController(client, client.pose)
    motion = MotionSupervisor(controller)
    progress = SimpleNamespace(completed=False)
    session = SimpleNamespace(spec=SimpleNamespace(revision=1, raw_instruction="Find a chair."),
        progress=progress, anchor_search_ledger=AnchorSearchLedger(), snapshot=SemanticMapSnapshot(
            0., Pose3D((0., 0., 1.5)), objects=(ObjectNodeSnapshot("obj", "chair", (2., 0., 1.5)),),
            viewpoints=(ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("obj",)),)))
    tools = NavigationTools(session, motion, None, None)
    tools.query_scene = lambda: {"candidates": [{"candidate_uid": "obj", "target_id": "t1",
                                                 "role": role}], "reason": "observed"}
    verified = []
    def verify(*args):
        verified.append(args)
        progress.completed = True
        return {"decision": "accept"}
    tools.verify_target = verify
    guided = IsaacGuidedNavigation(tools, SimpleNamespace(model_calls=0), tmp_path)
    assert guided.select()
    controller.advance(motion.active_goal_id)
    guided.motion_finished(motion.poll_active())
    assert bool(verified) == (role == "primary" and status == "reached")
    assert progress.completed == bool(verified)
    if role == "anchor":
        assert session.anchor_search_ledger.is_blocked("Find a chair.", "t1", "obj")
    assert not guided.select()
    assert motion.active_request_id is None


@pytest.mark.parametrize("revision,epoch", [(1, 1), (0, 0)])
def test_version_rejection_preserves_guard_and_allows_corrected_proposal(revision, epoch):
    from backends.isaac.runtime import IsaacNavigationTools
    from scene.contracts import FrontierSnapshot, SemanticMapSnapshot

    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    tools = object.__new__(IsaacNavigationTools)
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
