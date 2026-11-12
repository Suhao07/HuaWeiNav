import base64
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

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


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    """Fail before provider discovery, key lookup, network access or GPU setup."""
    for module in ("harness.runtime", "instruction_adapter.verifier",
                   "instruction_adapter.relation_verifier", "instruction_adapter.concept_matcher",
                   "llm_utils.cognav_llm_adapter"):
        monkeypatch.setattr(f"{module}.get_client_and_model",
                            Mock(side_effect=AssertionError("unexpected real model construction")))
    # 概念匹配同样声明离线边界：replay 只验证几何与调度，不允许暗中回落到真实模型。
    monkeypatch.setenv("LLM_OFFLINE", "1")
    monkeypatch.setattr("urllib.request.urlopen",
                        Mock(side_effect=AssertionError("unexpected network request")))
    monkeypatch.setattr("backends.isaac.runtime.IsaacObservationMap",
                        Mock(side_effect=AssertionError("unexpected GPU perception construction")))


class FakeExplorationMap:
    """Explicit online-map adapter; packet-provided frontiers are never policy input."""

    def __init__(self, positions=(), *, by_frame=None, viewpoints=(), rooms=()):
        self.nodes = {i: SimpleNamespace(idx=i, position=tuple(position), state=0)
                      for i, position in enumerate(positions, 1)}
        self.by_frame = by_frame or {}
        self.viewpoints, self.rooms = viewpoints, rooms
        self.added, self.refreshed, self.snapshot_objects, self.marked = [], [], [], []
        self.events = []
        self.unreachable = set()
        self.approaches = []
        self.approach_queries = []

    @property
    def candidates(self):
        return [{"node_idx": node.idx, "frontier_uid": f"node:{node.idx}",
                 "position": list(node.position), "raw_distance_m": float(node.idx),
                 "frontier_idxs": [[node.idx, 0]], "room_id": "room:1"}
                for node in self.nodes.values() if not node.state and node.idx not in self.unreachable]

    def begin_observation(self):
        self.events.append(("begin",))

    def finish_observation(self):
        self.events.append(("finish",))
        return True

    def arrive(self, node_index):
        self.events.append(("arrive", node_index))

    def add(self, packet):
        self.added.append(packet)
        self.events.append(("add", packet["frame"]["frame_id"]))
        positions = self.by_frame.get(packet["frame"]["frame_id"])
        if positions is not None:
            self.nodes = {i: SimpleNamespace(idx=i, position=tuple(position), state=0)
                          for i, position in enumerate(positions, 1)}

    def refresh(self, client, pose):
        self.refreshed.append((client, pose))
        self.events.append(("refresh", tuple(pose["position"])))
        return self.candidates

    def snapshot_fields(self, objects=()):
        self.snapshot_objects.append(objects)
        return {"rooms": self.rooms, "viewpoints": self.viewpoints,
                "frontiers": tuple(FrontierSnapshot(f"node:{node.idx}", node.position,
                    metadata={"frame_id": "map", "source": "observed_rgbd", "node_idx": node.idx})
                    for node in self.nodes.values() if not node.state)}

    def select(self, choose_room=None):
        rows = self.candidates
        return (self.nodes[rows[-1]["node_idx"]] if rows else None, "current_room", rows)

    def approach_targets(self, position):
        self.approach_queries.append(tuple(position))
        return [list(point) for point in self.approaches]

    def mark_observed(self, node_index):
        self.nodes[node_index].state = 1
        self.marked.append(node_index)
        self.events.append(("mark", node_index))


def _session(snapshot=None, spec=None):
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from task.session import TaskSession

    return TaskSession(snapshot or SemanticMapSnapshot(0., Pose3D((0., 0., 1.5))),
                       runtime_plan(spec or TaskSpec("Find a chair.", (TargetSpec("t1", "chair"),))))


def _agent(tools, output, *, max_model_calls=5, parse=None):
    from harness.runtime import AgentRuntime

    def choose(**kwargs):
        context = json.loads(kwargs["messages"][1]["content"][0]["text"])
        result = kwargs["response_format"](frontier_uid=context["baseline_recommendation"],
                                            reason="Continue searching")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))])

    parse_call = Mock(side_effect=parse or choose)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_call))))
    return AgentRuntime(tools, output / "tools.jsonl", max_model_calls=max_model_calls,
                        client_model=(model, "offline-replay")), parse_call


def _tools(session, client, output, observation_map=None, exploration_map=None):
    motion = MotionSupervisor(IsaacMotionController(client, client.pose))
    return IsaacNavigationTools(session, motion, observation_map, client, output, "self_hosted",
                                exploration_map if exploration_map is not None else FakeExplorationMap())


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

    replay = _episode_replay(monkeypatch, tmp_path, initial_calls=1)
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=2, max_moves=5,
                                 panorama_views=12, search_review_moves=1)
    replay.observation_factory.assert_called_once_with(runtime.BACKGROUND_CLASSES,
        replay.client.packets[0], replay.client, save_dir=tmp_path, device="cuda:0",
        floor_world_z=0.08, ceiling_world_z=1.68)
    assert replay.observation_factory.call_args.args[1] is replay.client.packets[0]
    assert replay.agents[0].tools.exploration_map is replay.observation_map.mapping is replay.exploration
    assert result["frontier_source"] == "original_instruct_mapper"
    assert "espresso machine" not in replay.observation_factory.call_args.args[0]
    assert "espresso machine" in replay.sessions[0].detector_terms
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert not result["task_complete"]
    # 最后一次决策接纳的移动仍须执行，初始基线移动单独计入移动预算。
    assert result["moves"] == 3 and result["model_calls"] == 2
    assert result["initial_model_calls"] == 1
    assert len([a for a, _ in replay.client.commands if a == "move"]) == 3
    assert replay.parse_calls[0].call_count == 2
    assert replay.client.commands[-2][0] == "finish"
    assert replay.agents[0].tools.motion.active_request_id is None
    snapshots = [json.loads(line) for line in (tmp_path / "scene_snapshots.jsonl").read_text().splitlines()]
    assert len(snapshots) >= 3
    assert snapshots[0]["robot_pose"] != snapshots[-1]["robot_pose"]
    assert all(row["metadata"]["captured_frames"] == 12 for row in snapshots)


def test_frontier_arrival_precedes_scan_and_mark_in_guided_loop(monkeypatch, tmp_path):
    from backends.isaac import runtime

    replay = _episode_replay(monkeypatch, tmp_path)
    # Exercise the default full twelve-view scan, not a partial panorama.
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=3, max_moves=1)
    start = tuple(replay.public["start_position"])
    destination = replay.exploration.nodes[9].position
    assert replay.exploration.events == [
        ("begin",), *[("add", i) for i in range(2, 14)], ("finish",),
        ("refresh", start), ("refresh", start),
        # Successful after_motion must not scan before the original mapper arrives.
        ("arrive", 9), ("begin",), *[("add", i) for i in range(14, 26)], ("finish",),
        ("refresh", destination), ("refresh", destination), ("mark", 9)]
    calls = replay.observation_map.add_panorama.call_args_list
    assert [[packet["frame"]["frame_id"] for packet in invocation.args[0]]
            for invocation in calls] == [list(range(start, start + 4)) for start in range(2, 26, 4)]
    assert all(invocation.args[1] is replay.sessions[0] for invocation in calls)
    assert replay.sessions[0].snapshot.metadata["captured_frames"] == 12
    assert replay.exploration.marked == [9] and replay.exploration.nodes[9].state == 1
    assert result["status"] == "completed" and result["reason"] == "movement_budget_exhausted"
    assert result["moves"] == 1 and not result["task_complete"]
    assert replay.agents[0].tools.motion.active_request_id is None
    replay.parse_calls[0].assert_not_called()


@pytest.mark.parametrize("after_arrival", [False, True])
def test_incomplete_full_scan_stops_without_marking_or_retrying(monkeypatch, tmp_path, after_arrival):
    from backends.isaac import runtime

    replay = _episode_replay(monkeypatch, tmp_path)
    outcomes = [True, False] if after_arrival else [False]
    replay.exploration.finish_observation = Mock(side_effect=outcomes)
    with pytest.raises(RuntimeError, match="Original mapper scan produced no navigable geometry"):
        runtime.run_episode(replay.public, tmp_path, max_model_calls=3, max_moves=10,
                            panorama_views=12)
    assert (("arrive", 9) in replay.exploration.events) is after_arrival
    assert replay.exploration.finish_observation.call_count == len(outcomes)
    assert replay.exploration.marked == [] and replay.exploration.nodes[9].state == 0
    assert replay.sessions[0].snapshot.metadata["mapping_cycle_completed"] is False
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["status"] == "error" and not result["task_complete"]
    assert result["moves"] == int(after_arrival)
    assert replay.client.commands[-1][0] == "STOP"
    assert not any(action == "finish" for action, _ in replay.client.commands)
    assert replay.agents[0].tools.motion.active_request_id is None
    replay.parse_calls[0].assert_not_called()


@pytest.mark.parametrize("role,status", [("primary", "reached"), ("primary", "blocked"), ("anchor", "reached")])
def test_guided_candidate_verification_and_recovery(tmp_path, role, status):
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.tools import NavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from verification.contracts import VerificationResult

    client = FakeClient()
    client.status = status
    target = TargetSpec("t1", "chair", role=role, terminal=role == "primary")
    targets = (target,) if role == "primary" else (target, TargetSpec("terminal", "sofa"))
    session = _session(SemanticMapSnapshot(0., pose_value(client.pose),
        objects=(ObjectNodeSnapshot("obj", "chair", (2., 0., 1.5)),),
        viewpoints=(ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("obj",)),),
        metadata={"covered_headings": list(range(12))}), TaskSpec("Find a chair.", targets))
    tools = _tools(session, client, tmp_path)
    motion, controller = tools.motion, tools.motion.controller
    tools.query_scene = Mock(return_value={"candidates": [{"candidate_uid": "obj", "target_id": "t1",
                                                           "role": role}], "reason": "observed"})
    tools.query_viewpoints = lambda uid: NavigationTools.query_viewpoints(tools, uid)
    evidence = {"observation_identity": {"observation_id": "fresh"}, "geometry": {}}
    tools.last_evidence = evidence
    tools.evidence_provider = Mock(return_value=evidence)
    tools.verifier = SimpleNamespace(verify=Mock(return_value=VerificationResult(
        True, "accept", semantic_satisfied=True, task_revision=1, target_uid="obj")))
    agent, _ = _agent(tools, tmp_path)
    guided = IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.scan_complete = True
    assert guided.select()
    assert not guided.view_attempted("obj", (1., 0., 1.5))
    controller.advance(motion.active_goal_id)
    feedback = motion.poll_active()
    guided.motion_finished(feedback)
    tools.verifier.verify.assert_not_called()  # Terminal bookkeeping is not verification.
    assert guided.candidate_motion_goals[(1, "obj", 0)] == {(1., 0., 1.5): status}
    if status == "reached":
        assert guided.verify(feedback.intent.goal_pose.position) == (
            "verified_finish" if role == "primary" else "select")
    else:
        assert guided.recover({"reason": "motion_terminal_failure", "candidate_uid": "obj"}) == "search_exhausted"
    accepted = role == "primary" and status == "reached"
    assert tools.verifier.verify.call_count == int(accepted)
    assert tools.evidence_provider.call_count == int(accepted)
    assert session.progress.completed == accepted
    assert tools.propose_finish()["accepted"] == accepted
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


def _frame_packet(root, frame_id, depth):
    rgb_path, depth_path = f"{frame_id}-rgb.png", f"{frame_id}-depth.npy"
    Image.new("RGB", (depth.shape[1], depth.shape[0]), (17, 33, 65)).save(root / rgb_path)
    np.save(root / depth_path, depth, allow_pickle=False)
    return {"capture_root": os.path.relpath(root),
            "frame": {"frame_id": frame_id, "rgb_path": rgb_path, "depth_path": depth_path,
                      "intrinsics": np.eye(3).tolist(), "camera_to_map": np.eye(4).tolist(),
                      "depth_convention": "optical_z", "depth_unit": "m",
                      "size_hw": list(depth.shape)},
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


class EpisodeClient(FakeClient):
    """Capture replay packets and execute bounded moves without a simulator."""

    def __init__(self, root):
        super().__init__()
        self.root, self.frame_id = root, 0
        self.packets = []
        self.status = "reached"

    def command(self, action, **arguments):
        if action == "observe":
            self.commands.append((action, arguments))
            self.frame_id += 1
            packet = _frame_packet(self.root, self.frame_id, np.ones((10, 10)))
            packet["pose"] = {"position": list(self.pose["position"]), "yaw": self.pose["yaw"]}
            camera = np.eye(4)
            c, s = np.cos(self.pose["yaw"]), np.sin(self.pose["yaw"])
            camera[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            packet["frame"]["camera_to_map"] = camera.tolist()
            packet["frontiers"] = [{"uid": "forbidden-official", "position": [99., 99., 1.5]}]
            self.packets.append(packet)
            return packet
        if action == "look":
            self.commands.append((action, arguments))
            self.pose = {"position": list(self.pose["position"]), "yaw": arguments["yaw"]}
            return {"pose": self.pose, "status": "reached"}
        if action == "move":
            self.commands.append((action, arguments))
            self.pose = {"position": list(arguments["position"]), "yaw": arguments["yaw"]}
            return {"pose": self.pose, "status": self.status, "zero_velocity": True,
                    "path_length_remaining": 0.}
        return super().command(action, **arguments)


def _episode_replay(monkeypatch, output, *, initial_calls=1, parse=None, spec=None):
    """Run the real session/tools/policy/loop with only sensor and model boundaries faked."""
    from backends.isaac import runtime
    from instruction.contracts import CompileResult, TaskSpec, TargetSpec
    from task.session import TaskSession

    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    spec = spec or TaskSpec(public["instruction"], (TargetSpec("t1", "chair"),),
                           search_priors={"destination_hypotheses": {"t1": ["espresso machine"]}})
    public["instruction"] = spec.raw_instruction
    initial = SimpleNamespace(model_calls=initial_calls, compile=Mock(return_value=CompileResult("success", spec)))
    monkeypatch.setattr(runtime, "InitialInstructionCompiler", Mock(return_value=initial))
    client = EpisodeClient(output)
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    observation_map = SimpleNamespace(objects=lambda: (), points=Mock(side_effect=AssertionError("no objects")),
        mapper=SimpleNamespace(initial_position=np.array(client.pose["position"]),
            current_obj_indices=[], objects=[], _is_target_tag=Mock(side_effect=lambda tag: tag == "chair")))
    observation_map.add_panorama = Mock(side_effect=lambda packets, session: [{
        "frame_id": packet["frame"]["frame_id"], "observation_image": {
            "id": str(packet["frame"]["frame_id"]),
            "image_ref": str(output / packet["frame"]["rgb_path"]),
            "observed_object_uids": [], "object_boxes": {}}} for packet in packets])
    observation_factory = Mock(return_value=observation_map)
    monkeypatch.setattr(runtime, "IsaacObservationMap", observation_factory)
    exploration = FakeExplorationMap([(float(i), 2., 1.5) for i in range(1, 10)])
    observation_map.mapping = exploration
    sessions, agents, parse_calls = [], [], []

    def make_session(snapshot, plan, **kwargs):
        assert kwargs["constraint_evaluator"].relation_service.vlm_callback.vlm == "self_hosted"
        session = TaskSession(snapshot, plan, **kwargs)
        sessions.append(session)
        return session

    def make_agent(tools, trace_path, **kwargs):
        agent, parse_call = _agent(tools, output, max_model_calls=kwargs["max_model_calls"], parse=parse)
        assert agent.model_calls == 0
        agents.append(agent)
        parse_calls.append(parse_call)
        return agent

    monkeypatch.setattr(runtime, "TaskSession", make_session)
    monkeypatch.setattr(runtime, "AgentRuntime", make_agent)
    return SimpleNamespace(public=public, initial=initial, client=client, observation_map=observation_map,
                           observation_factory=observation_factory, exploration=exploration,
                           sessions=sessions, agents=agents, parse_calls=parse_calls)


def _sam(*boxes):
    """Authored current-frame instance regions for the replayed geometry, one mask per [u1, v1, u2, v2)."""

    def segment(image, label):
        masks = np.zeros((len(boxes), image.shape[0], image.shape[1]), dtype=bool)
        for index, (u1, v1, u2, v2) in enumerate(boxes):
            masks[index, v1:v2, u1:u2] = True
        return masks

    return SimpleNamespace(sam=SimpleNamespace(segment_current_target=Mock(side_effect=segment)))


def _surface_map():
    points, _, _, _ = projection_fixture()
    surfaces = {"candidate": points, "anchor": points + [2., 0., 0.],
                "occluded": points * 2, "behind": -points,
                "outside": points + [20., 0., 0.], "sparse": points[:9]}
    objects = tuple(ObjectNodeSnapshot(uid, "chair", (2., 0., 1.5)) for uid in surfaces)
    return SimpleNamespace(objects=lambda: objects, points=Mock(side_effect=surfaces.__getitem__),
        mapper=SimpleNamespace(initial_position=np.array([0., 0., 1.5]), current_obj_indices=[],
            objects=[SimpleNamespace(uid=obj.uid, tag=obj.label) for obj in objects],
            object_perceiver=_sam((2, 3, 7, 7)),
            _is_target_tag=Mock(side_effect=lambda tag: tag == "chair")),
        add_panorama=Mock(side_effect=AssertionError("unexpected detector call")))


def _tensor(value):
    value = np.asarray(value)
    result = SimpleNamespace(shape=value.shape, numpy=lambda: value)
    result.cpu = lambda: result
    return result


def _cloud(points, *args):
    result = SimpleNamespace(point=SimpleNamespace(positions=_tensor(points)))
    result.voxel_down_sample = lambda voxel: result
    return result


@pytest.mark.parametrize("target_list", [None, ["espresso machine", "chair"]])
def test_add_panorama_reports_only_current_fusion_bindings(monkeypatch, tmp_path, target_list):
    from instruction_adapter.contracts import InstructionPlan, TargetQuery
    from mapping_utils.representation import ObjectNode
    from task.session import TaskSession

    packet = _frame_packet(tmp_path, 41, np.ones((10, 10)))
    camera = np.eye(4)
    camera[:3, 3] = [3., -2., 4.]
    packet["frame"]["camera_to_map"] = camera.tolist()
    points, _, _, _ = projection_fixture()
    origin = np.array([10., -20., 2.])
    surfaces = {"unseen": points * 2, "fused": points,
                "missing-image": -points, "fused:new": points + [2., 0., 0.]}
    entities, originals = [], {}
    for uid, surface in surfaces.items():
        local = surface + camera[:3, 3] - origin
        entity = ObjectNode(_cloud(local), _cloud(local), "chair", .9,
                            local.mean(axis=0), len(local), None, [0, 0, 1, 1])
        entity.uid = uid
        entities.append(entity)
        originals[uid] = local.copy()
    observation_map = object.__new__(IsaacObservationMap)
    observation_map.mapping = SimpleNamespace(intrinsic=np.eye(3), shape=(10, 10), _step=7)
    observation_map.mapper = SimpleNamespace(objects=entities[:3], initial_position=origin,
        current_obj_indices=[], update_object_pcd=Mock(return_value=object()))
    old = {"image_ref": str(tmp_path / "old.png"), "frame_id": 5, "bbox": [0, 0, 1, 1]}
    observation_map.latest = {"unseen": dict(old), "fused": dict(old)}
    session = TaskSession(SemanticMapSnapshot(0., Pose3D((0., 0., 1.5))),
        InstructionPlan("Find a chair.", targets=[TargetQuery(
            "goal", "chair", detector_terms=target_list or [])]))
    active = session.plan.terminal_targets[0]

    def process(adapter, images, depths, positions, rotations, *, partial, project_view):
        assert adapter is observation_map and adapter.task is session and adapter.vlm == session.vlm
        assert adapter.episode_steps == 7
        assert adapter.mapper.target == active.name
        assert adapter.mapper.target_list == (target_list or [active.name])
        assert adapter.mapper.target_aliases == list(active.match_terms)
        assert len(images) == len(depths) == len(positions) == len(rotations) == 1
        np.testing.assert_array_equal(images[0][0, 0], [17, 33, 65])
        np.testing.assert_array_equal(positions[0], camera[:3, 3] - origin)
        np.testing.assert_array_equal(rotations[0], np.diag([1., -1., -1.]))
        assert partial is True and callable(project_view)
        # The shared pipeline owns fusion; the adapter must report the fused UIDs,
        # never synthetic per-detection identities or stale image boxes.
        adapter.mapper.objects = entities
        adapter.mapper.current_obj_indices = [1, 3]
        count = 1 if np.any(depths[0]) else 0
        return [{"background_detections": 3 * count, "target_detections": count}]

    pipeline = Mock(side_effect=process)
    monkeypatch.setitem(sys.modules, "navigation.observation_pipeline", SimpleNamespace(
        process_panoramic_detections=pipeline))
    monkeypatch.setitem(sys.modules, "mapping.object_fusion", SimpleNamespace(
        label_probabilities=lambda entity: {"chair": 1.}))
    report, = observation_map.add_panorama([packet], session)
    assert (report["detected"], report["map_objects"], report["frontend"]) == (4, 4, "original_panoramic_bc")
    image = report["observation_image"]
    assert image == {"id": "41", "image_ref": str(tmp_path / "41-rgb.png"),
        "observed_object_uids": ["fused", "fused:new"],
        "object_boxes": {"fused": [2, 3, 7, 7], "fused:new": [4, 3, 9, 7]}}
    objects = {obj.uid: obj for obj in observation_map.objects()}
    for uid in ("fused", "fused:new"):
        assert objects[uid].metadata["frame_id"] == 41
        assert objects[uid].image_ref == image["image_ref"]
        assert objects[uid].bbox2d_xyxy == tuple(image["object_boxes"][uid])
    assert observation_map.latest["unseen"] == old
    assert objects["missing-image"].image_ref is None
    assert objects["missing-image"].bbox2d_xyxy is None
    assert objects["missing-image"].metadata["frame_id"] is None
    empty_packet = _frame_packet(tmp_path, 57, np.zeros((10, 10)))
    empty_packet["frame"]["camera_to_map"] = camera.tolist()
    empty, = observation_map.add_panorama([empty_packet], session)
    assert empty["detected"] == 0
    assert empty["observation_image"] == {"id": "57", "image_ref": str(tmp_path / "57-rgb.png"),
                                         "observed_object_uids": [], "object_boxes": {}}
    assert observation_map.objects() == tuple(objects.values())
    assert pipeline.call_count == 2
    assert observation_map.mapper.current_obj_indices == [1, 3]
    assert observation_map.mapper.update_object_pcd.call_args_list == [call(), call()]
    assert observation_map.mapper.object_pcd is observation_map.mapper.update_object_pcd.return_value
    for entity in entities:
        np.testing.assert_array_equal(entity.pcd.point.positions.cpu().numpy(), originals[entity.uid])
        np.testing.assert_array_equal(entity.pcd_all.point.positions.cpu().numpy(), originals[entity.uid])
        np.testing.assert_array_equal(entity.position, originals[entity.uid].mean(axis=0))
    np.testing.assert_array_equal(origin, [10., -20., 2.])
    json.dumps([report, empty], allow_nan=False)


def test_observation_objects_and_points_return_world_without_mutating_local_cloud(monkeypatch):
    from mapping_utils.representation import ObjectNode

    local = np.array([[-2., 1., .5], [4., 3., 2.5], [1., 2., 1.5]])
    original = local.copy()
    origin = np.array([10., -20., 3.])
    entity = ObjectNode(_cloud(local), _cloud(local), "chair", .9,
                        local.mean(axis=0), len(local), None, [1, 2, 6, 8])
    entity.uid = "chair:1"
    observation_map = object.__new__(IsaacObservationMap)
    observation_map.mapper = SimpleNamespace(objects=[entity], initial_position=origin,
        current_obj_indices=[], associate_object_entities=Mock(), update_object_pcd=Mock())
    observation_map.latest = {entity.uid: {"bbox": entity.bbox, "image_ref": "frame.png", "frame_id": 7}}
    monkeypatch.setitem(sys.modules, "mapping.object_fusion", SimpleNamespace(
        label_probabilities=lambda item: {"chair": 1.}))

    for _ in range(2):
        snapshot, = observation_map.objects()
        assert snapshot.position == tuple(original.mean(axis=0) + origin)
        assert snapshot.bbox3d_center == tuple((original.min(axis=0) + original.max(axis=0)) / 2 + origin)
        assert snapshot.bbox3d_extent == tuple(original.max(axis=0) - original.min(axis=0))
        world = observation_map.points(entity.uid)
        np.testing.assert_array_equal(world, original + origin)
        assert not np.shares_memory(world, local)
        world[:] = 0.
    np.testing.assert_array_equal(entity.pcd.point.positions.cpu().numpy(), original)
    np.testing.assert_array_equal(entity.pcd_all.point.positions.cpu().numpy(), original)
    np.testing.assert_array_equal(entity.position, original.mean(axis=0))
    np.testing.assert_array_equal(origin, [10., -20., 3.])
    observation_map.mapper.associate_object_entities.assert_not_called()
    observation_map.mapper.update_object_pcd.assert_not_called()


@pytest.mark.parametrize("panorama_views", [0, 1, 3, 4, 5, 12])
def test_refresh_publishes_only_this_batch_and_depth_visible_final_frame(tmp_path, panorama_views):
    observation_map = _surface_map()
    packets = [_frame_packet(tmp_path, 40 + i * 7, np.ones((10, 10)))
               for i in range(max(1, panorama_views))]
    start_yaw = .7
    reports = []
    for i, packet in enumerate(packets):
        packet["pose"]["yaw"] = start_yaw - i * np.pi / 6
        if i < len(packets) - 1:
            packet["frame"]["camera_to_map"] = [[0., -1., 0., float(i)], [1., 0., 0., 2.],
                                                  [0., 0., 1., 1.2], [0., 0., 0., 1.]]
        boxes = {} if i % 2 else {"anchor": [4, 3, 9, 7]}
        reports.append({"frame_id": packet["frame"]["frame_id"], "observation_image": {
            "id": str(packet["frame"]["frame_id"]),
            "image_ref": str(tmp_path / packet["frame"]["rgb_path"]),
            "observed_object_uids": list(boxes), "object_boxes": boxes}})
    batches = [packets[i:i + 4] for i in range(0, panorama_views, 4)]
    processed = 0

    def add_panorama(batch, active_session):
        nonlocal processed
        assert batch == packets[processed:processed + 4] and active_session is session
        processed += len(batch)
        assert exploration.added == packets[:processed]
        assert sum(action == "observe" for action, _ in client.commands) == processed
        exploration.events.append(("perception_batch", tuple(p["frame"]["frame_id"] for p in batch)))
        return reports[processed - len(batch):processed]

    observation_map.add_panorama = Mock(side_effect=add_panorama)
    old_images = [{"id": "old", "image_ref": str(tmp_path / "old.png"),
                   "observed_object_uids": ["occluded"], "object_boxes": {"occluded": [0, 0, 1, 1]}}]
    session = SimpleNamespace(detector_terms=["espresso machine", "table"],
        snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 1.5)),
                                     metadata={"observation_images": old_images}))
    client = FrameClient(packets)
    client.pose["yaw"] = start_yaw
    controller = IsaacMotionController(client, client.pose)
    exploration = FakeExplorationMap([(3., 2., 1.5)])
    covered = set()
    last_packet = refresh_observations(session, observation_map, controller, client, tmp_path,
                                      panorama_views, exploration, covered_headings=covered)
    assert last_packet is packets[-1]
    assert exploration.added == packets
    expected_events = [("begin",)]
    for batch in batches:
        expected_events.extend(("add", p["frame"]["frame_id"]) for p in batch)
        expected_events.append(("perception_batch", tuple(p["frame"]["frame_id"] for p in batch)))
    if not panorama_views:
        expected_events.append(("add", packets[0]["frame"]["frame_id"]))
    assert exploration.events == expected_events + [
        ("finish",), ("refresh", tuple(packets[-1]["pose"]["position"]))]
    assert exploration.refreshed == [(client, packets[-1]["pose"])]
    assert exploration.snapshot_objects == [observation_map.objects()]
    assert session.snapshot.frontiers == exploration.snapshot_fields()["frontiers"]
    assert session.snapshot.metadata["captured_frames"] == max(1, panorama_views)
    assert session.snapshot.metadata["perception_frames"] == panorama_views
    assert session.snapshot.metadata["frontier_source"] == "original_instruct_mapper"
    assert session.snapshot.metadata["mapping_cycle_completed"] is True
    assert session.snapshot.metadata["panorama_complete"] is (panorama_views == 12)
    assert session.snapshot.metadata["perception_frontend"] == "original_panoramic_bc"
    assert session.snapshot.metadata["observation_schedule"] == ("adaptive4" if panorama_views else "geometry_only")
    assert len(covered) == panorama_views
    images = session.snapshot.metadata["observation_images"]
    assert len(images) == len({image["id"] for image in images}) == max(1, panorama_views)
    assert images[:-1] == [{**report["observation_image"],
        "position": np.asarray(packet["frame"]["camera_to_map"])[:3, 3].tolist(),
        "rotation": np.asarray(packet["frame"]["camera_to_map"])[:3, :3].tolist()}
        for packet, report in zip(packets[:-1], reports)]
    assert images[-1] == {"id": str(packets[-1]["frame"]["frame_id"]),
        "image_ref": str(tmp_path / packets[-1]["frame"]["rgb_path"]),
        "position": [0., 0., 0.], "rotation": np.eye(3).tolist(),
        "observed_object_uids": ["candidate", "anchor"],
        "object_boxes": {"candidate": [2, 3, 7, 7], "anchor": [4, 3, 9, 7]}}
    assert old_images[0]["id"] == "old"
    assert observation_map.add_panorama.call_args_list == [call(batch, session) for batch in batches]
    assert [invocation.args[0] for invocation in observation_map.points.call_args_list] == [
        obj.uid for obj in observation_map.objects()]
    assert [action for action, _ in client.commands] == ["observe"] + ["look", "observe"] * (len(packets) - 1)
    assert [args["yaw"] for action, args in client.commands if action == "look"] == pytest.approx(
        [start_yaw - index * np.pi / 6 for index in range(1, panorama_views)])
    assert controller.pose is packets[-1]["pose"]
    assert controller.pose["yaw"] == pytest.approx(start_yaw - (len(packets) - 1) * np.pi / 6)
    assert json.loads((tmp_path / "scene_snapshots.jsonl").read_text())["metadata"]["observation_images"] == images

    observation_map.add_panorama = Mock(side_effect=AssertionError("unexpected detector call"))
    exploration.events.clear()
    previous_coverage = covered.copy()
    client.packets = iter([_frame_packet(tmp_path, 199, np.zeros((10, 10)))])
    before = len(client.commands)
    last_packet = refresh_observations(session, observation_map, controller, client, tmp_path, 0, exploration,
                                      covered_headings=covered)
    assert last_packet["frame"]["frame_id"] == 199
    assert client.commands[before:] == [("observe", {})]
    assert covered == previous_coverage
    assert session.snapshot.metadata["covered_headings"] == sorted(previous_coverage)
    assert exploration.events == [("begin",), ("add", 199), ("finish",),
                                  ("refresh", tuple(last_packet["pose"]["position"]))]
    assert session.snapshot.metadata["captured_frames"] == 1
    assert session.snapshot.metadata["perception_frames"] == 0
    assert session.snapshot.metadata["observation_images"] == [{
        "id": "199", "image_ref": str(tmp_path / "199-rgb.png"),
        "position": [0., 0., 0.], "rotation": np.eye(3).tolist(),
        "observed_object_uids": [], "object_boxes": {}}]
    observation_map.add_panorama.assert_not_called()


@pytest.mark.parametrize("panorama_views", [2, 12])
def test_refresh_forwards_replaced_terms_to_actual_observation_map(monkeypatch, tmp_path, panorama_views):
    """Retrieval revisions reopen semantic coverage without resetting the scan heading."""
    from backends.isaac.runtime import IsaacGuidedExecution, IsaacGuidedNavigation
    from instruction_adapter.contracts import InstructionPlan, TargetQuery
    from task.session import TaskSession

    points, _, _, _ = projection_fixture()
    objects = (ObjectNodeSnapshot("historical", "chair", (2., 0., 1.5)),)
    origin, camera_position = np.array([10., -20., 2.]), np.array([3., -2., 4.])
    local = points - origin
    historical = [SimpleNamespace(uid="historical", tag="chair", alive=True, pcd=_cloud(local))]
    observation_map = object.__new__(IsaacObservationMap)
    observation_map.mapping = SimpleNamespace(intrinsic=np.eye(3), shape=(10, 10), _step=7)
    observation_map.mapper = SimpleNamespace(objects=historical, initial_position=origin,
        current_obj_indices=[], _is_target_tag=Mock(side_effect=lambda tag: tag == "reading spot"),
        update_object_pcd=Mock(return_value=object()))
    observation_map.latest = {}
    observation_map.objects = lambda: objects
    observation_map.points = lambda uid: points
    session = TaskSession(SemanticMapSnapshot(0., Pose3D((0., 0., 1.5))),
        InstructionPlan("Find a reading spot.", targets=[TargetQuery(
            "goal", "reading spot", detector_terms=["espresso machine", "table"])]))
    spec, ledger, revision = session.spec, session.verification_ledger, session.search_revision
    depth = np.ones((10, 10))
    depth[0, :3] = [float("nan"), -1., float("inf")]
    packets = [_frame_packet(tmp_path, i, depth) for i in range(2 * panorama_views)]
    optical_rotations = []
    for index, packet in enumerate(packets):
        slot = index % panorama_views
        yaw = -slot * np.pi / 6
        packet["pose"]["yaw"] = yaw
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array(
            [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        optical_rotations.append(rotation)
        camera = np.eye(4)
        camera[:3, :3], camera[:3, 3] = rotation, camera_position
        packet["frame"]["camera_to_map"] = camera.tolist()
    terms_seen = []
    processed = 0

    def process(adapter, images, depths, positions, rotations, *, partial, project_view):
        nonlocal processed
        assert adapter is observation_map and adapter.task is session and adapter.vlm == session.vlm
        assert adapter.mapper.target == "reading spot" and adapter.episode_steps == 7
        assert adapter.mapper.target_aliases == list(session.plan.terminal_targets[0].match_terms)
        terms_seen.append((list(adapter.mapper.target_list), session.search_revision))
        batch_size = min(4, panorama_views - processed % panorama_views)
        assert len(images) == len(depths) == len(positions) == len(rotations) == batch_size
        assert partial is True and callable(project_view)
        clean_depth = np.ones((10, 10))
        clean_depth[0, :3] = 0.
        for image, depth_value, position, rotation, optical in zip(
                images, depths, positions, rotations, optical_rotations[processed:processed + batch_size]):
            np.testing.assert_array_equal(image[0, 0], [17, 33, 65])
            np.testing.assert_array_equal(depth_value, clean_depth)
            np.testing.assert_array_equal(position, camera_position - origin)
            np.testing.assert_array_equal(rotation, optical @ np.diag([1., -1., -1.]))
        processed += batch_size
        assert exploration.added == packets[:processed]
        return [{"background_detections": 0, "target_detections": 0} for _ in images]

    pipeline = Mock(side_effect=process)
    monkeypatch.setitem(sys.modules, "navigation.observation_pipeline", SimpleNamespace(
        process_panoramic_detections=pipeline))
    client = FrameClient(packets)
    exploration = FakeExplorationMap()
    tools = _tools(session, client, tmp_path, observation_map, exploration)
    tools.query_scene = Mock(return_value={"candidates": []})
    guided = IsaacGuidedNavigation(tools, SimpleNamespace(), tmp_path)
    guided.record = Mock()
    execution = IsaacGuidedExecution(guided, max_moves=10, panorama_views=panorama_views)
    execution.observe(panorama_views)
    first_snapshot = session.snapshot
    scan = execution.current_scan
    scan['node_index'] = 9
    exploration.nodes[9] = SimpleNamespace(idx=9, position=(0., 0., 1.5), state=0)
    session.revise_search([{"concept_id": "goal", "detector_terms": ["bookshelf"]}],
        task_revision=session.spec.revision, search_revision=session.search_revision,
        snapshot=session.snapshot)
    assert guided.select(None)
    assert execution.current_scan is scan
    assert scan['search_revision'] == session.search_revision
    assert exploration.marked == ([9] if panorama_views == 12 else [])
    assert exploration.added == packets
    assert session.spec is spec and session.verification_ledger is ledger
    assert session.search_revision == revision + 1
    batches_per_scan = (panorama_views + 3) // 4
    assert terms_seen == ([(["espresso machine", "table"], revision)] * batches_per_scan
                          + [(["bookshelf"], revision + 1)] * batches_per_scan)
    assert pipeline.call_count == 2 * batches_per_scan
    assert processed == len(packets)
    assert first_snapshot.objects is objects and session.snapshot.objects is objects
    assert observation_map.mapper.objects is historical and historical[0].tag == "chair"
    assert observation_map.mapper.current_obj_indices == []
    assert observation_map.mapper.update_object_pcd.call_args_list == [call()] * (2 * batches_per_scan)
    assert observation_map.mapper.object_pcd is observation_map.mapper.update_object_pcd.return_value
    np.testing.assert_array_equal(historical[0].pcd.point.positions.cpu().numpy(), local)
    np.testing.assert_array_equal(origin, [10., -20., 2.])
    for scan, snapshot in enumerate((first_snapshot, session.snapshot)):
        assert snapshot.metadata["perception_frames"] == panorama_views
        assert snapshot.metadata["captured_frames"] == panorama_views
        assert snapshot.metadata["panorama_complete"] is (panorama_views == 12)
        assert snapshot.metadata["covered_headings"] == list(range(panorama_views))
        images = snapshot.metadata["observation_images"]
        assert [image["id"] for image in images] == [
            str(i) for i in range(scan * panorama_views, (scan + 1) * panorama_views)]
        final_packet = packets[(scan + 1) * panorama_views - 1]
        assert snapshot.robot_pose == pose_value(final_packet["pose"])
        assert images[-1]["image_ref"] == str(tmp_path / final_packet["frame"]["rgb_path"])
        assert images[-1]["position"] == camera_position.tolist()
        assert images[-1]["rotation"] == optical_rotations[(scan + 1) * panorama_views - 1].tolist()


@pytest.mark.parametrize("observe_elsewhere", [False, True], ids=["same-position", "other-position-first"])
@pytest.mark.parametrize("refresh_fails", [False, True])
def test_guided_scan_resumes_fixed_yaw_slots_and_marks_only_completed_arrival(
        tmp_path, observe_elsewhere, refresh_fails):
    from backends.isaac.runtime import IsaacGuidedExecution, IsaacGuidedNavigation
    from instruction.contracts import TaskSpec, TargetSpec

    start_yaw = np.pi / 4
    position = np.array([12.34567, -18.76543, 3.5])
    origin = np.array([10., -20., 2.])
    count = 24 if observe_elsewhere else 12
    packets = [_frame_packet(tmp_path, 300 + i,
                            np.zeros((10, 10)) if i == count - 1 else np.ones((10, 10)))
               for i in range(count)]

    def optical_rotation(yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array(
            [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])

    class PoseTrackingClient(FrameClient):
        """Capture the commanded pose, not a pre-scripted yaw that could hide bad looks."""

        def command(self, action, **arguments):
            if action == "look":
                self.pose = {**self.pose, "yaw": arguments["yaw"]}
            packet = super().command(action, **arguments)
            if action == "observe":
                packet["pose"] = {"position": list(self.pose["position"]), "yaw": self.pose["yaw"]}
                camera = np.eye(4)
                camera[:3, :3] = optical_rotation(self.pose["yaw"])
                camera[:3, 3] = self.pose["position"]
                packet["frame"]["camera_to_map"] = camera.tolist()
            return packet

    observation_map = _surface_map()
    observation_map.mapper.initial_position = origin
    local_points = observation_map.points.side_effect
    first_final_rotation = optical_rotation(start_yaw - 3 * np.pi / 6)
    observation_map.points.side_effect = lambda uid: local_points(uid) @ first_final_rotation.T + position
    session = _session(spec=TaskSpec("Find a chair.", (TargetSpec("t1", "chair"),), revision=7))
    client = PoseTrackingClient(packets)
    client.pose = {"position": position.tolist(), "yaw": start_yaw}
    exploration = FakeExplorationMap([position])
    tools = _tools(session, client, tmp_path, observation_map, exploration)
    controller = tools.motion.controller
    guided = IsaacGuidedNavigation(tools, SimpleNamespace(), tmp_path)
    execution = IsaacGuidedExecution(guided, max_moves=10, panorama_views=12)
    batches = []

    def add_panorama(batch, active_session):
        assert active_session is session and len(batch) == 4
        assert exploration.marked == []
        assert exploration.added[-4:] == batch
        batches.append(list(batch))
        # Only the first fused batch finds a candidate; later batches allow completion.
        observation_map.mapper.current_obj_indices = [0] if len(batches) == 1 else []
        # Deliberately stale boxes must not survive fresh final-frame depth projection.
        return [{"frame_id": packet["frame"]["frame_id"], "observation_image": {
            "id": str(packet["frame"]["frame_id"]),
            "image_ref": str(tmp_path / packet["frame"]["rgb_path"]),
            "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [2, 3, 7, 7]}}}
            for packet in batch]

    observation_map.add_panorama = Mock(side_effect=add_panorama)
    feedback = SimpleNamespace(intent=SimpleNamespace(metadata={"node_index": 1}))
    key = (7, 2.346, 1.235, 1.5)
    execution.frontier_arrived(feedback)
    execution.frontier_observed(feedback)
    scan = execution.semantic_scan_headings[key]
    assert scan == {"start_yaw": start_yaw, "covered": set(range(4)), "node_index": 1,
                    "search_revision": session.search_revision}
    assert guided.scan_complete is False and exploration.marked == []
    assert exploration.nodes[1].state == 0
    assert exploration.events[0] == ("arrive", 1)
    assert tools.last_packet is packets[3]
    assert session.snapshot.metadata["perception_frames"] == session.snapshot.metadata["captured_frames"] == 4
    assert session.snapshot.metadata["panorama_complete"] is False
    assert session.snapshot.metadata["covered_headings"] == list(range(4))
    first_images = session.snapshot.metadata["observation_images"]
    assert [image["id"] for image in first_images] == [str(300 + i) for i in range(4)]
    assert first_images[-1]["observed_object_uids"] == ["candidate", "anchor"]
    assert [action for action, _ in client.commands] == ["observe"] + ["look", "observe"] * 3
    assert [args["yaw"] for action, args in client.commands if action == "look"] == pytest.approx(
        [start_yaw - i * np.pi / 6 for i in range(1, 4)])

    if observe_elsewhere:
        other_yaw = np.deg2rad(23.)
        client.pose = {"position": (position + [1., 0., 0.]).tolist(), "yaw": other_yaw}
        controller.pose = client.pose
        execution.observe(12)
        other_key = (7, 3.346, 1.235, 1.5)
        assert execution.semantic_scan_headings[other_key] == {
            "start_yaw": other_yaw, "covered": set(range(12)), "search_revision": session.search_revision}
        assert guided.scan_complete is True and exploration.marked == []
        assert exploration.nodes[1].state == 0
        assert scan == {"start_yaw": start_yaw, "covered": set(range(4)), "node_index": 1,
                    "search_revision": session.search_revision}
        assert [packet["pose"]["yaw"] for packet in packets[4:16]] == pytest.approx(
            [other_yaw - i * np.pi / 6 for i in range(12)])
        client.pose = {"position": position.tolist(), "yaw": client.pose["yaw"]}

    # Candidate viewing may turn the camera away before resuming at the same position.
    controller.pose = client.command("look", yaw=np.deg2rad(17.))["pose"]
    before = len(client.commands)
    execution.observe(12)
    assert [action for action, _ in client.commands[before:]] == ["look", "observe"] * 8
    assert [args["yaw"] for action, args in client.commands[before:] if action == "look"] == pytest.approx(
        [start_yaw - i * np.pi / 6 for i in range(4, 12)])
    same_position = packets[:4] + packets[-8:]
    assert [packet["pose"]["yaw"] for packet in same_position] == pytest.approx(
        [start_yaw - i * np.pi / 6 for i in range(12)])
    assert all(packet["pose"]["position"] == position.tolist() for packet in same_position)
    assert execution.semantic_scan_headings[key] is scan
    assert guided.scan_complete is True
    assert exploration.marked == [] and scan['node_index'] == 1
    guided.candidate_view_states[(7, "candidate")] = object()

    def context(uid):
        exploration.events.append(("evidence", uid))
        if refresh_fails:
            raise ValueError("evidence refresh failed")
        return set()

    guided.observation_context = context
    if refresh_fails:
        with pytest.raises(ValueError, match="evidence refresh failed"):
            guided.observation_updated()
        assert exploration.marked == [] and scan['node_index'] == 1
        assert exploration.nodes[1].state == 0
        return
    guided.observation_updated()
    assert exploration.events[-2:] == [("evidence", "candidate"), ("mark", 1)]
    assert scan == {"start_yaw": start_yaw, "covered": set(range(12)),
                    "search_revision": session.search_revision}
    assert exploration.marked == [1] and exploration.nodes[1].state == 1
    assert exploration.events[-1] == ("mark", 1)
    assert len(batches) == (6 if observe_elsewhere else 3)
    assert [packet for batch in batches for packet in batch] == packets == exploration.added
    assert tools.last_packet is packets[-1]
    assert session.snapshot.metadata["perception_frames"] == session.snapshot.metadata["captured_frames"] == 8
    assert session.snapshot.metadata["panorama_complete"] is True
    assert session.snapshot.metadata["covered_headings"] == list(range(12))
    images = session.snapshot.metadata["observation_images"]
    assert [image["id"] for image in images] == [str(packet["frame"]["frame_id"]) for packet in packets[-8:]]
    assert images[-1] == {"id": str(packets[-1]["frame"]["frame_id"]),
        "image_ref": str(tmp_path / packets[-1]["frame"]["rgb_path"]),
        "position": position.tolist(), "rotation": optical_rotation(start_yaw - 11 * np.pi / 6).tolist(),
        "observed_object_uids": [], "object_boxes": {}}
    assert first_images[-1]["observed_object_uids"] == ["candidate", "anchor"]


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
    exploration = FakeExplorationMap([(7., 8., 1.5)] if anchor_visible else [])
    old_images = [{"id": "panorama", "image_ref": str(tmp_path / "old.png"),
                   "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [0, 0, 1, 1]}}]
    previous = SemanticMapSnapshot(11., Pose3D((0., 0., 1.5)), objects=observation_map.objects(),
        viewpoints=(ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("candidate",)),),
        frontiers=(FrontierSnapshot("stale", (4., 0., 1.5)),), source="isaac_rgbd_strive_fusion",
        metadata={"observation_images": old_images, "frame_revision": 7, "panorama_views": 3,
                  "semantic_gt_used": False, "frontier_source": "original_instruct_mapper"})
    from instruction.contracts import TaskSpec, TargetSpec
    from scene.dependencies import object_dependencies
    session = _session(previous, TaskSpec("Find a chair.", (TargetSpec("t1", "chair"),), revision=4))
    tools = _tools(session, client, tmp_path, observation_map, exploration)
    dependencies = object_dependencies(previous, "candidate", "anchor")
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
        "position": [0., 0., 0.], "rotation": np.eye(3).tolist(),
        "observed_object_uids": list(boxes), "object_boxes": boxes}]
    snapshot = tools.session.snapshot
    assert snapshot == replace(previous, timestamp=74., robot_pose=pose_value(packet["pose"]),
        rooms=exploration.rooms, frontiers=exploration.snapshot_fields()["frontiers"],
        metadata={**previous.metadata, "observation_images": evidence["observation_images"],
                  "panorama_views": 1, "captured_frames": 1, "perception_frames": 0})
    assert snapshot.objects is previous.objects and snapshot.viewpoints is previous.viewpoints
    assert object_dependencies(snapshot, "candidate", "anchor") == dependencies
    assert exploration.added == [packet] and exploration.refreshed == [(client, packet["pose"])]
    assert exploration.events == [("begin",), ("add", 73), ("finish",),
                                  ("refresh", tuple(packet["pose"]["position"]))]
    assert exploration.snapshot_objects[0] is previous.objects
    assert tools.last_packet is packet and tools.last_evidence is evidence
    assert not {"current", "stale"} & {frontier.uid for frontier in snapshot.frontiers}
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
    assert evidence["geometry"]["distance_source"] == "observed_instance_position_xy"
    assert evidence["geometry"]["stop_reference_position"] == [2., 0., 1.5]
    assert evidence["geometry"]["observation_identity"] == evidence["observation_identity"]
    expected_distance = np.linalg.norm(np.array([2., 0.]) - packet["pose"]["position"][:2])
    assert evidence["geometry"]["distance_to_object"] == (pytest.approx(expected_distance)
                                                          if candidate_visible else None)
    surface_distance = np.linalg.norm(np.array([2., 3.]) - packet["pose"]["position"][:2])
    assert evidence["geometry"]["visible_surface_distance"] == (pytest.approx(surface_distance)
                                                               if candidate_visible else None)
    assert evidence["geometry"]["visible_surface_distance_source"] == "depth_visible_surface_xy"
    assert evidence["geometry"]["projection_failed_in_final_view"] == (not candidate_visible)
    assert project.call_count == len(observation_map.objects())
    assert [call.args[0] for call in observation_map.points.call_args_list] == [
        obj.uid for obj in observation_map.objects()]
    observation_map.add_panorama.assert_not_called()
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
def test_guided_frontier_uses_same_current_packet_after_verification(tmp_path, candidate_visible, frontier_count):
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.runtime import AgentRuntime
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from task.session import TaskSession
    from verification.contracts import VerificationResult

    observation_map = _surface_map()
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
    observation_map.add_panorama = Mock(return_value=[{"frame_id": 47, "observation_image": {
        "id": "47", "image_ref": str(tmp_path / refreshed["frame"]["rgb_path"]),
        "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [2, 3, 7, 7]}}}])
    client = FrameClient([refreshed, current])
    client.status = "reached"
    controller = IsaacMotionController(client, client.pose)
    motion = MotionSupervisor(controller)
    view = ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("candidate",))
    session = TaskSession(SemanticMapSnapshot(0., pose_value(client.pose),
        objects=observation_map.objects(), viewpoints=(view,)),
        runtime_plan(TaskSpec("Find a chair.", (TargetSpec("t1", "chair"),))))
    online_positions = [(float(i), 4., 1.5) for i in range(1, frontier_count + 1)]
    exploration = FakeExplorationMap([(9., 9., 1.5)], by_frame={73: online_positions}, viewpoints=(view,))
    tools = IsaacNavigationTools(session, motion, observation_map, client, tmp_path, "cognav", exploration)
    tools.evidence_provider = Mock(wraps=tools.capture_evidence)
    # Even a visible target can need further search; never invoke a real verifier model.
    tools.verifier.final.verify = Mock(return_value=VerificationResult(False, "need_better_view",
        task_revision=session.spec.revision, target_uid="candidate", view_sufficient_for_stop=False))
    tools.query_scene = Mock(return_value={"candidates": [], "reason": "no_untried_candidate_view"})

    def parse(**kwargs):
        choice = kwargs["response_format"](frontier_uid=f"node:{frontier_count}", reason="Continue searching")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=choice))])

    parse_call = Mock(side_effect=parse)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_call))))
    agent = AgentRuntime(tools, tmp_path / "tools.jsonl", client_model=(model, "offline-replay"))
    from instruction_adapter.view_control import ViewControlState
    guided = IsaacGuidedNavigation(tools, agent, tmp_path, search_review_moves=1)
    guided.scan_complete = True
    guided.selected = {"candidate_uid": "candidate", "target_id": "t1", "role": "primary"}
    tools.view_states[(1, "candidate")] = ViewControlState()
    guided.submit(view.pose.position, candidate_uid="candidate", source="baseline_viewpoint")
    controller.advance(motion.active_goal_id)
    feedback = motion.poll_active()
    assert not feedback.tracking_active
    tools.last_packet = refresh_observations(session, observation_map, controller, client, tmp_path, 1, exploration)
    assert tools.last_packet is refreshed
    assert [image["id"] for image in session.snapshot.metadata["observation_images"]] == ["47"]
    guided.motion_finished(feedback)
    tools.evidence_provider.assert_not_called()
    assert guided.verify(view.pose.position) == "recovery"
    assert not session.progress.completed
    tools.evidence_provider.assert_called_once_with("candidate")
    assert tools.verifier.final.verify.call_count == int(candidate_visible)
    assert guided.select()

    parse_call.assert_called_once()
    content = parse_call.call_args.kwargs["messages"][1]["content"]
    context = json.loads(content[0]["text"])
    observation = context["observation"]
    evidence = json.loads((tmp_path / "verification" / "000073-evidence.json").read_text())
    assert evidence["observation_status"] == ("current" if candidate_visible else "need_observation")
    assert observation["robot_pose"] == pose_value(current["pose"]).as_dict()
    expected_frontiers = json.loads(json.dumps([asdict(item) for item in exploration.snapshot_fields()["frontiers"]]))
    table = observation["frontiers"]
    projected_frontiers = [dict(zip(table["columns"], row)) for row in table["rows"]]
    assert projected_frontiers == [
        {**{key: item[key] for key in ("uid", "position", "room_id", "score")},
         "frame_id": item["metadata"]["frame_id"]} for item in expected_frontiers]
    assert [item["frontier_uid"] for item in context["options"]] == [
        f"node:{i}" for i in range(1, frontier_count + 1)]
    assert context["baseline_recommendation"] == f"node:{frontier_count}"
    assert context["event"]["reason"] == "prolonged_target_search"
    assert exploration.added == [refreshed, current]
    assert all(not item["uid"].startswith(("current:", "stale:")) for item in projected_frontiers)
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
    assert rows[-1]["frontiers"] == expected_frontiers
    assert rows[-1]["metadata"]["observation_images"] == evidence["observation_images"]
    observation_map.add_panorama.assert_called_once_with([refreshed], session)
    assert observation_map.points.call_count == 2 * len(observation_map.objects())
    assert [action for action, _ in client.commands] == ["move", "observe", "look", "observe"]
    assert agent.model_calls == 1 and controller.moves == 1
    goal = controller.goals[motion.active_goal_id]
    assert goal.mode == MotionGoalMode.GO_TO_FRONTIER
    assert goal.goal_pose.position == online_positions[-1]
    assert motion.active_intent.metadata["task_revision"] == session.spec.revision


@pytest.mark.parametrize("outcome", ["installed", "unchanged", "invalid", "guard_refused"])
def test_guided_interpretation_handoff_preserves_physical_history(tmp_path, outcome):
    """Revision scopes attempts anew, without deleting old evidence or reopening blocked paths."""
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.tools import NavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from instruction_adapter.view_control import ViewControlState

    client = FakeClient()
    spec = TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),))
    view = ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("obj",))
    snapshot = SemanticMapSnapshot(12., pose_value(client.pose),
        objects=(ObjectNodeSnapshot("obj", "sofa", (2., 0., 1.5)),), viewpoints=(view,))
    session = _session(snapshot, spec)
    exploration = FakeExplorationMap([(1., 2., 1.5), (2., 2., 1.5)])
    exploration.unreachable.add(2)
    tools = _tools(session, client, tmp_path, exploration_map=exploration)
    motion, controller = tools.motion, tools.motion.controller
    controller.moves = 7
    tools.query_viewpoints = lambda uid: NavigationTools.query_viewpoints(tools, uid)
    queried = []

    def query():
        active = session.progress.active_target()
        queried.append((session.spec.revision, active.name))
        return {"candidates": [{"candidate_uid": "obj", "target_id": active.id,
                                "label": active.name, "role": "primary"}], "reason": "observed"}

    tools.query_scene = query
    runtime, _ = _agent(tools, tmp_path)
    runtime.model_calls = 2
    no_choice = outcome == "invalid"

    def choose(options, recommended, reason):
        runtime.model_calls += 1
        assert [option["frontier_uid"] for option in options] == ["node:1"]
        assert recommended == "node:1" and motion.active_request_id is None
        assert reason["reason"] == "prolonged_target_search"
        if outcome == "installed":
            revised = replace(spec, revision=2, targets=(TargetSpec("goal", "sofa"),))
            assert tools.install_interpretation(runtime_plan(revised), task_revision=1,
                search_revision=session.search_revision, snapshot=session.snapshot,
                record={"control_epoch": motion.control_epoch, "reason": "new interpretation"})
            return None
        return None if no_choice else recommended

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

    runtime.choose_frontier = Mock(side_effect=choose)
    runtime.choice_guard = Mock(side_effect=choice_guard)
    guided = IsaacGuidedNavigation(tools, runtime, tmp_path, search_review_moves=1)
    guided.scan_complete, guided.search_review_step = True, 0
    original_submit = guided.submit

    def submit(position, **kwargs):
        if kwargs.get("node") is not None:
            assert guard_events == ["enter"]
        return original_submit(position, **kwargs)

    guided.submit = Mock(side_effect=submit)
    guided.sequence = 4
    attempts = guided.candidate_motion_goals
    attempts[(1, "obj", 0)] = {view.pose.position: "blocked"}
    old_state = ViewControlState()
    guided.candidate_view_states[(1, "obj")] = old_state
    blocked = exploration.unreachable
    if outcome == "guard_refused":
        with pytest.raises(ValueError, match="choice refused"):
            guided.select()
    else:
        assert guided.select()
    assert attempts is guided.candidate_motion_goals
    assert attempts == {(1, "obj", 0): {view.pose.position: "blocked"}}
    assert guided.deferred_evidence == {(1, "obj"): {None}}
    assert guided.candidate_view_states[(1, "obj")] is old_state
    if outcome == "installed":
        assert session.spec.revision == 2 and guided.sequence == 4
        assert motion.active_request_id is None and controller.goals == {}
        guided.submit.assert_not_called()
        assert guided.deferred_candidates == set()
        assert not guided.view_attempted("obj", view.pose.position)
        assert guided.select()
        assert queried == [(1, "chair"), (1, "chair"), (2, "sofa")]
        assert guided.selected["label"] == "sofa"
        assert guided.candidate_view_states[(2, "obj")] is not old_state
        assert motion.active_intent.target_object_uid == "obj"
        assert motion.active_intent.metadata["task_revision"] == 2
    else:
        assert session.spec == spec and queried == [(1, "chair"), (1, "chair")]
        assert guided.deferred_candidates == {"obj"}
        if outcome == "unchanged":
            assert motion.active_intent.metadata == {"task_revision": 1, "frontier_uid": "node:1",
                                                     "node_index": 1, "decision_source": "agent_search_review"}
        else:
            assert motion.active_request_id is None and controller.goals == {}
            guided.submit.assert_not_called()
    submitted = int(outcome in {"installed", "unchanged"})
    assert guided.sequence == 4 + submitted and len(controller.goals) == submitted
    if outcome in {"unchanged", "guard_refused"}:
        runtime.choice_guard.assert_called_once_with()
        assert guard_events == ["enter", "exit"]
    else:
        runtime.choice_guard.assert_not_called()
        assert guard_events == []
    assert exploration.unreachable is blocked and blocked == {2}
    assert exploration.marked == []
    assert controller.moves == 7 and client.commands == []
    assert controller.pose is client.pose and session.snapshot.robot_pose == snapshot.robot_pose
    assert runtime.model_calls == 3 and runtime.max_model_calls == 5
    runtime.choose_frontier.assert_called_once()
    events = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "submitted" for event in events) == submitted
    assert sum(event["event"] == "interpretation_changed" for event in events) == int(outcome == "installed")
    assert sum(event["event"] == "decision_reselect" for event in events) == int(no_choice)
    if no_choice:
        assert events[-1] == {"event": "decision_reselect", "moves": 7, "task_revision": 1,
                              "decision_source": "agent_search_review"}


@pytest.mark.parametrize("failure", ["invalid", "compile_exception", "setup_exception"])
def test_initial_admission_failures_retain_calls_without_motion(monkeypatch, tmp_path, failure):
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
    client = EpisodeClient(tmp_path)
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
    factory.assert_called_once_with(max_model_calls=2,
        trace_path=tmp_path / "initial_interpretation.jsonl", vlm="self_hosted")
    initial.compile.assert_called_once()
    assert observation_map.call_count == int(failure == "setup_exception")
    task_factory.assert_not_called()
    motion_factory.assert_not_called()
    assert client.commands == ([("observe", {})] if failure == "setup_exception" else []) + [("STOP", {})]
    if failure == "setup_exception":
        observation_map.assert_called_once_with(runtime.BACKGROUND_CLASSES, client.packets[0], client,
            save_dir=tmp_path, device="cuda:0", floor_world_z=0.08, ceiling_world_z=1.68)
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["status"] == "error" and result["reason"] == exception.__name__
    assert result["model_calls"] == result["moves"] == 0
    assert result["initial_model_calls"] == 2
    assert not result["task_complete"]
    assert "guided_selection_and_review_only" in result["model_calls_scope"]
    assert (tmp_path / "instruction_plan.json").exists() == (failure == "setup_exception")


@pytest.mark.parametrize("initial_calls", [1, 2])
@pytest.mark.parametrize("budget", [1, 3])
def test_episode_startup_calls_are_separate_from_guided_budget(monkeypatch, tmp_path, initial_calls, budget):
    from backends.isaac import runtime

    replay = _episode_replay(monkeypatch, tmp_path, initial_calls=initial_calls)
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=budget, max_moves=6,
                                 panorama_views=12, search_review_moves=1)
    runtime.InitialInstructionCompiler.assert_called_once_with(max_model_calls=2,
        trace_path=tmp_path / "initial_interpretation.jsonl", vlm="self_hosted")
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert result["model_calls"] == replay.agents[0].model_calls == replay.parse_calls[0].call_count == budget
    assert result["initial_model_calls"] == initial_calls
    assert "guided_selection_and_review_only" in result["model_calls_scope"]
    # 初始基线移动不消耗决策调用，已接纳的复查动作执行完再判断预算。
    assert result["moves"] == 1 + budget
    assert sum(action == "move" for action, _ in replay.client.commands) == result["moves"]
    assert not result["task_complete"]
    assert replay.client.commands[-2][0] == "finish"
    assert replay.agents[0].tools.motion.active_request_id is None
    plan = json.loads((tmp_path / "instruction_plan.json").read_text())
    assert plan["diagnostics"]["initial_model_calls"] == initial_calls


@pytest.mark.parametrize("revision", [1, 4])
def test_fixed_task_skips_initial_calls_and_locks_whole_episode(monkeypatch, tmp_path, revision):
    import hashlib
    from backends.isaac import runtime
    from instruction.contracts import TargetSpec, TaskSpec

    task = TaskSpec("Find a chair.", (TargetSpec("chair", "chair"),), revision=revision)
    replay = _episode_replay(monkeypatch, tmp_path, initial_calls=0, spec=task)
    forbidden = Mock(side_effect=AssertionError("fixed task must not recompile"))
    monkeypatch.setattr(runtime, "InitialInstructionCompiler", forbidden)
    monkeypatch.setattr(runtime, "compile_instruction_plan", forbidden)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task.as_dict()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=3, max_moves=6,
        panorama_views=12, search_review_moves=1, task_spec_path=path, task_spec_sha256=digest)
    forbidden.assert_not_called()
    replay.initial.compile.assert_not_called()
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert result["initial_model_calls"] == 0 and result["model_calls"] == 3
    assert replay.parse_calls[0].call_count == 3 and result["moves"] == 4
    assert result["task_spec_sha256"] == digest and result["interpretation_locked"] is True
    assert result["task_revision"] == revision and "guided_selection_and_review_only" in result["model_calls_scope"]
    assert replay.sessions[0].interpretation_locked and replay.sessions[0].spec == task
    assert replay.sessions[0].interpretation_history == []
    assert not (tmp_path / "initial_interpretation.jsonl").exists()
    plan = json.loads((tmp_path / "instruction_plan.json").read_text())
    assert plan["diagnostics"]["initial_model_calls"] == 0
    assert plan["diagnostics"]["task_spec"] == task.as_dict()
    assert replay.client.commands[-2][0] == "finish"


@pytest.mark.parametrize("case", ["hash", "text", "path_only", "hash_only"])
def test_fixed_task_rejection_precedes_all_runtime_side_effects(monkeypatch, tmp_path, case):
    import hashlib
    from backends.isaac import runtime
    from instruction.contracts import TargetSpec, TaskSpec

    task = TaskSpec("Find a chair.", (TargetSpec("chair", "chair"),))
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task.as_dict()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": task.raw_instruction, "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    if case == "text":
        public["instruction"] = "Find a table."
    forbidden = Mock(side_effect=AssertionError("invalid task must fail before runtime setup"))
    for name in ("IsaacClient", "InitialInstructionCompiler", "IsaacObservationMap", "set_trace_dir"):
        monkeypatch.setattr(runtime, name, forbidden)
    with pytest.raises(ValueError, match="mismatch|supplied together"):
        runtime.run_episode(public, tmp_path, task_spec_path=None if case == "hash_only" else path,
            task_spec_sha256=None if case == "path_only" else ("0" * 64 if case == "hash" else digest))
    forbidden.assert_not_called()
    assert not (tmp_path / "policy_result.json").exists()


def test_fixed_task_setup_failure_records_zero_initial_calls(monkeypatch, tmp_path):
    import hashlib
    from backends.isaac import runtime
    from instruction.contracts import TargetSpec, TaskSpec

    task = TaskSpec("Find a chair.", (TargetSpec("chair", "chair"),))
    replay = _episode_replay(monkeypatch, tmp_path, initial_calls=0, spec=task)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task.as_dict()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "IsaacObservationMap", Mock(side_effect=RuntimeError("perception unavailable")))
    with pytest.raises(RuntimeError, match="perception unavailable"):
        runtime.run_episode(replay.public, tmp_path, task_spec_path=path, task_spec_sha256=digest)
    replay.initial.compile.assert_not_called()
    assert replay.client.commands == [("observe", {}), ("STOP", {})]
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["model_calls"] == result["initial_model_calls"] == result["moves"] == 0
    assert result["task_spec_sha256"] == digest and result["interpretation_locked"] is True
    assert result["status"] == "error" and not result["task_complete"]


@pytest.mark.parametrize("option", ["--task-spec", "--task-spec-sha256"])
def test_policy_cli_rejects_partial_fixed_input_before_loading(monkeypatch, tmp_path, capsys, option):
    from backends.isaac import runtime
    from scripts import run_intentionnav, serve_intentionnav

    forbidden = Mock(side_effect=AssertionError("partial fixed input cannot start runtime"))
    monkeypatch.setattr(serve_intentionnav, "load_public_episode", forbidden)
    monkeypatch.setattr(runtime, "run_episode", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_intentionnav.py", "--public-episode", str(tmp_path / "public.json"),
                                    "--output", str(tmp_path), option, "unused"])
    with pytest.raises(SystemExit) as error:
        run_intentionnav.main()
    assert error.value.code == 2
    assert "--task-spec and --task-spec-sha256 must be supplied together" in capsys.readouterr().err
    forbidden.assert_not_called()
    assert not (tmp_path / "policy_result.json").exists()


@pytest.mark.parametrize("fixed", [False, True])
def test_policy_cli_runs_bounded_episode_and_refuses_output_reuse(monkeypatch, tmp_path, capsys, fixed):
    import hashlib
    from instruction.contracts import TargetSpec, TaskSpec
    from scripts import run_intentionnav

    task = TaskSpec("Find a chair.", (TargetSpec("chair", "chair"),), revision=4)
    replay = _episode_replay(monkeypatch, tmp_path, initial_calls=0 if fixed else 1, spec=task)
    public = tmp_path / "public.json"
    public.write_text(json.dumps(replay.public))
    argv = ["run_intentionnav.py", "--public-episode", str(public), "--output", str(tmp_path),
            "--max-model-calls", "3", "--max-moves", "6", "--panorama-views", "12",
            "--search-review-moves", "1"]
    digest = None
    if fixed:
        path = tmp_path / "task.json"
        path.write_text(json.dumps(task.as_dict()))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        argv += ["--task-spec", str(path), "--task-spec-sha256", digest]
    monkeypatch.setattr(sys, "argv", argv)
    run_intentionnav.main()
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    receipt = tmp_path / "policy_result.json"
    before = receipt.read_bytes()
    assert result == json.loads(before)
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert result["task_spec_sha256"] == digest and result["interpretation_locked"] is fixed
    assert result["initial_model_calls"] == (0 if fixed else 1) and result["model_calls"] == 3
    assert result["moves"] == 4
    assert replay.sessions[0].spec == task and result["task_revision"] == 4
    assert replay.initial.compile.call_count == (0 if fixed else 1)
    commands = list(replay.client.commands)
    with pytest.raises(SystemExit) as error:
        run_intentionnav.main()
    assert error.value.code == 2 and "fresh run directory" in capsys.readouterr().err
    assert receipt.read_bytes() == before and replay.client.commands == commands
