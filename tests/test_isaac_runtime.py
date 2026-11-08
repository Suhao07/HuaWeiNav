import base64
from contextlib import contextmanager, nullcontext
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


@pytest.mark.parametrize("settings,mode,configured,effective", [
    ({}, "single", 4, 1),
    ({"observation_mode": "single", "panorama_views": 7}, "single", 7, 1),
    ({"observation_mode": "single", "panorama_views": 0}, "single", 0, 1),
    ({"observation_mode": "multi"}, "multi", 4, 4),
    ({"observation_mode": "multi", "panorama_views": 3}, "multi", 3, 3),
])
def test_last_model_call_motion_executes_before_budget_stop(
        monkeypatch, tmp_path, settings, mode, configured, effective):
    from backends.isaac import runtime

    client = FakeClient()
    command = client.command
    move_statuses = iter(["running", "reached"])

    def advance_command(action, **arguments):
        if action == "move":
            client.status = next(move_statuses)
        return command(action, **arguments)

    client.command = advance_command
    from harness.tools import NavigationTools
    from scene.contracts import FrontierSnapshot, SemanticMapSnapshot

    plan = SimpleNamespace(as_dict=lambda: {},
        concept_queries=[SimpleNamespace(detector_terms=["espresso machine"])],
        terminal_targets=[SimpleNamespace(name="chair")])
    progress = SimpleNamespace(completed=False, as_dict=lambda: {"completed": False})
    session = SimpleNamespace(progress=progress, spec=SimpleNamespace(revision=1),
        snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 1.5)), frontiers=tuple(
            FrontierSnapshot(f"f:{i}", (float(i), 0., 1.5), score=float(i),
                             metadata={"frame_id": "map", "explored_density": 0.}) for i in (1, 2))))
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    monkeypatch.setattr(runtime, "compile_instruction_plan", lambda *a, **k: plan)
    observation_map_factory = Mock(return_value=object())
    monkeypatch.setattr(runtime, "IsaacObservationMap", observation_map_factory)
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
    monkeypatch.setattr(runtime, "refresh_observations", lambda *a, **k: refreshes.append(
        (client.pose.copy(), a[-1], k["observation_mode"])))
    class LastCallAgent:
        def __init__(self, tools, *a, **k):
            self.tools, self.model_calls = tools, 0
        def choose_frontier(self, options, recommended_uid, reason):
            self.model_calls += 1
            return recommended_uid
        def choice_guard(self):
            return nullcontext()
    monkeypatch.setattr(runtime, "AgentRuntime", LastCallAgent)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "model_requests"))
    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    original_public = dict(public)
    result = runtime.run_episode(public, tmp_path, max_model_calls=1, max_moves=3, **settings)
    assert public == original_public and len(public) == 6
    assert result["observation_mode"] == mode
    assert result["panorama_views"] == configured
    assert result["effective_views"] == effective
    assert json.loads((tmp_path / "policy_result.json").read_text()) == result
    observation_map_factory.assert_called_once_with(runtime.BACKGROUND_CLASSES, device="cuda:0")
    assert "espresso machine" not in observation_map_factory.call_args.args[0]
    assert result["status"] == "completed"
    assert result["reason"] == "decision_budget_exhausted"
    assert not result["task_complete"]
    assert result["moves"] == 2
    assert sum(action == "move" for action, _ in client.commands) == 2
    assert client.commands[-2][0] == "finish"
    assert len(refreshes) == 3 and refreshes[0][0] != refreshes[1][0]
    assert [(views, selected) for _, views, selected in refreshes] == [
        (effective, mode), (1, mode), (effective, mode)]


@pytest.mark.parametrize("settings,message", [
    ({"observation_mode": "panorama"}, "observation_mode"),
    ({"observation_mode": "multi", "panorama_views": 0}, "panorama_views"),
    ({"observation_mode": "multi", "panorama_views": 1}, "panorama_views"),
    ({"observation_mode": "multi", "panorama_views": -1}, "panorama_views"),
    ({"observation_mode": "multi", "panorama_views": 2.5}, "panorama_views"),
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


@pytest.mark.parametrize("panorama_views,observation_mode", [(0, None), (1, None), (2, None), (3, None), (1, "multi")])
def test_refresh_publishes_only_this_batch_and_depth_visible_final_frame(
        tmp_path, capsys, panorama_views, observation_mode):
    observation_map = _surface_map()
    image_count = 1 if panorama_views == 1 else panorama_views + 1
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
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views,
                         observation_mode=observation_mode)
    images = session.snapshot.metadata["observation_images"]
    assert len(images) == image_count
    assert images[:-1] == ([report["observation_image"] for report in reports] if panorama_views >= 2 else [])
    assert controller.pose is packets[-1]["pose"]
    assert session.snapshot.robot_pose == pose_value(packets[-1]["pose"])
    assert session.snapshot.frontiers == (FrontierSnapshot(**packets[-1]["frontiers"][0]),)
    assert session.snapshot.objects is observation_map.objects()
    selected_mode = observation_mode or ("multi" if panorama_views >= 2 else "single")
    assert session.snapshot.metadata["observation_mode"] == selected_mode
    assert session.snapshot.metadata["effective_views"] == panorama_views
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
    if panorama_views != 1:
        for i in range(panorama_views):
            expected_commands.extend([("look", {"yaw": start_yaw + 2 * np.pi * i / panorama_views}),
                                      ("observe", {})])
        expected_commands.append(("look", {"yaw": start_yaw}))
    expected_commands.append(("observe", {}))
    assert client.commands == expected_commands
    logged = [json.loads(line) for line in (tmp_path / "perception.jsonl").read_text().splitlines()] if panorama_views else []
    assert logged == [{**report, "observation_mode": selected_mode, "effective_views": panorama_views}
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


@pytest.mark.parametrize("panorama_views", [1, 2])
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
    images_per_refresh = 1 if panorama_views == 1 else panorama_views + 1
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
    if panorama_views == 1:
        assert client.commands == [("observe", {})] * 2
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
    observation_map.add = Mock(return_value={"frame_id": 47, "observation_image": {
        "id": "47", "image_ref": str(tmp_path / refreshed["frame"]["rgb_path"]),
        "observed_object_uids": ["candidate"], "object_boxes": {"candidate": [2, 3, 7, 7]}}})
    client = FrameClient([refreshed, current])
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

    def parse(**kwargs):
        choice = kwargs["response_format"](frontier_uid=f"current:{frontier_count}", reason="Continue searching")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=choice))])

    parse_call = Mock(side_effect=parse)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_call))))
    agent = AgentRuntime(tools, tmp_path / "tools.jsonl", client_model=(model, "offline-replay"))
    guided = IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.submit(candidate={"candidate_uid": "candidate", "target_id": "t1", "role": "primary"},
                  view=asdict(view))
    controller.advance(motion.active_goal_id)
    feedback = motion.poll_active()
    assert not feedback.tracking_active
    refresh_observations(session, observation_map, controller, client, tmp_path, panorama_views=1)
    assert [image["id"] for image in session.snapshot.metadata["observation_images"]] == ["47"]
    guided.motion_finished(feedback)
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
    assert observation["frontiers"] == [asdict(FrontierSnapshot(**item)) for item in current["frontiers"]]
    assert [item["frontier_uid"] for item in context["options"]] == [
        f"current:{i}" for i in range(1, frontier_count + 1)]
    assert context["baseline_recommendation"] == f"current:{frontier_count}"
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
    observation_map.add.assert_called_once_with(refreshed, target_list=session.detector_terms)
    assert observation_map.points.call_count == 2 * len(observation_map.objects())
    assert [action for action, _ in client.commands] == ["move", "observe", "look", "observe"]
    assert agent.model_calls == 1 and controller.moves == 1
    goal = controller.goals[motion.active_goal_id]
    assert goal.mode == MotionGoalMode.GO_TO_FRONTIER
    assert list(goal.goal_pose.position) == current["frontiers"][-1]["position"]


@pytest.mark.parametrize("outcome", ["installed", "unchanged", "invalid", "guard_refused"])
def test_guided_interpretation_handoff_preserves_physical_history(tmp_path, outcome):
    """Only an installed revision releases candidate attempts, never blocked paths."""
    from backends.isaac.runtime import IsaacGuidedNavigation
    from harness.tools import NavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from task.session import TaskSession

    client = FakeClient()
    controller = IsaacMotionController(client, client.pose)
    controller.moves = 7
    motion = MotionSupervisor(controller)
    spec = TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),))
    view = ViewpointSnapshot("view", Pose3D((1., 0., 1.5)), visible_objects=("obj",))
    frontiers = tuple(FrontierSnapshot(f"f:{i}", (float(i), 2., 1.5), score=float(i),
        metadata={"frame_id": "map", "explored_density": 0.}) for i in (1, 2))
    snapshot = SemanticMapSnapshot(12., pose_value(client.pose),
        objects=(ObjectNodeSnapshot("obj", "sofa", (2., 0., 1.5)),),
        viewpoints=(view,), frontiers=frontiers)
    session = TaskSession(snapshot, runtime_plan(spec))
    tools = NavigationTools(session, motion, None, None)
    queried = []

    def query():
        active = session.progress.active_target()
        queried.append((session.spec.revision, active.name))
        return {"candidates": [{"candidate_uid": "obj", "target_id": active.id,
                                "label": active.name, "role": "primary"}], "reason": "observed"}

    tools.query_scene = query
    runtime = SimpleNamespace(model_calls=2, max_model_calls=5)
    no_choice = outcome == "invalid"

    def choose(options, recommended, reason):
        runtime.model_calls += 1
        assert [option["frontier_uid"] for option in options] == ["f:1"]
        assert recommended == "f:1" and motion.active_request_id is None
        if outcome == "installed":
            revised = replace(spec, revision=2, targets=(TargetSpec("goal", "sofa"),))
            session.install_plan(runtime_plan(revised))
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
    guided = IsaacGuidedNavigation(tools, runtime, tmp_path)
    original_submit = guided.submit

    def submit(**kwargs):
        if "frontier" in kwargs:
            assert guard_events == ["enter"]
        return original_submit(**kwargs)

    guided.submit = Mock(side_effect=submit)
    guided.sequence = 4
    attempts = guided.attempted_views
    attempts["obj"] = [view.pose.position]
    blocked = guided.blocked_frontiers
    blocked.append(frontiers[1].position)
    if outcome == "guard_refused":
        with pytest.raises(ValueError, match="choice refused"):
            guided.select()
    else:
        assert guided.select()
    if outcome == "installed":
        assert session.spec.revision == 2
        assert guided.sequence == 4 and guided.pending is None
        assert motion.active_request_id is None and controller.goals == {}
        guided.submit.assert_not_called()
        assert attempts == {} and guided.attempted_views is attempts
        assert guided.select()
        assert queried == [(1, "chair"), (2, "sofa")]
        assert guided.pending["candidate"]["label"] == "sofa"
        assert motion.active_intent.target_object_uid == "obj"
        assert motion.active_intent.metadata["task_revision"] == 2
    else:
        assert session.spec == spec and queried == [(1, "chair")]
        assert attempts == {"obj": [view.pose.position]} and guided.attempted_views is attempts
        if outcome == "unchanged":
            assert motion.active_intent.metadata == {"task_revision": 1, "frontier_uid": "f:1"}
        else:
            assert guided.pending is None and motion.active_request_id is None
            assert controller.goals == {}
            guided.submit.assert_not_called()
    submitted = int(outcome in {"installed", "unchanged"})
    assert guided.sequence == 4 + submitted and len(controller.goals) == submitted
    if outcome in {"unchanged", "guard_refused"}:
        runtime.choice_guard.assert_called_once_with()
        assert guard_events == ["enter", "exit"]
    else:
        runtime.choice_guard.assert_not_called()
        assert guard_events == []
    assert guided.blocked_frontiers is blocked and blocked == [frontiers[1].position]
    assert controller.moves == 7 and client.commands == []
    assert controller.pose is client.pose and session.snapshot.robot_pose == snapshot.robot_pose
    assert runtime.model_calls == 3 and runtime.max_model_calls == 5
    runtime.choose_frontier.assert_called_once()
    events = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "submitted" for event in events) == submitted
    assert sum(event["event"] == "interpretation_changed" for event in events) == int(outcome == "installed")
    assert sum(event["event"] == "decision_reselect" for event in events) == int(no_choice)
    if no_choice:
        assert events[-1] == {"event": "decision_reselect", "moves": 7, "task_revision": 1}


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


@pytest.mark.parametrize("initial_calls", [1, 2])
def test_episode_startup_calls_share_guided_budget(monkeypatch, tmp_path, initial_calls):
    from backends.isaac import runtime
    from harness.runtime import AgentRuntime
    from harness.tools import NavigationTools
    from instruction.contracts import CompileResult, TargetSpec, TaskSpec
    from task.session import TaskSession

    public = {"selection_id": "SEL_TEST", "scene_id": "test", "style": "formal",
              "instruction": "Find a chair.", "start_position": [0., 0., 1.5],
              "start_rotation_quat_wxyz": [1., 0., 0., 0.]}
    initial = SimpleNamespace(model_calls=initial_calls, compile=Mock(return_value=CompileResult(
        "success", TaskSpec(public["instruction"], (TargetSpec("goal", "chair"),)))))
    monkeypatch.setattr(runtime, "InitialInstructionCompiler", Mock(return_value=initial))
    client = FakeClient()
    client.status = "reached"
    monkeypatch.setattr(runtime, "IsaacClient", lambda port: client)
    monkeypatch.setattr(runtime, "IsaacObservationMap", Mock(return_value=object()))
    monkeypatch.setattr(runtime, "refresh_observations", Mock())
    frontiers = tuple(FrontierSnapshot(f"f:{i}", (float(i), 0., 1.5), score=float(i),
        metadata={"frame_id": "map", "explored_density": 0.}) for i in (1, 2))

    def task_session(snapshot, plan, **kwargs):
        return TaskSession(replace(snapshot, frontiers=frontiers), plan, **kwargs)

    def make_tools(session, motion, *args):
        tools = NavigationTools(session, motion, None, None)
        tools.query_scene = Mock(return_value={"candidates": [], "reason": "no_observed_candidate"})
        return tools

    def parse(**kwargs):
        choice = kwargs["response_format"](frontier_uid="f:1", reason="Continue searching")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=choice))])

    parse_call = Mock(side_effect=parse)
    model = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_call))))
    agents = []

    def make_agent(tools, trace_path, **kwargs):
        agent = AgentRuntime(tools, trace_path, client_model=(model, "offline-replay"), **kwargs)
        assert agent.model_calls == initial_calls
        agents.append(agent)
        return agent

    monkeypatch.setattr(runtime, "TaskSession", task_session)
    monkeypatch.setattr(runtime, "IsaacNavigationTools", make_tools)
    monkeypatch.setattr(runtime, "AgentRuntime", make_agent)
    monkeypatch.setenv("STRIVE_LVLM_TRACE_DIR", str(tmp_path / "model_requests"))
    result = runtime.run_episode(public, tmp_path, max_model_calls=3, max_moves=3)
    assert result["status"] == "completed" and result["reason"] == "decision_budget_exhausted"
    assert result["model_calls"] == agents[0].model_calls == 3
    assert parse_call.call_count == result["moves"] == 3 - initial_calls
    assert sum(action == "move" for action, _ in client.commands) == 3 - initial_calls
    assert not result["task_complete"]
    assert client.commands[-2][0] == "finish"
    plan = json.loads((tmp_path / "instruction_plan.json").read_text())
    assert plan["diagnostics"]["initial_model_calls"] == initial_calls
