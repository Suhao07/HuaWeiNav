"""Replay Agent calls through real task/query/motion/verifier tool bindings."""
from types import SimpleNamespace
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import base64
import hashlib
import json
import re

import pytest

from harness.runtime import AgentRuntime, ToolCall, compact_guided_context
from harness.tools import NavigationTools
from instruction.contracts import TaskSpec, TargetSpec
from instruction_adapter.task_compat import runtime_plan
from motion.contracts import NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from scene.contracts import FrontierSnapshot, ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot, ViewpointSnapshot
from task.session import TaskSession
from verification.contracts import VerificationResult


def tool_call(*, name, arguments, reason):
    return ToolCall(action={"name": name, "arguments": arguments, "reason": reason})


class Controller:
    def __init__(self):
        self.sent = []
        self.status = NavigationStatusCode.RUNNING

    def send_goal(self, goal):
        self.sent.append(goal)
        return f"goal-{len(self.sent)}"

    def poll_status(self, goal_id):
        return NavigationStatus(self.status, goal_id=goal_id)

    def cancel(self, goal_id):
        self.status = NavigationStatusCode.PREEMPTED

    def hold(self):
        pass


class Verifier:
    def verify(self, *, raw_instruction, instruction_plan, candidate, evidence):
        assert evidence["arrival_frame"] == "frame-1"
        assert "object:cup-1" in evidence["dependencies"]
        return VerificationResult(True, "accept", semantic_satisfied=True,
                                  task_revision=1, target_uid=candidate.uid)


def make_tools(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    task = TaskSpec("find cup", (TargetSpec("cup", "cup"),))
    snapshot = SemanticMapSnapshot(
        1, Pose3D((0, 0, 0)),
        objects=(ObjectNodeSnapshot("cup-1", "cup", (2, 0, 0), visible_viewpoints=("view-1",)),),
        viewpoints=(ViewpointSnapshot("view-1", Pose3D((1, 0, 0))),),
    )
    session = TaskSession(snapshot, runtime_plan(task))
    owner = MotionSupervisor(Controller())
    return NavigationTools(session, owner, Verifier(), lambda uid: {"arrival_frame": "frame-1"})


def model_calls(calls):
    calls = iter(calls)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        parsed = next(calls)
        if callable(parsed):
            parsed = parsed(kwargs)
        if isinstance(parsed, dict):
            parsed = kwargs["response_format"](**parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    return (client, "replay-model"), requests


def test_tool_loop_moves_verifies_and_finishes_from_domain_state(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    specs = [
        ("observe", {}), ("query_scene", {}),
        ("query_viewpoints", {"candidate_uid": "cup-1"}),
        ("submit_motion", {"candidate_uid": "cup-1", "viewpoint_uid": "view-1",
                           "task_revision": 1, "control_epoch": 0, "request_id": "motion-1"}),
        ("motion_status", {}), ("cancel_motion", {}), ("motion_status", {}),
        ("submit_motion", {"candidate_uid": "cup-1", "viewpoint_uid": "view-1",
                           "task_revision": 1, "control_epoch": 1, "request_id": "motion-2"}),
        ("motion_status", {}),
        ("verify_target", {"candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}),
        ("propose_finish", {}),
    ]
    client_model, requests = model_calls([tool_call(name=n, arguments=a, reason="replay") for n, a in specs])
    trace = tmp_path / "events.jsonl"
    runtime = AgentRuntime(tools, trace, max_model_calls=len(specs), client_model=client_model)
    from prompting import templates
    # Check model-facing instructions; task inputs may remain multilingual.
    prompt_texts = [runtime.system_prompt, json.dumps(ToolCall.schema(), ensure_ascii=False)]
    prompt_texts.extend(value for name, value in vars(templates).items()
                        if name.endswith("_PROMPT") and isinstance(value, str))
    assert all(re.search(r"[\u3400-\u9fff]", text) is None for text in prompt_texts)
    assert not tools.propose_finish()["accepted"]
    for index in range(len(specs)):
        if index == 7:
            tools.motion.controller.status = NavigationStatusCode.RUNNING
        if index == 8:
            tools.motion.controller.status = NavigationStatusCode.REACHED
        outcome = runtime.step()
        if index < 9:
            assert not tools.session.progress.completed
    assert outcome["result"]["accepted"]
    assert tools.session.progress.target_progress["cup"].count == 1
    assert len(tools.motion.controller.sent) == 2
    assert all(goal.goal_pose.position == (1, 0, 0) for goal in tools.motion.controller.sent)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(events) == 2 * len(specs)
    assert len(runtime.history) == 4
    assert len(request_context(requests[-1])["recent_tools"]) == 4
    with pytest.raises(RuntimeError, match="budget"):
        runtime.step()


def test_model_motion_with_stale_revision_is_recorded_and_propagated(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    client_model, _ = model_calls([tool_call(name="submit_motion", arguments={
        "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 2,
        "control_epoch": 0, "request_id": "stale",
    }, reason="stale")])
    trace = tmp_path / "events.jsonl"
    runtime = AgentRuntime(tools, trace, client_model=client_model)
    with pytest.raises(ValueError, match="revision"):
        runtime.step()
    assert tools.motion.controller.sent == []
    assert json.loads(trace.read_text().splitlines()[-1])["event"] == "failed"


def test_scene_change_during_model_request_prevents_motion(monkeypatch, tmp_path):
    from dataclasses import replace
    tools = make_tools(monkeypatch)

    def parse(**kwargs):
        snapshot = tools.session.snapshot
        obj = replace(snapshot.objects[0], position=(8, 0, 0))
        tools.session.snapshot = replace(snapshot, objects=(obj,))
        call = tool_call(name="submit_motion", arguments={
            "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
            "control_epoch": 0, "request_id": "stale-geometry",
        }, reason="late response")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=call))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    runtime = AgentRuntime(tools, tmp_path / "events.jsonl", client_model=(client, "replay"))
    with pytest.raises(ValueError, match="dependencies changed"):
        runtime.step()
    assert tools.motion.controller.sent == []


def test_verification_scene_change_preserves_task_progress(monkeypatch):
    from dataclasses import replace
    tools = make_tools(monkeypatch)

    class DelayedVerifier(Verifier):
        def verify(self, **kwargs):
            result = super().verify(**kwargs)
            snapshot = tools.session.snapshot
            tools.session.snapshot = replace(snapshot, objects=())
            return result

    tools.verifier = DelayedVerifier()
    with pytest.raises(ValueError, match="dependencies changed"):
        tools.verify_target("cup-1", "cup", 1)
    assert not tools.session.progress.completed
    assert tools.session.verification_ledger.history == []


def test_frontier_motion_uses_exported_frame_and_keeps_receipt(monkeypatch):
    from dataclasses import replace
    from scene.contracts import FrontierSnapshot
    tools = make_tools(monkeypatch)
    tools.session.snapshot = replace(tools.session.snapshot, frontiers=(
        FrontierSnapshot("frontier-1", (3, 2, 1), metadata={"frame_id": "mapper_local"}),
    ))
    tools.submit_motion(1, 0, "explore-1", frontier_uid="frontier-1")
    goal = tools.motion.controller.sent[0]
    assert goal.goal_pose.position == (3, 2, 1)
    assert goal.goal_pose.frame_id == "mapper_local"
    tools.motion.controller.status = NavigationStatusCode.REACHED
    tools.motion.poll_active()
    assert tools.motion_status()["status"]["status"] == NavigationStatusCode.REACHED


@pytest.fixture(params=["object", "frontier"])
def motion_submission(request, monkeypatch):
    from scene.contracts import FrontierSnapshot

    tools = make_tools(monkeypatch)
    if request.param == "frontier":
        tools.session.snapshot = replace(tools.session.snapshot, frontiers=(
            FrontierSnapshot("frontier-1", (3, 2, 1), metadata={"frame_id": "mapper_local"}),
        ))
        arguments = {"frontier_uid": "frontier-1"}
    else:
        arguments = {"candidate_uid": "cup-1", "viewpoint_uid": "view-1"}
    return tools, arguments


@pytest.mark.parametrize("checkpoint", ["lookup", "reservation"])
def test_motion_submission_serializes_interpretation(motion_submission, monkeypatch, checkpoint):
    from threading import Event, Thread

    tools, arguments = motion_submission
    session, motion = tools.session, tools.motion
    original = session.spec
    plan = runtime_plan(replace(original, revision=2, exhaustive=True))
    attempted = Event()
    lock_available, outcomes = [], []

    def adopt():
        available = motion._lock.acquire(blocking=False)
        if available:
            motion._lock.release()
        lock_available.append(available)
        attempted.set()
        try:
            outcomes.append(tools.install_interpretation(
                plan, task_revision=1, search_revision=0, snapshot=session.snapshot,
                record={"control_epoch": 0}))
        except Exception as error:
            outcomes.append(error)

    worker = Thread(target=adopt, daemon=True)

    def interleave():
        worker.start()
        assert attempted.wait(5), "Adoption thread did not reach the shared lock"
        # 非阻塞探测确认互斥，避免用等待时长猜测线程是否被阻塞。
        assert lock_available == [False]
        assert session.spec is original
        assert motion.controller.sent == []
        assert motion.active_request_id is None

    if checkpoint == "lookup":
        class GatedItems(tuple):
            def __iter__(self):
                interleave()
                return super().__iter__()

        field = "frontiers" if "frontier_uid" in arguments else "objects"
        session.snapshot = replace(session.snapshot, **{
            field: GatedItems(getattr(session.snapshot, field)),
        })
    else:
        submit = motion.submit

        def gated_submit(*args, **kwargs):
            interleave()
            return submit(*args, **kwargs)

        monkeypatch.setattr(motion, "submit", gated_submit)

    try:
        result = tools.submit_motion(1, 0, "racing-motion", **arguments)
    finally:
        if worker.ident is not None:
            worker.join(5)
    assert not worker.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], RuntimeError)
    assert "idle, unheld" in str(outcomes[0])
    assert session.spec is original and session.interpretation_history == []
    assert motion.control_epoch == 0 and motion.active_request_id == "racing-motion"
    assert motion.active_intent.metadata["task_revision"] == 1
    assert result == {"goal_id": "goal-1", "phase": "accepted"}
    assert len(motion.controller.sent) == 1


def test_motion_submission_rejects_revision_adopted_at_same_epoch(motion_submission):
    tools, arguments = motion_submission
    session, motion = tools.session, tools.motion
    tools.install_interpretation(
        runtime_plan(replace(session.spec, revision=2, exhaustive=True)),
        task_revision=1, search_revision=0, snapshot=session.snapshot,
        record={"control_epoch": 0})
    assert session.spec.revision == 2 and motion.control_epoch == 0
    with pytest.raises(ValueError, match="another task revision"):
        tools.submit_motion(1, 0, "stale-motion", **arguments)
    assert motion.controller.sent == [] and motion.active_request_id is None
    assert tools.submit_motion(2, 0, "current-motion", **arguments)["goal_id"] == "goal-1"
    assert motion.active_intent.metadata["task_revision"] == 2


def test_motion_submission_retries_active_request_without_redispatch(motion_submission):
    tools, arguments = motion_submission
    first = tools.submit_motion(1, 0, "same-request", **arguments)
    assert tools.motion.active_request_id == "same-request"
    assert tools.submit_motion(1, 0, "same-request", **arguments) == first
    with pytest.raises(RuntimeError, match="ownership"):
        tools.submit_motion(1, 0, "different-request", **arguments)
    assert tools.motion.active_request_id == "same-request"
    assert tools.last_goal_id == first["goal_id"] == "goal-1"
    assert len(tools.motion.controller.sent) == 1


def test_tool_schema_binds_names_to_exact_arguments():
    from pydantic import ValidationError
    call = tool_call(name="submit_motion", arguments={
        "frontier_uid": "node:2", "task_revision": 1, "control_epoch": 0, "request_id": "r1",
    }, reason="explore")
    assert call.action.arguments.dict() == {
        "frontier_uid": "node:2", "task_revision": 1, "control_epoch": 0, "request_id": "r1",
    }
    with pytest.raises(ValidationError):
        tool_call(name="query_scene", arguments={"target_id": "cup"}, reason="invalid protocol")
    with pytest.raises(ValidationError):
        tool_call(name="submit_motion", arguments={}, reason="missing motion fields")
    assert tool_call(name="query_scene", arguments={}, reason="query").action.arguments.dict() == {}


def test_observation_preserves_late_discovered_objects_and_frontiers(monkeypatch):
    from dataclasses import replace
    from scene.contracts import FrontierSnapshot
    tools = make_tools(monkeypatch)
    objects = tuple(ObjectNodeSnapshot(f"object-{i}", "chair", (i, 0, 0)) for i in range(40))
    frontiers = tuple(FrontierSnapshot(f"node:{i}", (i, 0, 0)) for i in range(40))
    tools.session.snapshot = replace(tools.session.snapshot, objects=objects, frontiers=frontiers)
    observed = tools.observe()
    assert [obj["uid"] for obj in observed["objects"]] == [obj.uid for obj in objects]
    assert [node["uid"] for node in observed["frontiers"]] == [node.uid for node in frontiers]


def test_arrival_refresh_preserves_object_recheck_until_instance_disappears(monkeypatch):
    from dataclasses import replace
    from backends.habitat.agent_runtime import HabitatNavigationTools
    import backends.habitat.query as backend
    tools = make_tools(monkeypatch)
    snapshot = tools.session.snapshot
    recheck = ViewpointSnapshot("recheck-1", Pose3D((1, 1, 0)), visible_objects=("cup-1",))
    tools.session.snapshot = replace(snapshot, viewpoints=snapshot.viewpoints + (recheck,))
    tools.agent = SimpleNamespace(task=tools.session, mapper=SimpleNamespace(nodes=[]), episode_steps=5, current_node_idx=0)
    monkeypatch.setattr(tools.session.constraint_evaluator, "register_mapper_objects", lambda **kwargs: None)
    arrived = replace(snapshot, robot_pose=Pose3D((1, 0, 0)), viewpoints=())
    monkeypatch.setattr(backend, "snapshot_from_mapper", lambda *args, **kwargs: (arrived, {"cup-1": object()}))
    observation = {"id": "arrival:5", "image_ref": "", "observed_object_uids": [], "object_boxes": {}}
    monkeypatch.setattr(backend, "current_observation", lambda agent: observation)
    HabitatNavigationTools.refresh_scene(tools)
    tools.submit_motion(1, 0, "recheck-after-arrival", candidate_uid="cup-1", viewpoint_uid="recheck-1")
    assert tools.motion.controller.sent[-1].goal_pose.position == (1, 1, 0)

    monkeypatch.setattr(backend, "snapshot_from_mapper", lambda *args, **kwargs: (replace(arrived, objects=()), {}))
    HabitatNavigationTools.refresh_scene(tools)
    assert tools.session.snapshot.viewpoints == ()


def test_recovery_schema_only_selects_supplied_domain_action(monkeypatch, tmp_path):
    from pydantic import ValidationError
    tools = make_tools(monkeypatch)
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        schema = kwargs["response_format"]
        with pytest.raises(ValidationError):
            schema(action_uid="invented-coordinate", reason="invalid")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=schema(action_uid="approach:u", reason="untried view")))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    runtime = AgentRuntime(tools, tmp_path / "recovery.jsonl", client_model=(client,"replay"))
    selected = runtime.choose_recovery([
        {"action_uid": "approach:u", "kind": "viewpoint"},
        {"action_uid": "frontier:2", "kind": "frontier"}], {"reason": "view_options_exhausted"})
    assert selected == "approach:u" and runtime.model_calls == 1 and len(requests) == 1
    assert tools.motion.active_request_id is None  # selection leaves execution to the driver


def test_skill_loading_uses_utf8_after_native_locale_change(tmp_path):
    import locale
    from types import SimpleNamespace
    from harness.runtime import AgentRuntime
    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, 'C')
        runtime = AgentRuntime(SimpleNamespace(names=(), session=SimpleNamespace(
            plan=SimpleNamespace(diagnostics={}))), tmp_path/'trace.jsonl',
                               client_model=(object(), 'mock'))
        assert 'Return one ToolCall' in runtime.system_prompt
        assert len(runtime.system_prompt) > 500
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def request_context(request):
    return json.loads(request["messages"][1]["content"][0]["text"])


def _table_rows(table):
    """Decode tables while rejecting duplicate columns or misaligned rows."""
    assert set(table) == {"columns", "rows"}
    assert len(table["columns"]) == len(set(table["columns"]))
    assert all(len(row) == len(table["columns"]) for row in table["rows"])
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def test_compact_guided_tables_preserve_geometry_associations_and_quality(monkeypatch):
    tools = make_tools(monkeypatch)
    pose = Pose3D((1.234567, -2.345678, 3.456789), (.123456, .234567, .345678, .876543),
                  frame_id="mapper_local", stamp=12.345678)
    tools.session.snapshot = replace(tools.session.snapshot, robot_pose=pose, objects=(
        ObjectNodeSnapshot("cup-1", "cup", pose.position, confidence=.987654,
            bbox2d_xyxy=(1.234567, 2.345678, 30.456789, 40.567891), room_id="kitchen",
            visible_viewpoints=("view-1", "view-2"), verified_state="rejected",
            metadata={"label_probabilities": {"cup": .876543, "bowl": .123457}, "frame_id": "frame:17"}),
    ), frontiers=(FrontierSnapshot("node:1", pose.position, room_id="hall", score=.765432,
        metadata={"frame_id": "mapper_local", "path_length": 5.678912, "reachable": True,
                  "supported": False, "frontier_idxs": [[1, 2], [3, 4], [5, 6]]}),))
    quality = {"score": .654321, "center_score": .765432, "area_ratio": .345678,
               "visible_ratio": .876543, "projected_points": 37, "occlusion_score": .123456}
    options = [{"action_uid": "approach:view-1", "candidate_uid": "cup-1", "kind": "viewpoint",
        "uid": "view-1", "room_id": "kitchen", "visible_objects": ["cup-1", "table-1"],
        "explored": False, "pose": pose.as_dict(), "metadata": {
            "pitch": -.123456, "travel_distance": 2.345678, "path_length": 3.456789,
            "stop_feasible": False, "visible_ratio": .456789, "area_ratio": .234567,
            "distance_to_target": .987654, "recommended": True, "source": "projection",
            "visibility": {"status": "predicted", "confidence": .567891},
            "predicted_quality": quality, "score": 9.9, "center_score": 8.8,
            "border_score": .222222, "distance_score": .333333,
        }}, {"action_uid": "explore:node:1", "frontier_uid": "node:1", "kind": "frontier",
             "position": pose.position, "raw_distance_m": 5.678912, "reachable": True,
             "supported": False, "frame_id": "mapper_local", "frontier_idxs": [[1, 2], [3, 4]]}]
    observation = tools.observe()
    observation["verification_records"] = [{"confidence": .987654}]
    observation["search_context"]["basis"] = {
        "instruction": {"type": "original_instruction"},
        "verification:0": {"type": "stored_judgment", "record": {"confidence": .987654}},
    }
    before, options_before, snapshot_before = deepcopy((observation, options, tools.session.snapshot))

    projected, identities, geometry = compact_guided_context(observation, options)

    assert _table_rows(projected["objects"]) == [{
        "uid": "cup-1", "label": "cup", "position": [1.235, -2.346, 3.457],
        "confidence": .988, "room_id": "kitchen", "bbox2d_xyxy": [1.235, 2.346, 30.457, 40.568],
        "visible_viewpoints": ["view-1", "view-2"], "verified_state": "rejected",
        "label_probabilities": {"cup": .877, "bowl": .123}, "observed_frame": "frame:17",
    }]
    assert _table_rows(projected["frontiers"]) == [{
        "uid": "node:1", "position": [1.235, -2.346, 3.457], "room_id": "hall", "score": .765,
        "frame_id": "mapper_local", "path_length": 5.679, "reachable": True,
        "supported": False, "frontier_cell_count": 3,
    }]
    assert identities == [
        {"action_uid": "approach:view-1", "candidate_uid": "cup-1", "kind": "viewpoint"},
        {"action_uid": "explore:node:1", "frontier_uid": "node:1", "kind": "frontier"},
    ]
    assert _table_rows(geometry["viewpoints"]) == [{
        **identities[0], "uid": "view-1", "room_id": "kitchen", "visible_objects": ["cup-1", "table-1"],
        "explored": False, "position": [1.235, -2.346, 3.457],
        "orientation_xyzw": [.123, .235, .346, .877], "frame_id": "mapper_local", "stamp": 12.346,
        "pitch": -.123, "travel_distance": 2.346, "path_length": 3.457, "stop_feasible": False,
        "visible_ratio": .457, "area_ratio": .235, "distance_to_target": .988, "recommended": True,
        "source": "projection", "visibility": {"status": "predicted", "confidence": .568},
        **{"quality_" + key: round(value, 3) for key, value in quality.items()},
        "quality_border_score": .222, "quality_distance_score": .333,
    }]
    assert _table_rows(geometry["frontiers"]) == [{
        **identities[1], "position": [1.235, -2.346, 3.457], "raw_distance_m": 5.679,
        "reachable": True, "supported": False, "frame_id": "mapper_local", "frontier_cell_count": 2,
    }]
    assert projected["verification_records"] == ["verification:0"]
    assert [projected["search_context"]["basis"][uid]["record"]
            for uid in projected["verification_records"]] == before["verification_records"]
    assert {key: value for key, value in projected.items()
            if key not in {"objects", "frontiers", "verification_records"}} == {
        key: value for key, value in before.items()
        if key not in {"objects", "frontiers", "verification_records"}}
    assert observation == before and options == options_before
    assert tools.session.snapshot == snapshot_before


def decision_call(runtime, mode):
    if mode == "frontier":
        return runtime.choose_frontier([
            {"frontier_uid": "node:1", "raw_distance_m": 2.0},
            {"frontier_uid": "node:2", "raw_distance_m": 3.0},
        ], "node:1", {"reason": "prolonged_target_search", "steps_since_review_or_candidate": 60})
    if mode == "recovery":
        return runtime.choose_recovery([
            {"action_uid": "approach:u", "kind": "viewpoint", "candidate_uid": "cup-1"},
            {"action_uid": "frontier:2", "kind": "frontier"},
        ], {"reason": "view_options_exhausted", "candidate_uid": "cup-1"})
    return runtime.step()


def decision_payload(mode):
    if mode == "frontier":
        return {"frontier_uid": "node:2", "reason": "explore another branch"}
    if mode == "recovery":
        return {"action_uid": "approach:u", "reason": "collect missing evidence"}
    return tool_call(name="observe", arguments={}, reason="observe")


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
def test_large_guided_context_keeps_last_candidate_and_bounds_repeated_options(monkeypatch, tmp_path, mode):
    tools = make_tools(monkeypatch)
    count = 225  # Later discoveries must not disappear behind a prefix/TopK cap.
    cells = [[index, index + 1] for index in range(200)]
    objects = tuple(ObjectNodeSnapshot(f"cup:{i}", "cup", (i + .123456, 0, 0),
                    visible_viewpoints=(f"view:{i}",)) for i in range(count))
    views = tuple(ViewpointSnapshot(f"view:{i}", Pose3D((i + .234567, 0, 0)),
                  visible_objects=(f"cup:{i}",), metadata={"predicted_quality": {"score": .876543}})
                  for i in range(count))
    frontiers = tuple(FrontierSnapshot(f"node:{i}", (i + .345678, 1, 0),
                     metadata={"frontier_idxs": cells, "frame_id": "map", "path_length": i + .456789})
                     for i in range(count))
    tools.session.snapshot = replace(tools.session.snapshot, objects=objects, viewpoints=views, frontiers=frontiers)
    options = [{"frontier_uid": node.uid, "action_uid": f"explore:{node.uid}", "kind": "frontier",
                "position": node.position, **node.metadata} for node in frontiers]
    if mode == "recovery":
        options.extend({**asdict(view), "action_uid": f"approach:{view.uid}", "kind": "viewpoint",
                        "candidate_uid": obj.uid} for obj, view in zip(objects, views))
    raw = json.dumps({"observation": tools.observe(), "options": options})
    assert len(raw) > 330_000
    options_before, snapshot_before = deepcopy((options, tools.session.snapshot))
    field = "frontier_uid" if mode == "frontier" else "action_uid"
    payload = {field: options[-1][field], "reason": "Inspect the last discovered candidate"}
    client_model, requests = model_calls([payload, payload])
    runtime = AgentRuntime(tools, tmp_path / "large.jsonl", max_model_calls=2, client_model=client_model)
    for _ in range(2):
        selected = (runtime.choose_frontier(options, options[0][field], {"reason": "review"})
                    if mode == "frontier" else runtime.choose_recovery(options, {"reason": "exhausted"}))
        assert selected == options[-1][field]
    for request in requests:
        context = request_context(request)
        assert request["response_format"].schema()["properties"][field]["enum"] == [o[field] for o in options]
        assert context["options"] == [{key: value for key, value in option.items()
                                      if key in {"action_uid", "frontier_uid", "candidate_uid", "kind"}}
                                     for option in options]
        assert [row["uid"] for row in _table_rows(context["observation"]["objects"])] == [o.uid for o in objects]
        assert [row["uid"] for row in _table_rows(context["observation"]["frontiers"])] == [f.uid for f in frontiers]
        branches = _table_rows(context["option_geometry"]["frontiers"])
        assert [row["frontier_uid"] for row in branches] == [f.uid for f in frontiers]
        assert all(row["frontier_cell_count"] == len(cells) for row in branches)
        projected_views = _table_rows(context["option_geometry"]["viewpoints"])
        assert [(row["action_uid"], row["candidate_uid"], row["uid"]) for row in projected_views] == (
            [(f"approach:{v.uid}", o.uid, v.uid) for o, v in zip(objects, views)] if mode == "recovery" else [])
        assert "frontier_idxs" not in request["messages"][1]["content"][0]["text"]
        assert len(request["messages"][1]["content"][0]["text"]) < len(raw) / 2
    previous = request_context(requests[-1])["recent_tools"]
    assert previous == [{**{key: value for key, value in runtime.history[0].items() if key != "options"},
                         "option_count": len(options)}]
    assert all(event["options"] == options_before for event in runtime.history)
    trace = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    assert trace == json.loads(json.dumps(runtime.history))
    assert options == options_before and tools.session.snapshot == snapshot_before
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("empty", [True, False])
def test_guided_tables_allow_empty_lists_and_missing_optional_metadata(monkeypatch, tmp_path, empty):
    tools = make_tools(monkeypatch)
    tools.session.snapshot = replace(tools.session.snapshot,
        objects=() if empty else tools.session.snapshot.objects,
        frontiers=() if empty else (FrontierSnapshot("node:empty", (0, 0, 0)),
            FrontierSnapshot("node:known", (1, 0, 0), metadata={"path_length": .123456, "reachable": False})))
    options = [] if empty else [
        {"action_uid": "view:sparse", "kind": "viewpoint", "candidate_uid": "cup-1"},
        {"frontier_uid": "node:empty", "frontier_idxs": []},
        {"frontier_uid": "node:known", "reachable": False},
    ]
    before, options_before = deepcopy((tools.observe(), options))
    runtime = AgentRuntime(tools, tmp_path / "sparse.jsonl", client_model=(object(), "unused"))
    plain = request_context({"messages": runtime._messages("tool")})
    guided = request_context({"messages": runtime._messages("guided", options=options)})
    assert "option_geometry" not in plain
    for key in ("objects", "frontiers"):
        assert plain["observation"][key] == json.loads(json.dumps(before[key]))
    if empty:
        assert guided["options"] == []
        assert all(table == {"columns": [], "rows": []} for table in (
            guided["observation"]["objects"], guided["observation"]["frontiers"],
            *guided["option_geometry"].values()))
    else:
        obj, = _table_rows(guided["observation"]["objects"])
        assert all(obj[key] is None for key in ("label_probabilities", "observed_frame", "room_id", "bbox2d_xyxy"))
        sparse, known = _table_rows(guided["observation"]["frontiers"])
        assert sparse["path_length"] is None and sparse["reachable"] is None
        assert known["path_length"] == .123 and known["reachable"] is False
        assert _table_rows(guided["option_geometry"]["viewpoints"]) == [options[0]]
        assert _table_rows(guided["option_geometry"]["frontiers"]) == [
            {"frontier_uid": "node:empty", "frontier_cell_count": 0, "reachable": None},
            {"frontier_uid": "node:known", "frontier_cell_count": None, "reachable": False}]
    assert tools.observe() == before and options == options_before


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
def test_guided_requests_share_skills_and_real_scoped_task_evidence(monkeypatch, tmp_path, mode):
    from instruction.contracts import ConstraintSpec
    tools = make_tools(monkeypatch)
    task = TaskSpec(
        "依次找两个桌子旁的杯子", (TargetSpec("cup", "cup", min_count=2),
                            TargetSpec("table", "table", role="anchor", terminal=False)),
        constraints=(ConstraintSpec("beside", "relation", "cup", "near", "table"),),
        revision=2, ordered=True, exhaustive=True,
    )
    tools.session.install_plan(runtime_plan(task))
    ledger = tools.session.verification_ledger
    record = ledger.put(task.raw_instruction, "cup-1", VerificationResult(
        False, "need_better_view", confidence=.987654, task_revision=2, target_uid="cup-1",
        predicate_id="beside", anchor_uid="table-1", failed_constraints=["beside"],
        reason="Anchor is occluded", evidence_ids=["frame:checked-1"],
        dependencies={"object:cup-1": "observed-version"},
    ), step=14)
    # Neither another instruction nor another revision is evidence for this task.
    ledger.put("find something else", "cup-1", VerificationResult(True, "accept", task_revision=2))
    ledger.put(task.raw_instruction, "cup-old", VerificationResult(True, "accept", task_revision=1))
    basis = {"instruction": {"type": "original_instruction"},
             "verification:0": {"type": "stored_judgment", "record": record.as_dict()}}
    tools.session.revise_search([{"concept_id": "cup", "detector_terms": ["mug"],
        "basis_ids": ["verification:0"], "basis": {"verification:0": basis["verification:0"]},
        "reason": "Try another retrieval term for the occluded instance"}],
        task_revision=2, search_revision=0, snapshot=tools.session.snapshot)
    search_before = deepcopy(tools.observe()["search_context"])
    before = tools.session.progress.as_dict()
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "guided.jsonl", client_model=client_model)
    decision_call(runtime, mode)
    request = requests[0]
    prompt = request["messages"][0]["content"]
    for name in ("understand_instruction.md", "search.md", "active_observation.md", "recovery.md", "completion.md"):
        skill = (Path(__file__).resolve().parents[1] / "skills" / name).read_text(encoding="utf-8")
        assert skill in prompt
    assert "ToolCall" not in prompt
    assert re.search(r"[\u3400-\u9fff]", prompt) is None
    schema = request["response_format"].schema()
    expected = "FrontierChoice" if mode == "frontier" else "RecoveryChoice"
    field = "frontier_uid" if mode == "frontier" else "action_uid"
    assert schema["title"] == expected and "ToolCall" not in json.dumps(schema)
    assert set(schema["properties"]) == {field, "reason", "search_updates", "interpretation_update"}
    context = request_context(request)
    assert schema["properties"][field]["enum"] == [option[field] for option in context["options"]]
    observation = context["observation"]
    assert observation["task"] == task.as_dict()
    assert observation["progress"] == before
    assert observation["verification_records"] == ["verification:0"]
    assert observation["search_context"]["basis"]["verification:0"]["record"] == record.as_dict()
    assert observation["search_context"]["basis"]["verification:0"]["record"]["verdict"] == "unknown"
    assert observation["search_context"] == {**search_before, "basis": basis}
    assert tools.observe()["search_context"] == search_before
    assert "event" in context and context["recent_tools"] == []
    assert tools.session.progress.as_dict() == before
    assert not tools.propose_finish()["accepted"]
    assert tools.motion.controller.sent == []


def search_payload(mode, terms, *, concept_id="resource", basis_ids=None):
    payload = decision_payload(mode)
    if isinstance(payload, ToolCall):
        payload = payload.dict()
    return {**payload, "search_updates": [{
        "concept_id": concept_id, "detector_terms": terms,
        "basis_ids": basis_ids or ["instruction"],
        "reason": "Inspect another possible realization without changing the requested function.",
    }]}


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_search_revision_changes_candidate_retrieval_not_task_or_rejections(monkeypatch, tmp_path, mode):
    from harness.tools import query_scene
    tools = make_tools(monkeypatch)
    task = TaskSpec("I need a soft place to sit", (TargetSpec(
        "resource", "seating surface", attributes={"softness": "soft"}, demand={
            "activity": "sit", "required_resource": "seating surface", "manipulated_objects": [],
        }),), revision=2)
    tools.session.install_plan(runtime_plan(task))
    session = tools.session
    session.snapshot = replace(session.snapshot, objects=tuple(
        ObjectNodeSnapshot(uid, label, (2, 0, 0)) for uid, label in (
            ("bad-couch", "couch"), ("another-couch", "couch"), ("sofa-1", "sofa"))))
    ledger = session.verification_ledger
    ledger.put(task.raw_instruction, "bad-couch", VerificationResult(
        False, "reject_candidate", task_revision=2, reason="Instance is not soft"))
    before = session.progress.as_dict()
    record_before = ledger.as_dict()
    assert query_scene(session).candidates == []
    payloads = [search_payload(mode, [term]) for term in ("couch", "couch", "sofa")]
    client_model, requests = model_calls(payloads)
    runtime = AgentRuntime(tools, tmp_path / "revisions.jsonl", max_model_calls=3, client_model=client_model)
    for index, expected in enumerate(("another-couch", "another-couch", "sofa-1")):
        decision_call(runtime, mode)
        result = query_scene(session)
        assert [candidate.obj.uid for candidate in result.candidates] == [expected]
        assert all(candidate.record["verdict"] == "unknown" for candidate in result.candidates)
        assert session.spec.as_dict() == task.as_dict()
        assert session.progress.as_dict() == before
        assert session.verification_ledger is ledger and ledger.as_dict() == record_before
        assert not tools.propose_finish()["accepted"]
        assert tools.motion.controller.sent == []
        assert session.search_revision == (1 if index < 2 else 2)
    target = session.plan.targets[0]
    assert session.detector_terms == target.detector_terms == target.concept_query().detector_terms == ["sofa"]
    assert "sofa" not in target.match_terms and "couch" not in target.match_terms
    assert session.search_hypotheses["resource"]["basis_ids"] == ["instruction"]
    context = request_context(requests[-1])["observation"]["search_context"]
    assert context["hypotheses"][0]["detector_terms"] == ["couch"]
    assert context["basis"]["verification:0"]["record"]["candidate_uid"] == "bad-couch"
    with pytest.raises(RuntimeError, match="budget"):
        decision_call(runtime, mode)
    assert len(requests) == runtime.model_calls == 3


@pytest.mark.parametrize("invalid_second", [False, True])
def test_search_revision_updates_shared_anchor_atomically(monkeypatch, tmp_path, invalid_second):
    from instruction.contracts import ConstraintSpec
    from perception.detector_adapter import ground_plan
    tools = make_tools(monkeypatch)
    task = TaskSpec("cup near table", (TargetSpec("cup", "cup"),
        TargetSpec("table", "table", role="anchor", terminal=False)),
        constraints=(ConstraintSpec("near", "relation", "cup", "near", "table"),), revision=2)
    plan = ground_plan(runtime_plan(task), ["cup", "table"])
    tools.session.install_plan(plan)
    original = plan.as_dict()
    payload = search_payload("frontier", ["mug"], concept_id="cup")
    payload["search_updates"].extend(search_payload(
        "frontier", [" " if invalid_second else "desk"], concept_id="table")["search_updates"])
    client_model, _ = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "anchors.jsonl", client_model=client_model)
    if invalid_second:
        with pytest.raises(ValueError, match="retrieval phrases"):
            decision_call(runtime, "frontier")
        assert plan.as_dict() == original
        assert tools.session.search_revision == 0
    else:
        decision_call(runtime, "frontier")
        target, anchor = plan.targets
        assert target.detector_terms == target.concept.detector_terms == ["mug"]
        assert anchor.detector_terms == anchor.concept.detector_terms == ["desk"]
        assert plan.constraints[0].subject_concept is target.concept
        assert plan.constraints[0].object_concept is anchor.concept
        assert [concept.detector_terms for concept in plan.concept_queries] == [["mug"], ["desk"]]
        assert tools.session.search_revision == 1
    assert tools.session.spec == task and tools.motion.controller.sent == []


def test_same_task_plan_replacement_preserves_search_and_evidence(monkeypatch, tmp_path):
    from perception.detector_adapter import ground_plan
    tools = make_tools(monkeypatch)
    session = tools.session
    progress, ledger = session.progress, session.verification_ledger
    record = ledger.put(session.spec.raw_instruction, "cup-1", VerificationResult(
        False, "reject_candidate", task_revision=1, reason="Wrong instance"))
    client_model, _ = model_calls([search_payload("frontier", ["mug"], concept_id="cup")])
    decision_call(AgentRuntime(tools, tmp_path / "replacement.jsonl", client_model=client_model), "frontier")
    replacement = ground_plan(runtime_plan(session.spec), ["cup"])
    session.install_plan(replacement)
    assert session.search_revision == 2 and session.plan is replacement
    assert session.detector_terms == replacement.targets[0].detector_terms == ["mug"]
    assert all(concept.detector_terms == ["mug"] for concept in session.search_concepts)
    assert session.progress is progress and session.verification_ledger is ledger
    assert tools.observe()["verification_records"] == [record.as_dict()]
    session.install_plan(replacement)
    assert session.search_revision == 2 and tools.motion.controller.sent == []


def test_search_hypothesis_retains_cited_judgment_after_ledger_update(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    session = tools.session
    ledger = session.verification_ledger
    original = ledger.put(session.spec.raw_instruction, "cup-1", VerificationResult(
        False, "need_better_view", task_revision=1, reason="Occluded", evidence_ids=["frame:1"]), step=1)
    payload = search_payload("frontier", ["mug"], concept_id="cup", basis_ids=["verification:0"])
    client_model, requests = model_calls([payload, decision_payload("frontier")])
    runtime = AgentRuntime(tools, tmp_path / "provenance.jsonl", client_model=client_model)
    decision_call(runtime, "frontier")
    cited = session.search_hypotheses["cup"]["basis"]["verification:0"]
    assert cited == {"type": "stored_judgment", "record": original.as_dict()}
    revised = ledger.put(session.spec.raw_instruction, "cup-1", VerificationResult(
        False, "reject_candidate", task_revision=1, reason="Wrong instance", evidence_ids=["frame:2"]), step=2)
    decision_call(runtime, "frontier")
    context = request_context(requests[-1])["observation"]["search_context"]
    assert context["basis"]["verification:0"]["record"] == revised.as_dict()
    assert context["hypotheses"][0]["basis"]["verification:0"] == cited
    assert cited["record"]["reason"] == "Occluded"
    assert tools.motion.controller.sent == [] and not session.progress.completed


def test_search_revision_cannot_accompany_motion(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    payload = search_payload("step", ["mug"], concept_id="cup")
    payload["action"] = tool_call(name="submit_motion", arguments={
        "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
        "control_epoch": 0, "request_id": "mixed-update"}, reason="move").action.dict()
    client_model, _ = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "execution.jsonl", client_model=client_model)
    with pytest.raises(ValueError, match="not an execution action"):
        runtime.step()
    assert tools.session.search_revision == 0 and tools.session.detector_terms == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
@pytest.mark.parametrize("change", ["snapshot", "task", "plan", "search_revision"])
def test_stale_search_revision_never_applies_or_moves(monkeypatch, tmp_path, mode, change):
    tools = make_tools(monkeypatch)
    original_terms = list(tools.session.detector_terms)
    def parse(**kwargs):
        payload = kwargs["response_format"](**search_payload(mode, ["mug"], concept_id="cup"))
        if change == "snapshot":
            tools.session.snapshot = replace(tools.session.snapshot, timestamp=2)
        elif change == "task":
            tools.session.install_plan(runtime_plan(TaskSpec(
                "find cup", (TargetSpec("cup", "cup", attributes={"color": "red"}),), revision=2)))
        elif change == "plan":
            tools.session.install_plan(runtime_plan(tools.session.spec))
        else:
            tools.session.search_revision += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=payload))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    runtime = AgentRuntime(tools, tmp_path / "stale-search.jsonl", client_model=(client, "mock"))
    with pytest.raises(ValueError, match="changed|stale"):
        decision_call(runtime, mode)
    assert tools.session.search_hypotheses == {}
    assert tools.session.detector_terms == original_terms
    assert tools.motion.controller.sent == []
    assert not tools.session.progress.completed


@pytest.mark.parametrize("invalid", ["concept", "missing_frame", "identity", "empty_terms", "duplicate"])
def test_search_update_schema_and_domain_reject_illegal_changes(monkeypatch, tmp_path, invalid):
    from pydantic import ValidationError
    tools = make_tools(monkeypatch)
    original_terms = list(tools.session.detector_terms)
    tools.session.snapshot = replace(tools.session.snapshot, metadata={"observation_images": [{
        "id": "not-sent", "image_ref": "", "observed_object_uids": [], "object_boxes": {}}]})
    payload = search_payload("frontier", ["mug"], concept_id="cup")
    update = payload["search_updates"][0]
    if invalid == "concept":
        update["concept_id"] = "invented"
    elif invalid == "missing_frame":
        update["basis_ids"] = ["frame:not-sent"]
    elif invalid == "identity":
        update["aliases"] = ["mug"]
    elif invalid == "empty_terms":
        update["detector_terms"] = []
    else:
        payload["search_updates"].append(dict(update))
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "illegal-search.jsonl", client_model=client_model)
    with pytest.raises((ValidationError, ValueError)):
        decision_call(runtime, "frontier")
    assert tools.session.search_revision == 0 and tools.session.search_hypotheses == {}
    assert tools.session.detector_terms == original_terms
    assert "frame:not-sent" not in request_context(requests[0])["observation"]["search_context"]["basis"]
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_current_observation_frames_become_actual_image_blocks(monkeypatch, tmp_path, mode):
    from PIL import Image
    tools = make_tools(monkeypatch)
    frames = []
    for index, uids in enumerate((["cup-1"], [])):
        path = tmp_path / f"camera-{index}.png"
        Image.new("RGB", (2, 2), (index * 120, 40, 60)).save(path)
        frames.append({"id": f"camera:frame-{index}", "image_ref": str(path),
                       "observed_object_uids": uids,
                       "object_boxes": {uid: [0, 0, 1, 1] for uid in uids}})
    tools.session.snapshot = replace(tools.session.snapshot, metadata={"observation_images": frames})

    def forbidden(*args, **kwargs):
        pytest.fail("A decision must not capture evidence, query or poll motion")

    monkeypatch.setattr(tools, "evidence_provider", forbidden)
    monkeypatch.setattr(tools.verifier, "verify", forbidden)
    monkeypatch.setattr(tools, "query_scene", forbidden)
    monkeypatch.setattr(tools, "motion_status", forbidden)
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "images.jsonl", client_model=client_model)
    decision_call(runtime, mode)
    content = requests[0]["messages"][1]["content"]
    observation = request_context(requests[0])["observation"]
    assert len(content) == 5
    for index, frame in enumerate(frames):
        label, image = content[1 + 2 * index:3 + 2 * index]
        assert label == {"type": "text", "text": "Current observation image: " + json.dumps(frame["id"])
                         + "; metadata in observation.observation_images."}
        projected = observation["observation_images"][index]
        assert {key: projected[key] for key in frame} == frame
        assert image["type"] == "image_url"
        data_url = image["image_url"]["url"]
        assert data_url.startswith("data:image/png;base64,")
        assert base64.b64decode(data_url.split(",", 1)[1]) == Path(frame["image_ref"]).read_bytes()
        projected = observation["observation_images"][index]
        assert projected["image_status"] == "available"
        assert observation["search_context"]["basis"][f"frame:{frame['id']}"] == {
            "type": "current_frame", "id": frame["id"],
            "image_sha256": hashlib.sha256(Path(frame["image_ref"]).read_bytes()).hexdigest(),
        }
        assert Path(projected["request_image_ref"]).read_bytes() == Path(frame["image_ref"]).read_bytes()
    assert tools.observe()["observation_images"] == frames  # no snapshot mutation
    assert all("image_status" not in frame for frame in frames)
    assert tools.motion.controller.sent == []
    assert not tools.session.progress.completed


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
@pytest.mark.parametrize("published", [False, True])
def test_no_current_camera_never_promotes_historical_object_images(monkeypatch, tmp_path, mode, published):
    from PIL import Image
    tools = make_tools(monkeypatch)
    old_image = tmp_path / "historical.png"
    Image.new("RGB", (1, 1)).save(old_image)
    snapshot = tools.session.snapshot
    tools.session.snapshot = replace(
        snapshot, objects=(replace(snapshot.objects[0], image_ref=str(old_image)),),
        metadata={"observation_images": []} if published else {},
    )
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "no-camera.jsonl", client_model=client_model)
    decision_call(runtime, mode)
    content = requests[0]["messages"][1]["content"]
    assert [block["type"] for block in content] == ["text"]
    assert request_context(requests[0])["observation"]["observation_images"] == []
    assert runtime.model_calls == 1
    assert not tools.session.progress.completed


@pytest.mark.parametrize("ref", [None, "", "unknown", "relative.png", "https://example.invalid/image.png", "absent"])
def test_missing_images_remain_explicit_facts_without_fallback(monkeypatch, tmp_path, ref):
    tools = make_tools(monkeypatch)
    if ref == "absent":
        ref = str(tmp_path / "absent-with-unknown-suffix")
    frame = {"id": "frame:unavailable", "image_ref": ref,
             "observed_object_uids": ["cup-1"], "object_boxes": {"cup-1": [0, 0, 1, 1]}}
    tools.session.snapshot = replace(tools.session.snapshot, metadata={"observation_images": [frame]})
    client_model, requests = model_calls([decision_payload("frontier")])
    runtime = AgentRuntime(tools, tmp_path / "missing.jsonl", client_model=client_model)
    assert decision_call(runtime, "frontier") == "node:2"
    observed = request_context(requests[0])["observation"]["observation_images"][0]
    assert observed["id"] == frame["id"] and observed["image_ref"] == ref
    assert observed["image_status"] == "missing"
    assert "missing_reason" in observed
    assert [block["type"] for block in requests[0]["messages"][1]["content"]] == ["text"]
    assert not tools.propose_finish()["accepted"]


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
@pytest.mark.parametrize("failure", ["read", "archive"])
def test_image_io_errors_propagate_without_request_or_budget_charge(monkeypatch, tmp_path, mode, failure):
    from PIL import Image
    tools = make_tools(monkeypatch)
    image = tmp_path / "unreadable.png"
    Image.new("RGB", (1, 1)).save(image)
    tools.session.snapshot = replace(tools.session.snapshot, metadata={"observation_images": [
        {"id": "camera:error", "image_ref": str(image), "observed_object_uids": [], "object_boxes": {}},
    ]})
    if failure == "read":
        error = PermissionError(13, "source read denied", str(image))
        original = Path.read_bytes
        def read_bytes(path):
            if path == image:
                raise error
            return original(path)
        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    else:
        error = FileNotFoundError(2, "archive unavailable", str(tmp_path / "archive.png"))
        def write_bytes(path, data):
            raise error
        monkeypatch.setattr(Path, "write_bytes", write_bytes)
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "io.jsonl", client_model=client_model)
    with pytest.raises(type(error)) as raised:
        decision_call(runtime, mode)
    assert raised.value is error
    assert requests == [] and runtime.model_calls == 0 and runtime.history == []


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
def test_first_guided_request_includes_bounded_baseline_execution(monkeypatch, tmp_path, mode):
    from backends.habitat.guided_runtime import GuidedNavigation

    tools = make_tools(monkeypatch)
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "tools.jsonl", max_model_calls=1, client_model=client_model)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent = SimpleNamespace(task=tools.session, episode_steps=32)
    guided.directory, guided.runtime = tmp_path, runtime
    guided.record("candidate_evidence_updated", candidate_uid="cup-1", evidence_fingerprint="new")
    events = [
        {"event": "submitted", "candidate_uid": "cup-1", "position": [1, 0, 0],
         "decision_source": "baseline_viewpoint"},
        {"event": "motion_terminal", "candidate_uid": "cup-1", "position": [1, 0, 0],
         "status": "reached", "metadata": {"task_revision": 1}},
        {"event": "verified", "candidate_uid": "cup-1", "decision": "need_better_view",
         "task_complete": False, "pose_gap": {"observation_id": "0:32"}},
        {"event": "candidate_deferred", "candidate_uid": "cup-1", "reason": "view_options_exhausted"},
    ]
    for event in events:
        guided.record(**event)
    guided.record("search_review", options=[{"frontier_uid": "node:2"}])
    assert runtime.model_calls == 0 and requests == []
    expected = [{"step": 32, "task_revision": 1, **event} for event in events]
    assert runtime.history == expected
    decision_call(runtime, mode)
    context = request_context(requests[0])
    assert context["recent_tools"] == expected
    assert runtime.model_calls == 1 and len(runtime.history) == 4
    assert tools.motion.controller.sent == [] and not tools.session.progress.completed
    assert not tools.propose_finish()["accepted"]
    decisions = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert decisions[1:5] == expected
    trace = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    assert trace[1:5] == expected


def test_guided_choices_and_tool_failures_share_bounded_history(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    client_model, requests = model_calls([
        tool_call(name="submit_motion", arguments={"frontier_uid": "missing", "task_revision": 2,
                  "control_epoch": 0, "request_id": "stale"}, reason="stale attempt"),
        decision_payload("frontier"), decision_payload("recovery"), decision_payload("frontier"),
        decision_payload("recovery"), decision_payload("step"),
    ])
    runtime = AgentRuntime(tools, tmp_path / "history.jsonl", max_model_calls=6, client_model=client_model)
    with pytest.raises(ValueError, match="revision"):
        runtime.step()
    for mode in ["frontier", "recovery", "frontier", "recovery", "step"]:
        decision_call(runtime, mode)
    assert request_context(requests[1])["recent_tools"][0]["event"] == "failed"
    assert "revision" in request_context(requests[1])["recent_tools"][0]["error"]
    frontier_event = request_context(requests[2])["recent_tools"][-1]
    assert frontier_event["event"] == "frontier_choice" and frontier_event["selected_uid"] == "node:2"
    assert frontier_event["trigger"]["reason"] == "prolonged_target_search"
    recovery_event = request_context(requests[3])["recent_tools"][-1]
    assert recovery_event["event"] == "recovery_choice" and recovery_event["selected_uid"] == "approach:u"
    assert recovery_event["trigger"]["reason"] == "view_options_exhausted"
    assert [event["call_index"] for event in request_context(requests[-1])["recent_tools"]] == [2, 3, 4, 5]
    assert [event["call_index"] for event in runtime.history] == [3, 4, 5, 6]
    trace = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    assert all(json.loads(json.dumps(event)) in trace for event in runtime.history)
    assert tools.motion.controller.sent == [] and not tools.session.progress.completed
    for mode in ["frontier", "recovery", "step"]:
        with pytest.raises(RuntimeError, match="budget"):
            decision_call(runtime, mode)
    assert len(requests) == 6


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
def test_guided_invalid_ids_and_tool_protocol_are_rejected(monkeypatch, tmp_path, mode):
    from pydantic import ValidationError
    tools = make_tools(monkeypatch)
    field = "frontier_uid" if mode == "frontier" else "action_uid"
    client_model, requests = model_calls([
        {field: "invented", "reason": "not a domain action"},
        {"action": {"name": "propose_finish", "arguments": {}, "reason": "not a choice"}},
        {**decision_payload(mode), "task_complete": True},
    ])
    runtime = AgentRuntime(tools, tmp_path / "invalid.jsonl", max_model_calls=3, client_model=client_model)
    for _ in range(3):
        with pytest.raises(ValidationError):
            decision_call(runtime, mode)
    assert runtime.model_calls == 3 and len(requests) == 3
    assert runtime.history == []
    assert tools.motion.active_request_id is None and tools.motion.controller.sent == []
    assert not tools.propose_finish()["accepted"]
    with pytest.raises(RuntimeError, match="budget"):
        decision_call(runtime, mode)


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
@pytest.mark.parametrize("change", ["task", "search", "snapshot", "progress", "control_epoch", "motion_owner"])
def test_plain_decision_cannot_use_changed_context(monkeypatch, tmp_path, mode, change):
    tools = make_tools(monkeypatch)
    session = tools.session
    if change == "motion_owner":
        tools.submit_motion(1, 0, "existing", candidate_uid="cup-1", viewpoint_uid="view-1")

    def mutate(request):
        if change == "task":
            session.install_plan(runtime_plan(replace(session.spec, revision=2, exhaustive=True)))
        elif change == "search":
            session.search_revision += 1
        elif change == "snapshot":
            session.snapshot = replace(session.snapshot)
        elif change == "progress":
            session.progress.completed = True
        elif change == "control_epoch":
            tools.motion.hold()
            tools.motion.resume()
        else:
            tools.motion.controller.status = NavigationStatusCode.REACHED
            tools.motion.poll_active()
            tools.submit_motion(1, 0, "replacement", candidate_uid="cup-1", viewpoint_uid="view-1")
        return decision_payload(mode)

    client_model, requests = model_calls([mutate])
    runtime = AgentRuntime(tools, tmp_path / "stale-plain.jsonl", client_model=client_model)
    error = RuntimeError if change in {"control_epoch", "motion_owner"} else ValueError
    with pytest.raises(error, match="changed|stale"):
        decision_call(runtime, mode)
    assert runtime._choice_context is None and len(requests) == 1
    assert len(tools.motion.controller.sent) == (2 if change == "motion_owner" else 0)


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
@pytest.mark.parametrize("change", ["none", "task", "search", "snapshot", "pose", "progress", "control_epoch", "held"])
def test_guided_submission_rechecks_and_consumes_choice(monkeypatch, tmp_path, mode, change):
    tools = make_tools(monkeypatch)
    session = tools.session
    pose = Pose3D((1.123411, 2.234522, 3.345633), (.123456, 0, 0, .992350), frame_id="mapper_local")
    view = replace(session.snapshot.viewpoints[0], pose=pose)
    frontier = FrontierSnapshot("node:2", pose.position, metadata={"frame_id": pose.frame_id})
    session.snapshot = replace(session.snapshot, viewpoints=(view,), frontiers=(frontier,))
    snapshot = session.snapshot
    client_model, requests = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "guard.jsonl", client_model=client_model)
    if mode == "frontier":
        runtime.choose_frontier([{"frontier_uid": frontier.uid, "position": frontier.position}], frontier.uid, {})
        arguments = {"frontier_uid": frontier.uid}
    else:
        runtime.choose_recovery([{**asdict(view), "action_uid": "approach:u", "kind": "viewpoint",
                                  "candidate_uid": "cup-1"}], {})
        arguments = {"candidate_uid": "cup-1", "viewpoint_uid": view.uid}
    geometry = request_context(requests[0])["option_geometry"]
    displayed, = _table_rows(geometry["frontiers" if mode == "frontier" else "viewpoints"])
    assert displayed["position"] == [1.123, 2.235, 3.346]
    assert runtime._choice_context[0] is snapshot
    assert runtime._choice_context[1]["task"] == session.spec.as_dict()
    if change == "task":
        session.install_plan(runtime_plan(replace(session.spec, revision=2, exhaustive=True)))
    elif change == "search":
        session.search_revision += 1
    elif change == "snapshot":
        session.snapshot = replace(session.snapshot)
    elif change == "pose":
        changed_pose = replace(pose, position=(pose.position[0] + .00001, *pose.position[1:]))
        assert changed_pose != pose
        assert [round(value, 3) for value in changed_pose.position] == displayed["position"]
        session.snapshot = replace(snapshot, viewpoints=(replace(view, pose=changed_pose),),
                                   frontiers=(replace(frontier, position=changed_pose.position),))
    elif change == "progress":
        session.progress.completed = True
    elif change in {"control_epoch", "held"}:
        tools.motion.hold()
        if change == "control_epoch":
            tools.motion.resume()

    def submit():
        with runtime.choice_guard():
            tools.submit_motion(session.spec.revision, tools.motion.control_epoch, "guided", **arguments)

    if change == "none":
        submit()
        goal, = tools.motion.controller.sent
        assert goal.goal_pose.position == pose.position
        assert goal.goal_pose.frame_id == pose.frame_id
        if mode == "recovery":
            assert goal.goal_pose == pose
        assert tools.motion.active_intent.metadata["task_revision"] == session.spec.revision
    else:
        error = RuntimeError if change in {"control_epoch", "held"} else ValueError
        with pytest.raises(error):
            submit()
        assert tools.motion.controller.sent == []
    with pytest.raises(ValueError, match="No current guided choice"):
        submit()


@pytest.fixture
def interpretation_tools(monkeypatch):
    import instruction.compiler as compiler
    import perception.detector_adapter as detector
    from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES
    from prompting.schemas import ParsedInstruction

    tools = make_tools(monkeypatch)
    compiled = compiler.compile_parsed_instruction(
        "I need a flat surface to dry my wet cup.",
        ParsedInstruction(targets=[{"name": "cup"}]), RUNTIME_CONSTRAINT_TYPES)
    assert compiled.task is not None
    tools.session = TaskSession(tools.session.snapshot, runtime_plan(compiled.task))

    def forbidden(*args, **kwargs):
        pytest.fail("Interpretation correction must not make hidden parsing or grounding calls")

    monkeypatch.setattr(compiler, "parse_instruction_with_llm", forbidden)
    monkeypatch.setattr(detector, "_llm_ground_terms", forbidden)
    monkeypatch.setattr(detector, "get_client_and_model", forbidden)
    ground_plan = detector.ground_plan

    def ground_without_model(plan, **kwargs):
        assert kwargs["available_classes"] == ()
        assert kwargs["use_legacy_similarity_fallback"] is False
        return ground_plan(plan, **kwargs)

    monkeypatch.setattr(detector, "ground_plan", ground_without_model)
    return tools


def interpretation_payload(mode, *, parsed=None, basis_ids=None):
    from prompting.schemas import ParsedInstruction
    if parsed is None:
        parsed = {"targets": [{"name": "drying surface", "attributes": {"shape": "flat"},
            "demand": {"activity": "dry", "manipulated_objects": ["wet cup"],
                       "required_resource": "drying surface",
                       "destination_hypotheses": ["table", "counter"]}}]}
    payload = decision_payload(mode)
    if isinstance(payload, ToolCall):
        payload = payload.dict()
    return {**payload, "interpretation_update": {
        "interpretation": ParsedInstruction(**parsed).dict(),
        "basis_ids": ["instruction"] if basis_ids is None else basis_ids,
        "reason": "The cup is manipulated material; the requested destination is a drying resource.",
    }}


@pytest.mark.parametrize("mode,action", [
    ("frontier", None), ("recovery", None), ("step", "observe"), ("step", "query_scene"),
])
def test_interpretation_corrects_material_goal_and_archives_revision(
        interpretation_tools, monkeypatch, tmp_path, mode, action):
    from copy import deepcopy
    from task.reducer import apply_verification
    from verification.contracts import CandidateInstance

    tools = interpretation_tools
    session = tools.session
    old_task, old_plan = session.spec, session.plan
    old_target_id = old_task.targets[0].id
    stale_result = VerificationResult(**tools.verify_target("cup-1", old_target_id, 1))
    assert tools.propose_finish()["accepted"]
    session.revise_search([{
        "concept_id": old_target_id, "detector_terms": ["mug"], "basis_ids": ["instruction"],
        "reason": "Previous retrieval hypothesis",
    }], task_revision=1, search_revision=0, snapshot=session.snapshot)
    session.destination_hypotheses = [{"candidate_uid": "cup-1", "task_revision": 1}]
    old_progress, old_ledger = session.progress, session.verification_ledger
    old_anchor, old_pairs = session.anchor_search_ledger, session.relation_pair_ledger
    progress_before, evidence_before = old_progress.as_dict(), old_ledger.as_dict()
    search_before = deepcopy(session.search_hypotheses)

    def forbidden(*args, **kwargs):
        pytest.fail("Adoption must not dispatch, capture, verify, or poll motion")

    monkeypatch.setattr(tools, "dispatch", forbidden)
    monkeypatch.setattr(tools, "evidence_provider", forbidden)
    monkeypatch.setattr(tools.verifier, "verify", forbidden)
    monkeypatch.setattr(tools.motion.controller, "poll_status", forbidden)
    payload = interpretation_payload(mode, basis_ids=["instruction", "verification:0"])
    if action:
        payload["action"] = tool_call(name=action, arguments={}, reason="correct interpretation").action.dict()
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "interpretation.jsonl", max_model_calls=1, client_model=client_model)
    execution = {"event": "candidate_deferred", "task_revision": 1, "step": 32,
                 "candidate_uid": "cup-1", "reason": "view_options_exhausted"}
    runtime.record_event(execution)

    outcome = decision_call(runtime, mode)
    assert request_context(requests[0])["recent_tools"] == [execution]
    assert execution not in runtime.history
    assert json.dumps(execution) in runtime.trace_path.read_text()

    if mode == "step":
        assert outcome["result"] == {"interpretation_changed": True, "task_revision": 2}
    else:
        assert outcome is None  # The previously selected action is no longer applicable.
    assert len(requests) == runtime.model_calls == 1
    assert requests[0]["trace_label"] == "navigation_agent"
    assert requests[0]["model"] == "replay-model"

    assert session.spec.revision == 2 and session.spec.raw_instruction == old_task.raw_instruction
    target, = session.spec.targets
    assert target.name == "drying surface" and dict(target.attributes) == {"shape": "flat"}
    demand = session.spec.as_dict()["targets"][0]["demand"]
    assert demand == {"activity": "dry", "manipulated_objects": ["wet cup"],
                      "required_resource": "drying surface"}
    constraint, = session.spec.constraints
    assert (constraint.type, constraint.hardness, constraint.subject) == ("affordance", "hard", target.id)
    assert dict(constraint.value) == dict(target.demand)
    assert "table" in session.detector_terms and "counter" in session.detector_terms
    assert "table" not in target.aliases and "counter" not in target.aliases
    assert session.plan is not old_plan and session.progress is not old_progress
    assert session.verification_ledger is not old_ledger
    assert session.anchor_search_ledger is not old_anchor and session.relation_pair_ledger is not old_pairs
    assert session.search_revision == 0 and session.search_hypotheses == {}
    assert session.destination_hypotheses == [] and session.verification_ledger.history == []
    assert session.progress.target_progress[target.id].count == 0
    assert not tools.propose_finish()["accepted"]
    archived, = session.interpretation_history
    assert archived["previous_task"] == old_task.as_dict()
    assert archived["previous_progress"] == progress_before
    assert archived["previous_verification"] == evidence_before
    assert archived["previous_search_hypotheses"] == search_before
    assert archived["basis_ids"] == ["instruction", "verification:0"]
    supplied_basis = request_context(requests[0])["observation"]["search_context"]["basis"]
    assert archived["basis"] == supplied_basis
    assert archived["reason"] == payload["interpretation_update"]["reason"]
    assert "review" not in archived
    assert set(archived["changes"]) == {"target_0", "constraint_0"}
    assert archived["changes"]["target_0"]["before"]["name"] == "cup"
    assert archived["changes"]["target_0"]["after"]["name"] == "drying surface"
    events = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    adoption_event, = [event for event in events if event["event"] == "interpretation_adopted"]
    assert adoption_event["changes"] == archived["changes"]
    assert adoption_event["basis"] == archived["basis"]
    assert adoption_event["reason"] == archived["reason"]
    assert "review" not in adoption_event
    assert not any(event["event"] == "interpretation_review" for event in events)

    with pytest.raises(ValueError, match="revision"):
        tools.verify_target("cup-1", old_target_id, 1)
    assert not apply_verification(session.progress, target, CandidateInstance("cup-1", "cup"), stale_result, {})
    session.verification_ledger.put(old_task.raw_instruction, "cup-1", stale_result)
    assert tools.observe()["verification_records"] == []
    assert session.verification_ledger.get(old_task.raw_instruction, "cup-1") is None
    assert not tools.propose_finish()["accepted"]
    old_progress.completed = False
    old_ledger.reset()
    assert archived["previous_progress"] == progress_before
    assert archived["previous_verification"] == evidence_before
    assert tools.motion.controller.sent == []
    with pytest.raises(RuntimeError, match="budget"):
        decision_call(runtime, mode)


def test_complete_interpretation_installs_with_one_decision_call(interpretation_tools, tmp_path):
    tools = interpretation_tools
    payload = interpretation_payload("step")
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "single-decision.jsonl", max_model_calls=1,
                           client_model=client_model)

    assert runtime.step()["result"] == {"interpretation_changed": True, "task_revision": 2}

    assert len(requests) == runtime.model_calls == runtime.max_model_calls == 1
    assert requests[0]["trace_label"] == "navigation_agent"
    assert tools.session.spec.targets[0].name == "drying surface"
    assert len(tools.session.interpretation_history) == 1
    assert tools.motion.controller.sent == [] and not tools.propose_finish()["accepted"]
    with pytest.raises(ValueError, match="No current guided choice"):
        with runtime.choice_guard():
            pytest.fail("An adopted interpretation must discard the old executable choice")
    with pytest.raises(RuntimeError, match="budget"):
        runtime.step()
    assert len(requests) == 1


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_noop_interpretation_keeps_task_revision(interpretation_tools, tmp_path, mode):
    tools = interpretation_tools
    original, plan, progress = tools.session.spec, tools.session.plan, tools.session.progress
    payload = interpretation_payload(mode, parsed={"targets": [{"name": "cup"}]})
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "noop.jsonl", max_model_calls=1, client_model=client_model)
    result = decision_call(runtime, mode)
    if mode == "step":
        assert result["result"]["task"] == original.as_dict()
        assert "interpretation_changed" not in result["result"]
    else:
        assert result == ("node:2" if mode == "frontier" else "approach:u")
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec is original and tools.session.plan is plan and tools.session.progress is progress
    assert tools.session.search_revision == 0 and tools.session.interpretation_history == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("phase", ["active", "held", "cancel_pending"])
@pytest.mark.parametrize("during_decision", [False, True])
def test_interpretation_requires_idle_unheld_motion(interpretation_tools, tmp_path, phase, during_decision):
    tools = interpretation_tools
    original = tools.session.spec

    def change_motion():
        if phase == "held":
            tools.motion.hold()
        else:
            tools.submit_motion(1, 0, "existing", candidate_uid="cup-1", viewpoint_uid="view-1")
            if phase == "cancel_pending":
                tools.cancel_motion()
                assert tools.motion.phase == "cancel_requested"
                assert tools.motion.active_request_id == "existing"

    def decision(request):
        if during_decision:
            change_motion()
        return interpretation_payload("frontier")

    if not during_decision:
        change_motion()
    client_model, requests = model_calls([decision])
    runtime = AgentRuntime(tools, tmp_path / "owned.jsonl", client_model=client_model)
    # 响应期间改变运动状态由上下文守卫拒绝；已有运动由 idle_guard 拒绝。
    message = "motion state or control epoch changed" if during_decision else "idle, unheld"
    with pytest.raises(RuntimeError, match=message):
        decision_call(runtime, "frontier")
    assert len(requests) == runtime.model_calls == 1
    assert runtime._choice_context is None
    assert tools.session.spec is original and tools.session.interpretation_history == []
    assert len(tools.motion.controller.sent) == (0 if phase == "held" else 1)
    assert tools.motion.held == (phase == "held")
    assert tools.motion.active_request_id == (None if phase == "held" else "existing")


@pytest.mark.parametrize("change", ["task", "search", "snapshot", "control_epoch"])
def test_interpretation_rejects_stale_decision(interpretation_tools, tmp_path, change):
    tools = interpretation_tools
    session = tools.session
    concurrent_task = []

    def decision(request):
        if change == "task":
            session.install_plan(runtime_plan(replace(session.spec, revision=2, exhaustive=True)))
        elif change == "search":
            session.revise_search([{"concept_id": session.spec.targets[0].id, "detector_terms": ["mug"],
                                   "basis_ids": ["instruction"], "reason": "Concurrent retrieval"}],
                                  task_revision=1, search_revision=0, snapshot=session.snapshot)
        elif change == "snapshot":
            session.snapshot = replace(session.snapshot)  # Equal content is still a newer observation.
        else:
            tools.motion.hold()
            tools.motion.resume()  # Idle again, but the proposal's epoch is obsolete.
        concurrent_task.append(session.spec)
        return interpretation_payload("recovery")

    client_model, requests = model_calls([decision])
    runtime = AgentRuntime(tools, tmp_path / "stale-interpretation.jsonl", client_model=client_model)
    error = RuntimeError if change == "control_epoch" else ValueError
    with pytest.raises(error, match="stale|changed|epoch"):
        decision_call(runtime, "recovery")
    assert len(requests) == runtime.model_calls == 1
    assert runtime._choice_context is None
    assert session.spec is concurrent_task[0]
    assert session.interpretation_history == []
    if change == "search":
        assert session.search_revision == 1 and session.detector_terms == ["mug"]
    assert not tools.propose_finish()["accepted"] and tools.motion.controller.sent == []


@pytest.mark.parametrize("basis", [[], ["verification:0"], ["instruction", "verification:99"],
                                  ["instruction", "frame:missing"]])
def test_interpretation_requires_instruction_and_supplied_evidence(interpretation_tools, tmp_path, basis):
    tools = interpretation_tools
    session = tools.session
    original = session.spec
    session.verification_ledger.put(original.raw_instruction, "cup-1", VerificationResult(
        False, "need_better_view", task_revision=1, reason="Unresolved resource"))
    session.snapshot = replace(session.snapshot, metadata={"observation_images": [
        {"id": "missing", "image_ref": "", "observed_object_uids": [], "object_boxes": {}},
    ]})
    client_model, requests = model_calls([interpretation_payload("frontier", basis_ids=basis)])
    runtime = AgentRuntime(tools, tmp_path / "bad-basis.jsonl", client_model=client_model)
    with pytest.raises(ValueError):
        decision_call(runtime, "frontier")
    supplied = request_context(requests[0])["observation"]["search_context"]["basis"]
    assert set(supplied) == {"instruction", "verification:0"}
    assert len(requests) == runtime.model_calls == 1
    assert session.spec is original and session.interpretation_history == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("invalid", ["unsupported", "unresolved", "empty", "metadata", "raw_instruction"])
def test_interpretation_compiler_and_source_gates_precede_installation(interpretation_tools, tmp_path, invalid):
    tools = interpretation_tools
    payload = interpretation_payload("frontier")
    parsed = payload["interpretation_update"]["interpretation"]
    if invalid == "unsupported":
        parsed["constraints"] = [{"type": "teleport", "subject": "drying surface", "hardness": "hard"}]
    elif invalid == "unresolved":
        parsed["unresolved"] = ["The intended resource is ambiguous"]
    elif invalid == "empty":
        parsed["targets"] = []
    elif invalid == "metadata":
        tools.session = TaskSession(tools.session.snapshot, runtime_plan(replace(tools.session.spec, source="dataset")))
    else:
        payload["interpretation_update"]["raw_instruction"] = "Go to any nearby object"
    original, plan = tools.session.spec, tools.session.plan
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "invalid-interpretation.jsonl", client_model=client_model)
    if invalid in {"metadata", "raw_instruction"}:
        with pytest.raises(ValueError):
            decision_call(runtime, "frontier")
    else:
        assert decision_call(runtime, "frontier") is None
        feedback = runtime.history[-1]["interpretation_result"]
        assert feedback["interpretation_changed"] is False and feedback["reason"]
        assert feedback["interpretation_status"] in {"invalid", "unsupported", "ambiguous"}
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec is original and tools.session.plan is plan
    assert tools.session.interpretation_history == [] and tools.motion.controller.sent == []


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_interpretation_and_search_revisions_are_mutually_exclusive(interpretation_tools, tmp_path, mode):
    tools = interpretation_tools
    payload = interpretation_payload(mode)
    payload["search_updates"] = search_payload(
        mode, ["mug"], concept_id=tools.session.spec.targets[0].id)["search_updates"]
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "mixed-revisions.jsonl", client_model=client_model)
    with pytest.raises(ValueError, match="mutually exclusive"):
        decision_call(runtime, mode)
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec.revision == 1 and tools.session.search_revision == 0
    assert tools.session.search_hypotheses == {} and tools.session.interpretation_history == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("name,arguments", [
    ("query_viewpoints", {"candidate_uid": "cup-1"}),
    ("motion_status", {}), ("cancel_motion", {}), ("propose_finish", {}),
    ("verify_target", {"candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}),
    ("submit_motion", {"candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
                       "control_epoch": 0, "request_id": "must-not-move"}),
])
def test_interpretation_cannot_accompany_other_tools(interpretation_tools, tmp_path, name, arguments):
    tools = interpretation_tools
    payload = interpretation_payload("step")
    payload["action"] = tool_call(name=name, arguments=arguments, reason="mixed turn").action.dict()
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "mixed-action.jsonl", client_model=client_model)
    with pytest.raises(ValueError, match="not an execution action"):
        runtime.step()
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec.revision == 1 and tools.session.interpretation_history == []
    assert tools.motion.control_epoch == 0 and tools.motion.controller.sent == []
    assert tools.session.verification_ledger.history == [] and not tools.propose_finish()["accepted"]


@pytest.fixture
def habitat_evidence_tools(monkeypatch, tmp_path):
    import os
    import sys
    from types import MethodType
    from unittest.mock import Mock

    import numpy as np
    from PIL import Image
    from backends.habitat.agent_runtime import HabitatNavigationTools
    from instruction_adapter.verifier import hard_stop_constraints_from_evidence
    from instruction_adapter.view_control import ViewControlState
    from test_navigation_root_causes import methods

    projections = {
        "cup-1": [[1, 1, 1.], [3, 1, 1.], [1, 3, 1.]],
        "table-1": [[4, 1, 1.], [6, 1, 1.], [4, 4, 1.]],
        "occluded-1": [[1, 1, 2.], [3, 1, 2.], [1, 3, 2.]],
    }
    objects = []
    for uid, points in projections.items():
        points = np.array(points)
        positions = SimpleNamespace(cpu=lambda points=points: SimpleNamespace(numpy=lambda: points))
        objects.append(SimpleNamespace(
            uid=uid, tag=uid.split("-")[0], position=points.mean(axis=0),
            bbox=[0, 0, 8, 6], confidence=.9,
            pcd=SimpleNamespace(point=SimpleNamespace(positions=positions)),
        ))
    monkeypatch.setitem(sys.modules, "mapping_utils.geometry", SimpleNamespace(
        project_to_camera=lambda pcd, *_: pcd.point.positions.cpu().numpy().T))
    writer = Mock(return_value=True)
    loaded = methods("objnav_agent_with_process_obs.py", {
        "_build_final_verifier_evidence", "_observation_identity", "_attach_view_control_context",
    }, {"np": np, "os": os, "json": json, "cv2": SimpleNamespace(imwrite=writer),
        "hard_stop_constraints_from_evidence": hard_stop_constraints_from_evidence})
    session = make_tools(monkeypatch).session
    controller = Controller()
    agent = SimpleNamespace(
        task=session, episode_samples=3, episode_steps=16, save_dir=str(tmp_path), current_node_idx=0,
        mapper=SimpleNamespace(objects=objects, nodes=[], camera_intrinsic=np.eye(3),
                               current_position=np.array([1., 2., 3.]), current_rotation=np.eye(3)),
        execution=SimpleNamespace(controller=controller, motion=MotionSupervisor(controller), step=Mock()),
        view_control_state=ViewControlState(), final_instruction_verifier=Mock(),
        _augment_final_stop_geometry=lambda geometry: {**geometry, "distance_to_object": .5,
                                                       "required_stop_distance": 1.},
        _view_quality_facts=lambda geometry: {"visible_projected_points": 3},
        _nearby_objects_for_final_evidence=lambda: [], _overlapping_instance_observations=lambda path: [],
        _raw_instruction_for_verifier=lambda: session.spec.raw_instruction,
    )
    for name in ("_build_final_verifier_evidence", "_observation_identity", "_attach_view_control_context"):
        setattr(agent, name, MethodType(loaded[name], agent))
    agent._current_instance_image = lambda bbox: agent.rgb_trajectory[-1].copy()
    bbox = SimpleNamespace(squeeze=lambda dim: SimpleNamespace(
        cpu=lambda: SimpleNamespace(numpy=lambda: np.array([1, 1, 4, 4]))))
    agent._project_object_bbox_on_current_view = Mock(return_value=(bbox, {"visible_projected_points": 3}))

    def frame(depth, rotation):
        agent.episode_steps += 1
        agent.mapper.current_rotation = rotation.copy()
        rgb = np.full((6, 8, 3), agent.episode_steps, dtype=np.uint8)
        agent.obs = {"depth": np.full((6, 8, 1), depth), "rgb": rgb}
        agent.rgb_trajectory = [rgb]
        path = tmp_path / "episode-2" / "rgb" / f"monitor-rgb_{agent.episode_steps}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return True

    frame(2., np.eye(3))
    pitched = np.array([[1., 0., 0.], [0., .866, .5], [0., -.5, .866]])
    rotate = Mock(side_effect=lambda current: frame(1., pitched))
    tilt = Mock(side_effect=lambda current, pitch: frame(2., np.eye(3)))
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller", SimpleNamespace(
        rotate_toward_object_for_recheck=rotate, tilt_camera=tilt))
    tools = HabitatNavigationTools(agent)
    tools.refresh_scene()
    yield SimpleNamespace(tools=tools, rotate=rotate, tilt=tilt, writer=writer)
    agent.execution.step.assert_not_called()
    assert controller.sent == []
    assert tools.motion.active_request_id is None


def test_final_builder_persists_current_covisible_frame_before_attach(habitat_evidence_tools):
    from instruction_adapter.verifier import candidate_from_object
    fixture = habitat_evidence_tools
    tools, agent = fixture.tools, fixture.tools.agent
    previous = tools.observe()["observation_images"]
    fixture.rotate(agent)
    agent.object_final = tools.objects["cup-1"]
    candidate = candidate_from_object(agent.object_final)
    bbox, geometry = agent._project_object_bbox_on_current_view()

    evidence = agent._build_final_verifier_evidence(candidate, bbox, geometry)

    saved = Path(agent.save_dir) / "episode-2" / "final_verifier" / "evidence_18.json"
    assert json.loads(saved.read_text())["evidence"] == evidence
    frame, = evidence["observation_images"]
    identity = evidence["observation_identity"]
    assert frame["id"] == identity["observation_id"] == "2:18"
    assert identity["task_revision"] == 1 and identity["candidate_uid"] == "cup-1"
    assert frame["position"] == identity["sensor_position"] == agent.mapper.current_position.tolist()
    assert frame["rotation"] == identity["sensor_rotation"] == agent.mapper.current_rotation.tolist()
    assert frame["observed_object_uids"] == ["cup-1", "table-1"]
    assert frame["object_boxes"] == {"cup-1": [1, 1, 4, 4], "table-1": [4, 1, 7, 5]}
    image = Path(agent.save_dir) / "episode-2" / "rgb" / "monitor-rgb_18.jpg"
    assert frame["image_ref"] == str(image) and image.is_file()
    assert tools.observe()["observation_images"] == previous
    assert previous[0]["id"] != frame["id"]

    override = {"observation_images": previous,
                "geometry": {"distance_to_object": 2., "required_stop_distance": 1.}}
    attached = agent._attach_view_control_context(override, candidate)
    assert attached["observation_images"] == [frame]
    assert not attached["hard_stop_constraints"]["satisfied"]
    assert override["observation_images"] == previous
    agent.final_instruction_verifier.verify.assert_not_called()
    fixture.tilt.assert_not_called()


@pytest.mark.parametrize("has_bbox", [True, False])
def test_capture_refreshes_current_snapshot_even_without_bbox(habitat_evidence_tools, has_bbox):
    from backends.habitat.scene import current_observation
    fixture = habitat_evidence_tools
    tools, agent = fixture.tools, fixture.tools.agent
    before = tools.session.snapshot
    if not has_bbox:
        agent._project_object_bbox_on_current_view.return_value = (None, {})

    evidence = tools.capture_evidence("cup-1")

    frame = current_observation(agent)
    assert before.metadata["observation_images"][0]["id"] == "2:17"
    assert tools.session.snapshot.timestamp == frame["step"] == 18
    assert tools.session.snapshot.metadata["observation_images"] == [frame]
    assert tools.observe()["observation_images"] == [frame]
    assert frame["observed_object_uids"] == ["cup-1", "table-1"]
    assert evidence["observation_identity"]["observation_id"] == frame["id"]
    if has_bbox:
        assert evidence["observation_images"] == [frame]
    else:
        assert evidence["observation_status"] == "need_observation"
        fixture.writer.assert_not_called()
    fixture.rotate.assert_called_once_with(agent)
    fixture.tilt.assert_not_called()
    agent.final_instruction_verifier.verify.assert_not_called()
    assert not tools.propose_finish()["accepted"]


@pytest.mark.parametrize("accepted", [False, True])
def test_verification_publishes_restored_frame_only_when_incomplete(habitat_evidence_tools, accepted):
    from backends.habitat.scene import current_observation
    fixture = habitat_evidence_tools
    tools, agent = fixture.tools, fixture.tools.agent
    verifier = agent.final_instruction_verifier.verify
    verifier.return_value = VerificationResult(
        accepted, "accept" if accepted else "need_better_view", semantic_satisfied=True,
        view_sufficient_for_stop=accepted, task_revision=1, target_uid="cup-1")

    result = tools.verify_target("cup-1", "cup", 1)

    assert result["decision"] == verifier.return_value.decision
    assert tools.session.progress.completed == accepted
    assert tools.propose_finish()["accepted"] == accepted
    fixture.rotate.assert_called_once_with(agent)
    verifier.assert_called_once()
    captured = verifier.call_args.kwargs["evidence"]
    assert captured["observation_images"][0]["id"] == "2:18"
    frame = current_observation(agent)
    assert tools.session.snapshot.metadata["observation_images"] == [frame]
    assert tools.observe()["observation_images"] == [frame]
    assert tools.session.snapshot.timestamp == frame["step"]
    if accepted:
        fixture.tilt.assert_not_called()
        assert frame["id"] == "2:18"
        assert agent.accepted_candidate_uid == "cup-1"
    else:
        fixture.tilt.assert_called_once_with(agent, 0.0)
        assert frame["id"] == "2:19" and frame["observed_object_uids"] == ["occluded-1"]
        assert frame["rotation"] != captured["observation_images"][0]["rotation"]
        stale = tools.verifier.verify(**verifier.call_args.kwargs)
        assert stale.decision == "need_better_view" and not stale.accepted
        verifier.assert_called_once()


def test_habitat_verification_revision_gate_precedes_capture(habitat_evidence_tools):
    fixture = habitat_evidence_tools
    with pytest.raises(ValueError, match="revision"):
        fixture.tools.verify_target("cup-1", "cup", 2)
    fixture.rotate.assert_not_called()
    fixture.tilt.assert_not_called()
    fixture.tools.agent.final_instruction_verifier.verify.assert_not_called()
    assert fixture.tools.session.verification_ledger.history == []


def test_habitat_refreshed_evidence_preserves_stale_geometry_gate(habitat_evidence_tools):
    fixture = habitat_evidence_tools
    tools = fixture.tools

    def changed_scene(**kwargs):
        snapshot = tools.session.snapshot
        objects = (replace(snapshot.objects[0], position=(8., 0., 0.)),) + snapshot.objects[1:]
        tools.session.snapshot = replace(snapshot, objects=objects)
        return VerificationResult(True, "accept", semantic_satisfied=True,
                                  task_revision=1, target_uid="cup-1")

    tools.agent.final_instruction_verifier.verify.side_effect = changed_scene
    with pytest.raises(ValueError, match="dependencies changed"):
        tools.verify_target("cup-1", "cup", 1)
    assert not tools.propose_finish()["accepted"]
    assert tools.session.verification_ledger.history == []
    fixture.rotate.assert_called_once_with(tools.agent)
    fixture.tilt.assert_not_called()
