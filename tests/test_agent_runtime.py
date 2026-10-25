"""Replay Agent calls through real task/query/motion/verifier tool bindings."""
from types import SimpleNamespace
import json

import pytest

from harness.runtime import AgentRuntime, ToolCall
from harness.tools import NavigationTools
from instruction.contracts import TaskSpec, TargetSpec
from instruction_adapter.task_compat import runtime_plan
from motion.contracts import NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot, ViewpointSnapshot
from task.session import TaskSession
from verification.contracts import VerificationResult


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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=next(calls)))])

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
    client_model, requests = model_calls([ToolCall(name=n, arguments=a, reason="replay") for n, a in specs])
    trace = tmp_path / "events.jsonl"
    runtime = AgentRuntime(tools, trace, max_model_calls=len(specs), client_model=client_model)
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
    assert len(json.loads(requests[-1]["messages"][1]["content"])["recent_tools"]) == 4
    with pytest.raises(RuntimeError, match="budget"):
        runtime.step()


def test_model_motion_with_stale_revision_is_recorded_and_propagated(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    client_model, _ = model_calls([ToolCall(name="submit_motion", arguments={
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
        call = ToolCall(name="submit_motion", arguments={
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


def test_structured_tool_arguments_expose_frontier_submission_fields():
    from harness.runtime import ToolArguments
    schema = ToolArguments.schema()
    assert {"frontier_uid", "task_revision", "control_epoch", "request_id"} <= set(schema["properties"])
    call = ToolCall(name="submit_motion", arguments={
        "frontier_uid": "node:2", "task_revision": 1, "control_epoch": 0, "request_id": "r1",
    }, reason="explore")
    assert call.arguments.dict(exclude_none=True) == {
        "frontier_uid": "node:2", "task_revision": 1, "control_epoch": 0, "request_id": "r1",
    }
