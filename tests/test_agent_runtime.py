"""Replay Agent calls through real task/query/motion/verifier tool bindings."""
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
import base64
import hashlib
import json
import re

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
                                  task_revision=1, target_uid=candidate.uid,
                                  diagnostics={"original_instruction_review": {
                                      "coverage_complete": True, "interpretation_consistent": True,
                                      "task_satisfied": True}})


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


@pytest.mark.parametrize("stage", ["initial", "arrival", "tool"])
def test_habitat_nested_skill_budget_returns_normal_termination(monkeypatch, tmp_path, stage):
    from unittest.mock import Mock
    import backends.habitat.agent_runtime as habitat_runtime
    import navigation.planner_loop as planner
    from backends.habitat.motion import HabitatExecution

    episode = SimpleNamespace(episode_id="0", scene_id="scene", start_position=[0, 0, 0],
                              start_rotation=[0, 0, 0, 1])
    sim = SimpleNamespace(sim_config=SimpleNamespace(agents=[SimpleNamespace(
        action_space={1: SimpleNamespace(actuation=SimpleNamespace(amount=.25))})]))
    env = SimpleNamespace(current_episode=episode, sim=sim, episode_over=False,
                          step=Mock(return_value={}))
    execution = HabitatExecution(env)
    node = SimpleNamespace(idx=0)
    agent = SimpleNamespace(env=env, save_dir=str(tmp_path), vlm="test", episode_steps=1,
        current_node_idx=0, execution=execution, task=SimpleNamespace(
            search_revision=0, progress=SimpleNamespace(completed=False)),
        mapper=SimpleNamespace(nodes=[node], change_state=Mock()))
    execution.controller.agent = agent
    execution.controller.advance = Mock()
    motion = SimpleNamespace(active_request_id="frontier" if stage == "arrival" else None)
    def poll():
        motion.active_request_id = None
        return SimpleNamespace(tracking_active=False, status=NavigationStatus(NavigationStatusCode.REACHED),
            intent=SimpleNamespace(metadata={"frontier_uid": "node:0"}, target_object_uid=None,
                                   goal_pose=Pose3D((1, 0, 0))))
    motion.poll_active = poll
    tools = SimpleNamespace(motion=motion, refresh_scene=Mock(), exploration_options=Mock(),
                            motion_finished=Mock())
    runtime = SimpleNamespace(model_calls=0, record_event=Mock())
    def nested_actions():
        for _ in range(12):
            execution.step(3)
            agent.episode_steps += 1
    def observe(current, *, node, episode_idx):
        if stage == "initial" or node is not None:
            nested_actions()
        return planner.PlanningCycleResult(agent.episode_steps)
    def decide():
        assert stage == "tool"
        assert tools.movement_budget_remaining == 2
        runtime.model_calls += 1
        nested_actions()
    runtime.step = decide
    monkeypatch.setattr(habitat_runtime, "HabitatNavigationTools", lambda current: tools)
    monkeypatch.setattr(habitat_runtime, "AgentRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(planner, "run_observation_mapping_cycle", observe)

    result = habitat_runtime.run_agent_episode(agent, episode_idx=0, max_steps=2, stop_on_success=False)

    assert result["reason"] == "episode_limit" and not result["task_complete"]
    assert result["steps"] == 3 and result["model_calls"] == int(stage == "tool")
    assert execution.controller.action_count == env.step.call_count == 2
    assert execution.motion.active_request_id is None and motion.active_request_id is None
    assert tools.motion_finished.call_count == int(stage == "arrival")
    assert json.loads((tmp_path / "episode-0/harness/termination.json").read_text()) == result
    agent.mapper.change_state.assert_not_called()


@pytest.mark.parametrize("supported", [False, True])
def test_scan_schema_is_exposed_only_by_capable_backend(monkeypatch, tmp_path, supported):
    from pydantic import ValidationError
    tools = make_tools(monkeypatch)
    if supported:
        tools.names = (*tools.names, "scan_scene")
        tools.scan_scene = lambda: {"scan_complete": True}
    runtime = AgentRuntime(tools, tmp_path / "tools.jsonl", client_model=model_calls([])[0])
    response = runtime._response_type("ToolCall", runtime._messages(runtime.system_prompt), base=ToolCall)
    payload = {"action": {"name": "scan_scene", "arguments": {}, "reason": "unseen headings"}}
    if supported:
        assert response.model_validate(payload).action.name == "scan_scene"
    else:
        with pytest.raises(ValidationError):
            response.model_validate(payload)


def test_habitat_view_queries_do_not_consume_execution_attempts(monkeypatch):
    import sys
    import numpy as np
    from unittest.mock import Mock
    from backends.habitat.agent_runtime import HabitatNavigationTools
    base = make_tools(monkeypatch)
    obj = SimpleNamespace(rgb=np.zeros((4, 4, 3), dtype=np.uint8), bbox=[0, 0, 4, 4])
    agent = SimpleNamespace(task=base.session, execution=SimpleNamespace(
        motion=base.motion, controller=base.motion.controller), final_instruction_verifier=base.verifier,
        mapper=SimpleNamespace(current_position=np.zeros(3)), planner=SimpleNamespace(_goal_radius=.2),
        episode_steps=1, find_final_waypoint=Mock())
    tools = HabitatNavigationTools(agent)
    tools.objects = {"cup-1": obj}
    proposals = [{"position": [1., 0., 0.], "score": .8, "stop_feasible": True, "travel_distance": 1.}]
    monkeypatch.setitem(sys.modules, "navigation.view_verification_controller", SimpleNamespace(
        build_viewpoint_options=lambda current: (proposals, [])))
    monkeypatch.setitem(sys.modules, "navigation.action_controller", SimpleNamespace(
        geodesic_distance_to_waypoint=lambda current, position: float(np.linalg.norm(position))))
    first = tools.query_viewpoints("cup-1")
    assert first == tools.query_viewpoints("cup-1")
    assert not agent.view_control_state.proposals[0].attempted
    tools.submit_motion(1, 0, "view-1", candidate_uid="cup-1", viewpoint_uid=first[0]["uid"])
    assert not agent.view_control_state.proposals[0].attempted
    base.motion.controller.status = NavigationStatusCode.BLOCKED
    tools.motion_finished(base.motion.poll_active())
    assert tools.query_viewpoints("cup-1") == []
    assert "cup-1" in tools.deferred_candidates
    key = (1, "cup-1", tools.candidate_evidence(obj))
    assert tools.candidate_motion_goals[key][(1., 0., 0.)] == "blocked"
    with pytest.raises(ValueError):
        tools.submit_motion(1, 0, "stale-view", candidate_uid="cup-1", viewpoint_uid=first[0]["uid"])


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


@pytest.mark.parametrize("missing", ["coverage_complete", "interpretation_consistent", "task_satisfied"])
def test_structured_progress_never_replaces_original_instruction_review(monkeypatch, missing):
    tools = make_tools(monkeypatch)
    original_verify = tools.verifier.verify

    def incomplete(**kwargs):
        result = original_verify(**kwargs)
        result.diagnostics["original_instruction_review"][missing] = False
        return result

    tools.verifier.verify = incomplete
    tools.verify_target("cup-1", "cup", 1)
    assert tools.session.progress.completed
    assert not tools.propose_finish()["accepted"]
    assert not tools.propose_finish()["original_instruction_satisfied"]


@pytest.mark.parametrize("change", ["observation", "pose", "object", "revision", "motion"])
def test_full_instruction_finish_receipt_expires_with_its_execution_context(monkeypatch, change):
    tools = make_tools(monkeypatch)
    tools.verify_target("cup-1", "cup", 1)
    assert tools.propose_finish()["accepted"]
    snapshot = tools.session.snapshot
    if change == "observation":
        tools.session.snapshot = replace(snapshot, timestamp=snapshot.timestamp + 1)
    elif change == "pose":
        tools.session.snapshot = replace(snapshot, robot_pose=Pose3D((1, 1, 0)))
    elif change == "object":
        tools.session.snapshot = replace(snapshot, objects=(replace(snapshot.objects[0], position=(9, 0, 0)),))
    elif change == "revision":
        tools.session.install_plan(runtime_plan(replace(tools.session.spec, revision=2)))
    else:
        tools.submit_motion(1, 0, "after-verification", candidate_uid="cup-1", viewpoint_uid="view-1")
        tools.motion.controller.status = NavigationStatusCode.REACHED
        tools.motion.poll_active()
    assert not tools.propose_finish()["accepted"]


@pytest.mark.parametrize("in_place", [False, True])
def test_finish_receipt_expires_on_image_change_at_same_timestamp_and_pose(monkeypatch, in_place):
    tools = make_tools(monkeypatch)
    images = [{"id": "frame-1", "image_ref": "", "observed_object_uids": ["cup-1"],
               "object_boxes": {"cup-1": [0, 0, 2, 2]}}]
    tools.session.snapshot = replace(tools.session.snapshot, metadata={"observation_images": images})
    tools.verify_target("cup-1", "cup", 1)
    assert tools.propose_finish()["accepted"]
    snapshot = tools.session.snapshot
    progress = tools.session.progress.as_dict()
    ledger = tools.session.verification_ledger.as_dict()
    if in_place:
        images[0]["object_boxes"]["cup-1"][0] = 1
        assert tools.session.snapshot is snapshot
        assert tools.session.snapshot.metadata["observation_images"] is images
    else:
        tools.session.snapshot = replace(snapshot, metadata={"observation_images": [
            {**images[0], "id": "replacement-frame"}]})
    assert tools.session.snapshot.timestamp == snapshot.timestamp
    assert tools.session.snapshot.robot_pose == snapshot.robot_pose
    assert tools.session.progress.as_dict() == progress and tools.session.progress.completed
    assert tools.session.verification_ledger.as_dict() == ledger
    assert tools.session.spec.revision == 1 and tools.motion.control_epoch == 0
    assert tools.motion.controller.sent == []
    result = tools.propose_finish()
    assert result["accepted"] is False and result["original_instruction_satisfied"] is False
    assert result["task_complete"] is True and result["verified_target_id"] is None


def test_zero_movement_budget_still_allows_verification_and_finish(monkeypatch):
    tools = make_tools(monkeypatch)
    tools.movement_budget_remaining = 0
    result = tools.submit_motion(1, 0, "over-budget", candidate_uid="cup-1", viewpoint_uid="view-1")
    assert result == {"accepted": False, "reason": "movement_budget_exhausted"}
    assert tools.motion.controller.sent == []
    tools.verify_target("cup-1", "cup", 1)
    assert tools.propose_finish()["accepted"]


def test_every_turn_retains_authoritative_full_original_after_history_truncation(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    raw = "Find a red cup, not the blue one, near the small tray with fresh snacks."
    tools.session.install_plan(runtime_plan(TaskSpec(raw, (TargetSpec("cup", "cup"),), revision=2)))
    calls = [tool_call(name="observe", arguments={}, reason="Inspect original requirements") for _ in range(6)]
    client_model, requests = model_calls(calls)
    runtime = AgentRuntime(tools, tmp_path / "tools.jsonl", client_model=client_model)
    for _ in calls:
        runtime.step()
    for request in requests:
        observation = request_context(request)["observation"]
        assert observation["raw_instruction"] == raw
        assert observation["semantic_authority"] == "raw_instruction"
        assert observation["structured_interpretation_role"] == "revisable_execution_aid"
        assert "final semantic authority" in request["messages"][0]["content"]
    assert len(runtime.history) == 4


def _assert_proposal_rejected(runtime, outcome, match):
    assert outcome["event"] == "returned"
    assert outcome["result"]["accepted"] is False
    assert outcome["result"]["proposal_rejected"] is True
    assert re.search(match, outcome["result"]["reason"])
    assert "error" not in outcome and "error_type" not in outcome
    assert runtime.history[-1] == outcome
    events = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    assert events[-2]["event"] == "proposed"
    assert events[-1] == json.loads(json.dumps(outcome))


def test_model_motion_with_stale_revision_is_recorded_as_rejected(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    client_model, _ = model_calls([tool_call(name="submit_motion", arguments={
        "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 2,
        "control_epoch": 0, "request_id": "stale",
    }, reason="stale")])
    trace = tmp_path / "events.jsonl"
    runtime = AgentRuntime(tools, trace, client_model=client_model)
    _assert_proposal_rejected(runtime, runtime.step(), "revision")
    assert tools.motion.controller.sent == []
    assert not tools.session.progress.completed
    assert tools.session.spec.revision == 1 and tools.motion.control_epoch == 0


@pytest.mark.parametrize("name,invalid,match", [
    ("query_viewpoints", {"candidate_uid": "missing"}, "Unknown observed instance"),
    ("submit_motion", {"candidate_uid": "missing"}, "Unknown observed instance"),
    ("submit_motion", {"frontier_uid": "missing"}, "Unknown observed frontier"),
    ("submit_motion", {"viewpoint_uid": "missing"}, "not associated"),
    ("submit_motion", {"viewpoint_uid": "unrelated-view"}, "not associated"),
    ("submit_motion", {"control_epoch": 0}, "control epoch"),
    ("submit_motion", {"task_revision": 0}, "task revision"),
    ("verify_target", {"candidate_uid": "missing"}, "Unknown observed instance"),
    ("verify_target", {"target_id": "missing"}, "Unknown terminal target"),
    ("verify_target", {"target_id": "anchor"}, "Unknown terminal target"),
    ("verify_target", {"task_revision": 0}, "task revision"),
])
def test_invalid_proposal_allows_legal_reselection_without_backend_calls(
        monkeypatch, tmp_path, name, invalid, match):
    from scene.contracts import FrontierSnapshot

    tools = make_tools(monkeypatch)
    snapshot = replace(tools.session.snapshot,
        viewpoints=tools.session.snapshot.viewpoints + (
            ViewpointSnapshot("unrelated-view", Pose3D((9, 0, 0))),),
        frontiers=(FrontierSnapshot("frontier-1", (3, 2, 1), metadata={"frame_id": "map"}),))
    task = replace(tools.session.spec, targets=tools.session.spec.targets + (
        TargetSpec("anchor", "table", role="anchor", terminal=False),))
    tools.session = TaskSession(snapshot, runtime_plan(task))
    if "control_epoch" in invalid:
        tools.motion.hold()
        tools.motion.resume()
    epoch = tools.motion.control_epoch
    valid = {"candidate_uid": "cup-1"}
    if name == "verify_target":
        valid.update(target_id="cup", task_revision=1)
    elif name == "submit_motion":
        valid.update(viewpoint_uid="view-1", task_revision=1, control_epoch=epoch, request_id="reselect")
        if "frontier_uid" in invalid:
            valid.pop("candidate_uid")
            valid.pop("viewpoint_uid")
            valid["frontier_uid"] = "frontier-1"
    invalid_arguments = {**valid, **invalid}
    method = getattr(tools, name)
    invoked = []

    def execute(**arguments):
        invoked.append(arguments)
        return method(**arguments)

    monkeypatch.setattr(tools, name, execute)
    client_model, requests = model_calls([
        tool_call(name=name, arguments=invalid_arguments, reason="inadmissible proposal"),
        tool_call(name=name, arguments=valid, reason="reselect using current observed IDs"),
    ])
    runtime = AgentRuntime(tools, tmp_path / "reselect.jsonl", max_model_calls=2, client_model=client_model)
    progress = tools.session.progress.as_dict()
    ledger = tools.session.verification_ledger.as_dict()
    rejection = runtime.step()
    _assert_proposal_rejected(runtime, rejection, match)
    assert invoked == []
    assert tools.motion.controller.sent == [] and tools.motion.active_request_id is None
    assert tools.last_goal_id is None and tools.motion.control_epoch == epoch
    assert tools.session.spec == task and tools.session.search_revision == 0
    assert tools.session.progress.as_dict() == progress
    assert tools.session.verification_ledger.as_dict() == ledger
    assert tools.session.verification_ledger.history == []
    assert not tools.propose_finish()["accepted"]

    outcome = runtime.step()
    assert outcome["event"] == "returned"
    assert invoked == [valid]
    assert request_context(requests[1])["recent_tools"] == [rejection]
    assert tools.session.spec == task and tools.session.search_revision == 0
    assert tools.motion.control_epoch == epoch and runtime.model_calls == len(requests) == 2
    if name == "submit_motion":
        assert outcome["result"]["goal_id"] == "goal-1"
        assert len(tools.motion.controller.sent) == 1
        goal, = tools.motion.controller.sent
        assert goal.goal_pose.position == ((3, 2, 1) if "frontier_uid" in valid else (1, 0, 0))
        assert tools.motion.active_request_id == "reselect"
        assert tools.session.progress.as_dict() == progress
        assert tools.session.verification_ledger.history == []
        assert not tools.propose_finish()["accepted"]
    elif name == "verify_target":
        assert VerificationResult(**outcome["result"]).accepted
        assert tools.session.progress.target_progress["cup"].accepted_candidate_uids == ["cup-1"]
        assert len(tools.session.verification_ledger.history) == 1
        assert tools.propose_finish()["accepted"]
        assert tools.motion.controller.sent == []
    else:
        assert [view["uid"] for view in outcome["result"]] == ["view-1"]
        assert tools.session.progress.as_dict() == progress
        assert tools.session.verification_ledger.history == []
        assert tools.motion.controller.sent == []
        assert not tools.propose_finish()["accepted"]
    events = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["proposed", "returned", "proposed", "returned"]


@pytest.mark.parametrize("failure", ["model", "send_goal", "evidence_provider", "verifier"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_execution_failures_propagate_original_exception_not_proposal_rejection(
        monkeypatch, tmp_path, failure, error_type):
    tools = make_tools(monkeypatch)
    error = error_type(f"{failure} service failed")
    invoked = []

    def fail(*args, **kwargs):
        invoked.append((args, kwargs))
        raise error

    if failure == "send_goal":
        monkeypatch.setattr(tools.motion.controller, "send_goal", fail)
        proposal = tool_call(name="submit_motion", arguments={
            "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
            "control_epoch": 0, "request_id": "backend-failure"}, reason="valid motion")
    else:
        proposal = tool_call(name="verify_target", arguments={
            "candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}, reason="valid verification")
        if failure == "evidence_provider":
            monkeypatch.setattr(tools, "evidence_provider", fail)
        elif failure == "verifier":
            monkeypatch.setattr(tools.verifier, "verify", fail)
    client_model, requests = model_calls([fail if failure == "model" else proposal])
    runtime = AgentRuntime(tools, tmp_path / "failure.jsonl", client_model=client_model)
    progress = tools.session.progress.as_dict()
    with pytest.raises(error_type) as caught:
        runtime.step()
    assert caught.value is error
    assert len(invoked) == len(requests) == runtime.model_calls == 1
    assert tools.session.progress.as_dict() == progress
    assert tools.session.verification_ledger.history == []
    assert tools.session.spec.revision == 1 and tools.motion.control_epoch == 0
    assert tools.motion.controller.sent == [] and not tools.propose_finish()["accepted"]
    if failure == "send_goal":
        assert tools.motion.phase == "unknown"
        assert tools.motion.active_request_id == "backend-failure"
    else:
        assert tools.motion.active_request_id is None
    if failure == "model":
        assert runtime.history == [] and not runtime.trace_path.exists()
    else:
        outcome, = runtime.history
        assert outcome["event"] == "failed" and "result" not in outcome
        assert outcome["error_type"] == error_type.__name__ and outcome["error"] == str(error)
        events = [json.loads(line) for line in runtime.trace_path.read_text().splitlines()]
        assert [event["event"] for event in events] == ["proposed", "failed"]
        assert events[-1] == outcome


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
    _assert_proposal_rejected(runtime, runtime.step(), "dependencies changed")
    assert tools.motion.controller.sent == []
    assert not tools.session.progress.completed
    assert tools.session.spec.revision == 1 and tools.motion.control_epoch == 0


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


@pytest.mark.parametrize("mode", ["direct", "step"])
def test_verification_freezes_stop_context_after_active_evidence_capture(monkeypatch, tmp_path, mode):
    from copy import deepcopy

    tools = make_tools(monkeypatch)
    source = tools.session.snapshot
    captured = replace(source, timestamp=source.timestamp + 1,
        robot_pose=replace(source.robot_pose, orientation_xyzw=(0., 0., 1., 0.)),
        metadata={"observation_images": [{"id": "turned-frame", "image_ref": "",
            "observed_object_uids": ["cup-1"], "object_boxes": {"cup-1": [0, 0, 2, 2]}}]})
    provide, verify = tools.evidence_provider, tools.verifier.verify
    calls = []

    def capture(uid):
        calls.append("capture")
        # Backend active perception may turn and publish a fresh frame before returning evidence.
        tools.session.snapshot = captured
        return {**provide(uid), "observation_images": deepcopy(captured.metadata["observation_images"])}

    def verify_captured(**kwargs):
        calls.append("verify")
        assert tools.session.snapshot is captured
        assert kwargs["evidence"]["observation_images"] == captured.metadata["observation_images"]
        return verify(**kwargs)

    monkeypatch.setattr(tools, "evidence_provider", capture)
    monkeypatch.setattr(tools.verifier, "verify", verify_captured)
    if mode == "direct":
        result = tools.verify_target("cup-1", "cup", 1)
    else:
        client_model, _ = model_calls([tool_call(name="verify_target", arguments={
            "candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}, reason="capture then verify")])
        runtime = AgentRuntime(tools, tmp_path / "capture.jsonl", client_model=client_model)
        outcome = runtime.step()
        assert outcome["event"] == "returned"
        result = outcome["result"]
    assert VerificationResult(**result).accepted and calls == ["capture", "verify"]
    assert tools.session.progress.target_progress["cup"].accepted_candidate_uids == ["cup-1"]
    assert len(tools.session.verification_ledger.history) == 1
    assert tools.propose_finish()["accepted"]
    assert tools.session.spec.revision == 1 and tools.motion.control_epoch == 0
    assert tools.motion.controller.sent == [] and tools.motion.active_request_id is None
    tools.session.snapshot = source
    assert not tools.propose_finish()["accepted"]  # Receipt belongs to the captured, not pre-turn, context.


@pytest.mark.parametrize("mode", ["direct", "step"])
@pytest.mark.parametrize("change", ["position", "orientation", "timestamp", "observation_replaced",
                                     "observation_in_place", "progress", "pending_progress", "task"])
def test_verification_context_change_during_inference_never_writes_proof(
        monkeypatch, tmp_path, mode, change):
    from copy import deepcopy

    tools = make_tools(monkeypatch)
    session = tools.session
    images = [{"id": "frame-1", "image_ref": "", "observed_object_uids": ["cup-1"],
               "object_boxes": {"cup-1": [0, 0, 2, 2]}}]
    session.snapshot = replace(session.snapshot, metadata={"observation_images": images})
    if change == "pending_progress":
        from task.reducer import apply_verification
        from verification.contracts import CandidateInstance
        apply_verification(session.progress, session.spec.targets[0], CandidateInstance("cup-1", "cup"),
            VerificationResult(False, "need_better_view", semantic_satisfied=True,
                               task_revision=1, target_uid="cup-1"),
            {"verified_relation_context": {"edge": {"anchor_uid": "table-old"}}})
        assert not session.progress.completed and session.progress.mode == "better_view_for_verified_pair"
    snapshot = session.snapshot
    ledger = session.verification_ledger
    before = deepcopy(ledger.as_dict())
    verify = tools.verifier.verify
    concurrent = []

    def delayed_verify(**kwargs):
        result = verify(**kwargs)
        if change == "position":
            session.snapshot = replace(snapshot, robot_pose=replace(snapshot.robot_pose, position=(1, 0, 0)))
        elif change == "orientation":
            session.snapshot = replace(snapshot, robot_pose=replace(
                snapshot.robot_pose, orientation_xyzw=(0., 0., 1., 0.)))
        elif change == "timestamp":
            session.snapshot = replace(snapshot, timestamp=snapshot.timestamp + 1)
        elif change == "observation_replaced":
            session.snapshot = replace(snapshot, metadata={"observation_images": [
                {**images[0], "id": "new-frame"}]})
        elif change == "observation_in_place":
            images[0]["object_boxes"]["cup-1"][0] = 1
            assert session.snapshot is snapshot and session.snapshot.metadata["observation_images"] is images
        elif change == "progress":
            session.progress.target_progress["cup"].rejected_candidate_uids.append("other-cup")
        elif change == "pending_progress":
            session.progress.pending_verified_pair["edge"]["anchor_uid"] = "table-new"
        else:
            session.install_plan(runtime_plan(replace(session.spec, revision=2, exhaustive=True)))
        concurrent.append((session.spec, deepcopy(session.progress.as_dict())))
        return result

    monkeypatch.setattr(tools.verifier, "verify", delayed_verify)
    message = "task changed" if change == "task" else "stop context changed"
    if mode == "direct":
        with pytest.raises(ValueError, match=message):
            tools.verify_target("cup-1", "cup", 1)
    else:
        client_model, _ = model_calls([tool_call(name="verify_target", arguments={
            "candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}, reason="verify current evidence")])
        runtime = AgentRuntime(tools, tmp_path / "verification-race.jsonl", client_model=client_model)
        with pytest.raises(ValueError, match=message):
            runtime.step()
        outcome, = runtime.history
        assert outcome["event"] == "failed" and outcome["error_type"] == "ValueError"
        assert "result" not in outcome and message in outcome["error"]
    task, progress = concurrent[0]
    assert len(concurrent) == 1 and session.spec is task
    assert session.spec.revision == (2 if change == "task" else 1)
    assert session.progress.as_dict() == progress and not session.progress.completed
    assert session.progress.target_progress["cup"].accepted_candidate_uids == []
    assert ledger.history == [] and ledger.as_dict() == before
    assert session.verification_ledger.history == []
    assert tools._finish_verification is None and not tools.propose_finish()["accepted"]
    assert tools.motion.controller.sent == [] and tools.motion.active_request_id is None
    assert tools.motion.control_epoch == 0


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


def test_revision_schema_is_single_object_with_scoped_action_arguments(monkeypatch, tmp_path):
    from pydantic import ValidationError

    tools = make_tools(monkeypatch)
    client_model, _ = model_calls([])
    runtime = AgentRuntime(tools, tmp_path / "schema.jsonl", client_model=client_model)
    response_type = runtime._response_type("ToolCall", runtime._messages(runtime.system_prompt), base=ToolCall)
    for schema_type in (ToolCall, response_type):
        schema = schema_type.schema()
        assert schema["type"] == "object" and set(schema["properties"]) == {"action"}
        assert schema["required"] == ["action"] and schema["additionalProperties"] is False
        assert "anyOf" not in schema and "oneOf" not in schema
        definitions = schema.get("$defs", schema.get("definitions"))
        for definition in definitions.values():
            if "properties" in definition:
                assert definition["additionalProperties"] is False
        variants = [definitions[ref["$ref"].split("/")[-1]]
                    for ref in schema["properties"]["action"]["anyOf"]]
        assert len(variants) == 10
        for variant in variants:
            assert set(variant["properties"]) == {"name", "arguments", "reason"}
            assert variant["additionalProperties"] is False
        for payload in (search_payload("step", ["mug"], concept_id="cup"), interpretation_payload("step")):
            assert schema_type(**payload).action.name.startswith("revise_")
            payload["action"]["arguments"]["extra"] = True
            with pytest.raises(ValidationError):
                schema_type(**payload)
        with pytest.raises(ValidationError):
            schema_type(action={"name": "revise_search", "arguments": {"search_updates": []}, "reason": "empty"})
        with pytest.raises(ValidationError):
            schema_type(action={"name": "revise_interpretation", "arguments": {
                "interpretation_update": None}, "reason": "empty"})
        for location in ("interpretation", "target", "demand", "constraint"):
            payload = interpretation_payload("step")
            parsed = payload["action"]["arguments"]["interpretation_update"]["interpretation"]
            if location == "interpretation":
                container = parsed
            elif location == "target":
                container = parsed["targets"][0]
            elif location == "demand":
                container = parsed["targets"][0]["demand"]
            else:
                parsed["constraints"] = [{"type": "attribute", "subject": "drying surface"}]
                container = parsed["constraints"][0]
            container["hidden_action"] = "submit_motion"
            with pytest.raises(ValidationError):
                schema_type(**payload)
    for invalid in ("concept", "basis", "update_extra", "dual", "too_many"):
        payload = search_payload("step", ["mug"], concept_id="cup")
        arguments = payload["action"]["arguments"]
        update = arguments["search_updates"][0]
        if invalid == "concept":
            update["concept_id"] = "not-in-context"
        elif invalid == "basis":
            update["basis_ids"] = ["frame:not-supplied"]
        elif invalid == "update_extra":
            update["aliases"] = ["mug"]
        elif invalid == "dual":
            arguments.update(interpretation_payload("step")["action"]["arguments"])
        else:
            arguments["search_updates"].append(dict(update))
        with pytest.raises(ValidationError):
            response_type(**payload)
    for invalid in ("basis", "extra"):
        payload = interpretation_payload("step")
        update = payload["action"]["arguments"]["interpretation_update"]
        if invalid == "basis":
            update["basis_ids"] = ["instruction", "frame:not-supplied"]
        else:
            update["raw_instruction"] = "Change the original request"
        with pytest.raises(ValidationError):
            response_type(**payload)
    assert "Accepted revisions persist; do not repeat them every turn" in runtime.system_prompt
    assert "pair a revision only with observe" not in runtime.system_prompt
    assert "revise_search" not in tools.names and "revise_interpretation" not in tools.names


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
    tools.deferred_candidates = set()
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
        False, "need_better_view", task_revision=2, target_uid="cup-1",
        predicate_id="beside", anchor_uid="table-1", failed_constraints=["beside"],
        reason="Anchor is occluded", evidence_ids=["frame:checked-1"],
        dependencies={"object:cup-1": "observed-version"},
    ), step=14)
    # Neither another instruction nor another revision is evidence for this task.
    ledger.put("find something else", "cup-1", VerificationResult(True, "accept", task_revision=2))
    ledger.put(task.raw_instruction, "cup-old", VerificationResult(True, "accept", task_revision=1))
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
    assert observation["verification_records"] == [record.as_dict()]
    assert observation["verification_records"][0]["verdict"] == "unknown"
    assert "event" in context and context["recent_tools"] == []
    assert tools.session.progress.as_dict() == before
    assert not tools.propose_finish()["accepted"]
    assert tools.motion.controller.sent == []


def search_payload(mode, terms, *, concept_id="resource", basis_ids=None):
    arguments = {"search_updates": [{
        "concept_id": concept_id, "detector_terms": terms,
        "basis_ids": basis_ids or ["instruction"],
        "reason": "Inspect another possible realization without changing the requested function.",
    }]}
    if mode == "step":
        return {"action": {"name": "revise_search", "arguments": arguments, "reason": "Revise retrieval"}}
    return {**decision_payload(mode), **arguments}


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

    def forbidden(*args, **kwargs):
        pytest.fail("Changed and noop revisions must not dispatch hidden actions")

    monkeypatch.setattr(tools, "dispatch", forbidden)
    monkeypatch.setattr(tools, "evidence_provider", forbidden)
    monkeypatch.setattr(tools.verifier, "verify", forbidden)
    monkeypatch.setattr(tools.motion.controller, "poll_status", forbidden)
    for index, expected in enumerate(("another-couch", "another-couch", "sofa-1")):
        outcome = decision_call(runtime, mode)
        if mode == "step":
            assert outcome["name"] == "revise_search"
            assert outcome["result"]["accepted"]
            assert outcome["result"]["search_changed"] == (index != 1)
            assert bool(outcome["result"]["search_updates"]) == (index != 1)
            assert outcome["result"]["task_revision"] == 2
            assert runtime._choice_context is None
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


@pytest.mark.parametrize("mode", ["frontier", "step"])
@pytest.mark.parametrize("invalid_second", [False, True])
def test_search_revision_updates_shared_anchor_atomically(monkeypatch, tmp_path, invalid_second, mode):
    from instruction.contracts import ConstraintSpec
    from perception.detector_adapter import ground_plan
    tools = make_tools(monkeypatch)
    task = TaskSpec("cup near table", (TargetSpec("cup", "cup"),
        TargetSpec("table", "table", role="anchor", terminal=False)),
        constraints=(ConstraintSpec("near", "relation", "cup", "near", "table"),), revision=2)
    plan = ground_plan(runtime_plan(task), ["cup", "table"])
    tools.session.install_plan(plan)
    original = plan.as_dict()
    payload = search_payload(mode, ["mug"], concept_id="cup")
    arguments = payload["action"]["arguments"] if mode == "step" else payload
    arguments["search_updates"].extend(search_payload(
        "frontier", [" " if invalid_second else "desk"], concept_id="table")["search_updates"])
    client_model, _ = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "anchors.jsonl", client_model=client_model)
    if invalid_second:
        if mode == "step":
            _assert_proposal_rejected(runtime, runtime.step(), "retrieval phrases")
        else:
            with pytest.raises(ValueError, match="retrieval phrases"):
                decision_call(runtime, mode)
        assert plan.as_dict() == original
        assert tools.session.search_revision == 0
    else:
        decision_call(runtime, mode)
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
    from pydantic import ValidationError

    tools = make_tools(monkeypatch)
    payload = tool_call(name="submit_motion", arguments={
        "candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
        "control_epoch": 0, "request_id": "mixed-update"}, reason="move").dict()
    payload["search_updates"] = search_payload("frontier", ["mug"], concept_id="cup")["search_updates"]
    client_model, _ = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "execution.jsonl", client_model=client_model)
    with pytest.raises(ValidationError):
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
    if mode == "step":
        _assert_proposal_rejected(runtime, decision_call(runtime, mode), "changed|stale")
    else:
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
        assert label["type"] == "text" and frame["id"] in label["text"]
        assert all(uid in label["text"] for uid in frame["observed_object_uids"])
        assert json.dumps(frame["object_boxes"]) in label["text"]
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


def test_guided_choices_and_tool_rejections_share_bounded_history(monkeypatch, tmp_path):
    tools = make_tools(monkeypatch)
    client_model, requests = model_calls([
        tool_call(name="submit_motion", arguments={"frontier_uid": "missing", "task_revision": 2,
                  "control_epoch": 0, "request_id": "stale"}, reason="stale attempt"),
        decision_payload("frontier"), decision_payload("recovery"), decision_payload("frontier"),
        decision_payload("recovery"), decision_payload("step"),
    ])
    runtime = AgentRuntime(tools, tmp_path / "history.jsonl", max_model_calls=6, client_model=client_model)
    rejection = runtime.step()
    _assert_proposal_rejected(runtime, rejection, "revision")
    for mode in ["frontier", "recovery", "frontier", "recovery", "step"]:
        decision_call(runtime, mode)
    assert request_context(requests[1])["recent_tools"][0] == rejection
    assert "revision" in request_context(requests[1])["recent_tools"][0]["result"]["reason"]
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
    if mode == "step":
        _assert_proposal_rejected(runtime, decision_call(runtime, mode), "changed|stale")
    else:
        with pytest.raises(error, match="changed|stale"):
            decision_call(runtime, mode)
    assert runtime._choice_context is None and len(requests) == 1
    assert len(tools.motion.controller.sent) == (2 if change == "motion_owner" else 0)


@pytest.mark.parametrize("mode", ["frontier", "recovery"])
@pytest.mark.parametrize("change", ["none", "task", "search", "snapshot", "progress", "control_epoch", "held"])
def test_guided_submission_rechecks_and_consumes_choice(monkeypatch, tmp_path, mode, change):
    tools = make_tools(monkeypatch)
    client_model, _ = model_calls([decision_payload(mode)])
    runtime = AgentRuntime(tools, tmp_path / "guard.jsonl", client_model=client_model)
    decision_call(runtime, mode)
    session = tools.session
    if change == "task":
        session.install_plan(runtime_plan(replace(session.spec, revision=2, exhaustive=True)))
    elif change == "search":
        session.search_revision += 1
    elif change == "snapshot":
        session.snapshot = replace(session.snapshot)
    elif change == "progress":
        session.progress.completed = True
    elif change in {"control_epoch", "held"}:
        tools.motion.hold()
        if change == "control_epoch":
            tools.motion.resume()

    def submit():
        with runtime.choice_guard():
            tools.submit_motion(session.spec.revision, tools.motion.control_epoch, "guided",
                                candidate_uid="cup-1", viewpoint_uid="view-1")

    if change == "none":
        submit()
        assert len(tools.motion.controller.sent) == 1
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
    arguments = {"interpretation_update": {
        "interpretation": ParsedInstruction(**parsed).dict(),
        "basis_ids": ["instruction"] if basis_ids is None else basis_ids,
        "reason": "The cup is manipulated material; the requested destination is a drying resource.",
    }}
    if mode == "step":
        return {"action": {"name": "revise_interpretation", "arguments": arguments,
                           "reason": "Correct interpretation"}}
    return {**decision_payload(mode), **arguments}


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_interpretation_corrects_material_goal_and_archives_revision(
        interpretation_tools, monkeypatch, tmp_path, mode):
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
        assert outcome["name"] == "revise_interpretation"
        assert outcome["result"] == {"accepted": True, "interpretation_changed": True,
            "task_revision_before": 1, "task_revision": 2,
            "search_revision_before": 1, "search_revision": 0, "search_changed": True}
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
    update = payload["action"]["arguments"] if mode == "step" else payload
    assert archived["reason"] == update["interpretation_update"]["reason"]
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

    assert runtime.step()["result"] == {"accepted": True, "interpretation_changed": True,
        "task_revision_before": 1, "task_revision": 2,
        "search_revision_before": 0, "search_revision": 0, "search_changed": False}

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


@pytest.mark.parametrize("invalid", ["search_terms", "duplicate_concept", "missing_instruction", "empty_interpretation"])
def test_invalid_revision_can_be_corrected_without_hidden_execution(
        interpretation_tools, monkeypatch, tmp_path, invalid):
    from unittest.mock import Mock

    tools = interpretation_tools
    session = tools.session
    original, plan, ledger = session.spec, session.plan, session.verification_ledger
    if invalid == "duplicate_concept":
        session.install_plan(runtime_plan(replace(original, targets=original.targets + (
            TargetSpec("anchor", "table", role="anchor", terminal=False),), revision=2)))
        original, plan, ledger = session.spec, session.plan, session.verification_ledger
    concept_id = original.targets[0].id
    search = invalid in {"search_terms", "duplicate_concept"}
    if search:
        corrected = search_payload("step", ["mug"], concept_id=concept_id)
        payload = search_payload("step", [" " if invalid == "search_terms" else "mug"], concept_id=concept_id)
        if invalid == "duplicate_concept":
            updates = payload["action"]["arguments"]["search_updates"]
            updates.append(dict(updates[0]))
    else:
        ledger.put(original.raw_instruction, "cup-1", VerificationResult(
            False, "need_better_view", task_revision=original.revision, reason="Unresolved resource"))
        corrected = interpretation_payload("step")
        payload = (interpretation_payload("step", basis_ids=["verification:0"]) if invalid == "missing_instruction"
                   else interpretation_payload("step", parsed={"targets": []}))
    progress, records = session.progress.as_dict(), ledger.as_dict()
    client_model, requests = model_calls([payload, corrected])
    runtime = AgentRuntime(tools, tmp_path / "corrected-revision.jsonl", client_model=client_model)
    dispatch = Mock(side_effect=AssertionError("Revision cannot dispatch"))
    monkeypatch.setattr(tools, "dispatch", dispatch)
    rejected = runtime.step()
    if invalid == "empty_interpretation":
        assert not rejected["result"]["accepted"] and not rejected["result"]["interpretation_changed"]
        assert rejected["result"]["interpretation_status"] == "invalid"
    else:
        match = {"search_terms": "retrieval phrases", "duplicate_concept": "duplicate concept",
                 "missing_instruction": "cite the original instruction"}[invalid]
        _assert_proposal_rejected(runtime, rejected, match)
    assert session.spec is original and session.plan is plan
    assert session.progress.as_dict() == progress and session.verification_ledger is ledger
    assert ledger.as_dict() == records
    assert session.search_revision == 0 and session.interpretation_history == []
    assert tools.motion.controller.sent == [] and tools.motion.active_request_id is None
    accepted = runtime.step()
    assert accepted["result"]["accepted"]
    assert accepted["result"]["search_changed"] is search
    assert accepted["result"]["interpretation_changed"] is (not search)
    assert session.spec.raw_instruction == original.raw_instruction
    assert request_context(requests[1])["recent_tools"] == [rejected]
    assert runtime.model_calls == len(requests) == 2 and runtime._choice_context is None
    assert tools.motion.controller.sent == [] and tools.motion.active_request_id is None
    assert not tools.propose_finish()["accepted"]
    dispatch.assert_not_called()


@pytest.mark.parametrize("failure", ["compiler", "grounding", "installation"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, ConnectionError])
def test_revision_service_errors_are_not_proposal_rejections(
        interpretation_tools, monkeypatch, tmp_path, failure, error_type):
    from unittest.mock import Mock

    tools = interpretation_tools
    error = error_type(f"{failure} failed")
    fail = Mock(side_effect=error)
    if failure == "compiler":
        monkeypatch.setattr("instruction.compiler.compile_parsed_instruction", fail)
    elif failure == "grounding":
        monkeypatch.setattr("perception.detector_adapter.ground_plan", fail)
    else:
        monkeypatch.setattr(tools, "install_interpretation", fail)
    client_model, _ = model_calls([interpretation_payload("step")])
    runtime = AgentRuntime(tools, tmp_path / "revision-failure.jsonl", client_model=client_model)
    original = tools.session.spec
    with pytest.raises(error_type) as caught:
        runtime.step()
    assert caught.value is error and fail.call_count == 1
    assert tools.session.spec is original and tools.session.interpretation_history == []
    assert tools.session.search_revision == 0 and tools.motion.controller.sent == []
    outcome, = runtime.history
    assert outcome["name"] == "revise_interpretation" and outcome["event"] == "failed"
    assert outcome["error_type"] == error_type.__name__ and "result" not in outcome


@pytest.mark.parametrize("mode", ["frontier", "recovery", "step"])
def test_noop_interpretation_keeps_task_revision(interpretation_tools, monkeypatch, tmp_path, mode):
    tools = interpretation_tools
    original, plan, progress = tools.session.spec, tools.session.plan, tools.session.progress
    payload = interpretation_payload(mode, parsed={"targets": [{"name": "cup"}]})
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "noop.jsonl", max_model_calls=1, client_model=client_model)

    def forbidden(*args, **kwargs):
        pytest.fail("A noop revision must not dispatch or implicitly execute anything")

    monkeypatch.setattr(tools, "dispatch", forbidden)
    monkeypatch.setattr(tools, "evidence_provider", forbidden)
    monkeypatch.setattr(tools.verifier, "verify", forbidden)
    monkeypatch.setattr(tools.motion.controller, "poll_status", forbidden)
    result = decision_call(runtime, mode)
    if mode == "step":
        assert result["name"] == "revise_interpretation"
        assert result["result"] == {"accepted": True, "interpretation_changed": False,
            "task_revision_before": 1, "task_revision": 1,
            "search_revision_before": 0, "search_revision": 0, "search_changed": False}
        assert runtime._choice_context is None
    else:
        assert result == ("node:2" if mode == "frontier" else "approach:u")
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec is original and tools.session.plan is plan and tools.session.progress is progress
    assert tools.session.search_revision == 0 and tools.session.interpretation_history == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("mode", ["frontier", "step"])
@pytest.mark.parametrize("phase", ["active", "held", "cancel_pending"])
@pytest.mark.parametrize("during_decision", [False, True])
def test_interpretation_requires_idle_unheld_motion(interpretation_tools, tmp_path, phase, during_decision, mode):
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
        return interpretation_payload(mode)

    if not during_decision:
        change_motion()
    client_model, requests = model_calls([decision])
    runtime = AgentRuntime(tools, tmp_path / "owned.jsonl", client_model=client_model)
    # 响应期间改变运动状态由上下文守卫拒绝；已有运动由 idle_guard 拒绝。
    message = "motion state or control epoch changed" if during_decision else "idle, unheld"
    if mode == "step":
        _assert_proposal_rejected(runtime, runtime.step(), message)
    else:
        with pytest.raises(RuntimeError, match=message):
            decision_call(runtime, mode)
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
    arguments = payload["action"]["arguments"] if mode == "step" else payload
    arguments["search_updates"] = search_payload(
        "frontier", ["mug"], concept_id=tools.session.spec.targets[0].id)["search_updates"]
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "mixed-revisions.jsonl", client_model=client_model)
    if mode == "step":
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            decision_call(runtime, mode)
    else:
        with pytest.raises(ValueError, match="mutually exclusive"):
            decision_call(runtime, mode)
    assert len(requests) == runtime.model_calls == 1
    assert tools.session.spec.revision == 1 and tools.session.search_revision == 0
    assert tools.session.search_hypotheses == {} and tools.session.interpretation_history == []
    assert tools.motion.controller.sent == []


@pytest.mark.parametrize("name,arguments", [
    ("observe", {}), ("query_scene", {}),
    ("query_viewpoints", {"candidate_uid": "cup-1"}),
    ("motion_status", {}), ("cancel_motion", {}), ("propose_finish", {}),
    ("verify_target", {"candidate_uid": "cup-1", "target_id": "cup", "task_revision": 1}),
    ("submit_motion", {"candidate_uid": "cup-1", "viewpoint_uid": "view-1", "task_revision": 1,
                       "control_epoch": 0, "request_id": "must-not-move"}),
])
@pytest.mark.parametrize("location", ["root", "action", "arguments"])
@pytest.mark.parametrize("revision", ["search", "interpretation"])
def test_revision_cannot_accompany_other_tools(
        interpretation_tools, tmp_path, name, arguments, location, revision):
    from pydantic import ValidationError

    tools = interpretation_tools
    update = (interpretation_payload("step") if revision == "interpretation" else
              search_payload("step", ["mug"], concept_id=tools.session.spec.targets[0].id))
    payload = tool_call(name=name, arguments=arguments, reason="mixed turn").dict()
    container = (payload if location == "root" else payload["action"] if location == "action"
                 else payload["action"]["arguments"])
    container.update(update["action"]["arguments"])
    with pytest.raises(ValidationError):
        ToolCall(**payload)
    client_model, requests = model_calls([payload])
    runtime = AgentRuntime(tools, tmp_path / "mixed-action.jsonl", client_model=client_model)
    with pytest.raises(ValidationError):
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
        env=SimpleNamespace(episode_over=False),
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


def test_capture_budget_boundary_returns_unknown_without_model_or_more_motion(habitat_evidence_tools):
    fixture = habitat_evidence_tools
    tools, agent = fixture.tools, fixture.tools.agent
    fixture.rotate.side_effect = lambda current: setattr(current.env, "episode_over", True) or False
    result = tools.verify_target("cup-1", "cup", 1)
    assert result["decision"] == "need_better_view"
    assert not tools.propose_finish()["accepted"]
    agent._project_object_bbox_on_current_view.assert_not_called()
    agent.final_instruction_verifier.verify.assert_not_called()
    fixture.tilt.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
def test_verification_publishes_restored_frame_only_when_incomplete(habitat_evidence_tools, accepted):
    from backends.habitat.scene import current_observation
    fixture = habitat_evidence_tools
    tools, agent = fixture.tools, fixture.tools.agent
    verifier = agent.final_instruction_verifier.verify
    verifier.return_value = VerificationResult(
        accepted, "accept" if accepted else "need_better_view", semantic_satisfied=True,
        view_sufficient_for_stop=accepted, task_revision=1, target_uid="cup-1",
        diagnostics={"original_instruction_review": {"coverage_complete": True,
            "interpretation_consistent": True, "task_satisfied": accepted}})

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
