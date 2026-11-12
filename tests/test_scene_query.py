"""Queries preserve observations and share identity across backend exports."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from backends.habitat.scene import snapshot_from_mapper
from harness.tools import query_scene
from instruction_adapter.contracts import ConceptQuery, Constraint, InstructionPlan, SearchPriors, TargetQuery
from instruction_adapter.verifier import VerificationResult, candidate_from_object
from planning.semantic_snapshot_context import select_target_candidate_from_snapshot
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot
from task.session import TaskSession


def test_query_preserves_native_snapshot_and_distinct_uids(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    objects = (ObjectNodeSnapshot("cup-a", "cup", (1., 0., 0.), 0.8),
               ObjectNodeSnapshot("cup-b", "cup", (1., 0., 0.), 0.9))
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects)
    before = deepcopy(snapshot)
    plan = InstructionPlan("cup", targets=[TargetQuery("cup", "cup")], valid=True)
    selection = select_target_candidate_from_snapshot(snapshot, plan=plan)
    assert snapshot == before
    assert selection.result.obj is objects[1]
    assert candidate_from_object(objects[0]).uid != candidate_from_object(objects[1]).uid
    assert isinstance(selection.result.obj.confidence, float)
    assert not hasattr(selection.result.obj, "tag")


@pytest.mark.parametrize("metadata,eligible", [({}, True), ({"label_probabilities": {}}, False),
    ({"label_probabilities": {"cup": 1.}}, True)])
@pytest.mark.parametrize("pinned", [False, True])
def test_empty_category_posterior_cannot_become_an_exact_candidate(monkeypatch, metadata, eligible, pinned):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    obj = ObjectNodeSnapshot("observed", "cup", (1., 0., 0.), 0., metadata=metadata)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(obj,))
    plan = InstructionPlan("cup", targets=[TargetQuery("cup", "cup")], valid=True)
    session = TaskSession(snapshot, plan)
    if pinned:
        session.progress.mode = "better_view_for_verified_pair"
        session.progress.pending_verified_pair = {"candidate_uid": obj.uid, "target_id": "cup"}
    result = query_scene(session)
    assert bool(result.candidates) is eligible
    assert session.snapshot.object_by_uid(obj.uid) is obj
    assert not session.progress.completed


def test_habitat_export_and_native_input_select_same_object(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    mapper = SimpleNamespace(current_position=[0, 0, 0], objects=[
        SimpleNamespace(uid="cup-a", tag="cup", position=[1, 0, 0], confidence=0.9,
                        bbox=[1, 2, 3, 4]),
        SimpleNamespace(uid="cup-b", tag="cup", position=[2, 0, 0], confidence=0.5,
                        bbox=[1, 2, 3, 4]),
    ])
    plan = InstructionPlan("cup", targets=[TargetQuery("cup", "cup")], valid=True)
    exported, lookup = snapshot_from_mapper(mapper, step=3)
    native = SemanticMapSnapshot(3, Pose3D((0, 0, 0)), objects=exported.objects)
    a = select_target_candidate_from_snapshot(exported, plan=plan)
    b = select_target_candidate_from_snapshot(native, plan=plan)
    assert a.selected_snapshot_uid == b.selected_snapshot_uid == "cup-a"
    assert lookup[a.selected_snapshot_uid] is mapper.objects[0]
    assert mapper.objects[0].tag == "cup"


def test_candidate_without_ros_bbox_serializes_as_unknown_not_nan():
    import json
    from dataclasses import asdict

    obj = ObjectNodeSnapshot("sysnav_object:2", "chair", (1., 2., 0.), 0.8)
    candidate = candidate_from_object(obj)
    assert candidate.bbox_2d == []
    assert json.loads(json.dumps(asdict(candidate), allow_nan=False))["uid"] == obj.uid


def test_candidate_rejects_invalid_box_and_preserves_valid_xyxy():
    for box in (None, [], [float("nan")], [0, 0, float("inf"), 2],
                [0, 0, 0, 2], [3, 0, 1, 2], [1, 2, 3]):
        obj = SimpleNamespace(uid="chair", label="chair", bbox=box)
        assert candidate_from_object(obj).bbox_2d == []
    obj = SimpleNamespace(uid="chair", label="chair", bbox=[1, 2, 3, 4])
    assert candidate_from_object(obj).bbox_2d == [1., 2., 3., 4.]


@pytest.mark.parametrize("candidate_end", ["subject", "object"])
def test_relation_lookup_keeps_direction_and_anchor_attributes(monkeypatch, candidate_end):
    import json
    from instruction_adapter.verifier import FinalInstructionVerifier
    from instruction_adapter.constraints import ConstraintEvaluator
    from planning.mode_policy import is_anchor_first_relation_search
    from scene.query import anchor_concepts
    from unittest.mock import Mock
    from instruction_adapter.semantic_edges import SemanticEdge

    monkeypatch.setenv("LLM_OFFLINE", "1")
    subject, reference = (("goal", "anchor") if candidate_end == "subject" else ("anchor", "goal"))
    plan = InstructionPlan("Find the seat with a red carton beneath it", targets=[
        TargetQuery("goal", "seat"),
        TargetQuery("anchor", "carton", terminal=False, role="anchor", attributes={"color": "red"})],
        constraints=[Constraint("spatial", subject=subject, object=reference, relation="under",
                               value={"source_text": "beneath it", "requirement": "beneath the bound instance"},
                               hardness="hard", verifier="vlm"),
                     Constraint("state", subject="anchor", value="open", hardness="hard", verifier="vlm")])
    objects = (ObjectNodeSnapshot("seat-1", "seat", (0., 0., 0.)),
               ObjectNodeSnapshot("carton-1", "carton", (.5, 0., 0.)))
    service = SimpleNamespace(verify=Mock(side_effect=lambda **kw: SemanticEdge(
        subject_id=kw["subject"]["id"], relation=kw["relation"], object_id=kw["object_"]["id"], verified=True)))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), plan,
                          constraint_evaluator=ConstraintEvaluator(relation_service=service))
    target = session.plan.targets[0]
    assert is_anchor_first_relation_search(plan)
    assert [c.id for c in anchor_concepts(plan, target)] == ["anchor"]
    projection = FinalInstructionVerifier._visual_task(plan.as_dict(), {"target_id": "goal"})
    assert projection["domain_predicates"][0]["subject"] == subject
    assert projection["referenced_targets"] == [{"id": "anchor", "name": "carton", "attributes": {"color": "red"}}]
    result = session.constraint_evaluator.evaluate_before_final_verifier(mapper=session, plan=plan,
        target=target, candidate=candidate_from_object(objects[0]), candidate_obj=objects[0], evidence={})
    assert result.satisfied
    request = service.verify.call_args.kwargs
    expected = ("seat-1", "carton-1") if candidate_end == "subject" else ("carton-1", "seat-1")
    assert (request["subject"]["id"], request["object_"]["id"]) == expected
    assert request["relation"] == "under"
    assert request["predicate"] == {**session.spec.as_dict()["constraints"][0],
                                    "subject_concept": None, "object_concept": None}
    assert request["predicate"]["value"] == {
        "source_text": "beneath it", "requirement": "beneath the bound instance"}
    anchor = request["object_"] if candidate_end == "subject" else request["subject"]
    assert anchor["attributes"] == {"color": "red"}
    assert [(c["id"], c["subject"], c["value"]) for c in anchor["constraints"]] == [("c1", "anchor", "open")]
    assert json.loads(json.dumps(plan.as_dict()))["concept_queries"][1]["attributes"] == {"color": "red"}


def _relation_plan():
    return InstructionPlan("mug on work desk", targets=[
        TargetQuery("goal", "mug", detector_terms=["container"]),
        TargetQuery("desk", "work desk", role="anchor", terminal=False, detector_terms=["table"])], constraints=[
        Constraint("spatial", subject="mug", relation="on", object="work desk",
                   hardness="hard", object_concept=ConceptQuery(
                       "desk", "work desk", role="anchor", detector_terms=["table"]))], valid=True)


@pytest.mark.parametrize("label", ["mug", "container"])
@pytest.mark.parametrize("anchor_status", ["missing", "available", "searched_no_terminal_found"])
def test_relation_query_retains_target_and_prioritizes_unsearched_anchor_without_completion(
        monkeypatch, label, anchor_status):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    target = ObjectNodeSnapshot("target", label, (1., 0., 0.), 0.9)
    anchor = ObjectNodeSnapshot("anchor", "work desk", (2., 0., 0.), 0.1)
    objects = (target,) if anchor_status == "missing" else (target, anchor)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects)
    plan = _relation_plan()
    session = TaskSession(snapshot, plan)
    if anchor_status == "searched_no_terminal_found":
        session.anchor_search_ledger.mark(raw_instruction=plan.raw_instruction, concept_id="desk",
                                          anchor_uid="anchor", status=anchor_status)
    before = deepcopy((snapshot, session.progress.as_dict(), session.anchor_search_ledger.as_dict()))
    result = query_scene(session)
    expected = [(anchor, "anchor"), (target, "target")] if anchor_status == "available" else [(target, "target")]
    assert [(c.obj, c.role) for c in result.candidates] == expected
    selection = select_target_candidate_from_snapshot(snapshot, plan=plan, context=session)
    assert selection.result.obj is expected[0][0]
    assert selection.result.selection_role == expected[0][1]
    assert selection.result.skipped_objs == result.skipped_objs == []
    intent = selection.to_navigation_intent()
    assert intent.mode.value == ("go_to_anchor" if anchor_status == "available" else "go_to_object")
    assert not intent.stop_allowed
    assert not session.progress.completed
    assert not session.verification_ledger.records
    assert (snapshot, session.progress.as_dict(), session.anchor_search_ledger.as_dict()) == before
    if anchor_status != "missing":
        row = session.concept_matcher.match_many(raw_instruction=plan.raw_instruction,
                                                concept=plan.constraints[0].object_concept,
                                                objects=[anchor])[0]
        assert row.matches_concept is True and not row.terminal_eligible


@pytest.mark.parametrize("pin_target", [False, True])
def test_deferred_terminal_allows_anchor_search_without_semantic_rejection(monkeypatch, pin_target):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    target = ObjectNodeSnapshot("target", "mug", (1., 0., 0.), 0.9)
    anchor = ObjectNodeSnapshot("anchor", "table", (2., 0., 0.), 0.8)
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(target, anchor)), _relation_plan())
    if pin_target:
        session.progress.mode = "better_view_for_verified_pair"
        session.progress.pending_verified_pair = {"candidate_uid": target.uid, "target_id": "goal"}
    before = deepcopy(session.snapshot), deepcopy(session.progress.as_dict())
    first = query_scene(session)
    expected = [(target, "target")] if pin_target else [(anchor, "anchor"), (target, "target")]
    assert [(c.obj, c.role) for c in first.candidates] == expected
    if pin_target:
        assert {row["concept_id"] for row in first.concept_debug} == {"goal"}
    result = query_scene(session, deferred_uids={target.uid})
    assert [(c.obj, c.role) for c in result.candidates] == [(anchor, "anchor")]
    assert result.selected.obj is anchor
    assert result.skipped_objs == [{"uid": target.uid, "target_id": "goal",
                                   "skip_reason": "waiting_for_relevant_evidence"}]
    assert not query_scene(session, deferred_uids={target.uid, anchor.uid}).candidates
    assert not session.anchor_search_ledger.records
    session.anchor_search_ledger.mark(raw_instruction=session.plan.raw_instruction,
        concept_id="desk", anchor_uid=anchor.uid, status="searched_no_terminal_found")
    ledger_before = deepcopy(session.anchor_search_ledger.as_dict())
    assert not query_scene(session, deferred_uids={target.uid}).candidates
    assert query_scene(session).selected.obj is target
    assert session.anchor_search_ledger.as_dict() == ledger_before
    assert (session.snapshot, session.progress.as_dict()) == before
    assert not session.progress.completed
    assert not session.verification_ledger.records


@pytest.mark.parametrize("include_anchor", [False, True])
def test_terminal_priority_is_unchanged_without_relation_search(monkeypatch, include_anchor):
    from planning.mode_policy import is_anchor_first_relation_search

    monkeypatch.setenv("LLM_OFFLINE", "1")
    target = ObjectNodeSnapshot("target", "mug", (1., 0., 0.), 0.1)
    anchor = ObjectNodeSnapshot("anchor", "work desk", (2., 0., 0.), 0.9)
    plan = InstructionPlan("Find a mug", targets=[TargetQuery("goal", "mug")], valid=True)
    if include_anchor:
        plan.targets.append(TargetQuery("desk", "work desk", role="anchor", terminal=False))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(target, anchor)), plan)
    before = deepcopy((session.snapshot, session.spec.as_dict(), session.plan.as_dict(), session.progress.as_dict()))
    assert not is_anchor_first_relation_search(session.plan)
    result = query_scene(session)
    expected = [(target, "target"), (anchor, "anchor")] if include_anchor else [(target, "target")]
    assert [(c.obj, c.role) for c in result.candidates] == expected
    assert result.selected.obj is target
    assert not session.progress.completed
    assert not session.verification_ledger.records
    assert not session.anchor_search_ledger.records
    assert (session.snapshot, session.spec.as_dict(), session.plan.as_dict(), session.progress.as_dict()) == before


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("active_index", [0, 1])
def test_ordered_query_limits_anchors_and_terminals_to_active_target(monkeypatch, ordered, active_index):
    from instruction_adapter.contracts import ExecutionPolicy
    from planning.mode_policy import is_anchor_first_relation_search

    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = InstructionPlan("Find two mugs on the desk, then a lamp near the shelf", targets=[
        TargetQuery("mugs", "mug", min_count=2), TargetQuery("light", "lamp"),
        TargetQuery("desk", "desk", role="anchor", terminal=False),
        TargetQuery("shelf", "shelf", role="anchor", terminal=False)], constraints=[
        Constraint("spatial", subject="mugs", relation="on", object="desk", hardness="hard"),
        Constraint("spatial", subject="shelf", relation="near", object="light", hardness="hard")],
        execution=ExecutionPolicy(ordered=ordered), valid=True)
    objects = (ObjectNodeSnapshot("mug-1", "mug", (1., 0., 0.), 0.9),
               ObjectNodeSnapshot("lamp-1", "lamp", (2., 0., 0.), 0.8),
               ObjectNodeSnapshot("desk-1", "desk", (3., 0., 0.), 0.7),
               ObjectNodeSnapshot("shelf-1", "shelf", (4., 0., 0.), 0.6))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), plan)
    session.progress.active_target_index = active_index
    before = deepcopy((session.snapshot, session.spec.as_dict(), session.plan.as_dict(), session.progress.as_dict()))
    assert is_anchor_first_relation_search(session.plan)
    result = query_scene(session)
    expected = [("desk-1", "desk", "anchor"), ("shelf-1", "shelf", "anchor"),
                ("mug-1", "mugs", "target"), ("lamp-1", "light", "target")]
    if ordered:
        expected = [expected[active_index], expected[active_index + 2]]
    assert [(c.obj.uid, c.target_id, c.role) for c in result.candidates] == expected
    assert all(c.obj is session.snapshot.object_by_uid(c.obj.uid) for c in result.candidates)
    assert result.skipped_objs == []
    assert not session.progress.completed
    assert not session.verification_ledger.records
    assert not session.anchor_search_ledger.records
    assert (session.snapshot, session.spec.as_dict(), session.plan.as_dict(), session.progress.as_dict()) == before


def test_broad_target_is_unknown_candidate_without_relabeling_unrelated_objects(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    broad = ObjectNodeSnapshot("broad", "container", (1., 0., 0.), 0.01)
    unrelated = ObjectNodeSnapshot("unrelated", "wall", (2., 0., 0.), 0.99)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(unrelated, broad))
    session = TaskSession(snapshot, _relation_plan())
    before = deepcopy(snapshot)
    result = query_scene(session)
    assert [c.obj.uid for c in result.candidates] == ["broad"]
    assert result.selected.record["verdict"] == "unknown"
    assert result.selected.record["candidate_eligible"] is True
    assert result.selected.record["terminal_eligible"] is False
    assert snapshot == before and result.selected.obj.label == "container"


@pytest.mark.parametrize("status,evidence,blocked", [
    ("searched_no_terminal_found", {}, True),
    ("rejected_as_wrong_anchor", {}, True),
    ("search_incomplete", {"panorama_complete": False}, False),
    ("terminal_candidate_available", {}, False),
    ("terminal_candidate_available", {"panorama_complete": False}, False),
    ("terminal_candidate_available", {"panorama_complete": True}, True),
])
def test_broad_anchor_can_be_inspected_and_search_memory_only_blocks_repeat_search(
        monkeypatch, status, evidence, blocked):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    anchor = ObjectNodeSnapshot("anchor", "table", (2., 0., 0.), 0.8)
    unrelated = ObjectNodeSnapshot("unrelated", "wall", (3., 0., 0.), 0.9)
    plan = _relation_plan()
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(anchor, unrelated))
    session = TaskSession(snapshot, plan)
    first = query_scene(session)
    assert [c.obj.uid for c in first.candidates] == ["anchor"]
    assert first.selected.role == "anchor"
    assert first.selected.record["verdict"] == "unknown"
    assert first.selected.record["terminal_eligible"] is False
    session.anchor_search_ledger.mark(raw_instruction=plan.raw_instruction, concept_id="desk",
                                      anchor_uid="anchor", status=status, evidence=evidence)
    before = deepcopy((session.progress.as_dict(), session.anchor_search_ledger.as_dict()))
    assert session.anchor_search_ledger.is_blocked(plan.raw_instruction, "desk", anchor.uid) is blocked
    assert [c.obj.uid for c in query_scene(session).candidates] == ([] if blocked else [anchor.uid])
    new_target = ObjectNodeSnapshot("target", "container", (1., 0., 0.), 0.8)
    session.snapshot = SemanticMapSnapshot(2, snapshot.robot_pose, objects=(anchor, unrelated, new_target))
    expected = [(new_target, "target")] if blocked else [(anchor, "anchor"), (new_target, "target")]
    assert [(c.obj, c.role) for c in query_scene(session).candidates] == expected
    another_anchor = ObjectNodeSnapshot("another-anchor", "table", (4., 0., 0.), 0.8)
    session.snapshot = SemanticMapSnapshot(3, snapshot.robot_pose, objects=(anchor, another_anchor, unrelated))
    observed_before = deepcopy(session.snapshot)
    assert [c.obj.uid for c in query_scene(session).candidates] == (
        [another_anchor.uid] if blocked else [anchor.uid, another_anchor.uid])
    assert session.snapshot == observed_before
    assert session.snapshot.object_by_uid("anchor") is anchor
    assert (session.progress.as_dict(), session.anchor_search_ledger.as_dict()) == before
    assert not session.progress.completed
    assert not session.verification_ledger.records


def test_confirmed_identity_can_approach_without_terminal_eligibility(monkeypatch):
    from instruction_adapter.concept_matcher import ConceptMatchRecord
    monkeypatch.setenv("LLM_OFFLINE", "1")
    obj = ObjectNodeSnapshot("target", "mug", (1., 0., 0.), 0.8)
    row = ConceptMatchRecord("instruction", "goal", obj.uid, obj.label,
                             matches_concept=True, candidate_eligible=True,
                             terminal_eligible=False)
    matcher = SimpleNamespace(match_many=lambda **kwargs: [row] if kwargs["concept"].id == "goal" else [])
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(obj,)),
                          _relation_plan(), concept_matcher=matcher)
    result = query_scene(session)
    assert [(c.obj, c.role) for c in result.candidates] == [(obj, "target")]
    assert result.selected.obj is obj
    assert result.selected.record["terminal_eligible"] is False
    assert not session.progress.completed


def test_unknown_candidates_still_respect_counted_and_rejected_instances(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    objects = tuple(ObjectNodeSnapshot(uid, "container", (1., 0., 0.), 0.8)
                    for uid in ("counted", "rejected", "new"))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), _relation_plan())
    session.progress.target_progress["goal"].accepted_candidate_uids.append("counted")
    session.verification_ledger.put(session.plan.raw_instruction, "rejected", VerificationResult(
        satisfied=False, decision="reject_candidate", task_revision=session.spec.revision, target_uid="rejected"))
    result = query_scene(session)
    assert [c.obj.uid for c in result.candidates] == ["new"]
    assert {row["uid"]: row["skip_reason"] for row in result.skipped_objs} == {
        "counted": "already_counted_instance", "rejected": "rejected_instance"}


def _demand_session(*objects, revision=1):
    """Build a real compiled demand without using the adapter capability gate."""
    from instruction.compiler import task_from_parsed_plan
    from instruction_adapter.task_compat import runtime_plan

    parsed = InstructionPlan(
        "I need somewhere soft and unoccupied to lie down.",
        targets=[TargetQuery(
            "rest", "resting surface", attributes={"softness": "soft", "state": "unoccupied"},
            demand={"activity": "lie down", "manipulated_objects": ["body"],
                    "required_resource": "soft body-supporting surface"})],
        constraints=[Constraint("state", subject="rest", value="unoccupied", hardness="hard")],
        search_priors=SearchPriors(destination_hypotheses={"rest": ["bed", "sofa"]}),
        valid=True,
    )
    spec = task_from_parsed_plan(parsed, revision=revision)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects)
    return TaskSession(snapshot, runtime_plan(spec))


def _structured_concept_provider(monkeypatch):
    """Script provider judgments, keeping real batching, schemas and matching."""
    import json
    from instruction_adapter import concept_matcher

    requests = []

    def parse(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
        requests.append(payload)
        rows = {}
        for obj in payload["observed_objects"]:
            eligible = obj["label"] in {"bed", "sofa", "daybed"}
            rows[obj["uid"]] = {
                "matches_concept": None if eligible else False,
                "candidate_eligible": eligible,
                "terminal_eligible": False,
                "confidence": 0.8,
                "reason": "Inspect this facility's current state." if eligible else "Not a supporting surface.",
            }
        parsed = kwargs["response_format"].model_validate(rows)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(concept_matcher, "get_client_and_model", lambda _: (client, "test-only"))
    # Only the intercepted provider runs while the real matcher's offline guard
    # is disabled. No matcher, query, ledger or progress method is mocked.
    monkeypatch.setenv("LLM_OFFLINE", "0")
    return requests


def _reject_demand_instance(session, obj):
    """Feed an instance rejection into the real evidence store and reducer."""
    from task.reducer import apply_verification

    target = session.spec.targets[0]
    result = VerificationResult(
        satisfied=False, decision="reject_candidate", semantic_satisfied=False,
        task_revision=session.spec.revision, target_uid=obj.uid,
        failed_constraints=[f"demand:{target.id}"], reason="This instance is occupied.",
    )
    record = session.verification_ledger.put(session.spec.raw_instruction, obj.uid, result)
    assert record.status == "rejected_hard"
    assert not apply_verification(
        session.progress, target, candidate_from_object(obj), result, {})


@pytest.mark.parametrize("facility", ["bed", "sofa"])
def test_demand_retrieval_only_creates_observed_unverified_hypotheses(monkeypatch, facility):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    obj = ObjectNodeSnapshot("observed", facility, (1., 0., 0.), 0.8)
    wall = ObjectNodeSnapshot("wall", "wall", (2., 0., 0.), 0.99)
    session = _demand_session(obj, wall)
    before = session.spec.as_dict(), session.plan.as_dict(), deepcopy(session.snapshot)
    progress = deepcopy(session.progress.as_dict())
    assert session.destination_hypotheses == []

    result = query_scene(session)
    assert [c.obj.uid for c in result.candidates] == [obj.uid]
    assert result.selected.obj is obj
    assert result.selected.record["verdict"] == "unknown"
    assert result.selected.record["candidate_eligible"] is True
    assert result.selected.record["terminal_eligible"] is False
    assert session.destination_hypotheses == [{
        "target_id": "rest", "candidate_uid": obj.uid, "facility_concept": facility,
        "status": "requires_instance_verification", "task_revision": session.spec.revision,
        "observation_timestamp": session.snapshot.timestamp,
        "reason": result.selected.record["reason"],
    }]
    assert session.progress.as_dict() == progress
    assert not session.progress.completed
    assert not session.verification_ledger.records
    assert (session.spec.as_dict(), session.plan.as_dict(), session.snapshot) == before


@pytest.mark.parametrize("replacement_label", ["sofa", "daybed"])
def test_rejected_demand_instance_can_be_replaced_without_rewriting_spec(monkeypatch, replacement_label):
    requests = _structured_concept_provider(monkeypatch)
    rejected = ObjectNodeSnapshot("occupied-bed", "bed", (1., 0., 0.), 0.99)
    wall = ObjectNodeSnapshot("wall", "wall", (2., 0., 0.), 1.0)
    session = _demand_session(rejected, wall)
    original_spec = session.spec
    before = original_spec.as_dict(), session.plan.as_dict()
    snapshot_before = deepcopy(session.snapshot)
    progress_before = deepcopy(session.progress.as_dict())
    first = query_scene(session)
    assert first.selected.obj is rejected
    assert session.progress.as_dict() == progress_before
    assert [h["candidate_uid"] for h in session.destination_hypotheses] == [rejected.uid]

    _reject_demand_instance(session, rejected)
    assert session.verification_ledger.is_hard_rejected(session.spec.raw_instruction, rejected.uid)
    assert not query_scene(session).candidates
    assert session.destination_hypotheses == []
    assert session.snapshot == snapshot_before

    replacement = ObjectNodeSnapshot("replacement", replacement_label, (3., 0., 0.), 0.9)
    same_category = ObjectNodeSnapshot("another-bed", "bed", (4., 0., 0.), 0.7)
    session.snapshot = SemanticMapSnapshot(
        2, session.snapshot.robot_pose, objects=(rejected, wall, replacement, same_category))
    observed_before = deepcopy(session.snapshot)
    rejected_progress = deepcopy(session.progress.as_dict())
    result = query_scene(session)

    assert result.selected.obj is replacement
    assert [c.obj.uid for c in result.candidates] == [replacement.uid, same_category.uid]
    assert result.skipped_objs == [{
        "uid": rejected.uid, "target_id": "rest", "skip_reason": "rejected_instance"}]
    assert [h["candidate_uid"] for h in session.destination_hypotheses] == [
        replacement.uid, same_category.uid]
    assert [h["facility_concept"] for h in session.destination_hypotheses] == [replacement_label, "bed"]
    assert all(h["status"] == "requires_instance_verification"
               and h["task_revision"] == session.spec.revision
               and h["observation_timestamp"] == 2 for h in session.destination_hypotheses)
    assert all(c.record["terminal_eligible"] is False for c in result.candidates)
    assert session.spec is original_spec
    assert (session.spec.as_dict(), session.plan.as_dict()) == before
    assert session.snapshot == observed_before
    assert session.progress.as_dict() == rejected_progress
    assert session.progress.target_progress["rest"].rejected_candidate_uids == [rejected.uid]
    assert session.progress.target_progress["rest"].accepted_candidate_uids == []
    assert not session.progress.completed
    assert requests
    assert all(p["instruction"] == original_spec.raw_instruction for p in requests)
    assert all(p["concept"]["demand"] == before[0]["targets"][0]["demand"] for p in requests)
    if replacement_label == "daybed":
        assert replacement_label not in session.plan.targets[0].detector_terms
        assert replacement_label not in session.plan.targets[0].match_terms
        assert any({"uid": replacement.uid, "label": replacement_label} in p["observed_objects"]
                   for p in requests)


def test_new_revision_clears_destination_hypotheses_and_instance_rejections(monkeypatch):
    from dataclasses import replace
    from instruction_adapter.task_compat import runtime_plan

    monkeypatch.setenv("LLM_OFFLINE", "1")
    bed = ObjectNodeSnapshot("bed-1", "bed", (1., 0., 0.), 0.9)
    sofa = ObjectNodeSnapshot("sofa-1", "sofa", (2., 0., 0.), 0.8)
    session = _demand_session(bed, sofa, revision=4)
    query_scene(session)
    _reject_demand_instance(session, bed)
    assert query_scene(session).selected.obj is sofa
    previous_hypotheses = deepcopy(session.destination_hypotheses)
    assert previous_hypotheses and previous_hypotheses[0]["task_revision"] == 4
    old_spec, old_progress, old_ledger = session.spec, session.progress, session.verification_ledger
    before = old_spec.as_dict()

    # Refreshing the same projection is not a new revision.
    session.install_plan(runtime_plan(old_spec))
    assert session.destination_hypotheses == previous_hypotheses
    assert session.progress is old_progress and session.verification_ledger is old_ledger

    # Reusing the exact instruction and target IDs tests revision isolation,
    # rather than accidentally relying on a different instruction-hash key.
    revised = replace(old_spec, revision=5)
    session.install_plan(runtime_plan(revised))
    assert session.spec == revised
    assert session.destination_hypotheses == []
    assert session.progress is not old_progress and session.verification_ledger is not old_ledger
    assert not session.verification_ledger.records
    assert not session.progress.target_progress["rest"].rejected_candidate_uids
    assert not session.progress.completed
    assert query_scene(session).selected.obj is bed
    assert {h["task_revision"] for h in session.destination_hypotheses} == {5}
    assert {h["candidate_uid"] for h in session.destination_hypotheses} == {bed.uid, sofa.uid}
    assert old_spec.as_dict() == before
    assert old_ledger.is_hard_rejected(old_spec.raw_instruction, bed.uid)


@pytest.mark.parametrize("category", ["chair", "bed"])
def test_explicit_objectnav_remains_exact_identity_not_a_destination_hypothesis(monkeypatch, category):
    from backends.habitat.task_input import adapt_objectnav_task
    from instruction_adapter.task_compat import runtime_plan

    monkeypatch.setenv("LLM_OFFLINE", "1")
    obj = ObjectNodeSnapshot("explicit-target", category, (1., 0., 0.), 0.8)
    unrelated = ObjectNodeSnapshot("unrelated", "sofa", (2., 0., 0.), 0.99)
    spec = adapt_objectnav_task(category)
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(obj, unrelated)),
                          runtime_plan(spec))
    before = session.spec.as_dict(), session.plan.as_dict(), deepcopy(session.progress.as_dict())
    result = query_scene(session)
    assert [c.obj.uid for c in result.candidates] == [obj.uid]
    assert result.selected.record["source"] == "exact"
    assert result.selected.record["matches_concept"] is True
    assert result.selected.record["terminal_eligible"] is True
    assert not session.spec.targets[0].demand
    assert session.destination_hypotheses == []
    assert not session.progress.completed
    assert (session.spec.as_dict(), session.plan.as_dict(), session.progress.as_dict()) == before


def _evaluate_relation_precheck(session, evidence=None):
    obj = session.objects[0]
    return session.constraint_evaluator.evaluate_before_final_verifier(
        mapper=session, plan=session.plan, target=session.plan.targets[0],
        candidate=candidate_from_object(obj), candidate_obj=obj, evidence=evidence or {})


@pytest.mark.parametrize("candidate_end", ["subject", "object"])
@pytest.mark.parametrize("status", ["missing_anchor", "rejected_relation", "insufficient_evidence", "verified"])
def test_relation_precheck_outcomes_preserve_direction_and_verdict(monkeypatch, tmp_path, candidate_end, status):
    from unittest.mock import Mock
    from PIL import Image
    from instruction_adapter.constraints import ConstraintEvaluator
    from instruction_adapter.relation_verifier import DynamicRelationService
    from instruction_adapter.verifier import instruction_hash
    from task.reducer import apply_verification

    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = _relation_plan()
    if candidate_end == "object":
        plan.constraints[0].subject, plan.constraints[0].object = "work desk", "mug"
    target = ObjectNodeSnapshot("mug-1", "mug", (0., 0., 0.))
    anchor = ObjectNodeSnapshot("desk-1", "work desk", (0., 0., 0.))
    objects = (target,) if status == "missing_anchor" else (target, anchor)
    service = DynamicRelationService()
    service.vlm_callback = Mock(return_value={
        "verified": status == "verified", "confidence": 0.83, "reason": "Scripted pair judgment."})
    service.verify = Mock(wraps=service.verify)
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), plan,
                          constraint_evaluator=ConstraintEvaluator(relation_service=service))
    session.constraint_evaluator.register_mapper_objects(mapper=session)
    evidence = {}
    view_ids = []
    if status in ("rejected_relation", "verified"):
        frame = tmp_path / "shared.png"
        Image.new("RGB", (40, 40), "white").save(frame)
        view_ids = ["shared-view"]
        evidence = {"observation_images": [{
            "id": view_ids[0], "rgb_path": str(frame), "observed_object_uids": [target.uid, anchor.uid],
            "object_boxes": {target.uid: [0, 0, 20, 20], anchor.uid: [20, 20, 40, 40]},
        }]}
    evaluation = _evaluate_relation_precheck(session, evidence)
    subject, reference = ("goal", "desk") if candidate_end == "subject" else ("desk", "goal")
    anchor_uid = "" if status == "missing_anchor" else anchor.uid
    assert evaluation.relation_outcomes == [{
        "predicate_id": "c0", "subject_id": subject, "object_id": reference,
        "candidate_uid": target.uid, "anchor_concept_id": "desk", "anchor_uid": anchor_uid,
        "relation": "on", "status": status, "evidence_view_ids": view_ids,
        "reason": ("Anchor object has not been observed yet." if status == "missing_anchor" else
                   "No shared instance-bound observation is available." if status == "insufficient_evidence" else
                   "Scripted pair judgment."),
    }]
    decision = ("pass" if status == "verified" else
                "reject_relation" if status == "rejected_relation" else "need_relation_check")
    result = evaluation.to_verification_result(session.spec.revision, target.uid)
    assert result.decision == evaluation.decision == decision
    assert result.confidence == evaluation.confidence
    assert result.satisfied == evaluation.satisfied == (status == "verified")
    assert result.satisfied_constraints == evaluation.satisfied_constraints
    assert result.failed_constraints == evaluation.failed_constraints
    assert result.reason == evaluation.reason
    assert result.diagnostics == {"constraint_eval": evaluation.as_dict()}
    assert result.view_objective == {"relation_outcomes": evaluation.relation_outcomes}
    assert (result.task_revision, result.target_uid) == (session.spec.revision, target.uid)
    from scene.dependencies import object_dependencies
    assert result.dependencies == object_dependencies(session.snapshot, target.uid, anchor_uid)
    assert result.predicate_id == ("final_stop" if status == "verified" else "c0")
    assert result.anchor_uid == ("" if status == "verified" else anchor_uid)
    assert not result.accepted
    assert result.verdict == ("false" if status == "rejected_relation" else "unknown")
    assert service.vlm_callback.call_count == (1 if view_ids else 0)
    if status == "missing_anchor":
        service.verify.assert_not_called()
    else:
        service.verify.assert_called_once()
        request = service.verify.call_args.kwargs
        pair = (target.uid, anchor.uid) if candidate_end == "subject" else (anchor.uid, target.uid)
        assert (request["subject"]["id"], request["object_"]["id"]) == pair
        record = session.relation_pair_ledger.get(instruction_hash(plan.raw_instruction), pair[0], "on", pair[1])
        assert record.status == {"verified": "accepted_relation", "rejected_relation": "rejected_relation",
                                 "insufficient_evidence": "unknown"}[status]
    if status == "rejected_relation":
        record = session.verification_ledger.put(plan.raw_instruction, target.uid, result)
        assert record.verdict == "false" and record.status == "rejected_soft"
        assert (record.predicate_id, record.anchor_uid) == ("c0", anchor.uid)
        assert not session.verification_ledger.is_hard_rejected(plan.raw_instruction, target.uid)
        assert not apply_verification(session.progress, session.spec.targets[0], candidate_from_object(target), result, evidence)
        assert not session.progress.target_progress["goal"].rejected_candidate_uids
        assert [c.obj for c in query_scene(session).candidates if c.role == "target"] == [target]


def test_relation_precheck_keeps_all_predicates_and_conflicting_false(monkeypatch):
    from unittest.mock import Mock
    from instruction_adapter.constraints import ConstraintEvaluator
    from instruction_adapter.semantic_edges import SemanticEdge

    monkeypatch.setenv("LLM_OFFLINE", "1")
    statuses = ["rejected_relation", "missing_anchor", "insufficient_evidence", "verified"]
    labels = ["desk", "shelf", "tray", "lamp"]
    target = ObjectNodeSnapshot("mug-1", "mug", (0., 0., 0.))
    anchors = [ObjectNodeSnapshot(f"anchor-{i}", label, (0., 0., 0.)) for i, label in enumerate(labels)]
    plan = InstructionPlan("mug near the desk, shelf, tray and lamp", targets=[TargetQuery("goal", "mug"), *[
        TargetQuery(f"ref-{i}", label, role="anchor", terminal=False) for i, label in enumerate(labels)]],
        constraints=[Constraint("spatial", subject="goal", object=f"ref-{i}", relation="near", hardness="hard")
                     for i in range(len(labels))], valid=True)
    service = SimpleNamespace(verify=Mock(side_effect=lambda **kw: SemanticEdge(
        subject_id=kw["subject"]["id"], relation=kw["relation"], object_id=kw["object_"]["id"],
        verified=kw["object_"]["id"] == "anchor-3", need_better_view=kw["object_"]["id"] == "anchor-2",
        confidence=0.8, evidence_view_ids=[f"view-{kw['object_']['id']}"], reason="Scripted pair judgment.")))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)),
                          objects=(target, anchors[0], anchors[2], anchors[3])), plan,
                          constraint_evaluator=ConstraintEvaluator(relation_service=service))
    evaluation = _evaluate_relation_precheck(session)
    outcomes = evaluation.relation_outcomes
    assert [row["status"] for row in outcomes] == statuses
    assert [row["predicate_id"] for row in outcomes] == [f"c{i}" for i in range(4)]
    assert [row["object_id"] for row in outcomes] == [f"ref-{i}" for i in range(4)]
    assert [row["anchor_concept_id"] for row in outcomes] == [f"ref-{i}" for i in range(4)]
    assert [row["anchor_uid"] for row in outcomes] == ["anchor-0", "", "anchor-2", "anchor-3"]
    assert all(row["subject_id"] == "goal" and row["candidate_uid"] == target.uid for row in outcomes)
    assert [row["evidence_view_ids"] for row in outcomes] == [["view-anchor-0"], [], ["view-anchor-2"], ["view-anchor-3"]]
    assert len(evaluation.failed_constraints) == 3 and len(evaluation.satisfied_constraints) == 1
    result = evaluation.to_verification_result(session.spec.revision, target.uid)
    assert result.decision == evaluation.decision == "reject_relation"
    assert result.verdict == "false" and result.confidence == 0.8
    assert (result.predicate_id, result.anchor_uid) == ("final_stop", "")
    assert result.view_objective == {"relation_outcomes": outcomes}
    assert result.diagnostics["constraint_eval"]["relation_outcomes"] == outcomes
    assert service.verify.call_count == 3


@pytest.mark.parametrize("second_status", ["verified", "insufficient_evidence", "rejected_relation"])
def test_relation_precheck_keeps_alternative_pairs_without_candidate_rejection(monkeypatch, second_status):
    from unittest.mock import Mock
    from instruction_adapter.constraints import ConstraintEvaluator
    from instruction_adapter.semantic_edges import SemanticEdge
    from task.reducer import apply_verification

    monkeypatch.setenv("LLM_OFFLINE", "1")
    objects = (ObjectNodeSnapshot("mug-1", "mug", (0., 0., 0.)), *[
        ObjectNodeSnapshot(uid, "work desk", (0., 0., 0.)) for uid in ("desk-1", "desk-2")])
    service = SimpleNamespace(verify=Mock(side_effect=lambda **kw: SemanticEdge(
        subject_id=kw["subject"]["id"], relation=kw["relation"], object_id=kw["object_"]["id"],
        verified=kw["object_"]["id"] == "desk-2" and second_status == "verified",
        need_better_view=kw["object_"]["id"] == "desk-2" and second_status == "insufficient_evidence")))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), _relation_plan(),
                          constraint_evaluator=ConstraintEvaluator(relation_service=service))
    evaluation = _evaluate_relation_precheck(session)
    assert [row["status"] for row in evaluation.relation_outcomes] == ["rejected_relation", second_status]
    assert [row["anchor_uid"] for row in evaluation.relation_outcomes] == ["desk-1", "desk-2"]
    assert [row["predicate_id"] for row in evaluation.relation_outcomes] == ["c0", "c0"]
    result = evaluation.to_verification_result(session.spec.revision, objects[0].uid)
    assert result.decision == {"verified": "pass", "insufficient_evidence": "need_relation_check",
                               "rejected_relation": "reject_relation"}[second_status]
    assert (result.predicate_id, result.anchor_uid) == ("final_stop", "")
    assert result.view_objective["relation_outcomes"] == evaluation.relation_outcomes
    session.verification_ledger.put(session.plan.raw_instruction, objects[0].uid, result)
    assert not session.verification_ledger.is_hard_rejected(session.plan.raw_instruction, objects[0].uid)
    assert not apply_verification(session.progress, session.spec.targets[0], candidate_from_object(objects[0]), result, {})
    assert not session.progress.target_progress["goal"].rejected_candidate_uids
    assert [c.obj for c in query_scene(session).candidates if c.role == "target"] == [objects[0]]
    assert service.verify.call_count == 2


def test_relation_precheck_without_evaluated_pairs_remains_unknown(monkeypatch):
    from unittest.mock import Mock
    from instruction_adapter.constraints import ConstraintEvaluator

    monkeypatch.setenv("LLM_OFFLINE", "1")
    objects = (ObjectNodeSnapshot("mug-1", "mug", (0., 0., 0.)),
               ObjectNodeSnapshot("desk-1", "work desk", (0., 0., 0.)))
    service = SimpleNamespace(verify=Mock(side_effect=AssertionError("Identity-rejected pair must not invoke VLM")))
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects), _relation_plan(),
                          constraint_evaluator=ConstraintEvaluator(relation_service=service))
    session.constraint_evaluator.record_identity(session.plan.raw_instruction, session.plan.targets[0].concept_query(),
        {"instance_id": objects[0].uid, "verdict": "false", "source_crop_sha256": [], "reason": "Not a mug."}, 0.9)
    evaluation = _evaluate_relation_precheck(session)
    assert evaluation.decision == "need_relation_check" and evaluation.relation_edges == []
    assert evaluation.relation_outcomes[0]["status"] == "insufficient_evidence"
    assert evaluation.relation_outcomes[0]["anchor_uid"] == ""
    assert evaluation.to_verification_result(session.spec.revision, objects[0].uid).verdict == "unknown"
    service.verify.assert_not_called()
