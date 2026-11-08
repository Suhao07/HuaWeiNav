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


def _relation_plan():
    return InstructionPlan("mug on work desk", targets=[
        TargetQuery("goal", "mug", detector_terms=["container"])], constraints=[
        Constraint("spatial", subject="mug", relation="on", object="work desk",
                   hardness="hard", object_concept=ConceptQuery(
                       "desk", "work desk", role="anchor", detector_terms=["table"]))], valid=True)


@pytest.mark.parametrize("label", ["mug", "container"])
@pytest.mark.parametrize("anchor_status", ["missing", "available", "searched_no_terminal_found"])
def test_target_can_approach_without_unsearched_anchor_but_does_not_complete(monkeypatch, label, anchor_status):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    target = ObjectNodeSnapshot("target", label, (1., 0., 0.), 0.1)
    anchor = ObjectNodeSnapshot("anchor", "work desk", (2., 0., 0.), 0.9)
    objects = (target,) if anchor_status == "missing" else (target, anchor)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects)
    plan = _relation_plan()
    session = TaskSession(snapshot, plan)
    if anchor_status == "searched_no_terminal_found":
        session.anchor_search_ledger.mark(raw_instruction=plan.raw_instruction, concept_id="desk",
                                          anchor_uid="anchor", status=anchor_status)
    before = deepcopy(snapshot), session.progress.as_dict(), session.anchor_search_ledger.as_dict()
    selection = select_target_candidate_from_snapshot(snapshot, plan=plan, context=session)
    assert selection.result.obj is target
    assert selection.result.selection_role == "target"
    assert selection.result.skipped_objs == []
    intent = selection.to_navigation_intent()
    assert intent.mode.value == "go_to_object" and not intent.stop_allowed
    assert not session.progress.completed
    assert (snapshot, session.progress.as_dict(), session.anchor_search_ledger.as_dict()) == before
    if anchor_status != "missing":
        row = session.concept_matcher.match_many(raw_instruction=plan.raw_instruction,
                                                concept=plan.constraints[0].object_concept,
                                                objects=[anchor])[0]
        assert row.matches_concept is True and not row.terminal_eligible


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


def test_broad_anchor_can_be_inspected_and_search_memory_only_blocks_repeat_search(monkeypatch):
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
                                      anchor_uid="anchor", status="searched_no_terminal_found")
    assert not query_scene(session).candidates
    new_target = ObjectNodeSnapshot("target", "container", (1., 0., 0.), 0.8)
    session.snapshot = SemanticMapSnapshot(2, snapshot.robot_pose, objects=(anchor, unrelated, new_target))
    assert query_scene(session).selected.obj is new_target
    another_anchor = ObjectNodeSnapshot("another-anchor", "table", (4., 0., 0.), 0.8)
    session.snapshot = SemanticMapSnapshot(3, snapshot.robot_pose, objects=(anchor, another_anchor, unrelated))
    assert query_scene(session).selected.obj is another_anchor
    assert session.snapshot.object_by_uid("anchor") is anchor


def test_confirmed_identity_can_approach_without_terminal_eligibility(monkeypatch):
    from instruction_adapter.concept_matcher import ConceptMatchRecord
    monkeypatch.setenv("LLM_OFFLINE", "1")
    obj = ObjectNodeSnapshot("target", "mug", (1., 0., 0.), 0.8)
    row = ConceptMatchRecord("instruction", "goal", obj.uid, obj.label,
                             matches_concept=True, candidate_eligible=True,
                             terminal_eligible=False)
    matcher = SimpleNamespace(match_many=lambda **kwargs: [row])
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=(obj,)),
                          _relation_plan(), concept_matcher=matcher)
    assert query_scene(session).selected.obj is obj
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
