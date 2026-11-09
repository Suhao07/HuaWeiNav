import base64
import json
import types
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image


def _requirement(quote, verdict="true", scope="candidate", support=None):
    return {"quote": quote, "scope": scope, "verdict": verdict,
            "reason": "The supplied observation establishes this requirement.",
            "supporting_observation_ids": ["current"] if support is None else support}


def _complete_review(quote):
    return {"coverage_complete": True, "interpretation_consistent": True,
            "task_satisfied": True, "requirements": [_requirement(quote)]}


def _current_evidence(tmp_path):
    path = tmp_path / "current.png"
    Image.new("RGB", (8, 8), "red").save(path)
    return {"current_rgb_with_bbox_path": str(path),
            "observation_identity": {"candidate_uid": "u", "task_revision": 1, "observation_id": "current"},
            "geometry": {"bbox_xyxy": [0, 0, 7, 7]}}

from instruction_adapter.verifier import (
    CandidateInstance,
    EvidenceStore,
    VerificationResult,
    FinalInstructionVerifier,
    _planner_hard_constraint_infeasible,
    hard_stop_constraints_from_evidence,
)


@pytest.mark.parametrize("reason", ["disabled", "llm_offline", "pydantic_unavailable", "empty_response", "request_error"])
def test_unavailable_verification_never_accepts(reason):
    result = FinalInstructionVerifier()._fallback(reason)
    ledger = EvidenceStore()
    record = ledger.put("find a chair", "chair-1", result)
    assert not result.accepted
    assert result.decision == "uncertain"
    assert result.diagnostics["verdict"] == "unknown"
    assert record.status != "accepted"
    assert not ledger.is_hard_rejected("find a chair", "chair-1")


def test_disabled_verifier_returns_unknown(monkeypatch):
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "off")
    result = FinalInstructionVerifier().verify("find a chair", {}, CandidateInstance("c1", "chair"), {})
    assert not result.accepted
    assert result.diagnostics["unavailable"] == "disabled"


def test_verifier_request_error_preserves_original_exception(monkeypatch, tmp_path):
    import instruction_adapter.verifier as module

    failure = RuntimeError("provider unavailable")

    def fail_client(vlm):
        raise failure

    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "HAS_PYDANTIC", True)
    monkeypatch.setattr(module, "get_client_and_model", fail_client)
    with pytest.raises(RuntimeError) as raised:
        FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
            "find a chair", {}, CandidateInstance("u", "chair"), _current_evidence(tmp_path))
    assert raised.value is failure


@pytest.mark.parametrize("payload", [
    {"satisfied": False, "decision": "accept"},
    {"satisfied": True, "decision": "uncertain"},
    {"satisfied": True, "decision": "accept", "semantic_satisfied": False},
    {"satisfied": True, "decision": "accept", "semantic_satisfied": True, "view_sufficient_for_stop": False},
])
def test_inconsistent_result_cannot_be_accepted(payload):
    result = VerificationResult(**payload)
    assert not result.accepted
    assert EvidenceStore().put("chair", "c1", result).status != "accepted"


def test_view_guidance_reports_distance_as_prompt_fact():
    """Geometry distance should guide the prompt without hard-blocking accept."""

    guidance, preferred = FinalInstructionVerifier._view_guidance(
        {
            "geometry": {
                "distance_to_object": 2.42,
                "required_stop_distance": 1.5,
            },
            "view_quality_facts": {
                "bbox_center_norm": [0.5, 0.5],
                "bbox_area_ratio": 0.05,
                "center_offset_norm": 0.0,
                "border_margin_norm": 0.5,
                "projection_failed": False,
            },
        }
    )

    assert any("current_distance_to_object=2.420m" in item for item in guidance)
    assert any("predicted_instance_stop_distance=1.500m" in item for item in guidance)
    assert "remaining task predicate" in preferred


def test_view_guidance_reports_projection_failure_as_prompt_fact():
    """Projection failures are VLM guidance, not Python-side rejection rules."""

    guidance, preferred = FinalInstructionVerifier._view_guidance(
        {
            "geometry": {"projection_failed_in_final_view": True},
            "view_quality_facts": {
                "bbox_area_ratio": 0.001,
                "center_offset_norm": 0.6,
                "border_margin_norm": 0.0,
                "projection_failed": True,
            },
            "view_control": {
                "budget_exhausted": True,
                "best_visual_evidence": {
                    "image_paths": {"current_rgb_with_bbox_path": "best_target.jpg"},
                },
            },
        }
    )

    assert "current target projection failed" in guidance[0]
    assert all("budget" not in item for item in guidance)
    assert "candidate" in preferred


def test_hard_stop_constraints_report_unsatisfied_distance():
    """Distance is a structured stop contract, not only natural-language advice."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.512,
                "required_stop_distance": 1.5,
            },
            "view_control": {
                "best_visual_evidence": {
                    "image_paths": {"current_rgb_with_bbox_path": "best.jpg"},
                },
            },
        }
    )

    assert constraints["satisfied"] is False
    assert constraints["failed"][0]["name"] == "within_final_stop_distance"
    assert constraints["failed"][0]["margin"] < 0


def test_vlm_report_cannot_override_planner_hard_stop_contract():
    """Only planner/geometry evidence may mark a physical stop contract infeasible."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.278,
                "required_stop_distance": 1.5,
            }
        }
    )

    allowed, reason = _planner_hard_constraint_infeasible({
        **constraints,
        "vlm_report": {
            "infeasible_or_not_applicable": True,
            "reason": "The image suggests closer views might crop the target.",
        },
    })

    assert allowed is False
    assert reason == ""


def test_planner_proof_can_mark_hard_stop_contract_infeasible():
    """Best-available stop requires an explicit geometry-side infeasibility proof."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.278,
                "required_stop_distance": 1.5,
            },
            "planner_infeasibility_proof": {
                "infeasible_by_geometry": True,
                "reason": "all pathfinder-validated closer stop poses collide",
            },
        }
    )

    allowed, reason = _planner_hard_constraint_infeasible(constraints)

    assert allowed is True
    assert "collide" in reason


@pytest.mark.parametrize("semantic,visual,physical,infeasible", [
    (False, True, True, False), (True, False, True, False),
    (True, True, False, False), (True, True, False, True),
    (True, True, True, False),
])
def test_final_accept_requires_independent_predicates(monkeypatch, tmp_path, semantic, visual, physical, infeasible):
    """A model's overall accept cannot override its own missing predicates."""
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification
    parsed = ParsedVerification(satisfied=True, decision="accept", semantic_satisfied=semantic,
                                view_sufficient_for_stop=visual,
                                original_instruction_review=_complete_review("find object"))
    parse = lambda **kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(parsed=parsed))])
    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    result = FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find object", {}, CandidateInstance("u", "object"), {
            **_current_evidence(tmp_path),
            "hard_stop_constraints": {"satisfied": physical, "planner_infeasible": infeasible,
                                      "planner_infeasibility_proof": "blocked" if infeasible else ""}})
    assert result.accepted == (semantic and visual and physical)
    assert result.semantic_satisfied == semantic


def test_final_request_separates_execution_history_from_visual_evidence(monkeypatch, tmp_path):
    import json
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(
            parsed=ParsedVerification(satisfied=False, decision="uncertain")))])
    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    evidence = _current_evidence(tmp_path)
    FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find chair", {}, CandidateInstance("u", "chair", confidence=.99), {**evidence,
        "geometry": {**evidence["geometry"], "distance_to_object": .293},
        "view_control": {"budget_exhausted": True, "attempts": [{"semantic_satisfied": True}]},
        "final_verifier_attempt": {"budget_exhausted": True}, "bbox_verified": True})
    payload = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert payload["candidate"] == {"uid": "u"}
    assert "evidence" not in payload
    assert "geometry" not in payload
    assert "hard_constraints" not in requests[0]["response_format"].model_json_schema()["properties"]


def test_visual_task_scopes_user_semantics_to_active_target():
    from instruction.contracts import TaskSpec, TargetSpec, ConstraintSpec
    from instruction_adapter.task_compat import runtime_plan
    task = TaskSpec("Find a red cup, then the book on the shelf.", (
        TargetSpec("cup", "cup", attributes={"color": "red"}),
        TargetSpec("book", "book"), TargetSpec("shelf", "shelf", terminal=False)), (
        ConstraintSpec("on", "spatial", subject="book", relation="on", object="shelf", verifier="vlm"),), ordered=True)
    plan = runtime_plan(task).as_dict()
    plan["targets"][1]["detector_terms"] = ["predicted vocabulary"]
    scoped = FinalInstructionVerifier._visual_task(plan, {"target_id": "book"})
    assert scoped["active_target"] == {"id": "book", "name": "book", "attributes": {}, "demand": {}}
    assert scoped["referenced_targets"] == [task.as_dict()["targets"][2]]
    assert scoped["visual_predicates"][0]["relation"] == "on"
    assert scoped["targets"] == task.as_dict()["targets"]
    assert scoped["constraints"] == task.as_dict()["constraints"]
    assert scoped["ordered"] is True
    assert "predicted vocabulary" not in str(scoped)
    with pytest.raises(ValueError, match="explicit target_id"):
        FinalInstructionVerifier._visual_task(plan, {})


def test_visual_projection_uses_declared_constraint_owner_and_runtime_delegation():
    from instruction.contracts import TaskSpec, TargetSpec, ConstraintSpec
    from instruction_adapter.task_compat import runtime_plan
    task = TaskSpec("Find the cup in the kitchen within one metre, then find the book on a shelf.", (
        TargetSpec("cup", "cup"), TargetSpec("book", "book"),
        TargetSpec("shelf", "shelf", terminal=False)), (
        ConstraintSpec("distance", "distance", subject="cup", value=1, verifier="planner"),
        ConstraintSpec("room", "room", subject="cup", value="kitchen"),
        ConstraintSpec("color", "attribute", subject="cup", value="red"),
        ConstraintSpec("on", "spatial", subject="book", relation="on", object="shelf", verifier="vlm")),
        ordered=True)
    plan = runtime_plan(task)
    evidence = {"target_id": "cup", "declared_room_constraints": [plan.constraints[1].as_dict()],
                "declared_attribute_constraints": [plan.constraints[2].as_dict()]}
    projected = FinalInstructionVerifier._visual_task(plan.as_dict(), evidence)
    assert [item["type"] for item in projected["visual_predicates"]] == ["room", "attribute"]
    assert projected["domain_predicates"][0]["type"] == "distance"
    assert projected["domain_predicates"][0]["verifier"] == "planner"
    assert all(item["verifier"] == "planner" for item in projected["visual_predicates"])
    next_target = FinalInstructionVerifier._visual_task(plan.as_dict(), {"target_id": "book"})
    assert next_target["visual_predicates"][0]["relation"] == "on"
    assert next_target["visual_predicates"][0]["verifier"] == "vlm"
    assert next_target["domain_predicates"] == []


def test_current_stop_requires_distance_even_with_pinned_reference():
    evidence = {'geometry': {'required_stop_distance': 1.,
                              'projection_failed_in_final_view': True},
                'view_control': {'pinned_visual_evidence': {'image_path': 'old.jpg'}}}
    constraints = hard_stop_constraints_from_evidence(evidence)
    assert not constraints['satisfied']
    assert len(constraints['failed']) == 2


def test_surface_identity_must_match_current_observation():
    evidence = {'observation_identity': {'observation_id': 'new'},
                'geometry': {'surface_match': True, 'surface_status': 'matched',
                             'observation_identity': {'observation_id': 'old'},
                             'distance_to_object': .5, 'required_stop_distance': 1.}}
    assert not hard_stop_constraints_from_evidence(evidence)['satisfied']


@pytest.fixture
def fulltext_case(monkeypatch, tmp_path):
    """Exercise production verification with real local PNGs and a scripted provider."""
    import instruction_adapter.verifier as module

    raw = "Find the red container beside the blue cabinet."
    task = {"raw_instruction": raw, "revision": 1, "targets": [
        {"id": "goal", "name": "container", "attributes": {"color": "red"}, "terminal": True}],
        "constraints": []}
    review = _complete_review("Find the red container")
    review["requirements"].append(_requirement("beside the blue cabinet"))
    response = {"decision": "accept", "satisfied": True, "semantic_satisfied": True,
                "view_sufficient_for_stop": True, "original_instruction_review": review}
    case = types.SimpleNamespace(raw=raw, task=task, evidence=_current_evidence(tmp_path),
                                 response=response, review=review, requests=[], make_response=None)

    def parse(**kwargs):
        case.requests.append(kwargs)
        if case.make_response:
            case.make_response(json.loads(kwargs["messages"][1]["content"][0]["text"]))
        parsed = kwargs["response_format"](**case.response)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))])

    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")  # Only the scripted boundary below is reachable.
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "offline-fulltext-replay"))
    case.verifier = FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs"))
    case.verify = lambda: case.verifier.verify(case.raw, case.task, CandidateInstance("u", "detector guess"), case.evidence)
    return case


def test_full_text_review_accepts_complete_supported_candidate(fulltext_case):
    case = fulltext_case
    before = deepcopy((case.task, case.evidence, case.response))
    result = case.verify()
    assert result.accepted and result.verdict == "true"
    assert result.diagnostics["original_instruction_review"]["task_satisfied"]
    assert result.diagnostics["sent_current_observation_ids"] == ["current"]
    assert len(case.requests) == 1
    content = case.requests[0]["messages"][1]["content"]
    payload = json.loads(content[0]["text"])
    assert payload["raw_instruction"] == case.raw
    assert payload["semantic_authority"] == "raw_instruction"
    assert payload["candidate"] == {"uid": "u"}
    assert "detector guess" not in str(payload)
    sent_bytes = base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1])
    assert sent_bytes == Path(case.evidence["current_rgb_with_bbox_path"]).read_bytes()
    assert sent_bytes == Path(result.evidence_ids[0]).read_bytes()
    assert before == (case.task, case.evidence, case.response)


@pytest.mark.parametrize("verdict,decision", [("false", "reject_candidate"), ("unknown", "need_better_view")])
def test_omitted_relation_fulltext_clause_overrules_self_reported_accept(fulltext_case, verdict, decision):
    case = fulltext_case
    assert case.task["constraints"] == []  # Compilation lost the relation and anchor color.
    case.review["requirements"][1].update(verdict=verdict, reason="The required relation is contradicted or unresolved.")
    result = case.verify()
    assert not result.accepted and not result.semantic_satisfied
    assert result.decision == decision
    assert result.verdict == ("false" if verdict == "false" else "unknown")
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]


@pytest.mark.parametrize("broken", ["missing", "coverage", "interpretation", "no_rows", "task_only",
                                    "fake_quote", "different_case", "whitespace_quote", "empty_reason",
                                    "invented_id", "mixed_ids", "no_support", "unknown_fake_id"])
@pytest.mark.parametrize("reported", ["accept", "reject_candidate"])
def test_invalid_fulltext_review_is_unknown_not_wrong_instance(fulltext_case, broken, reported):
    case = fulltext_case
    case.response["decision"] = reported
    row = case.review["requirements"][0]
    if broken == "missing":
        case.response.pop("original_instruction_review")
    elif broken in ("coverage", "interpretation"):
        case.review["coverage_complete" if broken == "coverage" else "interpretation_consistent"] = False
    elif broken == "no_rows":
        case.review["requirements"] = []
    elif broken == "task_only":
        for item in case.review["requirements"]:
            item["scope"] = "task"
    elif broken in ("fake_quote", "different_case", "whitespace_quote"):
        row["quote"] = {"fake_quote": "unmentioned property", "different_case": "RED container",
                        "whitespace_quote": " "}[broken]
    elif broken == "empty_reason":
        row["reason"] = "   "
    else:
        row["supporting_observation_ids"] = {
            "invented_id": ["invented"], "mixed_ids": ["current", "invented"],
            "no_support": [], "unknown_fake_id": ["invented"]}[broken]
        if broken == "unknown_fake_id":
            row["verdict"] = "unknown"
    result = case.verify()
    assert not result.accepted and result.verdict == "unknown"
    assert result.decision == "need_better_view"
    review = result.diagnostics["original_instruction_review"]
    assert review["validation_errors"] and not review["task_satisfied"]


@pytest.mark.parametrize("missing", ["image", "file", "identity", "revision", "candidate", "bbox"])
def test_fulltext_requires_current_candidate_bound_image(fulltext_case, missing):
    case = fulltext_case
    if missing == "image":
        case.evidence.pop("current_rgb_with_bbox_path")
    elif missing == "file":
        case.evidence["current_rgb_with_bbox_path"] += ".missing.png"
    elif missing == "identity":
        case.evidence.pop("observation_identity")
    elif missing == "bbox":
        case.evidence["geometry"]["bbox_xyxy"] = [0, 0, 0, 0]
    else:
        case.evidence["observation_identity"]["task_revision" if missing == "revision" else "candidate_uid"] = 0
    result = case.verify()
    assert not result.accepted and result.verdict == "unknown"
    assert result.diagnostics["sent_current_observation_ids"] == []


@pytest.mark.parametrize("binding", ["valid", "wrong_revision", "wrong_candidate"])
def test_earlier_view_cannot_replace_current_candidate_support(fulltext_case, tmp_path, binding):
    case = fulltext_case
    path = tmp_path / "earlier.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    identity = {**case.evidence["observation_identity"], "observation_id": "earlier"}
    if binding == "wrong_revision":
        identity["task_revision"] = 0
    if binding == "wrong_candidate":
        identity["candidate_uid"] = "other"
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "image_paths": {"current_rgb_with_bbox_path": str(path)},
        "observation_identity": identity, "geometry": {"bbox_xyxy": [1, 1, 6, 6]}}}
    case.review["requirements"][0]["supporting_observation_ids"] = ["earlier"]
    result = case.verify()
    assert not result.accepted and result.verdict == "unknown"
    assert ("earlier" in result.diagnostics["sent_candidate_observation_ids"]) == (binding == "valid")


def test_raw_instruction_overrides_conflicting_structure_and_preserves_all_semantics(fulltext_case):
    case = fulltext_case
    case.task["raw_instruction"] = "a conflicting compiled paraphrase"
    case.task["targets"].extend([
        {"id": "anchor", "name": "cabinet", "terminal": False, "min_count": 1,
         "attributes": {"color": "blue", "state": "closed"}, "detector_terms": ["hidden vocabulary"]},
        {"id": "other", "name": "writing resource", "terminal": True, "min_count": 2,
         "demand": {"activity": "write", "required_resource": "surface"}}])
    case.task.update(ordered=True, exhaustive=True, completion_mode="all_targets_success",
                     search_priors={"destination_hypotheses": ["vanity"]}, constraints=[
        {"id": "relation", "type": "spatial", "subject": "goal", "relation": "beside",
         "object": "anchor", "verifier": "planner", "hardness": "hard"},
        {"id": "other_relation", "type": "spatial", "subject": "anchor", "relation": "near",
         "object": "other", "verifier": "planner"}])
    case.evidence["target_id"] = "goal"
    case.evidence["task_progress"] = {"visited_targets": [], "sequence_index": 0}
    case.verify()
    payload = json.loads(case.requests[0]["messages"][1]["content"][0]["text"])
    task = payload["task"]
    assert payload["raw_instruction"] == case.raw
    assert "conflicting compiled paraphrase" not in str(payload)
    assert "hidden vocabulary" not in str(payload)
    assert task["targets"][2]["demand"] == case.task["targets"][2]["demand"]
    assert task["targets"][2]["min_count"] == 2
    assert task["constraints"] == case.task["constraints"]
    assert task["domain_predicates"][0]["id"] == "relation"
    assert task["visual_predicates"] == []
    assert task["referenced_targets"][0]["attributes"] == {"color": "blue", "state": "closed"}
    assert task["referenced_targets"][0]["terminal"] is False
    assert task["ordered"] and task["exhaustive"]
    assert task["completion_mode"] == "all_targets_success"
    assert task["search_priors"] == {"destination_hypotheses": ["vanity"]}
    assert payload["task_progress"] == case.evidence["task_progress"]


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("task_verdict", ["unknown", "false"])
def test_candidate_acceptance_does_not_complete_unestablished_task(fulltext_case, multiple, task_verdict):
    case = fulltext_case
    if multiple:
        case.raw += " Then find two lamps."
        case.task["targets"].append({"id": "other", "name": "lamp", "min_count": 2, "terminal": True})
        case.evidence["target_id"] = "goal"
    quote = "Then find two lamps" if multiple else "Find"
    case.review["requirements"].append(_requirement(quote, task_verdict, "task",
                                                   [] if task_verdict == "unknown" else ["current"]))
    case.review["task_satisfied"] = True  # An inconsistent self-report is normalized.
    result = case.verify()
    assert result.accepted
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]


def _install_history(case, tmp_path):
    from verification.images import image_input
    from verification.store import instruction_hash
    path = tmp_path / "history.png"
    Image.new("RGB", (8, 8), "green").save(path)
    _, archive, digest = image_input(str(path), tmp_path / "history-blobs")
    record = {"candidate_uid": "previous", "status": "accepted", "verdict": "true",
              "instruction_hash": instruction_hash(case.raw), "task_revision": 1,
              "evidence_paths": [archive], "dependencies": {"image:" + archive: digest}, "step": 3,
              "candidate_observations": [{"archive_path": archive, "candidate_uid": "previous",
                                          "task_revision": 1, "observation_id": "previous:3",
                                          "bbox": [1, 1, 6, 6], "role": "current"}]}
    case.evidence["prior_verifications"] = [record]
    return record


def test_verified_history_is_sent_and_only_task_scope_can_cite_it(fulltext_case, tmp_path):
    case = fulltext_case
    case.raw += " after visiting the lamp"
    record = _install_history(case, tmp_path)
    row = _requirement("after visiting the lamp", scope="task", support=[])
    case.review["requirements"].append(row)
    case.make_response = lambda payload: row.update(
        supporting_observation_ids=payload["observation_support"]["verified_history"])
    result = case.verify()
    assert result.accepted and result.diagnostics["original_instruction_review"]["task_satisfied"]
    history_ids = result.diagnostics["sent_history_observation_ids"]
    assert len(history_ids) == 1
    content = case.requests[0]["messages"][1]["content"]
    payload = json.loads(content[0]["text"])
    assert payload["prior_verifications"][0]["supporting_observation_ids"] == history_ids
    label = json.loads(content[3]["text"])
    assert label["role"] == "verified_history_candidate"
    assert label["candidate_uid"] == "previous" and label["task_revision"] == 1
    assert label["instruction_hash"] == record["instruction_hash"]
    assert label["observation_identity"] == {
        "candidate_uid": "previous", "task_revision": 1, "observation_id": "previous:3"}
    assert label["bbox_xyxy_in_this_image"] == [1, 1, 6, 6]
    assert label["original_role"] == "current"
    assert label["image_sha256"] == record["dependencies"]["image:" + record["evidence_paths"][0]]
    assert label["image_sha256"] in label["observation_id"]
    assert base64.b64decode(content[4]["image_url"]["url"].split(",", 1)[1]) == Path(record["evidence_paths"][0]).read_bytes()
    assert case.verify().diagnostics["sent_history_observation_ids"] == history_ids
    row["scope"] = "candidate"
    result = case.verify()
    assert not result.accepted and result.verdict == "unknown"


@pytest.mark.parametrize("invalid", ["revision", "instruction", "status", "verdict", "missing_file", "changed_bytes"])
def test_stale_or_unverified_history_cannot_support_task(fulltext_case, tmp_path, invalid):
    case = fulltext_case
    record = _install_history(case, tmp_path)
    if invalid == "revision":
        record["task_revision"] = 0
    elif invalid == "instruction":
        record["instruction_hash"] = "other-instruction"
    elif invalid == "status":
        record["status"] = "needs_better_view"
    elif invalid == "verdict":
        record["verdict"] = "unknown"
    elif invalid == "missing_file":
        record["evidence_paths"][0] += ".missing.png"
    else:
        record["dependencies"] = {"image:" + record["evidence_paths"][0]: "obsolete-digest"}
    case.review["requirements"].append(_requirement("Find", scope="task", support=["old-history"]))
    result = case.verify()
    assert result.verdict == "unknown" and not result.accepted
    assert result.diagnostics["sent_history_observation_ids"] == []
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]


@pytest.mark.parametrize("invalid", [
    "legacy", "empty", "malformed", "missing_role", "reference", "relation", "overlap", "history",
    "wrong_uid", "wrong_revision", "missing_id", "empty_id", "bbox", "no_sha", "empty_sha",
    "path_not_dependency", "path_not_evidence", "changed_archive",
])
def test_history_requires_explicit_candidate_binding_and_actual_image_dependency(fulltext_case, tmp_path, invalid):
    case = fulltext_case
    record = _install_history(case, tmp_path)
    binding = record["candidate_observations"][0]
    archive = record["evidence_paths"][0]
    if invalid == "legacy":
        record.pop("candidate_observations")
    elif invalid == "empty":
        record["candidate_observations"] = []
    elif invalid == "malformed":
        record["candidate_observations"] = ["not a binding"]
    elif invalid == "missing_role":
        binding.pop("role")
    elif invalid in ("reference", "relation", "overlap", "history"):
        binding["role"] = invalid
    elif invalid == "wrong_uid":
        binding["candidate_uid"] = "different-candidate"
    elif invalid == "wrong_revision":
        binding["task_revision"] = 0
    elif invalid == "missing_id":
        binding.pop("observation_id")
    elif invalid == "empty_id":
        binding["observation_id"] = " "
    elif invalid == "bbox":
        binding["bbox"] = [0, 0, 0, 0]
    elif invalid == "no_sha":
        record["dependencies"] = {}
    elif invalid == "empty_sha":
        record["dependencies"]["image:" + archive] = ""
    elif invalid == "path_not_evidence":
        record["evidence_paths"] = []
    elif invalid == "path_not_dependency":
        # Even an extant path listed as evidence has no image dependency for this binding.
        binding["archive_path"] = case.evidence["current_rgb_with_bbox_path"]
        record["evidence_paths"].append(binding["archive_path"])
    else:
        Image.new("RGB", (8, 8), "blue").save(archive)
    result = case.verify()
    assert result.diagnostics["sent_history_observation_ids"] == []
    content = case.requests[0]["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 1
    assert "verified_history_candidate" not in str(content)
    assert {item["candidate_uid"] for item in result.candidate_observations} == {"u"}


def test_relation_ids_are_stable_and_cannot_replace_current_support(fulltext_case, tmp_path):
    case = fulltext_case
    path = tmp_path / "relation.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    case.evidence["relation_evidence_paths"] = [str(path)]
    row = case.review["requirements"][1]
    case.make_response = lambda payload: row.update(
        supporting_observation_ids=["current", *payload["observation_support"]["relation"]])
    result = case.verify()
    assert result.accepted
    ids = result.diagnostics["sent_relation_observation_ids"]
    assert len(ids) == 1
    content = case.requests[0]["messages"][1]["content"]
    assert json.loads(content[3]["text"])["observation_id"] == ids[0]
    assert base64.b64decode(content[4]["image_url"]["url"].split(",", 1)[1]) == path.read_bytes()
    assert case.verify().diagnostics["sent_relation_observation_ids"] == ids
    case.make_response = lambda payload: row.update(supporting_observation_ids=payload["observation_support"]["relation"])
    assert case.verify().verdict == "unknown"


@pytest.mark.parametrize("gate", ["distance", "projection", "surface", "reachability"])
def test_complete_review_cannot_bypass_physical_gates_on_cache_hit(fulltext_case, monkeypatch, gate):
    case = fulltext_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    assert case.verify().accepted
    if gate == "distance":
        case.evidence["geometry"].update(distance_to_object=3, required_stop_distance=1)
    elif gate == "projection":
        case.evidence["geometry"]["projection_failed_in_final_view"] = True
    elif gate == "surface":
        case.evidence["geometry"].update(surface_match=False, surface_status="not matched",
                                        observation_identity=case.evidence["observation_identity"])
    else:
        case.evidence["hard_stop_constraints"] = {"satisfied": False, "failed": [{"name": "reachability"}]}
    case.evidence["planner_infeasibility_proof"] = {"infeasible_by_geometry": True}
    result = case.verify()
    assert not result.accepted and result.semantic_satisfied and result.verdict == "unknown"
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]
    assert len(case.requests) == 1


@pytest.mark.parametrize("verdict,support,decision", [
    ("true", ["current"], "accept"), ("false", ["current"], "reject_candidate"),
    ("unknown", [], "need_better_view"), ("true", ["invented"], "need_better_view"),
    ("false", [], "need_better_view")])
def test_fulltext_review_keeps_exact_hard_affordance_evidence_gate(fulltext_case, verdict, support, decision):
    case = fulltext_case
    case.task["constraints"] = [{"id": "capability:goal", "type": "affordance", "subject": "goal",
                                 "verifier": "vlm", "hardness": "hard"}]
    case.response["affordance_judgments"] = {"capability:goal": {
        "verdict": verdict, "reason": "Observed functional structure or missing evidence.",
        "supporting_observation_ids": support}}
    result = case.verify()
    assert result.decision == decision
    assert result.accepted == (decision == "accept")
    assert result.diagnostics["original_instruction_review"]["task_satisfied"] == result.accepted
    assert len(case.requests) == 1


@pytest.mark.parametrize("consistent,decision", [(True, "reject_candidate"), (False, "need_better_view")])
def test_fulltext_false_and_affordance_unknown_preserve_review_contract(fulltext_case, consistent, decision):
    case = fulltext_case
    case.review["interpretation_consistent"] = consistent
    case.review["requirements"][1]["verdict"] = "false"
    case.task["constraints"] = [{"id": "capability:goal", "type": "affordance", "subject": "goal",
                                 "verifier": "vlm", "hardness": "hard"}]
    case.response["affordance_judgments"] = {"capability:goal": {
        "verdict": "unknown", "reason": "Capability cannot be seen.", "supporting_observation_ids": []}}
    result = case.verify()
    assert result.decision == decision and not result.accepted


@pytest.mark.parametrize("quote", ["freestanding", "not attached to the wall", "suitable for writing"])
def test_uncompiled_attribute_negation_and_function_are_reviewed_from_original(fulltext_case, quote):
    case = fulltext_case
    case.raw = f"Find a storage unit, {quote}."
    case.task["targets"][0].update(name="storage unit", attributes={})
    case.task["search_priors"] = {"destination_hypotheses": ["cabinet"]}
    case.review["requirements"] = [_requirement("Find a storage unit"), _requirement(quote, "false")]
    result = case.verify()
    assert result.decision == "reject_candidate" and not result.accepted
    payload = json.loads(case.requests[0]["messages"][1]["content"][0]["text"])
    assert payload["raw_instruction"] == case.raw
    assert payload["task"]["targets"][0]["attributes"] == {}
    assert payload["task"]["constraints"] == []


def test_goal_reference_does_not_authorize_fulltext_support(fulltext_case, tmp_path):
    case = fulltext_case
    path = tmp_path / "goal-reference.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    case.task["targets"][0]["attributes"]["goal_reference_image"] = str(path)
    case.review["requirements"][0]["supporting_observation_ids"] = ["current", "goal_reference"]
    result = case.verify()
    assert result.verdict == "unknown" and not result.accepted
    assert str(path) in result.diagnostics["image_sources"]
    assert "goal_reference" not in result.diagnostics["sent_candidate_observation_ids"]


def test_fulltext_review_does_not_use_reason_keyword_heuristics(fulltext_case):
    case = fulltext_case
    case.response["reason"] = "not false; no missing evidence; unknown labels are not semantic truth"
    for row in case.review["requirements"]:
        row["reason"] = "The image supports the quote; do not infer identity from chair/stool/table labels."
    assert case.verify().accepted


def test_prompt_requires_literal_fulltext_review_without_domain_semantic_exemption():
    from prompting.templates import FINAL_VERIFIER_PROMPT
    assert "only task.visual_predicates" not in FINAL_VERIFIER_PROMPT
    for text in ("FULL original instruction", "reference-object attributes", "negations and functional needs",
                 "Do not\nchange any literal requirement's meaning", "NOT an\nexemption",
                 "not whole-task completion", "task_progress and prior_verifications"):
        assert text in FINAL_VERIFIER_PROMPT
    for text in ("every clause", "background narrative is not an additional completion goal",
                 "lexical omissions, not semantic truth", "constraint_precheck",
                 "explicitly revise the task", "never silently"):
        assert text in FINAL_VERIFIER_PROMPT


@pytest.mark.parametrize("raw,quotes,covered", [
    ("Find the red container beside the blue cabinet.", ["the"], False),
    ("Find the red container beside the blue cabinet.",
     ["Find the red container", "beside the blue cabinet"], True),
    ("I left my bag outside; find the red container, not the blue one.",
     ["I left my bag outside; find the red container", "not the blue one"], True),
    ("找到蓝色柜子旁边的红色容器，然后停下。", ["找到蓝色柜子旁边的红色容器", "然后停下"], True),
    ("找到蓝色柜子旁边的红色容器，然后停下。", ["红色容器"], False),
    ("Find\tred,\nblue; ２ lamps!", ["Find", "red", "blue", "２ lamps"], True),
    ("Find red, blue; ２ lamps!", ["Find", "red", "blue", "lamps"], False),
    ("Find red, blue!", [",", "!"], False),
    ("Find the red container", ["Find the red", "the red container"], True),
    ("Find the red container", ["Find", "red container"], False),
    ("go go", ["go"], True),
    ("aaaaa", ["aaa"], True),
])
def test_fulltext_exact_quote_union_covers_every_original_alphanumeric(fulltext_case, raw, quotes, covered):
    case = fulltext_case
    case.raw = raw
    case.review["requirements"] = [_requirement(quote) for quote in quotes]
    result = case.verify()
    review = result.diagnostics["original_instruction_review"]
    assert result.accepted is covered
    assert review["coverage_complete"] is covered
    assert review["task_satisfied"] is covered
    audit = review["lexical_coverage"]
    intervals = audit["covered_intervals"]
    assert all(end < next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:]))
    assert all(any(start <= i < end for start, end in intervals)
               for i, char in enumerate(raw) if char.isalnum()) is covered
    assert bool(audit["uncovered_fragments"]) is not covered
    for fragment in audit["uncovered_fragments"]:
        assert fragment["text"] == raw[fragment["start"]:fragment["end"]]
        assert fragment["text"] in result.reason
    if not covered:
        assert result.decision == "need_better_view" and result.verdict == "unknown"
        assert not result.semantic_satisfied
        assert "Uncovered raw_instruction fragments" in result.reason
    assert len(case.requests) == 1


def _domain_precheck(satisfied=False, failure="missing_anchor"):
    return {"satisfied": satisfied, "decision": "accept" if satisfied else "need_relation_check",
            "reason": "Declared relation established." if satisfied else "Declared hard relation unresolved: " + failure,
            "failed_constraints": [] if satisfied else [failure], "satisfied_constraints": ["known_geometry"],
            "relation_edges": [{"subject": "u", "object": "anchor", "relation": "beside",
                                "verdict": "true" if satisfied else "unknown"}]}


@pytest.mark.parametrize("failure", ["missing_anchor", "geometry_false"])
def test_fulltext_accept_cannot_ignore_failed_domain_precheck(fulltext_case, failure):
    case = fulltext_case
    precheck = _domain_precheck(failure=failure)
    if failure == "geometry_false":
        precheck["decision"] = "reject_relation"
        precheck["relation_edges"][0]["verdict"] = "false"
    case.evidence["constraint_precheck"] = precheck
    case.response["satisfied_constraints"] = [failure]  # A contradictory self-report cannot erase failure.
    before = deepcopy((case.evidence, case.response))
    result = case.verify()
    assert not result.accepted and not result.satisfied
    assert result.decision == "need_relation_check" and result.verdict == "unknown"
    assert result.semantic_satisfied  # Domain failure does not erase independent visual analysis.
    review = result.diagnostics["original_instruction_review"]
    assert review["coverage_complete"] and not review["validation_errors"] and not review["task_satisfied"]
    assert result.diagnostics["parsed_satisfied"] is False
    assert result.diagnostics["constraint_precheck"] == precheck
    assert failure in result.failed_constraints and failure not in result.satisfied_constraints
    assert precheck["reason"] in result.reason
    assert len(case.requests) == 1  # Failed precheck must not early-block full-text review.
    payload = json.loads(case.requests[0]["messages"][1]["content"][0]["text"])
    assert payload["raw_instruction"] == case.raw
    assert payload["constraint_precheck"] == precheck
    assert "evidence" not in payload["constraint_precheck"]
    assert "not raw_instruction interpretation authority" in payload["constraint_precheck_policy"]
    assert "explicit task revision" in payload["constraint_precheck_policy"]
    assert (case.evidence, case.response) == before
    result.diagnostics["constraint_precheck"]["relation_edges"][0]["verdict"] = "changed"
    assert case.evidence == before[0]


@pytest.mark.parametrize("present", [False, True])
def test_successful_or_absent_precheck_preserves_compatible_acceptance(fulltext_case, present):
    case = fulltext_case
    if present:
        case.evidence["constraint_precheck"] = _domain_precheck(satisfied=True)
    result = case.verify()
    assert result.accepted and result.diagnostics["original_instruction_review"]["task_satisfied"]
    assert result.diagnostics["constraint_precheck"] == case.evidence.get("constraint_precheck", {})
    assert len(case.requests) == 1


@pytest.mark.parametrize("review_verdict,malformed,decision", [
    ("false", False, "reject_candidate"), ("unknown", False, "need_relation_check"),
    ("false", True, "need_relation_check"),
])
def test_failed_precheck_preserves_only_reliable_candidate_rejection(fulltext_case, review_verdict, malformed, decision):
    case = fulltext_case
    case.evidence["constraint_precheck"] = _domain_precheck()
    case.review["requirements"][1]["verdict"] = review_verdict
    if malformed:
        case.review["requirements"][1]["supporting_observation_ids"] = ["invented"]
    result = case.verify()
    assert result.decision == decision and not result.accepted
    assert result.diagnostics["constraint_precheck"]["failed_constraints"] == ["missing_anchor"]
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]
    assert len(case.requests) == 1


def test_failed_precheck_keeps_missing_current_no_model_contract(fulltext_case):
    case = fulltext_case
    case.evidence["constraint_precheck"] = _domain_precheck()
    case.evidence.pop("observation_identity")
    result = case.verify()
    assert result.decision == "need_better_view" and not result.accepted
    assert result.failed_constraints == ["missing_anchor"]
    assert not case.requests and result.candidate_observations == []


def test_precheck_changes_invalidate_fulltext_cache_and_keep_physical_gate(fulltext_case, monkeypatch):
    case = fulltext_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    case.evidence["constraint_precheck"] = _domain_precheck(satisfied=True)
    assert case.verify().accepted and case.verify().accepted
    assert len(case.requests) == 1
    case.evidence["constraint_precheck"] = _domain_precheck()
    result = case.verify()
    assert result.decision == "need_relation_check" and not result.accepted
    assert case.verify().decision == "need_relation_check"
    assert len(case.requests) == 2
    case.evidence["hard_stop_constraints"] = {"satisfied": False, "failed": [{"name": "distance"}]}
    result = case.verify()
    assert not result.hard_constraints["satisfied"] and not result.accepted
    assert set(result.failed_constraints) == {"missing_anchor", "distance"}
    case.evidence["constraint_precheck"] = _domain_precheck(satisfied=True)
    result = case.verify()
    assert result.decision == "need_better_view" and not result.accepted
    assert not result.diagnostics["original_instruction_review"]["task_satisfied"]
    assert len(case.requests) == 2
