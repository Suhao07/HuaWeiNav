"""Scripted shared verifier contracts; these tests do not measure real VLM ability."""
import json
import types
import pytest

from instruction_adapter.verifier import (
    CandidateInstance,
    EvidenceStore,
    VerificationResult,
    FinalInstructionVerifier,
    _planner_hard_constraint_infeasible,
    hard_stop_constraints_from_evidence,
)


def _scripted_response(request, verdict="unknown"):
    if request["trace_label"] == "instance_observer":
        # 观察器输出多实例结构化事实，授权一次格式重生成；终验只做一次判定，不重生成。
        assert request["parse_retries"] == 1
        content = request["messages"][1]["content"]
        identities = [json.loads(item["text"])["observation_identity"]
                      for item in content if item["type"] == "text"]
        parsed = request["response_format"](observations={identity["observation_id"]: {
            "observed_category": "chair", "visible_parts": ["seat", "back", "legs"],
            "support_and_attachment": "Legs on the floor.", "visible_surface_content": "",
            "visible_context": [], "uncertainties": []} for identity in identities})
    else:
        assert request["trace_label"] == "final_instruction_verifier"
        assert request["parse_retries"] == 0
        assert any(item["type"] == "image_url" for item in request["messages"][1]["content"])
        parsed = request["response_format"](observed_category="chair", observed_structure="Seat, back and legs.",
            interpretation_conflicts=[],
            predicate_judgments={"candidate_identity": {"verdict": verdict,
                "reason": "Scripted instance evidence.", "supporting_observation_ids": ["now"]}})
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))])


@pytest.mark.parametrize("field,limit", [("observed_category", 80), ("observed_structure", 240)])
def test_visual_instance_descriptions_have_guided_output_bounds(field, limit):
    from pydantic import ValidationError
    from prompting.schemas import visual_verification_response_model

    model = visual_verification_response_model(["candidate_identity"])
    assert model.model_json_schema()["properties"][field]["maxLength"] == limit
    payload = {field: "x" * limit, "interpretation_conflicts": [], "predicate_judgments": {
        "candidate_identity": {"verdict": "unknown", "reason": "Instance is unresolved.",
                               "supporting_observation_ids": []}}}
    assert getattr(model.model_validate(payload), field) == "x" * limit
    payload[field] += "x"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


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
    image = tmp_path / "current.png"
    image.write_bytes(b"frame")
    with pytest.raises(RuntimeError) as raised:
        FinalInstructionVerifier().verify("find a chair", {}, CandidateInstance("c1", "chair"), {
            "current_rgb_with_bbox_path": str(image), "geometry": {"bbox_xyxy": [0, 0, 3, 3]},
            "observation_identity": {"candidate_uid": "c1", "task_revision": 1, "observation_id": "now"}})
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


@pytest.mark.parametrize("verdict", ["true", "false", "unknown"])
@pytest.mark.parametrize("physical,infeasible", [(True, False), (False, False), (False, True)])
def test_final_accept_requires_evidence_and_physical_contract(monkeypatch, tmp_path, verdict, physical, infeasible):
    import instruction_adapter.verifier as module
    image = tmp_path / "current.png"
    image.write_bytes(b"frame")
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return _scripted_response(kwargs, verdict)
    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    result = FinalInstructionVerifier().verify("find object", {}, CandidateInstance("u", "object"), {
        "current_rgb_with_bbox_path": str(image), "geometry": {"bbox_xyxy": [0, 0, 3, 3]},
        "observation_identity": {"candidate_uid": "u", "task_revision": 1, "observation_id": "now"},
        "hard_stop_constraints": {"satisfied": physical, "planner_infeasible": infeasible,
                                  "planner_infeasibility_proof": "blocked" if infeasible else ""},
        "relation_outcomes": [{"predicate_id": "c0", "status": "verified", "anchor_uid": "reference"}]})
    assert [request["trace_label"] for request in requests] == [
        "instance_observer", "final_instruction_verifier"]
    assert result.view_objective["relation_outcomes"] == [
        {"predicate_id": "c0", "status": "verified", "anchor_uid": "reference"}]
    assert result.accepted == (verdict == "true" and physical)
    assert result.semantic_satisfied == (verdict == "true")
    if verdict == "false":
        assert result.decision == "reject_candidate"


def test_final_request_separates_execution_history_from_visual_evidence(monkeypatch, tmp_path):
    import json
    import instruction_adapter.verifier as module
    image = tmp_path / "current.png"
    image.write_bytes(b"frame")
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return _scripted_response(kwargs)
    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    FinalInstructionVerifier().verify("find chair", {}, CandidateInstance("u", "chair", confidence=.99), {
        "current_rgb_with_bbox_path": str(image),
        "observation_identity": {"candidate_uid": "u", "task_revision": 1, "observation_id": "now"},
        "geometry": {"distance_to_object": .293, "bbox_xyxy": [0, 0, 3, 3]},
        "view_control": {"budget_exhausted": True, "attempts": [{"semantic_satisfied": True}]},
        "final_verifier_attempt": {"budget_exhausted": True}, "bbox_verified": True})
    assert [request["trace_label"] for request in requests] == [
        "instance_observer", "final_instruction_verifier"]
    observer, final = requests
    payload = json.loads(final["messages"][1]["content"][0]["text"])
    assert payload["candidate"] == {"uid": "u"}
    assert payload["raw_instruction"] == "find chair"
    assert "evidence" not in payload
    assert "geometry" not in payload
    for request in requests:
        assert "hard_constraints" not in request["response_format"].model_json_schema()["properties"]
        assert all(token not in json.dumps(request["messages"][1]["content"]) for token in (
            "distance_to_object", "budget_exhausted", "semantic_satisfied", "bbox_verified", "0.99"))
    assert "find chair" not in json.dumps(observer["messages"])
    content = final["messages"][1]["content"]
    packet = next(value for item in content if item["type"] == "text"
                  if "bound_observations" in (value := json.loads(item["text"])))
    assert set(packet["bound_observations"]) == {"now"}
    assert json.loads(content[1]["text"]) == {
        "role": "current_candidate_observation", "image_kind": "current_rgb_with_bbox_path",
        "observation_identity": {"candidate_uid": "u", "task_revision": 1, "observation_id": "now"},
        "bbox_xyxy_in_this_image": [0, 0, 3, 3], "depth_support": None}
    assert [item for item in content if item["type"] == "image_url"] == [
        observer["messages"][1]["content"][1]]


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
    assert scoped["referenced_targets"] == [{"id": "shelf", "name": "shelf", "attributes": {}}]
    assert scoped["domain_predicates"][0]["relation"] == "on"
    assert scoped["visual_predicates"] == []
    assert "cup" not in str(scoped) and "predicted vocabulary" not in str(scoped)
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
    assert next_target["domain_predicates"][0]["relation"] == "on"
    assert next_target["domain_predicates"][0]["verifier"] == "vlm"
    assert next_target["visual_predicates"] == []


@pytest.mark.parametrize("locked", [False, True, None])
@pytest.mark.parametrize("verdict", ["true", "false", "unknown"])
@pytest.mark.parametrize("multiple", [False, True])
def test_original_visual_requirements_need_owner_authority_and_bound_evidence(monkeypatch, tmp_path, locked, verdict, multiple):
    from instruction.contracts import TaskSpec, TargetSpec, ConstraintSpec
    from instruction_adapter.task_compat import runtime_plan
    import instruction_adapter.verifier as module

    targets = [TargetSpec("surface", "surface", attributes={"material": "wood"})]
    text = "Find a surface for writing while standing."
    if multiple:
        targets.append(TargetSpec("second", "surface", attributes={"material": "metal"}))
        text = "Find a wooden surface for writing while standing, then a metal surface."
    task = TaskSpec(text, tuple(targets), (
        ConstraintSpec("use", "affordance", subject="surface", value="writing", verifier="vlm"),))
    plan = runtime_plan(task)
    original = plan.as_dict()
    image = tmp_path / "current.png"
    image.write_bytes(b"frame")
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        if kwargs["trace_label"] == "instance_observer":
            return _scripted_response(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
        original_mode = locked is False
        assert payload["verification_basis"] == ("original_instruction" if original_mode else "fixed_task")
        ids = [predicate["id"] for predicate in payload["verification_predicates"]]
        judgment = {"verdict": "true", "reason": "Scripted current support.",
                    "supporting_observation_ids": ["now"]}
        fields = {"predicate_judgments": {uid: dict(judgment) for uid in ids}}
        if original_mode:
            assert ids == ["candidate_identity"]
            scope = payload["task"]["active_target"]
            assert scope["id"] == "surface"
            if multiple:
                assert scope["compiled_scope_hint"] == {
                    "id": "surface", "name": "surface", "attributes": {"material": "wood"}, "demand": {}}
            else:
                assert scope == {"id": "surface"}
            assert payload["task"]["visual_predicates"] == []
            fields["instruction_judgments"] = [{**judgment, "verdict": verdict,
                "instruction_quote": "writing while standing", "requirement": "Supports writing while standing."}]
        else:
            assert ids == ["candidate_identity", "attribute:surface:material", "use"]
            fields["predicate_judgments"]["attribute:surface:material"]["verdict"] = "false"
        parsed = kwargs["response_format"](observed_category="surface", observed_structure="Raised horizontal top.",
                                           interpretation_conflicts=[], **fields)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=parsed))])

    client = types.SimpleNamespace(beta=types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    evidence = {"target_id": "surface", "current_rgb_with_bbox_path": str(image),
                "geometry": {"bbox_xyxy": [0, 0, 3, 3]},
                "observation_identity": {"candidate_uid": "u", "task_revision": 1, "observation_id": "now"},
                "hard_stop_constraints": {"satisfied": True}}
    if locked is not None:
        evidence["interpretation_locked"] = locked
    result = FinalInstructionVerifier().verify(task.raw_instruction, plan, CandidateInstance("u", "surface"), evidence)
    assert [request["trace_label"] for request in requests] == ["instance_observer", "final_instruction_verifier"]
    assert result.accepted == (locked is False and verdict == "true")
    if locked is False:
        assert result.diagnostics["predicate_judgments"]["instruction:0"]["verdict"] == verdict
        if verdict != "true":
            record = EvidenceStore().put(task.raw_instruction, "u", result)
            assert "writing while standing" in record.reason
            assert "Supports writing while standing." in record.reason
            assert "Scripted current support." in record.reason
    else:
        assert result.decision == "reject_candidate"
    assert plan.as_dict() == original


def test_original_visual_requirements_without_current_image_remain_unknown(monkeypatch):
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    import instruction_adapter.verifier as module

    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda _: pytest.fail("No image must mean zero model calls"))
    task = TaskSpec("Find a desk for standing work.", (TargetSpec("desk", "desk"),))
    result = FinalInstructionVerifier().verify(task.raw_instruction, runtime_plan(task), CandidateInstance("u", "desk"),
                                                {"interpretation_locked": False})
    assert not result.semantic_satisfied and not result.accepted
    assert result.diagnostics["predicate_judgments"]["instruction:0"]["verdict"] == "unknown"


def test_original_visual_schema_requires_nonempty_requirements():
    from pydantic import ValidationError
    from prompting.schemas import visual_verification_response_model

    model = visual_verification_response_model(["candidate_identity"], original_instruction=True)
    fields = list(model.model_fields)
    assert fields[0] == "instruction_reading"
    assert fields.index("instruction_judgments") < fields.index("predicate_judgments") < fields.index("confidence")
    from prompting.schemas import InstructionRequirementJudgment, VisualPredicateJudgment
    assert list(InstructionRequirementJudgment.model_fields) == [
        "instruction_quote", "requirement", "reason", "supporting_observation_ids", "verdict"]
    assert list(VisualPredicateJudgment.model_fields)[-1] == "verdict"
    payload = {"interpretation_conflicts": [], "predicate_judgments": {"candidate_identity": {
        "verdict": "unknown", "reason": "No current evidence.", "supporting_observation_ids": []}}}
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    with pytest.raises(ValidationError):
        model.model_validate({**payload, "instruction_judgments": []})


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
