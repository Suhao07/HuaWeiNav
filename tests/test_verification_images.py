"""The archived evidence is exactly the byte sequence sent to the model."""
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verification.images import image_input, image_dependencies_valid
from verification.contracts import CandidateInstance
from verification.store import EvidenceStore


def test_source_overwrite_preserves_archive_and_archive_loss_expires_evidence(tmp_path):
    source = tmp_path / "camera.png"
    source.write_bytes(b"first observed frame")
    block, reference, digest = image_input(str(source), tmp_path / "blobs")
    dependencies = {"image:" + reference: digest}
    assert base64.b64decode(block["image_url"]["url"].split(",", 1)[1]) == Path(reference).read_bytes()
    source.write_bytes(b"next observed frame")
    assert image_dependencies_valid(dependencies)
    Path(reference).unlink()
    assert not image_dependencies_valid(dependencies)


def test_verifier_records_exact_sent_images_and_expired_cache_is_ignored(tmp_path, monkeypatch):
    import instruction_adapter.verifier as module

    image_path = tmp_path / "camera.png"
    image_path.write_bytes(b"observed frame")
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        from prompting.schemas import ParsedVerification
        parsed = ParsedVerification(decision="accept", satisfied=True, semantic_satisfied=True,
                                 view_sufficient_for_stop=True, confidence=1.0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test-model"))
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find a chair", {}, CandidateInstance("chair-1", "chair"),
        {"current_rgb_with_bbox_path": str(image_path)},
    )
    assert result.accepted
    assert result.target_uid == "chair-1"
    archived = Path(result.evidence_ids[0])
    sent = next(item for item in requests[0]["messages"][1]["content"] if item["type"] == "image_url")
    assert base64.b64decode(sent["image_url"]["url"].split(",", 1)[1]) == archived.read_bytes()
    store = EvidenceStore(1)
    store.put("chair", "chair-1", result)
    assert store.get("chair", "chair-1").status == "accepted"
    archived.write_bytes(b"changed archive")
    assert store.get("chair", "chair-1") is None


def test_each_image_uses_its_own_observed_box_and_identity(tmp_path, monkeypatch):
    import json
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification
    current, earlier, unrelated = (tmp_path / name for name in ("current.jpg", "earlier.jpg", "unrelated.jpg"))
    for path in (current, earlier, unrelated):
        path.write_bytes(path.name.encode())
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=ParsedVerification(satisfied=False, decision="uncertain")))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client,"test"))
    identity = {"candidate_uid": "u", "task_revision": 1, "observation_id": "current"}
    evidence = {"observation_identity": identity, "current_rgb_with_bbox_path": str(current),
                "centered_view_path": str(current), "geometry": {"bbox_xyxy": [1,2,3,4], "distance_to_object": .29},
                "nearby_objects": [{"tag": "chair"}], "view_control": {
                    "pinned_visual_evidence": {"observation_identity": {**identity, "observation_id": "earlier"},
                        "geometry": {"bbox_xyxy": [5,6,7,8]},
                        "image_paths": {"current_rgb_with_bbox_path": str(earlier)}},
                    "best_visual_evidence": {"image_paths": {"current_rgb_with_bbox_path": str(unrelated)}}}}
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find chair", {}, CandidateInstance("u", "chair", bbox_2d=[100,200,300,400]), evidence)
    content = requests[0]["messages"][1]["content"]
    texts = [json.loads(item["text"]) for item in content if item["type"] == "text"]
    assert texts[0]["candidate"] == {"uid": "u"}
    assert texts[1]["bbox_xyxy_in_this_image"] == [1,2,3,4]
    assert texts[2]["bbox_xyxy_in_this_image"] == [5,6,7,8]
    assert texts[2]["observation_identity"]["observation_id"] == "earlier"
    assert sum(item["type"] == "image_url" for item in content) == 2
    assert len(result.diagnostics["excluded_history"]) == 1
    assert all("distance_to_object" not in str(text) and "nearby_objects" not in str(text) for text in texts)


def test_overlapping_instance_view_is_sent_without_its_map_label(tmp_path, monkeypatch):
    import json
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification
    current, overlap = tmp_path / "current.jpg", tmp_path / "overlap.jpg"
    current.write_bytes(b"current")
    overlap.write_bytes(b"front view")
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=ParsedVerification(satisfied=False, decision="reject_candidate")))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find sofa", {}, CandidateInstance("u", "sofa"),
        {"current_rgb_with_bbox_path": str(current), "overlapping_instance_observations": [
            {"image_path": str(overlap), "bbox_xyxy": [1, 2, 3, 4], "candidate_surface_fraction": .9}]})
    content = requests[0]["messages"][1]["content"]
    texts = [json.loads(item["text"]) for item in content if item["type"] == "text"]
    assert texts[-1] == {"role": "overlapping_instance_observation",
                         "candidate_surface_fraction_on_this_instance": .9,
                         "bbox_xyxy_in_this_image": [1, 2, 3, 4]}
    assert content[-1]["type"] == "image_url" and len(result.evidence_ids) == 2


def test_exact_visual_cache_rechecks_current_physical_stop(tmp_path, monkeypatch):
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification
    image = tmp_path/'current.jpg'
    image.write_bytes(b'current image')
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=ParsedVerification(
            decision='accept', satisfied=True, semantic_satisfied=True, view_sufficient_for_stop=True)))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(module, 'get_client_and_model', lambda _: (client,'test'))
    monkeypatch.setenv('LLM_OFFLINE','0')
    monkeypatch.setenv('STRIVE_FINAL_VERIFIER','on')
    monkeypatch.setenv('STRIVE_EVIDENCE_REUSE','1')
    verifier = module.FinalInstructionVerifier(evidence_dir=str(tmp_path/'blobs'))
    evidence = {'current_rgb_with_bbox_path':str(image), 'hard_stop_constraints':{'satisfied':True}}
    candidate = CandidateInstance('u','chair')
    first = verifier.verify('chair',{},candidate,evidence)
    evidence['hard_stop_constraints'] = {'satisfied':False,'failed':[{'name':'distance'}]}
    second = verifier.verify('chair',{},candidate,evidence)
    assert first.accepted and not second.accepted
    assert len(calls) == 1 and first.evidence_ids == second.evidence_ids
    image.write_bytes(b'new image')
    verifier.verify('chair',{},candidate,evidence)
    assert len(calls) == 2
    verifier.verify('red chair',{},candidate,evidence)
    assert len(calls) == 3


def test_goal_reference_is_sent_separately_and_recorded_as_image_dependency(tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    from PIL import Image
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification

    current, reference = tmp_path / "current.png", tmp_path / "goal.png"
    Image.new("RGB", (2, 2), "red").save(current)
    Image.new("RGB", (2, 2), "blue").save(reference)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=ParsedVerification(satisfied=False, decision="uncertain")))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "test"))
    plan = {"targets": [
        {"id": "other", "name": "chair", "attributes": {"goal_reference_image": str(tmp_path / "unused.png")}},
        {"id": "goal", "name": "chair", "attributes": {"goal_reference_image": str(reference)}},
    ]}
    identity = {"candidate_uid": "u", "task_revision": 1, "observation_id": "current"}
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find the pictured chair", plan, CandidateInstance("u", "chair"),
        {"target_id": "goal", "observation_identity": identity, "current_rgb_with_bbox_path": str(current)})
    content = requests[0]["messages"][1]["content"]
    assert len(content) == 5
    assert json.loads(content[1]["text"])["role"] == "current_candidate_observation"
    assert json.loads(content[1]["text"])["observation_identity"] == identity
    goal_role = json.loads(content[3]["text"])
    assert goal_role["role"] == "goal_reference"
    assert "user's goal reference image" in goal_role["instruction"]
    assert "same physical instance" in goal_role["instruction"]
    assert "category alone is insufficient" in goal_role["instruction"]
    assert "not a current candidate observation" in goal_role["instruction"]
    assert "observation_identity" not in goal_role
    for source, block in ((current, content[2]), (reference, content[4])):
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        sent_bytes = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        archived = result.diagnostics["image_sources"][str(source)]
        assert sent_bytes == source.read_bytes() == Path(archived).read_bytes()
        assert archived in result.evidence_ids
        assert result.dependencies["image:" + archived] == sha256(sent_bytes).hexdigest()
    assert image_dependencies_valid(result.dependencies)
    Path(result.diagnostics["image_sources"][str(reference)]).unlink()
    assert not image_dependencies_valid(result.dependencies)


def test_goal_reference_content_and_revision_invalidate_visual_cache(tmp_path, monkeypatch):
    import json
    from PIL import Image
    import instruction_adapter.verifier as module
    from prompting.schemas import ParsedVerification

    current, reference = tmp_path / "current.png", tmp_path / "goal.png"
    Image.new("RGB", (2, 2), "red").save(current)
    Image.new("RGB", (2, 2), "blue").save(reference)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=ParsedVerification(satisfied=False, decision="uncertain")))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "test"))
    verifier = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs"))
    attributes = {"goal_reference_image": str(reference)}
    task = {"revision": 1, "targets": [{"id": "goal", "name": "chair", "attributes": attributes}]}
    plan = {"diagnostics": {"task_spec": task}}
    candidate = CandidateInstance("u", "chair")
    evidence = {"current_rgb_with_bbox_path": str(current)}
    first = verifier.verify("find the pictured chair", plan, candidate, evidence)
    repeat = verifier.verify("find the pictured chair", plan, candidate, evidence)
    assert len(requests) == 1 and repeat.dependencies == first.dependencies
    Image.new("RGB", (2, 2), "green").save(reference)
    changed = verifier.verify("find the pictured chair", plan, candidate, evidence)
    assert len(requests) == 2 and changed.dependencies != first.dependencies
    task["revision"] = 2
    revised = verifier.verify("find the pictured chair", plan, candidate, evidence)
    assert len(requests) == 3 and revised.task_revision == 2
    assert requests[1]["messages"] == requests[2]["messages"]
    attributes.pop("goal_reference_image")
    absent = verifier.verify("find the pictured chair", plan, candidate, evidence)
    assert len(requests) == 4
    content = requests[-1]["messages"][1]["content"]
    assert len(content) == 3
    assert json.loads(content[1]["text"])["role"] == "current_candidate_observation"
    assert "goal_reference" not in str(requests[-1]["messages"])
    assert absent.evidence_ids == [absent.diagnostics["image_sources"][str(current)]]
    assert {key for key in absent.dependencies if key.startswith("image:")} == {
        "image:" + absent.evidence_ids[0]}


def test_affordance_schema_requires_exact_complete_predicate_judgments():
    from pydantic import ValidationError
    from prompting.schemas import ParsedVerification, affordance_verification_response_model

    assert affordance_verification_response_model([]) is ParsedVerification
    model = affordance_verification_response_model(["demand:goal", "design:goal"])
    judgment = {"verdict": "unknown", "reason": "Occluded resource.", "supporting_observation_ids": []}
    complete = {uid: dict(judgment) for uid in ("demand:goal", "design:goal")}
    parsed = model(affordance_judgments=complete)
    assert parsed.affordance_judgments.model_dump() == complete
    schema = model.model_json_schema()
    nested = schema["$defs"]["AffordanceJudgments"]
    assert set(nested["required"]) == set(complete)
    assert nested["additionalProperties"] is False
    assert set(schema["$defs"]["AffordanceJudgment"]["required"]) == set(judgment)
    for incomplete in ({}, {"demand:goal": judgment}, {**complete, "invented": judgment}):
        with pytest.raises(ValidationError):
            model(affordance_judgments=incomplete)
    for invalid in ({**judgment, "verdict": True}, {**judgment, "verdict": "maybe"},
                    {**judgment, "reason": ""}, {**judgment, "reason": "x" * 241},
                    {"verdict": "true", "reason": "Visible structure."}):
        with pytest.raises(ValidationError):
            model(affordance_judgments={**complete, "demand:goal": invalid})
    with pytest.raises(ValidationError):
        model(decision="accept")
    for invalid_ids in ([None], [""], ["duplicate", "duplicate"]):
        with pytest.raises(ValueError):
            affordance_verification_response_model(invalid_ids)


@pytest.fixture
def affordance_case(monkeypatch, tmp_path):
    """Keep the final verifier real, replacing only the provider boundary."""
    from PIL import Image
    import instruction_adapter.verifier as module

    current = tmp_path / "affordance.png"
    Image.new("RGB", (8, 8), "blue").save(current)
    demand = {"activity": "work with a notebook", "manipulated_objects": ["notebook"],
              "required_resource": "a usable work surface"}
    predicates = [{"id": uid, "type": "affordance", "subject": "goal", "value": demand,
                   "hardness": "hard", "verifier": "vlm"} for uid in ("demand:goal", "design:goal")]
    task = {"revision": 4, "targets": [{"id": "goal", "name": "work surface", "demand": demand}],
            "constraints": predicates, "search_priors": {"destination_hypotheses": ["desk"]}}
    identity = {"candidate_uid": "candidate", "task_revision": 4, "observation_id": "camera:4"}
    evidence = {"observation_identity": identity, "current_rgb_with_bbox_path": str(current),
                "geometry": {"bbox_xyxy": [0, 0, 7, 7], "distance_to_object": .5, "required_stop_distance": 1.0}}
    reports = {uid: {"verdict": "true", "reason": "A broad exposed horizontal top supports the notebook.",
                     "supporting_observation_ids": ["camera:4"]} for uid in ("demand:goal", "design:goal")}
    response = {"decision": "accept", "satisfied": True, "semantic_satisfied": True,
                "view_sufficient_for_stop": True, "affordance_judgments": reports}

    def parse(**kwargs):
        parsed = kwargs["response_format"](**response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    parse_mock = Mock(side_effect=parse)
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse_mock))))
    provider = Mock(return_value=(client, "offline-affordance-replay"))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")  # Only the mocked provider can be reached.
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "0")
    monkeypatch.setattr(module, "get_client_and_model", provider)
    verifier = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs"))
    candidate = CandidateInstance("candidate", "unfamiliar detector category")
    plan = {"diagnostics": {"task_spec": task}}
    return SimpleNamespace(verifier=verifier, task=task, plan=plan, evidence=evidence,
                           reports=reports, response=response, current=current, candidate=candidate,
                           parse=parse_mock, provider=provider,
                           verify=lambda: verifier.verify("Find somewhere to work with my notebook", plan, candidate, evidence))


def test_affordance_task_projection_preserves_demand_and_ids_without_search_priors(affordance_case):
    from copy import deepcopy

    case = affordance_case
    case.task["targets"].append({"id": "other", "name": "other resource", "demand": {"activity": "other"}})
    other = {**case.task["constraints"][0], "id": "demand:other", "subject": "other"}
    soft = {**case.task["constraints"][0], "id": "optional", "hardness": "soft"}
    case.task["constraints"].extend([other, soft])
    case.task["constraints"][0]["verifier"] = "planner"
    case.evidence.update(target_id="goal", declared_attribute_constraints=[case.task["constraints"][0]])
    before = deepcopy(case.plan)
    projected = case.verifier._visual_task(case.plan, case.evidence)
    assert projected["active_target"]["demand"] == case.task["targets"][0]["demand"]
    assert [p["id"] for p in projected["visual_predicates"]] == ["demand:goal", "design:goal", "optional"]
    assert "search_priors" not in str(projected) and "destination_hypotheses" not in str(projected)
    projected["active_target"]["demand"]["manipulated_objects"].append("changed")
    projected["visual_predicates"][0]["value"]["activity"] = "changed"
    assert case.plan == before
    assert case.verify().accepted
    schema = case.parse.call_args.kwargs["response_format"].model_json_schema()
    assert set(schema["$defs"]["AffordanceJudgments"]["required"]) == {"demand:goal", "design:goal"}


def test_affordance_request_sends_current_candidate_bytes_and_records_judgments(affordance_case):
    import json
    from copy import deepcopy

    case = affordance_case
    before = deepcopy((case.plan, case.evidence))
    result = case.verify()
    assert result.accepted and result.task_revision == 4
    assert case.parse.call_count == 1
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert json.loads(content[0]["text"])["task"]["active_target"]["demand"] == case.task["targets"][0]["demand"]
    bound = json.loads(content[1]["text"])
    assert bound["observation_identity"] == case.evidence["observation_identity"]
    assert bound["bbox_xyxy_in_this_image"] == [0, 0, 7, 7]
    sent_bytes = base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1])
    assert sent_bytes == case.current.read_bytes() == Path(result.evidence_ids[0]).read_bytes()
    assert result.diagnostics["affordance_judgments"] == case.reports
    assert result.diagnostics["sent_current_observation_ids"] == ["camera:4"]
    assert set(result.satisfied_constraints) == set(case.reports)
    assert (case.plan, case.evidence) == before


@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError])
def test_affordance_archive_failure_is_not_missing_observation(affordance_case, monkeypatch, error_type):
    import instruction_adapter.verifier as module

    case = affordance_case
    assert case.current.is_file()
    archive = Mock(side_effect=error_type("Evidence archive unavailable"))
    monkeypatch.setattr(module, "image_input", archive)
    with pytest.raises(error_type, match="Evidence archive unavailable"):
        case.verify()
    archive.assert_called_once()
    case.provider.assert_not_called()
    case.parse.assert_not_called()


def test_affordance_current_crop_uses_crop_coordinates(affordance_case):
    import json

    case = affordance_case
    case.evidence["object_crop_path"] = case.evidence.pop("current_rgb_with_bbox_path")
    case.evidence["object_crop_bbox_xyxy"] = [1, 1, 5, 5]
    case.evidence["geometry"].pop("bbox_xyxy")
    assert case.verify().accepted
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert json.loads(content[1]["text"])["bbox_xyxy_in_this_image"] == [1, 1, 5, 5]
    assert content[2]["type"] == "image_url"


def test_affordance_direct_immutable_taskspec_retains_revision(affordance_case):
    from instruction.contracts import TaskSpec

    case = affordance_case
    task = TaskSpec.from_dict({**case.task, "raw_instruction": "Find a work surface"})
    before = task.as_dict()
    result = case.verifier.verify("Find a work surface", task, case.candidate, case.evidence)
    assert result.accepted and result.task_revision == 4
    assert task.as_dict() == before


def test_soft_affordance_keeps_original_response_schema(affordance_case):
    from prompting.schemas import ParsedVerification

    case = affordance_case
    for predicate in case.task["constraints"]:
        predicate["hardness"] = "soft"
    case.response.pop("affordance_judgments")
    result = case.verify()
    assert result.accepted and "affordance_judgments" not in result.diagnostics
    assert case.parse.call_args.kwargs["response_format"] is ParsedVerification


@pytest.mark.parametrize("verdict,decision", [("false", "reject_candidate"), ("unknown", "need_better_view")])
def test_affordance_single_nontrue_predicate_blocks_aggregate_accept(affordance_case, verdict, decision):
    case = affordance_case
    case.reports["design:goal"].update(verdict=verdict, reason="Required structure is absent or occluded.")
    result = case.verify()
    assert not result.accepted and not result.semantic_satisfied and result.decision == decision
    assert result.diagnostics["affordance_judgments"]["design:goal"]["verdict"] == verdict
    assert "design:goal" not in result.satisfied_constraints
    assert ("design:goal" in result.failed_constraints) == (verdict == "false")


@pytest.mark.parametrize("support", [[], ["invented"], ["camera:4", "invented"], ["reference"], ["earlier"]])
@pytest.mark.parametrize("verdict", ["true", "false"])
def test_affordance_unsent_or_unbound_support_is_unknown(affordance_case, support, verdict):
    case = affordance_case
    case.reports["demand:goal"].update(verdict=verdict, supporting_observation_ids=support)
    result = case.verify()
    assert not result.accepted and not result.semantic_satisfied and result.decision == "need_better_view"
    judgment = result.diagnostics["affordance_judgments"]["demand:goal"]
    assert judgment["verdict"] == "unknown" and judgment["reported_verdict"] == verdict


@pytest.mark.parametrize("missing", ["path", "file", "identity", "candidate", "revision", "frame_id",
                                     "bbox", "degenerate_box", "nonfinite_box"])
def test_affordance_without_current_candidate_frame_is_unknown_without_call(affordance_case, missing):
    from copy import deepcopy

    case = affordance_case
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "earlier"},
        "geometry": deepcopy(case.evidence["geometry"]),
        "image_paths": {"current_rgb_with_bbox_path": str(case.current)}}}
    case.task["targets"][0]["attributes"] = {"goal_reference_image": str(case.current)}
    if missing == "path":
        case.evidence.pop("current_rgb_with_bbox_path")
    elif missing == "file":
        case.evidence["current_rgb_with_bbox_path"] = str(case.current.with_name("missing.png"))
    elif missing == "identity":
        case.evidence.pop("observation_identity")
    elif missing == "candidate":
        case.evidence["observation_identity"]["candidate_uid"] = "other-instance"
    elif missing == "revision":
        case.evidence["observation_identity"]["task_revision"] = 3
    elif missing == "frame_id":
        case.evidence["observation_identity"]["observation_id"] = ""
    elif missing == "bbox":
        case.evidence["geometry"].pop("bbox_xyxy")
    else:
        case.evidence["geometry"]["bbox_xyxy"] = [0, 0, 0 if missing == "degenerate_box" else float("nan"), 7]
    result = case.verify()
    assert result.decision == "need_better_view" and not result.accepted
    assert result.diagnostics["sent_current_observation_ids"] == []
    assert result.diagnostics["sent_candidate_observation_ids"] == [] and result.evidence_ids == []
    assert all(j["verdict"] == "unknown" for j in result.diagnostics["affordance_judgments"].values())
    case.provider.assert_not_called()
    case.parse.assert_not_called()


@pytest.mark.parametrize("bound", [True, False])
def test_affordance_history_support_requires_sent_matching_frame(affordance_case, bound, tmp_path):
    from PIL import Image

    case = affordance_case
    earlier = tmp_path / "earlier-affordance.png"
    Image.new("RGB", (8, 8), "green").save(earlier)
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "earlier",
                                 "candidate_uid": "candidate" if bound else "unrelated"},
        "geometry": {"bbox_xyxy": [1, 1, 6, 6]},
        "image_paths": {"current_rgb_with_bbox_path": str(earlier)}}}
    case.reports["demand:goal"]["supporting_observation_ids"] = ["earlier"]
    result = case.verify()
    assert result.accepted == bound
    assert ("earlier" in result.diagnostics["sent_candidate_observation_ids"]) == bound
    assert (str(earlier) in result.diagnostics["image_sources"]) == bound


def test_affordance_visual_reason_required_and_hidden_state_remains_unknown(affordance_case):
    case = affordance_case
    case.reports["demand:goal"]["reason"] = "   "
    assert case.verify().decision == "need_better_view"
    case.reports["demand:goal"].update(verdict="unknown", reason="Operational state is not visible.")
    assert case.verify().decision == "need_better_view"
    prompt = case.parse.call_args.kwargs["messages"][0]["content"]
    assert "Visible functional structure is sufficient" in prompt
    assert "category typicality are never" in prompt
    assert "operational or softness cannot be asserted solely" in prompt


@pytest.mark.parametrize("gate", ["distance", "projection", "surface", "surface_identity", "reachability"])
def test_affordance_all_true_preserves_physical_gates_on_cache_hit(affordance_case, monkeypatch, gate):
    case = affordance_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    assert case.verify().accepted
    geometry = case.evidence["geometry"]
    if gate == "distance":
        geometry["distance_to_object"] = 2.0
    elif gate == "projection":
        geometry["projection_failed_in_final_view"] = True
    elif gate in ("surface", "surface_identity"):
        geometry.update(surface_match=gate != "surface", surface_status="unmatched current surface",
                        observation_identity={**case.evidence["observation_identity"],
                                              "observation_id": "old" if gate == "surface_identity" else "camera:4"})
    else:
        case.evidence["hard_stop_constraints"] = {"satisfied": False, "failed": [{"name": "reachability"}]}
    case.evidence["planner_infeasibility_proof"] = {"infeasible_by_geometry": True}
    result = case.verify()
    assert not result.accepted and result.semantic_satisfied and result.decision == "need_better_view"
    assert result.failed_constraints and not result.hard_constraints["satisfied"]
    assert case.parse.call_count == 1


def test_affordance_cache_fingerprint_uses_actual_response_schema(affordance_case, monkeypatch):
    from pydantic import create_model
    import instruction_adapter.verifier as module

    case = affordance_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    assert case.verify().accepted and case.verify().accepted
    assert case.parse.call_count == 1
    factory = module.affordance_verification_response_model
    monkeypatch.setattr(module, "affordance_verification_response_model", lambda ids: create_model(
        "RevisedAffordanceVerification", __base__=factory(ids), schema_revision=(int, 2)))
    assert case.verify().accepted
    assert case.parse.call_count == 2
    assert case.parse.call_args_list[0].kwargs["messages"] == case.parse.call_args_list[1].kwargs["messages"]
    assert case.verify().accepted and case.parse.call_count == 2


@pytest.fixture
def relation_case(monkeypatch, tmp_path):
    """Exercise the real relation service with only its provider replaced."""
    from PIL import Image
    import instruction_adapter.relation_verifier as module
    from prompting.schemas import ParsedRelationResult

    image = tmp_path / "pair.png"
    pixels = Image.new("RGB", (8, 4), "red")
    pixels.paste("blue", (4, 0, 8, 4))
    pixels.save(image)
    parsed = ParsedRelationResult(
        subject_identity={"verdict": "true", "reason": "The subject region shows a cup."},
        reference_identity={"verdict": "true", "reason": "The reference region shows a table."},
        relation={"verdict": "true", "reason": "The bound cup is near the bound table."}, confidence=.9)
    parse = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]))
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    provider = Mock(return_value=(client, "offline-relation-replay"))
    monkeypatch.setattr(module, "get_client_and_model", provider)
    monkeypatch.setenv("STRIVE_RELATION_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")  # The provider boundary above cannot reach a live client.
    args = {
        "subject": {"id": "cup-1", "requested_concept": "cup", "detector_label": "cup", "center": [0, 0, 0]},
        "relation": "near",
        "object_": {"id": "table-1", "requested_concept": "table", "detector_label": "table", "center": [.5, 0, 0]},
        "evidence_views": [{"id": "camera:1", "rgb_path": str(image),
                            "observed_object_uids": ["cup-1", "table-1"],
                            "object_boxes": {"cup-1": [0, 0, 3, 3], "table-1": [4, 0, 7, 3]}}],
    }
    return SimpleNamespace(service=module.DynamicRelationService(), args=args, image=image,
                           parse=parse, provider=provider, parsed=parsed)


@pytest.mark.parametrize("suffix,mime", [("png", "image/png"), ("jpg", "image/jpeg")])
def test_relation_request_sends_bound_pair_exact_image_bytes_and_mime(relation_case, suffix, mime):
    import json
    from copy import deepcopy
    from hashlib import sha256
    from PIL import Image

    case = relation_case
    path = case.image.with_suffix("." + suffix)
    if path != case.image:
        with Image.open(case.image) as pixels:
            pixels.save(path)
    case.args["evidence_views"][0]["rgb_path"] = str(path)
    before = deepcopy(case.args)
    result = case.service.verify(**case.args)
    assert result.verdict == "true" and case.parse.call_count == 1
    request = case.parse.call_args.kwargs
    assert request["model"] == "offline-relation-replay"
    content = request["messages"][1]["content"]
    payload = json.loads(content[1]["text"])
    bound = json.loads(content[2]["text"])
    assert payload["subject"]["id"] == "cup-1" and payload["object"]["id"] == "table-1"
    assert payload["relation"] == "near"
    assert payload["evidence_views"] == [bound]
    assert bound["id"] == "camera:1"
    assert bound["observed_object_uids"] == ["cup-1", "table-1"]
    assert bound["object_boxes"] == before["evidence_views"][0]["object_boxes"]
    assert [block["type"] for block in content] == ["text", "text", "text", "image_url",
                                                   "text", "image_url", "text", "image_url"]
    data_url = content[3]["image_url"]["url"]
    assert data_url.startswith(f"data:{mime};base64,")
    sent = base64.b64decode(data_url.split(",", 1)[1])
    assert sent == path.read_bytes() == Path(bound["rgb_path"]).read_bytes()
    assert bound["image_sha256"] == sha256(sent).hexdigest()
    from io import BytesIO
    for offset, role, uid in ((4, "subject", "cup-1"), (6, "reference", "table-1")):
        region = json.loads(content[offset]["text"])
        assert region in bound["regions"]
        assert (region["role"], region["instance_id"], region["source_frame_id"]) == (role, uid, "camera:1")
        assert region["source_image_sha256"] == bound["image_sha256"]
        assert region["bbox_xyxy_in_source"] == before["evidence_views"][0]["object_boxes"][uid]
        crop_url = content[offset + 1]["image_url"]["url"]
        assert crop_url.startswith("data:image/png;base64,")
        pixels = base64.b64decode(crop_url.split(",", 1)[1])
        assert pixels == Path(region["rgb_path"]).read_bytes()
        assert sha256(pixels).hexdigest() == region["image_sha256"]
        with Image.open(BytesIO(sent)) as frame, Image.open(BytesIO(pixels)) as presentation:
            from PIL import ImageDraw

            expected = frame.crop(region["bbox_xyxy_in_source"]).convert("RGB")
            metadata = region["presentation"]
            crop = presentation.crop(metadata["unscaled_crop_box_in_image"])
            assert crop.size == expected.size and crop.tobytes() == expected.tobytes()
            buffer = BytesIO()
            expected.save(buffer, format="PNG")
            assert metadata["source_crop_sha256"] == sha256(buffer.getvalue()).hexdigest()
            assert metadata["outside_crop_role_caption"] == role
            assert metadata["unscaled_crop_box_in_image"][1] == 80
            assert presentation.size == (max(360, expected.width), expected.height + 80)
            banner = Image.new("RGB", (presentation.width, 80), "white")
            title = Image.new("RGB", (120, 16), "white")
            ImageDraw.Draw(title).text((2, 2), role.upper(), fill="black")
            banner.paste(title.resize((360, 48), resample=Image.NEAREST), (0, 0))
            draw = ImageDraw.Draw(banner)
            draw.text((6, 48), "Frame camera:1", fill="black")
            draw.text((6, 62), uid, fill="black")
            assert presentation.crop((0, 0, presentation.width, 80)).tobytes() == banner.tobytes()
    assert result.evidence_view_ids == ["camera:1"]
    assert case.args == before


def test_relation_request_keeps_demand_in_bound_region_and_invalidates_cache(relation_case):
    import json
    case = relation_case
    demand = {"activity": "write", "manipulated_objects": ["notebook"],
              "required_resource": "writing surface"}
    case.args["subject"].update(requested_concept="writing surface", demand=demand)
    case.parsed.subject_identity.verdict = "unknown"
    first = case.service.verify(**case.args)
    assert first.verdict == "unknown"
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert json.loads(content[1]["text"])["subject"]["demand"] == demand
    assert json.loads(content[4]["text"])["demand"] == demand
    assert json.loads(content[6]["text"])["requested_concept"] == "table"
    assert case.service.verify(**case.args).evidence_key == first.evidence_key
    assert case.parse.call_count == 1
    case.args["subject"]["demand"] = {**demand, "required_resource": "inclined writing surface"}
    changed = case.service.verify(**case.args)
    assert changed.verdict == "unknown" and changed.evidence_key != first.evidence_key
    assert case.parse.call_count == 2


@pytest.mark.parametrize("subject", ["true", "false", "unknown"])
@pytest.mark.parametrize("reference", ["true", "false", "unknown"])
@pytest.mark.parametrize("relation", ["true", "false", "unknown"])
def test_relation_requires_both_visual_identities_before_acceptance(relation_case, subject, reference, relation):
    case = relation_case
    case.args["object_"]["detector_label"] = "cup"
    case.parsed.subject_identity.verdict = subject
    case.parsed.reference_identity.verdict = reference
    case.parsed.relation.verdict = relation
    expected = ("false" if "false" in (subject, reference) else
                relation if subject == reference == "true" else "unknown")
    edge = case.service.verify(**case.args)
    assert edge.verdict == expected
    assert edge.verified == (expected == "true")
    assert edge.need_better_view == (expected == "unknown")
    assert f"reference_identity={reference}" in edge.reason
    assert edge.identity_judgments["subject"]["verdict"] == subject
    assert edge.identity_judgments["reference"]["verdict"] == reference
    assert edge.identity_judgments["reference"]["instance_id"] == "table-1"
    assert edge.identity_judgments["reference"]["requested_concept"] == "table"
    assert edge.identity_judgments["reference"]["source_crop_sha256"]
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert "Required reference concept: table" in content[0]["text"]
    assert "Reference detector label: cup" in content[0]["text"]
    assert case.service.verify(**case.args).verdict == expected
    assert case.parse.call_count == 1


@pytest.fixture
def identity_feedback_case(relation_case, monkeypatch):
    import json
    from dataclasses import replace
    from harness.tools import query_scene
    from instruction_adapter import concept_matcher
    from instruction_adapter.constraints import ConstraintEvaluator
    from instruction_adapter.contracts import ConceptQuery, Constraint, InstructionPlan, TargetQuery
    from instruction_adapter.verifier import candidate_from_object
    from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    case = relation_case
    objects = (ObjectNodeSnapshot("cup-1", "cup", (0., 0., 0.)),
               ObjectNodeSnapshot("cup-2", "cup", (.1, 0., 0.)),
               ObjectNodeSnapshot("table-1", "table", (.5, 0., 0.)))
    frame = {**case.args["evidence_views"][0], "step": 1,
             "observed_object_uids": [obj.uid for obj in objects],
             "object_boxes": {"cup-1": [0, 0, 3, 3], "cup-2": [0, 0, 3, 3], "table-1": [4, 0, 7, 3]}}
    anchor = ConceptQuery("reference", "table", role="anchor")
    target = TargetQuery("goal", "cup")
    plan = InstructionPlan("Find the cup near the table", targets=[target], constraints=[
        Constraint("relation", subject="cup", relation="near", object="table", object_concept=anchor)], valid=True)
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects,
                                   metadata={"observation_images": [frame]})
    session = TaskSession(snapshot, plan, constraint_evaluator=ConstraintEvaluator(relation_service=case.service))

    def parse_concepts(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
        rows = {}
        for obj in payload["observed_objects"]:
            matched = obj["label"] == payload["concept"]["name"] or (
                obj["uid"] == "table-1" and payload["concept"]["name"] == "table")
            rows[obj["uid"]] = dict(matches_concept=matched, candidate_eligible=matched,
                                   terminal_eligible=False, confidence=.9, reason="Scripted label judgment.")
        parsed = kwargs["response_format"].model_validate(rows)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse_concepts))))
    monkeypatch.setattr(concept_matcher, "get_client_and_model", lambda _: (client, "offline-concept-replay"))

    def observe(*frames, objects=objects):
        session.snapshot = replace(session.snapshot, objects=objects, metadata={"observation_images": list(frames)})
        return query_scene(session)

    def evaluate(uid="cup-1"):
        obj = session.snapshot.object_by_uid(uid)
        return session.constraint_evaluator.evaluate_before_final_verifier(
            mapper=session, plan=session.plan, target=session.plan.targets[0], candidate=candidate_from_object(obj),
            candidate_obj=obj, evidence={}, step=2)

    def match(concept=anchor, obj=objects[-1], method="match_many"):
        kwargs = dict(raw_instruction=session.plan.raw_instruction, concept=concept)
        if method == "match_many":
            return session.concept_matcher.match_many(**kwargs, objects=[obj])[0]
        return session.concept_matcher.match_object(**kwargs, obj=obj)

    observe(frame)
    return SimpleNamespace(case=case, session=session, objects=objects, frame=frame, anchor=anchor,
                           target=target, observe=observe, evaluate=evaluate, match=match)


@pytest.mark.parametrize("method", ["match_object", "match_many"])
@pytest.mark.parametrize("label", ["table", "cabinet"])
def test_identity_feedback_overrides_label_cache_without_poisoning_other_instances(identity_feedback_case, method, label):
    from copy import deepcopy
    from dataclasses import replace

    f = identity_feedback_case
    obj = replace(f.objects[-1], label=label)
    objects = (*f.objects[:2], obj)
    f.observe(f.frame, objects=objects)
    assert f.match(obj=obj).matches_concept is True
    before = deepcopy(f.session.progress.as_dict()), deepcopy(f.session.snapshot)
    f.case.parsed.reference_identity.verdict = "false"
    f.case.parsed.relation.verdict = "unknown"
    assert f.evaluate().decision == "reject_relation"
    row = f.match(obj=obj, method=method)
    assert row.matches_concept is False and not row.candidate_eligible and not row.terminal_eligible
    assert row.source == "relation_identity"
    assert f.evaluate("cup-2").decision == "need_relation_check"
    assert f.case.parse.call_count == 1
    assert f.match(obj=replace(obj, uid="table-other")).matches_concept is True
    assert f.match(concept=f.target.concept_query(), obj=f.objects[0]).candidate_eligible
    assert (f.session.progress.as_dict(), f.session.snapshot) == before
    result = f.observe({**f.frame, "id": "camera:2", "step": 2}, objects=(obj,))
    assert not result.candidates
    assert not f.session.verification_ledger.is_hard_rejected(f.session.plan.raw_instruction, "cup-1")
    assert f.case.parse.call_count == 1


@pytest.mark.parametrize("change", ["frame_id", "subject_pixels", "reference_pixels", "reference_box", "same_path_pixels"])
def test_identity_feedback_reopens_only_changed_instance_pixels(identity_feedback_case, tmp_path, change):
    from PIL import Image

    f = identity_feedback_case
    f.case.parsed.reference_identity.verdict = "false"
    f.evaluate()
    frame = {**f.frame, "id": "camera:2", "step": 2}
    path = f.case.image if change == "same_path_pixels" else tmp_path / "changed.png"
    with Image.open(f.case.image) as original:
        pixels = original.copy()
    if change == "subject_pixels":
        pixels.paste("green", (0, 0, 4, 4))
    elif change in {"reference_pixels", "same_path_pixels"}:
        pixels.paste("green", (4, 0, 8, 4))
    elif change == "reference_box":
        frame["object_boxes"] = {**frame["object_boxes"], "table-1": [0, 0, 3, 3]}
    pixels.save(path)
    frame["rgb_path"] = str(path)
    f.observe(frame)
    row = f.match()
    changed = change in {"reference_pixels", "reference_box", "same_path_pixels"}
    assert row.matches_concept is (None if changed else False)
    assert row.candidate_eligible is changed and not row.terminal_eligible
    f.case.parsed.reference_identity.verdict = "true"
    result = f.evaluate("cup-2")
    assert result.satisfied is changed
    assert f.case.parse.call_count == (2 if changed else 1)
    assert not f.session.progress.completed
    assert f.match().matches_concept is (True if changed else False)
    assert not f.match().terminal_eligible


def test_identity_feedback_does_not_recheck_old_pair_using_new_unshared_pixels(identity_feedback_case, tmp_path):
    from PIL import Image

    f = identity_feedback_case
    f.case.parsed.reference_identity.verdict = "false"
    f.evaluate()
    path = tmp_path / "reference-only.png"
    Image.new("RGB", (8, 4), "green").save(path)
    solo = {**f.frame, "id": "camera:2", "step": 2, "rgb_path": str(path),
            "observed_object_uids": ["table-1"], "object_boxes": {"table-1": [4, 0, 7, 3]}}
    f.observe(solo)
    assert f.match().candidate_eligible and f.match().matches_concept is None
    assert f.evaluate("cup-2").decision == "need_relation_check"
    assert f.case.parse.call_count == 1
    assert f.match().matches_concept is None
    f.case.parsed.reference_identity.verdict = "true"
    f.observe({**f.frame, "id": "camera:3", "step": 3, "rgb_path": str(path)})
    assert f.evaluate("cup-2").satisfied and f.case.parse.call_count == 2


@pytest.mark.parametrize("initial,conclusive", [("unknown", "false"), ("unknown", "true"), ("true", "false")])
@pytest.mark.parametrize("unshared", [False, True])
def test_same_pixels_can_gain_conclusive_identity_without_rebasing_unreviewed_views(
        identity_feedback_case, tmp_path, initial, conclusive, unshared):
    from copy import deepcopy
    from PIL import Image

    f = identity_feedback_case
    f.case.parsed.reference_identity.verdict = initial
    f.case.parsed.relation.verdict = "unknown"
    f.evaluate()
    evaluator = f.session.constraint_evaluator
    key = evaluator._identity_key(f.session.plan.raw_instruction, f.anchor, "table-1")
    previous = deepcopy(evaluator.identity_records[key])
    if unshared:
        path = tmp_path / "unreviewed-reference.png"
        Image.new("RGB", (8, 4), "green").save(path)
        f.observe({**f.frame, "id": "camera:2", "step": 2, "rgb_path": str(path),
                   "observed_object_uids": ["table-1"], "object_boxes": {"table-1": [4, 0, 7, 3]}})
    f.case.parsed.reference_identity.verdict = conclusive
    f.evaluate("cup-2")
    assert f.case.parse.call_count == 2
    stored = evaluator.identity_records[key]
    assert stored["verdict"] == conclusive
    assert stored["observed_crop_sha256"] == previous["observed_crop_sha256"]
    assert stored["source_crop_sha256"] == previous["source_crop_sha256"]
    assert f.match().matches_concept is (None if unshared else conclusive == "true")
    f.evaluate()
    assert f.case.parse.call_count == 2
    assert evaluator.identity_records[key]["verdict"] == conclusive


@pytest.mark.parametrize("field", ["id", "name", "demand", "raw_instruction", "task_revision", "detector_terms"])
def test_identity_feedback_semantic_scope_and_task_revision(identity_feedback_case, field):
    from dataclasses import replace
    from instruction_adapter.task_compat import runtime_plan

    f = identity_feedback_case
    assert f.match().matches_concept is True
    f.case.parsed.reference_identity.verdict = "false"
    f.evaluate()
    assert f.match().matches_concept is False
    concept, raw = f.anchor, f.session.plan.raw_instruction
    if field == "task_revision":
        f.session.install_plan(runtime_plan(replace(f.session.spec, revision=f.session.spec.revision + 1)))
        assert not f.session.constraint_evaluator.identity_records
        assert not f.session.constraint_evaluator.relation_service.as_dict()
        assert f.match().matches_concept is True
        return
    if field == "raw_instruction":
        raw += " instead"
    else:
        value = {"id": "other", "name": "desk", "demand": {"activity": "write"},
                 "detector_terms": ["cabinet"]}[field]
        concept = replace(concept, **{field: value})
    row = f.session.constraint_evaluator.identity_match(
        raw_instruction=raw, concept=concept, object_uid="table-1", observed_label="table")
    if field == "detector_terms":
        assert row.matches_concept is False and f.match(concept=concept).matches_concept is False
    else:
        assert row is None


@pytest.mark.parametrize("subject,reference,relation", [
    ("true", "true", "false"), ("unknown", "unknown", "unknown"), ("false", "true", "unknown")])
def test_relation_rejection_only_affects_explicitly_false_identity(identity_feedback_case, subject, reference, relation):
    f = identity_feedback_case
    f.case.parsed.subject_identity.verdict = subject
    f.case.parsed.reference_identity.verdict = reference
    f.case.parsed.relation.verdict = relation
    f.evaluate()
    assert f.match().candidate_eligible
    row = f.match(concept=f.target.concept_query(), obj=f.objects[0])
    assert row.candidate_eligible is (subject != "false")
    assert row.matches_concept is {"true": True, "false": False, "unknown": None}[subject]
    f.session.progress.mode = "better_view_for_verified_pair"
    f.session.progress.pending_verified_pair = {"candidate_uid": "cup-1", "target_id": "goal"}
    result = f.observe(f.frame)
    assert (result.selected.obj.uid == "cup-1") is (subject != "false")
    assert not f.session.progress.completed


@pytest.mark.parametrize("missing", ["subject_identity", "reference_identity", "relation"])
def test_relation_schema_requires_each_visual_judgment(relation_case, missing):
    from pydantic import ValidationError
    from prompting.schemas import ParsedRelationResult

    payload = relation_case.parsed.model_dump()
    payload.pop(missing)
    with pytest.raises(ValidationError):
        ParsedRelationResult(**payload)


@pytest.mark.parametrize("missing", [
    "views", "image_ref", "image_file", "frame_id", "observed_uids",
    "subject_box", "anchor_box", "short_box", "foreign_uid", "split_frames",
])
def test_relation_missing_or_unbound_image_is_unknown_without_model_call(relation_case, missing):
    from copy import deepcopy

    case = relation_case
    view = case.args["evidence_views"][0]
    if missing == "views":
        case.args["evidence_views"] = []
    elif missing == "image_ref":
        view.pop("rgb_path")
    elif missing == "image_file":
        view["rgb_path"] = str(case.image.with_name("missing.png"))
    elif missing == "frame_id":
        view.pop("id")
    elif missing == "observed_uids":
        view["observed_object_uids"] = []
    elif missing in ("subject_box", "anchor_box"):
        view["object_boxes"].pop("cup-1" if missing == "subject_box" else "table-1")
    elif missing == "short_box":
        view["object_boxes"]["table-1"] = [4, 0, 7]
    elif missing == "foreign_uid":
        case.args["subject"]["id"] = "cup-2"
    else:
        other = deepcopy(view)
        view["observed_object_uids"] = ["cup-1"]
        view["object_boxes"].pop("table-1")
        other.update(id="camera:2", observed_object_uids=["table-1"])
        other["object_boxes"].pop("cup-1")
        case.args["evidence_views"].append(other)
    edge = case.service.verify(**case.args)
    assert edge.verdict == "unknown" and edge.need_better_view and not edge.verified
    case.provider.assert_not_called()
    case.parse.assert_not_called()


@pytest.mark.parametrize("missing", ["image_file", "anchor_binding"])
def test_relation_unbound_image_cannot_become_negative_geometry_evidence(relation_case, missing):
    case = relation_case
    case.args["object_"]["center"] = [10, 0, 0]
    view = case.args["evidence_views"][0]
    if missing == "image_file":
        view["rgb_path"] = str(case.image.with_name("missing.png"))
    else:
        view["observed_object_uids"] = ["cup-1"]
        view["object_boxes"].pop("table-1")
    edge = case.service.verify(**case.args)
    assert edge.verdict == "unknown" and edge.need_better_view
    case.provider.assert_not_called()
    case.parse.assert_not_called()


@pytest.mark.parametrize("boundary", ["client", "parse"])
def test_relation_provider_error_propagates_without_caching_false(relation_case, boundary):
    case = relation_case
    failure = RuntimeError("relation provider unavailable")
    failing = case.provider if boundary == "client" else case.parse
    failing.side_effect = failure
    with pytest.raises(RuntimeError) as raised:
        case.service.verify(**case.args)
    assert raised.value is failure
    assert case.service.as_dict() == {}
    failing.side_effect = None
    assert case.service.verify(**case.args).verdict == "true"
    assert failing.call_count == 2


@pytest.mark.parametrize("change", ["frame", "image_bytes", "subject_geometry", "anchor_geometry",
                                    "subject_box", "anchor_box", "subject_concept", "reference_concept"])
def test_relation_cache_rechecks_changed_evidence_and_can_reverse_true(relation_case, change):
    from copy import deepcopy
    from PIL import Image

    case = relation_case
    first = case.service.verify(**case.args)
    repeated = case.service.verify(**deepcopy(case.args))
    assert first.verdict == repeated.verdict == "true"
    assert case.parse.call_count == 1
    view = case.args["evidence_views"][0]
    if change == "frame":
        view["id"] = "camera:2"
    elif change == "image_bytes":
        Image.new("RGB", (8, 4), "green").save(case.image)
    elif change.endswith("geometry"):
        case.args["subject" if change == "subject_geometry" else "object_"]["center"][0] += .1
    elif change.endswith("concept"):
        case.args["subject" if change == "subject_concept" else "object_"]["requested_concept"] = "storage cabinet"
    else:
        view["object_boxes"]["cup-1" if change == "subject_box" else "table-1"][0] += 1
    case.parsed.relation.verdict = "false"
    case.parsed.relation.reason = "The new evidence disproves the relation."
    changed = case.service.verify(**case.args)
    assert changed.verdict == "false" and not changed.need_better_view
    assert case.parse.call_count == 2
    assert first.verdict == "true"  # Historical evidence is not rewritten.
    assert case.service.verify(**deepcopy(case.args)).verdict == "false"
    assert case.parse.call_count == 2
