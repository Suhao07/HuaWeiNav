"""Image provenance and scripted verifier contracts, not real VLM image ability."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verification.images import image_input, image_dependencies_valid
from verification.contracts import CandidateInstance
from verification.store import EvidenceStore


def _bound_evidence(path, uid="u", revision=1):
    return {"current_rgb_with_bbox_path": str(path),
            "observation_identity": {"candidate_uid": uid, "task_revision": revision,
                                     "observation_id": "current"},
            "geometry": {"bbox_xyxy": [1, 2, 3, 4]}}


def _stage_requests(requests, trace):
    """Select actual provider calls, asserting both stages disable parse retries."""
    assert all(request["trace_label"] in {"instance_observer", "final_instruction_verifier"}
               and request["parse_retries"] == 0 for request in requests)
    return [request for request in requests if request["trace_label"] == trace]


def _assert_stages(requests, stages):
    _stage_requests(requests, "instance_observer")
    assert [request["trace_label"] for request in requests] == stages


def _observer_reply(request, facts=None, comparison=None):
    """Script facts for the sent observation IDs, never infer them from task labels."""
    assert request["trace_label"] == "instance_observer"
    facts = facts if facts is not None else {
        "observed_category": "chair", "visible_parts": ["seat", "back", "legs"],
        "support_and_attachment": "Supported by legs.", "visible_surface_content": "",
        "visible_context": [], "uncertainties": []}
    metadata = [json.loads(block["text"]) for block in request["messages"][1]["content"]
                if block["type"] == "text"]
    observations = {item["observation_identity"]["observation_id"]: dict(facts)
                    for item in metadata if "observation_identity" in item}
    response = {"observations": observations}
    if any(item["role"] == "goal_reference" for item in metadata):
        response["reference_comparison"] = comparison if comparison is not None else {
            "verdict": "unknown", "reason": "Scripted unresolved instance comparison.",
            "supporting_observation_ids": []}
    parsed = request["response_format"](**response)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


def _visual_reply(request, verdict="unknown"):
    """Instantiate each actual request schema with scripted, current-bound reports."""
    if request["trace_label"] == "instance_observer":
        return _observer_reply(request)
    assert request["trace_label"] == "final_instruction_verifier"
    content = request["messages"][1]["content"]
    assert all(item["type"] == "text" for item in content)
    payload = json.loads(content[0]["text"])
    observations = json.loads(content[1]["text"])["bound_observations"]
    current = next(uid for uid, record in observations.items() if record["current"])
    parsed = request["response_format"](
        observed_category="chair", observed_structure="A seat and back supported by legs.", confidence=1.0,
        predicate_judgments={p["id"]: {"verdict": verdict, "reason": "Scripted instance observation.",
                                      "supporting_observation_ids": [current]}
                             for p in payload["verification_predicates"]})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])


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
        return _visual_reply(kwargs, "true")

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test-model"))
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find a chair", {}, CandidateInstance("chair-1", "chair"),
        _bound_evidence(image_path, "chair-1"),
    )
    assert result.accepted
    assert result.target_uid == "chair-1"
    archived = Path(result.evidence_ids[0])
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"])
    observer = _stage_requests(requests, "instance_observer")[0]
    sent = next(item for item in observer["messages"][1]["content"] if item["type"] == "image_url")
    assert base64.b64decode(sent["image_url"]["url"].split(",", 1)[1]) == archived.read_bytes()
    store = EvidenceStore(1)
    store.put("chair", "chair-1", result)
    assert store.get("chair", "chair-1").status == "accepted"
    archived.write_bytes(b"changed archive")
    assert store.get("chair", "chair-1") is None


def test_each_image_uses_its_own_observed_box_and_identity(tmp_path, monkeypatch):
    import json
    import instruction_adapter.verifier as module
    current, earlier, unrelated = (tmp_path / name for name in ("current.jpg", "earlier.jpg", "unrelated.jpg"))
    for path in (current, earlier, unrelated):
        path.write_bytes(path.name.encode())
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return _visual_reply(kwargs)
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
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"])
    content = _stage_requests(requests, "instance_observer")[0]["messages"][1]["content"]
    texts = [json.loads(item["text"]) for item in content if item["type"] == "text"]
    final = _stage_requests(requests, "final_instruction_verifier")[0]["messages"][1]["content"]
    assert json.loads(final[0]["text"])["candidate"] == {"uid": "u"}
    assert texts[0]["bbox_xyxy_in_this_image"] == [1,2,3,4]
    assert texts[1]["bbox_xyxy_in_this_image"] == [5,6,7,8]
    assert texts[1]["observation_identity"]["observation_id"] == "earlier"
    records = json.loads(final[1]["text"])["bound_observations"]
    assert set(records) == {"current", "earlier"}
    for text in texts:
        record = records[text["observation_identity"]["observation_id"]]
        assert record["observation_identity"] == text["observation_identity"]
        assert record["images"] == [{"image_ref": text["image_ref"],
                                     "image_sha256": text["image_sha256"],
                                     "bbox_xyxy": text["bbox_xyxy_in_this_image"]}]
    assert records["current"]["current"] is True and records["earlier"]["current"] is False
    assert sum(item["type"] == "image_url" for item in content) == 2
    assert len(result.diagnostics["excluded_history"]) == 1
    assert all("distance_to_object" not in str(text) and "nearby_objects" not in str(text) for text in texts)


def test_overlapping_instance_view_is_sent_without_its_map_label(tmp_path, monkeypatch):
    import json
    import instruction_adapter.verifier as module
    current, overlap = tmp_path / "current.jpg", tmp_path / "overlap.jpg"
    current.write_bytes(b"current")
    overlap.write_bytes(b"front view")
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        return _visual_reply(kwargs, "false")
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "get_client_and_model", lambda vlm: (client, "test"))
    result = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs")).verify(
        "find sofa", {}, CandidateInstance("u", "sofa"),
        {**_bound_evidence(current), "overlapping_instance_observations": [
            {"image_path": str(overlap), "bbox_xyxy": [1, 2, 3, 4], "candidate_surface_fraction": .9}]})
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"])
    content = _stage_requests(requests, "instance_observer")[0]["messages"][1]["content"]
    texts = [json.loads(item["text"]) for item in content if item["type"] == "text"]
    assert texts[-1] == {"role": "overlapping_instance_observation",
                         "candidate_surface_fraction_on_this_instance": .9,
                         "bbox_xyxy_in_this_image": [1, 2, 3, 4],
                         "image_ref": result.diagnostics["image_sources"][str(overlap)],
                         "image_sha256": result.dependencies[
                             "image:" + result.diagnostics["image_sources"][str(overlap)]]}
    assert "sofa" not in json.dumps(content)
    assert content[-1]["type"] == "image_url" and len(result.evidence_ids) == 2


def test_exact_visual_cache_rechecks_current_physical_stop(tmp_path, monkeypatch):
    from PIL import Image
    import instruction_adapter.verifier as module
    image = tmp_path / 'current.png'
    Image.new("RGB", (8, 8), "red").save(image)
    scripted_categories = {image.read_bytes(): "chair"}
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        if kwargs["trace_label"] == "instance_observer":
            # Scripted replay keyed by actual sent bytes, not a claim of image understanding.
            block = next(item for item in kwargs["messages"][1]["content"] if item["type"] == "image_url")
            sent = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
            category = scripted_categories[sent]
            return _observer_reply(kwargs, {"observed_category": category, "visible_parts": ["seat"],
                "support_and_attachment": "Legs.", "visible_surface_content": "",
                "visible_context": [], "uncertainties": []})
        return _visual_reply(kwargs, "true")
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(module, 'get_client_and_model', lambda _: (client,'test'))
    monkeypatch.setenv('LLM_OFFLINE','0')
    monkeypatch.setenv('STRIVE_FINAL_VERIFIER','on')
    monkeypatch.setenv('STRIVE_EVIDENCE_REUSE','1')
    verifier = module.FinalInstructionVerifier(evidence_dir=str(tmp_path/'blobs'))
    evidence = {**_bound_evidence(image), 'hard_stop_constraints': {'satisfied': True}}
    candidate = CandidateInstance('u','chair')
    first = verifier.verify('chair',{},candidate,evidence)
    evidence['hard_stop_constraints'] = {'satisfied':False,'failed':[{'name':'distance'}]}
    second = verifier.verify('chair',{},candidate,evidence)
    assert first.accepted and not second.accepted
    _assert_stages(calls, ["instance_observer", "final_instruction_verifier"])
    assert first.evidence_ids == second.evidence_ids
    assert first.observed_category == second.observed_category == "chair"
    evidence['hard_stop_constraints'] = {'satisfied': True}
    assert verifier.verify('chair', {}, candidate, evidence).accepted
    _assert_stages(calls, ["instance_observer", "final_instruction_verifier"])
    Image.new("RGB", (8, 8), "green").save(image)
    scripted_categories[image.read_bytes()] = "stool"
    changed = verifier.verify('chair', {}, candidate, evidence)
    assert changed.accepted and changed.observed_category == "stool"
    assert first.observed_category == "chair" and changed.evidence_ids != first.evidence_ids
    _assert_stages(calls, ["instance_observer", "final_instruction_verifier"] * 2)
    reworded = verifier.verify('red chair', {}, candidate, evidence)
    assert reworded.observed_category == "stool"
    _assert_stages(calls, ["instance_observer", "final_instruction_verifier"] * 2 + ["final_instruction_verifier"])
    assert changed.diagnostics["instance_observation_key"] == reworded.diagnostics["instance_observation_key"]
    assert len(verifier.evidence_store.instance_observations) == 2
    assert len(verifier.evidence_store.visual_judgments) == 3


def test_goal_reference_is_sent_separately_and_recorded_as_image_dependency(tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    from PIL import Image
    import instruction_adapter.verifier as module

    current, reference = tmp_path / "current.png", tmp_path / "goal.png"
    Image.new("RGB", (2, 2), "red").save(current)
    Image.new("RGB", (2, 2), "blue").save(reference)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return _visual_reply(kwargs)

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
        {**_bound_evidence(current), "target_id": "goal", "observation_identity": identity})
    from prompting.templates import INSTANCE_OBSERVER_PROMPT

    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"])
    observer = _stage_requests(requests, "instance_observer")[0]
    content = observer["messages"][1]["content"]
    assert len(content) == 4
    assert json.loads(content[0]["text"])["role"] == "current_candidate_observation"
    assert json.loads(content[0]["text"])["observation_identity"] == identity
    goal_role = json.loads(content[2]["text"])
    assert goal_role["role"] == "goal_reference"
    assert "instruction" not in goal_role and "observation_identity" not in goal_role
    assert observer["messages"][0]["content"] == INSTANCE_OBSERVER_PROMPT
    assert "compare physical instance identity" in INSTANCE_OBSERVER_PROMPT
    assert "not merely category similarity" in INSTANCE_OBSERVER_PROMPT
    assert "reference is not a candidate observation" in INSTANCE_OBSERVER_PROMPT
    final = _stage_requests(requests, "final_instruction_verifier")[0]["messages"][1]["content"]
    assert all(item["type"] == "text" for item in final)
    assert set(json.loads(final[1]["text"])["bound_observations"]) == {"current"}
    for source, block in ((current, content[1]), (reference, content[3])):
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

    current, reference = tmp_path / "current.png", tmp_path / "goal.png"
    Image.new("RGB", (2, 2), "red").save(current)
    Image.new("RGB", (2, 2), "blue").save(reference)
    requests = []

    def parse(**kwargs):
        requests.append(kwargs)
        return _visual_reply(kwargs)

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "test"))
    verifier = module.FinalInstructionVerifier(evidence_dir=str(tmp_path / "blobs"))
    attributes = {"goal_reference_image": str(reference)}
    task = {"raw_instruction": "find the pictured chair", "revision": 1, "constraints": [],
            "targets": [{"id": "goal", "name": "chair", "attributes": attributes}]}
    plan = {"diagnostics": {"task_spec": task}}
    candidate = CandidateInstance("u", "chair")
    evidence = _bound_evidence(current)
    first = verifier.verify("find the pictured chair", plan, candidate, evidence)
    repeat = verifier.verify("find the pictured chair", plan, candidate, evidence)
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"])
    assert repeat.dependencies == first.dependencies
    Image.new("RGB", (2, 2), "green").save(reference)
    changed = verifier.verify("find the pictured chair", plan, candidate, evidence)
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"] * 2)
    assert changed.dependencies != first.dependencies
    assert changed.diagnostics["instance_observation_key"] != first.diagnostics["instance_observation_key"]
    task["revision"] = 2
    stale = verifier.verify("find the pictured chair", plan, candidate, evidence)
    assert not stale.accepted and stale.diagnostics["source"] == "missing_candidate_image"
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"] * 2)
    evidence["observation_identity"]["task_revision"] = 2
    revised = verifier.verify("find the pictured chair", plan, candidate, evidence)
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"] * 3)
    assert revised.task_revision == 2
    observers = _stage_requests(requests, "instance_observer")
    assert observers[1]["messages"][1]["content"][1:] == observers[2]["messages"][1]["content"][1:]
    assert revised.diagnostics["instance_observation_key"] != changed.diagnostics["instance_observation_key"]
    attributes.pop("goal_reference_image")
    absent = verifier.verify("find the pictured chair", plan, candidate, evidence)
    _assert_stages(requests, ["instance_observer", "final_instruction_verifier"] * 4)
    observer = _stage_requests(requests, "instance_observer")[-1]
    content = observer["messages"][1]["content"]
    assert len(content) == 2
    assert json.loads(content[0]["text"])["role"] == "current_candidate_observation"
    assert all(json.loads(item["text"])["role"] != "goal_reference"
               for item in content if item["type"] == "text")
    assert "reference_comparison" not in observer["response_format"].model_fields
    assert absent.evidence_ids == [absent.diagnostics["image_sources"][str(current)]]
    assert {key for key in absent.dependencies if key.startswith("image:")} == {
        "image:" + absent.evidence_ids[0]}


def test_visual_schema_requires_exact_complete_predicate_judgments():
    from pydantic import ValidationError
    from prompting.schemas import ParsedVerification, visual_verification_response_model

    ids = ["candidate_identity", "attribute:goal:color", "demand:goal", "design:goal"]
    model = visual_verification_response_model(ids)
    judgment = {"verdict": "unknown", "reason": "Occluded resource.", "supporting_observation_ids": []}
    complete = {uid: dict(judgment) for uid in ids}
    parsed = model(predicate_judgments=complete)
    assert parsed.predicate_judgments.model_dump() == complete
    schema = model.model_json_schema()
    assert schema["required"] == ["predicate_judgments"]
    assert schema["additionalProperties"] is False
    nested = schema["$defs"]["VisualPredicateJudgments"]
    assert set(nested["required"]) == set(complete)
    assert nested["additionalProperties"] is False
    assert set(schema["$defs"]["VisualPredicateJudgment"]["required"]) == set(judgment)
    assert set(ParsedVerification.model_fields) == {
        "observed_category", "observed_structure", "confidence", "view_feedback",
        "preferred_view_goal", "view_objective", "reason"}
    for missing in ids:
        with pytest.raises(ValidationError):
            model(predicate_judgments={uid: value for uid, value in complete.items() if uid != missing})
    for incomplete in ({}, {**complete, "invented": judgment}):
        with pytest.raises(ValidationError):
            model(predicate_judgments=incomplete)
    for invalid in ({**judgment, "verdict": True}, {**judgment, "verdict": "maybe"},
                    {**judgment, "reason": ""}, {**judgment, "reason": "x" * 241},
                    {**judgment, "decision": "accept"},
                    {"verdict": "true", "reason": "Visible structure."}):
        with pytest.raises(ValidationError):
            model(predicate_judgments={**complete, "demand:goal": invalid})
    with pytest.raises(ValidationError):
        model()
    for legacy in ({"decision": "accept"}, {"satisfied": True}, {"semantic_satisfied": True},
                   {"view_sufficient_for_stop": True}, {"affordance_judgments": complete}):
        with pytest.raises(ValidationError):
            model(predicate_judgments=complete, **legacy)
    for confidence in (-.1, 1.1):
        with pytest.raises(ValidationError):
            model(predicate_judgments=complete, confidence=confidence)
    for invalid_ids in ([], [None], [""], ["duplicate", "duplicate"]):
        with pytest.raises(ValueError):
            visual_verification_response_model(invalid_ids)


@pytest.fixture
def visual_case(monkeypatch, tmp_path):
    """Script reports at the provider boundary; placeholder pixels prove no VLM ability."""
    from PIL import Image
    import instruction_adapter.verifier as module

    current = tmp_path / "affordance.png"
    Image.new("RGB", (8, 8), "blue").save(current)
    demand = {"activity": "work with a notebook", "manipulated_objects": ["notebook"],
              "required_resource": "a usable work surface"}
    predicates = [{"id": uid, "type": "affordance", "subject": "goal", "value": demand,
                   "hardness": "hard", "verifier": "vlm"} for uid in ("demand:goal", "design:goal")]
    task = {"raw_instruction": "Find somewhere to work with my notebook", "revision": 4,
            "targets": [{"id": "goal", "name": "work surface", "demand": demand}],
            "constraints": predicates, "search_priors": {"destination_hypotheses": ["desk"]}}
    identity = {"candidate_uid": "candidate", "task_revision": 4, "observation_id": "camera:4"}
    evidence = {"observation_identity": identity, "current_rgb_with_bbox_path": str(current),
                "geometry": {"bbox_xyxy": [0, 0, 7, 7], "distance_to_object": .5, "required_stop_distance": 1.0}}
    reports = {uid: {"verdict": "true", "reason": "A broad exposed horizontal top supports the notebook.",
                     "supporting_observation_ids": ["camera:4"]}
               for uid in ("candidate_identity", "demand:goal", "design:goal")}
    response = {"observed_category": "desk", "observed_structure": "An exposed horizontal top on four legs.",
                "confidence": .95, "reason": "Scripted visual reports.",
                "view_objective": {"improve_goals": ["expose the candidate's usable top and support"]},
                "predicate_judgments": reports}

    facts = {"observed_category": "desk", "visible_parts": ["exposed horizontal top", "four legs"],
             "support_and_attachment": "Freestanding on the floor.",
             "visible_surface_content": "An empty top.", "visible_context": [], "uncertainties": []}
    comparison = {"verdict": "unknown", "reason": "Scripted unresolved reference comparison.",
                  "supporting_observation_ids": []}

    def parse(**kwargs):
        if kwargs["trace_label"] == "instance_observer":
            return _observer_reply(kwargs, facts, comparison)
        assert kwargs["trace_label"] == "final_instruction_verifier"
        assert all(item["type"] == "text" for item in kwargs["messages"][1]["content"])
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
                           parse=parse_mock, provider=provider, facts=facts, comparison=comparison,
                           requests=lambda: [call.kwargs for call in parse_mock.call_args_list],
                           verify=lambda: verifier.verify("Find somewhere to work with my notebook", plan, candidate, evidence))


def test_affordance_task_projection_preserves_demand_and_ids_without_search_priors(visual_case):
    from copy import deepcopy

    case = visual_case
    case.task["targets"].append({"id": "other", "name": "other resource",
                                 "demand": {"activity": "other", "required_resource": "other resource"}})
    other = {**case.task["constraints"][0], "id": "demand:other", "subject": "other"}
    soft = {**case.task["constraints"][0], "id": "optional", "hardness": "soft"}
    case.task["constraints"].extend([other, soft])
    case.task["constraints"][0]["verifier"] = "planner"
    case.evidence.update(target_id="goal", declared_attribute_constraints=[{
        **case.task["constraints"][0], "relation": "", "object": ""}])
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
    assert set(schema["$defs"]["VisualPredicateJudgments"]["required"]) == {
        "candidate_identity", "demand:goal", "design:goal"}


def test_original_instruction_reaches_only_text_reasoner_and_remains_detached(visual_case):
    import json
    from copy import deepcopy

    case = visual_case
    case.task["constraints"] = case.task["constraints"][:1]
    case.response["predicate_judgments"].pop("design:goal")
    raw = case.task["raw_instruction"]
    before = deepcopy(case.plan)
    projected = case.verifier._visual_task(case.plan, case.evidence)
    assert projected["active_target"]["demand"] == case.task["targets"][0]["demand"]
    projected["active_target"]["demand"]["manipulated_objects"].clear()
    assert case.plan == before
    assert case.verify().accepted
    observer = _stage_requests(case.requests(), "instance_observer")[0]
    assert raw not in json.dumps(observer["messages"])
    assert "raw_instruction" not in json.dumps(observer["messages"][1])
    assert "notebook" not in json.dumps(observer["messages"][1])
    final = _stage_requests(case.requests(), "final_instruction_verifier")[0]
    content = final["messages"][1]["content"]
    assert all(item["type"] == "text" for item in content)
    assert json.loads(content[0]["text"])["raw_instruction"] == raw
    assert case.plan == before


@pytest.mark.parametrize("plan_form", ["runtime", "task", "task_dict", "uncached_runtime"])
@pytest.mark.parametrize("reverse_relation", [False, True])
def test_canonical_id_name_collision_never_transfers_anchor_predicates(
        visual_case, plan_form, reverse_relation):
    from copy import deepcopy
    from instruction.compiler import compile_parsed_instruction
    from instruction_adapter.task_compat import runtime_plan
    from prompting.schemas import ParsedInstruction

    raw = "Find a red chair beside a blue chair."
    endpoints = ("chair", "red-chair") if reverse_relation else ("red-chair", "chair")
    parsed = ParsedInstruction(
        targets=[{"id": "red-chair", "name": "chair", "attributes": {"color": "red"}, "demand": None},
                 {"id": "chair", "name": "chair", "role": "anchor", "terminal": False, "demand": None}],
        constraints=[{"id": "beside", "type": "spatial", "subject": endpoints[0],
                      "relation": "beside", "object": endpoints[1], "hardness": "hard"},
                     {"id": "anchor-blue", "type": "color", "subject": "chair",
                      "value": "blue", "hardness": "hard", "verifier": "vlm"}])
    compilation = compile_parsed_instruction(raw, parsed, {"spatial", "color"}, revision=4)
    assert compilation.status == "success", compilation.diagnostics
    task = compilation.task
    if plan_form in {"runtime", "uncached_runtime"}:
        plan = runtime_plan(task)
        if plan_form == "uncached_runtime":
            del plan.diagnostics["task_spec"]
    else:
        plan = task if plan_form == "task" else task.as_dict()
    plan_dict = plan.as_dict() if hasattr(plan, "as_dict") else plan
    before = deepcopy(plan_dict)
    case = visual_case
    if plan_form == "uncached_runtime":
        case.evidence["observation_identity"]["task_revision"] = 1
    projected = case.verifier._visual_task(plan_dict, case.evidence)
    assert projected["active_target"]["id"] == "red-chair"
    assert projected["active_target"]["attributes"] == {"color": "red"}
    assert projected["visual_predicates"] == []
    assert [(p["id"], p["subject"], p["object"]) for p in projected["domain_predicates"]] == [
        ("beside", *endpoints)]
    assert [target["id"] for target in projected["referenced_targets"]] == ["chair"]
    case.parse.side_effect = lambda **kwargs: _visual_reply(kwargs, "true")
    case.verifier.verify(raw, plan, case.candidate, case.evidence)
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    final = _stage_requests(case.requests(), "final_instruction_verifier")[0]
    payload = json.loads(final["messages"][1]["content"][0]["text"])
    assert payload["task"] == projected
    assert {p["id"] for p in payload["verification_predicates"]} == {
        "candidate_identity", "attribute:red-chair:color"}
    schema = final["response_format"].model_json_schema()
    assert set(schema["$defs"]["VisualPredicateJudgments"]["required"]) == {
        "candidate_identity", "attribute:red-chair:color"}
    assert (plan.as_dict() if hasattr(plan, "as_dict") else plan) == before


def test_affordance_request_sends_current_candidate_bytes_and_records_judgments(visual_case):
    import json
    from copy import deepcopy

    case = visual_case
    before = deepcopy((case.plan, case.evidence))
    result = case.verify()
    assert result.accepted and result.task_revision == 4
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    final = _stage_requests(case.requests(), "final_instruction_verifier")[0]["messages"][1]["content"]
    assert json.loads(final[0]["text"])["task"]["active_target"]["demand"] == case.task["targets"][0]["demand"]
    assert all(item["type"] == "text" for item in final)
    content = _stage_requests(case.requests(), "instance_observer")[0]["messages"][1]["content"]
    bound = json.loads(content[0]["text"])
    assert bound["observation_identity"] == case.evidence["observation_identity"]
    assert bound["bbox_xyxy_in_this_image"] == [0, 0, 7, 7]
    record = json.loads(final[1]["text"])["bound_observations"]["camera:4"]
    assert record["observation_identity"] == bound["observation_identity"]
    assert record["facts"] == case.facts and record["current"] is True
    assert record["images"] == [{"image_ref": bound["image_ref"], "image_sha256": bound["image_sha256"],
                                 "bbox_xyxy": bound["bbox_xyxy_in_this_image"]}]
    assert result.diagnostics["bound_observations"] == {"camera:4": record}
    sent_bytes = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
    assert sent_bytes == case.current.read_bytes() == Path(result.evidence_ids[0]).read_bytes()
    assert result.diagnostics["predicate_judgments"] == case.reports
    assert result.diagnostics["sent_current_observation_ids"] == ["camera:4"]
    assert set(result.satisfied_constraints) == set(case.reports)
    assert (case.plan, case.evidence) == before


@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError])
def test_affordance_archive_failure_is_not_missing_observation(visual_case, monkeypatch, error_type):
    import instruction_adapter.verifier as module

    case = visual_case
    assert case.current.is_file()
    archive = Mock(side_effect=error_type("Evidence archive unavailable"))
    monkeypatch.setattr(module, "image_input", archive)
    with pytest.raises(error_type, match="Evidence archive unavailable"):
        case.verify()
    archive.assert_called_once()
    case.provider.assert_not_called()
    case.parse.assert_not_called()


def test_affordance_current_crop_uses_crop_coordinates(visual_case):
    import json

    case = visual_case
    case.evidence["object_crop_path"] = case.evidence.pop("current_rgb_with_bbox_path")
    case.evidence["object_crop_bbox_xyxy"] = [1, 1, 5, 5]
    case.evidence["geometry"].pop("bbox_xyxy")
    assert case.verify().accepted
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    content = _stage_requests(case.requests(), "instance_observer")[0]["messages"][1]["content"]
    assert json.loads(content[0]["text"])["bbox_xyxy_in_this_image"] == [1, 1, 5, 5]
    assert content[1]["type"] == "image_url"


def test_affordance_direct_immutable_taskspec_retains_revision(visual_case):
    from instruction.contracts import TaskSpec

    case = visual_case
    task = TaskSpec.from_dict({**case.task, "raw_instruction": "Find a work surface"})
    before = task.as_dict()
    result = case.verifier.verify("Find a work surface", task, case.candidate, case.evidence)
    assert result.accepted and result.task_revision == 4
    assert task.as_dict() == before


def test_soft_affordance_does_not_replace_required_identity_judgment(visual_case):
    case = visual_case
    for predicate in case.task["constraints"]:
        predicate["hardness"] = "soft"
    case.response["predicate_judgments"] = {"candidate_identity": case.reports["candidate_identity"]}
    result = case.verify()
    assert result.accepted
    assert set(result.diagnostics["predicate_judgments"]) == {"candidate_identity"}


@pytest.mark.parametrize("verdict,decision", [("false", "reject_candidate"), ("unknown", "need_better_view")])
def test_affordance_single_nontrue_predicate_blocks_aggregate_accept(visual_case, verdict, decision):
    case = visual_case
    case.reports["design:goal"].update(verdict=verdict, reason="Required structure is absent or occluded.")
    result = case.verify()
    assert not result.accepted and not result.semantic_satisfied and result.decision == decision
    assert result.diagnostics["predicate_judgments"]["design:goal"]["verdict"] == verdict
    assert "design:goal" not in result.satisfied_constraints
    assert ("design:goal" in result.failed_constraints) == (verdict == "false")


@pytest.mark.parametrize("support", [[], ["invented"], ["camera:4", "invented"], ["reference"], ["earlier"]])
@pytest.mark.parametrize("verdict", ["true", "false"])
def test_affordance_unsent_or_unbound_support_is_unknown(visual_case, support, verdict):
    case = visual_case
    case.reports["demand:goal"].update(verdict=verdict, supporting_observation_ids=support)
    result = case.verify()
    assert not result.accepted and not result.semantic_satisfied and result.decision == "need_better_view"
    judgment = result.diagnostics["predicate_judgments"]["demand:goal"]
    assert judgment["verdict"] == "unknown" and judgment["reported_verdict"] == verdict


@pytest.mark.parametrize("missing", ["path", "file", "identity", "candidate", "revision", "frame_id",
                                     "bbox", "degenerate_box", "nonfinite_box"])
def test_affordance_without_current_candidate_frame_is_unknown_without_call(visual_case, missing):
    from copy import deepcopy

    case = visual_case
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
    if missing == "file":
        with pytest.raises(FileNotFoundError):
            case.verify()
        case.provider.assert_not_called()
        return
    result = case.verify()
    assert result.decision == "need_better_view" and not result.accepted
    assert result.diagnostics["sent_current_observation_ids"] == []
    assert result.diagnostics["sent_candidate_observation_ids"] == [] and result.evidence_ids == []
    assert all(j["verdict"] == "unknown" for j in result.diagnostics["predicate_judgments"].values())
    case.provider.assert_not_called()
    case.parse.assert_not_called()


@pytest.mark.parametrize("bound", [True, False])
def test_affordance_history_support_requires_sent_matching_frame(visual_case, bound, tmp_path):
    from PIL import Image

    case = visual_case
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


def test_identity_cannot_be_established_by_history_without_current_support(visual_case, tmp_path):
    from PIL import Image

    case = visual_case
    earlier = tmp_path / "identity-history.png"
    Image.new("RGB", (8, 8), "green").save(earlier)
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "earlier"},
        "geometry": {"bbox_xyxy": [0, 0, 7, 7]},
        "image_paths": {"current_rgb_with_bbox_path": str(earlier)}}}
    case.reports["candidate_identity"]["supporting_observation_ids"] = ["earlier"]
    result = case.verify()
    assert not result.accepted
    assert result.diagnostics["predicate_judgments"]["candidate_identity"]["verdict"] == "unknown"
    assert set(result.diagnostics["sent_candidate_observation_ids"]) == {"earlier", "camera:4"}


@pytest.mark.parametrize("raw,category,structure", [
    ("Find a printed photo standing nearby.", "wall art", "The framed image hangs above the bed, without a stand."),
    ("Find smiling faces in a wall-mounted glass display.", "tabletop mirror", "A round mirror stands on the tabletop."),
    ("Find sculptural glass for blooms by the TV.", "window", "Glazing fills an opening in the wall, without a vessel cavity."),
])
def test_scripted_counterevidence_rejects_without_rewriting_original_requirements(visual_case, raw, category, structure):
    import json

    case = visual_case
    case.task["raw_instruction"] = raw
    case.task["targets"][0] = {"id": "goal", "name": "functional resource"}
    case.task["constraints"] = []
    case.facts.update(observed_category=category, visible_parts=[],
                      support_and_attachment=structure, visible_surface_content="")
    case.response.update(observed_category=category, observed_structure=structure,
        predicate_judgments={"candidate_identity": {"verdict": "false", "reason": structure,
                                                   "supporting_observation_ids": ["camera:4"]}})
    result = case.verifier.verify(raw, case.plan, case.candidate, case.evidence)
    assert result.decision == "reject_candidate" and not result.accepted
    assert result.observed_category == category != case.task["targets"][0]["name"]
    payload = json.loads(case.parse.call_args.kwargs["messages"][1]["content"][0]["text"])
    assert payload["raw_instruction"] == raw
    assert case.task["raw_instruction"] == raw
    assert result.diagnostics["observed_structure"] == structure
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("selected_role", ["subject", "anchor"])
def test_final_relation_context_keeps_pair_regions_without_rejudging_relation(visual_case, bound, selected_role):
    import json

    case = visual_case
    case.task["targets"].append({"id": "anchor", "name": "lamp", "terminal": False})
    case.task["constraints"].append({"id": "relation:goal", "type": "spatial", "subject": "goal",
        "object": "anchor", "relation": "near", "hardness": "hard", "verifier": "vlm"})
    view = {"id": "pair:earlier", "rgb_path": str(case.current),
            "observed_object_uids": ["candidate", "lamp-1"],
            "object_boxes": {"candidate": [0, 0, 3, 7], "lamp-1": [4, 0, 7, 7]}}
    if not bound:
        view["object_boxes"].pop("lamp-1")
    selected = {"id": "candidate", "requested_concept": "requested-work-surface",
                "demand": {"activity": "requested-writing"}}
    other = {"id": "lamp-1", "requested_concept": "requested-lamp",
             "attributes": {"color": "requested-purple"}}
    subject, reference = (selected, other) if selected_role == "subject" else (other, selected)
    case.evidence["verified_relation_contexts"] = [{"predicate_id": "relation:goal",
        "subject_record": subject, "anchor_record": reference,
        "edge": {"relation": "near"}, "observation_images": [view]}]
    result = case.verify()
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    observer = _stage_requests(case.requests(), "instance_observer")[0]["messages"][1]["content"]
    assert all(token not in json.dumps(observer) for token in (
        "requested-", "relation:goal", "lamp-1", "requested_concept", "demand", "object_boxes"))
    earlier = [json.loads(item["text"]) for item in observer if item["type"] == "text"
               and json.loads(item["text"])["role"] == "earlier_candidate_observation"]
    assert bool(earlier) is bound
    content = _stage_requests(case.requests(), "final_instruction_verifier")[0]["messages"][1]["content"]
    assert all(item["type"] == "text" for item in content)
    texts = [json.loads(item["text"]) for item in content]
    contexts = texts[2:]
    assert bool(contexts) is bound
    assert "relation:goal" not in {p["id"] for p in texts[0]["verification_predicates"]}
    assert texts[0]["task"]["domain_predicates"][0]["id"] == "relation:goal"
    if bound:
        assert earlier[0]["bbox_xyxy_in_this_image"] == view["object_boxes"]["candidate"]
        assert earlier[0]["observation_identity"] == {
            "candidate_uid": "candidate", "task_revision": 4, "observation_id": "pair:earlier"}
        assert contexts == [{"role": "verified_relation_observation", "predicate_id": "relation:goal",
                             "subject": subject, "reference": reference, "relation": "near", "observation": view}]
        assert texts[1]["bound_observations"]["pair:earlier"]["current"] is False
        assert "pair:earlier" in result.diagnostics["sent_candidate_observation_ids"]


def test_affordance_visual_reason_required_and_hidden_state_remains_unknown(visual_case):
    case = visual_case
    case.reports["demand:goal"]["reason"] = "   "
    assert case.verify().decision == "need_better_view"
    case.reports["demand:goal"].update(verdict="unknown", reason="Operational state is not visible.")
    assert case.verify().decision == "need_better_view"
    prompt = case.parse.call_args.kwargs["messages"][0]["content"]
    assert "Material/category similarity alone is insufficient" in prompt
    assert "hidden operational state is unknown" in prompt


@pytest.mark.parametrize("gate", ["distance", "projection", "surface", "surface_identity", "reachability"])
def test_affordance_all_true_preserves_physical_gates_on_cache_hit(visual_case, monkeypatch, gate):
    case = visual_case
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
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])


def test_affordance_cache_fingerprint_uses_actual_response_schema(visual_case, monkeypatch):
    from pydantic import create_model
    import instruction_adapter.verifier as module

    case = visual_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    assert case.verify().accepted and case.verify().accepted
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    factory = module.visual_verification_response_model
    monkeypatch.setattr(module, "visual_verification_response_model", lambda ids: create_model(
        "RevisedAffordanceVerification", __base__=factory(ids), schema_revision=(int, 2)))
    assert case.verify().accepted
    expected = ["instance_observer", "final_instruction_verifier", "final_instruction_verifier"]
    _assert_stages(case.requests(), expected)
    finals = _stage_requests(case.requests(), "final_instruction_verifier")
    assert finals[0]["messages"] == finals[1]["messages"]
    assert finals[0]["response_format"].model_json_schema() != finals[1]["response_format"].model_json_schema()
    assert case.verify().accepted
    _assert_stages(case.requests(), expected)


@pytest.mark.parametrize("reuse", [False, True])
def test_scripted_observer_is_blind_to_task_labels_priors_and_execution_history(visual_case, monkeypatch, reuse):
    from copy import deepcopy
    from prompting.templates import INSTANCE_OBSERVER_PROMPT

    case = visual_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1" if reuse else "0")
    case.facts["visible_context"] = ["A stove is visible across a doorway; shared space is unclear."]
    case.task["raw_instruction"] = "private-original-requirements"
    case.task["targets"][0]["name"] = "private-target-name"
    case.plan["targets"] = [{"id": "goal", "name": "private-target-name",
                             "detector_terms": ["private-detector-term"]}]
    case.task["targets"][0]["demand"]["activity"] = "private-activity"
    case.task["search_priors"] = {"destination_hypotheses": ["private-search-hypothesis"]}
    case.evidence.update(nearby_objects=[{"label": "private-neighbor"}],
                         final_verifier_attempt={"reason": "private-execution-history"})
    before = deepcopy((case.plan, case.evidence))
    first = case.verifier.verify(case.task["raw_instruction"], case.plan, case.candidate, case.evidence)
    assert first.accepted and (case.plan, case.evidence) == before
    case.task["raw_instruction"] = "changed-original-requirements"
    case.task["targets"][0]["name"] = "changed-target-name"
    case.task["targets"][0]["demand"]["activity"] = "changed-activity"
    changed = case.verifier.verify(case.task["raw_instruction"], case.plan, case.candidate, case.evidence)
    assert changed.accepted
    assert changed.diagnostics["instance_observation_key"] == first.diagnostics["instance_observation_key"]
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"]
                   + (["final_instruction_verifier"] if reuse else ["instance_observer", "final_instruction_verifier"]))
    assert len(case.verifier.evidence_store.instance_observations) == (1 if reuse else 0)
    assert len(case.verifier.evidence_store.visual_judgments) == (2 if reuse else 0)
    observers = _stage_requests(case.requests(), "instance_observer")
    assert observers[0]["messages"] == observers[-1]["messages"]
    assert observers[0]["messages"][0]["content"] == INSTANCE_OBSERVER_PROMPT
    for observer in observers:
        serialized = json.dumps(observer["messages"])
        assert all(token not in serialized for token in (
            "private-", "changed-", "unfamiliar detector category", "raw_instruction",
            '"task"', "search_priors", "destination_hypotheses", "detector_terms"))
        metadata = json.loads(observer["messages"][1]["content"][0]["text"])
        assert set(metadata) == {"role", "image_kind", "observation_identity", "bbox_xyxy_in_this_image",
                                 "image_ref", "image_sha256"}
    finals = _stage_requests(case.requests(), "final_instruction_verifier")
    payloads = [json.loads(request["messages"][1]["content"][0]["text"]) for request in finals]
    assert [payload["raw_instruction"] for payload in payloads] == [
        "private-original-requirements", "changed-original-requirements"]
    assert payloads[0]["task"]["active_target"]["name"] == "private-target-name"
    assert payloads[1]["task"]["active_target"]["name"] == "changed-target-name"
    packets = [json.loads(request["messages"][1]["content"][1]["text"]) for request in finals]
    assert packets[0] == packets[1]
    assert packets[1]["bound_observations"]["camera:4"]["facts"] == case.facts


@pytest.mark.parametrize("image_kind,visible_context,uncertainties", [
    pytest.param("current_rgb_with_bbox_path", [
        "椅子旁有炉灶和水槽。", "三者位于同一连续地面空间，之间没有门槛或隔墙。"], [], id="same-space"),
    pytest.param("current_rgb_with_bbox_path", [
        "炉灶位于椅子身后的门外；门框隔开两片可见地面，无法确认空间归属。"],
        ["无法确认门外炉灶与椅子是否属于同一空间。"], id="across-doorway"),
    pytest.param("object_crop_path", [], ["裁剪图只显示椅子，周围空间不可见。"], id="crop-only"),
])
def test_scripted_visible_context_stays_bound_without_resolving_room(
        visual_case, tmp_path, image_kind, visible_context, uncertainties):
    """Replay evidence transport, not recognition of rooms or fixtures from pixels."""
    from copy import deepcopy
    from PIL import Image

    case = visual_case
    raw = "Find the chair in the kitchen."
    room = {"id": "room:goal", "type": "room", "subject": "goal", "relation": "", "object": "",
            "value": "kitchen", "hardness": "hard", "verifier": "planner"}
    case.task.update(raw_instruction=raw, targets=[{"id": "goal", "name": "chair"}], constraints=[room])
    case.evidence.update(declared_room_constraints=[dict(room)], room_label="predicted-room-sentinel",
                         gt_room_label="gt-room-sentinel", predicted_room_label="predicted-room-sentinel",
                         nearby_objects=[{"label": "stove"}, {"label": "sink"}],
                         visible_context=["Unobserved planner-supplied surroundings."])
    case.facts.update(observed_category="chair", visible_parts=["seat", "back", "legs"],
                      support_and_attachment="Legs on the floor.", visible_surface_content="",
                      visible_context=visible_context, uncertainties=uncertainties)
    case.response["predicate_judgments"] = {
        "candidate_identity": {"verdict": "true", "reason": "Scripted chair identity.",
                               "supporting_observation_ids": ["camera:4"]},
        "room:goal": {"verdict": "unknown", "reason": "Scripted unresolved spatial membership.",
                      "supporting_observation_ids": ["camera:4"]},
    }
    if image_kind == "object_crop_path":
        case.evidence[image_kind] = case.evidence.pop("current_rgb_with_bbox_path")
        case.evidence["object_crop_bbox_xyxy"] = [1, 1, 5, 5]
        case.evidence["geometry"].pop("bbox_xyxy")
    earlier = tmp_path / "room-history.png"
    Image.new("RGB", (8, 8), "green").save(earlier)
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "earlier"},
        "geometry": {"bbox_xyxy": [0, 0, 7, 7]}, "image_paths": {"current_rgb_with_bbox_path": str(earlier)}}}
    historical_facts = {**deepcopy(case.facts), "visible_context": ["A bed and window were visible behind the chair."],
                        "uncertainties": []}
    original_parse = case.parse.side_effect

    def parse(**kwargs):
        completion = original_parse(**kwargs)
        if kwargs["trace_label"] == "instance_observer":
            payload = completion.choices[0].message.parsed.model_dump()
            payload["observations"]["earlier"] = historical_facts
            completion.choices[0].message.parsed = kwargs["response_format"](**payload)
        return completion

    case.parse.side_effect = parse
    before = deepcopy((case.plan, case.evidence, case.facts))
    result = case.verifier.verify(raw, case.plan, case.candidate, case.evidence)
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    observer, final = case.requests()
    serialized = json.dumps(observer["messages"], ensure_ascii=False)
    assert all(token not in serialized for token in (
        raw, "kitchen", "room:goal", "predicted-room-sentinel", "gt-room-sentinel", "room_label",
        "nearby_objects", "Unobserved planner-supplied", "unfamiliar detector category", '"task"'))
    metadata = [json.loads(item["text"]) for item in observer["messages"][1]["content"] if item["type"] == "text"]
    assert metadata[0]["image_kind"] == image_kind
    final_content = final["messages"][1]["content"]
    assert all(item["type"] == "text" for item in final_content)
    task_payload = json.loads(final_content[0]["text"])
    assert task_payload["raw_instruction"] == raw
    assert task_payload["task"]["visual_predicates"] == [room]
    assert {p["id"] for p in task_payload["verification_predicates"]} == {"candidate_identity", "room:goal"}
    records = json.loads(final_content[1]["text"])["bound_observations"]
    assert set(records) == {"camera:4", "earlier"}
    for item in metadata:
        record = records[item["observation_identity"]["observation_id"]]
        assert record["observation_identity"] == item["observation_identity"]
        assert record["images"] == [{"image_ref": item["image_ref"], "image_sha256": item["image_sha256"],
                                     "bbox_xyxy": item["bbox_xyxy_in_this_image"]}]
    assert records["camera:4"]["current"] is True and records["earlier"]["current"] is False
    assert records["camera:4"]["facts"] == before[2]
    assert records["earlier"]["facts"] == historical_facts
    assert result.diagnostics["bound_observations"] == records
    assert result.observed_category == "chair"
    assert result.diagnostics["observed_structure"] == "seat, back, legs; Legs on the floor."
    assert result.diagnostics["predicate_judgments"] == case.response["predicate_judgments"]
    assert not result.accepted and not result.semantic_satisfied and result.decision == "need_better_view"
    assert result.hard_constraints["satisfied"] is True
    assert result.satisfied_constraints == ["candidate_identity"] and result.failed_constraints == []
    assert (case.plan, case.evidence, case.facts) == before


@pytest.mark.parametrize("final_category,final_structure", [
    ("invented resource", "Invented usable parts."), ("", ""),
])
def test_scripted_current_facts_cannot_be_rewritten_by_final_or_history(
        visual_case, tmp_path, monkeypatch, final_category, final_structure):
    from copy import deepcopy
    from PIL import Image

    case = visual_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    earlier = tmp_path / "contradictory-history.png"
    Image.new("RGB", (8, 8), "green").save(earlier)
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "aaa:earlier"},
        "geometry": {"bbox_xyxy": [0, 0, 7, 7]}, "image_paths": {"current_rgb_with_bbox_path": str(earlier)}}}
    case.response.update(observed_category=final_category, observed_structure=final_structure)
    original_parse = case.parse.side_effect

    def parse(**kwargs):
        completion = original_parse(**kwargs)
        if kwargs["trace_label"] == "instance_observer":
            payload = completion.choices[0].message.parsed.model_dump()
            payload["observations"]["aaa:earlier"].update(
                observed_category="historical cabinet", visible_parts=["historical doors"])
            completion.choices[0].message.parsed = kwargs["response_format"](**payload)
        return completion

    case.parse.side_effect = parse
    before = deepcopy(case.facts)
    result, repeated = case.verify(), case.verify()
    expected_structure = "exposed horizontal top, four legs; Freestanding on the floor.; An empty top."
    for value in (result, repeated):
        assert value.accepted and value.observed_category == "desk"
        assert value.diagnostics["observed_structure"] == expected_structure
        assert value.diagnostics["bound_observations"]["camera:4"]["facts"] == before
        assert value.diagnostics["bound_observations"]["aaa:earlier"]["facts"]["observed_category"] == "historical cabinet"
    assert case.facts == before
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    # Applying the authoritative current facts must not mutate the cached final response.
    cached = next(iter(case.verifier.evidence_store.visual_judgments.values()))
    assert cached.observed_category == final_category and cached.observed_structure == final_structure


@pytest.mark.parametrize("missing_fact", ["category", "structure"])
def test_scripted_final_cannot_fill_missing_current_instance_facts(visual_case, missing_fact):
    case = visual_case
    if missing_fact == "category":
        case.facts["observed_category"] = ""
    else:
        case.facts.update(visible_parts=[], support_and_attachment="", visible_surface_content="")
    case.facts["uncertainties"] = ["Current region is occluded."]
    case.facts["visible_context"] = ["A separate stove and sink are visible beside the occluded candidate."]
    result = case.verify()
    assert result.diagnostics["bound_observations"]["camera:4"]["facts"] == case.facts
    assert not result.accepted and result.decision == "need_better_view"
    assert result.diagnostics["predicate_judgments"]["candidate_identity"]["verdict"] == "unknown"
    if missing_fact == "category":
        assert not result.observed_category
    else:
        assert not result.diagnostics["observed_structure"]
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])


@pytest.mark.parametrize("comparison_verdict,expected", [("same", "true"), ("different", "false"), ("unknown", "unknown")])
@pytest.mark.parametrize("support", [["camera:4"], ["camera:4", "earlier"], ["earlier"], [],
                                    ["goal_reference"], ["camera:4", "invented"]])
def test_scripted_reference_comparison_overrides_final_and_requires_current_support(
        visual_case, tmp_path, comparison_verdict, expected, support):
    from PIL import Image

    case = visual_case
    reference, earlier = tmp_path / "goal-reference.png", tmp_path / "reference-history.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    Image.new("RGB", (8, 8), "green").save(earlier)
    case.task["targets"][0]["attributes"] = {"goal_reference_image": str(reference)}
    case.evidence["view_control"] = {"pinned_visual_evidence": {
        "observation_identity": {**case.evidence["observation_identity"], "observation_id": "earlier"},
        "geometry": {"bbox_xyxy": [0, 0, 7, 7]}, "image_paths": {"current_rgb_with_bbox_path": str(earlier)}}}
    predicate = "attribute:goal:goal_reference_image"
    case.reports[predicate] = {"verdict": "false" if comparison_verdict == "same" else "true",
                              "reason": "Scripted final disagrees with blind comparison.",
                              "supporting_observation_ids": ["camera:4"]}
    case.comparison.update(verdict=comparison_verdict, reason="Scripted blind instance comparison.",
                           supporting_observation_ids=support)
    result = case.verify()
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    valid = "camera:4" in support and set(support) <= {"camera:4", "earlier"}
    verdict = expected if valid else "unknown"
    judgment = result.diagnostics["predicate_judgments"][predicate]
    assert judgment == {"verdict": verdict, "reason": case.comparison["reason"],
                        "supporting_observation_ids": sorted(support) if valid else []}
    assert result.accepted is (verdict == "true")
    assert result.decision == {"true": "accept", "false": "reject_candidate", "unknown": "need_better_view"}[verdict]
    assert result.diagnostics["reference_comparison"] == case.comparison
    final = _stage_requests(case.requests(), "final_instruction_verifier")[0]["messages"][1]["content"]
    packet = json.loads(final[1]["text"])
    assert packet["reference_comparison"] == case.comparison
    assert set(packet["bound_observations"]) == {"camera:4", "earlier"}
    assert all(item["type"] == "text" for item in final)


@pytest.mark.parametrize("compare_reference", [False, True])
def test_observer_schema_requires_exact_observation_ids_and_complete_facts(compare_reference):
    from copy import deepcopy
    from pydantic import ValidationError
    from prompting.schemas import instance_observation_response_model

    ids = ["camera:4", "earlier"]
    facts = {"observed_category": "desk", "visible_parts": ["top"], "support_and_attachment": "Legs.",
             "visible_surface_content": "", "visible_context": [], "uncertainties": ["Rear face occluded."]}
    model = instance_observation_response_model(ids, compare_reference=compare_reference)
    payload = {"observations": {uid: dict(facts) for uid in ids}}
    comparison = {"verdict": "same", "reason": "Scripted distinctive visual match.",
                  "supporting_observation_ids": ["camera:4"]}
    if compare_reference:
        payload["reference_comparison"] = comparison
    assert model(**payload).model_dump() == payload
    schema = model.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload)
    assert set(schema["$defs"]["BoundInstanceFacts"]["required"]) == set(ids)
    assert schema["$defs"]["BoundInstanceFacts"]["additionalProperties"] is False
    assert set(schema["$defs"]["ObservedInstanceFacts"]["required"]) == set(facts)
    assert schema["$defs"]["ObservedInstanceFacts"]["additionalProperties"] is False
    context_schema = schema["$defs"]["ObservedInstanceFacts"]["properties"]["visible_context"]
    assert context_schema["type"] == "array" and context_schema["items"] == {"type": "string"}
    assert "default" not in context_schema
    for uid in ids:
        incomplete = deepcopy(payload)
        del incomplete["observations"][uid]
        with pytest.raises(ValidationError):
            model(**incomplete)
    with pytest.raises(ValidationError):
        model(**{**payload, "observations": {**payload["observations"], "invented": facts}})
    for field in facts:
        incomplete = deepcopy(payload)
        del incomplete["observations"]["camera:4"][field]
        with pytest.raises(ValidationError):
            model(**incomplete)
    for invalid in ({**facts, "observed_category": "x" * 81}, {**facts, "task_satisfied": True},
                    *({**facts, "visible_context": value} for value in (None, "stove nearby", {}, [1]))):
        with pytest.raises(ValidationError):
            model(**{**payload, "observations": {**payload["observations"], "camera:4": invalid}})
    if compare_reference:
        with pytest.raises(ValidationError):
            model(observations=payload["observations"])
        for invalid in ({**comparison, "verdict": "true"}, {**comparison, "reason": ""},
                        {"verdict": "same", "reason": "No support field."}):
            with pytest.raises(ValidationError):
                model(**{**payload, "reference_comparison": invalid})
    else:
        with pytest.raises(ValidationError):
            model(**payload, reference_comparison=comparison)


@pytest.mark.parametrize("field", ["visible_parts", "visible_context", "uncertainties"])
@pytest.mark.parametrize("length", [0, 16, 17, 140])
def test_observer_fact_lists_are_bounded_without_silent_truncation(field, length):
    from pydantic import ValidationError
    from llm_utils.self_hosted_llm_adapter import _response_json_schema
    from prompting.schemas import instance_observation_response_model

    model = instance_observation_response_model(["view"])
    facts = {"observed_category": "shelf", "visible_parts": [], "support_and_attachment": "Wall.",
             "visible_surface_content": "", "visible_context": [], "uncertainties": []}
    facts[field] = ["visible detail"] * length
    for schema in (model.model_json_schema(), _response_json_schema(model)):
        assert schema["$defs"]["ObservedInstanceFacts"]["properties"][field]["maxItems"] == 16
    if length <= 16:
        assert model(observations={"view": facts}).model_dump()["observations"]["view"] == facts
    else:
        with pytest.raises(ValidationError) as error:
            model(observations={"view": facts})
        assert error.value.errors()[0]["loc"] == ("observations", "view", field)


@pytest.mark.parametrize("stage", ["instance_observer", "final_instruction_verifier"])
@pytest.mark.parametrize("failure_kind", ["provider", "empty", "missing_entry", "extra_entry"])
def test_stage_failure_propagates_without_retry_or_invalid_cache(visual_case, monkeypatch, stage, failure_kind):
    from pydantic import ValidationError

    case = visual_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    original_parse = case.parse.side_effect
    failure = RuntimeError("scripted provider failure")

    def parse(**kwargs):
        if kwargs["trace_label"] != stage:
            return original_parse(**kwargs)
        if failure_kind == "provider":
            raise failure
        if failure_kind == "empty":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))])
        payload = original_parse(**kwargs).choices[0].message.parsed.model_dump()
        key = "observations" if stage == "instance_observer" else "predicate_judgments"
        if failure_kind == "missing_entry":
            payload[key].pop(next(iter(payload[key])))
        else:
            payload[key]["invented"] = next(iter(payload[key].values()))
        # A malformed parsed object must be revalidated, not trusted as provider output.
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=SimpleNamespace(model_dump=lambda: payload)))])

    case.parse.side_effect = parse
    error = RuntimeError if failure_kind == "provider" else ValueError if failure_kind == "empty" else ValidationError
    with pytest.raises(error) as raised:
        case.verify()
    if failure_kind == "provider":
        assert raised.value is failure
    expected = ["instance_observer"] if stage == "instance_observer" else ["instance_observer", "final_instruction_verifier"]
    _assert_stages(case.requests(), expected)
    assert not case.verifier.evidence_store.visual_judgments
    assert len(case.verifier.evidence_store.instance_observations) == (stage == "final_instruction_verifier")
    # Only an explicit caller retry can proceed; valid observer facts may then be reused.
    case.parse.side_effect = original_parse
    assert case.verify().accepted
    expected += ["instance_observer", "final_instruction_verifier"] if stage == "instance_observer" else ["final_instruction_verifier"]
    _assert_stages(case.requests(), expected)


@pytest.mark.parametrize("change", ["schema", "prompt", "candidate_uid"])
def test_blind_cache_fingerprints_actual_schema_messages_and_instance(visual_case, monkeypatch, change):
    from pydantic import create_model
    import instruction_adapter.verifier as module

    case = visual_case
    monkeypatch.setenv("STRIVE_EVIDENCE_REUSE", "1")
    first, repeated = case.verify(), case.verify()
    assert first.accepted and repeated.accepted
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"])
    if change == "schema":
        factory = module.instance_observation_response_model
        monkeypatch.setattr(module, "instance_observation_response_model", lambda ids, **kwargs: create_model(
            "RevisedInstanceObservation", __base__=factory(ids, **kwargs), schema_revision=(int, 2)))
    elif change == "prompt":
        monkeypatch.setattr(module, "INSTANCE_OBSERVER_PROMPT", module.INSTANCE_OBSERVER_PROMPT + "\nDescribe edges.")
    else:
        case.candidate.uid = "other-candidate"
        case.evidence["observation_identity"]["candidate_uid"] = "other-candidate"
    changed = case.verify()
    assert changed.accepted
    assert changed.diagnostics["instance_observation_key"] != first.diagnostics["instance_observation_key"]
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"] * 2)
    observers = _stage_requests(case.requests(), "instance_observer")
    if change == "schema":
        assert observers[0]["messages"] == observers[1]["messages"]
        assert observers[0]["response_format"].model_json_schema() != observers[1]["response_format"].model_json_schema()
    else:
        assert observers[0]["messages"] != observers[1]["messages"]
    assert case.verify().accepted
    _assert_stages(case.requests(), ["instance_observer", "final_instruction_verifier"] * 2)
    assert len(case.verifier.evidence_store.instance_observations) == 2
    assert len(case.verifier.evidence_store.visual_judgments) == 2
    case.verifier.evidence_store.reset()
    assert not case.verifier.evidence_store.instance_observations
    assert not case.verifier.evidence_store.visual_judgments


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
    assert result.dependencies == {f"image:{image['rgb_path']}": image["image_sha256"]
                                   for image in [bound, *bound["regions"]]}
    assert result.evidence_ids == sorted(set(result.dependencies.values()))
    assert case.args == before


def test_relation_archives_remain_bound_through_precheck_conversion(relation_case):
    from instruction_adapter.constraints import ConstraintEvaluation
    from verification.images import image_dependencies_valid

    case = relation_case
    edge = case.service.verify(**case.args)
    evaluation = ConstraintEvaluation(False, "need_relation_check", relation_edges=[edge],
                                      evidence={"dependencies": {"object:cup-1": "geometry"}})
    result = evaluation.to_verification_result(3, "cup-1")
    assert result.dependencies == {"object:cup-1": "geometry", **edge.dependencies}
    assert result.evidence_ids == edge.evidence_ids
    assert image_dependencies_valid(result.dependencies)
    assert case.service.verify(**case.args) is edge
    archive = Path(next(iter(edge.dependencies)).removeprefix("image:"))
    archive.unlink()
    assert not image_dependencies_valid(result.dependencies)
    assert case.service.verify(**case.args) is not edge
    assert case.parse.call_count == 2
    assert archive.is_file()


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


def test_relation_predicate_reaches_model_and_changes_cache_key(relation_case):
    case = relation_case
    predicate = {"id": "relation:1", "type": "spatial", "subject": "goal", "object": "anchor",
                 "relation": "near", "hardness": "hard", "verifier": "vlm",
                 "value": {"source_text": "within reach", "requirement": "reachable while seated"}}
    first = case.service.verify(**case.args, predicate=predicate)
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert json.dumps(predicate, ensure_ascii=False) in content[0]["text"]
    assert case.service.verify(**case.args, predicate=predicate) is first
    assert case.parse.call_count == 1
    changed = {**predicate, "value": {"source_text": "visible from bed",
                                     "requirement": "visible while lying down"}}
    second = case.service.verify(**case.args, predicate=changed)
    assert second.evidence_key != first.evidence_key
    assert case.parse.call_count == 2
    content = case.parse.call_args.kwargs["messages"][1]["content"]
    assert json.dumps(changed, ensure_ascii=False) in content[0]["text"]


@pytest.mark.parametrize("relation,visual", [("on", True), ("above", True), ("on top of", False)])
def test_relation_preserves_attachment_and_elevation_semantics(relation_case, relation, visual):
    case = relation_case
    case.args["relation"] = relation
    case.args["subject"].update(center=[0, 0, 1], min_bound=[0, 0, .5], max_bound=[1, 1, 1.5])
    case.args["object_"].update(center=[.5, 0, 2], min_bound=[0, 0, 0], max_bound=[1, 1, 4])
    result = case.service.verify(**case.args)
    assert result.relation == relation
    assert result.source == ("vlm" if visual else "geometry")
    assert case.parse.call_count == int(visual)
    if visual:
        payload = json.loads(case.parse.call_args.kwargs["messages"][1]["content"][1]["text"])
        assert payload["relation"] == relation


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
    plan = InstructionPlan("Find the cup near the table", targets=[target,
        TargetQuery("reference", "table", role="anchor", terminal=False, concept=anchor)], constraints=[
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


def test_relation_attribute_counterevidence_does_not_poison_unqualified_identity(identity_feedback_case):
    import json
    from dataclasses import replace
    from instruction.compiler import task_from_parsed_plan

    f = identity_feedback_case
    task = task_from_parsed_plan(f.session.plan)
    task = replace(task, revision=task.revision + 1,
                   targets=(task.targets[0], replace(task.targets[1], attributes={"contents": "fresh snacks"})))
    from instruction_adapter.task_compat import runtime_plan
    f.session.install_plan(runtime_plan(task))
    f.observe(f.frame)
    f.case.parsed.reference_identity.verdict = "false"
    f.case.parsed.reference_identity.reason = "The visible surface has no snack container."
    f.case.parsed.relation.verdict = "unknown"
    assert f.evaluate().decision == "reject_relation"
    content = f.case.parse.call_args.kwargs["messages"][1]["content"]
    payload = json.loads(content[1]["text"])
    assert payload["object"]["attributes"] == {"contents": "fresh snacks"}
    qualified = f.session.plan.targets[1].concept_query()
    assert f.match(concept=qualified).matches_concept is False
    assert f.match(concept=replace(qualified, attributes={})).matches_concept is True


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
    ("true", "true", "false"), ("unknown", "unknown", "unknown"), ("false", "true", "unknown"),
    ("true", "true", "true")])
def test_relation_rejection_only_affects_explicitly_false_identity(identity_feedback_case, subject, reference, relation):
    f = identity_feedback_case
    f.case.parsed.subject_identity.verdict = subject
    f.case.parsed.reference_identity.verdict = reference
    f.case.parsed.relation.verdict = relation
    result = f.evaluate()
    assert result.evidence["relation_outcomes"] == result.relation_outcomes
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
