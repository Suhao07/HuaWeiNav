"""The archived evidence is exactly the byte sequence sent to the model."""
import base64
from pathlib import Path
from types import SimpleNamespace

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
