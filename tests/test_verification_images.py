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
