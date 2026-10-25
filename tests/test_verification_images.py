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
        parsed = SimpleNamespace(decision="accept", satisfied=True, semantic_satisfied=True,
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
