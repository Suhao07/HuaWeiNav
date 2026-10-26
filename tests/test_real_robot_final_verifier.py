"""Robot evidence reaches the shared visual protocol with capture-owned identity."""
import base64
import json
from types import SimpleNamespace

import pytest

from instruction_adapter.verifier import FinalInstructionVerifier
from real_robot.contracts import EvidenceSource, ViewEvidence, Pose3D, NavigationIntent, MotionGoal, MotionGoalMode
from real_robot.sysnav_runtime import FinalInstructionVerifierAdapter, _verification_context_from_intent
from verification.contracts import CandidateInstance


def _context():
    return {
        "raw_instruction": "find a chair then a table",
        "target_id": "table",
        "candidate_instance": CandidateInstance("object-2", "predicted-label", bbox_2d=[90, 90, 99, 99]),
        "instruction_plan": {"diagnostics": {"task_spec": {
            "revision": 7,
            "targets": [{"id": "chair", "name": "chair"}, {"id": "table", "name": "table"}],
        }}},
        "verifier_evidence": {"geometry": {"distance_to_target": 2.0, "bbox_xyxy": [90, 90, 99, 99]},
                              "hard_stop_constraints": {"satisfied": False, "failed": []}},
    }


def _evidence(path):
    return ViewEvidence(source=EvidenceSource.VIEWPOINT_CAPTURE, timestamp=12.5,
                        pose=Pose3D(position=(1, 2, 0), stamp=12.5), image_ref=str(path),
                        bbox_xyxy=(1, 2, 3, 4), target_object_uid="object-2")


def test_robot_capture_reaches_visual_model_and_keeps_domain_stop_independent(tmp_path, monkeypatch):
    import instruction_adapter.verifier as module
    image = tmp_path / "capture.png"
    # A valid one-pixel PNG: exact encoded capture bytes are preserved by image_input.
    data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXuoAAAAASUVORK5CYII=")
    image.write_bytes(data)
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        from prompting.schemas import ParsedVerification
        verdict = ParsedVerification(decision="accept", satisfied=True, semantic_satisfied=True,
                                  view_sufficient_for_stop=True, confidence=1.0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=verdict))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "stub"))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    adapter = FinalInstructionVerifierAdapter(FinalInstructionVerifier(evidence_dir=str(tmp_path / "images")))
    result = adapter.verify(_evidence(image), _context())
    content = requests[0]["messages"][1]["content"]
    task = json.loads(content[0]["text"])
    assert task["task"]["active_target"]["id"] == "table"
    binding = json.loads(content[1]["text"])
    assert binding["bbox_xyxy_in_this_image"] == [1, 2, 3, 4]
    assert binding["observation_identity"]["candidate_uid"] == "object-2"
    assert binding["observation_identity"]["task_revision"] == 7
    assert binding["observation_identity"]["pose"]["stamp"] == 12.5
    assert base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1]) == data
    assert "predicted-label" not in json.dumps(content)
    assert "distance_to_target" not in json.dumps(content)
    assert result["satisfied"] is False
    assert result["hard_constraints"]["satisfied"] is False


def test_robot_intent_preserves_active_target():
    intent = NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT, metadata={"target_id": "table", "task_revision": 9})
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(0, 0, 0)))
    context = _verification_context_from_intent(intent, goal)
    assert context["target_id"] == "table"
    assert context["task_revision"] == 9


def test_missing_capture_preserves_candidate_and_mismatched_identity_raises():
    adapter = FinalInstructionVerifierAdapter(SimpleNamespace(verify=lambda **_: pytest.fail("unexpected VLM call")))
    evidence = _evidence("capture.png")
    assert adapter.verify(evidence, {})["decision"] == "need_better_view"
    from dataclasses import replace
    assert adapter.verify(replace(evidence, image_ref=None), _context())["decision"] == "need_better_view"
    assert adapter.verify(replace(evidence, metadata={"image_ref_source": "object_image_ref"}), _context())["decision"] == "need_better_view"
    with pytest.raises(ValueError, match="instance differs"):
        adapter.verify(replace(evidence, target_object_uid="other"), _context())


def test_owner_revision_scopes_legacy_plan_result():
    context = _context()
    context["instruction_plan"] = {"targets": [{"id": "table", "name": "table"}]}
    context["task_revision"] = 9
    captured = []
    def verify(**kwargs):
        captured.append(kwargs)
        return {"task_revision": 1, "decision": "uncertain"}
    adapter = FinalInstructionVerifierAdapter(SimpleNamespace(verify=verify))
    assert adapter.verify(_evidence("capture.png"), context)["task_revision"] == 9
    assert captured[0]["evidence"]["observation_identity"]["task_revision"] == 9
    del context["task_revision"]
    with pytest.raises(ValueError, match="task owner's revision"):
        adapter.verify(_evidence("capture.png"), context)


def test_selection_reads_revision_from_task_owner():
    from planning.semantic_snapshot_context import _selection_metadata
    selection = SimpleNamespace(context=SimpleNamespace(plan=SimpleNamespace(raw_instruction="find table"),
                                                        spec=SimpleNamespace(revision=9)),
                                is_anchor_reference=False, selected_snapshot_uid="object-2",
                                result=SimpleNamespace(target_id="table", answer="yes", anchor_record={},
                                                       skipped_objs=[], concept_debug=[]))
    assert _selection_metadata(selection, CandidateInstance("object-2", "table"))["task_revision"] == 9


def test_missing_image_file_keeps_original_io_error(tmp_path, monkeypatch):
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    adapter = FinalInstructionVerifierAdapter(FinalInstructionVerifier(evidence_dir=str(tmp_path / "images")))
    with pytest.raises(FileNotFoundError):
        adapter.verify(_evidence(tmp_path / "missing.png"), _context())
