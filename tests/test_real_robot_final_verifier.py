"""Scripted robot evidence contracts with capture-owned identity, not real VLM ability."""
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
            "raw_instruction": "find a chair then a table", "revision": 7, "constraints": [],
            "targets": [{"id": "chair", "name": "chair"}, {"id": "table", "name": "table"}],
        }}},
        "verifier_evidence": {"geometry": {"distance_to_target": 2.0, "bbox_xyxy": [90, 90, 99, 99]},
                              "hard_stop_constraints": {"satisfied": False, "failed": []}},
    }


def _evidence(path):
    return ViewEvidence(source=EvidenceSource.VIEWPOINT_CAPTURE, timestamp=12.5,
                        pose=Pose3D(position=(1, 2, 0), stamp=12.5), image_ref=str(path),
                        bbox_xyxy=(1, 2, 3, 4), target_object_uid="object-2")


@pytest.mark.parametrize("visible_context", [
    [], ["A stove and sink stand beside the table on continuous flooring, with no intervening doorway."],
])
def test_robot_capture_reaches_visual_model_and_keeps_domain_stop_independent(tmp_path, monkeypatch, visible_context):
    import instruction_adapter.verifier as module
    image = tmp_path / "capture.png"
    # A valid one-pixel PNG: exact encoded capture bytes are preserved by image_input.
    data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXuoAAAAASUVORK5CYII=")
    image.write_bytes(data)
    requests = []
    def parse(**kwargs):
        requests.append(kwargs)
        content = kwargs["messages"][1]["content"]
        assert kwargs["parse_retries"] == 0
        if kwargs["trace_label"] == "instance_observer":
            current_id = json.loads(content[0]["text"])["observation_identity"]["observation_id"]
            verdict = kwargs["response_format"](observations={current_id: {
                "observed_category": "table", "visible_parts": ["flat top", "legs"],
                "support_and_attachment": "Legs on the floor.", "visible_surface_content": "",
                "visible_context": visible_context, "uncertainties": []}})
        else:
            assert kwargs["trace_label"] == "final_instruction_verifier"
            assert all(item["type"] == "text" for item in content)
            bound = json.loads(content[1]["text"])["bound_observations"]
            current_id = next(uid for uid, record in bound.items() if record["current"])
            verdict = kwargs["response_format"](observed_category="invented desk", observed_structure="Invented drawers.",
                confidence=1.0, predicate_judgments={"candidate_identity": {
                    "verdict": "true", "reason": "This bound region shows a table.",
                    "supporting_observation_ids": [current_id]}})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=verdict))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(module, "get_client_and_model", lambda _: (client, "stub"))
    monkeypatch.setenv("STRIVE_FINAL_VERIFIER", "on")
    monkeypatch.setenv("LLM_OFFLINE", "0")
    adapter = FinalInstructionVerifierAdapter(FinalInstructionVerifier(evidence_dir=str(tmp_path / "images")))
    result = adapter.verify(_evidence(image), _context())
    assert [request["trace_label"] for request in requests] == [
        "instance_observer", "final_instruction_verifier"]
    observer, final = requests
    content = observer["messages"][1]["content"]
    task = json.loads(final["messages"][1]["content"][0]["text"])
    assert task["task"]["active_target"]["id"] == "table"
    assert task["raw_instruction"] == _context()["raw_instruction"]
    assert "find a chair then a table" not in json.dumps(observer["messages"])
    binding = json.loads(content[0]["text"])
    assert binding["bbox_xyxy_in_this_image"] == [1, 2, 3, 4]
    assert binding["observation_identity"]["candidate_uid"] == "object-2"
    assert binding["observation_identity"]["task_revision"] == 7
    assert binding["observation_identity"]["pose"]["stamp"] == 12.5
    assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == data
    assert "predicted-label" not in json.dumps(content)
    assert "distance_to_target" not in json.dumps(content)
    final_content = final["messages"][1]["content"]
    assert all(item["type"] == "text" for item in final_content)
    bound = json.loads(final_content[1]["text"])["bound_observations"]
    record = bound[binding["observation_identity"]["observation_id"]]
    assert record["observation_identity"] == binding["observation_identity"]
    assert record["images"][0]["bbox_xyxy"] == [1, 2, 3, 4]
    assert record["images"][0]["image_sha256"] == binding["image_sha256"]
    assert record["facts"] == {
        "observed_category": "table", "visible_parts": ["flat top", "legs"],
        "support_and_attachment": "Legs on the floor.", "visible_surface_content": "",
        "visible_context": visible_context, "uncertainties": []}
    assert record["current"] is True
    assert result["diagnostics"]["bound_observations"] == bound
    assert result["observed_category"] == "table"
    assert result["diagnostics"]["observed_structure"] == "flat top, legs; Legs on the floor."
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
