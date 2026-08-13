"""Offline tests for prior-map multimodal evidence and prompt-first services."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from prior_map.high_level_selector import (
    HighLevelCandidate,
    PriorMapHighLevelSelector,
    build_runtime_candidates,
)
from prior_map.multimodal import PriorMapMultimodalContext
from prior_map.room_semantics import RoomEvidence, RoomSemanticCache, RoomSemanticClassifier
from prior_map.visualizer import FloorPlanOverlay, build_floorplan_overlay
from prior_map.contracts import PriorMapData, PriorRoom
from prior_map.alignment import PriorMapAlignment
from prompting.schemas import ParsedHighLevelSelection, ParsedRoomSemantic


class _FakeCompletions:
    """Minimal multimodal client used to assert request construction."""

    def __init__(self, parsed, content: str) -> None:
        self.parsed = parsed
        self.content = content
        self.messages = None

    def parse(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=self.parsed, content=self.content)
                )
            ]
        )


class _FakeClient:
    """OpenAI-compatible fake exposing the nested parse interface."""

    def __init__(self, parsed, content: str) -> None:
        completions = _FakeCompletions(parsed, content)
        self.completions = completions
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_multimodal_context_hash_and_image_block(tmp_path: Path) -> None:
    image = tmp_path / "bev.png"
    image.write_bytes(b"fake-png")

    context = PriorMapMultimodalContext(image_path=str(image), image_role="dynamic_bev")

    assert context.available is True
    assert len(context.image_sha256) == 64
    assert context.as_image_content()["type"] == "image_url"


def test_room_cache_is_evidence_versioned(tmp_path: Path) -> None:
    image = tmp_path / "room.png"
    image.write_bytes(b"room-v1")
    cache = RoomSemanticCache(tmp_path / "room_cache.json")
    import os

    os.environ["LLM_OFFLINE"] = "1"
    classifier = RoomSemanticClassifier(vlm="cognav", scene_id="scene", cache=cache)
    evidence = RoomEvidence(room_uid="room-1", rgb_path=str(image), geometry_summary={"area": 4})

    first = classifier.classify(evidence)
    cached = classifier.classify(evidence)
    image.write_bytes(b"room-v2")
    changed = classifier.classify(evidence)

    assert first.source == "fallback"
    assert cached.source == "cache"
    assert changed.evidence_hash != first.evidence_hash
    assert json.loads((tmp_path / "room_cache.json").read_text(encoding="utf-8"))


def test_room_classifier_does_not_treat_ros_mask_uri_as_rgb() -> None:
    evidence = RoomEvidence(room_uid="room-1", room_mask_path="ros:///room_mask/1")

    result = RoomSemanticClassifier(vlm="cognav").classify(evidence)

    assert result.label == "unknown"
    assert result.source == "fallback"


def test_room_classifier_sends_rgb_and_mask_and_records_audit_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_OFFLINE", raising=False)
    rgb = tmp_path / "room.png"
    mask = tmp_path / "room_mask.png"
    rgb.write_bytes(b"rgb")
    mask.write_bytes(b"mask")
    fake = _FakeClient(
        ParsedRoomSemantic(
            label="food-preparation area",
            description="counter and cabinets are visible",
            confidence=0.86,
            alternatives=["utility area"],
            evidence_summary="RGB and mask agree on the room extent",
            uncertainty="partial view",
        ),
        '{"label":"food-preparation area"}',
    )
    classifier = RoomSemanticClassifier(vlm="mock", client=fake, model="mock-room")
    result = classifier.classify(
        RoomEvidence(room_uid="room-1", rgb_path=str(rgb), room_mask_path=str(mask))
    )

    user_content = fake.completions.messages[1]["content"]
    assert [block["type"] for block in user_content] == ["text", "image_url", "image_url"]
    assert result.source == "vlm"
    assert result.raw_response["text"] == '{"label":"food-preparation area"}'
    assert result.raw_response["parsed"]["label"] == "food-preparation area"
    assert result.request_metadata["rgb_sha256"]
    assert result.request_metadata["room_mask_sha256"]
    assert result.latency_ms >= 0.0


def test_high_level_selector_offline_result_stays_inside_candidate_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_OFFLINE", "1")
    image = tmp_path / "bev.png"
    image.write_bytes(b"fake-png")
    context = PriorMapMultimodalContext(image_path=str(image), image_role="dynamic_bev")
    candidates = (
        HighLevelCandidate(uid="room-a", candidate_type="room", label="candidate a"),
        HighLevelCandidate(uid="room-b", candidate_type="room", label="candidate b"),
    )

    result = PriorMapHighLevelSelector(vlm="cognav", scene_id="scene").select(
        instruction="find the cup",
        instruction_plan={"target": "cup"},
        context=context,
        candidates=candidates,
    )

    assert result.selected_uid in {candidate.uid for candidate in candidates}
    assert result.source == "fallback"


def test_high_level_selector_sends_bev_and_candidate_json_and_rejects_unknown_uid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_OFFLINE", raising=False)
    image = tmp_path / "bev.png"
    image.write_bytes(b"fake-png")
    context = PriorMapMultimodalContext(image_path=str(image), image_role="dynamic_bev")
    candidates = (
        HighLevelCandidate(uid="room-a", candidate_type="room", label="candidate a"),
        HighLevelCandidate(uid="frontier-1", candidate_type="frontier", label="frontier"),
    )
    fake = _FakeClient(
        ParsedHighLevelSelection(
            selected_uid="not-a-candidate",
            selected_type="frontier",
            decision="select",
            confidence=0.9,
        ),
        '{"selected_uid":"not-a-candidate"}',
    )
    result = PriorMapHighLevelSelector(vlm="mock", client=fake, model="mock-selector").select(
        instruction="find a cup",
        instruction_plan={"target": "cup"},
        context=context,
        candidates=candidates,
    )

    user_content = fake.completions.messages[1]["content"]
    payload = json.loads(user_content[0]["text"])
    assert user_content[1]["type"] == "image_url"
    assert payload["candidate_uid_contract"] == ["frontier-1", "room-a"]
    assert result.selected_uid == "room-a"
    assert result.source == "fallback"
    assert result.raw_response["text"] == '{"selected_uid":"not-a-candidate"}'
    assert result.request_metadata["candidate_count"] == 2


def test_runtime_candidate_builder_keeps_room_and_frontier_identity() -> None:
    candidates = build_runtime_candidates(
        rooms=(SimpleNamespace(uid="room-a", label="unknown"),),
        frontiers=(SimpleNamespace(uid="frontier-1", room_id="room-a", position=(1.0, 2.0, 0.0)),),
    )

    assert [(item.uid, item.candidate_type) for item in candidates] == [
        ("room-a", "room"),
        ("frontier-1", "frontier"),
    ]
    assert candidates[1].room_uid == "room-a"


def test_runtime_candidate_builder_normalizes_numpy_like_position() -> None:
    class _Array:
        def tolist(self):
            return [1.0, 2.0, 3.0]

    candidates = build_runtime_candidates(
        frontiers=(SimpleNamespace(uid="frontier-1", position=_Array()),)
    )

    assert candidates[0].metadata["position"] == [1.0, 2.0, 3.0]


def test_dynamic_overlay_serializes_runtime_room_context() -> None:
    prior_map = PriorMapData(
        scene_id="scene",
        rooms=(PriorRoom(uid="room-a", label="unknown", centroid_xy=(1.0, 1.0)),),
    )

    overlay = build_floorplan_overlay(
        prior_map,
        robot_position_xyz=(0.5, 0.0, 0.5),
        current_room_uid="room-a",
        candidate_room_uids=("room-a",),
        selected_room_uid="room-a",
    )

    assert overlay.robot_position_xy == (0.5, 0.0)
    assert overlay.current_room_uid == "room-a"
    assert overlay.to_dict()["selected_room_uid"] == "room-a"


def test_dynamic_overlay_uses_runtime_to_prior_alignment() -> None:
    prior_map = PriorMapData(
        scene_id="scene",
        frame_id="prior_map",
        source_format="hm3d_groundtruth",
        rooms=(PriorRoom(uid="room-a", label="unknown", centroid_xy=(1.0, 1.0)),),
    )
    alignment = PriorMapAlignment.affine_2d(
        scale=1.0,
        rotation_rad=0.0,
        translation_xyz=(10.0, 0.0, 0.0),
        runtime_frame_id="habitat_world",
        confidence=1.0,
    )

    overlay = build_floorplan_overlay(
        prior_map,
        robot_position_xyz=(11.0, 0.0, 0.5),
        alignment=alignment,
    )

    assert overlay.robot_position_xy == (1.0, 0.5)


def test_dynamic_overlay_does_not_fabricate_runtime_position_without_alignment() -> None:
    prior_map = PriorMapData(
        scene_id="scene",
        frame_id="prior_map",
        source_format="hm3d_groundtruth",
        rooms=(PriorRoom(uid="room-a", label="unknown", centroid_xy=(1.0, 1.0)),),
    )

    overlay = build_floorplan_overlay(
        prior_map,
        robot_position_xyz=(11.0, 0.0, 0.5),
        alignment=PriorMapAlignment.unavailable(
            reason="calibration pending", prior_frame_id="prior_map", runtime_frame_id="map"
        ),
    )

    assert overlay.robot_position_xy is None
    assert overlay.trajectory_xy == ()
