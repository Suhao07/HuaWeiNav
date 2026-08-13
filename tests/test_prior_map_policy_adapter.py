from pathlib import Path
from types import SimpleNamespace

from prior_map.contracts import FrontierPrior, ObjectPrior, RoomPrior, SearchPriorResult
from prior_map.policy_adapter import (
    PriorAnnotatedCandidate,
    PriorMapPolicyAdapter,
    annotate_target_candidates,
    rank_frontiers,
    rank_rooms,
)


def _prior_result() -> SearchPriorResult:
    return SearchPriorResult(
        room_rankings=(
            RoomPrior(room_uid="room_b", label="kitchen", score=1.2, reason="instruction room hint"),
            RoomPrior(room_uid="room_a", label="living room", score=0.3, reason="weak fallback"),
        ),
        object_rankings=(
            ObjectPrior(
                object_uid="prior_book",
                label="book",
                score=1.7,
                reason="target concept match",
                parent_room_uid="room_b",
                exact=True,
                matched_runtime_uid="runtime_book",
                metadata={"score_components": {"concept_relevance": 1.0, "live_match": 1.0}},
            ),
            ObjectPrior(
                object_uid="prior_table",
                label="table",
                score=0.5,
                reason="support object hint",
                parent_room_uid="room_b",
                metadata={"score_components": {"support_relevance": 1.0}},
            ),
        ),
        frontier_biases=(
            FrontierPrior(
                frontier_uid="frontier_b",
                score_delta=0.9,
                reason="frontier leads to kitchen",
                prior_room_uid="room_b",
            ),
            FrontierPrior(
                frontier_uid="frontier_a",
                score_delta=0.2,
                reason="frontier leads to living room",
                prior_room_uid="room_a",
            ),
        ),
    )


def test_rank_rooms_uses_prior_scores_without_mutating_rooms() -> None:
    room_a = SimpleNamespace(uid="room_a", label="living room")
    room_b = SimpleNamespace(uid="room_b", label="kitchen")
    room_c = SimpleNamespace(uid="room_c", label="bathroom")
    rooms = [room_a, room_c, room_b]

    ranked = rank_rooms(rooms, _prior_result())

    assert [room.uid for room in ranked] == ["room_b", "room_a", "room_c"]
    assert ranked[0] is room_b
    assert not hasattr(room_b, "prior_map")


def test_rank_frontiers_uses_direct_and_room_prior_matches() -> None:
    frontier_a = SimpleNamespace(uid="frontier_x", room_uid="room_a")
    frontier_b = SimpleNamespace(uid="frontier_b", room_uid="room_z")
    frontier_c = SimpleNamespace(uid="frontier_c", room_uid="room_c")
    frontiers = [frontier_a, frontier_c, frontier_b]

    ranked = rank_frontiers(frontiers, _prior_result())

    assert [frontier.uid for frontier in ranked] == ["frontier_b", "frontier_x", "frontier_c"]


def test_annotate_target_candidates_preserves_candidate_order_and_adds_debug_context() -> None:
    table = SimpleNamespace(uid="runtime_table", tag="table")
    book = SimpleNamespace(uid="runtime_book", tag="book")
    unknown = SimpleNamespace(uid="runtime_unknown", tag="lamp")

    annotations = annotate_target_candidates([table, book, unknown], _prior_result())

    assert [annotation.candidate.uid for annotation in annotations] == [
        "runtime_table",
        "runtime_book",
        "runtime_unknown",
    ]
    assert all(isinstance(annotation, PriorAnnotatedCandidate) for annotation in annotations)
    assert annotations[0].prior.object_uid == "prior_table"
    assert annotations[1].prior.object_uid == "prior_book"
    assert annotations[1].metadata["prior_map"]["score_components"]["live_match"] == 1.0
    assert annotations[2].prior is None
    assert annotations[2].metadata["prior_map"]["matched"] is False
    assert not hasattr(book, "prior_map")


def test_disabled_adapter_preserves_existing_policy_behavior() -> None:
    adapter = PriorMapPolicyAdapter(enabled=False)
    rooms = [SimpleNamespace(uid="room_a"), SimpleNamespace(uid="room_b")]
    frontiers = [SimpleNamespace(uid="frontier_a"), SimpleNamespace(uid="frontier_b")]
    candidates = [SimpleNamespace(uid="runtime_book", tag="book")]

    assert adapter.rank_rooms(rooms, _prior_result()) == rooms
    assert adapter.rank_frontiers(frontiers, _prior_result()) == frontiers

    annotations = adapter.annotate_target_candidates(candidates, _prior_result())
    assert annotations[0].candidate is candidates[0]
    assert annotations[0].prior is None
    assert annotations[0].metadata["prior_map"] == {"enabled": False, "matched": False}


def test_policy_adapter_stays_platform_neutral_and_ranking_only() -> None:
    source = Path("prior_map/policy_adapter.py").read_text(encoding="utf-8")
    forbidden_imports = ("rclpy", "habitat", "cv2", "open3d", "geometry_msgs")
    for name in forbidden_imports:
        assert name not in source

    annotations = annotate_target_candidates(
        [SimpleNamespace(uid="runtime_book", tag="book")],
        _prior_result(),
    )
    assert not hasattr(annotations[0], "motion_goal")
    assert not hasattr(annotations[0], "navigation_intent")
