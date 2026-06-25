"""Policy-facing adapters for prior-map query results.

The adapter layer is intentionally small: it consumes ``SearchPriorResult`` and
returns reordered runtime records or debug annotations. It does not create
navigation intents, motion goals, verifier decisions, or stop conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import FrontierPrior, ObjectPrior, RoomPrior, SearchPriorResult


@dataclass(frozen=True)
class PriorAnnotatedCandidate:
    """Target-candidate wrapper carrying prior-map debug context.

    Args:
        candidate: Original target candidate object. The adapter never mutates
            it.
        prior: Matched object prior, when available.
        prior_score: Matched prior score.
        metadata: JSON-friendly debug annotation for logs or prompts.
    """

    candidate: Any
    prior: Optional[ObjectPrior] = None
    prior_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_prior(self) -> bool:
        """Return whether this candidate matched an object prior."""

        return self.prior is not None


class PriorMapPolicyAdapter:
    """Apply prior-map rankings as optional soft policy hints.

    Args:
        enabled: If ``False``, ranking methods preserve input order and
            annotations report that prior-map guidance is disabled.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Create a policy adapter."""

        self.enabled = bool(enabled)

    def rank_rooms(self, rooms: Iterable[Any], prior_result: Optional[SearchPriorResult]) -> List[Any]:
        """Return rooms sorted by prior-map score while preserving stability.

        Args:
            rooms: Runtime room records from mapper/planner code.
            prior_result: Query output from ``PriorMapQueryService``.

        Returns:
            A new list containing the original room objects in prior-biased
            order. Unknown rooms keep score ``0`` and retain relative order.
        """

        room_list = list(rooms)
        if not self.enabled or prior_result is None:
            return room_list

        room_priors = tuple(prior_result.room_rankings or ())
        return [
            room
            for _, _, room in sorted(
                (
                    (-_room_score(room, room_priors), index, room)
                    for index, room in enumerate(room_list)
                ),
                key=lambda item: (item[0], item[1]),
            )
        ]

    def rank_frontiers(
        self,
        frontiers: Iterable[Any],
        prior_result: Optional[SearchPriorResult],
    ) -> List[Any]:
        """Return frontiers sorted by prior-map score delta.

        Args:
            frontiers: Runtime frontier records.
            prior_result: Query output from ``PriorMapQueryService``.

        Returns:
            A new list containing the original frontier objects in prior-biased
            order. Unmatched frontiers keep score ``0`` and retain relative
            order.
        """

        frontier_list = list(frontiers)
        if not self.enabled or prior_result is None:
            return frontier_list

        frontier_priors = tuple(prior_result.frontier_biases or ())
        return [
            frontier
            for _, _, frontier in sorted(
                (
                    (-_frontier_score(frontier, frontier_priors), index, frontier)
                    for index, frontier in enumerate(frontier_list)
                ),
                key=lambda item: (item[0], item[1]),
            )
        ]

    def annotate_target_candidates(
        self,
        candidates: Iterable[Any],
        prior_result: Optional[SearchPriorResult],
    ) -> List[PriorAnnotatedCandidate]:
        """Attach prior-map debug annotation to target candidates.

        Args:
            candidates: Runtime target candidates from the existing selection
                path.
            prior_result: Query output from ``PriorMapQueryService``.

        Returns:
            Candidate wrappers in the same order as the input. This method does
            not filter, accept, reject, or reorder candidates.
        """

        candidate_list = list(candidates)
        if not self.enabled or prior_result is None:
            return [
                PriorAnnotatedCandidate(
                    candidate=candidate,
                    metadata={"prior_map": {"enabled": False, "matched": False}},
                )
                for candidate in candidate_list
            ]

        object_priors = tuple(prior_result.object_rankings or ())
        return [
            _annotate_candidate(candidate, object_priors)
            for candidate in candidate_list
        ]


def rank_rooms(
    rooms: Iterable[Any],
    prior_result: Optional[SearchPriorResult],
    *,
    enabled: bool = True,
) -> List[Any]:
    """Convenience wrapper for ``PriorMapPolicyAdapter.rank_rooms``."""

    return PriorMapPolicyAdapter(enabled=enabled).rank_rooms(rooms, prior_result)


def rank_frontiers(
    frontiers: Iterable[Any],
    prior_result: Optional[SearchPriorResult],
    *,
    enabled: bool = True,
) -> List[Any]:
    """Convenience wrapper for ``PriorMapPolicyAdapter.rank_frontiers``."""

    return PriorMapPolicyAdapter(enabled=enabled).rank_frontiers(frontiers, prior_result)


def annotate_target_candidates(
    candidates: Iterable[Any],
    prior_result: Optional[SearchPriorResult],
    *,
    enabled: bool = True,
) -> List[PriorAnnotatedCandidate]:
    """Convenience wrapper for ``PriorMapPolicyAdapter.annotate_target_candidates``."""

    return PriorMapPolicyAdapter(enabled=enabled).annotate_target_candidates(candidates, prior_result)


def _room_score(room: Any, room_priors: Sequence[RoomPrior]) -> float:
    room_uid = _first_text(room, ("room_uid", "uid", "id", "idx"))
    room_label = _first_text(room, ("label", "tag", "name", "room_label"))
    for prior in room_priors:
        if _matches_any(room_uid, (prior.room_uid,)) or _matches_any(room_label, (prior.label,)):
            return float(prior.score)
    return 0.0


def _frontier_score(frontier: Any, frontier_priors: Sequence[FrontierPrior]) -> float:
    frontier_uid = _first_text(frontier, ("frontier_uid", "uid", "id", "idx"))
    frontier_rooms = _texts(
        frontier,
        ("prior_room_uid", "room_uid", "room_id", "target_region_uid", "region_uid"),
    )
    best_score = 0.0
    for prior in frontier_priors:
        direct_match = _matches_any(frontier_uid, (prior.frontier_uid,))
        room_match = _any_match(frontier_rooms, (prior.prior_room_uid, prior.target_region_uid))
        if direct_match or room_match:
            best_score = max(best_score, float(prior.score_delta))
    return best_score


def _annotate_candidate(candidate: Any, object_priors: Sequence[ObjectPrior]) -> PriorAnnotatedCandidate:
    prior = _match_object_prior(candidate, object_priors)
    if prior is None:
        return PriorAnnotatedCandidate(
            candidate=candidate,
            metadata={"prior_map": {"enabled": True, "matched": False}},
        )

    prior_map = {
        "enabled": True,
        "matched": True,
        "object_uid": prior.object_uid,
        "label": prior.label,
        "score": prior.score,
        "reason": prior.reason,
        "parent_room_uid": prior.parent_room_uid,
        "exact": prior.exact,
        "matched_runtime_uid": prior.matched_runtime_uid,
        "score_components": dict(prior.metadata.get("score_components", {})),
    }
    return PriorAnnotatedCandidate(
        candidate=candidate,
        prior=prior,
        prior_score=float(prior.score),
        metadata={"prior_map": prior_map},
    )


def _match_object_prior(candidate: Any, object_priors: Sequence[ObjectPrior]) -> Optional[ObjectPrior]:
    candidate_ids = _texts(
        candidate,
        (
            "object_uid",
            "runtime_uid",
            "uid",
            "id",
            "idx",
            "prior_uid",
            "prior_object_uid",
            "track_id",
        ),
    )
    candidate_labels = _texts(candidate, ("label", "tag", "name", "category", "class_name"))

    for prior in object_priors:
        if _any_match(candidate_ids, (prior.object_uid, prior.matched_runtime_uid)):
            return prior
    for prior in object_priors:
        if _any_match(candidate_labels, (prior.label,)):
            return prior
    return None


def _first_text(value: Any, names: Sequence[str]) -> Optional[str]:
    for text in _texts(value, names):
        return text
    return None


def _texts(value: Any, names: Sequence[str]) -> Tuple[str, ...]:
    texts: List[str] = []
    for name in names:
        raw = _attr(value, name)
        if raw is None:
            continue
        if isinstance(raw, (str, int, float)):
            texts.append(str(raw))
            continue
        if isinstance(raw, (list, tuple, set)):
            texts.extend(str(item) for item in raw if item is not None)
    return tuple(text for text in texts if text.strip())


def _attr(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _matches_any(value: Optional[str], candidates: Iterable[Optional[str]]) -> bool:
    if value is None:
        return False
    value_norm = _normalize(value)
    return any(value_norm == _normalize(candidate) for candidate in candidates if candidate is not None)


def _any_match(values: Iterable[str], candidates: Iterable[Optional[str]]) -> bool:
    candidate_set = {_normalize(candidate) for candidate in candidates if candidate is not None}
    return any(_normalize(value) in candidate_set for value in values)


def _normalize(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


__all__ = [
    "PriorAnnotatedCandidate",
    "PriorMapPolicyAdapter",
    "annotate_target_candidates",
    "rank_frontiers",
    "rank_rooms",
]
