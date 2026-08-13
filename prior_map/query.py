"""Prior-map query service for soft search guidance.

The query service converts an ``InstructionPlan``-like object, runtime context,
and ``PriorMapMemory`` into ``SearchPriorResult``. It only produces ranking
signals and diagnostics; it never creates motion goals, navigation intents, or
stop decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    FrontierPrior,
    ObjectPrior,
    RoomPrior,
    SearchPriorResult,
    SupportRegionPrior,
)
from .memory import PriorMapMemory


@dataclass(frozen=True)
class PriorMapQueryWeights:
    """Scoring weights for prior-map query results.

    Args:
        concept_relevance: Weight for target/support concept matches.
        room_relevance: Weight for room hint matches.
        topology_hint: Weight for object-room support from prior topology.
        unvisited_bonus: Bonus for rooms that have not been visited.
        visited_penalty: Penalty for rooms already visited.
        exhausted_penalty: Penalty for rooms marked exhausted.
        live_evidence: Bonus for priors supported by live observations.
        verified_bonus: Bonus for verified prior objects.
        rejection_penalty: Penalty for rejected prior objects.
        exact_bonus: Bonus for exact object instance priors.
        target_object_room: Weight that lifts a target object's parent room.
        geometry_object_bonus: Bonus for target objects with usable geometry.
        frontier_distance: Weight for frontier-to-target geometric proximity.
        frontier_room: Weight for frontier association with a target room.
    """

    concept_relevance: float = 1.0
    room_relevance: float = 0.7
    topology_hint: float = 0.25
    unvisited_bonus: float = 0.2
    visited_penalty: float = -0.15
    exhausted_penalty: float = -0.7
    live_evidence: float = 0.45
    verified_bonus: float = 0.5
    rejection_penalty: float = -1.0
    exact_bonus: float = 0.1
    target_object_room: float = 0.9
    geometry_object_bonus: float = 0.15
    frontier_distance: float = 0.7
    frontier_room: float = 0.25


@dataclass(frozen=True)
class PriorMapQueryContext:
    """Normalized instruction/runtime context used by query scoring.

    Args:
        target_terms: Terms for terminal target concepts.
        room_hints: Room labels or ids from instruction search priors.
        support_terms: Support object terms from instruction search priors.
        affordances: Affordance terms from instruction search priors.
        live_object_uids: Runtime object ids observed recently.
        live_object_labels: Runtime object labels observed recently.
        live_room_uids: Runtime room ids observed recently.
        live_room_labels: Runtime room labels observed recently.
        runtime_frontiers: Runtime frontier records for frontier bias output.
    """

    target_terms: Tuple[str, ...] = ()
    room_hints: Tuple[str, ...] = ()
    support_terms: Tuple[str, ...] = ()
    affordances: Tuple[str, ...] = ()
    live_object_uids: Tuple[str, ...] = ()
    live_object_labels: Tuple[str, ...] = ()
    live_room_uids: Tuple[str, ...] = ()
    live_room_labels: Tuple[str, ...] = ()
    runtime_frontiers: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class _RuntimeFrontierRecord:
    """Normalized runtime frontier geometry for prior-map ranking.

    Args:
        uid: Stable runtime frontier uid.
        raw: Original runtime frontier object.
        position_xyz: Runtime-frame 3-D position when available.
        prior_xy: Position transformed into the prior-map 2-D plane.
        room_uid: Runtime or prior room uid associated with the frontier.
        source: Coordinate source used to compute ``position_xyz``.
        metadata: JSON-friendly extraction diagnostics.
    """

    uid: str
    raw: Any
    position_xyz: Optional[Tuple[float, float, float]] = None
    prior_xy: Optional[Tuple[float, float]] = None
    room_uid: Optional[str] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriorMapQueryService:
    """Generate soft prior-map search rankings.

    Args:
        weights: Optional scoring weights.
        max_rooms: Maximum room priors returned.
        max_objects: Maximum object priors returned.
        max_frontiers: Maximum frontier priors returned.
        max_support_regions: Maximum support region priors returned.
    """

    def __init__(
        self,
        weights: Optional[PriorMapQueryWeights] = None,
        max_rooms: int = 8,
        max_objects: int = 12,
        max_frontiers: int = 24,
        max_support_regions: int = 12,
    ) -> None:
        """Create a query service.

        Args:
            weights: Optional scoring weights.
            max_rooms: Maximum room priors returned.
            max_objects: Maximum object priors returned.
            max_frontiers: Maximum frontier priors returned.
            max_support_regions: Maximum support region priors returned.
        """

        self.weights = weights or PriorMapQueryWeights()
        self.max_rooms = max_rooms
        self.max_objects = max_objects
        self.max_frontiers = max_frontiers
        self.max_support_regions = max_support_regions

    def query(self, plan: Any, runtime_context: Any, memory: PriorMapMemory) -> SearchPriorResult:
        """Query prior-map guidance for the current instruction/runtime state.

        Args:
            plan: ``InstructionPlan``-like object. The service reads target
                names, detector terms, aliases, search priors, constraints, and
                optional target detector prompts by duck typing.
            runtime_context: Mapper/snapshot-like context containing live
                objects, rooms, and frontiers.
            memory: Prior-map runtime memory.

        Returns:
            ``SearchPriorResult`` with room/object/frontier/support rankings and
            diagnostics.
        """

        context = _build_query_context(plan, runtime_context, memory)
        current_map = memory.current_map()
        target_geometry = _target_object_geometry(current_map, context, memory)
        room_rankings = self._rank_rooms(current_map, context, memory, target_geometry)
        object_rankings = self._rank_objects(current_map, context, memory, room_rankings, target_geometry)
        support_regions = self._rank_support_regions(current_map, context, object_rankings)
        frontier_biases = self._rank_frontiers(
            context,
            room_rankings,
            object_rankings,
            current_map,
            memory,
        )
        geometry_disabled_reason = _geometry_disabled_reason(
            memory=memory,
            context=context,
            target_geometry=target_geometry,
            frontier_biases=frontier_biases,
        )

        return SearchPriorResult(
            room_rankings=tuple(room_rankings[: self.max_rooms]),
            object_rankings=tuple(object_rankings[: self.max_objects]),
            frontier_biases=tuple(frontier_biases[: self.max_frontiers]),
            support_regions=tuple(support_regions[: self.max_support_regions]),
            prompt_context={
                "target_terms": list(context.target_terms),
                "room_hints": list(context.room_hints),
                "support_terms": list(context.support_terms),
                "affordances": list(context.affordances),
                "top_rooms": [prior.to_dict() for prior in room_rankings[:3]],
                "top_objects": [prior.to_dict() for prior in object_rankings[:5]],
                "top_frontier_biases": [prior.to_dict() for prior in frontier_biases[:5]],
            },
            diagnostics={
                "score_weights": self.weights.__dict__,
                "live_object_uids": list(context.live_object_uids),
                "live_object_labels": list(context.live_object_labels),
                "live_room_uids": list(context.live_room_uids),
                "live_room_labels": list(context.live_room_labels),
                "alignment": memory.alignment.diagnostics_payload(),
                "geometry_enabled": geometry_disabled_reason is None,
                "geometry_disabled_reason": geometry_disabled_reason,
                "runtime_frontiers": [
                    _frontier_diagnostics(frontier)
                    for frontier in context.runtime_frontiers
                ],
                "authority": "ranking_only",
            },
        )

    def _rank_rooms(
        self,
        current_map: Any,
        context: PriorMapQueryContext,
        memory: PriorMapMemory,
        target_geometry: Dict[str, Dict[str, Any]],
    ) -> List[RoomPrior]:
        """Rank prior rooms from instruction hints and runtime state.

        Args:
            current_map: Runtime-annotated prior map.
            context: Normalized query context.
            memory: Prior-map runtime memory.
            target_geometry: Target-object geometry indexed by prior object uid.

        Returns:
            Sorted room priors.
        """

        results: List[RoomPrior] = []
        room_hints = tuple(_normalize(term) for term in context.room_hints)
        live_rooms = {_normalize(term) for term in (*context.live_room_uids, *context.live_room_labels)}
        for room in current_map.rooms:
            state = memory.room_states[room.uid]
            room_terms = {_normalize(room.uid), _normalize(room.label)}
            room_relevance = _best_match_score(room_terms, room_hints)
            target_object_room_relevance = _target_object_room_relevance(room.uid, target_geometry)
            live_match = 1.0 if room_terms & live_rooms else 0.0
            visited_adjustment = self.weights.visited_penalty if state.visited else self.weights.unvisited_bonus
            exhausted = bool(state.metadata.get("exhausted", False))
            exhausted_penalty = self.weights.exhausted_penalty if exhausted else 0.0
            alignment_confidence = memory.alignment.confidence()
            total = (
                self.weights.room_relevance * room_relevance
                + self.weights.target_object_room * target_object_room_relevance * alignment_confidence
                + self.weights.live_evidence * live_match
                + visited_adjustment
                + exhausted_penalty
            )
            components = {
                "room_relevance": room_relevance,
                "target_object_room_relevance": target_object_room_relevance,
                "live_match": live_match,
                "visited_adjustment": visited_adjustment,
                "exhausted_penalty": exhausted_penalty,
                "alignment_confidence": alignment_confidence,
                "runtime_confidence": state.confidence,
                "total": total,
            }
            results.append(
                RoomPrior(
                    room_uid=room.uid,
                    label=room.label,
                    score=total,
                    reason=_join_reasons(
                        [
                            "matches room hint" if room_relevance > 0 else "",
                            "contains target object prior" if target_object_room_relevance > 0 else "",
                            "supported by live room evidence" if live_match > 0 else "",
                            "unvisited prior room" if not state.visited else "already visited",
                            "marked exhausted" if exhausted else "",
                        ]
                    ),
                    visit_state="visited" if state.visited else "unvisited",
                    reachable_hint="aligned" if memory.alignment.can_rank_geometry() else "prompt_context_only",
                    metadata={"score_components": components},
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _rank_objects(
        self,
        current_map: Any,
        context: PriorMapQueryContext,
        memory: PriorMapMemory,
        room_rankings: Sequence[RoomPrior],
        target_geometry: Dict[str, Dict[str, Any]],
    ) -> List[ObjectPrior]:
        """Rank prior objects from instruction concepts and runtime state.

        Args:
            current_map: Runtime-annotated prior map.
            context: Normalized query context.
            memory: Prior-map runtime memory.
            room_rankings: Current room ranking output.
            target_geometry: Target-object geometry indexed by prior object uid.

        Returns:
            Sorted object priors.
        """

        target_terms = tuple(_normalize(term) for term in context.target_terms)
        support_terms = tuple(_normalize(term) for term in (*context.support_terms, *context.affordances))
        live_labels = {_normalize(term) for term in context.live_object_labels}
        live_uids = {_normalize(term) for term in context.live_object_uids}
        room_scores = {prior.room_uid: prior.score for prior in room_rankings}
        results: List[ObjectPrior] = []

        for obj in current_map.objects:
            state = memory.object_states[obj.uid]
            object_terms = {_normalize(obj.uid), _normalize(obj.label), *(_normalize(alias) for alias in obj.aliases)}
            concept_relevance = _best_match_score(object_terms, target_terms)
            support_relevance = _best_match_score(object_terms, support_terms)
            room_relevance = max(0.0, room_scores.get(obj.parent_room_uid or "", 0.0))
            live_match = 1.0 if object_terms & live_labels or _normalize(obj.uid) in live_uids else 0.0
            if state.matched_runtime_uid and _normalize(state.matched_runtime_uid) in live_uids:
                live_match = 1.0
            if state.observation_count > 0:
                live_match = max(live_match, 0.5)
            verified_bonus = self.weights.verified_bonus if state.verified else 0.0
            rejection_penalty = self.weights.rejection_penalty if state.rejected else 0.0
            exact_bonus = self.weights.exact_bonus if obj.exact else 0.0
            geometry = target_geometry.get(obj.uid, {})
            geometry_bonus = self.weights.geometry_object_bonus * float(geometry.get("target_object_relevance", 0.0))
            geometry_bonus *= memory.alignment.confidence()
            total = (
                self.weights.concept_relevance * concept_relevance
                + self.weights.concept_relevance * 0.6 * support_relevance
                + self.weights.topology_hint * room_relevance
                + self.weights.live_evidence * live_match
                + verified_bonus
                + rejection_penalty
                + exact_bonus
                + geometry_bonus
            )
            components = {
                "concept_relevance": concept_relevance,
                "support_relevance": support_relevance,
                "parent_room_score": room_relevance,
                "live_match": live_match,
                "verified_bonus": verified_bonus,
                "rejection_penalty": rejection_penalty,
                "exact_bonus": exact_bonus,
                "geometry_bonus": geometry_bonus,
                "has_position": obj.position_xyz is not None,
                "alignment_confidence": memory.alignment.confidence(),
                "runtime_confidence": state.confidence,
                "total": total,
            }
            results.append(
                ObjectPrior(
                    object_uid=obj.uid,
                    label=obj.label,
                    score=total,
                    reason=_join_reasons(
                        [
                            "matches target concept" if concept_relevance > 0 else "",
                            "matches support/affordance prior" if support_relevance > 0 else "",
                            "parent room ranked by prior" if room_relevance > 0 else "",
                            "supported by live observation" if live_match > 0 else "",
                            "verified by runtime evidence" if state.verified else "",
                            "rejected prior hypothesis" if state.rejected else "",
                        ]
                    ),
                    parent_room_uid=obj.parent_room_uid,
                    exact=obj.exact,
                    matched_runtime_uid=state.matched_runtime_uid,
                    metadata={"score_components": components, "confidence": state.confidence},
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _rank_support_regions(
        self,
        current_map: Any,
        context: PriorMapQueryContext,
        object_rankings: Sequence[ObjectPrior],
    ) -> List[SupportRegionPrior]:
        """Build support-region priors from support objects and affordances.

        Args:
            current_map: Runtime-annotated prior map.
            context: Normalized query context.
            object_rankings: Object ranking output.

        Returns:
            Sorted support-region priors.
        """

        support_terms = tuple(_normalize(term) for term in (*context.support_terms, *context.affordances))
        object_scores = {prior.object_uid: prior.score for prior in object_rankings}
        results: List[SupportRegionPrior] = []
        if not support_terms:
            return results
        for obj in current_map.objects:
            object_terms = {_normalize(obj.uid), _normalize(obj.label), *(_normalize(alias) for alias in obj.aliases)}
            relevance = _best_match_score(object_terms, support_terms)
            if relevance <= 0.0:
                continue
            score = self.weights.concept_relevance * relevance + 0.25 * max(0.0, object_scores.get(obj.uid, 0.0))
            results.append(
                SupportRegionPrior(
                    uid=obj.uid,
                    label=obj.label,
                    score=score,
                    reason="support object/affordance hint from instruction",
                    room_uid=obj.parent_room_uid,
                    metadata={
                        "source": "prior_object",
                        "score_components": {
                            "support_relevance": relevance,
                            "object_prior_score": object_scores.get(obj.uid, 0.0),
                            "total": score,
                        },
                    },
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _rank_frontiers(
        self,
        context: PriorMapQueryContext,
        room_rankings: Sequence[RoomPrior],
        object_rankings: Sequence[ObjectPrior],
        current_map: Any,
        memory: PriorMapMemory,
    ) -> List[FrontierPrior]:
        """Generate runtime frontier score deltas from geometric priors.

        Args:
            context: Normalized query context.
            room_rankings: Current room rankings.
            object_rankings: Current object rankings.
            current_map: Runtime-annotated prior map.
            memory: Prior-map runtime memory.

        Returns:
            Sorted frontier priors.
        """

        if not memory.alignment.can_rank_geometry():
            return []
        positioned_targets = _positioned_target_priors(current_map, object_rankings, memory)
        if not positioned_targets:
            return []
        room_scores = {prior.room_uid: prior.score for prior in room_rankings}
        room_priors = {prior.room_uid: prior for prior in room_rankings}
        alignment_confidence = memory.alignment.confidence()
        results: List[FrontierPrior] = []
        for frontier in context.runtime_frontiers:
            if not isinstance(frontier, _RuntimeFrontierRecord) or frontier.prior_xy is None:
                continue
            target = _nearest_positioned_target(frontier.prior_xy, positioned_targets)
            if target is None:
                continue
            room_uid = frontier.room_uid or target.get("parent_room_uid")
            room_prior = room_priors.get(str(room_uid)) if room_uid else None
            visited_exhausted_penalty = _room_visit_exhausted_adjustment(room_prior)
            distance_m = float(target["distance_m"])
            distance_score = 1.0 / (1.0 + max(0.0, distance_m))
            target_object_relevance = float(target.get("target_object_relevance", 0.0))
            target_room_relevance = _soft_positive(room_scores.get(str(room_uid), 0.0)) if room_uid else 0.0
            score_delta = alignment_confidence * (
                self.weights.frontier_distance * distance_score
                + self.weights.frontier_room * target_room_relevance
                + 0.25 * target_object_relevance
                + visited_exhausted_penalty
            )
            if score_delta <= 0.0:
                continue
            components = {
                "target_object_relevance": target_object_relevance,
                "target_room_relevance": target_room_relevance,
                "distance_score": distance_score,
                "alignment_confidence": alignment_confidence,
                "visited_exhausted_penalty": visited_exhausted_penalty,
                "distance_m": distance_m,
                "total": score_delta,
            }
            results.append(
                FrontierPrior(
                    frontier_uid=frontier.uid,
                    score_delta=score_delta,
                    reason=(
                        f"frontier is {distance_m:.2f}m from target prior "
                        f"{target['object_uid']}"
                    ),
                    prior_room_uid=str(room_uid) if room_uid else None,
                    target_region_uid=str(room_uid) if room_uid else None,
                    metadata={
                        "alignment_confidence": memory.alignment.confidence(),
                        "source": "geometry_prior",
                        "score_components": components,
                        "target_object_uid": target["object_uid"],
                        "target_object_label": target["label"],
                        "target_object_prior_xy": list(target["prior_xy"]),
                        "frontier_prior_xy": list(frontier.prior_xy),
                        "frontier_position_xyz": list(frontier.position_xyz) if frontier.position_xyz else None,
                        "frontier_position_source": frontier.source,
                    },
                )
            )
        return sorted(results, key=lambda item: item.score_delta, reverse=True)


def _build_query_context(plan: Any, runtime_context: Any, memory: PriorMapMemory) -> PriorMapQueryContext:
    """Normalize plan and runtime data for query scoring.

    Args:
        plan: Instruction plan-like object.
        runtime_context: Mapper/snapshot-like object.
        memory: Prior-map memory.

    Returns:
        Normalized query context.
    """

    latest_observations = memory.observations[-3:]
    live_object_uids = []
    live_object_labels = []
    for record in latest_observations:
        live_object_uids.extend(record.observed_object_uids)
        live_object_labels.extend(record.observed_object_labels)

    runtime_objects = tuple(_iter_records(_first_attr(runtime_context, ("objects", "object_nodes", "detected_objects"), ())))
    runtime_rooms = tuple(_iter_records(_first_attr(runtime_context, ("rooms", "room_nodes"), ())))
    for obj in runtime_objects:
        uid = _runtime_uid(obj)
        label = _runtime_label(obj)
        if uid:
            live_object_uids.append(uid)
        if label:
            live_object_labels.append(label)

    live_room_uids = []
    live_room_labels = []
    for room in runtime_rooms:
        uid = _runtime_uid(room)
        label = _runtime_label(room)
        if uid:
            live_room_uids.append(uid)
        if label:
            live_room_labels.append(label)

    return PriorMapQueryContext(
        target_terms=tuple(_dedupe(_extract_target_terms(plan))),
        room_hints=tuple(_dedupe(_extract_room_hints(plan))),
        support_terms=tuple(_dedupe(_extract_support_terms(plan))),
        affordances=tuple(_dedupe(_extract_affordances(plan))),
        live_object_uids=tuple(_dedupe(live_object_uids)),
        live_object_labels=tuple(_dedupe(live_object_labels)),
        live_room_uids=tuple(_dedupe(live_room_uids)),
        live_room_labels=tuple(_dedupe(live_room_labels)),
        runtime_frontiers=tuple(_runtime_frontier_records(runtime_context, memory)),
    )


def _target_object_geometry(
    current_map: Any,
    context: PriorMapQueryContext,
    memory: PriorMapMemory,
) -> Dict[str, Dict[str, Any]]:
    """Return positioned target-object geometry usable for ranking.

    Args:
        current_map: Runtime-annotated prior map.
        context: Normalized query context.
        memory: Prior-map memory with alignment.

    Returns:
        Mapping from prior object uid to geometry/relevance metadata.
    """

    if not memory.alignment.can_rank_geometry():
        return {}
    target_terms = tuple(_normalize(term) for term in context.target_terms)
    output: Dict[str, Dict[str, Any]] = {}
    for obj in getattr(current_map, "objects", ()) or ():
        if obj.position_xyz is None:
            continue
        object_terms = {_normalize(obj.uid), _normalize(obj.label), *(_normalize(alias) for alias in obj.aliases)}
        relevance = _best_match_score(object_terms, target_terms)
        if relevance <= 0.0:
            continue
        output[obj.uid] = {
            "object_uid": obj.uid,
            "label": obj.label,
            "parent_room_uid": obj.parent_room_uid,
            "prior_xy": _prior_plane_xy(obj.position_xyz, memory.alignment.prior_frame_id),
            "target_object_relevance": relevance,
            "confidence": obj.confidence,
            "exact": obj.exact,
        }
    return output


def _target_object_room_relevance(room_uid: str, target_geometry: Dict[str, Dict[str, Any]]) -> float:
    """Return the strongest positioned target-object support for a room.

    Args:
        room_uid: Prior room uid.
        target_geometry: Target-object geometry indexed by object uid.

    Returns:
        Relevance score in ``[0, 1]``.
    """

    best = 0.0
    for geometry in target_geometry.values():
        if geometry.get("parent_room_uid") != room_uid:
            continue
        confidence = float(geometry.get("confidence", 0.0) or 0.0)
        relevance = float(geometry.get("target_object_relevance", 0.0) or 0.0)
        best = max(best, relevance * max(0.0, min(1.0, confidence)))
    return best


def _positioned_target_priors(
    current_map: Any,
    object_rankings: Sequence[ObjectPrior],
    memory: PriorMapMemory,
) -> List[Dict[str, Any]]:
    """Build positioned target records from object rankings.

    Args:
        current_map: Runtime-annotated prior map.
        object_rankings: Object priors sorted by query score.
        memory: Prior-map memory with alignment.

    Returns:
        Positioned object-prior records sorted by object rank.
    """

    by_uid = {obj.uid: obj for obj in getattr(current_map, "objects", ()) or ()}
    positioned: List[Dict[str, Any]] = []
    for prior in object_rankings:
        obj = by_uid.get(prior.object_uid)
        if obj is None or obj.position_xyz is None:
            continue
        components = dict(prior.metadata.get("score_components", {}) or {})
        relevance = float(components.get("concept_relevance", 0.0) or 0.0)
        if relevance <= 0.0:
            continue
        positioned.append(
            {
                "object_uid": obj.uid,
                "label": obj.label,
                "score": float(prior.score),
                "parent_room_uid": obj.parent_room_uid,
                "prior_xy": _prior_plane_xy(obj.position_xyz, memory.alignment.prior_frame_id),
                "target_object_relevance": relevance,
            }
        )
    return positioned


def _nearest_positioned_target(
    frontier_prior_xy: Tuple[float, float],
    positioned_targets: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the closest positioned target prior to a frontier.

    Args:
        frontier_prior_xy: Frontier position in prior-map plane coordinates.
        positioned_targets: Positioned target records.

    Returns:
        Target record copy with ``distance_m`` field, or ``None``.
    """

    best: Optional[Dict[str, Any]] = None
    best_distance = math.inf
    for target in positioned_targets:
        target_xy = target.get("prior_xy")
        if not target_xy:
            continue
        distance = _euclidean_2d(frontier_prior_xy, target_xy)
        if distance < best_distance:
            best_distance = distance
            best = dict(target)
            best["distance_m"] = distance
    return best


def _room_visit_exhausted_adjustment(room_prior: Optional[RoomPrior]) -> float:
    """Return room state adjustment already encoded in room score components.

    Args:
        room_prior: Matching room prior.

    Returns:
        Sum of visited and exhausted score components.
    """

    if room_prior is None:
        return 0.0
    components = dict(room_prior.metadata.get("score_components", {}) or {})
    return float(components.get("visited_adjustment", 0.0) or 0.0) + float(
        components.get("exhausted_penalty", 0.0) or 0.0
    )


def _runtime_frontier_records(runtime_context: Any, memory: PriorMapMemory) -> List[_RuntimeFrontierRecord]:
    """Normalize runtime frontier or node records for geometric ranking.

    Args:
        runtime_context: Mapper/snapshot-like context.
        memory: Prior-map memory with alignment.

    Returns:
        List of normalized frontier records with stable ids and prior-plane
        positions when possible.
    """

    records = list(_iter_records(_first_attr(runtime_context, ("frontiers", "frontier_nodes"), ())))
    if not records:
        records = [
            node
            for node in _iter_records(_first_attr(runtime_context, ("nodes",), ()))
            if _truthy(_first_attr(node, ("has_frontier", "is_frontier"), False))
        ]

    output: List[_RuntimeFrontierRecord] = []
    seen: Dict[str, int] = {}
    for index, frontier in enumerate(records):
        uid = _dedupe_runtime_uid(_runtime_uid(frontier) or f"frontier_{index}", seen)
        room_uid = _text_or_none(
            _first_attr(frontier, ("prior_room_uid", "room_uid", "room_id", "target_region_uid", "region_uid"))
        )
        position_xyz, source = _runtime_world_position(frontier, runtime_context)
        prior_xy = _runtime_position_to_prior_xy(position_xyz, source, memory) if position_xyz is not None else None
        output.append(
            _RuntimeFrontierRecord(
                uid=uid,
                raw=frontier,
                position_xyz=position_xyz,
                prior_xy=prior_xy,
                room_uid=room_uid,
                source=source,
                metadata={
                    "index": index,
                    "has_position": position_xyz is not None,
                    "has_prior_xy": prior_xy is not None,
                },
            )
        )
    return output


def _runtime_world_position(frontier: Any, runtime_context: Any) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """Extract a runtime frontier position in a world-like frame.

    Args:
        frontier: Runtime frontier or node record.
        runtime_context: Mapper/snapshot-like context.

    Returns:
        ``(position_xyz, source)``. ``position_xyz`` is ``None`` when no usable
        coordinate is present.
    """

    direct_fields = (
        "habitat_world_position",
        "world_position",
        "world_position_xyz",
        "position_xyz",
        "point_xyz",
        "center_xyz",
    )
    for name in direct_fields:
        raw = _first_attr(frontier, (name,), None)
        vector = _vector3_or_none(raw)
        if vector is not None:
            return vector, name

    raw_position = _first_attr(frontier, ("position", "center", "centroid"), None)
    local = _vector3_or_none(raw_position)
    if local is None:
        return None, "missing_position"

    initial_position = _vector3_or_none(_first_attr(runtime_context, ("initial_position",), None))
    if initial_position is not None:
        return _mapper_local_to_habitat_world(local, initial_position), "mapper_local_position"
    return local, "position_untransformed"


def _runtime_position_to_prior_xy(
    position_xyz: Tuple[float, float, float],
    source: str,
    memory: PriorMapMemory,
) -> Optional[Tuple[float, float]]:
    """Transform a runtime position into prior-map plane coordinates.

    Args:
        position_xyz: Runtime-frame position.
        source: Position extraction source.
        memory: Prior-map memory with alignment.

    Returns:
        Prior-frame ``(x, y)`` plane point, or ``None`` when geometry cannot be
        safely transformed.
    """

    if not memory.alignment.can_rank_geometry():
        return None
    try:
        runtime_plane = _plane_xy_from_xyz(position_xyz, memory.alignment.runtime_frame_id, source)
        return memory.alignment.runtime_to_prior((runtime_plane[0], runtime_plane[1], 0.0))
    except Exception:
        return None


def _prior_plane_xy(position_xyz: Sequence[float], prior_frame_id: str) -> Tuple[float, float]:
    """Return the prior-map 2-D plane coordinate for a 3-D position.

    Args:
        position_xyz: Prior-frame 3-D position.
        prior_frame_id: Prior frame id.

    Returns:
        ``(x, z)`` for Habitat frames, otherwise ``(x, y)``.
    """

    vector = _vector3_or_none(position_xyz)
    if vector is None:
        return (0.0, 0.0)
    return _plane_xy_from_xyz(vector, prior_frame_id, "prior_object")


def _plane_xy_from_xyz(
    position_xyz: Sequence[float],
    frame_id: str,
    source: str = "",
) -> Tuple[float, float]:
    """Project a 3-D point onto the navigation plane.

    Args:
        position_xyz: 3-D point.
        frame_id: Coordinate frame id.
        source: Extraction source hint.

    Returns:
        Plane coordinate. Habitat-world points use \(x,z\); generic map points
        use \(x,y\).
    """

    point = _vector3_or_none(position_xyz) or (0.0, 0.0, 0.0)
    frame = _normalize(frame_id)
    source_text = _normalize(source)
    if "habitat" in frame or "habitat" in source_text or "mapper local" in source_text:
        return (point[0], point[2])
    return (point[0], point[1])


def _mapper_local_to_habitat_world(
    local_position: Tuple[float, float, float],
    initial_position: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Convert mapper-local node coordinates to Habitat world coordinates.

    Args:
        local_position: Mapper-local node position.
        initial_position: Episode initial Habitat position.

    Returns:
        Habitat-world position matching the existing planner conversion path.
    """

    return (
        local_position[0] + initial_position[0],
        initial_position[2] - 0.88,
        local_position[1] + initial_position[1],
    )


def _geometry_disabled_reason(
    *,
    memory: PriorMapMemory,
    context: PriorMapQueryContext,
    target_geometry: Dict[str, Dict[str, Any]],
    frontier_biases: Sequence[FrontierPrior],
) -> Optional[str]:
    """Return a concrete geometry fallback reason.

    Args:
        memory: Prior-map memory.
        context: Normalized query context.
        target_geometry: Positioned target-object geometry.
        frontier_biases: Generated frontier priors.

    Returns:
        ``None`` when geometry produced frontier bias; otherwise reason string.
    """

    if frontier_biases:
        return None
    if not memory.alignment.can_rank_geometry():
        return "alignment_unavailable"
    if not target_geometry:
        return "no_positioned_target_object_prior"
    if not context.runtime_frontiers:
        return "no_runtime_frontiers"
    if not any(isinstance(frontier, _RuntimeFrontierRecord) and frontier.prior_xy is not None for frontier in context.runtime_frontiers):
        return "no_runtime_frontier_world_position"
    return "frontier_bias_empty"


def _frontier_diagnostics(frontier: Any) -> Dict[str, Any]:
    """Return JSON-friendly runtime frontier diagnostics.

    Args:
        frontier: Runtime frontier record.

    Returns:
        Diagnostic payload.
    """

    if isinstance(frontier, _RuntimeFrontierRecord):
        return {
            "uid": frontier.uid,
            "room_uid": frontier.room_uid,
            "position_xyz": list(frontier.position_xyz) if frontier.position_xyz else None,
            "prior_xy": list(frontier.prior_xy) if frontier.prior_xy else None,
            "source": frontier.source,
            **dict(frontier.metadata),
        }
    return {"uid": _runtime_uid(frontier), "source": "raw_runtime_frontier"}


def _soft_positive(value: float) -> float:
    """Squash a positive score into ``[0, 1)`` without changing sign.

    Args:
        value: Raw score.

    Returns:
        Squashed non-negative score.
    """

    value = max(0.0, float(value or 0.0))
    return value / (1.0 + value)


def _euclidean_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Euclidean distance in the prior-map plane.

    Args:
        a: First 2-D point.
        b: Second 2-D point.

    Returns:
        Distance.
    """

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _vector3_or_none(value: Any) -> Optional[Tuple[float, float, float]]:
    """Convert a sequence-like value into a 3-D vector.

    Args:
        value: Candidate vector.

    Returns:
        Vector or ``None``.
    """

    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        z = value[2] if len(value) >= 3 else 0.0
        return (float(value[0]), float(value[1]), float(z))
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    """Return Python truthiness for scalar or numpy-like values."""

    try:
        return bool(value)
    except Exception:
        return False


def _dedupe_runtime_uid(uid: str, seen: Dict[str, int]) -> str:
    """Deduplicate runtime frontier ids while preserving first occurrence."""

    key = str(uid)
    count = seen.get(key, 0)
    seen[key] = count + 1
    if count == 0:
        return key
    return f"{key}:{count}"


def _extract_target_terms(plan: Any) -> List[str]:
    """Extract terminal target and detector terms from a plan.

    Args:
        plan: Instruction plan-like object.

    Returns:
        Target terms.
    """

    terms: List[str] = []
    terms.extend(_string_list(_first_attr(plan, ("target_detector_prompts", "detector_prompts"), ())))
    dataset_target = _first_attr(plan, ("dataset_target",), "")
    if dataset_target:
        terms.append(str(dataset_target))
    for target in _iter_records(_first_attr(plan, ("targets",), ())):
        if bool(_first_attr(target, ("terminal",), True)):
            terms.extend(_concept_terms(target))
    for concept in _iter_records(_first_attr(plan, ("concept_queries",), ())):
        if bool(_first_attr(concept, ("terminal",), False)):
            terms.extend(_concept_terms(concept))
    return terms


def _extract_room_hints(plan: Any) -> List[str]:
    """Extract room hints from search priors and constraints.

    Args:
        plan: Instruction plan-like object.

    Returns:
        Room hint terms.
    """

    priors = _first_attr(plan, ("search_priors",), None)
    terms = _string_list(_first_attr(priors, ("room_hints",), ()))
    for constraint in _iter_records(_first_attr(plan, ("constraints",), ())):
        ctype = _normalize(_first_attr(constraint, ("type",), ""))
        if ctype in {"room", "in room", "inside room"}:
            terms.extend(_string_list(_first_attr(constraint, ("value", "object"), ())))
    return terms


def _extract_support_terms(plan: Any) -> List[str]:
    """Extract support object hints from search priors.

    Args:
        plan: Instruction plan-like object.

    Returns:
        Support object terms.
    """

    priors = _first_attr(plan, ("search_priors",), None)
    return _string_list(_first_attr(priors, ("support_objects",), ()))


def _extract_affordances(plan: Any) -> List[str]:
    """Extract affordance hints from search priors.

    Args:
        plan: Instruction plan-like object.

    Returns:
        Affordance terms.
    """

    priors = _first_attr(plan, ("search_priors",), None)
    return _string_list(_first_attr(priors, ("affordances",), ()))


def _concept_terms(value: Any) -> List[str]:
    """Extract name, detector terms, aliases, and concept terms.

    Args:
        value: Target or concept-like object.

    Returns:
        Concept terms.
    """

    terms = []
    for field_name in ("name", "label", "id"):
        text = _first_attr(value, (field_name,), "")
        if text:
            terms.append(str(text))
    terms.extend(_string_list(_first_attr(value, ("detector_terms", "aliases", "match_terms"), ())))
    concept = _first_attr(value, ("concept",), None)
    if concept is not None:
        terms.extend(_concept_terms(concept))
    return terms


def _best_match_score(candidate_terms: Iterable[str], query_terms: Iterable[str]) -> float:
    """Compute lightweight text-match relevance.

    Args:
        candidate_terms: Prior object/room terms.
        query_terms: Instruction query terms.

    Returns:
        Score in ``[0, 1]``.
    """

    best = 0.0
    candidates = [term for term in candidate_terms if term]
    queries = [term for term in query_terms if term]
    for candidate in candidates:
        for query in queries:
            if candidate == query:
                best = max(best, 1.0)
            elif candidate in query or query in candidate:
                best = max(best, 0.7)
    return best


def _iter_records(value: Any) -> Iterable[Any]:
    """Iterate runtime records from common containers.

    Args:
        value: List, tuple, dict, or other value.

    Returns:
        Iterable records.
    """

    if isinstance(value, dict):
        return value.values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _first_attr(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """Return the first attribute or dict key found.

    Args:
        value: Object or dictionary.
        names: Candidate field names.
        default: Fallback value.

    Returns:
        Field value or fallback.
    """

    if value is None:
        return default
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            attr = getattr(value, name)
            if callable(attr) and name == "match_terms":
                try:
                    return attr()
                except TypeError:
                    return attr
            return attr
    return default


def _runtime_uid(value: Any) -> Optional[str]:
    """Extract a runtime uid.

    Args:
        value: Runtime record.

    Returns:
        Runtime uid or ``None``.
    """

    return _text_or_none(_first_attr(value, ("uid", "id", "object_uid", "room_uid", "frontier_uid", "idx")))


def _runtime_label(value: Any) -> str:
    """Extract a runtime label.

    Args:
        value: Runtime record.

    Returns:
        Runtime label.
    """

    return _text_or_none(_first_attr(value, ("label", "tag", "type", "category", "name", "class_name"))) or ""


def _string_list(value: Any) -> List[str]:
    """Convert text or sequences to a list of strings.

    Args:
        value: Source value.

    Returns:
        String list.
    """

    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _dedupe(values: Sequence[str]) -> List[str]:
    """Deduplicate strings while preserving order.

    Args:
        values: Source strings.

    Returns:
        Deduplicated list.
    """

    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        key = _normalize(text)
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _normalize(value: Any) -> str:
    """Normalize text for relevance matching.

    Args:
        value: Source value.

    Returns:
        Normalized lowercase text.
    """

    return str(value or "").strip().lower().replace("_", " ")


def _text_or_none(value: Any) -> Optional[str]:
    """Return stripped text or ``None``.

    Args:
        value: Source value.

    Returns:
        Text or ``None``.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _join_reasons(parts: Sequence[str]) -> str:
    """Join non-empty reason fragments.

    Args:
        parts: Reason fragments.

    Returns:
        Human-readable reason.
    """

    text = "; ".join(part for part in parts if part)
    return text or "no strong prior signal"
