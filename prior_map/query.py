"""Prior-map query service for soft search guidance.

The query service converts an ``InstructionPlan``-like object, runtime context,
and ``PriorMapMemory`` into ``SearchPriorResult``. It only produces ranking
signals and diagnostics; it never creates motion goals, navigation intents, or
stop decisions.
"""

from __future__ import annotations

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
        room_rankings = self._rank_rooms(current_map, context, memory)
        object_rankings = self._rank_objects(current_map, context, memory, room_rankings)
        support_regions = self._rank_support_regions(current_map, context, object_rankings)
        frontier_biases = self._rank_frontiers(context, room_rankings, memory)

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
            },
            diagnostics={
                "score_weights": self.weights.__dict__,
                "live_object_uids": list(context.live_object_uids),
                "live_object_labels": list(context.live_object_labels),
                "live_room_uids": list(context.live_room_uids),
                "live_room_labels": list(context.live_room_labels),
                "alignment": memory.alignment.diagnostics_payload(),
                "authority": "ranking_only",
            },
        )

    def _rank_rooms(
        self,
        current_map: Any,
        context: PriorMapQueryContext,
        memory: PriorMapMemory,
    ) -> List[RoomPrior]:
        """Rank prior rooms from instruction hints and runtime state.

        Args:
            current_map: Runtime-annotated prior map.
            context: Normalized query context.
            memory: Prior-map runtime memory.

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
            live_match = 1.0 if room_terms & live_rooms else 0.0
            visited_adjustment = self.weights.visited_penalty if state.visited else self.weights.unvisited_bonus
            exhausted = bool(state.metadata.get("exhausted", False))
            exhausted_penalty = self.weights.exhausted_penalty if exhausted else 0.0
            total = (
                self.weights.room_relevance * room_relevance
                + self.weights.live_evidence * live_match
                + visited_adjustment
                + exhausted_penalty
            )
            components = {
                "room_relevance": room_relevance,
                "live_match": live_match,
                "visited_adjustment": visited_adjustment,
                "exhausted_penalty": exhausted_penalty,
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
    ) -> List[ObjectPrior]:
        """Rank prior objects from instruction concepts and runtime state.

        Args:
            current_map: Runtime-annotated prior map.
            context: Normalized query context.
            memory: Prior-map runtime memory.
            room_rankings: Current room ranking output.

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
            total = (
                self.weights.concept_relevance * concept_relevance
                + self.weights.concept_relevance * 0.6 * support_relevance
                + self.weights.topology_hint * room_relevance
                + self.weights.live_evidence * live_match
                + verified_bonus
                + rejection_penalty
                + exact_bonus
            )
            components = {
                "concept_relevance": concept_relevance,
                "support_relevance": support_relevance,
                "parent_room_score": room_relevance,
                "live_match": live_match,
                "verified_bonus": verified_bonus,
                "rejection_penalty": rejection_penalty,
                "exact_bonus": exact_bonus,
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
        memory: PriorMapMemory,
    ) -> List[FrontierPrior]:
        """Generate runtime frontier score deltas from room priors.

        Args:
            context: Normalized query context.
            room_rankings: Current room rankings.
            memory: Prior-map runtime memory.

        Returns:
            Sorted frontier priors.
        """

        if not memory.alignment.can_rank_geometry():
            return []
        room_scores = {prior.room_uid: prior.score for prior in room_rankings}
        results: List[FrontierPrior] = []
        for frontier in context.runtime_frontiers:
            frontier_uid = _runtime_uid(frontier) or f"frontier_{len(results)}"
            room_uid = _text_or_none(_first_attr(frontier, ("prior_room_uid", "room_uid", "room_id", "target_region_uid")))
            if not room_uid:
                continue
            score_delta = 0.25 * room_scores.get(room_uid, 0.0)
            if score_delta == 0.0:
                continue
            results.append(
                FrontierPrior(
                    frontier_uid=frontier_uid,
                    score_delta=score_delta,
                    reason=f"frontier associated with prior room {room_uid}",
                    prior_room_uid=room_uid,
                    target_region_uid=room_uid,
                    metadata={
                        "alignment_confidence": memory.alignment.confidence(),
                        "source": "prior_room_ranking",
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
        runtime_frontiers=tuple(_iter_records(_first_attr(runtime_context, ("frontiers", "frontier_nodes"), ()))),
    )


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

    return _text_or_none(_first_attr(value, ("label", "type", "category", "name"))) or ""


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
