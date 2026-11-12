"""Read-only semantic candidate queries shared by simulation and robot inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instruction.contracts import RELATION_CONSTRAINT_TYPES, resolve_target_reference
from instruction_adapter.ontology import normalize_term
from planning.mode_policy import is_anchor_first_relation_search
from scene.contracts import ObjectNodeSnapshot, SemanticMapSnapshot


@dataclass
class TargetSelectionResult:
    found: bool
    obj: Any | None = None
    answer: str = ""
    target_id: str = ""
    canonical_label: str = ""
    selection_role: str = ""
    anchor_record: dict[str, Any] = field(default_factory=dict)
    skipped_objs: list[dict[str, Any]] = field(default_factory=list)
    concept_debug: list[dict[str, Any]] = field(default_factory=list)



@dataclass(frozen=True)
class CandidateMatch:
    """Task-specific interpretation of an unchanged scene object."""

    obj: ObjectNodeSnapshot
    target_id: str
    canonical_label: str
    role: str = "target"
    confidence: float = 0.0
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSet:
    """Ordered candidates plus reasons for excluding other observed instances."""

    candidates: list[CandidateMatch] = field(default_factory=list)
    skipped_objs: list[dict[str, Any]] = field(default_factory=list)
    concept_debug: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def selected(self):
        """Return the highest-ranked candidate, if available."""
        return self.candidates[0] if self.candidates else None


def anchor_concepts(plan, target=None):
    """Return distinct relation references, optionally restricted to a target."""
    values = []
    if target is None:
        values.extend(t.concept_query() for t in plan.anchor_targets)
    for constraint in plan.constraints:
        if normalize_term(constraint.type) not in RELATION_CONSTRAINT_TYPES:
            continue
        subject = resolve_target_reference(plan.targets, constraint.subject, predicate_id="anchor query", field="subject")
        reference = resolve_target_reference(plan.targets, constraint.object, predicate_id="anchor query", field="object")
        if target is None or subject == target.id:
            concept = constraint.object_concept
        elif reference == target.id:
            concept = constraint.subject_concept
        else:
            continue
        if concept is not None and not concept.terminal:
            values.append(concept)
    return list({c.id: c for c in values}.values())


def query_candidates(plan, progress, snapshot: SemanticMapSnapshot, evidence, *,
                     matcher, anchor_ledger, step=None, prior_adapter=None,
                     prior_result=None, deferred_uids=()) -> CandidateSet:
    """Query fixed observations without changing map labels or task progress.

    Args:
        plan: Compiled semantic concepts and detector bindings during migration.
        progress: Authoritative task progress, read only here.
        snapshot: Backend-independent observed scene.
        evidence: Instance rejection index.
        matcher: Batched concept matcher shared by both backends.
        anchor_ledger: Search-attempt index, distinct from semantic rejection.
        step: Observation sequence for diagnostics.
        prior_adapter: Optional existing soft-prior ranker.
        prior_result: Optional frozen prior result.
    """
    result = CandidateSet()
    raw = plan.raw_instruction
    objects = tuple(obj for obj in snapshot.objects
                    if obj.metadata.get("label_probabilities") != {})
    terminals = ([t for t in plan.terminal_targets if t.id == progress.active_target().id]
                 if progress.task.ordered else plan.terminal_targets)
    terminals = [t for t in terminals if t is not None]
    # Within one snapshot, each concept is matched once, even when multiple
    # terminal candidates share the same required anchor.
    matches = {}

    def records(concept):
        if concept.id not in matches:
            rows = matcher.match_many(raw_instruction=raw, concept=concept,
                                      objects=list(objects), step=step)
            matches[concept.id] = {r.object_uid: r for r in rows}
            result.concept_debug.extend(r.as_dict() for r in rows)
        return matches[concept.id]

    pending = progress.pending_verified_pair
    pinned = pending.get("candidate_uid") if progress.mode == "better_view_for_verified_pair" else None
    pinned_target = next((t for t in terminals if t.id == pending.get("target_id")), None)
    pinned_obj = snapshot.object_by_uid(pinned) if pinned else None
    if (pinned_obj is not None and pinned_target is not None and pinned_obj.uid not in deferred_uids
            and not evidence.is_hard_rejected(raw, pinned_obj.uid)):
        row = records(pinned_target.concept_query()).get(pinned_obj.uid)
        if row is not None and row.candidate_eligible:
            result.candidates.append(CandidateMatch(pinned_obj, pinned_target.id, pinned_target.name,
                                                   record=row.as_dict()))
            result.reason = "Continue observation of the same pending verified instance"
            return result

    for target in terminals:
        accepted = progress.target_progress[target.id].accepted_candidate_uids
        rows = records(target.concept_query())
        for obj in objects:
            row = rows.get(obj.uid)
            # 接近是取证动作，不要求身份及关系已被证实。
            if row is None or not row.candidate_eligible:
                continue
            reason = ""
            if obj.uid in accepted:
                reason = "already_counted_instance"
            elif evidence.is_hard_rejected(raw, obj.uid):
                reason = "rejected_instance"
            elif obj.uid in deferred_uids:
                reason = "waiting_for_relevant_evidence"
            if reason:
                result.skipped_objs.append({"uid": obj.uid, "target_id": target.id, "skip_reason": reason})
                continue
            result.candidates.append(CandidateMatch(obj, target.id, target.name,
                                                     confidence=obj.confidence, record=row.as_dict()))
    if result.candidates:
        prior_scores = {}
        if prior_adapter is not None and prior_result is not None:
            annotations = prior_adapter.annotate_target_candidates([c.obj for c in result.candidates], prior_result)
            prior_scores = {a.candidate.uid: a.prior_score for a in annotations}
            result.concept_debug.extend({"source": "prior_map_target_annotation", "candidate_uid": a.candidate.uid,
                                         **a.metadata.get("prior_map", {})} for a in annotations)
        result.candidates.sort(key=lambda c: (prior_scores.get(c.obj.uid, 0), c.confidence), reverse=True)
    anchors = []
    concepts = (anchor_concepts(plan, terminals[0]) if progress.task.ordered else anchor_concepts(plan))
    for concept in concepts:
        rows = records(concept)
        for obj in objects:
            row = rows.get(obj.uid)
            if (row is not None and row.candidate_eligible and obj.uid not in deferred_uids
                    and not anchor_ledger.is_blocked(raw, concept.id, obj.uid)):
                anchors.append(CandidateMatch(obj, concept.id, concept.name, "anchor", row.confidence, row.as_dict()))
    anchors.sort(key=lambda c: c.confidence, reverse=True)
    # 保留两类候选，使参照物到达后的查询仍能看到待验证的终点。
    result.candidates = (anchors + result.candidates if is_anchor_first_relation_search(plan)
                         else result.candidates + anchors)
    if not result.candidates:
        result.reason = "No eligible observed candidate"
    elif result.selected.role == "anchor":
        result.reason = "Search from a relation anchor"
    else:
        result.reason = "Observed target candidates for instance and constraint verification"
    return result


def query_viewpoints(candidate: CandidateMatch, snapshot: SemanticMapSnapshot):
    """Return existing viewpoints associated with a candidate; synthesize no pose."""
    return tuple(v for v in snapshot.viewpoints
                 if v.uid in candidate.obj.visible_viewpoints or candidate.obj.uid in v.visible_objects)
