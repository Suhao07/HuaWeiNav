"""Adapter from real-robot semantic snapshots to existing planning policies.

The classes here do not implement a second instruction policy. They expose a
`SemanticMapSnapshot` through the minimal mapper-like protocol already consumed
by `planning.target_selection_policy.select_target_candidate()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from instruction_adapter.concept_matcher import AnchorSearchLedger, RuntimeConceptMatcher
from instruction_adapter.contracts import InstructionPlan
from instruction_adapter.constraints import ConstraintEvaluator
from instruction_adapter.ontology import normalize_term
from instruction_adapter.relation_verifier import DynamicRelationService
from instruction_adapter.semantic_edges import RelationPairLedger
from instruction_adapter.spatial_graph import InstructionSpatialGraph
from instruction_adapter.verifier import VerificationLedger, VerificationResult, candidate_from_object
from planning.object_search_policy import InstructionObjectSearchPolicy
from planning.target_selection_policy import TargetSelectionResult, select_target_candidate
from real_robot.contracts import MotionGoalMode, NavigationIntent, ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot, ViewpointResult


RELATION_CONSTRAINT_TYPES = {
    "spatial",
    "relation",
    "object_relation",
    "object relation",
    "co_occurrence",
    "co occurrence",
}


class SnapshotConfidence:
    """Small confidence wrapper compatible with mapper object expectations.

    The existing simulator policy sorts objects with
    `obj.confidence.numpy().item()`. Real-robot snapshots carry plain floats, so
    this wrapper preserves that narrow interface without changing the policy.
    """

    def __init__(self, value: float) -> None:
        """Create a mapper-compatible confidence value.

        Args:
            value: Numeric confidence score.
        """

        self.value = float(value or 0.0)

    def numpy(self) -> "SnapshotConfidence":
        """Return self for `confidence.numpy().item()` compatibility.

        Returns:
            This object.
        """

        return self

    def item(self) -> float:
        """Return the scalar confidence value.

        Returns:
            Float confidence.
        """

        return self.value

    def __float__(self) -> float:
        """Return the confidence as a Python float."""

        return self.value


class SnapshotObjectAdapter:
    """Mapper-like object wrapper around `ObjectNodeSnapshot`.

    Args:
        snapshot: Source object snapshot.
    """

    def __init__(self, snapshot: ObjectNodeSnapshot) -> None:
        """Initialize the object adapter.

        Args:
            snapshot: Source object snapshot.
        """

        self.snapshot = snapshot
        self.uid = snapshot.uid
        self.tag = snapshot.label
        self.position = list(snapshot.position or snapshot.bbox3d_center or (0.0, 0.0, 0.0))
        self.confidence = SnapshotConfidence(snapshot.confidence)
        self.conf_list: dict[str, SnapshotConfidence] = {snapshot.label: self.confidence}
        self.bbox = list(snapshot.bbox2d_xyxy or ())
        self.room_id = snapshot.room_id
        self.image_ref = snapshot.image_ref
        self.track_ids = tuple(snapshot.track_ids or ())
        self.metadata = dict(snapshot.metadata or {})
        self._instruction_reference_role = ""
        self._instruction_anchor_concept_id = ""
        self._instruction_anchor_concept_name = ""
        self._instruction_anchor_candidate_uid = ""

    def as_snapshot(self) -> ObjectNodeSnapshot:
        """Return the underlying object snapshot.

        Returns:
            Source `ObjectNodeSnapshot`.
        """

        return self.snapshot


@dataclass
class SemanticMapSnapshotPolicyContext:
    """Mapper-like context that lets snapshots reuse existing policies.

    Args:
        snapshot: Current real-robot semantic map snapshot.
        plan: Parsed instruction plan.
        concept_matcher: Optional existing matcher instance.
        verification_ledger: Optional instruction-scoped verification ledger.
        anchor_search_ledger: Optional anchor search ledger.
        relation_pair_ledger: Optional relation pair ledger.
        constraint_evaluator: Optional existing constraint evaluator.
        instruction_object_search_policy: Optional existing object search policy.
        vlm: VLM provider name used when constructing default matchers.
    """

    snapshot: SemanticMapSnapshot
    plan: Any
    concept_matcher: Optional[Any] = None
    verification_ledger: Optional[VerificationLedger] = None
    anchor_search_ledger: Optional[AnchorSearchLedger] = None
    relation_pair_ledger: Optional[RelationPairLedger] = None
    constraint_evaluator: Optional[ConstraintEvaluator] = None
    instruction_object_search_policy: Optional[InstructionObjectSearchPolicy] = None
    vlm: str = "cognav"

    def __post_init__(self) -> None:
        """Initialize the mapper-like fields expected by planning policies."""

        self.instruction_plan = self.plan
        self.instruction_spec = None
        self.objects = [SnapshotObjectAdapter(obj) for obj in self.snapshot.objects]
        self.current_position = list(self.snapshot.robot_pose.position)
        self.current_node_idx = None
        self.target = _active_target_name(self.plan)
        self.target_list = list(getattr(self.plan, "target_detector_prompts", []) or [self.target])
        self.target_aliases = list(getattr(self.plan, "target_match_terms", []) or [self.target])
        self.concept_matcher = self.concept_matcher or RuntimeConceptMatcher(vlm=self.vlm)
        self.verification_ledger = self.verification_ledger or VerificationLedger()
        self.anchor_search_ledger = self.anchor_search_ledger or AnchorSearchLedger()
        self.relation_pair_ledger = self.relation_pair_ledger or RelationPairLedger()
        self.instruction_object_search_policy = self.instruction_object_search_policy or InstructionObjectSearchPolicy()
        if self.constraint_evaluator is None and hasattr(self, "instruction_constraint_evaluator"):
            pass
        elif self.constraint_evaluator is None:
            spatial_graph = InstructionSpatialGraph()
            relation_service = DynamicRelationService(vlm=self.vlm)
            self.instruction_constraint_evaluator = ConstraintEvaluator(
                spatial_graph=spatial_graph,
                relation_service=relation_service,
            )
        else:
            self.instruction_constraint_evaluator = self.constraint_evaluator

    def _target_match_terms(self) -> set[str]:
        """Return normalized terminal target terms.

        Returns:
            Normalized terms accepted as terminal target labels.
        """

        terms = list(self.target_aliases or []) + list(self.target_list or []) + [self.target]
        return {normalize_term(term) for term in terms if normalize_term(term)}

    def _is_target_tag(self, tag: str) -> bool:
        """Return whether a label may satisfy the active target.

        Args:
            tag: Candidate object label.

        Returns:
            True when the label matches active target terms.
        """

        return normalize_term(tag) in self._target_match_terms()

    def _raw_instruction_for_verifier(self) -> str:
        """Return the raw instruction used by verifier ledgers.

        Returns:
            Raw instruction string.
        """

        return str(getattr(self.plan, "raw_instruction", "") or self.target or "")

    def _is_verifier_rejected(self, obj: Any, step: int | None = None) -> bool:
        """Return whether final verifier hard-rejected this instance.

        Args:
            obj: Mapper-like object adapter.
            step: Optional runtime step.

        Returns:
            True if the existing verification ledger marks the instance as hard
            rejected for the active instruction.
        """

        candidate = candidate_from_object(obj, canonical_label=self.target, step=step)
        return self.verification_ledger.is_hard_rejected(
            self._raw_instruction_for_verifier(),
            candidate.uid,
        )

    def _anchor_concepts_for_plan(self) -> list[Any]:
        """Return non-terminal concepts used by anchor-first search.

        Returns:
            Deduplicated non-terminal anchor concepts.
        """

        concepts = []
        for target in getattr(self.plan, "anchor_targets", []):
            concepts.append(target.concept_query())
        for constraint in getattr(self.plan, "constraints", []) or []:
            concept = getattr(constraint, "object_concept", None)
            if concept is not None and not getattr(concept, "terminal", False):
                concepts.append(concept)
        return _dedupe_concepts(concepts)

    def _anchor_concepts_for_terminal_target(self, plan: Any, target: Any) -> list[Any]:
        """Return relation anchors required before chasing a terminal target.

        Args:
            plan: Instruction plan.
            target: Terminal target query.

        Returns:
            Non-terminal concepts required by hard relation constraints.
        """

        target_terms = {normalize_term(term) for term in getattr(target, "match_terms", [])}
        target_terms.add(normalize_term(getattr(target, "id", "")))
        concepts = []
        for constraint in getattr(plan, "constraints", []) or []:
            ctype = normalize_term(getattr(constraint, "type", ""))
            if ctype not in RELATION_CONSTRAINT_TYPES:
                continue
            subject = normalize_term(getattr(constraint, "subject", ""))
            if subject and subject not in target_terms:
                continue
            concept = getattr(constraint, "object_concept", None)
            if concept is not None and not getattr(concept, "terminal", False):
                concepts.append(concept)
        return _dedupe_concepts(concepts)

    def _has_unblocked_anchor_evidence(
        self,
        plan: Any,
        target: Any,
        candidate_obj: Any,
        step: int | None = None,
        debug: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """Return whether relation-constrained target has an available anchor.

        Args:
            plan: Instruction plan.
            target: Terminal target query.
            candidate_obj: Candidate terminal object.
            step: Optional runtime step.
            debug: Optional list receiving concept-match diagnostics.

        Returns:
            True when no anchor is required or a matching unblocked anchor is
            currently observed.
        """

        concepts = self._anchor_concepts_for_terminal_target(plan, target)
        if not concepts:
            return True
        raw_instruction = self._raw_instruction_for_verifier()
        for concept in concepts:
            objects = [obj for obj in self.objects if obj is not candidate_obj]
            records = self.concept_matcher.match_many(
                raw_instruction=raw_instruction,
                concept=concept,
                objects=objects,
                step=step,
            )
            records_by_uid = {record.object_uid: record for record in records}
            for obj in objects:
                anchor_uid = candidate_from_object(obj, canonical_label=concept.name, step=step).uid
                record = records_by_uid.get(anchor_uid)
                if record is None:
                    continue
                if debug is not None:
                    debug.append(record.as_dict())
                if not record.matches_concept:
                    continue
                if self.anchor_search_ledger.is_blocked(raw_instruction, concept.id, anchor_uid):
                    continue
                return True
        return False


@dataclass(frozen=True)
class SnapshotTargetSelection:
    """Result of running existing target selection on a semantic snapshot.

    Args:
        context: Mapper-like context used by the existing policy.
        result: Existing `TargetSelectionResult`.
    """

    context: SemanticMapSnapshotPolicyContext
    result: TargetSelectionResult

    @property
    def selected_snapshot_uid(self) -> str:
        """Return selected runtime object uid, or an empty string.

        Returns:
            Source `ObjectNodeSnapshot.uid` when a candidate was selected.
        """

        obj = self.result.obj
        return str(getattr(obj, "uid", "") if obj is not None else "")

    @property
    def selected_pose(self) -> Optional[Pose3D]:
        """Return selected object pose in the snapshot frame.

        Returns:
            Pose for the selected object, or None when no object was selected.
        """

        obj = self.result.obj
        if obj is None:
            return None
        source = obj.as_snapshot() if hasattr(obj, "as_snapshot") else None
        position = getattr(source, "position", None) if source is not None else getattr(obj, "position", None)
        if position is None:
            return None
        return Pose3D(
            position=tuple(float(x) for x in position[:3]),
            frame_id=self.context.snapshot.robot_pose.frame_id,
            stamp=self.context.snapshot.timestamp,
        )

    @property
    def is_anchor_reference(self) -> bool:
        """Return whether the selected object is an anchor reference.

        Returns:
            True when existing policy selected an anchor, not a terminal target.
        """

        return bool(getattr(self.result.obj, "_instruction_reference_role", "") == "anchor")

    @property
    def candidate_instance(self) -> Any | None:
        """Return the existing verifier candidate for the selected object.

        Returns:
            Candidate generated by `instruction_adapter.verifier`, or None when
            no object was selected.
        """

        if self.result.obj is None:
            return None
        return candidate_from_object(
            self.result.obj,
            canonical_label=str(getattr(self.result.obj, "tag", "") or self.context.target),
        )

    def to_navigation_intent(self) -> NavigationIntent:
        """Convert the existing target selection result to a motion intent.

        This method is an adapter, not a second navigation policy: target and
        anchor semantics already came from `select_target_candidate()`.

        Returns:
            `NavigationIntent` for the selected target/anchor, or WAIT when no
            executable pose is available.
        """

        if not self.result.found or self.result.obj is None:
            return NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason=self.result.answer or "no instruction target candidate selected",
                metadata=_selection_metadata(self, candidate=None),
            )
        pose = self.selected_pose
        if pose is None:
            return NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="selected semantic object has no executable pose",
                metadata=_selection_metadata(self, candidate=None),
            )
        candidate = self.candidate_instance
        mode = MotionGoalMode.GO_TO_ANCHOR if self.is_anchor_reference else MotionGoalMode.GO_TO_OBJECT
        return NavigationIntent(
            mode=mode,
            goal_pose=pose,
            target_object_uid=None if self.is_anchor_reference else self.selected_snapshot_uid,
            anchor_object_uid=self.selected_snapshot_uid if self.is_anchor_reference else None,
            reason=self.result.answer or f"selected {self.selected_snapshot_uid}",
            metadata=_selection_metadata(self, candidate=candidate),
        )


class StaticInstructionPlanProvider:
    """Return one precompiled instruction plan for runtime policy calls."""

    def __init__(self, plan: InstructionPlan) -> None:
        """Initialize the provider.

        Args:
            plan: Precompiled instruction plan.
        """

        self.plan = plan

    def __call__(self, instruction: Optional[str] = None) -> InstructionPlan:
        """Return the stored plan.

        Args:
            instruction: Ignored raw instruction compatibility argument.

        Returns:
            The precompiled plan.
        """

        return self.plan


class SemanticMapSnapshotIntentAdapter:
    """Adapt semantic snapshot target selection into `NavigationIntent`.

    The adapter implements the real-robot runtime policy protocol without
    owning terminal/anchor/relation semantics. Those semantics remain in
    `InstructionObjectSearchPolicy`, `ConstraintEvaluator`, and verifier
    ledgers.
    """

    def __init__(
        self,
        plan_provider: Callable[[Optional[str]], InstructionPlan],
        *,
        context: Optional[SemanticMapSnapshotPolicyContext] = None,
        vlm: str = "cognav",
    ) -> None:
        """Initialize the adapter.

        Args:
            plan_provider: Callable returning the active `InstructionPlan`.
            context: Optional persistent snapshot policy context.
            vlm: VLM provider name used for default matchers in the context.
        """

        self.plan_provider = plan_provider
        self.context = context
        self.vlm = vlm
        self.step = 0
        self.last_selection: Optional[SnapshotTargetSelection] = None
        self.last_viewpoint_result: Optional[ViewpointResult] = None

    def decide(self, snapshot: SemanticMapSnapshot, instruction: Optional[str] = None) -> NavigationIntent:
        """Return the next intent for a semantic snapshot.

        Args:
            snapshot: Current semantic map snapshot.
            instruction: Raw instruction string.

        Returns:
            Navigation intent produced from existing target-selection logic.
        """

        plan = self.plan_provider(instruction)
        if not getattr(plan, "valid", False) or not getattr(plan, "terminal_targets", []):
            return NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="compiled instruction plan is invalid or has no terminal target",
                metadata={
                    "policy": "semantic_map_snapshot_context",
                    "raw_instruction": instruction or "",
                    "plan_valid": bool(getattr(plan, "valid", False)),
                    "plan_diagnostics": dict(getattr(plan, "diagnostics", {}) or {}),
                },
            )

        if self.context is not None:
            state = getattr(self.context, "instruction_execution_state", None)
            if bool(getattr(state, "completed", False)):
                return NavigationIntent(
                    mode=MotionGoalMode.STOP,
                    stop_allowed=True,
                    reason="instruction execution state is completed",
                    metadata={
                        "policy": "semantic_map_snapshot_context",
                        "raw_instruction": plan.raw_instruction,
                        "instruction_plan": plan,
                        "execution_state": state.as_dict() if hasattr(state, "as_dict") else {},
                    },
                )

        selection = select_target_candidate_from_snapshot(
            snapshot,
            plan=plan,
            context=self.context,
            step=self.step,
            vlm=self.vlm,
        )
        self.context = selection.context
        self.last_selection = selection
        self.step += 1
        return selection.to_navigation_intent()

    def handle_viewpoint_result(self, result: ViewpointResult, intent: NavigationIntent) -> bool:
        """Record final verifier feedback in existing ledgers/state.

        Args:
            result: Evidence/verifier result captured after navigation reached.
            intent: Intent that led to the viewpoint.

        Returns:
            True when policy state changed and completed-goal suppression can be
            cleared.
        """

        self.last_viewpoint_result = result
        if self.context is None:
            return False
        candidate = intent.metadata.get("candidate_instance")
        if candidate is None and self.last_selection is not None:
            candidate = self.last_selection.candidate_instance
        candidate_uid = str(getattr(candidate, "uid", "") or intent.metadata.get("candidate_uid") or "")
        if not candidate_uid:
            return False

        verifier_decision = dict((result.metadata or {}).get("verifier_decision") or {})
        if not verifier_decision:
            return False
        verification_result = _verification_result_from_dict(verifier_decision)
        raw_instruction = self.context._raw_instruction_for_verifier()
        self.context.verification_ledger.put(
            raw_instruction,
            candidate_uid,
            verification_result,
            evidence_paths=_evidence_paths(result),
        )

        plan = self.context.plan
        state = self.context.instruction_constraint_evaluator.ensure_state(self.context, plan)
        target = self.context.instruction_constraint_evaluator.target_for_candidate(self.context, plan, candidate)
        if verification_result.satisfied or verification_result.decision == "accept":
            if target is not None:
                state.mark_candidate_accepted(plan, target, candidate_uid)
            return True
        if verification_result.decision == "reject_candidate":
            state.mark_candidate_rejected(target, candidate_uid)
            return True
        return verification_result.decision == "need_better_view"


def select_target_candidate_from_snapshot(
    snapshot: SemanticMapSnapshot,
    *,
    plan: Any,
    context: SemanticMapSnapshotPolicyContext | None = None,
    step: int | None = None,
    **context_kwargs: Any,
) -> SnapshotTargetSelection:
    """Run existing target selection on a real-robot semantic snapshot.

    Args:
        snapshot: Current semantic map snapshot.
        plan: Parsed instruction plan.
        context: Optional existing context carrying persistent ledgers.
        step: Optional runtime step.
        **context_kwargs: Constructor arguments for
            `SemanticMapSnapshotPolicyContext` when `context` is not provided.

    Returns:
        Snapshot target selection with the context and existing policy result.
    """

    if context is None:
        context = SemanticMapSnapshotPolicyContext(snapshot=snapshot, plan=plan, **context_kwargs)
    else:
        context.snapshot = snapshot
        context.plan = plan
        context.__post_init__()
    result = select_target_candidate(context, plan=plan, step=step)
    return SnapshotTargetSelection(context=context, result=result)


def _active_target_name(plan: Any) -> str:
    active = getattr(plan, "active_terminal_target", None)
    if active is not None:
        return str(getattr(active, "name", "") or "")
    terminals = list(getattr(plan, "terminal_targets", []) or [])
    if terminals:
        return str(getattr(terminals[0], "name", "") or "")
    return str(getattr(plan, "dataset_target", "") or "")


def _selection_metadata(selection: SnapshotTargetSelection, candidate: Any | None) -> dict[str, Any]:
    plan = selection.context.plan
    role = "anchor" if selection.is_anchor_reference else "target"
    return {
        "policy": "semantic_map_snapshot_context",
        "raw_instruction": str(getattr(plan, "raw_instruction", "")),
        "instruction_plan": plan,
        "selected_snapshot_uid": selection.selected_snapshot_uid,
        "selection_role": role,
        "selection_answer": selection.result.answer,
        "candidate_instance": candidate,
        "candidate_uid": str(getattr(candidate, "uid", "") if candidate is not None else ""),
        "anchor_record": dict(selection.result.anchor_record or {}),
        "skipped_objs": list(selection.result.skipped_objs or []),
        "concept_debug": list(selection.result.concept_debug or []),
    }


def _verification_result_from_dict(payload: dict[str, Any]) -> VerificationResult:
    return VerificationResult(
        satisfied=bool(payload.get("satisfied", payload.get("decision") == "accept")),
        decision=str(payload.get("decision", "")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        semantic_satisfied=bool(payload.get("semantic_satisfied", payload.get("satisfied", False))),
        view_sufficient_for_stop=bool(payload.get("view_sufficient_for_stop", True)),
        hard_constraints=dict(payload.get("hard_constraints") or {}),
        satisfied_constraints=list(payload.get("satisfied_constraints") or []),
        failed_constraints=list(payload.get("failed_constraints") or []),
        view_feedback=str(payload.get("view_feedback", "")),
        preferred_view_goal=str(payload.get("preferred_view_goal", "")),
        view_objective=dict(payload.get("view_objective") or {}),
        reason=str(payload.get("reason", "")),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _evidence_paths(result: ViewpointResult) -> list[str]:
    evidence = result.evidence
    refs = []
    if evidence is not None and evidence.image_ref:
        refs.append(str(evidence.image_ref))
    verifier_decision = dict((result.metadata or {}).get("verifier_decision") or {})
    for key in ("current_rgb_with_bbox_path", "object_crop_path", "centered_view_path"):
        value = verifier_decision.get(key)
        if value:
            refs.append(str(value))
    return refs


def _dedupe_concepts(concepts: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for concept in concepts:
        key = str(getattr(concept, "id", "") or normalize_term(getattr(concept, "name", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(concept)
    return out
