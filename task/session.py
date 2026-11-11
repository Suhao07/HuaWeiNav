"""Task-owned progress and services; the map only publishes observations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from scene.contracts import SemanticMapSnapshot
from instruction_adapter.concept_matcher import AnchorSearchLedger, RuntimeConceptMatcher
from instruction_adapter.constraints import ConstraintEvaluator
from instruction_adapter.semantic_edges import RelationPairLedger
from instruction_adapter.relation_verifier import DynamicRelationService
from verification.store import EvidenceStore
from instruction.contracts import TaskSpec
from instruction.compiler import task_from_parsed_plan
from instruction_adapter.task_compat import runtime_plan
from task.state import TaskProgress

@dataclass
class TaskSession:
    """Task-owned services and current native scene input.

    Args:
        snapshot: Current real-robot semantic map snapshot.
        plan: Parsed instruction plan.
        concept_matcher: Optional existing matcher instance.
        verification_ledger: Optional instruction-scoped verification ledger.
        anchor_search_ledger: Optional anchor search ledger.
        relation_pair_ledger: Optional relation pair ledger.
        constraint_evaluator: Optional existing constraint evaluator.
        prior_result: Optional `SearchPriorResult` used as ranking evidence.
        prior_map_policy_adapter: Optional adapter for target annotations.
        prior_map_prompt_context: Optional bounded prompt context bundle.
        prior_map_high_level_selection: Optional room/frontier selection result.
        prior_map_room_semantics: Optional evidence-versioned room labels.
        prior_map_diagnostics: Optional conflict/alignment diagnostics.
        vlm: VLM provider name used when constructing default matchers.
        interpretation_locked: Keep the task contract fixed while allowing search revisions.
    """

    snapshot: SemanticMapSnapshot
    plan: Any
    spec: TaskSpec = field(init=False)
    progress: TaskProgress = field(init=False)
    concept_matcher: Optional[Any] = None
    verification_ledger: Optional[EvidenceStore] = None
    anchor_search_ledger: Optional[AnchorSearchLedger] = None
    relation_pair_ledger: Optional[RelationPairLedger] = None
    constraint_evaluator: Optional[ConstraintEvaluator] = None
    prior_result: Optional[Any] = None
    prior_map_policy_adapter: Optional[Any] = None
    prior_map_prompt_context: Optional[Any] = None
    prior_map_high_level_selection: Optional[Any] = None
    prior_map_room_semantics: Optional[dict[str, Any]] = None
    prior_map_diagnostics: Optional[dict[str, Any]] = None
    save_dir: str = ""
    vlm: str = "cognav"
    destination_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    search_revision: int = 0
    search_hypotheses: dict[str, dict[str, Any]] = field(default_factory=dict)
    interpretation_history: list[dict[str, Any]] = field(default_factory=list)
    interpretation_locked: bool = False

    def __post_init__(self) -> None:
        """Keep task services outside the scene; observations remain native values."""
        self.concept_matcher = self.concept_matcher or RuntimeConceptMatcher(vlm=self.vlm)
        self.anchor_search_ledger = self.anchor_search_ledger or AnchorSearchLedger()
        self.relation_pair_ledger = self.relation_pair_ledger or RelationPairLedger()
        data = self.plan.diagnostics.get("task_spec")
        self.spec = TaskSpec.from_dict(data) if data else task_from_parsed_plan(self.plan)
        self.plan = runtime_plan(self.spec, plan=self.plan)
        self.verification_ledger = self.verification_ledger or EvidenceStore(self.spec.revision)
        self.progress = TaskProgress(self.spec)
        active = self.progress.active_target()
        self.target = active.name if active is not None else ""
        if self.constraint_evaluator is None:
            self.constraint_evaluator = ConstraintEvaluator(
                save_dir=self.save_dir, relation_service=DynamicRelationService(vlm=self.vlm))
        if isinstance(self.concept_matcher, RuntimeConceptMatcher):
            self.concept_matcher.identity_resolver = self.constraint_evaluator.identity_match

    def install_plan(self, plan) -> None:
        """Install a newer task revision and start its own progress/evidence scope.

        Args:
            plan: Compiled runtime plan carrying the immutable task contract.

        Raises:
            ValueError: The contract is locked, or a changed contract reuses or decreases the revision.
        """
        data = plan.diagnostics.get("task_spec")
        spec = TaskSpec.from_dict(data) if data else task_from_parsed_plan(plan)
        if spec == self.spec:
            same_plan = plan is self.plan
            self.plan = runtime_plan(spec, plan=plan)
            if same_plan:
                return
            for concept in self.search_concepts:
                if concept.id in self.search_hypotheses:
                    concept.detector_terms = list(self.search_hypotheses[concept.id]["detector_terms"])
            for target in plan.targets:
                if target.id in self.search_hypotheses:
                    target.detector_terms = list(self.search_hypotheses[target.id]["detector_terms"])
            self.search_revision += 1
            return
        if self.interpretation_locked:
            raise ValueError("Task interpretation is locked; only search revisions are allowed")
        if spec.revision <= self.spec.revision:
            raise ValueError("Changed task requires a newer revision")
        if spec.raw_instruction != self.spec.raw_instruction:
            self.interpretation_history = []
        self.plan, self.spec = runtime_plan(spec, plan=plan), spec
        self.progress = TaskProgress(spec)
        self.verification_ledger = EvidenceStore(spec.revision)
        self.anchor_search_ledger = AnchorSearchLedger()
        self.relation_pair_ledger = RelationPairLedger()
        self.constraint_evaluator.identity_records.clear()
        self.constraint_evaluator.relation_service.reset()
        if isinstance(self.concept_matcher, RuntimeConceptMatcher):
            self.concept_matcher.reset()
        self.target = self.progress.active_target().name
        self.destination_hypotheses = []
        self.search_revision = 0
        self.search_hypotheses = {}
        self.prior_result = None
        self.prior_map_prompt_context = None
        self.prior_map_high_level_selection = None
        self.prior_map_diagnostics = None

    def adopt_interpretation(self, plan, *, task_revision, search_revision, snapshot, record):
        from copy import deepcopy
        from instruction.contracts import interpretation_changes
        if task_revision != self.spec.revision or search_revision != self.search_revision:
            raise ValueError("Interpretation proposal belongs to a stale revision")
        if snapshot is not self.snapshot:
            raise ValueError("Interpretation observation changed during inference")
        proposed = TaskSpec.from_dict(plan.diagnostics["task_spec"])
        if self.spec.source != "language" or proposed.source != "language":
            raise ValueError("Only language interpretations may be revised")
        if proposed.raw_instruction != self.spec.raw_instruction:
            raise ValueError("Interpretation cannot replace the original instruction")
        if proposed.revision != task_revision + 1:
            raise ValueError("Interpretation requires the next task revision")
        if not interpretation_changes(self.spec, proposed):
            raise ValueError("Retrieval-only changes require search revision")
        archived = deepcopy({
            **record, "previous_task": self.spec.as_dict(), "task": proposed.as_dict(),
            "previous_progress": self.progress.as_dict(),
            "previous_verification": self.verification_ledger.as_dict(),
            "previous_search_hypotheses": self.search_hypotheses,
        })
        self.install_plan(plan)
        self.interpretation_history.append(archived)
        return deepcopy(archived)

    @property
    def search_concepts(self):
        values = [target.concept_query() for target in self.plan.targets]
        values.extend(self.plan.concept_queries)
        values.extend(concept for constraint in self.plan.constraints
                      for concept in (constraint.subject_concept, constraint.object_concept)
                      if concept is not None)
        return values

    @property
    def detector_terms(self):
        return list(dict.fromkeys(term for concept in self.search_concepts
                                  for term in concept.detector_terms))

    def revise_search(self, updates, *, task_revision, search_revision, snapshot):
        """Replace retrieval hypotheses without changing task semantics or evidence."""
        from copy import deepcopy
        from instruction_adapter.ontology import dedupe_terms
        if task_revision != self.spec.revision or search_revision != self.search_revision:
            raise ValueError("Search proposal belongs to a stale revision")
        if snapshot is not self.snapshot:
            raise ValueError("Search proposal observation changed during inference")
        concepts = self.search_concepts
        by_id = {concept.id: concept for concept in concepts}
        prepared = {}
        for update in updates:
            uid = update["concept_id"]
            if uid not in by_id or uid in prepared:
                raise ValueError("Search proposal has an unknown or duplicate concept ID")
            terms = dedupe_terms(update["detector_terms"])
            if not terms:
                raise ValueError("Search proposal requires retrieval phrases")
            prepared[uid] = {**deepcopy(update), "detector_terms": terms}
        changed = {uid: row for uid, row in prepared.items()
                   if row["detector_terms"] != by_id[uid].detector_terms}
        if not changed:
            return []
        self.search_revision += 1
        for uid, row in changed.items():
            for concept in concepts:
                if concept.id == uid:
                    concept.detector_terms = list(row["detector_terms"])
            for target in self.plan.targets:
                if target.id == uid:
                    target.detector_terms = list(row["detector_terms"])
            self.search_hypotheses[uid] = {
                **row, "task_revision": task_revision, "search_revision": self.search_revision,
                "observation_timestamp": snapshot.timestamp,
            }
        return deepcopy([self.search_hypotheses[uid] for uid in changed])

    @property
    def objects(self):
        """Expose native snapshots for consumers migrating from context.objects."""
        return self.snapshot.objects


    @property
    def current_position(self):
        """Return the pose associated with the current scene snapshot."""
        return self.snapshot.robot_pose.position
