"""Task-owned progress and services; the map only publishes observations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from scene.contracts import SemanticMapSnapshot
from instruction_adapter.concept_matcher import AnchorSearchLedger, RuntimeConceptMatcher
from instruction_adapter.constraints import ConstraintEvaluator
from instruction_adapter.semantic_edges import RelationPairLedger
from verification.store import EvidenceStore
from instruction.contracts import TaskSpec
from instruction.compiler import task_from_parsed_plan
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

    def __post_init__(self) -> None:
        """Keep task services outside the scene; observations remain native values."""
        self.concept_matcher = self.concept_matcher or RuntimeConceptMatcher(vlm=self.vlm)
        self.anchor_search_ledger = self.anchor_search_ledger or AnchorSearchLedger()
        self.relation_pair_ledger = self.relation_pair_ledger or RelationPairLedger()
        data = self.plan.diagnostics.get("task_spec")
        self.spec = TaskSpec.from_dict(data) if data else task_from_parsed_plan(self.plan)
        self.verification_ledger = self.verification_ledger or EvidenceStore(self.spec.revision)
        self.progress = TaskProgress(self.spec)
        active = self.progress.active_target()
        self.target = active.name if active is not None else ""
        if self.constraint_evaluator is None:
            self.constraint_evaluator = ConstraintEvaluator(save_dir=self.save_dir)

    def install_plan(self, plan) -> None:
        """Install a newer task revision and start its own progress/evidence scope.

        Args:
            plan: Compiled runtime plan carrying the immutable task contract.

        Raises:
            ValueError: A changed contract reuses or decreases the revision.
        """
        data = plan.diagnostics.get("task_spec")
        spec = TaskSpec.from_dict(data) if data else task_from_parsed_plan(plan)
        if spec == self.spec:
            self.plan = plan
            return
        if spec.revision <= self.spec.revision:
            raise ValueError("Changed task requires a newer revision")
        self.plan, self.spec = plan, spec
        self.progress = TaskProgress(spec)
        self.verification_ledger = EvidenceStore(spec.revision)
        self.anchor_search_ledger = AnchorSearchLedger()
        self.relation_pair_ledger = RelationPairLedger()
        self.target = self.progress.active_target().name

    @property
    def objects(self):
        """Expose native snapshots for consumers migrating from context.objects."""
        return self.snapshot.objects


    @property
    def current_position(self):
        """Return the pose associated with the current scene snapshot."""
        return self.snapshot.robot_pose.position
