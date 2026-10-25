"""Verification values shared by simulator and robot task execution."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class CandidateInstance:
    """A concrete mapped object candidate.

    The detector label is intentionally separate from uid.  For instructions
    such as "find the red chair", rejecting one blue chair must not reject every
    chair in the scene.
    """

    uid: str
    detector_label: str
    canonical_label: str = ""
    centroid: list[float] = field(default_factory=list)
    bbox_2d: list[float] = field(default_factory=list)
    confidence: float = 0.0
    step: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationRecord:
    """Stored judgment with its task, predicate and observation dependencies."""
    instruction_hash: str
    candidate_uid: str
    status: str
    confidence: float = 0.0
    decision: str = ""
    failed_constraints: list[str] = field(default_factory=list)
    reason: str = ""
    step: int | None = None
    evidence_paths: list[str] = field(default_factory=list)
    task_revision: int | None = None
    predicate_id: str = "final_stop"
    anchor_uid: str = ""
    verdict: str = "unknown"
    dependencies: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """A verifier judgment; task progress consumes its explicit scope."""
    satisfied: bool
    decision: str
    confidence: float = 0.0
    semantic_satisfied: bool = False
    view_sufficient_for_stop: bool = True
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    satisfied_constraints: list[str] = field(default_factory=list)
    failed_constraints: list[str] = field(default_factory=list)
    view_feedback: str = ""
    preferred_view_goal: str = ""
    view_objective: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task_revision: int | None = None
    target_uid: str = ""
    predicate_id: str = "final_stop"
    anchor_uid: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """Return three-valued evidence semantics for this predicate."""
        if self.accepted:
            return "true"
        if self.decision in ("reject_candidate", "reject_relation"):
            return "false"
        return "unknown"

    @property
    def accepted(self) -> bool:
        """Require explicit, consistent semantic and view confirmation."""
        return (
            self.satisfied and self.decision == "accept"
            and self.semantic_satisfied and self.view_sufficient_for_stop
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
