from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import InstructionPlan, TargetQuery
from .ontology import compact_key, normalize_term


def _constraint_key(value: Any) -> str:
    text = repr(value)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class ConstraintStatus:
    """Runtime status for one declared constraint.

    Parser 只负责声明约束；这里记录运行时是否已经验证、是否还需要更好视角、
    或者是否只拒绝某个对象对。这个状态是 Phase2/3 解耦的关键。
    """

    key: str
    type: str
    satisfied: bool = False
    status: str = "pending"
    confidence: float = 0.0
    candidate_uid: str = ""
    relation_key: str = ""
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetProgress:
    target_id: str
    target_name: str
    min_count: int = 1
    accepted_candidate_uids: list[str] = field(default_factory=list)
    rejected_candidate_uids: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(set(self.accepted_candidate_uids))

    @property
    def complete(self) -> bool:
        return self.count >= max(1, int(self.min_count or 1))

    def mark_accepted(self, candidate_uid: str):
        if candidate_uid and candidate_uid not in self.accepted_candidate_uids:
            self.accepted_candidate_uids.append(candidate_uid)

    def mark_rejected(self, candidate_uid: str):
        if candidate_uid and candidate_uid not in self.rejected_candidate_uids:
            self.rejected_candidate_uids.append(candidate_uid)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["count"] = self.count
        data["complete"] = self.complete
        return data


class InstructionExecutionState:
    """State machine for executable instruction plans.

    Phase2 的 count / any-success / sequence 都在这里完成；普通 benchmark
    不创建这个状态，因此不会改变原始 STRIVE 行为。
    """

    def __init__(self, plan: InstructionPlan | None = None):
        self.plan_hash = _constraint_key(plan.as_dict() if hasattr(plan, "as_dict") else plan)
        self.active_target_index = 0
        self.completed = False
        self.target_progress: dict[str, TargetProgress] = {}
        self.constraint_status: dict[str, ConstraintStatus] = {}
        if plan is not None:
            self.bind_plan(plan)

    def bind_plan(self, plan: InstructionPlan):
        self.active_target_index = max(0, int(getattr(plan.execution, "active_target_index", 0) or 0))
        for target in plan.terminal_targets:
            if target.id not in self.target_progress:
                self.target_progress[target.id] = TargetProgress(
                    target_id=target.id,
                    target_name=target.name,
                    min_count=max(1, int(target.min_count or 1)),
                )
        for idx, constraint in enumerate(plan.constraints):
            key = f"c{idx}_{constraint.type}_{_constraint_key(constraint.as_dict())}"
            self.constraint_status.setdefault(
                key,
                ConstraintStatus(key=key, type=normalize_term(constraint.type)),
            )

    def active_target(self, plan: InstructionPlan) -> TargetQuery | None:
        terminals = plan.terminal_targets
        if not terminals:
            return None
        idx = max(0, min(self.active_target_index, len(terminals) - 1))
        return terminals[idx]

    def target_for_candidate(self, plan: InstructionPlan, candidate_label: str) -> TargetQuery | None:
        """Return the target matched by a detector label.

        Ordered mode only exposes the active target. Any-success/count mode can
        accept any terminal target whose detector terms match the candidate.
        """

        candidate = normalize_term(candidate_label)
        targets = [self.active_target(plan)] if getattr(plan.execution, "ordered", False) else plan.terminal_targets
        for target in targets:
            if target is None:
                continue
            terms = {normalize_term(x) for x in target.match_terms}
            if candidate in terms:
                return target
        return self.active_target(plan) if getattr(plan.execution, "ordered", False) else None

    def mark_candidate_rejected(self, target: TargetQuery | None, candidate_uid: str):
        if target is not None and target.id in self.target_progress:
            self.target_progress[target.id].mark_rejected(candidate_uid)

    def mark_candidate_accepted(self, plan: InstructionPlan, target: TargetQuery, candidate_uid: str) -> bool:
        """Record an accepted instance and return whether the full task is done."""

        self.bind_plan(plan)
        progress = self.target_progress[target.id]
        progress.mark_accepted(candidate_uid)

        if getattr(plan.execution, "ordered", False):
            if progress.complete and self.active_target_index < len(plan.terminal_targets) - 1:
                self.active_target_index += 1
                plan.execution.active_target_index = self.active_target_index
                return False
            self.completed = progress.complete and self.active_target_index >= len(plan.terminal_targets) - 1
            return self.completed

        mode = compact_key(getattr(plan.execution, "mode", "")) or compact_key(plan.eval_mode)
        if mode in ("all_targets_success", "all", "exhaustive") or getattr(plan.execution, "exhaustive", False):
            self.completed = all(
                self.target_progress[target.id].complete
                for target in plan.terminal_targets
                if target.id in self.target_progress
            )
        else:
            # any-success 和单目标 min_count 都落在这个分支。
            self.completed = progress.complete
        return self.completed

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "active_target_index": self.active_target_index,
            "completed": self.completed,
            "target_progress": {
                key: value.as_dict() for key, value in self.target_progress.items()
            },
            "constraint_status": {
                key: value.as_dict() for key, value in self.constraint_status.items()
            },
        }
