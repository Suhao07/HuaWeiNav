"""Task-owned progress for an immutable instruction revision."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
from instruction.contracts import TaskSpec, TargetSpec


@dataclass
class TargetProgress:
    """Counts distinct accepted instances for one semantic target."""
    target_id: str
    target_name: str
    min_count: int
    accepted_candidate_uids: list[str] = field(default_factory=list)
    rejected_candidate_uids: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Return N_i = |A_i|, the number of distinct accepted instances."""
        return len(set(self.accepted_candidate_uids))

    @property
    def complete(self) -> bool:
        """Return whether the required instance count has been reached."""
        return self.count >= self.min_count

    def as_dict(self) -> dict:
        """Serialize progress and derived counts."""
        return {**asdict(self), "count": self.count, "complete": self.complete}


class TaskProgress:
    """Progress bound to one immutable TaskSpec.

    Args:
        task: Semantic contract for this task revision.

    同版本的顺序和数量由 task 决定；检测器词表和搜索策略独立演化。
    """
    def __init__(self, task: TaskSpec):
        self.task = task
        self.active_target_index = 0
        self.completed = False
        self.mode = "search"
        self.pending_verified_pair: dict[str, Any] = {}
        self.target_progress = {
            t.id: TargetProgress(t.id, t.name, t.min_count)
            for t in task.targets if t.terminal
        }

    def active_target(self) -> TargetSpec:
        """Return the current semantic target in instruction order."""
        terminals = [t for t in self.task.targets if t.terminal]
        return terminals[self.active_target_index]

    def pending_pair_candidate_uid(self) -> str:
        return str((self.pending_verified_pair or {}).get("candidate_uid", ""))

    def as_dict(self) -> dict:
        """Serialize mutable progress with its authoritative task revision."""
        return {
            "task_revision": self.task.revision,
            "active_target_index": self.active_target_index,
            "completed": self.completed,
            "mode": self.mode,
            "pending_verified_pair": self.pending_verified_pair,
            "target_progress": {key: value.as_dict() for key, value in self.target_progress.items()},
        }
