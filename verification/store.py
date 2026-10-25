"""Evidence records and instance-level query indexes."""
from __future__ import annotations
import hashlib
from typing import Any
from verification.contracts import VerificationRecord, VerificationResult
from verification.images import image_dependencies_valid

def instruction_hash(raw_instruction: str) -> str:
    raw = str(raw_instruction or "").strip().lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class EvidenceStore:
    """Instruction-scoped verifier memory.

    ledger 只屏蔽“某条指令下的某个对象实例”，不屏蔽 detector
    类别。这样找红色椅子时，蓝色椅子被拒绝后不会反复回来，但其它 chair
    实例仍然可以继续参与验证。
    """

    def __init__(self, task_revision: int | None = None):
        self.task_revision = task_revision
        self.records: dict[tuple, VerificationRecord] = {}
        self.history: list[VerificationRecord] = []

    def reset(self):
        self.records.clear()
        self.history.clear()

    def get(self, raw_instruction: str, candidate_uid: str, *,
            predicate_id: str = "final_stop", anchor_uid: str = "") -> VerificationRecord | None:
        """Read a judgment for the current revision and exact predicate/object pair."""
        record = self.records.get((instruction_hash(raw_instruction), self.task_revision,
                                   predicate_id, candidate_uid, anchor_uid))
        return record if record is not None and image_dependencies_valid(record.dependencies) else None

    def is_hard_rejected(self, raw_instruction: str, candidate_uid: str) -> bool:
        record = self.get(raw_instruction, candidate_uid)
        return bool(record and record.status == "rejected_hard")

    def put(
        self,
        raw_instruction: str,
        candidate_uid: str,
        result: VerificationResult,
        step: int | None = None,
        evidence_paths: list[str] | None = None,
    ) -> VerificationRecord:
        images_valid = image_dependencies_valid(result.dependencies)
        if not images_valid:
            status = "expired"
        elif result.accepted:
            status = "accepted"
        elif result.decision == "reject_candidate":
            status = "rejected_hard"
        elif result.decision == "need_better_view":
            status = "needs_better_view"
        else:
            status = "rejected_soft"
        record = VerificationRecord(
            instruction_hash=instruction_hash(raw_instruction),
            candidate_uid=candidate_uid,
            status=status,
            confidence=float(result.confidence or 0.0),
            decision=result.decision,
            failed_constraints=list(result.failed_constraints or []),
            reason=result.reason,
            step=step,
            evidence_paths=list(result.evidence_ids or evidence_paths or []),
            task_revision=result.task_revision,
            predicate_id=result.predicate_id,
            anchor_uid=result.anchor_uid,
            verdict=result.verdict if images_valid else "unknown",
            dependencies=dict(result.dependencies),
        )
        # 记录保留原始版本及对象对；新 anchor 使用独立索引。
        self.history.append(record)
        self.records[(record.instruction_hash, record.task_revision, record.predicate_id,
                      candidate_uid, record.anchor_uid)] = record
        return record

    def as_dict(self) -> dict[str, Any]:
        return {
            ":".join(map(str, key)): record.as_dict()
            for key, record in self.records.items()
        }
