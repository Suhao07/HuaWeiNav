from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _quality_from_facts(facts: dict[str, Any] | None) -> dict[str, float]:
    """Convert view facts into a generic comparable quality summary.

    这里不判断对象语义，只把“是否更清楚、更居中、更大、更不贴边”转成
    连续分数，供 view-control 判断一次移动是否真的改善了证据质量。
    """

    facts = dict(facts or {})
    center_offset = facts.get("center_offset_norm")
    border_margin = facts.get("border_margin_norm")
    area = facts.get("bbox_area_ratio")
    visible_points = facts.get("visible_projected_points")
    projection_failed = bool(facts.get("projection_failed", False))
    center_score = 0.0 if center_offset is None else max(0.0, 1.0 - _safe_float(center_offset) / 0.707)
    border_score = 0.0 if border_margin is None else max(0.0, min(1.0, _safe_float(border_margin) / 0.25))
    area_score = 0.0 if area is None else max(0.0, min(1.0, (_safe_float(area) ** 0.5) / 0.12))
    visible_score = max(0.0, min(1.0, _safe_float(visible_points) / 64.0))
    if projection_failed:
        center_score = border_score = area_score = visible_score = 0.0
    score = 0.35 * area_score + 0.25 * center_score + 0.20 * border_score + 0.20 * visible_score
    return {
        "score": float(score),
        "area_score": float(area_score),
        "center_score": float(center_score),
        "border_score": float(border_score),
        "visible_score": float(visible_score),
        "bbox_area_ratio": _safe_float(area),
        "center_offset_norm": _safe_float(center_offset),
        "border_margin_norm": _safe_float(border_margin),
    }


def view_quality_from_evidence(evidence: dict[str, Any] | None) -> dict[str, float]:
    return _quality_from_facts((evidence or {}).get("view_quality_facts") or {})


def _objective_hash(candidate_uid: str, objective: dict[str, Any]) -> str:
    raw = json.dumps([candidate_uid, objective], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class ViewpointProposal:
    pose: list[float]
    score: float
    predicted_quality: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    attempted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ViewAttempt:
    step: int
    proposal: dict[str, Any]
    observed_quality: dict[str, float]
    verifier_decision: str
    improvement_over_baseline: float
    evidence_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ViewControlState:
    """Prompt-first final verifier 的通用视角执行状态。

    状态里只保存 candidate、view objective、候选视角和质量改善历史；
    不保存任何目标类别规则。是否语义满足仍由 final verifier 决定。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.candidate_uid = ""
        self.objective: dict[str, Any] = {}
        self.pinned_relation_context: dict[str, Any] = {}
        self.objective_hash = ""
        self.baseline_quality: dict[str, float] = {}
        self.proposals: list[ViewpointProposal] = []
        self.attempts: list[ViewAttempt] = []
        self.last_selected_index: int | None = None
        self.exhausted = False

    def start(self, candidate_uid: str, objective: dict[str, Any], evidence: dict[str, Any]) -> None:
        new_hash = _objective_hash(candidate_uid, objective)
        if self.active and self.candidate_uid == candidate_uid and self.objective_hash == new_hash:
            return
        self.reset()
        self.active = True
        self.candidate_uid = candidate_uid
        self.objective = dict(objective or {})
        self.objective_hash = new_hash
        self.baseline_quality = view_quality_from_evidence(evidence)

    def pin_relation_context(self, context: dict[str, Any] | None) -> None:
        if context:
            self.pinned_relation_context = dict(context)

    def set_proposals(self, proposals: list[dict[str, Any]]) -> None:
        attempted_poses = {
            tuple(round(float(x), 3) for x in attempt.proposal.get("pose", []))
            for attempt in self.attempts
        }
        out: list[ViewpointProposal] = []
        for item in proposals:
            pose = [float(x) for x in item.get("pose", item.get("position", []))]
            key = tuple(round(x, 3) for x in pose)
            out.append(
                ViewpointProposal(
                    pose=pose,
                    score=_safe_float(item.get("score")),
                    predicted_quality=dict(item.get("predicted_quality") or {}),
                    reason=str(item.get("reason", "")),
                    attempted=key in attempted_poses,
                )
            )
        self.proposals = sorted(out, key=lambda p: p.score, reverse=True)
        self.exhausted = bool(self.proposals) and all(p.attempted for p in self.proposals)

    def next_proposal(self) -> ViewpointProposal | None:
        for idx, proposal in enumerate(self.proposals):
            if not proposal.attempted:
                proposal.attempted = True
                self.last_selected_index = idx
                return proposal
        self.exhausted = True
        return None

    def record_attempt(self, step: int, evidence: dict[str, Any], decision: str) -> ViewAttempt | None:
        if not self.active or self.last_selected_index is None:
            return None
        proposal = self.proposals[self.last_selected_index]
        observed = view_quality_from_evidence(evidence)
        improvement = observed.get("score", 0.0) - self.baseline_quality.get("score", 0.0)
        attempt = ViewAttempt(
            step=step,
            proposal=proposal.as_dict(),
            observed_quality=observed,
            verifier_decision=decision,
            improvement_over_baseline=float(improvement),
            evidence_path=str((evidence or {}).get("current_rgb_with_bbox_path", "")),
        )
        self.attempts.append(attempt)
        self.exhausted = all(p.attempted for p in self.proposals) if self.proposals else True
        return attempt

    def remaining_count(self) -> int:
        return sum(1 for proposal in self.proposals if not proposal.attempted)

    def min_required_improvement(self) -> float:
        try:
            return float(os.getenv("STRIVE_VIEW_CONTROL_MIN_IMPROVEMENT", "0.08"))
        except Exception:
            return 0.08

    def current_improvement(self, evidence: dict[str, Any]) -> float:
        current = view_quality_from_evidence(evidence)
        return current.get("score", 0.0) - self.baseline_quality.get("score", 0.0)

    def should_block_accept(self, evidence: dict[str, Any]) -> bool:
        if not self.active or self.exhausted:
            return False
        return self.current_improvement(evidence) < self.min_required_improvement() and self.remaining_count() > 0

    def as_context(self) -> dict[str, Any]:
        if not self.active:
            return {"active": False}
        return {
            "active": True,
            "candidate_uid": self.candidate_uid,
            "objective": self.objective,
            "pinned_relation_context": self.pinned_relation_context,
            "objective_hash": self.objective_hash,
            "baseline_quality": self.baseline_quality,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "remaining_feasible_proposals": self.remaining_count(),
            "exhausted": self.exhausted,
            "min_required_improvement": self.min_required_improvement(),
        }
