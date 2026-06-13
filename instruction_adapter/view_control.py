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


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer controller budget from the environment."""

    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, value)


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


def _visual_packet_from_evidence(
    evidence: dict[str, Any] | None,
    *,
    step: int | None = None,
    decision: str = "",
    reason: str = "",
    role: str = "visual_evidence",
) -> dict[str, Any]:
    """Normalize one visual evidence record for verifier prompts and logs."""

    evidence = dict(evidence or {})
    image_paths = {
        key: str(evidence.get(key) or "")
        for key in ("current_rgb_with_bbox_path", "object_crop_path", "centered_view_path")
        if evidence.get(key)
    }
    if not image_paths and not evidence.get("view_quality_facts"):
        return {}
    return {
        "step": step,
        "decision": decision,
        "reason": reason,
        "image_paths": image_paths,
        "geometry": dict(evidence.get("geometry") or {}),
        "view_quality_facts": dict(evidence.get("view_quality_facts") or {}),
        "quality": view_quality_from_evidence(evidence),
        "role": role,
    }


@dataclass
class ViewpointProposal:
    pose: list[float]
    score: float
    distance_to_target: float = 0.0
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
    semantic_satisfied: bool = False
    evidence_path: str = ""
    reason: str = ""

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
        self.pinned_visual_evidence: dict[str, Any] = {}
        self.latest_visual_evidence: dict[str, Any] = {}
        self.objective_hash = ""
        self.baseline_quality: dict[str, float] = {}
        self.proposals: list[ViewpointProposal] = []
        self.attempts: list[ViewAttempt] = []
        self.best_attempt_index: int | None = None
        self.best_visual_evidence: dict[str, Any] = {}
        self.last_selected_index: int | None = None
        self.exhausted = False

    def start(self, candidate_uid: str, objective: dict[str, Any], evidence: dict[str, Any]) -> None:
        new_hash = _objective_hash(candidate_uid, objective)
        if self.active and self.candidate_uid == candidate_uid:
            # 同一目标的 better-view 子任务不能因为 VLM 每次生成的文字略有
            # 不同就重置。否则 attempts/proposals 会被清空，控制器会反复
            # 选择同一视角，最终退回普通探索。
            self.objective.update({k: v for k, v in dict(objective or {}).items() if v not in (None, "", [])})
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

    def pin_visual_evidence(
        self,
        evidence: dict[str, Any] | None,
        *,
        step: int | None = None,
        decision: str = "",
        reason: str = "",
    ) -> None:
        """Pin the first semantically confirmed visual reference.

        Geometry proposals may be built from a noisy 3D cluster.  The final
        verifier, however, reasons over images.  Once the VLM says a candidate
        is semantically plausible, keep that first visual reference stable so
        later views can be compared against the same target identity.
        """

        packet = _visual_packet_from_evidence(
            evidence,
            step=step,
            decision=decision,
            reason=reason,
            role="stable_semantic_visual_reference",
        )
        if not packet:
            return
        packet["instruction"] = (
            "Compare future stop evidence with this VLM-confirmed visual reference. "
            "It is a target-identity reference, not automatic permission to stop."
        )
        # 首次语义确认锁定目标身份；后续更近视角不能覆盖该参照，
        # 否则污染的点云投影会把 reference 从目标漂移到支撑物或背景。
        if not self.pinned_visual_evidence:
            self.pinned_visual_evidence = packet
        self.latest_visual_evidence = packet

    def set_proposals(self, proposals: list[dict[str, Any]]) -> None:
        attempted_poses = {
            tuple(round(float(x), 3) for x in attempt.proposal.get("pose", []))
            for attempt in self.attempts
        }
        attempted_poses.update(
            tuple(round(float(x), 3) for x in proposal.pose)
            for proposal in self.proposals
            if proposal.attempted
        )
        out: list[ViewpointProposal] = []
        for item in proposals:
            pose = [float(x) for x in item.get("pose", item.get("position", []))]
            key = tuple(round(x, 3) for x in pose)
            out.append(
                ViewpointProposal(
                    pose=pose,
                    score=_safe_float(item.get("score")),
                    distance_to_target=_safe_float(item.get("distance_to_target")),
                    predicted_quality=dict(item.get("predicted_quality") or {}),
                    reason=str(item.get("reason", "")),
                    attempted=key in attempted_poses,
                )
            )
        self.proposals = sorted(out, key=self._proposal_sort_key, reverse=True)
        self.exhausted = bool(self.proposals) and all(p.attempted for p in self.proposals)

    def _proposal_sort_key(self, proposal: ViewpointProposal) -> tuple[float, float, float]:
        distance = float(proposal.distance_to_target or 0.0)
        required = self.required_stop_distance()
        if self.physical_contract_pending() and required > 0.0:
            satisfies_contract = 1.0 if 0.0 < distance <= required else 0.0
            distance_gap = max(0.0, distance - required) if distance > 0.0 else 999.0
            # 物理 stop contract 是 planner 事实，不是语义规则。若当前还没满足
            # 距离合同，优先选择能满足/接近合同且仍有可见性质量的视角。
            return (satisfies_contract, -distance_gap, proposal.score)
        return (proposal.score, -distance, 0.0)

    def next_proposal(self) -> ViewpointProposal | None:
        if self.budget_exhausted():
            self.exhausted = True
            return None
        for idx, proposal in enumerate(self.proposals):
            if not proposal.attempted:
                proposal.attempted = True
                self.last_selected_index = idx
                return proposal
        self.exhausted = True
        return None

    def record_attempt(
        self,
        step: int,
        evidence: dict[str, Any],
        decision: str,
        *,
        semantic_satisfied: bool = False,
        reason: str = "",
    ) -> ViewAttempt | None:
        """Record one verifier pass in a finite better-view optimization loop."""

        if not self.active:
            return None
        if self.last_selected_index is None:
            proposal_dict = {
                "pose": [],
                "score": 0.0,
                "distance_to_target": 0.0,
                "predicted_quality": {},
                "reason": "verifier called before a selected proposal was available",
                "attempted": True,
            }
        else:
            try:
                proposal_dict = self.proposals[self.last_selected_index].as_dict()
            except Exception:
                proposal_dict = {
                    "pose": [],
                    "score": 0.0,
                    "distance_to_target": 0.0,
                    "predicted_quality": {},
                    "reason": "selected proposal was no longer available in runtime buffer",
                    "attempted": True,
                }
        observed = view_quality_from_evidence(evidence)
        improvement = observed.get("score", 0.0) - self.baseline_quality.get("score", 0.0)
        attempt = ViewAttempt(
            step=step,
            proposal=proposal_dict,
            observed_quality=observed,
            verifier_decision=decision,
            improvement_over_baseline=float(improvement),
            semantic_satisfied=bool(semantic_satisfied),
            evidence_path=str((evidence or {}).get("current_rgb_with_bbox_path", "")),
            reason=reason,
        )
        self.attempts.append(attempt)
        if semantic_satisfied and decision in ("accept", "need_better_view", "uncertain"):
            packet = _visual_packet_from_evidence(
                evidence,
                step=step,
                decision=decision,
                reason=reason,
                role="best_available_stop_evidence",
            )
            if packet:
                current_best = _safe_float((self.best_visual_evidence.get("quality") or {}).get("score"), -1.0)
                if observed.get("score", 0.0) >= current_best:
                    self.best_visual_evidence = packet
                    self.best_attempt_index = len(self.attempts) - 1
        if self.budget_exhausted():
            self.exhausted = True
        return attempt

    def remaining_count(self) -> int:
        return sum(1 for proposal in self.proposals if not proposal.attempted)

    def required_stop_distance(self) -> float:
        """Return the active physical stop distance contract, if any."""

        try:
            value = self.objective.get("required_stop_distance")
            if value is None:
                hard = dict(self.objective.get("hard_stop_constraints") or {})
                for item in hard.get("failed", []) or hard.get("constraints", []) or []:
                    if isinstance(item, dict) and item.get("name") == "within_final_stop_distance":
                        value = item.get("required_stop_distance")
                        break
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def current_distance_to_object(self) -> float:
        """Return the current measured target distance from the view objective."""

        try:
            value = self.objective.get("current_distance_to_object")
            if value is None:
                hard = dict(self.objective.get("hard_stop_constraints") or {})
                for item in hard.get("failed", []) or hard.get("constraints", []) or []:
                    if isinstance(item, dict) and item.get("name") == "within_final_stop_distance":
                        value = item.get("current_distance_to_object")
                        break
            return max(0.0, float(value))
        except Exception:
            return 0.0

    def physical_contract_pending(self) -> bool:
        """Return whether geometry says final stop distance is still unmet."""

        required = self.required_stop_distance()
        current = self.current_distance_to_object()
        if required > 0.0 and current > required:
            return True
        hard = dict(self.objective.get("hard_stop_constraints") or {})
        if bool(hard.get("satisfied", True)):
            return False
        for item in hard.get("failed", []) or []:
            if isinstance(item, dict) and item.get("name") == "within_final_stop_distance":
                return True
        return False

    def remaining_physical_proposal_count(self) -> int:
        """Count untried proposals that can make progress on the stop contract."""

        if not self.physical_contract_pending():
            return 0
        required = self.required_stop_distance()
        current = self.current_distance_to_object()
        count = 0
        for proposal in self.proposals:
            if proposal.attempted:
                continue
            distance = float(proposal.distance_to_target or 0.0)
            if distance <= 0.0:
                continue
            if required > 0.0 and distance <= required:
                count += 1
            elif current > 0.0 and distance < current:
                count += 1
        return count

    def max_attempts(self) -> int:
        return _env_int("STRIVE_VIEW_CONTROL_MAX_ATTEMPTS", 5)

    def max_verifier_calls(self) -> int:
        return _env_int("STRIVE_VIEW_CONTROL_MAX_VERIFIER_CALLS", max(4, self.max_attempts()))

    def max_no_improvement_rounds(self) -> int:
        return _env_int("STRIVE_VIEW_CONTROL_MAX_NO_IMPROVEMENT_ROUNDS", 2)

    def no_improvement_rounds(self) -> int:
        """Count consecutive verifier attempts without meaningful view improvement."""

        threshold = self.min_required_improvement()
        rounds = 0
        for attempt in reversed(self.attempts):
            if attempt.improvement_over_baseline >= threshold:
                break
            rounds += 1
        return rounds

    def best_attempt(self) -> dict[str, Any]:
        if self.best_attempt_index is None:
            return {}
        try:
            return self.attempts[self.best_attempt_index].as_dict()
        except Exception:
            return {}

    def remaining_proposal_summaries(self, limit: int = 8) -> list[dict[str, Any]]:
        """Return compact remaining viewpoint facts for the verifier prompt."""

        out: list[dict[str, Any]] = []
        for proposal in self.proposals:
            if proposal.attempted:
                continue
            out.append({
                "pose": proposal.pose,
                "score": proposal.score,
                "distance_to_target": proposal.distance_to_target,
                "predicted_quality": proposal.predicted_quality,
                "reason": proposal.reason,
            })
            if len(out) >= limit:
                break
        return out

    def closest_remaining_proposal_distance(self) -> float | None:
        distances = [
            float(proposal.distance_to_target)
            for proposal in self.proposals
            if not proposal.attempted and float(proposal.distance_to_target or 0.0) > 0
        ]
        if not distances:
            return None
        return min(distances)

    def budget_exhausted(self) -> bool:
        """Return whether better-view control has no more useful work to run.

        这里的预算是机器人执行资源约束，不是类别规则。它限制同一已确认
        目标上的软视角优化次数，避免 VLM 无限返回 need_better_view。
        但当物理 stop contract 仍未满足且还有更近的可执行 proposal 时，
        attempt budget 不能被解释为“可以停止”。
        """

        if not self.active:
            return False
        proposal_exhausted = bool(self.proposals) and self.remaining_count() <= 0
        if self.physical_contract_pending() and self.remaining_physical_proposal_count() > 0:
            return False
        return bool(
            self.exhausted
            or proposal_exhausted
            or len(self.attempts) >= self.max_attempts()
            or len(self.attempts) >= self.max_verifier_calls()
        )

    def min_required_improvement(self) -> float:
        try:
            return float(os.getenv("STRIVE_VIEW_CONTROL_MIN_IMPROVEMENT", "0.08"))
        except Exception:
            return 0.08

    def as_context(self) -> dict[str, Any]:
        if not self.active:
            return {"active": False}
        remaining = self.remaining_count()
        remaining_physical = self.remaining_physical_proposal_count()
        budget_exhausted = self.budget_exhausted()
        no_improvement_rounds = self.no_improvement_rounds()
        progress_stalled = no_improvement_rounds >= self.max_no_improvement_rounds()
        return {
            "active": True,
            "candidate_uid": self.candidate_uid,
            "objective": self.objective,
            "pinned_relation_context": self.pinned_relation_context,
            "pinned_visual_evidence": self.pinned_visual_evidence,
            "latest_visual_evidence": self.latest_visual_evidence,
            "best_visual_evidence": self.best_visual_evidence,
            "best_attempt": self.best_attempt(),
            "objective_hash": self.objective_hash,
            "baseline_quality": self.baseline_quality,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "attempt_count": len(self.attempts),
            "max_attempts": self.max_attempts(),
            "max_verifier_calls": self.max_verifier_calls(),
            "max_no_improvement_rounds": self.max_no_improvement_rounds(),
            "no_improvement_rounds": no_improvement_rounds,
            "progress_stalled": progress_stalled,
            "remaining_feasible_proposals": remaining,
            "remaining_physical_contract_proposals": remaining_physical,
            "remaining_proposals": self.remaining_proposal_summaries(),
            "closest_remaining_proposal_distance": self.closest_remaining_proposal_distance(),
            "proposal_exhausted": bool(self.proposals) and remaining <= 0,
            "attempt_budget_exhausted": len(self.attempts) >= self.max_attempts(),
            "verifier_budget_exhausted": len(self.attempts) >= self.max_verifier_calls(),
            "physical_contract_pending": self.physical_contract_pending(),
            "budget_exhausted": budget_exhausted,
            "exhausted": budget_exhausted,
            "min_required_improvement": self.min_required_improvement(),
        }
