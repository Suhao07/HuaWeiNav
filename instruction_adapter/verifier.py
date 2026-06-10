from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import FINAL_VERIFY
from prompting.schemas import HAS_PYDANTIC, ParsedVerification
from prompting.templates import FINAL_VERIFIER_PROMPT

from .ontology import normalize_term


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
    instruction_hash: str
    candidate_uid: str
    status: str
    confidence: float = 0.0
    decision: str = ""
    failed_constraints: list[str] = field(default_factory=list)
    reason: str = ""
    step: int | None = None
    evidence_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    satisfied: bool
    decision: str
    confidence: float = 0.0
    semantic_satisfied: bool = False
    view_sufficient_for_stop: bool = True
    satisfied_constraints: list[str] = field(default_factory=list)
    failed_constraints: list[str] = field(default_factory=list)
    view_feedback: str = ""
    preferred_view_goal: str = ""
    view_objective: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def instruction_hash(raw_instruction: str) -> str:
    raw = str(raw_instruction or "").strip().lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy().item()
        return float(value)
    except Exception:
        return default


def candidate_uid_from_object(obj: Any, label: str = "") -> str:
    """Build a stable-enough instance key from geometry.

    STRIVE's mapped ObjectNode does not expose a permanent object id.  We use a
    quantized geometry signature so the same physical object is skipped after a
    hard verifier rejection, while another object of the same detector class can
    still be considered.
    """

    detector_label = normalize_term(label or getattr(obj, "tag", "object")) or "object"
    center = np.array(getattr(obj, "position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)[:3]
    if center.size < 3:
        center = np.pad(center, (0, 3 - center.size))
    size = np.zeros(3, dtype=float)
    try:
        pts = getattr(obj, "pcd").point.positions.cpu().numpy()
        if len(pts) > 0:
            size = np.max(pts, axis=0)[:3] - np.min(pts, axis=0)[:3]
    except Exception:
        pass
    # 25cm buckets tolerate small map updates but keep nearby distinct objects apart.
    center_key = tuple(np.round(center / 0.25).astype(int).tolist())
    size_key = tuple(np.round(size / 0.25).astype(int).tolist())
    raw = json.dumps([detector_label, center_key, size_key], sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{detector_label}:{digest}"


def candidate_from_object(
    obj: Any,
    canonical_label: str = "",
    step: int | None = None,
) -> CandidateInstance:
    label = normalize_term(getattr(obj, "tag", canonical_label)) or normalize_term(canonical_label)
    uid = candidate_uid_from_object(obj, label)
    centroid = np.array(getattr(obj, "position", []), dtype=float).reshape(-1)[:3].tolist()
    bbox = getattr(obj, "bbox", None)
    try:
        bbox = np.array(bbox, dtype=float).reshape(-1).tolist()
    except Exception:
        bbox = []
    return CandidateInstance(
        uid=uid,
        detector_label=label,
        canonical_label=normalize_term(canonical_label) or label,
        centroid=[float(x) for x in centroid],
        bbox_2d=[float(x) for x in bbox],
        confidence=_safe_float(getattr(obj, "confidence", 0.0)),
        step=step,
    )


class VerificationLedger:
    """Instruction-scoped verifier memory.

    ledger 只屏蔽“某条指令下的某个对象实例”，不屏蔽 detector
    类别。这样找红色椅子时，蓝色椅子被拒绝后不会反复回来，但其它 chair
    实例仍然可以继续参与验证。
    """

    def __init__(self):
        self.records: dict[tuple[str, str], VerificationRecord] = {}

    def reset(self):
        self.records.clear()

    def get(self, raw_instruction: str, candidate_uid: str) -> VerificationRecord | None:
        return self.records.get((instruction_hash(raw_instruction), candidate_uid))

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
        if result.satisfied or result.decision == "accept":
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
            evidence_paths=list(evidence_paths or []),
        )
        self.records[(record.instruction_hash, candidate_uid)] = record
        return record

    def as_dict(self) -> dict[str, Any]:
        return {
            f"{inst}:{uid}": record.as_dict()
            for (inst, uid), record in self.records.items()
        }


def _image_block(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


class FinalInstructionVerifier:
    """VLM verifier for original-instruction stop decisions."""

    def __init__(self, vlm: str = "cognav"):
        self.vlm = vlm

    @staticmethod
    def _view_guard(evidence: dict[str, Any]) -> tuple[bool, list[str], str]:
        """Generic geometry guard for final stop views.

        这不是目标语义规则，而是相机停止证据的硬约束：目标不能完全投影失败，
        也不应在最终 stop 图里过度贴边、过小或偏离中心。阈值全部可配置，
        用于补足 VLM 偶尔把“语义正确但视角很差”的图直接 accept 的问题。
        """

        mode = os.getenv("STRIVE_FINAL_VIEW_GUARD", "1").lower()
        if mode in ("0", "false", "no", "off"):
            return True, [], ""
        facts = dict((evidence or {}).get("view_quality_facts") or {})
        failures: list[str] = []
        if bool(facts.get("projection_failed", False)):
            failures.append("target projection failed in final view")
        center_offset = facts.get("center_offset_norm")
        border_margin = facts.get("border_margin_norm")
        area = facts.get("bbox_area_ratio")
        try:
            max_center_offset = float(os.getenv("STRIVE_FINAL_VIEW_MAX_CENTER_OFFSET", "0.35"))
        except Exception:
            max_center_offset = 0.35
        try:
            min_border_margin = float(os.getenv("STRIVE_FINAL_VIEW_MIN_BORDER_MARGIN", "0.08"))
        except Exception:
            min_border_margin = 0.08
        try:
            min_area = float(os.getenv("STRIVE_FINAL_VIEW_MIN_BBOX_AREA", "0.003"))
        except Exception:
            min_area = 0.003
        try:
            if center_offset is not None and float(center_offset) > max_center_offset:
                failures.append("target is too far from the image center")
        except Exception:
            pass
        try:
            if border_margin is not None and float(border_margin) < min_border_margin:
                failures.append("target is too close to the image border")
        except Exception:
            pass
        try:
            if area is not None and float(area) < min_area:
                failures.append("target projection is too small for final stop evidence")
        except Exception:
            pass
        if not failures:
            return True, [], ""
        preferred = (
            "Move to a viewpoint where the candidate remains visible, occupies a clearer area, "
            "and is closer to the image center; keep any required relation anchor visible."
        )
        return False, failures, preferred

    def _fallback(self, reason: str) -> VerificationResult:
        # When LLM is intentionally unavailable, preserve baseline behavior.
        return VerificationResult(
            satisfied=True,
            decision="accept",
            confidence=0.0,
            semantic_satisfied=True,
            view_sufficient_for_stop=True,
            reason=f"final verifier fallback accepted: {reason}",
            diagnostics={"fallback": reason},
        )

    def verify(
        self,
        raw_instruction: str,
        instruction_plan: Any,
        candidate: CandidateInstance,
        evidence: dict[str, Any],
    ) -> VerificationResult:
        verifier_mode = os.getenv("STRIVE_FINAL_VERIFIER", "auto").lower()
        if verifier_mode in ("0", "false", "no", "off"):
            return self._fallback("disabled")
        if verifier_mode == "auto" and instruction_plan is None:
            return self._fallback("no_instruction_plan")
        if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on"):
            return self._fallback("llm_offline")
        if not HAS_PYDANTIC:
            return self._fallback("pydantic_unavailable")

        plan_dict = instruction_plan.as_dict() if hasattr(instruction_plan, "as_dict") else instruction_plan
        payload = {
            "raw_instruction": raw_instruction,
            "instruction_plan": plan_dict or {},
            "candidate": candidate.as_dict(),
            "evidence": {k: v for k, v in evidence.items() if not str(k).endswith("_path")},
        }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        for key in ("current_rgb_with_bbox_path", "object_crop_path", "centered_view_path"):
            block = _image_block(str(evidence.get(key) or ""))
            if block is not None:
                content.append({"type": "text", "text": key})
                content.append(block)
        for item in evidence.get("relation_evidence_paths", []) or []:
            block = _image_block(str(item))
            if block is not None:
                content.append({"type": "text", "text": "relation_evidence"})
                content.append(block)

        try:
            client, model = get_client_and_model(self.vlm)
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": FINAL_VERIFIER_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=ParsedVerification,
                trace_label=FINAL_VERIFY.trace_label,
            )
            parsed = completion.choices[0].message.parsed
        except Exception as exc:
            return self._fallback(f"vlm_failed: {exc}")

        decision = normalize_term(getattr(parsed, "decision", "")).replace(" ", "_")
        allowed = {"accept", "reject_candidate", "need_better_view", "need_relation_check", "uncertain"}
        if decision not in allowed:
            decision = "accept" if bool(getattr(parsed, "satisfied", False)) else "uncertain"
        confidence = max(0.0, min(1.0, _safe_float(getattr(parsed, "confidence", 0.0))))
        semantic_satisfied = bool(getattr(parsed, "semantic_satisfied", False))
        view_sufficient = bool(getattr(parsed, "view_sufficient_for_stop", True))
        parsed_satisfied = bool(getattr(parsed, "satisfied", False))
        if parsed_satisfied:
            semantic_satisfied = True
        if decision == "accept" and not view_sufficient:
            decision = "need_better_view"
        view_guard_ok, view_guard_failures, view_guard_goal = self._view_guard(evidence)
        if semantic_satisfied and decision == "accept" and not view_guard_ok:
            decision = "need_better_view"
            view_sufficient = False
            parsed_satisfied = False
        failed_constraints = list(getattr(parsed, "failed_constraints", []) or [])
        for failure in view_guard_failures:
            if failure not in failed_constraints:
                failed_constraints.append(failure)
        view_feedback = str(getattr(parsed, "view_feedback", "") or "")
        preferred_view_goal = str(getattr(parsed, "preferred_view_goal", "") or "")
        reason = str(getattr(parsed, "reason", "") or "")
        if view_guard_failures and not view_feedback:
            view_feedback = "; ".join(view_guard_failures)
        if view_guard_goal and not preferred_view_goal:
            preferred_view_goal = view_guard_goal
        view_objective = dict(getattr(parsed, "view_objective", {}) or {})
        if decision == "need_better_view" and not view_objective:
            view_objective = {
                "keep_visible_roles": ["candidate"],
                "improve_goals": ["clarity", "centering", "scale"],
                "minimum_expected_improvement": "moderate",
                "accept_if_no_better_view": False,
                "reason": view_feedback or reason or "Need stronger final stop evidence.",
            }
        if view_guard_failures:
            suffix = "Generic view guard requested a better final stop view: " + "; ".join(view_guard_failures)
            reason = f"{reason} {suffix}".strip()
        return VerificationResult(
            satisfied=parsed_satisfied and decision == "accept" and view_sufficient,
            decision=decision,
            confidence=confidence,
            semantic_satisfied=semantic_satisfied,
            view_sufficient_for_stop=view_sufficient,
            satisfied_constraints=list(getattr(parsed, "satisfied_constraints", []) or []),
            failed_constraints=failed_constraints,
            view_feedback=view_feedback,
            preferred_view_goal=preferred_view_goal,
            view_objective=view_objective,
            reason=reason,
            diagnostics={
                "source": "vlm",
                "model_provider": self.vlm,
                "view_guard": {
                    "ok": view_guard_ok,
                    "failures": view_guard_failures,
                    "enabled": os.getenv("STRIVE_FINAL_VIEW_GUARD", "1").lower() not in ("0", "false", "no", "off"),
                },
            },
        )
