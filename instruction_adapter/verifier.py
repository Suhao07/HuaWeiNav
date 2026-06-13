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
    hard_constraints: dict[str, Any] = field(default_factory=dict)
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


def _nested_image_path(payload: dict[str, Any], *keys: str) -> str:
    """Read one nested image path from a JSON-like evidence packet."""

    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return str(cur or "")


def hard_stop_constraints_from_evidence(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate generic stop contracts from verifier evidence.

    These constraints are embodiment/benchmark contracts, not semantic object
    rules.  They are exposed to the VLM as structured facts and also used as a
    final consistency check so an `accept` cannot silently ignore distance.
    """

    evidence = dict(evidence or {})
    geometry = dict(evidence.get("geometry") or {})
    constraints: list[dict[str, Any]] = []

    distance = geometry.get("distance_to_object")
    required = geometry.get("required_stop_distance", geometry.get("success_distance"))
    if distance is not None and required is not None:
        try:
            distance_f = float(distance)
            required_f = float(required)
            constraints.append({
                "name": "within_final_stop_distance",
                "satisfied": distance_f <= required_f,
                "current_distance_to_object": distance_f,
                "required_stop_distance": required_f,
                "margin": required_f - distance_f,
                "reason": (
                    f"current target distance {distance_f:.3f}m "
                    f"vs required stop distance {required_f:.3f}m"
                ),
            })
        except Exception:
            constraints.append({
                "name": "within_final_stop_distance",
                "satisfied": False,
                "unknown": True,
                "reason": "distance_to_object or required_stop_distance is not numeric",
            })

    projection_failed = bool(geometry.get("projection_failed_in_final_view", False))
    constraints.append({
        "name": "target_projected_or_reference_available",
        "satisfied": not projection_failed or bool(
            (evidence.get("view_control") or {}).get("pinned_visual_evidence")
            or (evidence.get("view_control") or {}).get("best_visual_evidence")
        ),
        "projection_failed_in_final_view": projection_failed,
        "reason": "current projection exists or a pinned/best visual reference is available",
    })

    failed = [item for item in constraints if not bool(item.get("satisfied"))]
    planner_proof = (
        evidence.get("planner_infeasibility_proof")
        or evidence.get("physical_infeasibility_proof")
        or geometry.get("planner_infeasibility_proof")
        or geometry.get("physical_infeasibility_proof")
        or {}
    )
    if not isinstance(planner_proof, dict):
        planner_proof = {}
    return {
        "satisfied": len(failed) == 0,
        "constraints": constraints,
        "failed": failed,
        "source": "generic_final_stop_contract",
        "planner_infeasibility_proof": planner_proof,
        "planner_infeasible": bool(planner_proof.get("infeasible_by_geometry", False)),
    }


def _planner_hard_constraint_infeasible(hard_constraints: dict[str, Any]) -> tuple[bool, str]:
    """Return planner-owned proof that a physical stop contract is infeasible.

    The final VLM may judge instruction semantics and evidence quality, but it
    must not invent physical reachability facts.  A hard final-stop constraint
    may be bypassed only when the geometry/planning layer explicitly writes an
    infeasibility proof into the evidence packet.
    """

    hard = dict(hard_constraints or {})
    proof = hard.get("planner_infeasibility_proof") or hard.get("physical_infeasibility_proof")
    if isinstance(proof, dict) and bool(proof.get("infeasible_by_geometry")):
        reason = str(proof.get("reason", "") or proof.get("source", "") or "planner geometry proof")
        return True, reason
    if bool(hard.get("infeasible_by_geometry")) or bool(hard.get("planner_infeasible")):
        return True, str(hard.get("reason", "") or "planner geometry proof")
    for item in hard.get("failed", []) or []:
        if not isinstance(item, dict):
            continue
        if bool(item.get("infeasible_by_geometry")) or bool(item.get("planner_infeasible")):
            return True, str(item.get("reason", "") or item.get("proof", "") or "planner geometry proof")
    return False, ""


class FinalInstructionVerifier:
    """VLM verifier for original-instruction stop decisions."""

    def __init__(self, vlm: str = "cognav"):
        self.vlm = vlm

    @staticmethod
    def _view_guidance(evidence: dict[str, Any]) -> tuple[list[str], str]:
        """Build prompt-side view guidance without overriding the VLM decision.

        Final stop semantics are owned by the verifier prompt.  This helper only
        converts geometry into readable facts and a generic view objective; it
        deliberately avoids code thresholds such as "center must be < 0.2" or
        "distance must be <= 1.5m".
        """

        geometry = dict((evidence or {}).get("geometry") or {})
        facts = dict((evidence or {}).get("view_quality_facts") or {})
        view_control = dict((evidence or {}).get("view_control") or {})
        guidance: list[str] = []
        if bool(facts.get("projection_failed", False)):
            guidance.append("current target projection failed; compare against pinned/best evidence")
        center_offset = facts.get("center_offset_norm")
        border_margin = facts.get("border_margin_norm")
        area = facts.get("bbox_area_ratio")
        if center_offset is not None:
            guidance.append(f"center_offset_norm={_safe_float(center_offset):.3f}")
        if border_margin is not None:
            guidance.append(f"border_margin_norm={_safe_float(border_margin):.3f}")
        if area is not None:
            guidance.append(f"bbox_area_ratio={_safe_float(area):.5f}")
        try:
            distance_to_object = geometry.get("distance_to_object")
            required_stop_distance = geometry.get("required_stop_distance", geometry.get("success_distance"))
            if distance_to_object is not None and required_stop_distance is not None:
                guidance.append(
                    f"current_distance_to_object={float(distance_to_object):.3f}m; "
                    f"benchmark_success_distance={float(required_stop_distance):.3f}m"
                )
        except Exception:
            pass
        if view_control:
            guidance.append(
                "view_control_budget_exhausted="
                f"{bool(view_control.get('budget_exhausted', view_control.get('exhausted')))}"
            )
        preferred = (
            "Prefer the closest executable stop view that still keeps the candidate and any "
            "required relation anchor clearly visible, not clipped, and easy to verify."
        )
        return guidance, preferred

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
        evidence = dict(evidence or {})
        evidence.setdefault("hard_stop_constraints", hard_stop_constraints_from_evidence(evidence))
        payload = {
            "raw_instruction": raw_instruction,
            "instruction_plan": plan_dict or {},
            "candidate": candidate.as_dict(),
            "evidence": {k: v for k, v in evidence.items() if not str(k).endswith("_path")},
        }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        seen_image_paths: set[str] = set()
        for key in ("current_rgb_with_bbox_path", "object_crop_path", "centered_view_path"):
            block = _image_block(str(evidence.get(key) or ""))
            if block is not None:
                seen_image_paths.add(str(evidence.get(key) or ""))
                content.append({"type": "text", "text": key})
                content.append(block)
        for item in evidence.get("relation_evidence_paths", []) or []:
            block = _image_block(str(item))
            if block is not None:
                seen_image_paths.add(str(item))
                content.append({"type": "text", "text": "relation_evidence"})
                content.append(block)
        view_control = dict(evidence.get("view_control") or {})
        for label in ("pinned_visual_evidence", "latest_visual_evidence", "best_visual_evidence"):
            visual_packet = dict(view_control.get(label) or {})
            image_path = (
                _nested_image_path(visual_packet, "image_paths", "current_rgb_with_bbox_path")
                or _nested_image_path(visual_packet, "image_paths", "centered_view_path")
            )
            if not image_path or image_path in seen_image_paths:
                continue
            block = _image_block(image_path)
            if block is not None:
                seen_image_paths.add(image_path)
                content.append({
                    "type": "text",
                    "text": (
                        f"{label}: previous VLM-confirmed visual target reference; "
                        "compare target identity and view quality, but do not treat it as current stop evidence."
                    ),
                })
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
        parsed_hard_constraints = dict(getattr(parsed, "hard_constraints", {}) or {})
        hard_constraints = dict(evidence.get("hard_stop_constraints") or hard_stop_constraints_from_evidence(evidence))
        planner_infeasible, planner_infeasible_reason = _planner_hard_constraint_infeasible(hard_constraints)
        if parsed_satisfied:
            semantic_satisfied = True
        if decision == "accept" and not view_sufficient:
            decision = "need_better_view"
        if decision == "accept" and not bool(hard_constraints.get("satisfied", True)) and not planner_infeasible:
            decision = "need_better_view"
            view_sufficient = False
            parsed_satisfied = False
        view_guidance, view_guidance_goal = self._view_guidance(evidence)
        failed_constraints = list(getattr(parsed, "failed_constraints", []) or [])
        if decision == "need_better_view" and not bool(hard_constraints.get("satisfied", True)):
            for item in hard_constraints.get("failed", []) or []:
                name = str(item.get("name") or "hard_stop_constraint")
                if name not in failed_constraints:
                    failed_constraints.append(name)
        view_feedback = str(getattr(parsed, "view_feedback", "") or "")
        preferred_view_goal = str(getattr(parsed, "preferred_view_goal", "") or "")
        reason = str(getattr(parsed, "reason", "") or "")
        if view_guidance and not view_feedback and decision == "need_better_view":
            view_feedback = "; ".join(view_guidance)
        if view_guidance_goal and not preferred_view_goal and decision == "need_better_view":
            preferred_view_goal = view_guidance_goal
        view_objective = dict(getattr(parsed, "view_objective", {}) or {})
        if decision == "need_better_view" and not view_objective:
            view_objective = {
                "keep_visible_roles": ["candidate"],
                "improve_goals": [
                    "move closer when executable",
                    "keep target visible",
                    "keep relation anchor visible if required",
                    "avoid clipping",
                    "improve clarity",
                ],
                "minimum_expected_improvement": "moderate",
                "accept_if_no_better_view": False,
                "reason": view_feedback or reason or "Need stronger final stop evidence.",
            }
        if decision == "need_better_view":
            geometry = dict((evidence or {}).get("geometry") or {})
            try:
                required_stop_distance = float(
                    geometry.get("required_stop_distance", geometry.get("success_distance"))
                )
            except Exception:
                required_stop_distance = 0.0
            try:
                current_distance = float(geometry.get("distance_to_object"))
            except Exception:
                current_distance = 0.0
            if required_stop_distance > 0:
                view_objective["required_stop_distance"] = required_stop_distance
                view_objective["current_distance_to_object"] = current_distance
                goals = list(view_objective.get("improve_goals") or [])
                if "move closer while preserving visibility" not in goals:
                    goals.append("move closer while preserving visibility")
                view_objective["improve_goals"] = goals
        if decision == "need_better_view" and not bool(hard_constraints.get("satisfied", True)):
            view_objective["hard_stop_constraints"] = hard_constraints
            if not view_feedback:
                view_feedback = "Hard final-stop constraints are not yet satisfied; continue the view/approach objective."
        return VerificationResult(
            satisfied=parsed_satisfied and decision == "accept" and view_sufficient,
            decision=decision,
            confidence=confidence,
            semantic_satisfied=semantic_satisfied,
            view_sufficient_for_stop=view_sufficient,
            hard_constraints={
                **hard_constraints,
                "vlm_report": parsed_hard_constraints,
                "planner_infeasible": planner_infeasible,
                "planner_infeasible_reason": planner_infeasible_reason,
            },
            satisfied_constraints=list(getattr(parsed, "satisfied_constraints", []) or []),
            failed_constraints=failed_constraints,
            view_feedback=view_feedback,
            preferred_view_goal=preferred_view_goal,
            view_objective=view_objective,
            reason=reason,
            diagnostics={
                "source": "vlm",
                "model_provider": self.vlm,
                "view_guidance": {
                    "facts": view_guidance,
                    "preferred_view_goal": view_guidance_goal,
                    "enforced_by_code": False,
                },
                "hard_stop_constraints": {
                    **hard_constraints,
                    "vlm_report": parsed_hard_constraints,
                    "planner_infeasible": planner_infeasible,
                    "planner_infeasible_reason": planner_infeasible_reason,
                },
            },
        )
