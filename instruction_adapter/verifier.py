from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from verification.images import image_input
from typing import Any

import numpy as np

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import FINAL_VERIFY
from prompting.schemas import HAS_PYDANTIC, affordance_verification_response_model
from prompting.templates import FINAL_VERIFIER_PROMPT

from .ontology import normalize_term
from verification.contracts import CandidateInstance, VerificationResult
from verification.store import EvidenceStore, instruction_hash


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
    """Return native instance identity, with a geometry key for legacy inputs.

    Native map objects expose a stable instance uid. Legacy inputs without that
    field use the quantized geometry signature for compatibility.
    """

    if getattr(obj, "uid", None):
        return str(obj.uid)
    detector_label = normalize_term(label or getattr(obj, "tag", getattr(obj, "label", "object"))) or "object"
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
    label = normalize_term(getattr(obj, "label", getattr(obj, "tag", canonical_label))) or normalize_term(canonical_label)
    uid = candidate_uid_from_object(obj, label)
    position = getattr(obj, "position", None)
    if position is None:
        position = getattr(obj, "bbox3d_center", None)
    centroid = np.array(position if position is not None else [], dtype=float).reshape(-1)[:3].tolist()
    bbox = getattr(obj, "bbox2d_xyxy", getattr(obj, "bbox", None))
    try:
        values = np.asarray(bbox, dtype=float).reshape(-1) if bbox is not None else np.empty(0)
        # A missing ROS object box is unknown, not a one-element NaN box.
        # Keep only finite, non-degenerate xyxy geometry; fresh capture evidence
        # remains authoritative for final visual verification.
        valid = (values.size == 4 and np.isfinite(values).all()
                 and values[2] > values[0] and values[3] > values[1])
        bbox = values.tolist() if valid else []
    except (TypeError, ValueError):
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



def _nested_image_path(payload: dict[str, Any], *keys: str) -> str:
    """Read one nested image path from a JSON-like evidence packet."""

    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return str(cur or "")


def _candidate_observation_id(identity, bbox, candidate_uid, revision):
    """Only a located candidate in a frame from this task revision can support a predicate."""
    if (not isinstance(identity, dict) or identity.get("candidate_uid") != candidate_uid
            or identity.get("task_revision") != revision
            or not isinstance(identity.get("observation_id"), str)
            or not identity["observation_id"].strip()):
        return None
    try:
        box = np.asarray(bbox, dtype=float).reshape(-1)
        if (box.size == 4 and np.isfinite(box).all()
                and box[2] > box[0] and box[3] > box[1]):
            return identity["observation_id"]
    except (TypeError, ValueError):
        pass
    return None


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
    if required is not None:
        required_f = float(required)
        distance_f = None if distance is None else float(distance)
        constraints.append({
            "name": "within_final_stop_distance",
            "satisfied": distance_f is not None and distance_f <= required_f,
            "current_distance_to_object": distance_f,
            "required_stop_distance": required_f,
            "margin": None if distance_f is None else required_f - distance_f,
            "reason": f"current surface distance {distance_f}m; required {required_f}m",
        })

    projection_failed = bool(geometry.get("projection_failed_in_final_view", False))
    constraints.append({
        "name": "target_projected_or_reference_available",
        "satisfied": not projection_failed,
        "projection_failed_in_final_view": projection_failed,
        "reason": "Current target observation supplies final-stop geometry.",
    })
    if 'surface_match' in geometry:
        constraints.append({
            "name": "current_instance_surface_correspondence",
            "satisfied": bool(geometry['surface_match']) and
                geometry['observation_identity'] == evidence['observation_identity'],
            "reason": geometry['surface_status'],
        })

    failed = [item for item in constraints if not item["satisfied"]]
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

    def __init__(self, vlm: str = "cognav", evidence_dir: str = "outputs/evidence"):
        self.vlm = vlm
        self.evidence_dir = Path(evidence_dir)
        from verification.store import EvidenceStore
        self.evidence_store = EvidenceStore()

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
                    f"predicted_instance_stop_distance={float(required_stop_distance):.3f}m"
                )
        except Exception:
            pass
        preferred = "Resolve the remaining task predicate or stop-distance requirement with a new observation of the candidate."
        return guidance, preferred

    def _fallback(self, reason: str) -> VerificationResult:
        # Missing validation is unknown, never positive or negative evidence.
        return VerificationResult(
            satisfied=False,
            decision="uncertain",
            confidence=0.0,
            semantic_satisfied=False,
            view_sufficient_for_stop=False,
            reason=f"final verifier unavailable: {reason}",
            diagnostics={"unavailable": reason, "verdict": "unknown"},
        )

    @staticmethod
    def _visual_task(plan, evidence):
        """Project active user semantics while excluding grounding and execution."""
        semantic = plan.get("diagnostics", {}).get("task_spec", plan)
        targets = semantic.get("targets", [])
        terminal = [target for target in targets if target.get("terminal", True)]
        if not terminal:
            return {}
        target_id = evidence.get("target_id")
        if target_id is None:
            if len(terminal) != 1:
                raise ValueError("Visual verification of multiple targets requires an explicit target_id")
            active = terminal[0]
        else:
            active = next(target for target in terminal if target["id"] == target_id)
        predicates = [condition for condition in semantic.get("constraints", [])
                      if condition.get("subject", "") in ("", active["id"], active["name"])]
        # 默认 verifier=planner 不代表最终执行归属：ConstraintEvaluator 会
        # 显式把 room/attribute 等约束委派到 declared_*。沿用该委派事实。
        delegated = (list(evidence.get("declared_room_constraints", []))
                     + list(evidence.get("declared_attribute_constraints", [])))
        visual, domain = [], []
        for condition in predicates:
            projected = {key: condition[key] for key in (
                "id", "type", "subject", "relation", "object", "value", "hardness", "verifier") if key in condition}
            projected.setdefault("verifier", "planner")
            is_delegated = any(all(condition.get(key) == item.get(key) for key in (
                "type", "subject", "relation", "object", "value")) for item in delegated)
            if projected["verifier"] == "vlm" or is_delegated:
                visual.append(projected)
            else:
                domain.append(projected)
        anchors = {condition.get("object") for condition in visual}
        return deepcopy({
            "active_target": {key: active[key] for key in ("id", "name", "attributes", "demand") if key in active},
            "visual_predicates": visual,
            "domain_predicates": domain,
            "referenced_targets": [{key: target[key] for key in ("id", "name", "attributes") if key in target}
                                   for target in targets if target.get("id") in anchors or target.get("name") in anchors],
        })

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

        plan_dict = (instruction_plan.as_dict() if hasattr(instruction_plan, "as_dict") else instruction_plan) or {}
        semantic = plan_dict.get("diagnostics", {}).get("task_spec", plan_dict)
        revision = semantic.get("revision", 1)
        evidence = dict(evidence or {})
        evidence.setdefault("hard_stop_constraints", hard_stop_constraints_from_evidence(evidence))
        # V(I, task) 与 G(pose, map) 独立求值；预测标签和接近距离不参与视觉判定。
        payload = {"raw_instruction": raw_instruction,
                   "task": self._visual_task(plan_dict or {}, evidence),
                   "candidate": {"uid": candidate.uid}}
        affordance_ids = [predicate.get("id") for predicate in payload["task"].get("visual_predicates", [])
                          if predicate.get("type") == "affordance" and predicate.get("hardness", "hard") == "hard"]
        response_model = affordance_verification_response_model(affordance_ids)
        sent_current_ids: set[str] = set()
        sent_candidate_ids: set[str] = set()
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        seen_image_paths: set[str] = set()
        image_refs = {}
        image_dependencies = {}

        def load_image(path, missing_ok=False):
            if not path or (missing_ok and not Path(path).is_file()):
                return None
            block, reference, digest = image_input(path, self.evidence_dir)
            image_refs[path] = reference
            image_dependencies["image:" + reference] = digest
            return block

        for key in ("current_rgb_with_bbox_path", "object_crop_path", "centered_view_path"):
            path = str(evidence.get(key) or "")
            if path in seen_image_paths:
                continue
            bbox = (evidence.get("object_crop_bbox_xyxy") if key == "object_crop_path"
                    else evidence.get("geometry", {}).get("bbox_xyxy"))
            observation_id = _candidate_observation_id(
                evidence.get("observation_identity"), bbox, candidate.uid, revision)
            if affordance_ids and observation_id is None:
                continue
            block = load_image(path, missing_ok=bool(affordance_ids))
            if block is not None:
                seen_image_paths.add(path)
                if observation_id is not None:
                    sent_current_ids.add(observation_id)
                    sent_candidate_ids.add(observation_id)
                content.append({"type": "text", "text": json.dumps({
                    "role": "current_candidate_observation", "image_kind": key,
                    "observation_identity": evidence.get("observation_identity", {}),
                    "bbox_xyxy_in_this_image": bbox,
                }, ensure_ascii=False)})
                content.append(block)
        for item in evidence.get("relation_evidence_paths", []) or []:
            path = str(item)
            if path in seen_image_paths:
                continue
            block = load_image(path)
            if block is not None:
                seen_image_paths.add(path)
                content.append({"type": "text", "text": "relation observation; identify visible entities from this image"})
                content.append(block)
        view_control = dict(evidence.get("view_control") or {})
        excluded_history = []
        current_identity = evidence.get("observation_identity", {})
        for label in ("pinned_visual_evidence", "latest_visual_evidence", "best_visual_evidence"):
            visual_packet = dict(view_control.get(label) or {})
            image_path = (
                _nested_image_path(visual_packet, "image_paths", "current_rgb_with_bbox_path")
                or _nested_image_path(visual_packet, "image_paths", "centered_view_path")
            )
            if not image_path or image_path in seen_image_paths:
                continue
            identity = visual_packet.get("observation_identity", {})
            if (not {"candidate_uid", "task_revision", "observation_id"}.issubset(identity)
                    or identity["candidate_uid"] != candidate.uid
                    or identity["task_revision"] != current_identity.get("task_revision")):
                excluded_history.append({"image_path": image_path, "reason": "observation_binding_mismatch"})
                continue
            observation_id = _candidate_observation_id(
                identity, visual_packet.get("geometry", {}).get("bbox_xyxy"), candidate.uid, revision)
            if affordance_ids and observation_id is None:
                excluded_history.append({"image_path": image_path, "reason": "observation_binding_mismatch"})
                continue
            block = load_image(image_path, missing_ok=bool(affordance_ids))
            if block is not None:
                seen_image_paths.add(image_path)
                if observation_id is not None:
                    sent_candidate_ids.add(observation_id)
                content.append({"type": "text", "text": json.dumps({
                    "role": "earlier_candidate_observation",
                    "observation_identity": identity,
                    "bbox_xyxy_in_this_image": visual_packet.get("geometry", {}).get("bbox_xyxy"),
                }, ensure_ascii=False)})
                content.append(block)
        # 与候选共享表面的其它实例：同一物理物体可能被拆成多个实例，只给图像不给其标签。
        for item in evidence.get("overlapping_instance_observations", []) or []:
            block = load_image(item["image_path"])
            content.append({"type": "text", "text": json.dumps({
                "role": "overlapping_instance_observation",
                "candidate_surface_fraction_on_this_instance": item["candidate_surface_fraction"],
                "bbox_xyxy_in_this_image": item["bbox_xyxy"],
            }, ensure_ascii=False)})
            content.append(block)

        reference_path = _nested_image_path(
            payload, "task", "active_target", "attributes", "goal_reference_image")
        block = load_image(reference_path)
        if block is not None:
            content.append({"type": "text", "text": json.dumps({
                "role": "goal_reference",
                "instruction": (
                    "This is the user's goal reference image, not a current candidate observation. "
                    "Accept only if visual evidence establishes that the candidate is the same physical "
                    "instance as the target in this image; matching the object category alone is insufficient."
                ),
            }, ensure_ascii=False)})
            content.append(block)

        if affordance_ids and not sent_current_ids:
            image_refs.clear()
            image_dependencies.clear()
            sent_candidate_ids.clear()
            parsed = response_model(
                decision="need_better_view", view_sufficient_for_stop=False,
                reason="Hard affordances require a current candidate-bound image.",
                affordance_judgments={uid: {
                    "verdict": "unknown", "reason": "No current candidate-bound image was sent.",
                    "supporting_observation_ids": [],
                } for uid in affordance_ids})
        else:
            # Provider errors propagate; only exact visual judgments are cached.
            client, model = get_client_and_model(self.vlm)
            from llm_utils.structured_output import request_fingerprint
            from llm_utils.lvlm_call_tracker import record_cache_hit
            messages = [{"role": "system", "content": FINAL_VERIFIER_PROMPT},
                        {"role": "user", "content": content}]
            key = request_fingerprint(model, [messages, revision, evidence.get("observation_identity"),
                evidence.get("dependencies", {})], response_model, provider=self.vlm)
            reuse = os.getenv("STRIVE_EVIDENCE_REUSE", "0") == "1"
            parsed = self.evidence_store.visual_judgments.get(key) if reuse else None
            if parsed is None:
                completion = client.beta.chat.completions.parse(
                    model=model, messages=messages, response_format=response_model,
                    trace_label=FINAL_VERIFY.trace_label)
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Final verifier response has no structured judgment")
                if reuse:
                    self.evidence_store.visual_judgments[key] = parsed
            else:
                record_cache_hit("final_visual_judgment")

        affordance_judgments = {}
        if affordance_ids:
            parsed = response_model.model_validate(parsed.model_dump())
            affordance_judgments = parsed.affordance_judgments.model_dump()
            for judgment in affordance_judgments.values():
                supporting = judgment["supporting_observation_ids"]
                if judgment["verdict"] != "unknown" and (
                        not sent_current_ids or not supporting
                        or not set(supporting).issubset(sent_candidate_ids) or not judgment["reason"].strip()):
                    judgment["reported_verdict"] = judgment["verdict"]
                    judgment["verdict"] = "unknown"
                    judgment["validation_reason"] = "Missing visual reason or actually-sent candidate-bound observation support."

        decision = normalize_term(getattr(parsed, "decision", "")).replace(" ", "_")
        allowed = {"accept", "reject_candidate", "need_better_view", "need_relation_check", "uncertain"}
        if decision not in allowed:
            raise ValueError(f"Unsupported final verifier decision: {decision!r}")
        confidence = max(0.0, min(1.0, _safe_float(getattr(parsed, "confidence", 0.0))))
        semantic_satisfied = bool(getattr(parsed, "semantic_satisfied", False))
        view_sufficient = bool(getattr(parsed, "view_sufficient_for_stop", True))
        parsed_satisfied = bool(getattr(parsed, "satisfied", False))
        affordance_false = [uid for uid, judgment in affordance_judgments.items() if judgment["verdict"] == "false"]
        affordance_unknown = [uid for uid, judgment in affordance_judgments.items() if judgment["verdict"] == "unknown"]
        if affordance_false or affordance_unknown:
            semantic_satisfied = parsed_satisfied = False
            if affordance_false:
                decision = "reject_candidate"
            else:
                decision = "uncertain" if decision == "uncertain" else "need_better_view"
                view_sufficient = False
        hard_constraints = dict(evidence.get("hard_stop_constraints") or hard_stop_constraints_from_evidence(evidence))
        planner_infeasible, planner_infeasible_reason = _planner_hard_constraint_infeasible(hard_constraints)
        if decision == "accept" and not (semantic_satisfied and view_sufficient):
            decision = "need_better_view"
        if decision == "accept" and not bool(hard_constraints.get("satisfied", True)):
            decision = "need_better_view"
            parsed_satisfied = False
        view_guidance, view_guidance_goal = self._view_guidance(evidence)
        failed_constraints = list(getattr(parsed, "failed_constraints", []) or [])
        satisfied_constraints = list(getattr(parsed, "satisfied_constraints", []) or [])
        if affordance_ids:
            failed_constraints = [uid for uid in failed_constraints if uid not in affordance_ids] + affordance_false
            satisfied_constraints = [uid for uid in satisfied_constraints if uid not in affordance_ids] + [
                uid for uid, judgment in affordance_judgments.items() if judgment["verdict"] == "true"]
        if decision == "need_better_view" and not bool(hard_constraints.get("satisfied", True)):
            for item in hard_constraints.get("failed", []) or []:
                name = str(item.get("name") or "hard_stop_constraint")
                if name not in failed_constraints:
                    failed_constraints.append(name)
        view_feedback = str(getattr(parsed, "view_feedback", "") or "")
        preferred_view_goal = str(getattr(parsed, "preferred_view_goal", "") or "")
        reason = str(getattr(parsed, "reason", "") or "")
        if affordance_false or affordance_unknown:
            reason = "; ".join(filter(None, [reason, "Hard affordances not established: " + ", ".join(
                affordance_false + affordance_unknown)]))
        if view_guidance and not view_feedback and decision == "need_better_view":
            view_feedback = "; ".join(view_guidance)
        if view_guidance_goal and not preferred_view_goal and decision == "need_better_view":
            preferred_view_goal = view_guidance_goal
        view_objective = parsed.view_objective.model_dump(exclude_defaults=True)
        if decision == "need_better_view" and not view_objective:
            view_objective = {
                "keep_visible_roles": ["candidate"],
                "improve_goals": [
                    "resolve the unverified target or relation predicate",
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
            if required_stop_distance > 0 and current_distance > required_stop_distance:
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
            satisfied=parsed_satisfied and decision == "accept" and semantic_satisfied and view_sufficient,
            target_uid=candidate.uid,
            evidence_ids=sorted(set(image_refs.values())),
            dependencies={**evidence.get("dependencies", {}), **image_dependencies,
                          "observation_identity": json.dumps(evidence.get("observation_identity"), sort_keys=True)},
            task_revision=revision,
            decision=decision,
            confidence=confidence,
            semantic_satisfied=semantic_satisfied,
            view_sufficient_for_stop=view_sufficient,
            hard_constraints={
                **hard_constraints,
                "planner_infeasible": planner_infeasible,
                "planner_infeasible_reason": planner_infeasible_reason,
            },
            satisfied_constraints=satisfied_constraints,
            failed_constraints=failed_constraints,
            view_feedback=view_feedback,
            preferred_view_goal=preferred_view_goal,
            view_objective=view_objective,
            reason=reason,
            diagnostics={
                "source": "missing_candidate_image" if affordance_ids and not sent_current_ids else "vlm",
                "image_sources": image_refs,
                "excluded_history": excluded_history,
                **({"affordance_judgments": affordance_judgments,
                    "sent_current_observation_ids": sorted(sent_current_ids),
                    "sent_candidate_observation_ids": sorted(sent_candidate_ids)} if affordance_ids else {}),
                "model_provider": self.vlm,
                "view_guidance": {
                    "facts": view_guidance,
                    "preferred_view_goal": view_guidance_goal,
                    "enforced_by_code": False,
                },
                "hard_stop_constraints": {
                    **hard_constraints,
                    "planner_infeasible": planner_infeasible,
                    "planner_infeasible_reason": planner_infeasible_reason,
                },
            },
        )
