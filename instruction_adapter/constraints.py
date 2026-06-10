from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts import Constraint, InstructionPlan, TargetQuery
from .execution import InstructionExecutionState
from .ontology import normalize_term
from .relation_verifier import DynamicRelationService
from .semantic_edges import SemanticEdge, normalize_relation
from .spatial_graph import InstructionSpatialGraph
from .verifier import CandidateInstance, VerificationResult, candidate_from_object, instruction_hash
from llm_utils.lvlm_call_tracker import counts as lvlm_counts


RELATION_CONSTRAINT_TYPES = {
    "spatial",
    "relation",
    "object_relation",
    "co_occurrence",
}


@dataclass
class ConstraintEvaluation:
    satisfied: bool
    decision: str = "pass"
    confidence: float = 0.0
    failed_constraints: list[str] = field(default_factory=list)
    satisfied_constraints: list[str] = field(default_factory=list)
    relation_edges: list[SemanticEdge] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["relation_edges"] = [edge.as_dict() for edge in self.relation_edges]
        return data


def _target_names(target: TargetQuery) -> set[str]:
    return {normalize_term(item) for item in target.match_terms if normalize_term(item)}


def _constraint_subject_matches(constraint: Constraint, target: TargetQuery, candidate: CandidateInstance) -> bool:
    subject = normalize_term(constraint.subject)
    if not subject:
        return True
    if subject == normalize_term(target.id) or subject in _target_names(target):
        return True
    return subject in {
        normalize_term(candidate.detector_label),
        normalize_term(candidate.canonical_label),
    }


def _constraint_object_terms(constraint: Constraint) -> set[str]:
    terms = []
    if constraint.object:
        terms.append(constraint.object)
    concept = getattr(constraint, "object_concept", None)
    if concept is not None:
        terms.extend(getattr(concept, "match_terms", []))
    value = constraint.value
    if isinstance(value, str):
        terms.append(value)
    elif isinstance(value, dict):
        for key in ("object", "anchor", "target", "name"):
            if value.get(key):
                terms.append(value[key])
    return {normalize_term(item) for item in terms if normalize_term(item)}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy().item()
        return float(value)
    except Exception:
        return default


def _pcd_bounds(obj: Any) -> tuple[list[float], list[float]]:
    try:
        pts = getattr(obj, "pcd").point.positions.cpu().numpy()
        if len(pts) > 0:
            return (
                [float(x) for x in np.min(pts, axis=0)[:3].tolist()],
                [float(x) for x in np.max(pts, axis=0)[:3].tolist()],
            )
    except Exception:
        pass
    center = np.array(getattr(obj, "position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)[:3]
    if center.size < 3:
        center = np.pad(center, (0, 3 - center.size))
    return center.tolist(), center.tolist()


def object_record_from_mapper_object(obj: Any, canonical_label: str = "", step: int | None = None) -> dict[str, Any]:
    candidate = candidate_from_object(obj, canonical_label=canonical_label, step=step)
    min_bound, max_bound = _pcd_bounds(obj)
    return {
        "id": candidate.uid,
        "name": normalize_term(getattr(obj, "tag", canonical_label)),
        "tag": normalize_term(getattr(obj, "tag", canonical_label)),
        "canonical_label": normalize_term(canonical_label),
        "center": list(candidate.centroid or []),
        "centroid": list(candidate.centroid or []),
        "position": list(candidate.centroid or []),
        "bbox_2d": list(candidate.bbox_2d or []),
        "min_bound": min_bound,
        "max_bound": max_bound,
        "confidence": _safe_float(getattr(obj, "confidence", 0.0)),
    }


class ConstraintEvaluator:
    """Runtime evaluator for Phase2/Phase3 instruction constraints.

    设计原则：
    - Parser 不编码常识，只声明约束；
    - attribute 由 final verifier 结合原始指令判断；
    - room/count/sequence 属于执行状态；
    - spatial relation 先几何硬过滤，再按需调用 VLM 建动态语义边。
    """

    def __init__(
        self,
        *,
        spatial_graph: InstructionSpatialGraph | None = None,
        relation_service: DynamicRelationService | None = None,
        save_dir: str = "",
    ):
        self.spatial_graph = spatial_graph or InstructionSpatialGraph()
        self.relation_service = relation_service or DynamicRelationService()
        self.save_dir = save_dir

    def ensure_state(self, mapper: Any, plan: InstructionPlan) -> InstructionExecutionState:
        state = getattr(mapper, "instruction_execution_state", None)
        if state is None:
            state = InstructionExecutionState(plan)
            mapper.instruction_execution_state = state
        else:
            state.bind_plan(plan)
        return state

    def register_mapper_objects(
        self,
        *,
        mapper: Any,
        step: int | None = None,
        current_node_idx: int | None = None,
        rgb_path: str = "",
    ):
        view_id = f"step_{step}" if step is not None else ""
        position = []
        try:
            position = [float(x) for x in np.array(mapper.current_position, dtype=float).reshape(-1)[:3].tolist()]
        except Exception:
            pass
        for obj in getattr(mapper, "objects", []) or []:
            candidate = candidate_from_object(obj, canonical_label=getattr(mapper, "target", ""), step=step)
            self.spatial_graph.record_observation(
                candidate=candidate,
                view_id=view_id or candidate.uid,
                step=step,
                node_idx=current_node_idx,
                rgb_path=rgb_path,
                position=position,
            )

    def target_for_candidate(
        self,
        mapper: Any,
        plan: InstructionPlan,
        candidate: CandidateInstance,
    ) -> TargetQuery | None:
        state = self.ensure_state(mapper, plan)
        return state.target_for_candidate(plan, candidate.detector_label or candidate.canonical_label)

    def evaluate_before_final_verifier(
        self,
        *,
        mapper: Any,
        plan: InstructionPlan,
        target: TargetQuery,
        candidate: CandidateInstance,
        candidate_obj: Any,
        evidence: dict[str, Any],
        step: int | None = None,
    ) -> ConstraintEvaluation:
        """Evaluate hard runtime constraints before final stop acceptance."""

        self.ensure_state(mapper, plan)
        relation_edges: list[SemanticEdge] = []
        failed: list[str] = []
        satisfied: list[str] = []

        for constraint in plan.constraints:
            ctype = normalize_term(constraint.type)
            if not _constraint_subject_matches(constraint, target, candidate):
                continue
            if ctype in RELATION_CONSTRAINT_TYPES:
                relation_eval = self._evaluate_relation_constraint(
                    mapper=mapper,
                    plan=plan,
                    constraint=constraint,
                    target=target,
                    candidate=candidate,
                    candidate_obj=candidate_obj,
                    evidence=evidence,
                    step=step,
                )
                relation_edges.extend(relation_eval.relation_edges)
                failed.extend(relation_eval.failed_constraints)
                satisfied.extend(relation_eval.satisfied_constraints)
            elif ctype == "room":
                # 当前 STRIVE 房间没有自然语言标签；room 约束作为证据交给
                # final verifier，只有在未来接入 room caption 后才 hard reject。
                evidence.setdefault("declared_room_constraints", []).append(constraint.as_dict())
                satisfied.append("room constraint queued for final verifier")
            elif ctype in ("attribute", "color", "material", "state"):
                evidence.setdefault("declared_attribute_constraints", []).append(constraint.as_dict())
                satisfied.append("attribute constraint queued for final verifier")

        if failed:
            return ConstraintEvaluation(
                satisfied=False,
                decision="reject_relation",
                confidence=1.0,
                failed_constraints=failed,
                satisfied_constraints=satisfied,
                relation_edges=relation_edges,
                evidence=evidence,
                reason="One or more hard runtime constraints failed.",
            )
        return ConstraintEvaluation(
            satisfied=True,
            decision="pass",
            confidence=1.0,
            failed_constraints=[],
            satisfied_constraints=satisfied,
            relation_edges=relation_edges,
            evidence=evidence,
            reason="Runtime constraints passed or were deferred to final verifier.",
        )

    def apply_final_result(
        self,
        *,
        mapper: Any,
        plan: InstructionPlan,
        target: TargetQuery | None,
        candidate: CandidateInstance,
        result: VerificationResult,
    ) -> bool:
        state = self.ensure_state(mapper, plan)
        if target is None:
            target = state.target_for_candidate(plan, candidate.detector_label or candidate.canonical_label)
        if target is None:
            return bool(result.satisfied and result.decision == "accept")
        if not (result.satisfied and result.decision == "accept"):
            state.mark_candidate_rejected(target, candidate.uid)
            return False
        return state.mark_candidate_accepted(plan, target, candidate.uid)

    def _evaluate_relation_constraint(
        self,
        *,
        mapper: Any,
        plan: InstructionPlan,
        constraint: Constraint,
        target: TargetQuery,
        candidate: CandidateInstance,
        candidate_obj: Any,
        evidence: dict[str, Any],
        step: int | None = None,
    ) -> ConstraintEvaluation:
        relation = normalize_relation(constraint.relation or str(constraint.value or ""))
        anchor_terms = _constraint_object_terms(constraint)
        anchor_concept = getattr(constraint, "object_concept", None)
        if not relation or (not anchor_terms and anchor_concept is None):
            return ConstraintEvaluation(
                satisfied=False,
                decision="need_relation_check",
                failed_constraints=[f"incomplete relation constraint: {constraint.as_dict()}"],
                reason="Relation constraint lacks relation or anchor object.",
            )

        subject_record = object_record_from_mapper_object(candidate_obj, canonical_label=target.name, step=step)
        view_control = dict(evidence.get("view_control") or {})
        pinned_context = dict(view_control.get("pinned_relation_context") or {})
        pinned_edge = dict(pinned_context.get("edge") or {})
        if (
            view_control.get("active")
            and pinned_edge
            and normalize_relation(pinned_edge.get("relation", "")) == relation
            and pinned_edge.get("verified")
        ):
            # better-view 子任务追踪的是已验证的语义边，而不是
            # 每一帧重新发现关系。mapper 在近距离可能把同一书架区域切成
            # 新的 book uid；此时复用 pinned DynamicSemanticEdge，让 final
            # verifier 只重新判断视角质量。
            edge = SemanticEdge(
                subject_id=str(pinned_edge.get("subject_id", subject_record["id"])),
                relation=relation,
                object_id=str(pinned_edge.get("object_id", "")),
                confidence=float(pinned_edge.get("confidence", 1.0) or 1.0),
                verified=True,
                source="view_control_pinned_edge",
                evidence_view_ids=list(pinned_edge.get("evidence_view_ids", []) or []),
                reason=(
                    "Reused pinned relation edge during better-view control; "
                    "current candidate evidence is treated as the same accepted semantic region."
                ),
            )
            evidence.setdefault("relation_edges", []).append(edge.as_dict())
            evidence.setdefault("verified_relation_context", pinned_context)
            return ConstraintEvaluation(
                satisfied=True,
                decision="pass",
                confidence=edge.confidence,
                satisfied_constraints=[f"{edge.subject_id} {relation} {edge.object_id}"],
                relation_edges=[edge],
                evidence=evidence,
                reason=edge.reason,
            )
        anchors = []
        raw_instruction = str(getattr(plan, "raw_instruction", "") or getattr(mapper, "target", ""))
        mapper_objects = list(getattr(mapper, "objects", []) or [])
        concept_records = {}
        if anchor_concept is not None and hasattr(mapper, "concept_matcher"):
            records = mapper.concept_matcher.match_many(
                raw_instruction=raw_instruction,
                concept=anchor_concept,
                objects=[obj for obj in mapper_objects if obj is not candidate_obj],
                step=step,
            )
            concept_records = {record.object_uid: record for record in records}
        for obj in mapper_objects:
            if obj is candidate_obj:
                continue
            label = normalize_term(getattr(obj, "tag", ""))
            concept_match = None
            if anchor_concept is not None and concept_records:
                obj_uid = candidate_from_object(obj, canonical_label=getattr(anchor_concept, "name", ""), step=step).uid
                concept_match = concept_records.get(obj_uid)
            if label in anchor_terms or (concept_match is not None and concept_match.matches_concept):
                anchors.append(obj)

        if not anchors:
            # 没看到 anchor 时不能证明关系失败，只能要求继续搜索。
            return ConstraintEvaluation(
                satisfied=False,
                decision="need_relation_check",
                failed_constraints=[f"missing anchor object for relation {relation}: {sorted(anchor_terms)}"],
                reason="Anchor object has not been observed yet.",
            )

        edges = []
        evidence_paths = []
        rejected_pairs = []
        inst_hash = instruction_hash(raw_instruction)
        for anchor in anchors:
            anchor_record = object_record_from_mapper_object(anchor, canonical_label="", step=step)
            pair_ledger = getattr(mapper, "relation_pair_ledger", None)
            if pair_ledger is not None and pair_ledger.is_rejected(
                inst_hash,
                subject_record["id"],
                relation,
                anchor_record["id"],
            ):
                rejected_pairs.append(anchor_record["id"])
                continue
            if pair_ledger is not None and pair_ledger.is_accepted(
                inst_hash,
                subject_record["id"],
                relation,
                anchor_record["id"],
            ):
                edge = SemanticEdge(
                    subject_id=subject_record["id"],
                    relation=relation,
                    object_id=anchor_record["id"],
                    confidence=1.0,
                    verified=True,
                    source="relation_pair_ledger",
                    evidence_view_ids=[],
                    reason="Relation pair was already accepted for this instruction.",
                )
                evidence.setdefault("relation_edges", []).append(edge.as_dict())
                evidence["verified_relation_context"] = {
                    "edge": edge.as_dict(),
                    "subject_record": subject_record,
                    "anchor_record": anchor_record,
                    "source": "relation_pair_ledger",
                }
                return ConstraintEvaluation(
                    satisfied=True,
                    decision="pass",
                    confidence=edge.confidence,
                    satisfied_constraints=[f"{candidate.uid} {relation} {anchor_record['id']}"],
                    relation_edges=[edge],
                    evidence=evidence,
                    reason=edge.reason,
                )
            co_views = self.spatial_graph.co_visible_views(subject_record["id"], anchor_record["id"])
            if not co_views and evidence.get("current_rgb_with_bbox_path"):
                co_view = {
                    "id": f"current_{step}",
                    "step": step,
                    "rgb_path": evidence.get("current_rgb_with_bbox_path", ""),
                    "observed_object_uids": [subject_record["id"], anchor_record["id"]],
                    "allow_vlm_despite_geometry": bool(
                        isinstance(evidence.get("geometry"), dict)
                        and evidence.get("geometry", {}).get("source") == "check_again"
                    ),
                }
                co_views = [type("_View", (), {"as_dict": lambda self, data=co_view: data})()]
            elif evidence.get("current_rgb_with_bbox_path") and (
                isinstance(evidence.get("geometry"), dict)
                and evidence.get("geometry", {}).get("source") == "check_again"
            ):
                # check_again 图是当前最强视觉证据。即使历史 co-view
                # 存在，也允许它覆盖几何预筛，避免小物体/书架点云高度误差直接
                # hard reject，符合“几何硬约束 + VLM 软推理”的分层。
                co_view = {
                    "id": f"current_{step}",
                    "step": step,
                    "rgb_path": evidence.get("current_rgb_with_bbox_path", ""),
                    "observed_object_uids": [subject_record["id"], anchor_record["id"]],
                    "allow_vlm_despite_geometry": True,
                }
                co_views = [type("_View", (), {"as_dict": lambda self, data=co_view: data})()] + list(co_views)
            view_dicts = [view.as_dict() if hasattr(view, "as_dict") else dict(view) for view in co_views]
            edge = self.relation_service.verify(
                subject=subject_record,
                relation=relation,
                object_=anchor_record,
                evidence_views=view_dicts,
                use_vlm=True,
            )
            edges.append(edge)
            evidence_paths.extend([view.get("rgb_path", "") for view in view_dicts if view.get("rgb_path")])
            if edge.verified:
                if pair_ledger is not None:
                    pair_ledger.mark(
                        instruction_hash=inst_hash,
                        subject_id=subject_record["id"],
                        relation=relation,
                        object_id=anchor_record["id"],
                        status="accepted_relation",
                        confidence=edge.confidence,
                        step=step,
                        reason=edge.reason,
                        evidence_view_ids=edge.evidence_view_ids,
                    )
                evidence.setdefault("relation_edges", []).append(edge.as_dict())
                evidence["verified_relation_context"] = {
                    "edge": edge.as_dict(),
                    "subject_record": subject_record,
                    "anchor_record": anchor_record,
                    "source": "vlm",
                }
                evidence.setdefault("relation_evidence_paths", []).extend(evidence_paths)
                return ConstraintEvaluation(
                    satisfied=True,
                    decision="pass",
                    confidence=edge.confidence,
                    satisfied_constraints=[f"{candidate.uid} {relation} {anchor_record['id']}"],
                    relation_edges=edges,
                    evidence=evidence,
                    reason=edge.reason,
                )
            if pair_ledger is not None:
                pair_ledger.mark(
                    instruction_hash=inst_hash,
                    subject_id=subject_record["id"],
                    relation=relation,
                    object_id=anchor_record["id"],
                    status="rejected_relation",
                    confidence=edge.confidence,
                    step=step,
                    reason=edge.reason,
                    evidence_view_ids=edge.evidence_view_ids,
                )

        return ConstraintEvaluation(
            satisfied=False,
            decision="reject_relation",
            confidence=max([edge.confidence for edge in edges] or [0.0]),
            failed_constraints=[
                f"relation not verified: {candidate.uid} {relation} {sorted(anchor_terms)}"
                + (f"; skipped rejected pairs={rejected_pairs}" if rejected_pairs else "")
            ],
            relation_edges=edges,
            evidence=evidence,
            reason="No candidate anchor pair satisfied the required relation.",
        )

    def dump_state(self, *, mapper: Any, episode_idx: int, step: int | None = None):
        if not self.save_dir:
            return
        out_dir = os.path.join(self.save_dir, f"episode-{episode_idx}", "instruction_adapter")
        os.makedirs(out_dir, exist_ok=True)
        state = getattr(mapper, "instruction_execution_state", None)
        payload = {
            "execution_state": state.as_dict() if state is not None else None,
            "spatial_graph": self.spatial_graph.as_dict(),
            "semantic_edges": self.relation_service.as_dict(),
            "concept_matches": mapper.concept_matcher.as_dict() if hasattr(mapper, "concept_matcher") else {},
            "concept_matcher_stats": mapper.concept_matcher.stats_dict() if hasattr(mapper, "concept_matcher") else {},
            "lvlm_call_counts": lvlm_counts(),
            "anchor_search_ledger": mapper.anchor_search_ledger.as_dict() if hasattr(mapper, "anchor_search_ledger") else {},
            "relation_pair_ledger": mapper.relation_pair_ledger.as_dict() if hasattr(mapper, "relation_pair_ledger") else {},
        }
        suffix = f"_{step}" if step is not None else ""
        with open(os.path.join(out_dir, f"runtime_state{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
