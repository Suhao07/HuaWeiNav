from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts import ConceptQuery, Constraint, InstructionPlan, TargetQuery
from .concept_matcher import ConceptMatchRecord
from .ontology import normalize_term
from .relation_verifier import DynamicRelationService
from .semantic_edges import SemanticEdge, normalize_relation
from .spatial_graph import InstructionSpatialGraph
from .verifier import CandidateInstance, VerificationResult, candidate_from_object, instruction_hash
from llm_utils.lvlm_call_tracker import counts as lvlm_counts


from instruction.contracts import RELATION_CONSTRAINT_TYPES


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
    center_value = getattr(obj, "bbox3d_center", None)
    extent_value = getattr(obj, "bbox3d_extent", None)
    if center_value is not None and extent_value is not None:
        center, extent = np.asarray(center_value), np.asarray(extent_value)
        return (center - extent / 2).tolist(), (center + extent / 2).tolist()
    position = getattr(obj, "position", None)
    center = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=float).reshape(-1)[:3]
    if center.size < 3:
        center = np.pad(center, (0, 3 - center.size))
    return center.tolist(), center.tolist()


def object_record_from_mapper_object(obj: Any, requested_concept: str, step: int | None = None,
                                     *, demand: dict[str, Any] | None = None) -> dict[str, Any]:
    candidate = candidate_from_object(obj, step=step)
    min_bound, max_bound = _pcd_bounds(obj)
    return {
        "id": candidate.uid,
        "requested_concept": normalize_term(requested_concept),
        "demand": dict(demand or {}),
        "detector_label": normalize_term(candidate.detector_label),
        "center": list(candidate.centroid or []),
        "centroid": list(candidate.centroid or []),
        "position": list(candidate.centroid or []),
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
        self.identity_records: dict[tuple[str, ...], dict[str, Any]] = {}

    @staticmethod
    def _identity_key(raw_instruction, concept, object_uid):
        return (instruction_hash(raw_instruction), concept.id, normalize_term(concept.name),
                json.dumps(concept.demand, sort_keys=True), object_uid)

    def record_identity(self, raw_instruction, concept, judgment, confidence):
        uid = judgment["instance_id"]
        key = self._identity_key(raw_instruction, concept, uid)
        supplied = set(judgment["source_crop_sha256"])
        previous = self.identity_records.get(key)
        if previous is not None:
            if not supplied.difference(previous["source_crop_sha256"]):
                if (judgment["verdict"] == previous["verdict"] or judgment["verdict"] == "unknown"
                        or previous["verdict"] == "false"):
                    return
                self.identity_records[key] = {
                    **previous, "verdict": judgment["verdict"], "reason": judgment["reason"],
                    "confidence": confidence,
                }
                return
            supplied.update(previous["source_crop_sha256"])
        self.identity_records[key] = {
            **judgment, "source_crop_sha256": sorted(supplied), "confidence": confidence,
            "observed_crop_sha256": sorted(self.spatial_graph.object_crop_hashes(uid) | supplied),
        }

    def identity_match(self, *, raw_instruction, concept, object_uid, observed_label, evidence_views=None):
        stored = self.identity_records.get(self._identity_key(raw_instruction, concept, object_uid))
        if stored is None:
            return None
        current = self.spatial_graph.object_crop_hashes(object_uid, evidence_views)
        baseline = stored["observed_crop_sha256"] if evidence_views is None else stored["source_crop_sha256"]
        changed = bool(current.difference(baseline))
        verdict = "unknown" if changed else stored["verdict"]
        return ConceptMatchRecord(
            instruction_hash=instruction_hash(raw_instruction), concept_id=concept.id,
            object_uid=object_uid, observed_label=observed_label,
            matches_concept={"true": True, "false": False, "unknown": None}[verdict],
            candidate_eligible=verdict != "false", terminal_eligible=False,
            confidence=0.0 if changed else stored["confidence"], source="relation_identity",
            reason="New instance pixels require identity verification." if changed else stored["reason"],
        )

    def register_mapper_objects(
        self,
        *,
        mapper: Any,
        step: int | None = None,
        current_node_idx: int | None = None,
        rgb_path: str = "",
    ):
        for obj in mapper.objects:
            candidate = candidate_from_object(obj, step=step)
            self.spatial_graph.upsert_object(candidate)
            metadata = getattr(obj, "metadata", {})
            image_ref = getattr(obj, "image_ref", "")
            frame_id = metadata.get("frame_id")
            if image_ref and frame_id is not None and candidate.bbox_2d:
                self.spatial_graph.record_observation(
                    candidate=candidate, view_id=str(frame_id),
                    rgb_path=image_ref, step=metadata.get("step"),
                )
        snapshot = getattr(mapper, "snapshot", None)
        for view in getattr(snapshot, "metadata", {}).get("observation_images", []):
            self._register_view(view)

    def _register_view(self, view):
        self.spatial_graph.upsert_view(
            view_id=str(view["id"]), step=view.get("step"),
            rgb_path=view.get("image_ref") or view.get("rgb_path", ""),
            position=view.get("position"),
            observed_object_uids=view.get("observed_object_uids", []),
            object_boxes=view.get("object_boxes", {}),
        )

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

        relation_edges: list[SemanticEdge] = []
        failed: list[str] = []
        satisfied: list[str] = []
        decisions = []
        for view in evidence.get("observation_images", []):
            self._register_view(view)

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
                decisions.append(relation_eval.decision)
                failed.extend(relation_eval.failed_constraints)
                satisfied.extend(relation_eval.satisfied_constraints)
            elif ctype == "room":
                # 当前 VLN 房间没有自然语言标签；room 约束作为证据交给
                # final verifier，只有在未来接入 room caption 后才 hard reject。
                evidence.setdefault("declared_room_constraints", []).append(constraint.as_dict())
                satisfied.append("room constraint queued for final verifier")
            elif ctype in ("attribute", "color", "material", "state", "affordance"):
                evidence.setdefault("declared_attribute_constraints", []).append(constraint.as_dict())
                satisfied.append("attribute constraint queued for final verifier")

        if failed:
            return ConstraintEvaluation(
                satisfied=False,
                decision="reject_relation" if "reject_relation" in decisions else "need_relation_check",
                confidence=max((edge.confidence for edge in relation_edges), default=0.0),
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

        anchor_concept = anchor_concept or ConceptQuery(
            f"reference:{constraint.object}", constraint.object, role="anchor")
        subject_record = object_record_from_mapper_object(
            candidate_obj, requested_concept=target.name, step=step, demand=target.demand)
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
            if candidate_from_object(obj, step=step).uid == candidate.uid:
                continue
            label = normalize_term(getattr(obj, "label", getattr(obj, "tag", "")))
            concept_match = None
            if anchor_concept is not None and concept_records:
                obj_uid = candidate_from_object(obj, canonical_label=getattr(anchor_concept, "name", ""), step=step).uid
                concept_match = concept_records.get(obj_uid)
            identity = self.identity_match(
                raw_instruction=raw_instruction, concept=anchor_concept,
                object_uid=candidate_from_object(obj, step=step).uid, observed_label=label)
            concept_match = identity if identity is not None else concept_match
            eligible = concept_match.candidate_eligible if concept_match is not None else label in anchor_terms
            if eligible:
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
        inst_hash = instruction_hash(raw_instruction)
        for anchor in anchors:
            anchor_record = object_record_from_mapper_object(
                anchor, requested_concept=anchor_concept.name if anchor_concept else constraint.object,
                step=step, demand=anchor_concept.demand if anchor_concept else None)
            pair_ledger = getattr(mapper, "relation_pair_ledger", None)
            co_views = self.spatial_graph.co_visible_views(subject_record["id"], anchor_record["id"])
            view_dicts = [view.as_dict() if hasattr(view, "as_dict") else dict(view) for view in co_views]
            identities = [("subject", target.concept_query(), subject_record),
                          ("reference", anchor_concept, anchor_record)]
            judgments = [self.identity_match(
                raw_instruction=raw_instruction, concept=concept, object_uid=record["id"],
                observed_label=record["detector_label"], evidence_views=view_dicts)
                for _, concept, record in identities]
            if any(row is not None and row.matches_concept is False for row in judgments):
                continue
            edge = self.relation_service.verify(
                subject=subject_record,
                relation=relation,
                object_=anchor_record,
                evidence_views=view_dicts,
                use_vlm=True,
            )
            for role, concept, _ in identities:
                if role in edge.identity_judgments:
                    self.record_identity(raw_instruction, concept, edge.identity_judgments[role], edge.confidence)
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
                    status="unknown" if edge.need_better_view else "rejected_relation",
                    confidence=edge.confidence,
                    step=step,
                    reason=edge.reason,
                    evidence_view_ids=edge.evidence_view_ids,
                )

        evidence.setdefault("relation_edges", []).extend(edge.as_dict() for edge in edges)
        evidence.setdefault("relation_evidence_paths", []).extend(evidence_paths)
        return ConstraintEvaluation(
            satisfied=False,
            decision="need_relation_check" if not edges or any(edge.need_better_view for edge in edges) else "reject_relation",
            confidence=max([edge.confidence for edge in edges] or [0.0]),
            failed_constraints=[f"relation not verified: {candidate.uid} {relation} {sorted(anchor_terms)}"],
            relation_edges=edges,
            evidence=evidence,
            reason="No candidate anchor pair satisfied the required relation.",
        )

    def dump_state(self, *, mapper: Any, episode_idx: int, step: int | None = None):
        if not self.save_dir:
            return
        out_dir = os.path.join(self.save_dir, f"episode-{episode_idx}", "instruction_adapter")
        os.makedirs(out_dir, exist_ok=True)
        state = getattr(mapper, "progress", None)
        payload = {
            "execution_state": state.as_dict() if state is not None else None,
            "spatial_graph": self.spatial_graph.as_dict(),
            "semantic_edges": self.relation_service.as_dict(),
            "identity_judgments": {"|".join(key): value for key, value in self.identity_records.items()},
            "concept_matches": mapper.concept_matcher.as_dict() if hasattr(mapper, "concept_matcher") else {},
            "concept_matcher_stats": mapper.concept_matcher.stats_dict() if hasattr(mapper, "concept_matcher") else {},
            "lvlm_call_counts": lvlm_counts(),
            "anchor_search_ledger": mapper.anchor_search_ledger.as_dict() if hasattr(mapper, "anchor_search_ledger") else {},
            "relation_pair_ledger": mapper.relation_pair_ledger.as_dict() if hasattr(mapper, "relation_pair_ledger") else {},
        }
        suffix = f"_{step}" if step is not None else ""
        with open(os.path.join(out_dir, f"runtime_state{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
