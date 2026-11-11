from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .ontology import normalize_term


RelationCallback = Callable[[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], str], Any]


@dataclass(frozen=True)
class RelationQuery:
    """A lazy object-object relation query.

    subject/object are runtime object records, not parser concepts. The parser
    only declares relation constraints; this module verifies them on candidate
    pairs when VLN has actual map objects and shared observations.
    """

    subject_id: str
    relation: str
    object_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (str(self.subject_id), normalize_relation(self.relation), str(self.object_id))


@dataclass
class SemanticEdge:
    subject_id: str
    relation: str
    object_id: str
    confidence: float = 0.0
    verified: bool = False
    source: str = "unknown"
    evidence_view_ids: list[str] = field(default_factory=list)
    reason: str = ""
    need_better_view: bool = False
    evidence_key: str = ""
    identity_judgments: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.need_better_view:
            return "unknown"
        return "true" if self.verified else "false"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationPairRecord:
    """Instruction-scoped memory for one candidate-anchor relation pair.

    关系失败只说明“这个目标实例和这个 anchor 实例不满足该关系”，
    不能屏蔽目标类别，也不能屏蔽 anchor 类别。这个 ledger 的粒度与
    SysNav 的动态 object-object edge 对齐。
    """

    instruction_hash: str
    subject_id: str
    relation: str
    object_id: str
    status: str
    confidence: float = 0.0
    step: int | None = None
    reason: str = ""
    evidence_view_ids: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            str(self.instruction_hash),
            str(self.subject_id),
            normalize_relation(self.relation),
            str(self.object_id),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelationPairLedger:
    """Cache accepted/rejected object-object relation pairs."""

    def __init__(self):
        self.records: dict[tuple[str, str, str, str], RelationPairRecord] = {}

    def reset(self):
        self.records.clear()

    def get(self, instruction_hash: str, subject_id: str, relation: str, object_id: str) -> RelationPairRecord | None:
        return self.records.get((str(instruction_hash), str(subject_id), normalize_relation(relation), str(object_id)))

    def is_rejected(self, instruction_hash: str, subject_id: str, relation: str, object_id: str) -> bool:
        record = self.get(instruction_hash, subject_id, relation, object_id)
        return bool(record and record.status == "rejected_relation")

    def is_accepted(self, instruction_hash: str, subject_id: str, relation: str, object_id: str) -> bool:
        record = self.get(instruction_hash, subject_id, relation, object_id)
        return bool(record and record.status == "accepted_relation")

    def mark(
        self,
        *,
        instruction_hash: str,
        subject_id: str,
        relation: str,
        object_id: str,
        status: str,
        confidence: float = 0.0,
        step: int | None = None,
        reason: str = "",
        evidence_view_ids: list[str] | None = None,
    ) -> RelationPairRecord:
        record = RelationPairRecord(
            instruction_hash=str(instruction_hash),
            subject_id=str(subject_id),
            relation=normalize_relation(relation),
            object_id=str(object_id),
            status=str(status),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            step=step,
            reason=str(reason or ""),
            evidence_view_ids=list(evidence_view_ids or []),
        )
        self.records[record.key] = record
        return record

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {"|".join(key): record.as_dict() for key, record in self.records.items()}


class SemanticEdgeCache:
    """Small deterministic cache for expensive relation checks."""

    def __init__(self):
        self._edges: dict[tuple[str, str, str], SemanticEdge] = {}

    def get(self, query: RelationQuery) -> SemanticEdge | None:
        return self._edges.get(query.key)

    def put(self, edge: SemanticEdge) -> SemanticEdge:
        query = RelationQuery(edge.subject_id, edge.relation, edge.object_id)
        self._edges[query.key] = edge
        return edge

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {"|".join(key): edge.as_dict() for key, edge in self._edges.items()}


def normalize_relation(value: str) -> str:
    text = normalize_term(value)
    if text in ("next to", "beside", "adjacent to"):
        return "near"
    if text in ("in", "within"):
        return "inside"
    return text


def _point(obj: dict[str, Any], keys: Iterable[str]) -> tuple[float, float, float] | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError):
                continue
    return None


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[idx] - b[idx]) ** 2 for idx in range(3)))


def _stable_view_ids(evidence_views: list[dict[str, Any]]) -> list[str]:
    ids = []
    for view in evidence_views:
        view_id = view.get("id") or view.get("view_id") or view.get("step") or view.get("path")
        if view_id is None:
            digest = hashlib.sha1(json.dumps(view, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
            view_id = digest
        ids.append(str(view_id))
    return ids


def relation_view_is_bound(view: dict[str, Any], subject_id: str, object_id: str) -> bool:
    boxes = view.get("object_boxes", {})
    path = view.get("rgb_path") or view.get("image_ref") or view.get("path")
    return bool(view.get("id") and path and Path(path).is_file() and all(
        uid in view.get("observed_object_uids", []) and len(boxes.get(uid, [])) == 4
        for uid in (subject_id, object_id)
    ))


class DynamicSemanticEdgeVerifier:
    """Lazy semantic edge verifier following the SysNav split.

    几何只做硬过滤：明显不可能的候选对直接拒绝；语义关系是否成立交给 VLM
    callback。这样解析模块不需要知道“电视常在客厅”这类常识 运行时也不会对
    所有物体对做昂贵推理。
    """

    def __init__(
        self,
        cache: SemanticEdgeCache | None = None,
        *,
        max_near_distance: float = 1.5,
        vertical_axis: int = 2,
    ):
        self.cache = cache or SemanticEdgeCache()
        self.max_near_distance = max_near_distance
        self.vertical_axis = vertical_axis

    def verify(
        self,
        *,
        subject: dict[str, Any],
        relation: str,
        object_: dict[str, Any],
        predicate: dict[str, Any] | None = None,
        evidence_views: list[dict[str, Any]] | None = None,
        vlm_callback: RelationCallback | None = None,
    ) -> SemanticEdge:
        relation = normalize_relation(relation)
        query = RelationQuery(str(subject.get("id", "")), relation, str(object_.get("id", "")))
        evidence = [view for view in evidence_views or []
                    if relation_view_is_bound(view, query.subject_id, query.object_id)]
        image_hashes = []
        for view in evidence:
            path = view.get("rgb_path") or view.get("image_ref") or view.get("path")
            image_hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                if path and Path(path).is_file() else None)
        evidence_key = hashlib.sha256(json.dumps(
            [subject, relation, object_, predicate, evidence, image_hashes, vlm_callback is not None],
            sort_keys=True, default=str,
        ).encode("utf-8")).hexdigest()
        cached = self.cache.get(query)
        from verification.images import image_dependencies_valid

        if (cached is not None and cached.evidence_key == evidence_key
                and image_dependencies_valid(cached.dependencies)):
            return cached

        allow_visual_override = any(bool(view.get("allow_vlm_despite_geometry")) for view in evidence)
        if not evidence:
            edge = SemanticEdge(
                subject_id=query.subject_id, relation=relation, object_id=query.object_id,
                source="missing_observation", need_better_view=True,
                reason="No shared instance-bound observation is available.",
            )
        elif not self._geometry_allows(subject, relation, object_) and not allow_visual_override:
            edge = SemanticEdge(
                subject_id=query.subject_id, relation=relation, object_id=query.object_id,
                source="geometry", evidence_view_ids=_stable_view_ids(evidence),
                reason="Rejected by geometric hard constraint.",
            )
        elif vlm_callback is None:
            edge = SemanticEdge(
                subject_id=query.subject_id, relation=relation, object_id=query.object_id,
                source="geometry_prefilter", evidence_view_ids=_stable_view_ids(evidence),
                need_better_view=True, reason="Visual relation verification is unavailable.",
            )
        else:
            prompt = self._build_vlm_prompt(subject, relation, object_, predicate)
            result = vlm_callback(relation, subject, object_, evidence, prompt)
            edge = self._coerce_vlm_result(query, evidence, result)
        edge.evidence_key = evidence_key
        return self.cache.put(edge)

    def _geometry_allows(self, subject: dict[str, Any], relation: str, object_: dict[str, Any]) -> bool:
        subject_center = _point(subject, ("center", "centroid", "position"))
        object_center = _point(object_, ("center", "centroid", "position"))
        if subject_center is None or object_center is None:
            return True

        if relation in ("near", "with"):
            return _distance(subject_center, object_center) <= self.max_near_distance

        subject_min = _point(subject, ("min_bound",))
        subject_max = _point(subject, ("max_bound",))
        object_min = _point(object_, ("min_bound",))
        object_max = _point(object_, ("max_bound",))
        vertical_delta = subject_center[self.vertical_axis] - object_center[self.vertical_axis]
        horizontal_axes = [idx for idx in range(3) if idx != self.vertical_axis]
        horizontal_dist = math.sqrt(
            sum((subject_center[idx] - object_center[idx]) ** 2 for idx in horizontal_axes)
        )
        xy_overlap = True
        if subject_min and subject_max and object_min and object_max:
            overlaps = []
            for axis in horizontal_axes:
                overlaps.append(subject_min[axis] <= object_max[axis] and subject_max[axis] >= object_min[axis])
            xy_overlap = all(overlaps)
        if relation == "under":
            if subject_max and object_min:
                return subject_max[self.vertical_axis] <= object_min[self.vertical_axis] + 0.2 and xy_overlap
            return vertical_delta < 0 and horizontal_dist <= self.max_near_distance
        # 顶部支撑才适用垂向接触模型；泛指 on 的墙面附着交给视觉判断。
        if relation == "on top of":
            if subject_min and object_max:
                vertical_gap = subject_min[self.vertical_axis] - object_max[self.vertical_axis]
                return -0.15 <= vertical_gap <= 0.35 and xy_overlap
            return vertical_delta > 0 and horizontal_dist <= self.max_near_distance
        if relation == "inside":
            if subject_center and object_min and object_max:
                return all(object_min[idx] - 0.1 <= subject_center[idx] <= object_max[idx] + 0.1 for idx in range(3))
            return horizontal_dist <= self.max_near_distance
        return True

    @staticmethod
    def _build_vlm_prompt(subject: dict[str, Any], relation: str, object_: dict[str, Any],
                          predicate: dict[str, Any] | None) -> str:
        return (
            "Verify this exact directed relation between the two bound instances. "
            "Requested concepts are requirements, not established identities; detector labels "
            "are fallible observations. Do not substitute another visible object or relation. "
            "Use only each frame's object_boxes for its instance regions. "
            "Judge the relation together with all qualifiers in the required predicate.\n"
            f"Required predicate: {json.dumps(predicate, ensure_ascii=False)}\n"
            f"Subject instance: {subject['id']}\n"
            f"Required subject concept: {subject['requested_concept']}\n"
            f"Subject detector label: {subject['detector_label']}\n"
            f"Subject demand: {json.dumps(subject.get('demand', {}), ensure_ascii=False)}\n"
            f"Subject attributes: {json.dumps(subject.get('attributes', {}), ensure_ascii=False)}\n"
            f"Required relation: {relation}\n"
            f"Reference instance: {object_['id']}\n"
            f"Required reference concept: {object_['requested_concept']}\n"
            f"Reference detector label: {object_['detector_label']}\n"
            f"Reference demand: {json.dumps(object_.get('demand', {}), ensure_ascii=False)}\n"
            f"Reference attributes: {json.dumps(object_.get('attributes', {}), ensure_ascii=False)}"
        )

    @staticmethod
    def _coerce_vlm_result(
        query: RelationQuery,
        evidence_views: list[dict[str, Any]],
        result: Any,
    ) -> SemanticEdge:
        if isinstance(result, dict):
            verified = bool(result.get("verified", result.get("answer", False)))
            confidence = float(result.get("confidence", 1.0 if verified else 0.0))
            reason = str(result.get("reason", ""))
        elif isinstance(result, tuple) and len(result) >= 2:
            verified = bool(result[0])
            confidence = float(result[1])
            reason = str(result[2]) if len(result) > 2 else ""
        else:
            verified = bool(result)
            confidence = 1.0 if verified else 0.0
            reason = ""
        need_better_view = bool(result.get("need_better_view", False)) if isinstance(result, dict) else False
        return SemanticEdge(
            subject_id=query.subject_id,
            relation=query.relation,
            object_id=query.object_id,
            confidence=max(0.0, min(1.0, confidence)),
            verified=verified and not need_better_view,
            source="vlm",
            evidence_view_ids=_stable_view_ids(evidence_views),
            reason=reason,
            need_better_view=need_better_view,
            identity_judgments=result.get("identity_judgments", {}) if isinstance(result, dict) else {},
            evidence_ids=list(result.get("evidence_ids", [])) if isinstance(result, dict) else [],
            dependencies=dict(result.get("dependencies", {})) if isinstance(result, dict) else {},
        )
