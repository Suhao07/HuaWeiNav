from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from .ontology import normalize_term


RelationCallback = Callable[[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], str], Any]


@dataclass(frozen=True)
class RelationQuery:
    """A lazy object-object relation query.

    subject/object are runtime object records, not parser concepts. The parser
    only declares relation constraints; this module verifies them on candidate
    pairs when STRIVE has actual map objects and shared observations.
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    if text in ("on top of", "above"):
        return "on"
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


class DynamicSemanticEdgeVerifier:
    """Lazy semantic edge verifier following the SysNav split.

    几何只做硬过滤：明显不可能的候选对直接拒绝；语义关系是否成立交给 VLM
    callback。这样解析模块不需要知道“电视常在客厅”这类常识，运行时也不会对
    所有物体对做昂贵推理。
    """

    def __init__(
        self,
        cache: SemanticEdgeCache | None = None,
        *,
        max_near_distance: float = 1.5,
        vertical_axis: int = 1,
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
        evidence_views: list[dict[str, Any]] | None = None,
        vlm_callback: RelationCallback | None = None,
    ) -> SemanticEdge:
        relation = normalize_relation(relation)
        query = RelationQuery(str(subject.get("id", "")), relation, str(object_.get("id", "")))
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        evidence = list(evidence_views or [])
        if not self._geometry_allows(subject, relation, object_):
            return self.cache.put(
                SemanticEdge(
                    subject_id=query.subject_id,
                    relation=relation,
                    object_id=query.object_id,
                    confidence=0.0,
                    verified=False,
                    source="geometry",
                    evidence_view_ids=_stable_view_ids(evidence),
                    reason="Rejected by geometric hard constraint.",
                )
            )

        if vlm_callback is None:
            return self.cache.put(
                SemanticEdge(
                    subject_id=query.subject_id,
                    relation=relation,
                    object_id=query.object_id,
                    confidence=0.5,
                    verified=False,
                    source="geometry_prefilter",
                    evidence_view_ids=_stable_view_ids(evidence),
                    reason="Geometry passed; VLM callback was not provided.",
                )
            )

        prompt = self._build_vlm_prompt(subject, relation, object_)
        result = vlm_callback(relation, subject, object_, evidence, prompt)
        edge = self._coerce_vlm_result(query, evidence, result)
        return self.cache.put(edge)

    def _geometry_allows(self, subject: dict[str, Any], relation: str, object_: dict[str, Any]) -> bool:
        subject_center = _point(subject, ("center", "centroid", "position"))
        object_center = _point(object_, ("center", "centroid", "position"))
        if subject_center is None or object_center is None:
            return True

        if relation in ("near", "with"):
            return _distance(subject_center, object_center) <= self.max_near_distance

        vertical_delta = subject_center[self.vertical_axis] - object_center[self.vertical_axis]
        horizontal_axes = [idx for idx in range(3) if idx != self.vertical_axis]
        horizontal_dist = math.sqrt(
            sum((subject_center[idx] - object_center[idx]) ** 2 for idx in horizontal_axes)
        )
        if relation == "under":
            return vertical_delta < 0 and horizontal_dist <= self.max_near_distance
        if relation == "on":
            return vertical_delta > 0 and horizontal_dist <= self.max_near_distance
        if relation == "inside":
            return horizontal_dist <= self.max_near_distance
        return True

    @staticmethod
    def _build_vlm_prompt(subject: dict[str, Any], relation: str, object_: dict[str, Any]) -> str:
        subject_name = subject.get("name") or subject.get("tag") or subject.get("id")
        object_name = object_.get("name") or object_.get("tag") or object_.get("id")
        return (
            "Decide whether the spatial relation is visible in the provided views. "
            "Return whether the relation is true, a confidence in [0,1], and a short reason.\n"
            f"Subject: {subject_name}\nRelation: {relation}\nObject: {object_name}"
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
        return SemanticEdge(
            subject_id=query.subject_id,
            relation=query.relation,
            object_id=query.object_id,
            confidence=max(0.0, min(1.0, confidence)),
            verified=verified,
            source="vlm",
            evidence_view_ids=_stable_view_ids(evidence_views),
            reason=reason,
        )
