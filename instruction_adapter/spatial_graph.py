from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .ontology import normalize_term
from .verifier import CandidateInstance


@dataclass
class ViewNode:
    id: str
    step: int | None = None
    node_idx: int | None = None
    rgb_path: str = ""
    position: list[float] = field(default_factory=list)
    observed_object_uids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectNodeRecord:
    uid: str
    label: str
    canonical_label: str = ""
    centroid: list[float] = field(default_factory=list)
    bbox_2d: list[float] = field(default_factory=list)
    confidence: float = 0.0
    observed_view_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstructionSpatialGraph:
    """Lightweight semantic graph used by dynamic relation verification.

    已经维护对象点云和拓扑节点；这里不复制重建地图，只保存
    verifier 需要的对象实例、视角和共视关系索引。
    """

    def __init__(self):
        self.objects: dict[str, ObjectNodeRecord] = {}
        self.views: dict[str, ViewNode] = {}

    def reset(self):
        self.objects.clear()
        self.views.clear()

    def upsert_object(self, candidate: CandidateInstance, view_id: str = ""):
        record = self.objects.get(candidate.uid)
        if record is None:
            record = ObjectNodeRecord(
                uid=candidate.uid,
                label=normalize_term(candidate.detector_label),
                canonical_label=normalize_term(candidate.canonical_label),
                centroid=list(candidate.centroid or []),
                bbox_2d=list(candidate.bbox_2d or []),
                confidence=float(candidate.confidence or 0.0),
            )
            self.objects[candidate.uid] = record
        else:
            record.label = normalize_term(candidate.detector_label) or record.label
            record.canonical_label = normalize_term(candidate.canonical_label) or record.canonical_label
            if candidate.centroid:
                record.centroid = list(candidate.centroid)
            if candidate.bbox_2d:
                record.bbox_2d = list(candidate.bbox_2d)
            record.confidence = max(record.confidence, float(candidate.confidence or 0.0))
        if view_id and view_id not in record.observed_view_ids:
            record.observed_view_ids.append(view_id)

    def upsert_view(
        self,
        *,
        view_id: str,
        step: int | None = None,
        node_idx: int | None = None,
        rgb_path: str = "",
        position: list[float] | None = None,
        observed_object_uids: list[str] | None = None,
    ):
        view = self.views.get(view_id)
        if view is None:
            view = ViewNode(
                id=view_id,
                step=step,
                node_idx=node_idx,
                rgb_path=rgb_path,
                position=list(position or []),
                observed_object_uids=[],
            )
            self.views[view_id] = view
        if rgb_path:
            view.rgb_path = rgb_path
        if position:
            view.position = list(position)
        for uid in observed_object_uids or []:
            if uid and uid not in view.observed_object_uids:
                view.observed_object_uids.append(uid)

    def record_observation(
        self,
        *,
        candidate: CandidateInstance,
        view_id: str,
        step: int | None = None,
        node_idx: int | None = None,
        rgb_path: str = "",
        position: list[float] | None = None,
    ):
        self.upsert_object(candidate, view_id=view_id)
        self.upsert_view(
            view_id=view_id,
            step=step,
            node_idx=node_idx,
            rgb_path=rgb_path,
            position=position,
            observed_object_uids=[candidate.uid],
        )

    def co_visible_views(self, subject_uid: str, object_uid: str, limit: int = 4) -> list[ViewNode]:
        subject = self.objects.get(subject_uid)
        obj = self.objects.get(object_uid)
        if subject is None or obj is None:
            return []
        shared = [vid for vid in subject.observed_view_ids if vid in set(obj.observed_view_ids)]
        views = [self.views[vid] for vid in shared if vid in self.views]
        views = sorted(views, key=lambda item: (item.step is None, item.step or 0), reverse=True)
        return views[: max(1, limit)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "objects": {key: value.as_dict() for key, value in self.objects.items()},
            "views": {key: value.as_dict() for key, value in self.views.items()},
        }
