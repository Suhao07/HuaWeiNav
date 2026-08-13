"""Build prior maps from HM3D semantic annotation text files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .alignment import PriorMapAlignment
from .contracts import PriorMapData, PriorObject, PriorRoom, PriorTopologyEdge


@dataclass(frozen=True)
class HM3DSemanticInstance:
    """One row from an HM3D ``*.semantic.txt`` annotation file."""

    instance_id: int
    color_hex: str
    label: str
    region_id: str


@dataclass(frozen=True)
class HM3DSemanticPriorMapBuildResult:
    """Result of building a prior map from HM3D scene semantics."""

    prior_map: PriorMapData
    instances: tuple[HM3DSemanticInstance, ...]
    metadata: dict[str, Any]


class HM3DSemanticTxtPriorMapBuilder:
    """Convert HM3D semantic annotation txt files into STRIVE prior maps."""

    def build(
        self,
        semantic_txt_path: str | Path,
        *,
        scene_id: str = "",
        include_unknown: bool = False,
        include_structural: bool = False,
    ) -> HM3DSemanticPriorMapBuildResult:
        """Build a scene-level semantic prior map.

        Args:
            semantic_txt_path: HM3D ``*.semantic.txt`` path.
            scene_id: Optional scene id override.
            include_unknown: Keep ``unknown`` labels when true.
            include_structural: Keep structural labels such as wall/floor/ceiling.

        Returns:
            Build result with canonical ``PriorMapData``.
        """

        path = Path(semantic_txt_path)
        instances = _read_semantic_txt(path)
        filtered = tuple(
            inst
            for inst in instances
            if _keep_instance(inst, include_unknown=include_unknown, include_structural=include_structural)
        )
        inferred_scene_id = scene_id or _infer_scene_id(path)
        rooms = _rooms_from_instances(inferred_scene_id, filtered)
        objects = tuple(_object_from_instance(inferred_scene_id, inst) for inst in filtered)
        edges = tuple(
            PriorTopologyEdge(
                uid=f"edge:{obj.parent_room_uid}:{obj.uid}",
                source_uid=str(obj.parent_room_uid),
                target_uid=obj.uid,
                edge_type="room-object",
                relation="contains",
                confidence=obj.confidence,
                source="hm3d_semantic_txt",
            )
            for obj in objects
            if obj.parent_room_uid
        )
        label_counts: dict[str, int] = {}
        for inst in filtered:
            label_counts[inst.label] = label_counts.get(inst.label, 0) + 1
        prior_map = PriorMapData(
            scene_id=inferred_scene_id,
            rooms=rooms,
            objects=objects,
            topology_edges=edges,
            source_format="hm3d_semantic_txt",
            frame_id="habitat_world",
            metadata={
                "semantic_txt_path": str(path),
                "raw_instance_count": len(instances),
                "instance_count": len(filtered),
                "region_count": len(rooms),
                "label_count": len(label_counts),
                "include_unknown": bool(include_unknown),
                "include_structural": bool(include_structural),
                "label_counts": dict(sorted(label_counts.items())),
                "authority": "scene_semantic_inventory",
            },
        )
        return HM3DSemanticPriorMapBuildResult(
            prior_map=prior_map,
            instances=filtered,
            metadata=dict(prior_map.metadata),
        )


def build_hm3d_semantic_prior_map(
    semantic_txt_path: str | Path,
    *,
    scene_id: str = "",
    include_unknown: bool = False,
    include_structural: bool = False,
) -> HM3DSemanticPriorMapBuildResult:
    """Build a canonical prior map from HM3D semantic txt annotations."""

    return HM3DSemanticTxtPriorMapBuilder().build(
        semantic_txt_path,
        scene_id=scene_id,
        include_unknown=include_unknown,
        include_structural=include_structural,
    )


def write_hm3d_semantic_prior_map_with_alignment(
    result: HM3DSemanticPriorMapBuildResult,
    output_path: str | Path,
    *,
    alignment_output_path: str | Path = "",
    alignment_mode: str = "unavailable",
) -> dict[str, str]:
    """Write generated HM3D semantic prior map and optional alignment file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.prior_map.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = {"prior_map": str(output)}
    if alignment_output_path:
        alignment = _alignment_for_mode(alignment_mode, result.prior_map)
        alignment_path = Path(alignment_output_path)
        alignment.save(alignment_path)
        paths["alignment"] = str(alignment_path)
    return paths


def _read_semantic_txt(path: Path) -> tuple[HM3DSemanticInstance, ...]:
    instances: list[HM3DSemanticInstance] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or not str(header[0]).startswith("HM3D Semantic"):
            raise ValueError(f"Unexpected HM3D semantic txt header in {path}")
        for row in reader:
            if not row or len(row) < 4:
                continue
            instance_id = _safe_int(row[0])
            label = str(row[2]).strip()
            region_id = str(row[3]).strip()
            if instance_id is None or not label or not region_id:
                continue
            instances.append(
                HM3DSemanticInstance(
                    instance_id=instance_id,
                    color_hex=str(row[1]).strip(),
                    label=label,
                    region_id=region_id,
                )
            )
    return tuple(instances)


def _rooms_from_instances(scene_id: str, instances: tuple[HM3DSemanticInstance, ...]) -> tuple[PriorRoom, ...]:
    region_counts: dict[str, int] = {}
    for inst in instances:
        region_counts[inst.region_id] = region_counts.get(inst.region_id, 0) + 1
    rooms = []
    for region_id, count in sorted(region_counts.items(), key=lambda item: _safe_sort_key(item[0])):
        room_uid = _room_uid(scene_id, region_id)
        rooms.append(
            PriorRoom(
                uid=room_uid,
                label=f"region_{region_id}",
                confidence=0.45,
                source="hm3d_semantic_txt",
                metadata={
                    "hm3d_region_id": region_id,
                    "instance_count": count,
                    "coarse_region": True,
                },
            )
        )
    if rooms:
        return tuple(rooms)
    return (
        PriorRoom(
            uid=f"prior_room:{_safe_uid(scene_id)}:scene",
            label="scene",
            confidence=0.35,
            source="hm3d_semantic_txt",
            metadata={"fallback_scene_room": True},
        ),
    )


def _object_from_instance(scene_id: str, inst: HM3DSemanticInstance) -> PriorObject:
    return PriorObject(
        uid=f"prior_object:{_safe_uid(scene_id)}:{inst.instance_id}",
        label=inst.label,
        parent_room_uid=_room_uid(scene_id, inst.region_id),
        exact=True,
        confidence=0.65,
        source="hm3d_semantic_txt",
        metadata={
            "hm3d_instance_id": inst.instance_id,
            "hm3d_color_hex": inst.color_hex,
            "hm3d_region_id": inst.region_id,
            "geometry_available": False,
        },
    )


def _keep_instance(inst: HM3DSemanticInstance, *, include_unknown: bool, include_structural: bool) -> bool:
    label = _norm(inst.label)
    if not include_unknown and label in {"unknown", "misc", "miscellaneous"}:
        return False
    if not include_structural and label in _STRUCTURAL_LABELS:
        return False
    return True


def _alignment_for_mode(mode: str, prior_map: PriorMapData) -> PriorMapAlignment:
    normalized = str(mode or "unavailable").strip().lower()
    if normalized == "identity":
        return PriorMapAlignment.identity(prior_frame_id=prior_map.frame_id, runtime_frame_id="map")
    return PriorMapAlignment.unavailable(
        prior_frame_id=prior_map.frame_id,
        runtime_frame_id="map",
        reason=(
            "HM3D semantic txt provides instance labels and region ids but no "
            "runtime-calibrated object positions or room boundaries."
        ),
    )


def _infer_scene_id(path: Path) -> str:
    name = path.name
    for suffix in (".semantic.txt", ".txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _room_uid(scene_id: str, region_id: str) -> str:
    return f"prior_room:{_safe_uid(scene_id)}:region_{_safe_uid(region_id)}"


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_sort_key(value: str) -> tuple[int, str]:
    parsed = _safe_int(value)
    if parsed is not None:
        return parsed, ""
    return 10**9, str(value)


def _safe_uid(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return safe or "unknown"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


_STRUCTURAL_LABELS = {
    "ceiling",
    "door",
    "door frame",
    "floor",
    "railing",
    "stairs",
    "wall",
    "window",
    "window frame",
}


__all__ = [
    "HM3DSemanticInstance",
    "HM3DSemanticPriorMapBuildResult",
    "HM3DSemanticTxtPriorMapBuilder",
    "build_hm3d_semantic_prior_map",
    "write_hm3d_semantic_prior_map_with_alignment",
]
