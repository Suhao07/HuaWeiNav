"""FloorPlan-compatible room-layout contracts for VLN prior maps.

The layout contract deliberately contains room geometry and room topology only.
Object instances remain optional search hints and are not part of the canonical
floorplan representation used for global navigation context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .contracts import BoundaryXY, PriorMapData, PriorRoom, PriorTopologyEdge, Vector2


@dataclass(frozen=True)
class FloorplanRegion:
    """A semantic room polygon in a metric floorplan coordinate frame.

    Args:
        uid: Stable region identifier.
        label: Semantic room label, or ``unknown`` when unavailable.
        boundary_xy: Polygon in the floorplan ``(x, -z)`` plane.
        center_xy: Region center in the same metric plane.
        level: Floor index.
        height_range: Optional lower and upper world heights.
        connectivity: Neighboring region identifiers.
        confidence: Geometry/semantic confidence in ``[0, 1]``.
        metadata: JSON-friendly provenance and quality fields.
    """

    uid: str
    label: str = "unknown"
    boundary_xy: BoundaryXY = ()
    center_xy: Optional[Vector2] = None
    level: int = 0
    height_range: Optional[tuple[float, float]] = None
    connectivity: tuple[str, ...] = ()
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the region using FloorPlan-VLN-compatible field names."""

        return {
            "id": self.uid,
            "uid": self.uid,
            "label": self.label,
            "boundaries": [list(point) for point in self.boundary_xy],
            "center": list(self.center_xy) if self.center_xy is not None else None,
            "level": self.level,
            "region_height_range": list(self.height_range) if self.height_range else None,
            "connectivity": list(self.connectivity),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FloorplanLevel:
    """One floor of a structured floorplan.

    Args:
        level: Integer floor index.
        height_range: Optional lower and upper world heights for this floor.
        regions: Room or region polygons belonging to this floor.
    """

    level: int
    height_range: Optional[tuple[float, float]] = None
    regions: tuple[FloorplanRegion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the level and its regions."""

        region_map = {region.uid: region.to_dict() for region in self.regions}
        region_ids = [region.uid for region in self.regions]
        index = {uid: position for position, uid in enumerate(region_ids)}
        adjacency = [[0 for _ in region_ids] for _ in region_ids]
        for source_index, region in enumerate(self.regions):
            for target_uid in region.connectivity:
                target_index = index.get(target_uid)
                if target_index is not None:
                    adjacency[source_index][target_index] = 1
        return {
            "height_range": list(self.height_range) if self.height_range else None,
            "regions": region_map,
            "region_graph": adjacency,
        }


@dataclass(frozen=True)
class FloorplanLayout:
    """Metric room layout that can be exchanged with FloorPlan-VLN tooling.

    Args:
        scene_id: Stable scene or building identifier.
        frame_id: Coordinate frame declared by the serialized layout. Generated
            layouts use ``floorplan_metric`` for the ``(x, -z)`` plane.
        levels: Floor levels with room polygons and connectivity.
        metadata: Source, authority, and quality metadata.
    """

    scene_id: str
    frame_id: str = "habitat_world"
    levels: tuple[FloorplanLevel, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a FloorPlan-VLN-compatible JSON payload."""

        return {
            "name": self.scene_id,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "coordinate_convention": {
                "world_axes": ["x", "y", "z"],
                "floorplan_axes": ["x", "-z"],
                "unit": "meter",
            },
            "total_levels": len(self.levels),
            "total_regions": sum(len(level.regions) for level in self.levels),
            "levels": {str(level.level): level.to_dict() for level in self.levels},
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        """Write the layout as deterministic UTF-8 JSON.

        Args:
            path: Destination JSON path.

        Returns:
            The written path.
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def from_prior_map(cls, prior_map: PriorMapData) -> "FloorplanLayout":
        """Convert room-only canonical prior data to this exchange format.

        Args:
            prior_map: Canonical prior map containing room polygons and edges.

        Returns:
            Layout with object instances intentionally omitted.
        """

        room_edges = _room_neighbors(prior_map.topology_edges)
        grouped: dict[int, list[FloorplanRegion]] = {}
        for room in prior_map.rooms:
            height_range = _room_height_range(room)
            grouped.setdefault(int(room.level), []).append(
                FloorplanRegion(
                    uid=room.uid,
                    label=room.label,
                    boundary_xy=_habitat_xz_to_floorplan(room.boundary_xy),
                    center_xy=_habitat_xz_to_floorplan_point(room.centroid_xy),
                    level=int(room.level),
                    height_range=height_range,
                    connectivity=tuple(sorted(room_edges.get(room.uid, set(room.neighbors)))),
                    confidence=float(room.confidence),
                    metadata={**room.metadata, "source": room.source},
                )
            )
        levels = tuple(
            FloorplanLevel(
                level=level,
                height_range=_level_height_range(regions),
                regions=tuple(sorted(regions, key=lambda item: item.uid)),
            )
            for level, regions in sorted(grouped.items())
        )
        return cls(
            scene_id=prior_map.scene_id,
            frame_id="floorplan_metric",
            levels=levels,
            metadata={
                "authority": "room_layout_only",
                "source_format": prior_map.source_format,
                "source_frame_id": prior_map.frame_id,
                "object_instances_omitted": True,
                "coordinate_transform": "habitat_xz_to_floorplan_x_neg_z",
                **dict(prior_map.metadata),
            },
        )


def _room_neighbors(edges: Sequence[PriorTopologyEdge]) -> dict[str, set[str]]:
    """Collect only room-room topology edges."""

    neighbors: dict[str, set[str]] = {}
    for edge in edges:
        if edge.edge_type != "room-room":
            continue
        neighbors.setdefault(edge.source_uid, set()).add(edge.target_uid)
        if edge.bidirectional:
            neighbors.setdefault(edge.target_uid, set()).add(edge.source_uid)
    return neighbors


def _habitat_xz_to_floorplan(boundary: BoundaryXY) -> BoundaryXY:
    """Convert VLN room coordinates from ``(x, z)`` to ``(x, -z)``."""

    return tuple((float(x), -float(z)) for x, z in boundary)


def _habitat_xz_to_floorplan_point(point: Optional[Vector2]) -> Optional[Vector2]:
    """Convert one optional room center from ``(x, z)`` to ``(x, -z)``."""

    if point is None:
        return None
    return (float(point[0]), -float(point[1]))


def _room_height_range(room: PriorRoom) -> Optional[tuple[float, float]]:
    """Read an optional semantic AABB height range from room provenance."""

    metadata = room.metadata if isinstance(room.metadata, dict) else {}
    center = metadata.get("aabb_center")
    sizes = metadata.get("aabb_sizes")
    if not isinstance(center, Sequence) or not isinstance(sizes, Sequence) or len(center) < 2 or len(sizes) < 2:
        return None
    try:
        y = float(center[1])
        half_height = abs(float(sizes[1])) / 2.0
    except (TypeError, ValueError):
        return None
    return (y - half_height, y + half_height)


def _level_height_range(regions: Sequence[FloorplanRegion]) -> Optional[tuple[float, float]]:
    """Aggregate valid region height ranges into one level range."""

    ranges = [region.height_range for region in regions if region.height_range is not None]
    if not ranges:
        return None
    return (min(item[0] for item in ranges), max(item[1] for item in ranges))


__all__ = ["FloorplanLayout", "FloorplanLevel", "FloorplanRegion"]
