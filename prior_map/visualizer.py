"""Set-of-Marks SVG visualizer for prior maps.

The visualizer renders deterministic SVG text and marker metadata. It does not
depend on OpenCV, ROS, Habitat, or browser runtimes, so it can run in offline
tests and deployment smoke checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .contracts import PriorMapData, PriorObject, PriorRoom


@dataclass(frozen=True)
class SomMarker:
    """Stable Set-of-Marks marker metadata.

    Args:
        marker_id: Stable marker id derived from marker type and prior uid.
        marker_type: Marker family, such as ``room`` or ``object``.
        uid: Prior map uid represented by the marker.
        label: Human-readable label.
        xy: Prior-map coordinates used to place the marker.
        display_label: Short text shown in the rendered SVG.
        metadata: JSON-friendly marker diagnostics.
    """

    marker_id: str
    marker_type: str
    uid: str
    label: str
    xy: Tuple[float, float]
    display_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly marker dictionary.

        Returns:
            Dictionary containing marker identity, position, and metadata.
        """

        return {
            "marker_id": self.marker_id,
            "marker_type": self.marker_type,
            "uid": self.uid,
            "label": self.label,
            "xy": list(self.xy),
            "display_label": self.display_label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SomView:
    """Rendered prior-map SoM view.

    Args:
        svg: SVG document as text.
        markers: Stable markers shown in the view.
        legend: Legend mapping marker prefixes to semantics.
        view_type: View type, such as ``global`` or ``room``.
        view_box: Prior-map bounds represented by the view.
        metadata: JSON-friendly rendering diagnostics.
    """

    svg: str
    markers: Tuple[SomMarker, ...]
    legend: dict[str, str]
    view_type: str
    view_box: Tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly rendered view dictionary.

        Returns:
            Dictionary containing SVG text, marker list, legend, and bounds.
        """

        return {
            "svg": self.svg,
            "markers": [marker.to_dict() for marker in self.markers],
            "legend": dict(self.legend),
            "view_type": self.view_type,
            "view_box": list(self.view_box),
            "metadata": dict(self.metadata),
        }


class PriorMapSomVisualizer:
    """Render global and room-level SoM views for prior maps.

    Args:
        width: SVG canvas width in pixels.
        height: SVG canvas height in pixels.
        margin: Canvas margin in pixels.
        max_objects: Maximum object markers rendered.
    """

    def __init__(self, width: int = 900, height: int = 600, margin: int = 36, max_objects: int = 80) -> None:
        """Create a visualizer.

        Args:
            width: SVG canvas width in pixels.
            height: SVG canvas height in pixels.
            margin: Canvas margin in pixels.
            max_objects: Maximum object markers rendered.
        """

        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        self.max_objects = int(max_objects)

    def render_global_view(self, map_data: PriorMapData) -> SomView:
        """Render a global prior-map SoM SVG.

        Args:
            map_data: Prior map to render.

        Returns:
            ``SomView`` containing SVG, markers, legend, and diagnostics.
        """

        rooms = tuple(sorted(map_data.rooms, key=lambda room: room.uid))
        objects = tuple(sorted(map_data.objects, key=lambda obj: obj.uid))[: max(0, self.max_objects)]
        bounds = _map_bounds(map_data, rooms, objects)
        markers = self._build_markers(rooms=rooms, objects=objects)
        svg = self._render_svg(
            title=f"Prior map: {map_data.scene_id}",
            rooms=rooms,
            objects=objects,
            markers=markers,
            bounds=bounds,
        )
        return SomView(
            svg=svg,
            markers=tuple(markers),
            legend=_legend(),
            view_type="global",
            view_box=bounds,
            metadata={
                "scene_id": map_data.scene_id,
                "room_count": len(rooms),
                "object_count": len(objects),
                "frame_id": map_data.frame_id,
            },
        )

    def render_room_view(self, map_data: PriorMapData, room_uid: str) -> SomView:
        """Render a room-level SoM SVG.

        Args:
            map_data: Prior map to render.
            room_uid: Room uid to focus.

        Returns:
            ``SomView`` containing only the selected room and its object
            markers.

        Raises:
            ValueError: If ``room_uid`` is not present.
        """

        room = map_data.room_by_uid(room_uid)
        if room is None:
            raise ValueError(f"Unknown prior room uid: {room_uid}")
        rooms = (room,)
        objects = tuple(
            sorted((obj for obj in map_data.objects if obj.parent_room_uid == room_uid), key=lambda obj: obj.uid)
        )[: max(0, self.max_objects)]
        bounds = _map_bounds(map_data, rooms, objects)
        markers = self._build_markers(rooms=rooms, objects=objects)
        svg = self._render_svg(
            title=f"Prior room: {room.label or room.uid}",
            rooms=rooms,
            objects=objects,
            markers=markers,
            bounds=bounds,
        )
        return SomView(
            svg=svg,
            markers=tuple(markers),
            legend=_legend(),
            view_type="room",
            view_box=bounds,
            metadata={
                "scene_id": map_data.scene_id,
                "room_uid": room_uid,
                "object_count": len(objects),
                "frame_id": map_data.frame_id,
            },
        )

    def _build_markers(
        self,
        *,
        rooms: Sequence[PriorRoom],
        objects: Sequence[PriorObject],
    ) -> List[SomMarker]:
        markers: List[SomMarker] = []
        for room in rooms:
            xy = _room_anchor(room)
            if xy is None:
                continue
            marker_id = _marker_id("R", room.uid)
            markers.append(
                SomMarker(
                    marker_id=marker_id,
                    marker_type="room",
                    uid=room.uid,
                    label=room.label,
                    xy=xy,
                    display_label=marker_id,
                    metadata={"confidence": room.confidence, "neighbors": list(room.neighbors)},
                )
            )
        for obj in objects:
            xy = _object_anchor(obj)
            if xy is None:
                continue
            marker_id = _marker_id("O", obj.uid)
            markers.append(
                SomMarker(
                    marker_id=marker_id,
                    marker_type="object",
                    uid=obj.uid,
                    label=obj.label,
                    xy=xy,
                    display_label=marker_id,
                    metadata={
                        "confidence": obj.confidence,
                        "parent_room_uid": obj.parent_room_uid,
                        "exact": obj.exact,
                        "aliases": list(obj.aliases),
                    },
                )
            )
        return markers

    def _render_svg(
        self,
        *,
        title: str,
        rooms: Sequence[PriorRoom],
        objects: Sequence[PriorObject],
        markers: Sequence[SomMarker],
        bounds: Tuple[float, float, float, float],
    ) -> str:
        transform = _CanvasTransform(bounds=bounds, width=self.width, height=self.height, margin=self.margin)
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">',
            f"  <title>{escape(title)}</title>",
            '  <rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        ]
        for room in rooms:
            if room.boundary_xy:
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.to_canvas(point) for point in room.boundary_xy))
                lines.append(
                    f'  <polygon points="{points}" fill="#dbeafe" stroke="#2563eb" '
                    f'stroke-width="1.5" opacity="0.72"/>'
                )
            anchor = _room_anchor(room)
            if anchor is not None:
                x, y = transform.to_canvas(anchor)
                lines.append(
                    f'  <text x="{x + 6:.1f}" y="{y - 8:.1f}" font-family="sans-serif" '
                    f'font-size="12" fill="#1e293b">{escape(room.label or room.uid)}</text>'
                )
        for obj in objects:
            anchor = _object_anchor(obj)
            if anchor is None:
                continue
            x, y = transform.to_canvas(anchor)
            fill = "#f97316" if obj.exact else "#f59e0b"
            lines.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{fill}" stroke="#7c2d12"/>')
        for marker in markers:
            x, y = transform.to_canvas(marker.xy)
            if marker.marker_type == "room":
                lines.append(
                    f'  <rect x="{x - 13:.1f}" y="{y - 13:.1f}" width="26" height="18" '
                    f'rx="2" fill="#1d4ed8" opacity="0.92"/>'
                )
            else:
                lines.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="#c2410c" opacity="0.92"/>')
            lines.append(
                f'  <text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="8" fill="#ffffff">{escape(marker.display_label)}</text>'
            )
        lines.extend(_legend_svg(self.width, self.height))
        lines.append("</svg>")
        return "\n".join(lines)


@dataclass(frozen=True)
class _CanvasTransform:
    bounds: Tuple[float, float, float, float]
    width: int
    height: int
    margin: int

    def to_canvas(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        """Transform prior-map coordinates into SVG canvas coordinates."""

        min_x, min_y, max_x, max_y = self.bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        usable_w = max(self.width - 2 * self.margin, 1)
        usable_h = max(self.height - 2 * self.margin, 1)
        scale = min(usable_w / span_x, usable_h / span_y)
        x = self.margin + (float(xy[0]) - min_x) * scale
        y = self.height - self.margin - (float(xy[1]) - min_y) * scale
        return x, y


def render_global_view(map_data: PriorMapData, *, width: int = 900, height: int = 600) -> SomView:
    """Render a global prior-map SoM view.

    Args:
        map_data: Prior map to render.
        width: SVG width.
        height: SVG height.

    Returns:
        Rendered SoM view.
    """

    return PriorMapSomVisualizer(width=width, height=height).render_global_view(map_data)


def render_room_view(map_data: PriorMapData, room_uid: str, *, width: int = 900, height: int = 600) -> SomView:
    """Render a room-level prior-map SoM view.

    Args:
        map_data: Prior map to render.
        room_uid: Room uid to focus.
        width: SVG width.
        height: SVG height.

    Returns:
        Rendered SoM view.
    """

    return PriorMapSomVisualizer(width=width, height=height).render_room_view(map_data, room_uid)


def _map_bounds(
    map_data: PriorMapData,
    rooms: Sequence[PriorRoom],
    objects: Sequence[PriorObject],
) -> Tuple[float, float, float, float]:
    points: List[Tuple[float, float]] = []
    if map_data.world_min is not None and map_data.world_max is not None:
        points.extend((map_data.world_min, map_data.world_max))
    for room in rooms:
        points.extend((float(x), float(y)) for x, y in room.boundary_xy)
        anchor = _room_anchor(room)
        if anchor is not None:
            points.append(anchor)
    for obj in objects:
        anchor = _object_anchor(obj)
        if anchor is not None:
            points.append(anchor)
    if not points:
        return (0.0, 0.0, 1.0, 1.0)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    pad_x = max((max_x - min_x) * 0.06, 0.5)
    pad_y = max((max_y - min_y) * 0.06, 0.5)
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def _room_anchor(room: PriorRoom) -> Optional[Tuple[float, float]]:
    if room.centroid_xy is not None:
        return float(room.centroid_xy[0]), float(room.centroid_xy[1])
    if room.boundary_xy:
        x = sum(point[0] for point in room.boundary_xy) / len(room.boundary_xy)
        y = sum(point[1] for point in room.boundary_xy) / len(room.boundary_xy)
        return float(x), float(y)
    return None


def _object_anchor(obj: PriorObject) -> Optional[Tuple[float, float]]:
    if obj.position_xyz is None:
        return None
    return float(obj.position_xyz[0]), float(obj.position_xyz[1])


def _marker_id(prefix: str, uid: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(uid).strip()).strip("_")
    return f"{prefix}_{slug or 'unknown'}"


def _legend() -> dict[str, str]:
    return {
        "R_*": "prior room marker",
        "O_*": "prior object marker",
        "blue polygon": "room boundary",
        "orange dot": "prior object position; darker means exact instance",
    }


def _legend_svg(width: int, height: int) -> List[str]:
    x = max(width - 260, 12)
    y = max(height - 86, 12)
    return [
        f'  <g id="legend" transform="translate({x},{y})">',
        '    <rect x="0" y="0" width="248" height="74" rx="4" fill="#ffffff" stroke="#cbd5e1"/>',
        '    <rect x="10" y="14" width="22" height="14" rx="2" fill="#1d4ed8"/>',
        '    <text x="42" y="25" font-family="sans-serif" font-size="11" fill="#0f172a">room marker</text>',
        '    <circle cx="21" cy="48" r="8" fill="#c2410c"/>',
        '    <text x="42" y="52" font-family="sans-serif" font-size="11" fill="#0f172a">object marker</text>',
        "  </g>",
    ]


__all__ = [
    "PriorMapSomVisualizer",
    "SomMarker",
    "SomView",
    "render_global_view",
    "render_room_view",
]
