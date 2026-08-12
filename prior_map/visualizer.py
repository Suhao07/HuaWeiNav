"""Set-of-Marks and floorplan visualizers for prior maps.

The visualizers render deterministic SVG/PNG text and marker metadata. They do
not depend on OpenCV, ROS, simulator runtimes, or browser runtimes, so they can
run in offline tests and deployment smoke checks.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class FloorPlanOverlayPoint:
    """Overlay point rendered on a floorplan prior map.

    Args:
        uid: Stable overlay identifier.
        label: Human-readable label.
        xy: Prior-map plane coordinate.
        point_type: Overlay family, such as ``frontier`` or ``live_detection``.
        selected: Whether this point is the active selected element.
        metadata: JSON-friendly diagnostics for review artifacts.
    """

    uid: str
    label: str
    xy: Tuple[float, float]
    point_type: str
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly marker dictionary.

        Returns:
            Dictionary containing overlay point identity and geometry.
        """

        return {
            "uid": self.uid,
            "label": self.label,
            "xy": list(self.xy),
            "point_type": self.point_type,
            "selected": bool(self.selected),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FloorPlanOverlay:
    """Optional dynamic overlays for floorplan prior-map rendering.

    Args:
        target_prior_object_uids: Prior object ids highlighted as target priors.
        frontiers: Runtime frontier points in prior-map coordinates.
        live_detections: Prior object positions currently supported by live
            observations.
        trajectory_xy: Runtime trajectory points in prior-map coordinates.
        selected_frontier_uid: Active selected frontier uid.
        baseline_frontier_uid: Baseline frontier uid before prior ranking.
        metadata: JSON-friendly overlay diagnostics.
    """

    target_prior_object_uids: Tuple[str, ...] = ()
    frontiers: Tuple[FloorPlanOverlayPoint, ...] = ()
    live_detections: Tuple[FloorPlanOverlayPoint, ...] = ()
    trajectory_xy: Tuple[Tuple[float, float], ...] = ()
    selected_frontier_uid: Optional[str] = None
    baseline_frontier_uid: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly overlay dictionary.

        Returns:
            Dictionary containing target ids, frontier markers, live detections,
            trajectory points, and diagnostics.
        """

        return {
            "target_prior_object_uids": list(self.target_prior_object_uids),
            "frontiers": [point.to_dict() for point in self.frontiers],
            "live_detections": [point.to_dict() for point in self.live_detections],
            "trajectory_xy": [list(point) for point in self.trajectory_xy],
            "selected_frontier_uid": self.selected_frontier_uid,
            "baseline_frontier_uid": self.baseline_frontier_uid,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FloorPlanView:
    """Rendered FloorPlan-VLN-style prior-map view.

    Args:
        svg: SVG document as text.
        markers: Stable marker metadata rendered in the view.
        legend: Legend mapping visual elements to semantics.
        view_type: View type, such as ``floorplan_global`` or
            ``floorplan_room``.
        view_box: Prior-map bounds represented by the view.
        overlay: Dynamic overlay rendered on top of the static prior map.
        metadata: JSON-friendly rendering diagnostics.
    """

    svg: str
    markers: Tuple[dict[str, Any], ...]
    legend: dict[str, str]
    view_type: str
    view_box: Tuple[float, float, float, float]
    overlay: FloorPlanOverlay = field(default_factory=FloorPlanOverlay)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly rendered view dictionary.

        Returns:
            Dictionary containing SVG text, markers, legend, bounds, and
            overlay metadata.
        """

        return {
            "svg": self.svg,
            "markers": [dict(marker) for marker in self.markers],
            "legend": dict(self.legend),
            "view_type": self.view_type,
            "view_box": list(self.view_box),
            "overlay": self.overlay.to_dict(),
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

    def write_global_artifacts(
        self,
        map_data: PriorMapData,
        output_dir: str | Path,
        *,
        stem: str = "som_global",
    ) -> dict[str, str]:
        """Write global SVG, PNG, and marker metadata artifacts.

        Args:
            map_data: Prior map to render.
            output_dir: Destination directory.
            stem: File stem for the artifact set.

        Returns:
            Dictionary with ``svg``, ``png``, and ``markers`` paths.
        """

        return self._write_view_artifacts(self.render_global_view(map_data), output_dir, stem=stem)

    def write_room_artifacts(
        self,
        map_data: PriorMapData,
        room_uid: str,
        output_dir: str | Path,
        *,
        stem: Optional[str] = None,
    ) -> dict[str, str]:
        """Write room-level SVG, PNG, and marker metadata artifacts.

        Args:
            map_data: Prior map to render.
            room_uid: Room uid to focus.
            output_dir: Destination directory.
            stem: Optional file stem. Defaults to ``som_room_<room_uid>``.

        Returns:
            Dictionary with ``svg``, ``png``, and ``markers`` paths.
        """

        safe_room = re.sub(r"[^A-Za-z0-9]+", "_", str(room_uid)).strip("_") or "unknown"
        return self._write_view_artifacts(
            self.render_room_view(map_data, room_uid),
            output_dir,
            stem=stem or f"som_room_{safe_room}",
        )

    def _write_view_artifacts(self, view: SomView, output_dir: str | Path, *, stem: str) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        svg_path = output / f"{stem}.svg"
        png_path = output / f"{stem}.png"
        marker_path = output / f"{stem}_markers.json"
        svg_path.write_text(view.svg, encoding="utf-8")
        png_path.write_bytes(self._render_marker_png(view))
        marker_path.write_text(
            json.dumps(
                {
                    "view_type": view.view_type,
                    "view_box": list(view.view_box),
                    "legend": view.legend,
                    "markers": [marker.to_dict() for marker in view.markers],
                    "metadata": view.metadata,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"svg": str(svg_path), "png": str(png_path), "markers": str(marker_path)}

    def _render_marker_png(self, view: SomView) -> bytes:
        transform = _CanvasTransform(
            bounds=view.view_box,
            width=self.width,
            height=self.height,
            margin=self.margin,
        )
        canvas = _RasterCanvas(width=self.width, height=self.height, background=(248, 250, 252))
        for marker in view.markers:
            x_float, y_float = transform.to_canvas(marker.xy)
            x, y = int(round(x_float)), int(round(y_float))
            if marker.marker_type == "room":
                canvas.fill_rect(x - 10, y - 8, 20, 16, color=(29, 78, 216))
                canvas.stroke_rect(x - 10, y - 8, 20, 16, color=(30, 64, 175))
            else:
                canvas.fill_circle(x, y, 8, color=(194, 65, 12))
                canvas.stroke_circle(x, y, 8, color=(124, 45, 18))
        return canvas.to_png()

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


class PriorMapFloorPlanVisualizer:
    """Render FloorPlan-VLN-style prior-map views.

    The renderer uses the prior-map room polygons as floorplan regions, topology
    edges as room connectivity, prior object positions as object points, and an
    optional runtime overlay for frontiers, trajectory, target priors, and live
    detections. It is strictly diagnostic: it does not create motion goals or
    change planning authority.

    Args:
        width: SVG/PNG canvas width in pixels.
        height: SVG/PNG canvas height in pixels.
        margin: Canvas margin in pixels.
        max_objects: Maximum object points rendered in each view before target
            and live-detection pins are reinserted.
    """

    def __init__(self, width: int = 1400, height: int = 1000, margin: int = 80, max_objects: int = 800) -> None:
        """Create a floorplan visualizer.

        Args:
            width: SVG/PNG canvas width in pixels.
            height: SVG/PNG canvas height in pixels.
            margin: Canvas margin in pixels.
            max_objects: Maximum object points rendered.
        """

        self.width = int(width)
        self.height = int(height)
        self.margin = int(margin)
        self.max_objects = int(max_objects)

    def render_global_view(
        self,
        map_data: PriorMapData,
        overlay: Optional[FloorPlanOverlay] = None,
    ) -> FloorPlanView:
        """Render a global FloorPlan-VLN-style prior-map view.

        Args:
            map_data: Prior map to render.
            overlay: Optional dynamic overlay.

        Returns:
            Rendered floorplan view with SVG and marker metadata.
        """

        overlay = overlay or FloorPlanOverlay()
        rooms = tuple(sorted(map_data.rooms, key=lambda room: room.uid))
        objects = _select_floorplan_objects(map_data, rooms=rooms, overlay=overlay, max_objects=self.max_objects)
        bounds = _floorplan_bounds(map_data, rooms, objects, overlay)
        markers = _floorplan_markers(map_data, rooms, objects, overlay)
        svg = self._render_svg(
            title=f"Floorplan prior map: {map_data.scene_id}",
            map_data=map_data,
            rooms=rooms,
            objects=objects,
            overlay=overlay,
            bounds=bounds,
        )
        return FloorPlanView(
            svg=svg,
            markers=tuple(markers),
            legend=_floorplan_legend(),
            view_type="floorplan_global",
            view_box=bounds,
            overlay=overlay,
            metadata={
                "scene_id": map_data.scene_id,
                "source_format": map_data.source_format,
                "frame_id": map_data.frame_id,
                "room_count": len(rooms),
                "object_count": len(objects),
                "topology_edge_count": len(map_data.topology_edges),
                "authority": "diagnostic_only",
            },
        )

    def render_room_view(
        self,
        map_data: PriorMapData,
        room_uid: str,
        overlay: Optional[FloorPlanOverlay] = None,
    ) -> FloorPlanView:
        """Render a room-level FloorPlan-VLN-style view.

        Args:
            map_data: Prior map to render.
            room_uid: Room uid to focus.
            overlay: Optional dynamic overlay.

        Returns:
            Rendered floorplan room view.

        Raises:
            ValueError: If ``room_uid`` is absent.
        """

        room = map_data.room_by_uid(room_uid)
        if room is None:
            raise ValueError(f"Unknown prior room uid: {room_uid}")
        overlay = overlay or FloorPlanOverlay()
        rooms = (room,)
        objects = _select_floorplan_objects(map_data, rooms=rooms, overlay=overlay, max_objects=self.max_objects)
        bounds = _floorplan_bounds(map_data, rooms, objects, overlay)
        markers = _floorplan_markers(map_data, rooms, objects, overlay)
        svg = self._render_svg(
            title=f"Floorplan prior room: {room.label or room.uid}",
            map_data=map_data,
            rooms=rooms,
            objects=objects,
            overlay=overlay,
            bounds=bounds,
        )
        return FloorPlanView(
            svg=svg,
            markers=tuple(markers),
            legend=_floorplan_legend(),
            view_type="floorplan_room",
            view_box=bounds,
            overlay=overlay,
            metadata={
                "scene_id": map_data.scene_id,
                "room_uid": room_uid,
                "source_format": map_data.source_format,
                "frame_id": map_data.frame_id,
                "object_count": len(objects),
                "authority": "diagnostic_only",
            },
        )

    def write_global_artifacts(
        self,
        map_data: PriorMapData,
        output_dir: str | Path,
        *,
        stem: str = "floorplan_global",
        overlay: Optional[FloorPlanOverlay] = None,
    ) -> dict[str, str]:
        """Write global floorplan SVG, PNG, and marker metadata artifacts.

        Args:
            map_data: Prior map to render.
            output_dir: Destination directory.
            stem: File stem for artifact paths.
            overlay: Optional dynamic overlay.

        Returns:
            Dictionary with ``svg``, ``png``, and ``markers`` paths.
        """

        return self._write_view_artifacts(self.render_global_view(map_data, overlay=overlay), output_dir, stem=stem)

    def write_room_artifacts(
        self,
        map_data: PriorMapData,
        room_uid: str,
        output_dir: str | Path,
        *,
        stem: Optional[str] = None,
        overlay: Optional[FloorPlanOverlay] = None,
    ) -> dict[str, str]:
        """Write room-level floorplan SVG, PNG, and marker metadata artifacts.

        Args:
            map_data: Prior map to render.
            room_uid: Room uid to focus.
            output_dir: Destination directory.
            stem: Optional file stem. Defaults to ``floorplan_room_<room_uid>``.
            overlay: Optional dynamic overlay.

        Returns:
            Dictionary with ``svg``, ``png``, and ``markers`` paths.
        """

        safe_room = re.sub(r"[^A-Za-z0-9]+", "_", str(room_uid)).strip("_") or "unknown"
        return self._write_view_artifacts(
            self.render_room_view(map_data, room_uid, overlay=overlay),
            output_dir,
            stem=stem or f"floorplan_room_{safe_room}",
        )

    def _write_view_artifacts(self, view: FloorPlanView, output_dir: str | Path, *, stem: str) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        svg_path = output / f"{stem}.svg"
        png_path = output / f"{stem}.png"
        marker_path = output / f"{stem}_markers.json"
        svg_path.write_text(view.svg, encoding="utf-8")
        png_path.write_bytes(self._render_png(view))
        marker_path.write_text(
            json.dumps(
                {
                    "view_type": view.view_type,
                    "view_box": list(view.view_box),
                    "legend": view.legend,
                    "markers": list(view.markers),
                    "overlay": view.overlay.to_dict(),
                    "metadata": view.metadata,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"svg": str(svg_path), "png": str(png_path), "markers": str(marker_path)}

    def _render_svg(
        self,
        *,
        title: str,
        map_data: PriorMapData,
        rooms: Sequence[PriorRoom],
        objects: Sequence[PriorObject],
        overlay: FloorPlanOverlay,
        bounds: Tuple[float, float, float, float],
    ) -> str:
        transform = _CanvasTransform(bounds=bounds, width=self.width, height=self.height, margin=self.margin)
        visible_room_uids = {room.uid for room in rooms}
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">',
            f"  <title>{escape(title)}</title>",
            '  <rect x="0" y="0" width="100%" height="100%" fill="#f7f8fa"/>',
            f'  <text x="24" y="34" font-family="sans-serif" font-size="24" font-weight="700" '
            f'fill="#111827">{escape(title)} | rooms={len(rooms)} objects={len(objects)} '
            f'edges={len(map_data.topology_edges)}</text>',
        ]
        for edge in _visible_room_edges(map_data, visible_room_uids):
            a = _room_anchor(map_data.room_by_uid(edge.source_uid)) if map_data.room_by_uid(edge.source_uid) else None
            b = _room_anchor(map_data.room_by_uid(edge.target_uid)) if map_data.room_by_uid(edge.target_uid) else None
            if a is None or b is None:
                continue
            ax, ay = transform.to_canvas(a)
            bx, by = transform.to_canvas(b)
            lines.append(
                f'  <line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                f'stroke="#4b5563" stroke-width="2" opacity="0.55"/>'
            )
        for room in rooms:
            if len(room.boundary_xy) >= 3:
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.to_canvas(point) for point in room.boundary_xy))
                fill = _room_fill(room.uid)
                lines.append(
                    f'  <polygon points="{points}" fill="{fill}" stroke="#334155" '
                    f'stroke-width="2" opacity="0.72"/>'
                )
        for obj in objects:
            anchor = _object_floorplan_anchor(obj, map_data)
            if anchor is None:
                continue
            x, y = transform.to_canvas(anchor)
            is_target = obj.uid in set(overlay.target_prior_object_uids)
            fill = "#dc2626" if is_target else "#64748b"
            radius = 7 if is_target else 2.5
            opacity = "1.0" if is_target else "0.42"
            lines.append(
                f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
                f'stroke="#111827" stroke-width="{1.5 if is_target else 0.0}" opacity="{opacity}"/>'
            )
            if is_target:
                lines.append(
                    f'  <text x="{x + 12:.1f}" y="{y - 8:.1f}" font-family="sans-serif" '
                    f'font-size="12" font-weight="700" fill="#991b1b">{escape(obj.label)} {escape(obj.uid)}</text>'
                )
        for point in overlay.live_detections:
            x, y = transform.to_canvas(point.xy)
            lines.append(
                f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="none" stroke="#16a34a" '
                f'stroke-width="3" opacity="0.95"/>'
            )
        if len(overlay.trajectory_xy) >= 2:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (transform.to_canvas(point) for point in overlay.trajectory_xy))
            lines.append(
                f'  <polyline points="{points}" fill="none" stroke="#0ea5e9" stroke-width="3" '
                f'stroke-linecap="round" stroke-linejoin="round" opacity="0.86"/>'
            )
        for point in overlay.frontiers:
            x, y = transform.to_canvas(point.xy)
            fill = "#7c3aed" if point.selected else "#f59e0b"
            stroke = "#111827" if point.selected else "#92400e"
            size = 10 if point.selected else 7
            triangle = [
                (x, y - size),
                (x - size * 0.9, y + size * 0.75),
                (x + size * 0.9, y + size * 0.75),
            ]
            points_svg = " ".join(f"{px:.1f},{py:.1f}" for px, py in triangle)
            lines.append(
                f'  <polygon points="{points_svg}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{2 if point.selected else 1}" opacity="0.96"/>'
            )
            if point.selected:
                lines.append(
                    f'  <text x="{x + 12:.1f}" y="{y + 4:.1f}" font-family="sans-serif" '
                    f'font-size="12" font-weight="700" fill="#581c87">selected {escape(point.uid)}</text>'
                )
        for index, room in enumerate(rooms):
            anchor = _room_anchor(room)
            if anchor is None:
                continue
            x, y = transform.to_canvas(anchor)
            label = room.label or room.uid
            short_uid = str(room.uid).split(":")[-1]
            lines.append(
                f'  <rect x="{x - 26:.1f}" y="{y - 22:.1f}" width="104" height="54" rx="5" '
                f'fill="#ffffff" stroke="#64748b" opacity="0.84"/>'
            )
            lines.append(
                f'  <text x="{x - 20:.1f}" y="{y - 8:.1f}" font-family="sans-serif" font-size="12" '
                f'fill="#0f172a">{index}</text>'
            )
            lines.append(
                f'  <text x="{x - 20:.1f}" y="{y + 7:.1f}" font-family="sans-serif" font-size="12" '
                f'fill="#0f172a">{escape(label)}</text>'
            )
            lines.append(
                f'  <text x="{x - 20:.1f}" y="{y + 22:.1f}" font-family="sans-serif" font-size="11" '
                f'fill="#334155">{escape(short_uid)}</text>'
            )
        lines.extend(_floorplan_legend_svg(self.width, self.height))
        lines.append("</svg>")
        return "\n".join(lines)

    def _render_png(self, view: FloorPlanView) -> bytes:
        transform = _CanvasTransform(bounds=view.view_box, width=self.width, height=self.height, margin=self.margin)
        canvas = _RasterCanvas(width=self.width, height=self.height, background=(247, 248, 250))
        map_data = view.metadata.get("_map_data")
        # The map object is intentionally not serialized into metadata. Rebuild
        # enough render input from marker payloads for PNG output.
        for marker in view.markers:
            if marker.get("marker_type") != "room":
                continue
            boundary = marker.get("boundary_xy") or []
            if len(boundary) < 3:
                continue
            points = [tuple(int(round(value)) for value in transform.to_canvas((float(x), float(y)))) for x, y in boundary]
            canvas.fill_polygon(points, color=(196, 213, 234))
            canvas.stroke_polyline(points + [points[0]], color=(51, 65, 85), width=2)
        room_centers = {
            marker.get("uid"): tuple(marker.get("xy", ()))
            for marker in view.markers
            if marker.get("marker_type") == "room" and len(marker.get("xy", ())) == 2
        }
        for marker in view.markers:
            if marker.get("marker_type") != "room_edge":
                continue
            source_xy = room_centers.get(marker.get("source_uid"))
            target_xy = room_centers.get(marker.get("target_uid"))
            if source_xy is None or target_xy is None:
                continue
            canvas.draw_line(
                _int_point(transform.to_canvas(source_xy)),
                _int_point(transform.to_canvas(target_xy)),
                color=(75, 85, 99),
                width=2,
            )
        for marker in view.markers:
            if marker.get("marker_type") != "object":
                continue
            xy = marker.get("xy")
            if not xy or len(xy) != 2:
                continue
            point = _int_point(transform.to_canvas((float(xy[0]), float(xy[1]))))
            if marker.get("target_prior"):
                canvas.fill_circle(point[0], point[1], 8, color=(220, 38, 38))
                canvas.stroke_circle(point[0], point[1], 8, color=(127, 29, 29))
            else:
                canvas.fill_circle(point[0], point[1], 2, color=(100, 116, 139))
        if len(view.overlay.trajectory_xy) >= 2:
            canvas.stroke_polyline(
                [_int_point(transform.to_canvas(point)) for point in view.overlay.trajectory_xy],
                color=(14, 165, 233),
                width=3,
            )
        for point in view.overlay.live_detections:
            x, y = _int_point(transform.to_canvas(point.xy))
            canvas.stroke_circle(x, y, 9, color=(22, 163, 74))
            canvas.stroke_circle(x, y, 7, color=(22, 163, 74))
        for point in view.overlay.frontiers:
            x, y = transform.to_canvas(point.xy)
            size = 10 if point.selected else 7
            triangle = [
                (int(round(x)), int(round(y - size))),
                (int(round(x - size * 0.9)), int(round(y + size * 0.75))),
                (int(round(x + size * 0.9)), int(round(y + size * 0.75))),
            ]
            canvas.fill_polygon(triangle, color=(124, 58, 237) if point.selected else (245, 158, 11))
            canvas.stroke_polyline(triangle + [triangle[0]], color=(17, 24, 39), width=2 if point.selected else 1)
        return canvas.to_png()


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


def render_floorplan_global_view(
    map_data: PriorMapData,
    *,
    overlay: Optional[FloorPlanOverlay] = None,
    width: int = 1400,
    height: int = 1000,
) -> FloorPlanView:
    """Render a global FloorPlan-VLN-style prior-map view.

    Args:
        map_data: Prior map to render.
        overlay: Optional dynamic overlay.
        width: SVG width.
        height: SVG height.

    Returns:
        Rendered floorplan view.
    """

    return PriorMapFloorPlanVisualizer(width=width, height=height).render_global_view(map_data, overlay=overlay)


def render_floorplan_room_view(
    map_data: PriorMapData,
    room_uid: str,
    *,
    overlay: Optional[FloorPlanOverlay] = None,
    width: int = 1400,
    height: int = 1000,
) -> FloorPlanView:
    """Render a room-level FloorPlan-VLN-style prior-map view.

    Args:
        map_data: Prior map to render.
        room_uid: Room uid to focus.
        overlay: Optional dynamic overlay.
        width: SVG width.
        height: SVG height.

    Returns:
        Rendered floorplan room view.
    """

    return PriorMapFloorPlanVisualizer(width=width, height=height).render_room_view(
        map_data,
        room_uid,
        overlay=overlay,
    )


def write_som_artifacts(
    map_data: PriorMapData,
    output_dir: str | Path,
    *,
    max_room_views: int = 4,
    width: int = 900,
    height: int = 600,
) -> dict[str, Any]:
    """Write global and selected room SoM artifacts.

    Args:
        map_data: Prior map to render.
        output_dir: Destination directory.
        max_room_views: Maximum room-level views to write.
        width: SVG/PNG width.
        height: SVG/PNG height.

    Returns:
        Dictionary containing artifact path groups.
    """

    visualizer = PriorMapSomVisualizer(width=width, height=height)
    artifacts = {
        "global": visualizer.write_global_artifacts(map_data, output_dir),
        "rooms": {},
    }
    for room in sorted(map_data.rooms, key=lambda item: item.uid)[: max(0, int(max_room_views))]:
        artifacts["rooms"][room.uid] = visualizer.write_room_artifacts(map_data, room.uid, output_dir)
    return artifacts


def write_floorplan_artifacts(
    map_data: PriorMapData,
    output_dir: str | Path,
    *,
    max_room_views: int = 4,
    width: int = 1400,
    height: int = 1000,
    overlay: Optional[FloorPlanOverlay] = None,
) -> dict[str, Any]:
    """Write global and selected room floorplan artifacts.

    Args:
        map_data: Prior map to render.
        output_dir: Destination directory.
        max_room_views: Maximum room-level views to write.
        width: SVG/PNG width.
        height: SVG/PNG height.
        overlay: Optional dynamic overlay.

    Returns:
        Dictionary containing artifact path groups.
    """

    visualizer = PriorMapFloorPlanVisualizer(width=width, height=height)
    artifacts = {
        "global": visualizer.write_global_artifacts(map_data, output_dir, overlay=overlay),
        "rooms": {},
    }
    for room in sorted(map_data.rooms, key=lambda item: item.uid)[: max(0, int(max_room_views))]:
        artifacts["rooms"][room.uid] = visualizer.write_room_artifacts(
            map_data,
            room.uid,
            output_dir,
            overlay=overlay,
        )
    return artifacts


def build_floorplan_overlay(
    map_data: PriorMapData,
    *,
    prior_result: Any = None,
    chosen_frontier: Optional[Mapping[str, Any]] = None,
    observations: Sequence[Any] = (),
    object_states: Optional[Mapping[str, Any]] = None,
) -> FloorPlanOverlay:
    """Build a diagnostic floorplan overlay from runtime artifacts.

    Args:
        map_data: Prior map that owns target prior object coordinates.
        prior_result: Optional ``SearchPriorResult``-like ranking result.
        chosen_frontier: Optional active planner chosen-frontier payload.
        observations: Runtime observation records with optional pose.
        object_states: Runtime prior-object states keyed by prior object uid.

    Returns:
        Floorplan overlay that can be rendered without changing planner
        authority.
    """

    target_uids = _target_prior_uids(prior_result)
    frontier_points = _frontier_overlay_points(prior_result=prior_result, chosen_frontier=chosen_frontier)
    live_points = _live_detection_overlay_points(map_data, object_states or {})
    trajectory = _trajectory_overlay_points(observations)
    selected_uid = _text_or_none(_mapping_get(chosen_frontier, "selected_uid")) if chosen_frontier else None
    baseline_uid = _text_or_none(_mapping_get(chosen_frontier, "baseline_selected_uid")) if chosen_frontier else None
    return FloorPlanOverlay(
        target_prior_object_uids=tuple(target_uids),
        frontiers=tuple(frontier_points),
        live_detections=tuple(live_points),
        trajectory_xy=tuple(trajectory),
        selected_frontier_uid=selected_uid,
        baseline_frontier_uid=baseline_uid,
        metadata={
            "authority": "diagnostic_only",
            "chosen_frontier_available": chosen_frontier is not None,
            "frontier_count": len(frontier_points),
            "live_detection_count": len(live_points),
            "trajectory_point_count": len(trajectory),
        },
    )


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


def _floorplan_bounds(
    map_data: PriorMapData,
    rooms: Sequence[PriorRoom],
    objects: Sequence[PriorObject],
    overlay: FloorPlanOverlay,
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
        anchor = _object_floorplan_anchor(obj, map_data)
        if anchor is not None:
            points.append(anchor)
    for point in overlay.frontiers:
        points.append(point.xy)
    for point in overlay.live_detections:
        points.append(point.xy)
    points.extend(overlay.trajectory_xy)
    if not points:
        return (0.0, 0.0, 1.0, 1.0)
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    pad_x = max((max_x - min_x) * 0.08, 0.5)
    pad_y = max((max_y - min_y) * 0.08, 0.5)
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def _select_floorplan_objects(
    map_data: PriorMapData,
    *,
    rooms: Sequence[PriorRoom],
    overlay: FloorPlanOverlay,
    max_objects: int,
) -> Tuple[PriorObject, ...]:
    visible_room_uids = {room.uid for room in rooms}
    is_global_view = len(visible_room_uids) >= len(map_data.rooms)
    objects = [
        obj
        for obj in sorted(map_data.objects, key=lambda item: item.uid)
        if is_global_view or not visible_room_uids or obj.parent_room_uid in visible_room_uids
    ]
    selected = list(objects[: max(0, int(max_objects))])
    required_uids = set(overlay.target_prior_object_uids)
    required_uids.update(point.uid for point in overlay.live_detections)
    selected_uids = {obj.uid for obj in selected}
    for obj in objects:
        if obj.uid in required_uids and obj.uid not in selected_uids:
            selected.append(obj)
            selected_uids.add(obj.uid)
    return tuple(selected)


def _floorplan_markers(
    map_data: PriorMapData,
    rooms: Sequence[PriorRoom],
    objects: Sequence[PriorObject],
    overlay: FloorPlanOverlay,
) -> List[dict[str, Any]]:
    markers: List[dict[str, Any]] = []
    visible_room_uids = {room.uid for room in rooms}
    for room in rooms:
        anchor = _room_anchor(room)
        markers.append(
            {
                "marker_type": "room",
                "uid": room.uid,
                "label": room.label,
                "xy": list(anchor) if anchor is not None else None,
                "boundary_xy": [list(point) for point in room.boundary_xy],
                "neighbors": list(room.neighbors),
                "confidence": room.confidence,
            }
        )
    for edge in _visible_room_edges(map_data, visible_room_uids):
        markers.append(
            {
                "marker_type": "room_edge",
                "uid": edge.uid,
                "source_uid": edge.source_uid,
                "target_uid": edge.target_uid,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "weight": edge.weight,
            }
        )
    live_uids = {point.uid for point in overlay.live_detections}
    target_uids = set(overlay.target_prior_object_uids)
    for obj in objects:
        anchor = _object_floorplan_anchor(obj, map_data)
        markers.append(
            {
                "marker_type": "object",
                "uid": obj.uid,
                "label": obj.label,
                "xy": list(anchor) if anchor is not None else None,
                "parent_room_uid": obj.parent_room_uid,
                "exact": obj.exact,
                "confidence": obj.confidence,
                "target_prior": obj.uid in target_uids,
                "live_detected": obj.uid in live_uids,
            }
        )
    for point in overlay.frontiers:
        payload = point.to_dict()
        payload["marker_type"] = "frontier"
        markers.append(payload)
    for point in overlay.live_detections:
        payload = point.to_dict()
        payload["marker_type"] = "live_detection"
        markers.append(payload)
    return markers


def _visible_room_edges(map_data: PriorMapData, visible_room_uids: set[str]) -> Tuple[Any, ...]:
    return tuple(
        edge
        for edge in map_data.topology_edges
        if edge.edge_type == "room-room"
        and edge.source_uid in visible_room_uids
        and edge.target_uid in visible_room_uids
    )


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


def _object_floorplan_anchor(obj: PriorObject, map_data: PriorMapData) -> Optional[Tuple[float, float]]:
    if obj.position_xyz is None:
        return None
    return _plane_xy_from_xyz(obj.position_xyz, frame_id=map_data.frame_id, source_format=map_data.source_format)


def _plane_xy_from_xyz(
    position_xyz: Sequence[float],
    *,
    frame_id: str = "",
    source_format: str = "",
) -> Tuple[float, float]:
    point = _vector3_or_none(position_xyz) or (0.0, 0.0, 0.0)
    frame = str(frame_id or "").lower()
    source = str(source_format or "").lower()
    plane_axes = "xz" if "hm3d" in source or ("habi" + "tat") in frame else "xy"
    if plane_axes == "xz":
        return (point[0], point[2])
    return (point[0], point[1])


def _target_prior_uids(prior_result: Any) -> Tuple[str, ...]:
    uids: List[str] = []
    for item in _iter_records(_first_attr(prior_result, ("object_rankings",), ())):
        uid = _text_or_none(_first_attr(item, ("object_uid", "uid"), None))
        if uid and _is_target_object_prior(item):
            uids.append(uid)
    if uids:
        return tuple(_dedupe(uids))
    first = next(iter(_iter_records(_first_attr(prior_result, ("object_rankings",), ()))), None)
    fallback_uid = _text_or_none(_first_attr(first, ("object_uid", "uid"), None))
    return (fallback_uid,) if fallback_uid else ()


def _is_target_object_prior(item: Any) -> bool:
    metadata = _first_attr(item, ("metadata",), {}) or {}
    components = _mapping_get(metadata, "score_components", {}) or {}
    concept_relevance = _safe_float(_mapping_get(components, "concept_relevance", 0.0))
    if concept_relevance > 0.0:
        return True
    reason = str(_first_attr(item, ("reason",), "") or "").lower()
    return "target concept" in reason or "matches target" in reason


def _frontier_overlay_points(
    *,
    prior_result: Any,
    chosen_frontier: Optional[Mapping[str, Any]],
) -> List[FloorPlanOverlayPoint]:
    bias_by_uid: dict[str, Any] = {}
    for bias in _iter_records(_first_attr(prior_result, ("frontier_biases",), ())):
        uid = _text_or_none(_first_attr(bias, ("frontier_uid", "uid"), None))
        if uid:
            bias_by_uid[uid] = bias
    selected_uid = _text_or_none(_mapping_get(chosen_frontier, "selected_uid")) if chosen_frontier else None
    candidates = _mapping_get(chosen_frontier, "candidates", ()) if chosen_frontier else ()
    points: List[FloorPlanOverlayPoint] = []
    for index, candidate in enumerate(_iter_records(candidates)):
        uid = _text_or_none(_first_attr(candidate, ("frontier_uid", "uid", "id"), None)) or f"frontier_{index}"
        xy = _candidate_frontier_xy(candidate, bias_by_uid.get(uid))
        if xy is None:
            continue
        points.append(
            FloorPlanOverlayPoint(
                uid=uid,
                label=f"frontier {uid}",
                xy=xy,
                point_type="frontier",
                selected=uid == selected_uid,
                metadata={
                    "rank_input": _first_attr(candidate, ("rank_input",), None),
                    "node_idx": _first_attr(candidate, ("node_idx",), None),
                    "raw_distance_m": _first_attr(candidate, ("raw_distance_m",), None),
                    "prior_score": _first_attr(candidate, ("prior_score",), None),
                    "adjusted_distance_m": _first_attr(candidate, ("adjusted_distance_m",), None),
                    "source": "chosen_frontier_payload",
                },
            )
        )
    if points:
        return points
    for bias in bias_by_uid.values():
        uid = _text_or_none(_first_attr(bias, ("frontier_uid", "uid"), None))
        xy = _candidate_frontier_xy({}, bias)
        if uid and xy is not None:
            points.append(
                FloorPlanOverlayPoint(
                    uid=uid,
                    label=f"frontier {uid}",
                    xy=xy,
                    point_type="frontier",
                    selected=uid == selected_uid,
                    metadata={"source": "frontier_bias"},
                )
            )
    return points


def _candidate_frontier_xy(candidate: Any, bias: Any) -> Optional[Tuple[float, float]]:
    metadata = _first_attr(bias, ("metadata",), {}) or {}
    prior_xy = _mapping_get(metadata, "frontier_prior_xy")
    vector2 = _vector2_or_none(prior_xy)
    if vector2 is not None:
        return vector2
    position = _first_attr(candidate, ("position", "position_xyz", "world_position_xyz"), None)
    vector = _vector3_or_none(position)
    if vector is not None:
        return (vector[0], vector[1])
    return None


def _live_detection_overlay_points(
    map_data: PriorMapData,
    object_states: Mapping[str, Any],
) -> List[FloorPlanOverlayPoint]:
    points: List[FloorPlanOverlayPoint] = []
    for uid, state in sorted(object_states.items(), key=lambda item: str(item[0])):
        observed = _safe_int(_first_attr(state, ("observation_count",), 0)) > 0
        verified = bool(_first_attr(state, ("verified",), False))
        if not observed and not verified:
            continue
        obj = map_data.object_by_uid(str(uid))
        if obj is None:
            continue
        anchor = _object_floorplan_anchor(obj, map_data)
        if anchor is None:
            continue
        points.append(
            FloorPlanOverlayPoint(
                uid=obj.uid,
                label=obj.label,
                xy=anchor,
                point_type="live_detection",
                selected=verified,
                metadata={
                    "observation_count": _first_attr(state, ("observation_count",), 0),
                    "matched_runtime_uid": _first_attr(state, ("matched_runtime_uid",), None),
                    "verified": verified,
                },
            )
        )
    return points


def _trajectory_overlay_points(observations: Sequence[Any]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for record in observations:
        pose = _first_attr(record, ("pose_xyz",), None)
        vector = _vector3_or_none(pose)
        if vector is None:
            continue
        frame_id = _text_or_none(_first_attr(record, ("frame_id",), "")) or ""
        points.append(_plane_xy_from_xyz(vector, frame_id=frame_id, source_format="runtime"))
    return points


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


def _floorplan_legend() -> dict[str, str]:
    return {
        "room polygon": "prior room boundary or navigable component",
        "gray line": "room-room topology edge",
        "gray dot": "prior object position",
        "red dot": "target prior object",
        "green ring": "live detection matched to prior object",
        "blue line": "runtime trajectory",
        "orange triangle": "runtime frontier candidate",
        "purple triangle": "selected frontier",
        "authority": "diagnostic only; prior map does not generate goals or stop decisions",
    }


def _floorplan_legend_svg(width: int, height: int) -> List[str]:
    x = 18
    y = max(height - 126, 18)
    return [
        f'  <g id="floorplan-legend" transform="translate({x},{y})">',
        '    <rect x="0" y="0" width="430" height="108" rx="6" fill="#ffffff" stroke="#94a3b8" opacity="0.9"/>',
        '    <rect x="10" y="12" width="28" height="18" fill="#c4d5ea" stroke="#334155"/>',
        '    <text x="50" y="26" font-family="sans-serif" font-size="12" fill="#0f172a">prior room boundary / component</text>',
        '    <line x1="10" y1="46" x2="38" y2="46" stroke="#4b5563" stroke-width="3"/>',
        '    <text x="50" y="50" font-family="sans-serif" font-size="12" fill="#0f172a">room-room topology edge</text>',
        '    <circle cx="24" cy="68" r="8" fill="#dc2626" stroke="#7f1d1d"/>',
        '    <text x="50" y="72" font-family="sans-serif" font-size="12" fill="#0f172a">target prior object</text>',
        '    <polygon points="19,91 12,104 26,104" fill="#7c3aed" stroke="#111827"/>',
        '    <text x="50" y="101" font-family="sans-serif" font-size="12" fill="#0f172a">selected frontier / runtime overlay</text>',
        "  </g>",
    ]


def _room_fill(uid: str) -> str:
    palette = ("#bfdbfe", "#bae6fd", "#bbf7d0", "#fde68a", "#fecaca", "#ddd6fe", "#c7d2fe", "#fed7aa")
    checksum = sum(ord(ch) for ch in str(uid))
    return palette[checksum % len(palette)]


def _int_point(point: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _vector2_or_none(value: Any) -> Optional[Tuple[float, float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_safe_float(value[0]), _safe_float(value[1]))
    return None


def _vector3_or_none(value: Any) -> Optional[Tuple[float, float, float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        z = value[2] if len(value) >= 3 else 0.0
        return (_safe_float(value[0]), _safe_float(value[1]), _safe_float(z))
    return None


def _iter_records(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _first_attr(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _mapping_get(value: Optional[Mapping[str, Any]], key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def _text_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


class _RasterCanvas:
    def __init__(self, *, width: int, height: int, background: tuple[int, int, int]) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.pixels = bytearray(background * (self.width * self.height))

    def fill_rect(self, x: int, y: int, width: int, height: int, *, color: tuple[int, int, int]) -> None:
        for yy in range(max(0, y), min(self.height, y + height)):
            for xx in range(max(0, x), min(self.width, x + width)):
                self._set(xx, yy, color)

    def stroke_rect(self, x: int, y: int, width: int, height: int, *, color: tuple[int, int, int]) -> None:
        for xx in range(x, x + width):
            self._set(xx, y, color)
            self._set(xx, y + height - 1, color)
        for yy in range(y, y + height):
            self._set(x, yy, color)
            self._set(x + width - 1, yy, color)

    def fill_circle(self, x: int, y: int, radius: int, *, color: tuple[int, int, int]) -> None:
        rr = radius * radius
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= rr:
                    self._set(xx, yy, color)

    def stroke_circle(self, x: int, y: int, radius: int, *, color: tuple[int, int, int]) -> None:
        outer = radius * radius
        inner = max(0, radius - 1) * max(0, radius - 1)
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                dist = (xx - x) * (xx - x) + (yy - y) * (yy - y)
                if inner <= dist <= outer:
                    self._set(xx, yy, color)

    def draw_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a Bresenham line segment.

        Args:
            start: Start pixel.
            end: End pixel.
            color: RGB color.
            width: Stroke width in pixels.
        """

        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        radius = max(0, int(width) // 2)
        while True:
            self._paint_kernel(x0, y0, radius, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def stroke_polyline(
        self,
        points: Sequence[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a connected polyline.

        Args:
            points: Pixel points.
            color: RGB stroke color.
            width: Stroke width in pixels.
        """

        for start, end in zip(points, points[1:]):
            self.draw_line(start, end, color=color, width=width)

    def fill_polygon(self, points: Sequence[tuple[int, int]], *, color: tuple[int, int, int]) -> None:
        """Fill a simple polygon using scanline intersections.

        Args:
            points: Polygon vertices in pixel coordinates.
            color: RGB fill color.
        """

        if len(points) < 3:
            return
        min_y = max(0, min(y for _, y in points))
        max_y = min(self.height - 1, max(y for _, y in points))
        edges = list(zip(points, points[1:] + points[:1]))
        for y in range(min_y, max_y + 1):
            intersections: List[float] = []
            for (x0, y0), (x1, y1) in edges:
                if y0 == y1:
                    continue
                low_y = min(y0, y1)
                high_y = max(y0, y1)
                if not (low_y <= y < high_y):
                    continue
                t = (y - y0) / (y1 - y0)
                intersections.append(x0 + t * (x1 - x0))
            intersections.sort()
            for left, right in zip(intersections[0::2], intersections[1::2]):
                x_start = max(0, int(round(left)))
                x_end = min(self.width - 1, int(round(right)))
                for x in range(x_start, x_end + 1):
                    self._set(x, y, color)

    def to_png(self) -> bytes:
        raw_rows = []
        for y in range(self.height):
            start = y * self.width * 3
            end = start + self.width * 3
            raw_rows.append(b"\x00" + bytes(self.pixels[start:end]))
        compressed = zlib.compress(b"".join(raw_rows), level=6)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", compressed)
            + _png_chunk(b"IEND", b"")
        )

    def _set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if not 0 <= x < self.width or not 0 <= y < self.height:
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset : offset + 3] = bytes(color)

    def _paint_kernel(self, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                self._set(xx, yy, color)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum & 0xFFFFFFFF)


__all__ = [
    "FloorPlanOverlay",
    "FloorPlanOverlayPoint",
    "FloorPlanView",
    "PriorMapFloorPlanVisualizer",
    "PriorMapSomVisualizer",
    "SomMarker",
    "SomView",
    "build_floorplan_overlay",
    "render_floorplan_global_view",
    "render_floorplan_room_view",
    "render_global_view",
    "render_room_view",
    "write_floorplan_artifacts",
    "write_som_artifacts",
]
