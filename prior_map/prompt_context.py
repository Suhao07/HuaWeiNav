"""Prompt-context rendering for prior-map mode.

This module converts prior maps and query results into compact text snippets
for LLM/VLM prompts. The snippets are context only: they do not authorize
motion goals, waypoint publication, or final stop decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable, List, Optional, Sequence

from .contracts import PriorMapData, PriorObject, PriorRoom, PriorTopologyEdge, SearchPriorResult


@dataclass(frozen=True)
class PromptContextBundle:
    """Rendered prior-map prompt context.

    Args:
        natural_language: Human-readable summary of the map.
        compact_xml: OSM-like compact XML summary.
        search_prior_summary: Prompt-friendly summary of current query output.
        truncated: Whether any field was truncated by the character budget.
        multimodal_context: Optional dynamic BEV image metadata. The bytes are
            packaged only by the high-level selector.
        metadata: JSON-friendly diagnostics such as counts and character limits.
    """

    natural_language: str
    compact_xml: str
    search_prior_summary: str = ""
    truncated: bool = False
    multimodal_context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary containing rendered context and diagnostics.
        """

        return {
            "natural_language": self.natural_language,
            "compact_xml": self.compact_xml,
            "search_prior_summary": self.search_prior_summary,
            "truncated": self.truncated,
            "multimodal_context": dict(self.multimodal_context or {}),
            "metadata": dict(self.metadata or {}),
        }


class PriorMapPromptContextBuilder:
    """Build bounded prompt context from prior-map data.

    Args:
        max_chars: Maximum characters per rendered field.
        max_rooms: Maximum rooms included in summaries.
        max_objects: Maximum objects included in summaries.
        max_edges: Maximum topology edges included in XML summaries.
        max_rankings: Maximum rankings included in query summaries.
    """

    def __init__(
        self,
        max_chars: int = 4000,
        max_rooms: int = 12,
        max_objects: int = 24,
        max_edges: int = 32,
        max_rankings: int = 6,
    ) -> None:
        """Create a prompt context builder.

        Args:
            max_chars: Maximum characters per rendered field.
            max_rooms: Maximum rooms included in summaries.
            max_objects: Maximum objects included in summaries.
            max_edges: Maximum topology edges included in XML summaries.
            max_rankings: Maximum rankings included in query summaries.
        """

        self.max_chars = int(max_chars)
        self.max_rooms = int(max_rooms)
        self.max_objects = int(max_objects)
        self.max_edges = int(max_edges)
        self.max_rankings = int(max_rankings)

    def summarize_map(self, map_data: PriorMapData) -> str:
        """Render a natural-language prior-map summary.

        Args:
            map_data: Prior map to summarize.

        Returns:
            Bounded human-readable summary.
        """

        lines = [
            (
                f"Prior map for scene {map_data.scene_id}: "
                f"{len(map_data.rooms)} rooms, {len(map_data.objects)} objects, "
                f"{len(map_data.topology_edges)} topology edges; "
                f"source={map_data.source_format}, frame={map_data.frame_id}."
            ),
            "Use this map only as soft search context. Live observations and final verification remain authoritative.",
        ]
        rooms = sorted(map_data.rooms, key=lambda room: room.uid)[: max(0, self.max_rooms)]
        objects_by_room = _objects_by_room(map_data.objects)
        for room in rooms:
            room_objects = objects_by_room.get(room.uid, ())[:4]
            object_text = ", ".join(_object_label(obj) for obj in room_objects) or "no listed objects"
            lines.append(
                "- Room "
                f"{room.uid} ({room.label}, conf={room.confidence:.2f}): "
                f"centroid={_format_xy(room.centroid_xy)}, neighbors={_join_or_none(room.neighbors)}, "
                f"objects={object_text}."
            )

        unassigned = [obj for obj in map_data.objects if not obj.parent_room_uid]
        listed_room_object_uids = {
            obj.uid
            for room in rooms
            for obj in objects_by_room.get(room.uid, ())[:4]
        }
        extra_objects = [
            obj
            for obj in sorted(map_data.objects, key=lambda item: item.uid)
            if obj.uid not in listed_room_object_uids and obj not in unassigned
        ][: max(0, self.max_objects)]
        if extra_objects:
            lines.append(
                "Additional prior objects: "
                + "; ".join(
                    f"{obj.uid}={_object_label(obj)} in {obj.parent_room_uid or 'unknown room'}"
                    for obj in extra_objects
                )
                + "."
            )
        if unassigned:
            lines.append(
                "Unassigned prior objects: "
                + "; ".join(f"{obj.uid}={_object_label(obj)}" for obj in unassigned[: self.max_objects])
                + "."
            )
        text, _ = _limit_text("\n".join(lines), self.max_chars)
        return text

    def to_compact_xml(self, map_data: PriorMapData) -> str:
        """Render an OSM-like compact XML map summary.

        Args:
            map_data: Prior map to render.

        Returns:
            Bounded XML-like text containing room, object, and topology tags.
        """

        lines = [
            (
                f'<prior_map scene="{_xml(map_data.scene_id)}" '
                f'source="{_xml(map_data.source_format)}" frame="{_xml(map_data.frame_id)}">'
            )
        ]
        for room in sorted(map_data.rooms, key=lambda item: item.uid)[: max(0, self.max_rooms)]:
            attrs = {
                "id": room.uid,
                "label": room.label,
                "conf": f"{room.confidence:.2f}",
                "xy": _format_xy(room.centroid_xy),
                "neighbors": ",".join(room.neighbors),
            }
            lines.append("  " + _empty_tag("room", attrs))
        for obj in sorted(map_data.objects, key=lambda item: item.uid)[: max(0, self.max_objects)]:
            attrs = {
                "id": obj.uid,
                "label": obj.label,
                "room": obj.parent_room_uid or "",
                "exact": str(bool(obj.exact)).lower(),
                "conf": f"{obj.confidence:.2f}",
                "xyz": _format_xyz(obj.position_xyz),
                "aliases": ",".join(obj.aliases),
            }
            lines.append("  " + _empty_tag("object", attrs))
        for edge in sorted(map_data.topology_edges, key=lambda item: item.uid)[: max(0, self.max_edges)]:
            attrs = {
                "id": edge.uid,
                "type": edge.edge_type,
                "rel": edge.relation,
                "from": edge.source_uid,
                "to": edge.target_uid,
                "conf": f"{edge.confidence:.2f}",
            }
            lines.append("  " + _empty_tag("edge", attrs))
        lines.append("</prior_map>")
        text, _ = _limit_text("\n".join(lines), self.max_chars)
        return text

    def summarize_search_prior(self, prior_result: Optional[SearchPriorResult]) -> str:
        """Render a prompt-friendly summary of prior query output.

        Args:
            prior_result: Search prior result from ``PriorMapQueryService``.

        Returns:
            Bounded ranking summary. Empty string when no result is available.
        """

        if prior_result is None:
            return ""

        lines = [
            "Prior search guidance is ranking-only; it cannot authorize motion goals or STOP.",
        ]
        if prior_result.room_rankings:
            lines.append("Top rooms: " + _join_rankings(
                (
                    f"{item.label or item.room_uid}({item.room_uid}, score={item.score:.2f}, {item.reason})"
                    for item in prior_result.room_rankings[: self.max_rankings]
                )
            ))
        if prior_result.object_rankings:
            lines.append("Top objects: " + _join_rankings(
                (
                    f"{item.label or item.object_uid}({item.object_uid}, score={item.score:.2f}, {item.reason})"
                    for item in prior_result.object_rankings[: self.max_rankings]
                )
            ))
        if prior_result.support_regions:
            lines.append("Support regions: " + _join_rankings(
                (
                    f"{item.label or item.uid}({item.uid}, room={item.room_uid or 'unknown'}, score={item.score:.2f})"
                    for item in prior_result.support_regions[: self.max_rankings]
                )
            ))
        if prior_result.frontier_biases:
            lines.append("Frontier biases: " + _join_rankings(
                (
                    f"{item.frontier_uid}(delta={item.score_delta:.2f}, room={item.prior_room_uid or 'unknown'})"
                    for item in prior_result.frontier_biases[: self.max_rankings]
                )
            ))
        text, _ = _limit_text("\n".join(lines), self.max_chars)
        return text

    def build_bundle(
        self,
        map_data: PriorMapData,
        prior_result: Optional[SearchPriorResult] = None,
        *,
        multimodal_context: Any = None,
    ) -> PromptContextBundle:
        """Build a complete bounded prompt context bundle.

        Args:
            map_data: Prior map to render.
            prior_result: Optional query result to summarize.
            multimodal_context: Optional prior-map multimodal context object.

        Returns:
            Prompt context bundle with natural language, XML, and search
            summaries.
        """

        natural, natural_truncated = _limit_text(self.summarize_map(map_data), self.max_chars)
        compact_xml, xml_truncated = _limit_text(self.to_compact_xml(map_data), self.max_chars)
        search_summary, search_truncated = _limit_text(self.summarize_search_prior(prior_result), self.max_chars)
        return PromptContextBundle(
            natural_language=natural,
            compact_xml=compact_xml,
            search_prior_summary=search_summary,
            truncated=natural_truncated or xml_truncated or search_truncated,
            multimodal_context=(
                multimodal_context.to_dict()
                if hasattr(multimodal_context, "to_dict")
                else dict(multimodal_context or {})
            ),
            metadata={
                "max_chars": self.max_chars,
                "room_count": len(map_data.rooms),
                "object_count": len(map_data.objects),
                "topology_edge_count": len(map_data.topology_edges),
                "has_search_prior_result": prior_result is not None,
                "has_multimodal_context": multimodal_context is not None,
            },
        )


def summarize_prior_map(map_data: PriorMapData, *, max_chars: int = 4000) -> str:
    """Render a natural-language prior-map summary.

    Args:
        map_data: Prior map to summarize.
        max_chars: Maximum output length.

    Returns:
        Bounded summary string.
    """

    return PriorMapPromptContextBuilder(max_chars=max_chars).summarize_map(map_data)


def to_compact_xml(map_data: PriorMapData, *, max_chars: int = 4000) -> str:
    """Render an OSM-like compact XML prior-map summary.

    Args:
        map_data: Prior map to render.
        max_chars: Maximum output length.

    Returns:
        Bounded XML-like string.
    """

    return PriorMapPromptContextBuilder(max_chars=max_chars).to_compact_xml(map_data)


def summarize_search_prior(prior_result: SearchPriorResult, *, max_chars: int = 4000) -> str:
    """Render a prompt-friendly search-prior summary.

    Args:
        prior_result: Search prior result to summarize.
        max_chars: Maximum output length.

    Returns:
        Bounded summary string.
    """

    return PriorMapPromptContextBuilder(max_chars=max_chars).summarize_search_prior(prior_result)


def _objects_by_room(objects: Sequence[PriorObject]) -> dict[str, tuple[PriorObject, ...]]:
    by_room: dict[str, list[PriorObject]] = {}
    for obj in sorted(objects, key=lambda item: item.uid):
        if obj.parent_room_uid:
            by_room.setdefault(obj.parent_room_uid, []).append(obj)
    return {room_uid: tuple(room_objects) for room_uid, room_objects in by_room.items()}


def _object_label(obj: PriorObject) -> str:
    suffix = " exact" if obj.exact else " likely"
    aliases = f" aliases={','.join(obj.aliases)}" if obj.aliases else ""
    return f"{obj.label}{suffix}{aliases}"


def _join_or_none(values: Iterable[str]) -> str:
    text = ", ".join(str(value) for value in values if str(value).strip())
    return text or "none"


def _join_rankings(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    return "; ".join(items) if items else "none"


def _format_xy(value: Optional[tuple[float, float]]) -> str:
    if value is None:
        return "unknown"
    return f"{float(value[0]):.2f},{float(value[1]):.2f}"


def _format_xyz(value: Optional[tuple[float, float, float]]) -> str:
    if value is None:
        return "unknown"
    return f"{float(value[0]):.2f},{float(value[1]):.2f},{float(value[2]):.2f}"


def _empty_tag(name: str, attrs: dict[str, Any]) -> str:
    attr_text = " ".join(f'{key}="{_xml(value)}"' for key, value in attrs.items() if str(value) != "")
    return f"<{name} {attr_text}/>" if attr_text else f"<{name}/>"


def _xml(value: Any) -> str:
    return escape(str(value), quote=True)


def _limit_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    suffix = " ... [truncated]"
    if max_chars <= len(suffix):
        return text[:max_chars], True
    return text[: max_chars - len(suffix)].rstrip() + suffix, True


__all__ = [
    "PriorMapPromptContextBuilder",
    "PromptContextBundle",
    "summarize_prior_map",
    "summarize_search_prior",
    "to_compact_xml",
]
