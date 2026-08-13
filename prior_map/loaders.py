"""File loaders for STRIVE prior-map contracts.

This module converts supported static map file formats into ``PriorMapData``.
It performs only schema-level normalization, including the explicit reflection
marker used by STRIVE-generated FloorPlan layouts. Coordinate alignment,
runtime observation fusion, ranking, and navigation decisions belong to later
prior-map modules.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    BoundaryXY,
    PriorMapData,
    PriorObject,
    PriorObservationRecord,
    PriorRoom,
    PriorTopologyEdge,
    Vector2,
    Vector3,
)


class PriorMapLoaderError(ValueError):
    """Raised when a prior-map file cannot be parsed into contracts."""


class PriorMapLoader:
    """Load supported prior-map file formats into ``PriorMapData``.

    Args:
        default_frame_id: Frame id assigned when a source file does not declare
            one.
    """

    SUPPORTED_FORMATS = {
        "auto",
        "json",
        "canonical_json",
        "floorplan_json",
        "floorplan_vln_json",
        "hm3d_json",
        "hm3d_topdown_json",
        "generated_prior_map_json",
        "vlm_reconstruction_json",
        "vlm_reconstruction",
        "osm_xml",
        "osm",
        "xml",
    }

    def __init__(self, default_frame_id: str = "prior_map") -> None:
        """Create a prior-map loader.

        Args:
            default_frame_id: Frame id used for formats that do not declare a
                coordinate frame.
        """

        self.default_frame_id = default_frame_id

    def load(self, path: str | Path, source_format: str = "auto") -> PriorMapData:
        """Load a prior-map file.

        Args:
            path: JSON, OSM, or XML map file path.
            source_format: Explicit source format, or ``"auto"`` to infer from
                file extension and top-level fields.

        Returns:
            Parsed ``PriorMapData``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            PriorMapLoaderError: If the format is unsupported or invalid.
        """

        map_path = Path(path)
        if not map_path.exists():
            raise FileNotFoundError(f"Prior map file not found: {map_path}")

        normalized = _normalize_format(source_format)
        if normalized not in self.SUPPORTED_FORMATS:
            raise PriorMapLoaderError(f"Unsupported prior map source_format: {source_format}")

        if normalized == "auto":
            normalized = self._infer_format(map_path)

        if normalized in {"osm", "xml", "osm_xml"}:
            return self._load_osm_xml(map_path)

        if normalized in {
            "json",
            "canonical_json",
            "floorplan_json",
            "floorplan_vln_json",
            "hm3d_json",
            "hm3d_topdown_json",
            "generated_prior_map_json",
            "vlm_reconstruction_json",
            "vlm_reconstruction",
        }:
            return self._load_json(map_path, normalized)

        raise PriorMapLoaderError(f"Unsupported prior map source_format: {source_format}")

    def _infer_format(self, path: Path) -> str:
        """Infer a source format from extension and top-level JSON fields.

        Args:
            path: Map file path.

        Returns:
            Normalized source format.

        Raises:
            PriorMapLoaderError: If the extension cannot be inferred.
        """

        suffix = path.suffix.lower()
        if suffix in {".osm", ".xml"}:
            return "osm_xml"
        if suffix != ".json":
            raise PriorMapLoaderError(f"Cannot infer prior map format from extension: {path.suffix}")

        data = _read_json(path)
        if _looks_like_canonical_prior_map(data):
            return "canonical_json"
        if _nested_payload(data) is not data:
            return "vlm_reconstruction_json"
        if "levels" in data:
            if str(data.get("source_format", "")).lower() in {"hm3d_json", "hm3d_topdown_json"}:
                return "hm3d_json"
            return "floorplan_json"
        if "rooms" in data or "objects" in data:
            return "generated_prior_map_json"
        raise PriorMapLoaderError(f"Cannot infer prior map JSON schema: {path}")

    def _load_json(self, path: Path, source_format: str) -> PriorMapData:
        """Load a JSON prior-map file.

        Args:
            path: JSON file path.
            source_format: Normalized source format.

        Returns:
            Parsed ``PriorMapData``.
        """

        data = _read_json(path)
        payload = _nested_payload(data)

        if source_format in {"canonical_json", "json"} and _looks_like_canonical_prior_map(payload):
            return PriorMapData.from_dict(payload)

        if _looks_like_canonical_prior_map(payload):
            map_data = PriorMapData.from_dict(payload)
            return _replace_source_format(map_data, _format_label(source_format, payload))

        if source_format in {"floorplan_json", "floorplan_vln_json", "hm3d_json", "hm3d_topdown_json"}:
            return self._load_levels_json(path, payload, source_format)

        if source_format in {"generated_prior_map_json", "vlm_reconstruction_json", "vlm_reconstruction", "json"}:
            return self._load_generic_json(path, payload, source_format)

        return self._load_generic_json(path, payload, source_format)

    def _load_levels_json(self, path: Path, data: Dict[str, Any], source_format: str) -> PriorMapData:
        """Load floorplan, FloorPlan-VLN, or HM3D generated levels JSON.

        Args:
            path: JSON file path.
            data: JSON-decoded payload.
            source_format: Normalized source format.

        Returns:
            Parsed ``PriorMapData``.
        """

        scene_id = str(data.get("scene_id") or data.get("scan_id") or data.get("house_id") or path.stem)
        if source_format in {"hm3d_json", "hm3d_topdown_json"} and scene_id.endswith("_semantic"):
            scene_id = scene_id[: -len("_semantic")]

        rooms: List[PriorRoom] = []
        topology: List[PriorTopologyEdge] = []
        seen_edges: set[Tuple[str, str, str]] = set()
        levels = data.get("levels") or {}

        for level_key, level_data in _items(levels):
            level = _safe_int(level_key, default=0)
            regions = level_data.get("regions") if isinstance(level_data, dict) else {}
            for raw_room_uid, room_data in _items(regions or {}):
                room_uid = str(room_data.get("uid") or room_data.get("id") or raw_room_uid)
                neighbors = _string_tuple(
                    room_data.get("neighbors")
                    or room_data.get("connections")
                    or room_data.get("connectivity")
                    or ()
                )
                room = PriorRoom(
                    uid=room_uid,
                    label=_text(
                        room_data.get("label")
                        or room_data.get("type")
                        or room_data.get("room_type")
                        or room_data.get("name")
                        or "unknown"
                    ),
                    boundary_xy=_boundary_xy(
                        room_data.get("boundary_xy")
                        or room_data.get("boundaries")
                        or room_data.get("boundary")
                        or room_data.get("polygon")
                        or ()
                    ),
                    centroid_xy=_vector2(
                        room_data.get("centroid_xy") or room_data.get("centroid") or room_data.get("center")
                    ),
                    neighbors=neighbors,
                    level=_safe_int(room_data.get("level"), default=level),
                    confidence=_confidence(room_data.get("confidence"), default=0.5),
                    source=source_format,
                    description=_text(room_data.get("description") or room_data.get("area_description") or ""),
                    metadata=_metadata(room_data, exclude=_ROOM_KEYS),
                )
                rooms.append(room)
                _append_room_neighbor_edges(topology, seen_edges, room, source_format)

        objects, object_edges = self._parse_objects(data.get("objects") or (), source_format)
        topology.extend(object_edges)

        explicit_edges = _parse_topology_edges(data, source_format)
        topology.extend(_dedupe_edges(explicit_edges, existing=topology))

        metadata = _metadata(data, exclude=_MAP_KEYS)
        frame_id = _text(data.get("frame_id") or self.default_frame_id)
        rooms, frame_id, metadata = _normalize_floorplan_coordinates(
            data,
            rooms,
            frame_id,
            metadata,
        )

        return PriorMapData(
            scene_id=scene_id,
            rooms=tuple(rooms),
            objects=tuple(objects),
            topology_edges=tuple(topology),
            source_format=_format_label(source_format, data),
            frame_id=frame_id,
            world_min=_vector2(data.get("world_min")),
            world_max=_vector2(data.get("world_max")),
            metadata=metadata,
        )

    def _load_generic_json(self, path: Path, data: Dict[str, Any], source_format: str) -> PriorMapData:
        """Load generic generated or VLM reconstruction JSON.

        Args:
            path: JSON file path.
            data: JSON-decoded payload.
            source_format: Normalized source format.

        Returns:
            Parsed ``PriorMapData``.
        """

        scene_id = str(data.get("scene_id") or data.get("scan_id") or data.get("house_id") or path.stem)
        if scene_id.endswith("_semantic") and source_format in {"hm3d_json", "hm3d_topdown_json"}:
            scene_id = scene_id[: -len("_semantic")]

        rooms = tuple(self._parse_rooms(data.get("rooms") or data.get("regions") or (), source_format))
        objects, object_edges = self._parse_objects(data.get("objects") or data.get("object_priors") or (), source_format)

        topology: List[PriorTopologyEdge] = []
        seen_edges: set[Tuple[str, str, str]] = set()
        for room in rooms:
            _append_room_neighbor_edges(topology, seen_edges, room, source_format)
        topology.extend(object_edges)
        topology.extend(_dedupe_edges(_parse_topology_edges(data, source_format), existing=topology))

        observations = tuple(
            _parse_observation_record(item, source_format)
            for item in _as_sequence(data.get("observations") or data.get("observation_records") or ())
        )

        return PriorMapData(
            scene_id=scene_id,
            rooms=rooms,
            objects=tuple(objects),
            topology_edges=tuple(topology),
            source_format=_format_label(source_format, data),
            frame_id=_text(data.get("frame_id") or self.default_frame_id),
            world_min=_vector2(data.get("world_min")),
            world_max=_vector2(data.get("world_max")),
            observations=observations,
            metadata=_metadata(data, exclude=_MAP_KEYS),
        )

    def _load_osm_xml(self, path: Path) -> PriorMapData:
        """Load simplified OSM/XML prior-map data.

        Args:
            path: XML or OSM file path.

        Returns:
            Parsed ``PriorMapData``.
        """

        root = ET.parse(path).getroot()
        if root.tag.lower().endswith("osm"):
            return self._load_osm_root(path, root)
        return self._load_simplified_xml_root(path, root)

    def _load_osm_root(self, path: Path, root: ET.Element) -> PriorMapData:
        """Load an OSM-like XML root.

        Args:
            path: Source path.
            root: Parsed XML root.

        Returns:
            Parsed ``PriorMapData``.
        """

        nodes: Dict[str, Vector2] = {}
        for node in root.findall("node"):
            node_id = str(node.get("id") or "")
            if not node_id:
                continue
            nodes[node_id] = (_float(node.get("lon"), default=0.0), _float(node.get("lat"), default=0.0))

        rooms: List[PriorRoom] = []
        topology: List[PriorTopologyEdge] = []
        seen_edges: set[Tuple[str, str, str]] = set()

        for way in root.findall("way"):
            tags = _xml_tags(way)
            if _text(tags.get("osmAG:type")).lower() == "passage":
                continue
            label = (
                tags.get("semantic_osmAG:room_type")
                or tags.get("room_type")
                or tags.get("type")
                or tags.get("name")
                or ""
            )
            if not str(label).strip():
                continue
            room_uid = str(way.get("id") or f"room_{len(rooms)}")
            boundary = tuple(nodes[nd.get("ref")] for nd in way.findall("nd") if nd.get("ref") in nodes)
            neighbors = _split_text_list(tags.get("semantic_osmAG:connections") or tags.get("connections") or "")
            room = PriorRoom(
                uid=room_uid,
                label=_text(label),
                boundary_xy=boundary,
                centroid_xy=_centroid(boundary),
                neighbors=neighbors,
                level=_safe_int(tags.get("level"), default=0),
                confidence=_confidence(tags.get("confidence"), default=0.5),
                source="osm_xml",
                description=_text(tags.get("semantic_osmAG:area_description") or tags.get("description") or ""),
                metadata={"osm_tags": tags},
            )
            rooms.append(room)
            _append_room_neighbor_edges(topology, seen_edges, room, "osm_xml")

        objects, object_edges = self._parse_osm_objects(root)
        topology.extend(object_edges)

        return PriorMapData(
            scene_id=path.stem,
            rooms=tuple(rooms),
            objects=tuple(objects),
            topology_edges=tuple(topology),
            source_format="osm_xml",
            frame_id=self.default_frame_id,
            metadata={"path": str(path)},
        )

    def _load_simplified_xml_root(self, path: Path, root: ET.Element) -> PriorMapData:
        """Load a simple XML root containing ``room`` and ``object`` elements.

        Args:
            path: Source path.
            root: Parsed XML root.

        Returns:
            Parsed ``PriorMapData``.
        """

        rooms: List[PriorRoom] = []
        topology: List[PriorTopologyEdge] = []
        seen_edges: set[Tuple[str, str, str]] = set()

        for room_elem in root.findall(".//room"):
            room_uid = str(room_elem.get("uid") or room_elem.get("id") or f"room_{len(rooms)}")
            boundary = tuple(
                (_float(point.get("x"), 0.0), _float(point.get("y"), 0.0))
                for point in room_elem.findall(".//point")
            )
            neighbors = _split_text_list(room_elem.get("neighbors") or room_elem.get("connections") or "")
            room = PriorRoom(
                uid=room_uid,
                label=_text(room_elem.get("label") or room_elem.get("type") or room_elem.get("name") or "unknown"),
                boundary_xy=boundary,
                centroid_xy=_centroid(boundary),
                neighbors=neighbors,
                level=_safe_int(room_elem.get("level"), default=0),
                confidence=_confidence(room_elem.get("confidence"), default=0.5),
                source="xml",
                description=_text(room_elem.get("description") or ""),
            )
            rooms.append(room)
            _append_room_neighbor_edges(topology, seen_edges, room, "xml")

        objects: List[PriorObject] = []
        for obj_elem in root.findall(".//object"):
            position = _vector3(
                (
                    obj_elem.get("x"),
                    obj_elem.get("y"),
                    obj_elem.get("z", 0.0),
                )
            )
            parent_room_uid = obj_elem.get("parent_room_uid") or obj_elem.get("parent_room") or obj_elem.get("room")
            obj = PriorObject(
                uid=str(obj_elem.get("uid") or obj_elem.get("id") or f"object_{len(objects)}"),
                label=_text(obj_elem.get("label") or obj_elem.get("type") or obj_elem.get("name") or "unknown"),
                position_xyz=position,
                parent_room_uid=str(parent_room_uid) if parent_room_uid else None,
                exact=_bool(obj_elem.get("exact"), default=position is not None),
                confidence=_confidence(obj_elem.get("confidence"), default=0.5),
                source="xml",
                aliases=_split_text_list(obj_elem.get("aliases") or ""),
            )
            objects.append(obj)
            if obj.parent_room_uid:
                topology.append(_room_object_edge(obj.parent_room_uid, obj.uid, "xml"))

        return PriorMapData(
            scene_id=_text(root.get("scene_id") or root.get("id") or path.stem),
            rooms=tuple(rooms),
            objects=tuple(objects),
            topology_edges=tuple(topology),
            source_format="xml",
            frame_id=_text(root.get("frame_id") or self.default_frame_id),
            metadata={"path": str(path)},
        )

    def _parse_rooms(self, raw_rooms: Any, source_format: str) -> List[PriorRoom]:
        """Parse generic room records.

        Args:
            raw_rooms: List or dict of room records.
            source_format: Normalized source format.

        Returns:
            Parsed room contracts.
        """

        rooms: List[PriorRoom] = []
        for raw_uid, room_data in _items(raw_rooms):
            if not isinstance(room_data, dict):
                continue
            room_uid = str(room_data.get("uid") or room_data.get("id") or raw_uid)
            rooms.append(
                PriorRoom(
                    uid=room_uid,
                    label=_text(
                        room_data.get("label")
                        or room_data.get("type")
                        or room_data.get("room_type")
                        or room_data.get("name")
                        or "unknown"
                    ),
                    boundary_xy=_boundary_xy(
                        room_data.get("boundary_xy")
                        or room_data.get("boundary")
                        or room_data.get("boundaries")
                        or room_data.get("polygon")
                        or ()
                    ),
                    centroid_xy=_vector2(
                        room_data.get("centroid_xy") or room_data.get("centroid") or room_data.get("center")
                    ),
                    neighbors=_string_tuple(
                        room_data.get("neighbors")
                        or room_data.get("connections")
                        or room_data.get("connectivity")
                        or ()
                    ),
                    level=_safe_int(room_data.get("level"), default=0),
                    confidence=_confidence(room_data.get("confidence"), default=0.5),
                    source=source_format,
                    description=_text(room_data.get("description") or ""),
                    metadata=_metadata(room_data, exclude=_ROOM_KEYS),
                )
            )
        return rooms

    def _parse_objects(self, raw_objects: Any, source_format: str) -> Tuple[List[PriorObject], List[PriorTopologyEdge]]:
        """Parse generic object records and room-object topology.

        Args:
            raw_objects: List or dict of object records.
            source_format: Normalized source format.

        Returns:
            Parsed objects and generated room-object edges.
        """

        objects: List[PriorObject] = []
        edges: List[PriorTopologyEdge] = []
        for raw_uid, object_data in _items(raw_objects):
            if not isinstance(object_data, dict):
                continue
            obj_uid = str(object_data.get("uid") or object_data.get("id") or object_data.get("object_id") or raw_uid)
            position = _vector3(
                object_data.get("position_xyz")
                or object_data.get("position")
                or object_data.get("center")
                or object_data.get("centroid")
            )
            parent_room_uid = (
                object_data.get("parent_room_uid")
                or object_data.get("parent_room")
                or object_data.get("room_uid")
                or object_data.get("room_id")
                or object_data.get("room")
            )
            obj = PriorObject(
                uid=obj_uid,
                label=_text(
                    object_data.get("label")
                    or object_data.get("type")
                    or object_data.get("category")
                    or object_data.get("class")
                    or object_data.get("name")
                    or "unknown"
                ),
                position_xyz=position,
                parent_room_uid=str(parent_room_uid) if parent_room_uid is not None else None,
                exact=_bool(
                    object_data.get("exact")
                    if "exact" in object_data
                    else object_data.get("is_exact_object")
                    if "is_exact_object" in object_data
                    else object_data.get("exactness"),
                    default=position is not None,
                ),
                confidence=_confidence(object_data.get("confidence"), default=0.5),
                source=source_format,
                aliases=_string_tuple(object_data.get("aliases") or object_data.get("synonyms") or ()),
                metadata=_metadata(object_data, exclude=_OBJECT_KEYS),
            )
            objects.append(obj)
            if obj.parent_room_uid:
                edges.append(_room_object_edge(obj.parent_room_uid, obj.uid, source_format))
        return objects, _dedupe_edges(edges)

    def _parse_osm_objects(self, root: ET.Element) -> Tuple[List[PriorObject], List[PriorTopologyEdge]]:
        """Parse OSM node objects.

        Args:
            root: OSM XML root.

        Returns:
            Parsed objects and generated room-object edges.
        """

        objects: List[PriorObject] = []
        edges: List[PriorTopologyEdge] = []
        for node in root.findall("node"):
            tags = _xml_tags(node)
            label = (
                tags.get("semantic_osmAG:object_name")
                or tags.get("semantic_osmAG:observed_objects")
                or tags.get("object")
                or tags.get("object_name")
            )
            if not label:
                continue
            position = (_float(node.get("lon"), 0.0), _float(node.get("lat"), 0.0), 0.0)
            parent_room_uid = tags.get("osmAG:parent") or tags.get("parent") or tags.get("room")
            obj = PriorObject(
                uid=str(node.get("id") or f"object_{len(objects)}"),
                label=_text(label),
                position_xyz=position,
                parent_room_uid=str(parent_room_uid) if parent_room_uid else None,
                exact="semantic_osmAG:object_name" in tags,
                confidence=_confidence(tags.get("confidence"), default=0.5),
                source="osm_xml",
                aliases=_split_text_list(tags.get("aliases") or ""),
                metadata={"osm_tags": tags},
            )
            objects.append(obj)
            if obj.parent_room_uid:
                edges.append(_room_object_edge(obj.parent_room_uid, obj.uid, "osm_xml"))
        return objects, _dedupe_edges(edges)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read and validate a JSON object.

    Args:
        path: JSON file path.

    Returns:
        JSON object payload.

    Raises:
        PriorMapLoaderError: If the root is not an object.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise PriorMapLoaderError(f"Prior map JSON root must be an object: {path}")
    return data


def _nested_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return nested prior-map payloads used by reconstruction wrappers.

    Args:
        data: JSON-decoded root payload.

    Returns:
        Nested map payload when present, otherwise ``data``.
    """

    for key in ("prior_map_data", "prior_map", "map_data"):
        nested = data.get(key)
        if isinstance(nested, dict):
            return nested
    return data


def _looks_like_canonical_prior_map(data: Dict[str, Any]) -> bool:
    """Return whether JSON already matches ``PriorMapData.to_dict``.

    Args:
        data: JSON-decoded payload.

    Returns:
        ``True`` when the payload has canonical prior-map fields.
    """

    return "scene_id" in data and "source_format" in data and (
        "topology_edges" in data or any(_has_key(item, "uid") for _, item in _items(data.get("rooms", ())))
    )


def _format_label(source_format: str, payload: Dict[str, Any]) -> str:
    """Return the source format stored in ``PriorMapData``.

    Args:
        source_format: Loader format.
        payload: JSON payload.

    Returns:
        Stable source format label.
    """

    declared = payload.get("source_format")
    if declared and source_format in {"json", "canonical_json"}:
        return str(declared)
    if source_format == "canonical_json":
        return str(declared or "json")
    if source_format == "hm3d_topdown_json":
        return "hm3d_json"
    if source_format == "generated_prior_map_json":
        return str(declared or "generated_prior_map_json")
    if source_format == "vlm_reconstruction":
        return "vlm_reconstruction_json"
    return str(declared or source_format)


def _replace_source_format(map_data: PriorMapData, source_format: str) -> PriorMapData:
    """Return a copy with a normalized source format.

    Args:
        map_data: Parsed map.
        source_format: Desired source format label.

    Returns:
        New ``PriorMapData`` instance.
    """

    return PriorMapData(
        scene_id=map_data.scene_id,
        rooms=map_data.rooms,
        objects=map_data.objects,
        topology_edges=map_data.topology_edges,
        source_format=source_format,
        frame_id=map_data.frame_id,
        world_min=map_data.world_min,
        world_max=map_data.world_max,
        observations=map_data.observations,
        metadata=map_data.metadata,
    )


def _items(value: Any) -> Iterable[Tuple[str, Any]]:
    """Iterate list or dict records with stable synthetic keys.

    Args:
        value: List, tuple, or dict.

    Returns:
        Iterable of ``(key, item)`` pairs.
    """

    if isinstance(value, dict):
        return value.items()
    if isinstance(value, (list, tuple)):
        return ((str(index), item) for index, item in enumerate(value))
    return ()


def _as_sequence(value: Any) -> Sequence[Any]:
    """Return a sequence view for list-like JSON fields.

    Args:
        value: Arbitrary JSON value.

    Returns:
        Sequence, or an empty tuple when not list-like.
    """

    if isinstance(value, (list, tuple)):
        return value
    return ()


def _parse_observation_record(item: Dict[str, Any], source_format: str) -> PriorObservationRecord:
    """Parse a generic observation record.

    Args:
        item: Observation payload.
        source_format: Source format label.

    Returns:
        Parsed observation record.
    """

    pose = _vector3(item.get("pose_xyz") or item.get("pose") or item.get("position"))
    detected = item.get("detected_objects") or item.get("observed_objects") or ()
    labels = []
    uids = []
    for index, detected_item in enumerate(_as_sequence(detected)):
        if isinstance(detected_item, dict):
            labels.append(_text(detected_item.get("label") or detected_item.get("type") or detected_item.get("name") or ""))
            uids.append(_text(detected_item.get("uid") or detected_item.get("id") or f"observed_{index}"))
        else:
            labels.append(_text(detected_item))
            uids.append(f"observed_{index}")
    return PriorObservationRecord(
        timestamp=_float(item.get("timestamp") or item.get("step"), default=0.0),
        pose_xyz=pose,
        frame_id=_text(item.get("frame_id") or "map"),
        observed_object_uids=tuple(uid for uid in uids if uid),
        observed_object_labels=tuple(label for label in labels if label),
        room_hypothesis_uid=item.get("room_hypothesis_uid") or item.get("room_hypothesis"),
        source=_text(item.get("source") or source_format),
        metadata=_metadata(item, exclude=_OBSERVATION_KEYS),
    )


def _parse_topology_edges(data: Dict[str, Any], source_format: str) -> List[PriorTopologyEdge]:
    """Parse explicit topology edge arrays.

    Args:
        data: JSON payload.
        source_format: Source format label.

    Returns:
        Parsed topology edges.
    """

    raw_edges = data.get("topology_edges") or data.get("topology") or data.get("edges") or ()
    edges: List[PriorTopologyEdge] = []
    for raw_uid, edge_data in _items(raw_edges):
        if not isinstance(edge_data, dict):
            continue
        source_uid = edge_data.get("source_uid") or edge_data.get("source") or edge_data.get("from")
        target_uid = edge_data.get("target_uid") or edge_data.get("target") or edge_data.get("to")
        if not source_uid or not target_uid:
            continue
        edges.append(
            PriorTopologyEdge(
                uid=str(edge_data.get("uid") or edge_data.get("id") or raw_uid),
                source_uid=str(source_uid),
                target_uid=str(target_uid),
                edge_type=_text(edge_data.get("edge_type") or edge_data.get("type") or "room-room"),
                relation=_text(edge_data.get("relation") or "connected"),
                bidirectional=_bool(edge_data.get("bidirectional"), default=True),
                confidence=_confidence(edge_data.get("confidence"), default=0.5),
                weight=_optional_float(edge_data.get("weight")),
                source=_text(edge_data.get("source_format") or edge_data.get("source_name") or source_format),
                metadata=_metadata(edge_data, exclude=_EDGE_KEYS),
            )
        )
    return edges


def _append_room_neighbor_edges(
    edges: List[PriorTopologyEdge],
    seen_edges: set[Tuple[str, str, str]],
    room: PriorRoom,
    source_format: str,
) -> None:
    """Append deduplicated room-room neighbor edges.

    Args:
        edges: Mutable edge list.
        seen_edges: Deduplication keys.
        room: Room with neighbor ids.
        source_format: Source format label.
    """

    for neighbor in room.neighbors:
        key = tuple(sorted((room.uid, neighbor))) + ("adjacent",)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_uid = f"edge:{key[0]}:{key[1]}:adjacent"
        edges.append(
            PriorTopologyEdge(
                uid=edge_uid,
                source_uid=room.uid,
                target_uid=neighbor,
                edge_type="room-room",
                relation="adjacent",
                bidirectional=True,
                confidence=min(room.confidence, 0.5),
                source=source_format,
            )
        )


def _room_object_edge(room_uid: str, object_uid: str, source_format: str) -> PriorTopologyEdge:
    """Create a room-object contains edge.

    Args:
        room_uid: Parent room uid.
        object_uid: Object uid.
        source_format: Source format label.

    Returns:
        Topology edge.
    """

    return PriorTopologyEdge(
        uid=f"edge:{room_uid}:{object_uid}:contains",
        source_uid=room_uid,
        target_uid=object_uid,
        edge_type="room-object",
        relation="contains",
        bidirectional=False,
        confidence=0.5,
        source=source_format,
    )


def _dedupe_edges(
    edges: Iterable[PriorTopologyEdge],
    existing: Iterable[PriorTopologyEdge] = (),
) -> List[PriorTopologyEdge]:
    """Deduplicate edges by uid and endpoint relation.

    Args:
        edges: Candidate edges.
        existing: Already accepted edges.

    Returns:
        Deduplicated edge list.
    """

    seen = {(edge.uid, edge.source_uid, edge.target_uid, edge.relation) for edge in existing}
    output: List[PriorTopologyEdge] = []
    for edge in edges:
        key = (edge.uid, edge.source_uid, edge.target_uid, edge.relation)
        if key in seen:
            continue
        seen.add(key)
        output.append(edge)
    return output


def _xml_tags(element: ET.Element) -> Dict[str, str]:
    """Return OSM-style tags from an XML element.

    Args:
        element: XML element.

    Returns:
        Mapping from tag key to tag value.
    """

    return {str(tag.get("k")): str(tag.get("v")) for tag in element.findall("tag") if tag.get("k") is not None}


def _normalize_format(source_format: str) -> str:
    """Normalize source format aliases.

    Args:
        source_format: User-provided source format.

    Returns:
        Normalized source format.
    """

    text = str(source_format or "auto").strip().lower().replace("-", "_")
    aliases = {
        "canonical": "canonical_json",
        "floorplan": "floorplan_json",
        "floorplan_vln": "floorplan_vln_json",
        "hm3d": "hm3d_json",
        "hm3d_topdown": "hm3d_topdown_json",
        "generated": "generated_prior_map_json",
        "vlm": "vlm_reconstruction_json",
    }
    return aliases.get(text, text)


def _boundary_xy(value: Any) -> BoundaryXY:
    """Convert common boundary encodings to ``BoundaryXY``.

    Args:
        value: JSON boundary field.

    Returns:
        Tuple of 2-D points.
    """

    points = []
    for point in _as_sequence(value):
        vec = _vector2(point)
        if vec is not None:
            points.append(vec)
    return tuple(points)


def _normalize_floorplan_coordinates(
    data: Dict[str, Any],
    rooms: Sequence[PriorRoom],
    frame_id: str,
    metadata: Dict[str, Any],
) -> tuple[Sequence[PriorRoom], str, Dict[str, Any]]:
    """Normalize STRIVE-generated ``(x,-z)`` room geometry for runtime use.

    FloorPlan-VLN stores its 2-D layout in an ``(x, -z)`` convention. STRIVE's
    canonical prior-map query layer uses Habitat ``(x,z)`` projection, so the
    generated interchange format carries an explicit marker and is reflected
    exactly once at load time:

    .. math::

       (x, y_{floorplan}) \mapsto (x, -y_{floorplan})

    Untagged third-party FloorPlan files are left untouched for backward
    compatibility; their coordinate frame must be supplied by the caller's
    alignment configuration.

    Args:
        data: Decoded top-level floorplan payload.
        rooms: Parsed room contracts.
        frame_id: Declared source frame id.
        metadata: Parsed top-level metadata.

    Returns:
        Normalized rooms, runtime frame id, and augmented metadata.
    """

    convention = data.get("coordinate_convention")
    axes = convention.get("floorplan_axes") if isinstance(convention, dict) else None
    if frame_id != "floorplan_metric" or axes != ["x", "-z"]:
        return rooms, frame_id, metadata

    normalized = tuple(
        replace(
            room,
            boundary_xy=tuple((float(x), -float(y)) for x, y in room.boundary_xy),
            centroid_xy=(float(room.centroid_xy[0]), -float(room.centroid_xy[1]))
            if room.centroid_xy is not None
            else None,
        )
        for room in rooms
    )
    normalized_metadata = dict(metadata)
    normalized_metadata["coordinate_normalization"] = {
        "source_frame_id": "floorplan_metric",
        "runtime_frame_id": "habitat_world",
        "operation": "reflect_second_plane_axis",
    }
    return normalized, "habitat_world", normalized_metadata


def _vector2(value: Any) -> Optional[Vector2]:
    """Convert a JSON value to a 2-D vector.

    Args:
        value: JSON vector value.

    Returns:
        2-D vector or ``None``.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        if x is None or y is None:
            return None
        return (_float(x, 0.0), _float(y, 0.0))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_float(value[0], 0.0), _float(value[1], 0.0))
    return None


def _vector3(value: Any) -> Optional[Vector3]:
    """Convert a JSON value to a 3-D vector.

    Args:
        value: JSON vector value.

    Returns:
        3-D vector or ``None``.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        z = value.get("z", 0.0)
        if x is None or y is None:
            return None
        return (_float(x, 0.0), _float(y, 0.0), _float(z, 0.0))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        z = value[2] if len(value) >= 3 else 0.0
        return (_float(value[0], 0.0), _float(value[1], 0.0), _float(z, 0.0))
    return None


def _centroid(boundary: BoundaryXY) -> Optional[Vector2]:
    """Compute a simple arithmetic centroid from boundary points.

    Args:
        boundary: Boundary points.

    Returns:
        Centroid or ``None`` when no points are available.
    """

    if not boundary:
        return None
    return (
        sum(point[0] for point in boundary) / len(boundary),
        sum(point[1] for point in boundary) / len(boundary),
    )


def _string_tuple(value: Any) -> Tuple[str, ...]:
    """Convert list-like or delimited values to a string tuple.

    Args:
        value: JSON value.

    Returns:
        Tuple of non-empty strings.
    """

    if isinstance(value, str):
        return _split_text_list(value)
    return tuple(str(item) for item in _as_sequence(value) if str(item).strip())


def _split_text_list(value: str) -> Tuple[str, ...]:
    """Split comma/semicolon-separated text.

    Args:
        value: Delimited text.

    Returns:
        Tuple of stripped non-empty strings.
    """

    if not value:
        return ()
    text = str(value).replace(";", ",")
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _metadata(payload: Dict[str, Any], exclude: set[str]) -> Dict[str, Any]:
    """Return extension metadata from unclaimed fields.

    Args:
        payload: Source payload.
        exclude: Claimed field names.

    Returns:
        Metadata dictionary.
    """

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        result = dict(metadata)
    else:
        result = {}
    for key, value in payload.items():
        if key not in exclude and key != "metadata":
            result[key] = value
    return result


def _has_key(value: Any, key: str) -> bool:
    """Return whether ``value`` is a dict containing ``key``.

    Args:
        value: Candidate value.
        key: Key to check.

    Returns:
        ``True`` when present.
    """

    return isinstance(value, dict) and key in value


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int with fallback.

    Args:
        value: Source value.
        default: Fallback value.

    Returns:
        Integer value.
    """

    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with fallback.

    Args:
        value: Source value.
        default: Fallback value.

    Returns:
        Float value.
    """

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    """Convert an optional value to float.

    Args:
        value: Source value.

    Returns:
        Float or ``None``.
    """

    if value is None:
        return None
    return _float(value, default=0.0)


def _confidence(value: Any, default: float = 0.5) -> float:
    """Convert confidence to a bounded score.

    Args:
        value: Source value.
        default: Fallback confidence.

    Returns:
        Confidence clipped to ``[0, 1]``.
    """

    score = _float(value, default=default)
    return max(0.0, min(1.0, score))


def _bool(value: Any, default: bool = False) -> bool:
    """Convert common bool encodings.

    Args:
        value: Source value.
        default: Fallback bool.

    Returns:
        Boolean value.
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "exact"}:
        return True
    if text in {"false", "0", "no", "n", "hypothesis"}:
        return False
    return default


def _text(value: Any) -> str:
    """Convert a value to stripped text.

    Args:
        value: Source value.

    Returns:
        String value.
    """

    return str(value or "").strip()


_MAP_KEYS = {
    "scene_id",
    "scan_id",
    "house_id",
    "rooms",
    "regions",
    "objects",
    "object_priors",
    "topology",
    "topology_edges",
    "edges",
    "source_format",
    "frame_id",
    "world_min",
    "world_max",
    "observations",
    "observation_records",
    "levels",
}
_ROOM_KEYS = {
    "uid",
    "id",
    "label",
    "type",
    "room_type",
    "name",
    "boundary_xy",
    "boundaries",
    "boundary",
    "polygon",
    "centroid_xy",
    "centroid",
    "center",
    "neighbors",
    "connections",
    "connectivity",
    "level",
    "confidence",
    "source",
    "description",
    "area_description",
}
_OBJECT_KEYS = {
    "uid",
    "id",
    "object_id",
    "label",
    "type",
    "category",
    "class",
    "name",
    "position_xyz",
    "position",
    "center",
    "centroid",
    "parent_room_uid",
    "parent_room",
    "room_uid",
    "room_id",
    "room",
    "exact",
    "is_exact_object",
    "exactness",
    "confidence",
    "source",
    "aliases",
    "synonyms",
}
_EDGE_KEYS = {
    "uid",
    "id",
    "source_uid",
    "source",
    "from",
    "target_uid",
    "target",
    "to",
    "edge_type",
    "type",
    "relation",
    "bidirectional",
    "confidence",
    "weight",
    "source_format",
    "source_name",
}
_OBSERVATION_KEYS = {
    "timestamp",
    "step",
    "pose_xyz",
    "pose",
    "position",
    "frame_id",
    "detected_objects",
    "observed_objects",
    "room_hypothesis_uid",
    "room_hypothesis",
    "source",
}
