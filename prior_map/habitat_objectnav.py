"""Build prior maps from Habitat ObjectNav-style episode datasets.

The builder reads dataset JSON/JSON.GZ files directly and emits canonical
``PriorMapData``. It intentionally does not import Habitat, simulator APIs, or
detectors, so prior-map fixtures can be generated on the host or inside Docker.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .alignment import PriorMapAlignment
from .contracts import PriorMapData, PriorObject, PriorRoom, PriorTopologyEdge


@dataclass(frozen=True)
class HabitatPriorMapBuildResult:
    """Result of building a prior map from a Habitat dataset.

    Args:
        prior_map: Canonical prior map.
        selected_episode: Dataset episode used as the selection anchor.
        goal_count: Number of object goals converted into prior objects.
        metadata: JSON-friendly builder diagnostics.
    """

    prior_map: PriorMapData
    selected_episode: dict[str, Any]
    goal_count: int
    metadata: dict[str, Any]


class HabitatObjectNavPriorMapBuilder:
    """Convert Habitat ObjectNav datasets into STRIVE prior maps.

    Args:
        default_room_label: Label used for the coarse scene-level room when the
            dataset does not provide room segmentation.
        boundary_padding_m: Padding used around goal positions to create a
            coarse room boundary.
    """

    def __init__(self, default_room_label: str = "scene", boundary_padding_m: float = 1.0) -> None:
        """Create a builder.

        Args:
            default_room_label: Label for the generated scene-level room.
            boundary_padding_m: Padding around goal positions.
        """

        self.default_room_label = str(default_room_label or "scene")
        self.boundary_padding_m = float(boundary_padding_m)

    def build(
        self,
        dataset_path: str | Path,
        *,
        scene_id: str = "",
        object_category: str = "",
        episode_rank: int = 0,
        include_scene_categories: bool = False,
    ) -> HabitatPriorMapBuildResult:
        """Build a prior map from one Habitat ObjectNav dataset.

        Args:
            dataset_path: Habitat dataset JSON or JSON.GZ path.
            scene_id: Optional scene id filter.
            object_category: Optional object category filter.
            episode_rank: Rank among matching episodes.
            include_scene_categories: If true, include all goal categories from
                the selected scene, not only the selected episode target.

        Returns:
            Build result containing canonical ``PriorMapData``.

        Raises:
            ValueError: If no matching episode can be found.
        """

        path = Path(dataset_path)
        dataset = _read_dataset(path)
        if not dataset.get("episodes") and (path.parent / "content").is_dir():
            dataset = _read_split_content_dataset(
                path,
                root_dataset=dataset,
                scene_id=scene_id,
                object_category=object_category,
                episode_rank=episode_rank,
            )
        episodes = list(dataset.get("episodes", []) or [])
        selected = _select_episode(
            episodes,
            scene_id=scene_id,
            object_category=object_category,
            episode_rank=episode_rank,
        )
        selected_scene = _episode_scene_id(selected)
        selected_category = _episode_category(selected)
        goals = _extract_goals(
            dataset,
            selected_episode=selected,
            scene_id=selected_scene,
            object_category=selected_category,
            include_scene_categories=include_scene_categories,
        )
        room_uid = f"prior_room:{_safe_uid(_scene_short_id(selected_scene))}:scene"
        objects = tuple(
            _goal_to_prior_object(goal, index, room_uid=room_uid, fallback_label=selected_category)
            for index, goal in enumerate(goals)
        )
        room = _make_scene_room(
            uid=room_uid,
            label=self.default_room_label,
            objects=objects,
            padding_m=self.boundary_padding_m,
        )
        edges = tuple(
            PriorTopologyEdge(
                uid=f"edge:{room.uid}:{obj.uid}",
                source_uid=room.uid,
                target_uid=obj.uid,
                edge_type="room-object",
                relation="contains",
                confidence=obj.confidence,
                source="habitat_objectnav_dataset",
            )
            for obj in objects
        )
        prior_map = PriorMapData(
            scene_id=_scene_short_id(selected_scene) or "habitat_scene",
            rooms=(room,),
            objects=objects,
            topology_edges=edges,
            source_format="habitat_objectnav_json",
            frame_id="habitat_world",
            world_min=room.boundary_xy[0] if room.boundary_xy else None,
            world_max=room.boundary_xy[2] if len(room.boundary_xy) >= 3 else None,
            metadata={
                "dataset_path": str(path),
                "source_scene_id": selected_scene,
                "selected_episode_id": str(selected.get("episode_id", "")),
                "selected_object_category": selected_category,
                "include_scene_categories": bool(include_scene_categories),
                "goal_count": len(objects),
            },
        )
        return HabitatPriorMapBuildResult(
            prior_map=prior_map,
            selected_episode=dict(selected),
            goal_count=len(objects),
            metadata=dict(prior_map.metadata),
        )


def build_habitat_objectnav_prior_map(
    dataset_path: str | Path,
    *,
    scene_id: str = "",
    object_category: str = "",
    episode_rank: int = 0,
    include_scene_categories: bool = False,
) -> HabitatPriorMapBuildResult:
    """Build a canonical prior map from a Habitat ObjectNav dataset.

    Args:
        dataset_path: Dataset JSON or JSON.GZ path.
        scene_id: Optional scene id filter.
        object_category: Optional object category filter.
        episode_rank: Rank among matching episodes.
        include_scene_categories: Include all scene goals when true.

    Returns:
        Build result.
    """

    return HabitatObjectNavPriorMapBuilder().build(
        dataset_path,
        scene_id=scene_id,
        object_category=object_category,
        episode_rank=episode_rank,
        include_scene_categories=include_scene_categories,
    )


def write_prior_map_with_alignment(
    result: HabitatPriorMapBuildResult,
    output_path: str | Path,
    *,
    alignment_output_path: str | Path = "",
    alignment_mode: str = "unavailable",
) -> dict[str, str]:
    """Write a generated prior map and optional alignment file.

    Args:
        result: Build result to write.
        output_path: Prior map JSON destination.
        alignment_output_path: Optional alignment JSON destination.
        alignment_mode: ``unavailable`` or ``identity``.

    Returns:
        Paths written.
    """

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


def _read_dataset(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Habitat dataset root must be an object: {path}")
    return data


def _read_split_content_dataset(
    split_path: Path,
    *,
    root_dataset: dict[str, Any],
    scene_id: str,
    object_category: str,
    episode_rank: int,
) -> dict[str, Any]:
    content_dir = split_path.parent / "content"
    candidates = _content_candidates(content_dir, scene_id=scene_id)
    category_norm = _norm(object_category)
    for candidate in candidates:
        data = _read_dataset(candidate)
        episodes = list(data.get("episodes", []) or [])
        if not episodes:
            continue
        if category_norm and not any(_category_matches(category_norm, _episode_category(episode)) for episode in episodes):
            goal_keys = tuple(str(key) for key in dict(data.get("goals_by_category", {}) or {}).keys())
            if not any(category_norm in _norm(key) for key in goal_keys):
                continue
        if episode_rank >= len(episodes):
            continue
        merged = dict(root_dataset)
        merged.update(data)
        merged.setdefault("metadata", {})
        if isinstance(merged["metadata"], dict):
            merged["metadata"] = dict(merged["metadata"])
            merged["metadata"]["split_root_path"] = str(split_path)
            merged["metadata"]["content_path"] = str(candidate)
        return merged
    raise ValueError(
        f"No Habitat content dataset matched scene_id={scene_id!r}, "
        f"object_category={object_category!r} under {content_dir}"
    )


def _content_candidates(content_dir: Path, *, scene_id: str) -> list[Path]:
    files = sorted(content_dir.glob("*.json.gz"))
    scene_norm = _norm(scene_id)
    scene_short_norm = _norm(_scene_short_id(scene_id))
    if not scene_norm:
        return files
    matched = [
        path
        for path in files
        if scene_norm in _norm(_content_scene_key(path)) or scene_short_norm in _norm(_content_scene_key(path))
    ]
    return matched or files


def _content_scene_key(path: Path) -> str:
    name = path.name
    for suffix in (".glb.json.gz", "_episodes.json.gz", ".json.gz", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _select_episode(
    episodes: Sequence[dict[str, Any]],
    *,
    scene_id: str,
    object_category: str,
    episode_rank: int,
) -> dict[str, Any]:
    scene_norm = _norm(scene_id)
    scene_short_norm = _norm(_scene_short_id(scene_id))
    category_norm = _norm(object_category)
    matched = []
    for episode in episodes:
        episode_scene_norm = _norm(_episode_scene_id(episode))
        if scene_norm and scene_norm not in episode_scene_norm and scene_short_norm not in episode_scene_norm:
            continue
        if category_norm and not _category_matches(category_norm, _episode_category(episode)):
            continue
        matched.append(episode)
    if not matched:
        raise ValueError(
            f"No Habitat episode matched scene_id={scene_id!r}, object_category={object_category!r}."
        )
    if episode_rank < 0 or episode_rank >= len(matched):
        raise IndexError(f"episode_rank={episode_rank} is out of range for {len(matched)} matched episodes")
    return dict(matched[episode_rank])


def _extract_goals(
    dataset: dict[str, Any],
    *,
    selected_episode: dict[str, Any],
    scene_id: str,
    object_category: str,
    include_scene_categories: bool,
) -> list[dict[str, Any]]:
    episode_goals = [dict(goal) for goal in selected_episode.get("goals", []) or [] if isinstance(goal, dict)]
    if episode_goals and not include_scene_categories:
        return episode_goals

    goals_by_category = dict(dataset.get("goals_by_category", {}) or {})
    scene_norm = _norm(scene_id)
    scene_short_norm = _norm(_scene_short_id(scene_id))
    category_norm = _norm(object_category)
    selected_goals: list[dict[str, Any]] = []
    for key, goals in goals_by_category.items():
        key_norm = _norm(key)
        if scene_norm and scene_norm not in key_norm and scene_short_norm not in key_norm:
            continue
        if not include_scene_categories and category_norm and category_norm not in key_norm:
            continue
        for goal in goals or []:
            if isinstance(goal, dict):
                selected_goals.append(dict(goal))
    if selected_goals:
        return selected_goals
    return [
        {
            "object_category": object_category,
            "object_name": object_category,
            "position": None,
            "source": "episode_category_fallback",
        }
    ]


def _goal_to_prior_object(
    goal: dict[str, Any],
    index: int,
    *,
    room_uid: str,
    fallback_label: str,
) -> PriorObject:
    label = str(
        goal.get("object_category")
        or goal.get("object_category_name")
        or goal.get("object_name")
        or goal.get("category")
        or fallback_label
        or "object"
    )
    object_id = str(goal.get("object_id") or goal.get("object_name") or goal.get("goal_id") or index)
    position = _extract_goal_position(goal)
    aliases = tuple(
        str(value)
        for value in (
            goal.get("aliases")
            or goal.get("synonyms")
            or ()
        )
        if str(value).strip()
    )
    return PriorObject(
        uid=f"prior_object:{_safe_uid(label)}:{_safe_uid(object_id)}",
        label=label,
        position_xyz=position,
        parent_room_uid=room_uid,
        exact=position is not None,
        confidence=0.75 if position is not None else 0.35,
        source="habitat_objectnav_dataset",
        aliases=aliases,
        metadata={
            "source_goal": goal,
            "habitat_object_id": object_id,
        },
    )


def _make_scene_room(
    *,
    uid: str,
    label: str,
    objects: Sequence[PriorObject],
    padding_m: float,
) -> PriorRoom:
    points = [(obj.position_xyz[0], obj.position_xyz[2]) for obj in objects if obj.position_xyz is not None]
    if not points:
        boundary = ((-padding_m, -padding_m), (padding_m, -padding_m), (padding_m, padding_m), (-padding_m, padding_m))
        centroid = (0.0, 0.0)
    else:
        min_x = min(point[0] for point in points) - padding_m
        max_x = max(point[0] for point in points) + padding_m
        min_y = min(point[1] for point in points) - padding_m
        max_y = max(point[1] for point in points) + padding_m
        boundary = ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y))
        centroid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    return PriorRoom(
        uid=uid,
        label=label,
        boundary_xy=boundary,
        centroid_xy=centroid,
        confidence=0.4,
        source="habitat_objectnav_dataset",
        metadata={"coarse_scene_room": True},
    )


def _alignment_for_mode(mode: str, prior_map: PriorMapData) -> PriorMapAlignment:
    normalized = str(mode or "unavailable").strip().lower()
    if normalized == "identity":
        return PriorMapAlignment.identity(prior_frame_id=prior_map.frame_id, runtime_frame_id="map")
    return PriorMapAlignment.unavailable(
        prior_frame_id=prior_map.frame_id,
        runtime_frame_id="map",
        reason=(
            "Habitat dataset goal positions are scene/world coordinates; "
            "STRIVE mapper runtime is local to episode start unless calibrated."
        ),
    )


def _position_xyz(value: Any) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    try:
        values = [float(item) for item in list(value)[:3]]
    except Exception:
        return None
    if len(values) != 3:
        return None
    return values[0], values[1], values[2]


def _extract_goal_position(goal: dict[str, Any]) -> Optional[tuple[float, float, float]]:
    for key in ("position", "position_xyz", "center", "center_xyz"):
        position = _position_xyz(goal.get(key))
        if position is not None:
            return position
    view_points = goal.get("view_points") or ()
    if not isinstance(view_points, Iterable) or isinstance(view_points, (str, bytes, dict)):
        return None
    for view_point in view_points:
        if not isinstance(view_point, dict):
            continue
        agent_state = view_point.get("agent_state")
        if not isinstance(agent_state, dict):
            continue
        position = _position_xyz(agent_state.get("position"))
        if position is not None:
            return position
    return None


def _episode_scene_id(episode: dict[str, Any]) -> str:
    return str(episode.get("scene_id") or episode.get("scene") or "")


def _episode_category(episode: dict[str, Any]) -> str:
    return str(
        episode.get("object_category")
        or episode.get("object_category_name")
        or episode.get("goal_object_category")
        or ""
    )


def _scene_short_id(scene_id: str) -> str:
    if not scene_id:
        return ""
    name = Path(scene_id).name
    for suffix in (".basis.glb", ".semantic.glb", ".glb", ".navmesh"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _safe_uid(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return safe or "unknown"


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _category_matches(requested_norm: str, candidate: Any) -> bool:
    candidate_norm = _norm(candidate)
    return (
        not requested_norm
        or requested_norm == candidate_norm
        or requested_norm in candidate_norm
        or candidate_norm in requested_norm
    )


__all__ = [
    "HabitatObjectNavPriorMapBuilder",
    "HabitatPriorMapBuildResult",
    "build_habitat_objectnav_prior_map",
    "write_prior_map_with_alignment",
]
