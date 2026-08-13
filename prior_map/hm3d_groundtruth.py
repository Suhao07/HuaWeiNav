"""Build geometric HM3D prior maps from Habitat semantic-scene ground truth.

This module converts a Habitat simulator's semantic scene and navigation mesh
into STRIVE's canonical ``PriorMapData`` schema. It does not read ObjectNav
episode goal positions and does not depend on runtime mapper outputs, so the
result is a static scene prior rather than leaked task evidence.
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Sequence

from .alignment import PriorMapAlignment
from .contracts import BoundaryXY, PriorMapData, PriorObject, PriorRoom, PriorTopologyEdge, Vector2, Vector3

_GridCell = tuple[int, int, int]
_TEXTURE_COLOR_TOLERANCE_SQUARED = 20 * 20 * 3


@dataclass(frozen=True)
class HM3DGroundTruthBuildConfig:
    r"""Configuration for HM3D ground-truth geometric prior construction.

    Args:
        topdown_resolution: Grid-cell size in Habitat world meters. The builder
            samples navigability over the \(x,z\) plane at this spacing.
        floor_height_tolerance: Maximum vertical distance in meters between a
            region center and a sampled navmesh point before the sample is
            treated as belonging to another floor.
        min_room_area_m2: Minimum connected room mask area to keep.
        mask_dilation_radius_m: Radius in meters used when detecting
            room-room contact between navigable masks.
        split_disconnected_components: Whether disconnected 2-D navigable masks
            inside one semantic region should become separate room priors. The
            projection is performed before connected-component extraction so
            multiple sampled heights do not duplicate one BEV room.
        include_structural: Whether structural labels such as wall and floor
            should be retained as object priors.
        include_object_priors: Whether semantic object instances should be
            emitted into the canonical map. Layout-only builders disable this
            because runtime ObjectNode identities must remain online facts.
        use_mesh_region_fallback: Whether to reconstruct invalid semantic
            region AABBs from semantic-mesh object bounds grouped by parent
            region. HM3D releases may expose ``[-inf, -inf, -inf]`` region
            bounds through Habitat-Sim even though the mesh is valid.
        max_grid_cells: Safety limit for top-down grid sampling.
        source: Source label stored in generated contract metadata.
    """

    topdown_resolution: float = 0.25
    floor_height_tolerance: float = 1.0
    min_room_area_m2: float = 0.25
    mask_dilation_radius_m: float = 0.35
    split_disconnected_components: bool = True
    include_structural: bool = False
    include_object_priors: bool = True
    use_mesh_region_fallback: bool = True
    max_grid_cells: int = 2_000_000
    source: str = "hm3d_groundtruth_semantic_scene"

    def __post_init__(self) -> None:
        """Validate numeric build parameters.

        Raises:
            ValueError: If a resolution, tolerance, area, radius, or grid limit
                is invalid.
        """

        if self.topdown_resolution <= 0.0:
            raise ValueError("topdown_resolution must be positive")
        if self.floor_height_tolerance < 0.0:
            raise ValueError("floor_height_tolerance must be non-negative")
        if self.min_room_area_m2 < 0.0:
            raise ValueError("min_room_area_m2 must be non-negative")
        if self.mask_dilation_radius_m < 0.0:
            raise ValueError("mask_dilation_radius_m must be non-negative")
        if self.max_grid_cells <= 0:
            raise ValueError("max_grid_cells must be positive")


@dataclass(frozen=True)
class HM3DGroundTruthPriorMapBuildResult:
    """Result of building an HM3D geometric prior map.

    Args:
        prior_map: Canonical static prior map.
        alignment: Identity Habitat-world alignment for simulation ranking.
        metadata: JSON-friendly build diagnostics.
    """

    prior_map: PriorMapData
    alignment: PriorMapAlignment
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HM3DSceneAssets:
    """Resolved HM3D files used to initialize one Habitat-Sim scene.

    Args:
        basis_glb: Geometric HM3D mesh.
        semantic_glb: Semantic HM3D mesh.
        semantic_txt: Instance-color to semantic-label mapping.
        navmesh: Habitat navigation mesh.
        scene_dataset_config: Optional Habitat scene-dataset configuration.
    """

    basis_glb: Path
    semantic_glb: Path
    semantic_txt: Path
    navmesh: Path
    scene_dataset_config: Optional[Path] = None


@dataclass(frozen=True)
class _SemanticObjectRecord:
    """Normalized Habitat semantic object metadata."""

    uid: str
    semantic_id: str
    label: str
    center_xyz: Vector3
    sizes_xyz: Vector3
    region_id: Optional[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _SemanticRegionRecord:
    """Normalized Habitat semantic region metadata."""

    uid: str
    semantic_id: str
    label: str
    center_xyz: Vector3
    sizes_xyz: Vector3
    level: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _RoomComponent:
    """Connected 2-D navigable component derived from one semantic region."""

    region: _SemanticRegionRecord
    component_index: int
    component_count: int
    cells: frozenset[_GridCell]
    boundary_xy: BoundaryXY
    centroid_xy: Vector2
    area_m2: float
    uid: str


class HM3DGroundTruthPriorMapBuilder:
    r"""Construct ``PriorMapData`` from Habitat semantic scene and navmesh.

    The room construction algorithm rasterizes the Habitat world \(x,z\) plane
    into grid cells. A cell belongs to a room when its center is navigable and
    falls inside a semantic region AABB. For each retained connected component:

    \[
    A = N_{cells} r^2,\quad
    c = \frac{1}{N_{cells}}\sum_i (x_i, z_i)
    \]

    where \(r\) is ``topdown_resolution``. Room-room edges are created when two
    room masks touch after dilation by ``mask_dilation_radius_m``.
    """

    def build_from_sim(
        self,
        sim: Any,
        scene_id: str,
        config: Optional[HM3DGroundTruthBuildConfig] = None,
        *,
        mesh_object_bounds: Optional[dict[str, dict[str, Any]]] = None,
    ) -> HM3DGroundTruthPriorMapBuildResult:
        """Build a geometric prior map from an existing Habitat simulator.

        Args:
            sim: Habitat-like simulator exposing ``semantic_scene`` and
                ``pathfinder`` attributes.
            scene_id: Stable scene id for generated prior ids.
            config: Optional build configuration.
            mesh_object_bounds: Optional object bounds extracted from a
                semantic mesh. Keys should be Habitat semantic object ids or
                HM3D semantic txt instance ids.

        Returns:
            Build result containing a canonical ``PriorMapData`` and identity
            alignment.

        Raises:
            ValueError: If the simulator does not expose the required semantic
                scene or pathfinder fields.
        """

        cfg = config or HM3DGroundTruthBuildConfig()
        scene_id = _safe_uid(scene_id or _read_optional_text(sim, "scene_id") or "hm3d_scene")
        semantic_scene = getattr(sim, "semantic_scene", None)
        pathfinder = getattr(sim, "pathfinder", None)
        if semantic_scene is None:
            raise ValueError("sim must expose semantic_scene")
        if pathfinder is None:
            raise ValueError("sim must expose pathfinder")

        regions = _extract_regions(
            semantic_scene,
            mesh_object_bounds=mesh_object_bounds or {},
            use_mesh_region_fallback=cfg.use_mesh_region_fallback,
        )
        objects = (
            _extract_objects(
                semantic_scene,
                include_structural=cfg.include_structural,
                mesh_object_bounds=mesh_object_bounds or {},
            )
            if cfg.include_object_priors
            else ()
        )
        bounds_min, bounds_max = _pathfinder_bounds(pathfinder, regions, objects)
        traversable_cells = _sample_traversable_cells(
            pathfinder,
            bounds_min,
            bounds_max,
            _sample_heights(regions, bounds_min, bounds_max),
            cfg,
        )
        room_components = _build_room_components(scene_id, regions, traversable_cells, cfg)
        parent_lookup = _assign_objects_to_rooms(objects, room_components, pathfinder)
        room_neighbor_pairs = _room_neighbor_pairs(room_components, cfg)
        rooms = _make_rooms(room_components, room_neighbor_pairs, cfg)
        prior_objects = _make_objects(scene_id, objects, parent_lookup, pathfinder, cfg)
        topology_edges = _make_topology_edges(rooms, prior_objects, room_neighbor_pairs, pathfinder, cfg)
        world_min, world_max = _world_bounds(rooms, prior_objects)
        label_counts = _label_counts(prior_objects)
        prior_map = PriorMapData(
            scene_id=scene_id,
            rooms=rooms,
            objects=prior_objects,
            topology_edges=topology_edges,
            source_format=cfg.source,
            frame_id="habitat_world",
            world_min=world_min,
            world_max=world_max,
            metadata={
                "authority": "semantic_scene_plus_navmesh",
                "no_episode_goal_positions": True,
                "topdown_resolution": cfg.topdown_resolution,
                "floor_height_tolerance": cfg.floor_height_tolerance,
                "min_room_area_m2": cfg.min_room_area_m2,
                "mask_dilation_radius_m": cfg.mask_dilation_radius_m,
                "split_disconnected_components": cfg.split_disconnected_components,
                "include_structural": cfg.include_structural,
                "include_object_priors": cfg.include_object_priors,
                "use_mesh_region_fallback": cfg.use_mesh_region_fallback,
                "region_geometry_sources": _region_geometry_sources(regions),
                "raw_region_count": len(regions),
                "raw_object_count": len(getattr(semantic_scene, "objects", ()) or ()),
                "object_count": len(prior_objects),
                "room_count": len(rooms),
                "topology_edge_count": len(topology_edges),
                "traversable_cell_count": len(traversable_cells),
                "label_counts": label_counts,
            },
        )
        alignment = PriorMapAlignment.identity(
            prior_frame_id=prior_map.frame_id,
            runtime_frame_id="habitat_world",
            confidence=1.0,
        )
        return HM3DGroundTruthPriorMapBuildResult(
            prior_map=prior_map,
            alignment=alignment,
            metadata=dict(prior_map.metadata),
        )

    def build_from_scene_dir(
        self,
        scene_dir: str | Path,
        scene_id: str = "",
        config: Optional[HM3DGroundTruthBuildConfig] = None,
    ) -> HM3DGroundTruthPriorMapBuildResult:
        """Build a geometric prior map by initializing Habitat-Sim.

        Args:
            scene_dir: HM3D scene directory containing ``*.basis.glb``,
                ``*.semantic.glb``, and ``*.basis.navmesh``.
            scene_id: Optional scene id override.
            config: Optional build configuration.

        Returns:
            Build result from ``build_from_sim``.

        Raises:
            RuntimeError: If Habitat-Sim is not installed or cannot initialize
                the scene.
            FileNotFoundError: If required scene assets are missing.
        """

        try:
            with open_hm3d_simulator(scene_dir) as (simulator, assets):
                inferred_scene_id = scene_id or _infer_scene_id_from_dir(Path(scene_dir))
                mesh_bounds = {}
                if config is None or config.include_object_priors or config.use_mesh_region_fallback:
                    mesh_bounds = _extract_mesh_object_bounds(assets.semantic_glb, assets.semantic_txt)
                return self.build_from_sim(
                    simulator,
                    inferred_scene_id,
                    config=config,
                    mesh_object_bounds=mesh_bounds,
                )
        except RuntimeError as exc:  # pragma: no cover - depends on Habitat assets.
            raise RuntimeError(f"Failed to build HM3D prior map from {scene_dir}") from exc


@contextmanager
def open_hm3d_simulator(scene_dir: str | Path) -> Iterator[tuple[Any, HM3DSceneAssets]]:
    """Open a Habitat-Sim instance and resolve the HM3D asset contract.

    Args:
        scene_dir: Directory containing one HM3D basis mesh, semantic mesh,
            semantic mapping, and NavMesh.

    Yields:
        A ``(simulator, assets)`` pair. The simulator is closed when the
        context exits.

    Raises:
        FileNotFoundError: If a required HM3D asset is missing.
        RuntimeError: If Habitat-Sim is unavailable or initialization fails.

    Notes:
        This is the single scene-loading boundary for offline map builders.
        Keeping it shared prevents the canonical prior-map and FloorPlan export
        paths from silently using different scene or coordinate inputs.
    """

    path = Path(scene_dir)
    assets = HM3DSceneAssets(
        basis_glb=_single_scene_asset(path, "*.basis.glb"),
        semantic_glb=_single_scene_asset(path, "*.semantic.glb"),
        semantic_txt=_single_scene_asset(path, "*.semantic.txt"),
        navmesh=_single_scene_asset(path, "*.basis.navmesh"),
        scene_dataset_config=_find_scene_dataset_config(path),
    )
    try:
        import habitat_sim  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install.
        raise RuntimeError("Habitat-Sim is required to read HM3D assets") from exc

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(assets.basis_glb)
    if assets.scene_dataset_config is not None and hasattr(sim_cfg, "scene_dataset_config_file"):
        sim_cfg.scene_dataset_config_file = str(assets.scene_dataset_config)
    if hasattr(sim_cfg, "semantic_scene_id"):
        sim_cfg.semantic_scene_id = str(assets.semantic_glb)
    if hasattr(sim_cfg, "enable_physics"):
        sim_cfg.enable_physics = False
    simulator = None
    try:
        agent_cfg = habitat_sim.AgentConfiguration()
        simulator = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        if assets.navmesh.exists() and hasattr(simulator.pathfinder, "load_nav_mesh"):
            simulator.pathfinder.load_nav_mesh(str(assets.navmesh))
    except Exception as exc:  # pragma: no cover - depends on Habitat assets.
        if simulator is not None and hasattr(simulator, "close"):
            simulator.close()
        raise RuntimeError(f"Failed to initialize HM3D scene from {path}") from exc
    try:
        yield simulator, assets
    finally:
        if simulator is not None and hasattr(simulator, "close"):
            simulator.close()


def build_hm3d_groundtruth_prior_map_from_sim(
    sim: Any,
    scene_id: str,
    config: Optional[HM3DGroundTruthBuildConfig] = None,
    *,
    mesh_object_bounds: Optional[dict[str, dict[str, Any]]] = None,
) -> HM3DGroundTruthPriorMapBuildResult:
    """Build an HM3D ground-truth prior map from an existing simulator.

    Args:
        sim: Habitat-like simulator.
        scene_id: Scene id for generated stable identifiers.
        config: Optional build configuration.
        mesh_object_bounds: Optional object bounds extracted from a semantic
            mesh.

    Returns:
        Build result with canonical prior map and identity alignment.
    """

    return HM3DGroundTruthPriorMapBuilder().build_from_sim(
        sim,
        scene_id,
        config=config,
        mesh_object_bounds=mesh_object_bounds,
    )


def build_hm3d_groundtruth_prior_map_from_scene_dir(
    scene_dir: str | Path,
    scene_id: str = "",
    config: Optional[HM3DGroundTruthBuildConfig] = None,
) -> HM3DGroundTruthPriorMapBuildResult:
    """Build an HM3D ground-truth prior map from a scene directory.

    Args:
        scene_dir: HM3D scene asset directory.
        scene_id: Optional scene id override.
        config: Optional build configuration.

    Returns:
        Build result with canonical prior map and identity alignment.
    """

    return HM3DGroundTruthPriorMapBuilder().build_from_scene_dir(scene_dir, scene_id=scene_id, config=config)


def write_hm3d_groundtruth_prior_map_with_alignment(
    result: HM3DGroundTruthPriorMapBuildResult,
    output_path: str | Path,
    *,
    alignment_output_path: str | Path = "",
) -> dict[str, str]:
    """Write generated prior map and optional identity alignment JSON.

    Args:
        result: Build result returned by the ground-truth builder.
        output_path: Destination canonical prior-map JSON path.
        alignment_output_path: Optional destination alignment JSON path.

    Returns:
        Dictionary containing written artifact paths.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.prior_map.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths = {"prior_map": str(output)}
    if alignment_output_path:
        alignment_path = Path(alignment_output_path)
        result.alignment.save(alignment_path)
        paths["alignment"] = str(alignment_path)
    return paths


def _extract_regions(
    semantic_scene: Any,
    *,
    mesh_object_bounds: dict[str, dict[str, Any]],
    use_mesh_region_fallback: bool,
) -> tuple[_SemanticRegionRecord, ...]:
    """Extract semantic regions and repair invalid HM3D region bounds.

    Habitat-Sim can expose the HM3D region list while leaving each region AABB
    at ``[-inf, -inf, -inf]``. The semantic mesh still contains per-instance
    geometry and every semantic object points to its parent region. Aggregating
    those transformed object bounds gives a conservative region envelope that
    is suitable for intersecting with the NavMesh.

    Args:
        semantic_scene: Habitat semantic-scene object.
        mesh_object_bounds: Mesh bounds keyed by Habitat or HM3D instance id.
        use_mesh_region_fallback: Whether to use grouped mesh geometry when a
            region AABB is invalid.

    Returns:
        Normalized semantic regions with a geometry-source marker.
    """

    regions = []
    seen_ids: dict[str, int] = {}
    mesh_region_bounds = (
        _mesh_region_bounds(semantic_scene, mesh_object_bounds)
        if use_mesh_region_fallback and mesh_object_bounds
        else {}
    )
    for index, region in enumerate(getattr(semantic_scene, "regions", ()) or ()):
        aabb = _read_aabb(region)
        center = _vector3_or_default(aabb.get("center"), (0.0, 0.0, 0.0))
        sizes = _vector3_or_default(aabb.get("sizes"), (0.0, 0.0, 0.0))
        geometry_source = "semantic_scene_aabb"
        fallback: Optional[dict[str, Any]] = None
        # 中文：无效 AABB 只能保留为语义库存，不能参与房间几何计算。
        # 否则 inf/nan 会把整个导航平面吸收到一个伪房间中。
        if not _valid_region_geometry(center, sizes):
            semantic_id = _read_id(region, fallback=index)
            fallback = mesh_region_bounds.get(str(semantic_id))
            if fallback is None:
                continue
            center = _vector3_or_default(fallback.get("center"), (0.0, 0.0, 0.0))
            sizes = _vector3_or_default(fallback.get("sizes"), (0.0, 0.0, 0.0))
            geometry_source = str(
                fallback.get("geometry_source", "semantic_glb_region_mesh_bounds")
            )
        if not _valid_region_geometry(center, sizes):
            continue
        semantic_id = _read_id(region, fallback=index)
        unique_region_id = _unique_semantic_id(semantic_id, seen_ids)
        regions.append(
            _SemanticRegionRecord(
                uid=unique_region_id,
                semantic_id=str(semantic_id),
                label=_category_name(region, fallback=f"region_{semantic_id}"),
                center_xyz=center,
                sizes_xyz=sizes,
                level=_read_level(region),
                metadata={
                    "semantic_region_id": str(semantic_id),
                    "aabb_center": list(center),
                    "aabb_sizes": list(sizes),
                    "floor_id": _read_level(region),
                    "geometry_source": geometry_source,
                    "mesh_object_count": fallback.get("object_count") if fallback else None,
                },
            )
        )
    return tuple(regions)


def _mesh_region_bounds(
    semantic_scene: Any,
    mesh_object_bounds: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate transformed semantic-object bounds by parent region.

    Args:
        semantic_scene: Habitat semantic scene exposing ``objects``.
        mesh_object_bounds: Bounds already expressed in Habitat world frame.

    Returns:
        Region id to conservative ``min``/``max``/``center``/``sizes`` bounds.
    """

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for obj in getattr(semantic_scene, "objects", ()) or ():
        region_id = _read_parent_region_id(obj)
        if region_id is None:
            continue
        bound = _mesh_bound_for_semantic_id(mesh_object_bounds, _read_id(obj, fallback=0))
        if bound is not None:
            grouped.setdefault(str(region_id), []).append((_category_name(obj, fallback=""), bound))

    result: dict[str, dict[str, Any]] = {}
    for region_id, labeled_bounds in grouped.items():
        floor_bounds = [
            bound
            for label, bound in labeled_bounds
            if _norm(label) in {"floor", "ground", "flooring"}
        ]
        # 中文：优先使用 floor 几何确定房间的平面范围和楼层高度，避免把
        # 家具/墙体的竖直包围盒误当成房间高度；没有 floor 标注时才退化到
        # 该 region 的全部语义网格实例。
        bounds = floor_bounds or [bound for _, bound in labeled_bounds]
        points = [point for bound in bounds for point in (bound.get("min"), bound.get("max"))]
        points = [point for point in points if point is not None]
        if not points:
            continue
        minimum = tuple(min(float(point[index]) for point in points) for index in range(3))
        maximum = tuple(max(float(point[index]) for point in points) for index in range(3))
        center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))
        sizes = tuple(maximum[index] - minimum[index] for index in range(3))
        result[region_id] = {
            "min": minimum,
            "max": maximum,
            "center": center,
            "sizes": sizes,
            "object_count": len(bounds),
            "geometry_source": "semantic_glb_floor_mesh_bounds" if floor_bounds else "semantic_glb_region_mesh_bounds",
        }
    return result


def _region_geometry_sources(regions: Sequence[_SemanticRegionRecord]) -> dict[str, int]:
    """Count region geometry provenance for build diagnostics."""

    counts: dict[str, int] = {}
    for region in regions:
        source = str(region.metadata.get("geometry_source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _extract_objects(
    semantic_scene: Any,
    *,
    include_structural: bool,
    mesh_object_bounds: dict[str, dict[str, Any]],
) -> tuple[_SemanticObjectRecord, ...]:
    objects = []
    seen_ids: dict[str, int] = {}
    for index, obj in enumerate(getattr(semantic_scene, "objects", ()) or ()):
        label = _category_name(obj, fallback=f"object_{index}")
        if not include_structural and _norm(label) in _STRUCTURAL_LABELS:
            continue
        aabb = _read_aabb(obj)
        center = _vector3_or_default(aabb.get("center"), (0.0, 0.0, 0.0))
        sizes = _vector3_or_default(aabb.get("sizes"), (0.0, 0.0, 0.0))
        semantic_id = _read_id(obj, fallback=index)
        unique_object_id = _unique_semantic_id(semantic_id, seen_ids)
        mesh_bound = _mesh_bound_for_semantic_id(mesh_object_bounds, semantic_id)
        geometry_source = "semantic_scene_aabb"
        if mesh_bound is not None and _degenerate_box(center, sizes):
            center = _vector3_or_default(mesh_bound.get("center"), center)
            sizes = _vector3_or_default(mesh_bound.get("sizes"), sizes)
            geometry_source = "semantic_glb_texture_bounds"
        if not _valid_object_geometry(center, sizes):
            # 中文：没有可验证空间范围的对象不进入静态几何层，避免零点投影污染 BEV。
            continue
        region_id = _read_parent_region_id(obj)
        objects.append(
            _SemanticObjectRecord(
                uid=unique_object_id,
                semantic_id=str(semantic_id),
                label=label,
                center_xyz=center,
                sizes_xyz=sizes,
                region_id=region_id,
                metadata={
                    "semantic_object_id": str(semantic_id),
                    "semantic_region_id": region_id,
                    "aabb_center": list(center),
                    "aabb_sizes": list(sizes),
                    "geometry_source": geometry_source,
                    "semantic_glb_vertex_count": mesh_bound.get("vertex_count") if mesh_bound else None,
                },
            )
        )
    return tuple(objects)


def extract_hm3d_mesh_object_bounds(
    semantic_glb: str | Path,
    semantic_txt: str | Path,
) -> dict[str, dict[str, Any]]:
    """Extract semantic-mesh instance bounds in the Habitat world frame.

    Args:
        semantic_glb: HM3D semantic mesh path.
        semantic_txt: HM3D color-to-instance annotation path.

    Returns:
        Mapping from HM3D instance id to transformed AABB metadata. An empty
        mapping is returned when the optional mesh parsing dependencies are not
        available or the assets contain no readable textured geometry.
    """

    return _extract_mesh_object_bounds(Path(semantic_glb), Path(semantic_txt))


def _extract_mesh_object_bounds(semantic_glb: Path, semantic_txt: Path) -> dict[str, dict[str, Any]]:
    """Extract object AABBs from HM3D semantic mesh texture ids.

    HM3D Habitat semantic objects can expose degenerate semantic-scene AABBs for
    small objects. The semantic mesh stores instance identity in a texture
    color, and ``*.semantic.txt`` maps that color to an instance id. This helper
    samples face-center UV colors, groups matched faces by instance id, and
    computes per-instance mesh-space AABBs.

    Args:
        semantic_glb: Path to ``*.semantic.glb``.
        semantic_txt: Path to ``*.semantic.txt`` color metadata.

    Returns:
        Mapping from semantic txt instance id to mesh bounds metadata.
    """

    color_to_instance = _read_semantic_txt_color_lookup(semantic_txt)
    if not color_to_instance:
        return {}
    try:
        import numpy as np  # type: ignore
        import trimesh  # type: ignore
    except Exception:
        return {}

    target_colors = np.array([item["rgb"] for item in color_to_instance.values()], dtype=np.int32)
    target_keys = list(color_to_instance.keys())
    bounds: dict[str, dict[str, Any]] = {}
    try:
        scene = trimesh.load(str(semantic_glb), force="scene", process=False)
    except Exception:
        return {}
    for geom in getattr(scene, "geometry", {}).values():
        vertices = np.asarray(getattr(geom, "vertices", ()), dtype=float)
        if vertices.size == 0:
            continue
        visual = getattr(geom, "visual", None)
        uv = np.asarray(getattr(visual, "uv", ()), dtype=float)
        faces = np.asarray(getattr(geom, "faces", ()), dtype=int)
        image = _texture_image_array(visual)
        if uv.shape[0] != vertices.shape[0] or image is None:
            continue
        if faces.size:
            face_uv = np.mean(uv[faces], axis=1)
            recognized_faces = _recognized_texture_samples(face_uv, image, target_colors, target_keys, np)
            for key, face_indices in recognized_faces.items():
                selected = np.unique(faces[face_indices].reshape(-1))
                selected_vertices = _semantic_mesh_to_habitat(vertices[selected], np)
                if selected_vertices.size == 0:
                    continue
                _update_mesh_bound(bounds, color_to_instance[key]["instance_id"], selected_vertices, np)
            continue
        recognized_vertices = _recognized_texture_samples(uv, image, target_colors, target_keys, np)
        for key, vertex_indices in recognized_vertices.items():
            selected_vertices = _semantic_mesh_to_habitat(vertices[vertex_indices], np)
            if selected_vertices.size == 0:
                continue
            _update_mesh_bound(bounds, color_to_instance[key]["instance_id"], selected_vertices, np)
    return bounds


def _semantic_mesh_to_habitat(vertices: Any, np: Any) -> Any:
    """Convert HM3D semantic-mesh coordinates to Habitat world coordinates.

    HM3D semantic GLB uses an OpenGL-style ``(x, y, z)`` mesh frame while
    Habitat-Sim exposes the scene in ``(x, y, z)`` with the mesh's vertical
    axis mapped to Habitat ``y`` and the mesh forward axis mapped to negative
    Habitat ``z``. Thus ``(x, y, z)_mesh -> (x, z, -y)_habitat``.

    Args:
        vertices: Array-like mesh vertices in the semantic GLB frame.
        np: Imported numpy module.

    Returns:
        Vertices in the Habitat world frame.
    """

    values = np.asarray(vertices, dtype=float)
    return np.stack((values[:, 0], values[:, 2], -values[:, 1]), axis=1)


def _read_semantic_txt_color_lookup(semantic_txt: Path) -> dict[str, dict[str, Any]]:
    """Read HM3D semantic color-to-instance metadata.

    Args:
        semantic_txt: Path to the HM3D ``*.semantic.txt`` CSV-like file.

    Returns:
        Mapping from hex color string to instance id, label, region id, and RGB.
    """

    lookup: dict[str, dict[str, Any]] = {}
    try:
        with semantic_txt.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                instance_id = str(row[0]).strip()
                color_hex = str(row[1]).strip().lstrip("#").upper()
                if not instance_id or len(color_hex) != 6:
                    continue
                try:
                    rgb = tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))
                except ValueError:
                    continue
                lookup[color_hex] = {
                    "instance_id": instance_id,
                    "label": str(row[2]).strip().strip('"'),
                    "region_id": str(row[3]).strip(),
                    "rgb": rgb,
                }
    except OSError:
        return {}
    return lookup


def _texture_image_array(visual: Any) -> Any:
    """Return a visual material texture as an RGB array when available.

    Args:
        visual: Trimesh visual object.

    Returns:
        RGB numpy array, or ``None`` when the visual has no readable texture.
    """

    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None) if material is not None else None
    if image is None and material is not None:
        image = getattr(material, "image", None)
    if image is None:
        return None
    try:
        import numpy as np  # type: ignore

        return np.asarray(image.convert("RGB"))
    except Exception:
        return None


def _recognized_texture_samples(
    sample_uv: Any,
    image: Any,
    target_colors: Any,
    target_keys: list[str],
    np: Any,
) -> dict[str, Any]:
    """Map sampled UV colors to known HM3D semantic instance colors.

    Args:
        sample_uv: UV coordinates to sample.
        image: RGB texture image array.
        target_colors: Known semantic RGB colors.
        target_keys: Hex color keys aligned with ``target_colors``.
        np: Imported numpy module.

    Returns:
        Mapping from matched color key to sample indices. The function tests
        both V-axis conventions and keeps the one with more recognized samples.
    """

    height, width = image.shape[:2]
    best: dict[str, Any] = {}
    best_count = -1
    for flip_v in (True, False):
        u = np.clip(np.rint(sample_uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
        v_values = 1.0 - sample_uv[:, 1] if flip_v else sample_uv[:, 1]
        v = np.clip(np.rint(v_values * (height - 1)).astype(int), 0, height - 1)
        colors = image[v, u, :3].astype(np.int32)
        unique_colors, inverse = np.unique(colors, axis=0, return_inverse=True)
        mapped: dict[str, list[Any]] = {}
        for color_index, color in enumerate(unique_colors):
            distances = np.sum((target_colors - color) * (target_colors - color), axis=1)
            nearest = int(np.argmin(distances))
            if int(distances[nearest]) > _TEXTURE_COLOR_TOLERANCE_SQUARED:
                continue
            key = target_keys[nearest]
            mapped.setdefault(key, []).append(np.flatnonzero(inverse == color_index))
        recognized = {
            key: np.unique(np.concatenate(indices))
            for key, indices in mapped.items()
            if indices
        }
        count = sum(len(indices) for indices in recognized.values())
        if count > best_count:
            best = recognized
            best_count = count
    return best


def _update_mesh_bound(bounds: dict[str, dict[str, Any]], instance_id: str, vertices: Any, np: Any) -> None:
    """Merge selected mesh vertices into one per-instance AABB.

    Args:
        bounds: Mutable output bounds dictionary.
        instance_id: HM3D semantic txt instance id.
        vertices: Selected vertex positions.
        np: Imported numpy module.
    """

    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)
    current = bounds.get(str(instance_id))
    if current is not None:
        min_point = np.minimum(np.asarray(current["min"], dtype=float), min_point)
        max_point = np.maximum(np.asarray(current["max"], dtype=float), max_point)
        vertex_count = int(current.get("vertex_count", 0)) + int(vertices.shape[0])
    else:
        vertex_count = int(vertices.shape[0])
    center = (min_point + max_point) / 2.0
    sizes = max_point - min_point
    bounds[str(instance_id)] = {
        "min": tuple(float(value) for value in min_point),
        "max": tuple(float(value) for value in max_point),
        "center": tuple(float(value) for value in center),
        "sizes": tuple(float(value) for value in sizes),
        "vertex_count": vertex_count,
    }


def _mesh_bound_for_semantic_id(
    mesh_object_bounds: dict[str, dict[str, Any]],
    semantic_id: str,
) -> Optional[dict[str, Any]]:
    """Find mesh bounds for either Habitat semantic id or semantic txt id.

    Args:
        mesh_object_bounds: Bounds extracted from semantic mesh textures.
        semantic_id: Habitat semantic object id, such as ``tv_349``.

    Returns:
        Matching mesh bounds, or ``None``.
    """

    for key in (str(semantic_id), _semantic_txt_instance_id(semantic_id)):
        if key and key in mesh_object_bounds:
            return mesh_object_bounds[key]
    return None


def _semantic_txt_instance_id(semantic_id: str) -> str:
    """Convert Habitat semantic object ids to HM3D semantic txt instance ids.

    Args:
        semantic_id: Habitat object id such as ``tv_349`` or ``349``.

    Returns:
        Numeric suffix when present; otherwise the original id string.
    """

    text = str(semantic_id)
    if "_" in text:
        suffix = text.rsplit("_", maxsplit=1)[-1]
        if suffix.isdigit():
            return suffix
    return text


def _degenerate_box(center: Vector3, sizes: Vector3) -> bool:
    """Return whether an AABB lacks usable spatial extent.

    Args:
        center: AABB center. Present for interface symmetry.
        sizes: AABB side lengths in meters.

    Returns:
        ``True`` when all side lengths are effectively zero.
    """

    return all(abs(value) <= 1e-9 for value in sizes)


def _pathfinder_bounds(
    pathfinder: Any,
    regions: Sequence[_SemanticRegionRecord],
    objects: Sequence[_SemanticObjectRecord],
) -> tuple[Vector3, Vector3]:
    if hasattr(pathfinder, "get_bounds"):
        raw_bounds = pathfinder.get_bounds()
        if isinstance(raw_bounds, Sequence) and len(raw_bounds) == 2:
            try:
                return _vector3(raw_bounds[0]), _vector3(raw_bounds[1])
            except ValueError:
                pass
    points = [region.center_xyz for region in regions] + [obj.center_xyz for obj in objects]
    if not points:
        return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    margin = 1.0
    return (
        (
            min(point[0] for point in points) - margin,
            min(point[1] for point in points) - margin,
            min(point[2] for point in points) - margin,
        ),
        (
            max(point[0] for point in points) + margin,
            max(point[1] for point in points) + margin,
            max(point[2] for point in points) + margin,
        ),
    )


def _sample_traversable_cells(
    pathfinder: Any,
    bounds_min: Vector3,
    bounds_max: Vector3,
    sample_heights: Sequence[float],
    config: HM3DGroundTruthBuildConfig,
) -> frozenset[_GridCell]:
    resolution = config.topdown_resolution
    x_steps = max(1, int(math.ceil((bounds_max[0] - bounds_min[0]) / resolution)))
    z_steps = max(1, int(math.ceil((bounds_max[2] - bounds_min[2]) / resolution)))
    cell_count = (x_steps + 1) * (z_steps + 1)
    if cell_count > config.max_grid_cells:
        raise ValueError(
            f"navmesh grid would sample {cell_count} cells, above max_grid_cells={config.max_grid_cells}"
        )
    cells: set[_GridCell] = set()
    for ix in range(x_steps + 1):
        x = bounds_min[0] + ix * resolution
        for iz in range(z_steps + 1):
            z = bounds_min[2] + iz * resolution
            for y in sample_heights:
                if _is_navigable(pathfinder, (x, y, z)):
                    cells.add(
                        (
                            _world_to_grid(x, resolution),
                            _world_to_grid(z, resolution),
                            _world_to_grid(y, resolution),
                        )
                    )
    return frozenset(cells)


def _build_room_components(
    scene_id: str,
    regions: Sequence[_SemanticRegionRecord],
    traversable_cells: frozenset[_GridCell],
    config: HM3DGroundTruthBuildConfig,
) -> tuple[_RoomComponent, ...]:
    components: list[_RoomComponent] = []
    emitted_uids: dict[str, int] = {}
    for region in regions:
        region_cells = frozenset(
            cell
            for cell in traversable_cells
            if _cell_center_in_region(cell, region, config)
        )
        if not region_cells:
            continue
        # 中文：房间先验最终服务于 BEV 和高层语义推理，因此必须先把
        # NavMesh 的多个采样高度投影到同一楼层的 (x, z) 平面；如果直接
        # 在 (x, z, y) 索引上做连通域，同一个房间会因高度采样重复。
        projected_cells = _project_cells_to_floor(region_cells)
        raw_components = (
            _connected_components(projected_cells)
            if config.split_disconnected_components
            else (projected_cells,)
        )
        kept = [
            comp
            for comp in raw_components
            if _cell_area_m2(comp, config.topdown_resolution) >= config.min_room_area_m2
        ]
        for index, cells in enumerate(kept):
            component_count = len(kept)
            uid = _dedupe_uid(_room_uid(scene_id, region.uid, index, component_count), emitted_uids)
            boundary = _boundary_from_cells(cells, config.topdown_resolution)
            centroid = _centroid_from_cells(cells, config.topdown_resolution)
            components.append(
                _RoomComponent(
                    region=region,
                    component_index=index,
                    component_count=component_count,
                    cells=frozenset(cells),
                    boundary_xy=boundary,
                    centroid_xy=centroid,
                    area_m2=_cell_area_m2(cells, config.topdown_resolution),
                    uid=uid,
                )
            )
    return tuple(components)


def _project_cells_to_floor(cells: frozenset[_GridCell]) -> frozenset[_GridCell]:
    """Collapse NavMesh samples onto a single semantic-region BEV plane.

    Args:
        cells: Navigable cells represented as ``(ix, iz, iy)``.

    Returns:
        One deterministic representative cell for each ``(ix, iz)`` location.
    """

    representatives: dict[tuple[int, int], _GridCell] = {}
    for cell in cells:
        key = (cell[0], cell[1])
        current = representatives.get(key)
        if current is None or (cell[2], cell) < (current[2], current):
            representatives[key] = cell
    return frozenset(representatives.values())


def _connected_components(cells: frozenset[_GridCell]) -> tuple[frozenset[_GridCell], ...]:
    """Extract four-connected components in the projected ``(ix, iz)`` plane.

    Args:
        cells: One representative NavMesh cell per 2-D location.

    Returns:
        Deterministically ordered connected components.
    """

    by_xy = {(cell[0], cell[1]): cell for cell in cells}
    remaining = set(by_xy)
    components = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = {by_xy[start]}
        while queue:
            ix, iz = queue.popleft()
            for neighbor in ((ix + 1, iz), (ix - 1, iz), (ix, iz + 1), (ix, iz - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(by_xy[neighbor])
                    queue.append(neighbor)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda comp: (-len(comp), min(comp))))


def _assign_objects_to_rooms(
    objects: Sequence[_SemanticObjectRecord],
    room_components: Sequence[_RoomComponent],
    pathfinder: Any,
) -> dict[str, tuple[Optional[str], str]]:
    parent_lookup: dict[str, tuple[Optional[str], str]] = {}
    by_region: dict[str, list[_RoomComponent]] = {}
    for component in room_components:
        by_region.setdefault(component.region.semantic_id, []).append(component)
    for obj in objects:
        snap = _snap_point(pathfinder, obj.center_xyz)
        candidates = by_region.get(str(obj.region_id), []) if obj.region_id is not None else []
        containing = [room for room in candidates if _point_in_boundary(_xz(obj.center_xyz), room.boundary_xy)]
        if not containing:
            containing = [room for room in room_components if _point_in_boundary(_xz(obj.center_xyz), room.boundary_xy)]
        if not containing:
            containing = [room for room in room_components if _point_in_boundary(_xz(snap), room.boundary_xy)]
        if containing:
            room = min(containing, key=lambda item: _distance_xy(item.centroid_xy, _xz(obj.center_xyz)))
            source = "semantic_parent_region" if room in candidates else "geometry_containment"
        elif room_components:
            room = min(room_components, key=lambda item: _distance_xy(item.centroid_xy, _xz(snap)))
            source = "nearest_nav_room"
        else:
            continue
        parent_lookup[obj.uid] = (room.uid, source)
    return parent_lookup


def _room_neighbor_pairs(
    room_components: Sequence[_RoomComponent],
    config: HM3DGroundTruthBuildConfig,
) -> frozenset[tuple[str, str]]:
    dilation_cells = max(1, int(math.ceil(config.mask_dilation_radius_m / config.topdown_resolution)))
    # 中文：房间已经是 BEV 语义单元，拓扑也必须在同一个二维坐标系中
    # 判断相邻关系；保留 y 作为 key 会把不同采样高度的相邻区域错误断开。
    cell_owner: dict[tuple[int, int], int] = {}
    for index, component in enumerate(room_components):
        for cell in component.cells:
            cell_owner.setdefault((cell[0], cell[1]), index)
    pairs: set[tuple[str, str]] = set()
    for index, component in enumerate(room_components):
        for ix, iz, iy in component.cells:
            for dx in range(-dilation_cells, dilation_cells + 1):
                for dz in range(-dilation_cells, dilation_cells + 1):
                    if dx * dx + dz * dz > dilation_cells * dilation_cells:
                        continue
                    owner = cell_owner.get((ix + dx, iz + dz))
                    if owner is None or owner == index:
                        continue
                    left, right = sorted((component.uid, room_components[owner].uid))
                    pairs.add((left, right))
    return frozenset(pairs)


def _make_rooms(
    room_components: Sequence[_RoomComponent],
    room_neighbor_pairs: frozenset[tuple[str, str]],
    config: HM3DGroundTruthBuildConfig,
) -> tuple[PriorRoom, ...]:
    neighbors: dict[str, set[str]] = {component.uid: set() for component in room_components}
    for left, right in room_neighbor_pairs:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    rooms = []
    for component in room_components:
        rooms.append(
            PriorRoom(
                uid=component.uid,
                label=component.region.label,
                boundary_xy=component.boundary_xy,
                centroid_xy=component.centroid_xy,
                neighbors=tuple(sorted(neighbors.get(component.uid, ()))),
                level=component.region.level,
                confidence=1.0,
                source=config.source,
                metadata={
                    **component.region.metadata,
                    "component_index": component.component_index,
                    "component_count": component.component_count,
                    "nav_sample_count": len(component.cells),
                    "area_m2": component.area_m2,
                    "boundary_method": "navigable_mask_boundary",
                },
            )
        )
    return tuple(sorted(rooms, key=lambda room: room.uid))


def _make_objects(
    scene_id: str,
    objects: Sequence[_SemanticObjectRecord],
    parent_lookup: dict[str, tuple[str, str]],
    pathfinder: Any,
    config: HM3DGroundTruthBuildConfig,
) -> tuple[PriorObject, ...]:
    prior_objects = []
    for obj in objects:
        snap = _snap_point(pathfinder, obj.center_xyz)
        distance_to_navmesh = _distance_xyz(obj.center_xyz, snap)
        parent_room_uid, containment_source = parent_lookup.get(obj.uid, (None, "unassigned"))
        prior_objects.append(
            PriorObject(
                uid=f"prior_object:{_safe_uid(scene_id)}:{_safe_uid(obj.uid)}",
                label=obj.label,
                position_xyz=obj.center_xyz,
                parent_room_uid=parent_room_uid,
                exact=True,
                confidence=1.0,
                source=config.source,
                aliases=tuple(_aliases_for_label(obj.label)),
                metadata={
                    **obj.metadata,
                    "navmesh_snap_point": list(snap),
                    "distance_to_navmesh": distance_to_navmesh,
                    "containment_source": containment_source,
                },
            )
        )
    return tuple(sorted(prior_objects, key=lambda obj: obj.uid))


def _make_topology_edges(
    rooms: Sequence[PriorRoom],
    objects: Sequence[PriorObject],
    room_neighbor_pairs: frozenset[tuple[str, str]],
    pathfinder: Any,
    config: HM3DGroundTruthBuildConfig,
) -> tuple[PriorTopologyEdge, ...]:
    room_by_uid = {room.uid: room for room in rooms}
    edges = []
    for left, right in sorted(room_neighbor_pairs):
        if left not in room_by_uid or right not in room_by_uid:
            continue
        centroid_distance = _distance_xy(
            room_by_uid[left].centroid_xy or (0.0, 0.0),
            room_by_uid[right].centroid_xy or (0.0, 0.0),
        )
        geodesic_distance = _geodesic_distance(
            pathfinder,
            _room_centroid_xyz(room_by_uid[left]),
            _room_centroid_xyz(room_by_uid[right]),
        )
        edge_weight = geodesic_distance if geodesic_distance is not None else centroid_distance
        edges.append(
            PriorTopologyEdge(
                uid=f"edge:{left}:{right}",
                source_uid=left,
                target_uid=right,
                edge_type="room-room",
                relation="connected",
                bidirectional=True,
                confidence=1.0,
                weight=edge_weight,
                source=config.source,
                metadata={
                    "method": "dilated_navigable_mask_contact",
                    "mask_dilation_radius_m": config.mask_dilation_radius_m,
                    "centroid_distance_m": centroid_distance,
                    "geodesic_distance_m": geodesic_distance,
                    "distance_source": "pathfinder_geodesic" if geodesic_distance is not None else "centroid_euclidean",
                },
            )
        )
    for obj in objects:
        if not obj.parent_room_uid:
            continue
        edges.append(
            PriorTopologyEdge(
                uid=f"edge:{obj.parent_room_uid}:{obj.uid}",
                source_uid=obj.parent_room_uid,
                target_uid=obj.uid,
                edge_type="room-object",
                relation="contains",
                bidirectional=False,
                confidence=1.0,
                source=config.source,
                metadata={"method": obj.metadata.get("containment_source", "unknown")},
            )
        )
    return tuple(edges)


def _world_bounds(
    rooms: Sequence[PriorRoom],
    objects: Sequence[PriorObject],
) -> tuple[Optional[Vector2], Optional[Vector2]]:
    points: list[Vector2] = []
    for room in rooms:
        points.extend(room.boundary_xy)
    for obj in objects:
        if obj.position_xyz is not None:
            points.append((obj.position_xyz[0], obj.position_xyz[2]))
    if not points:
        return None, None
    return (
        (min(point[0] for point in points), min(point[1] for point in points)),
        (max(point[0] for point in points), max(point[1] for point in points)),
    )


def _label_counts(objects: Sequence[PriorObject]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obj in objects:
        counts[obj.label] = counts.get(obj.label, 0) + 1
    return dict(sorted(counts.items()))


def _single_scene_asset(scene_dir: Path, pattern: str) -> Path:
    matches = sorted(scene_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern} under {scene_dir}")
    return matches[0]


def _infer_scene_id_from_dir(scene_dir: Path) -> str:
    name = scene_dir.name
    if "-" in name:
        return name.split("-", maxsplit=1)[1]
    return name


def _find_scene_dataset_config(scene_dir: Path) -> Optional[Path]:
    """Find the HM3D scene-dataset config near a scene directory.

    Args:
        scene_dir: HM3D scene directory.

    Returns:
        Path to a scene-dataset config if present; otherwise ``None``.
    """

    candidates = (
        scene_dir.parent / f"hm3d_annotated_{scene_dir.parent.name}_basis.scene_dataset_config.json",
        scene_dir.parent / "hm3d_annotated_basis.scene_dataset_config.json",
        scene_dir.parent.parent / "hm3d_annotated_basis.scene_dataset_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_aabb(value: Any) -> dict[str, Any]:
    aabb = getattr(value, "aabb", None) or getattr(value, "bbox", None) or getattr(value, "obb", None)
    if aabb is None:
        return {"center": None, "sizes": None}
    center = _maybe_call(getattr(aabb, "center", None))
    sizes = _maybe_call(getattr(aabb, "sizes", None))
    if sizes is None:
        sizes = _maybe_call(getattr(aabb, "size", None))
    if sizes is None:
        sizes = _maybe_call(getattr(aabb, "extent", None))
    if sizes is None:
        half_extents = _maybe_call(getattr(aabb, "half_extents", None))
        if half_extents is not None:
            half = _vector3(half_extents)
            sizes = (half[0] * 2.0, half[1] * 2.0, half[2] * 2.0)
    if sizes is None and hasattr(aabb, "max") and hasattr(aabb, "min"):
        min_point = _vector3(_maybe_call(getattr(aabb, "min")))
        max_point = _vector3(_maybe_call(getattr(aabb, "max")))
        center = (
            (min_point[0] + max_point[0]) / 2.0,
            (min_point[1] + max_point[1]) / 2.0,
            (min_point[2] + max_point[2]) / 2.0,
        )
        sizes = (
            max_point[0] - min_point[0],
            max_point[1] - min_point[1],
            max_point[2] - min_point[2],
        )
    if sizes is not None and str(type(sizes)).endswith("Range3D'>"):
        sizes = None
    return {"center": center, "sizes": sizes}


def _category_name(value: Any, *, fallback: str) -> str:
    category = getattr(value, "category", None)
    candidates = (getattr(category, "name", None), getattr(value, "label", None), getattr(value, "name", None), category)
    for candidate in candidates:
        if candidate is None:
            continue
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                continue
        text = str(candidate).strip()
        if text:
            return text
    return fallback


def _maybe_call(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _read_id(value: Any, *, fallback: int) -> str:
    for name in ("id", "semantic_id", "object_id", "index"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            return str(candidate)
    return str(fallback)


def _read_parent_region_id(value: Any) -> Optional[str]:
    for name in ("region_id", "parent_region_id", "semantic_region_id"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            return str(candidate)
    region = getattr(value, "region", None)
    if region is not None:
        return _read_id(region, fallback=0)
    return None


def _unique_semantic_id(raw_id: str, seen_ids: dict[str, int]) -> str:
    """Return a stable unique id when Habitat exposes duplicate raw ids.

    Args:
        raw_id: Raw semantic id from Habitat-Sim.
        seen_ids: Mutable count table.

    Returns:
        Original id for first occurrence, suffixed id for duplicates.
    """

    key = str(raw_id)
    count = seen_ids.get(key, 0)
    seen_ids[key] = count + 1
    if count == 0:
        return key
    return f"{key}_{count}"


def _dedupe_uid(uid: str, seen_ids: dict[str, int]) -> str:
    """Return a unique uid while preserving the first emitted uid.

    Args:
        uid: Candidate uid.
        seen_ids: Mutable count table.

    Returns:
        Original uid for first occurrence, suffixed uid for duplicates.
    """

    count = seen_ids.get(uid, 0)
    seen_ids[uid] = count + 1
    if count == 0:
        return uid
    return f"{uid}:duplicate_{count}"


def _read_level(value: Any) -> int:
    for name in ("level_id", "floor_id", "level"):
        candidate = getattr(value, name, None)
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return 0


def _read_optional_text(value: Any, name: str) -> str:
    candidate = getattr(value, name, "")
    return str(candidate).strip() if candidate is not None else ""


def _sample_heights(
    regions: Sequence[_SemanticRegionRecord],
    bounds_min: Vector3,
    bounds_max: Vector3,
) -> tuple[float, ...]:
    heights = set()
    for region in regions:
        center_y = float(region.center_xyz[1])
        heights.add(round(center_y, 3))
        if region.sizes_xyz[1] > 0.0:
            heights.add(round(center_y - abs(region.sizes_xyz[1]) / 2.0, 3))
            heights.add(round(center_y + abs(region.sizes_xyz[1]) / 2.0, 3))
    heights = sorted(heights)
    if heights:
        return tuple(heights)
    return ((bounds_min[1] + bounds_max[1]) / 2.0,)


def _is_navigable(pathfinder: Any, point_xyz: Vector3) -> bool:
    if hasattr(pathfinder, "is_navigable"):
        try:
            return bool(pathfinder.is_navigable(point_xyz))
        except TypeError:
            return bool(pathfinder.is_navigable(list(point_xyz)))
    return False


def _snap_point(pathfinder: Any, point_xyz: Vector3) -> Vector3:
    if hasattr(pathfinder, "snap_point"):
        try:
            snapped = pathfinder.snap_point(point_xyz)
        except TypeError:
            snapped = pathfinder.snap_point(list(point_xyz))
        try:
            return _vector3(snapped)
        except ValueError:
            return point_xyz
    return point_xyz


def _cell_center(cell: _GridCell, resolution: float) -> Vector2:
    return (float(cell[0]) * resolution, float(cell[1]) * resolution)


def _world_to_grid(value: float, resolution: float) -> int:
    return int(round(float(value) / resolution))


def _cell_center_in_region(
    cell: _GridCell,
    region: _SemanticRegionRecord,
    config: HM3DGroundTruthBuildConfig,
) -> bool:
    resolution = config.topdown_resolution
    x, z = _cell_center(cell, resolution)
    y = float(cell[2]) * resolution
    half_x = abs(region.sizes_xyz[0]) / 2.0
    half_y = abs(region.sizes_xyz[1]) / 2.0
    half_z = abs(region.sizes_xyz[2]) / 2.0
    if half_x <= 0.0 or half_z <= 0.0:
        return False
    if not (
        region.center_xyz[1] - half_y - config.floor_height_tolerance
        <= y
        <= region.center_xyz[1] + half_y + config.floor_height_tolerance
    ):
        return False
    return (
        region.center_xyz[0] - half_x <= x <= region.center_xyz[0] + half_x
        and region.center_xyz[2] - half_z <= z <= region.center_xyz[2] + half_z
    )


def _cell_area_m2(cells: Iterable[_GridCell], resolution: float) -> float:
    return float(len(tuple(cells))) * resolution * resolution


def _boundary_from_cells(cells: Iterable[_GridCell], resolution: float) -> BoundaryXY:
    """Extract the outer polygon of a connected grid component.

    Args:
        cells: Connected grid cells represented as ``(ix, iz, iy)``.
        resolution: Cell size in meters.

    Returns:
        A counter-clockwise polygon in the Habitat ``(x, z)`` plane. If the
        component cannot be polygonized, its conservative bounding rectangle
        is returned as a deterministic fallback.
    """

    cell_tuple = tuple(cells)
    if not cell_tuple:
        return ()
    cell_set = set(cell_tuple)
    edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    half = resolution / 2.0

    for ix, iz, iy in cell_tuple:
        left, right = ix * resolution - half, ix * resolution + half
        bottom, top = iz * resolution - half, iz * resolution + half
        # 中文：只保留没有同层邻居的四条边，得到自由空间连通域的真实轮廓。
        if (ix - 1, iz, iy) not in cell_set:
            edges.add(((left, bottom), (left, top)))
        if (ix + 1, iz, iy) not in cell_set:
            edges.add(((right, top), (right, bottom)))
        if (ix, iz - 1, iy) not in cell_set:
            edges.add(((right, bottom), (left, bottom)))
        if (ix, iz + 1, iy) not in cell_set:
            edges.add(((left, top), (right, top)))

    loops: list[list[tuple[float, float]]] = []
    outgoing: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    for key in outgoing:
        outgoing[key].sort()
    while edges:
        start, end = min(edges)
        edges.remove((start, end))
        loop = [start, end]
        current = end
        while current != start:
            candidates = [candidate for candidate in outgoing.get(current, ()) if (current, candidate) in edges]
            if not candidates:
                break
            next_point = candidates[0]
            edges.remove((current, next_point))
            loop.append(next_point)
            current = next_point
        if current == start and len(loop) >= 4:
            loops.append(loop[:-1])

    if loops:
        def area(loop: list[tuple[float, float]]) -> float:
            return abs(sum(
                loop[index][0] * loop[(index + 1) % len(loop)][1]
                - loop[(index + 1) % len(loop)][0] * loop[index][1]
                for index in range(len(loop))
            ) / 2.0)

        return tuple(max(loops, key=area))

    min_ix = min(cell[0] for cell in cell_tuple)
    max_ix = max(cell[0] for cell in cell_tuple)
    min_iz = min(cell[1] for cell in cell_tuple)
    max_iz = max(cell[1] for cell in cell_tuple)
    min_x = min_ix * resolution - half
    max_x = max_ix * resolution + half
    min_z = min_iz * resolution - half
    max_z = max_iz * resolution + half
    return ((min_x, min_z), (max_x, min_z), (max_x, max_z), (min_x, max_z))


def _centroid_from_cells(cells: Iterable[_GridCell], resolution: float) -> Vector2:
    centers = tuple(_cell_center(cell, resolution) for cell in cells)
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def _point_in_boundary(point_xy: Vector2, boundary_xy: BoundaryXY) -> bool:
    if not boundary_xy:
        return False
    min_x = min(point[0] for point in boundary_xy)
    max_x = max(point[0] for point in boundary_xy)
    min_y = min(point[1] for point in boundary_xy)
    max_y = max(point[1] for point in boundary_xy)
    return min_x <= point_xy[0] <= max_x and min_y <= point_xy[1] <= max_y


def _room_uid(scene_id: str, region_id: str, component_index: int, component_count: int) -> str:
    base = f"prior_room:{_safe_uid(scene_id)}:region_{_safe_uid(region_id)}"
    if component_count <= 1:
        return base
    return f"{base}:component_{component_index}"


def _aliases_for_label(label: str) -> tuple[str, ...]:
    normalized = _norm(label)
    aliases = {normalized, normalized.replace(" ", "_")}
    return tuple(sorted(alias for alias in aliases if alias and alias != label))


def _xz(point_xyz: Vector3) -> Vector2:
    return (point_xyz[0], point_xyz[2])


def _distance_xy(left: Vector2, right: Vector2) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _distance_xyz(left: Vector3, right: Vector3) -> float:
    return math.sqrt(
        (left[0] - right[0]) * (left[0] - right[0])
        + (left[1] - right[1]) * (left[1] - right[1])
        + (left[2] - right[2]) * (left[2] - right[2])
    )


def _room_centroid_xyz(room: PriorRoom) -> Vector3:
    centroid = room.centroid_xy or (0.0, 0.0)
    return (centroid[0], 0.0, centroid[1])


def _geodesic_distance(pathfinder: Any, start_xyz: Vector3, end_xyz: Vector3) -> Optional[float]:
    if not hasattr(pathfinder, "geodesic_distance"):
        return None
    try:
        value = pathfinder.geodesic_distance(start_xyz, end_xyz)
    except TypeError:
        try:
            value = pathfinder.geodesic_distance(list(start_xyz), list(end_xyz))
        except Exception:
            return None
    except Exception:
        return None
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(distance) and distance >= 0.0:
        return distance
    return None


def _vector3_or_default(value: Any, default: Vector3) -> Vector3:
    try:
        return _vector3(value)
    except ValueError:
        return default


def _vector3(value: Any) -> Vector3:
    if value is None:
        raise ValueError("expected vector3, got None")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError("expected vector3")
    return values


def _valid_region_geometry(center: Vector3, sizes: Vector3) -> bool:
    """Return whether a semantic region has finite horizontal geometry."""

    return (
        all(math.isfinite(float(value)) for value in center)
        and all(math.isfinite(float(value)) for value in sizes)
        and abs(float(sizes[0])) > 1e-6
        and abs(float(sizes[2])) > 1e-6
    )


def _valid_object_geometry(center: Vector3, sizes: Vector3) -> bool:
    """Return whether an object has finite center and non-zero extent."""

    return (
        all(math.isfinite(float(value)) for value in center)
        and all(math.isfinite(float(value)) for value in sizes)
        and any(abs(float(value)) > 1e-6 for value in sizes)
    )


def _safe_uid(value: Any) -> str:
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
    "HM3DGroundTruthBuildConfig",
    "HM3DGroundTruthPriorMapBuildResult",
    "HM3DGroundTruthPriorMapBuilder",
    "HM3DSceneAssets",
    "build_hm3d_groundtruth_prior_map_from_scene_dir",
    "build_hm3d_groundtruth_prior_map_from_sim",
    "open_hm3d_simulator",
    "extract_hm3d_mesh_object_bounds",
    "write_hm3d_groundtruth_prior_map_with_alignment",
]
