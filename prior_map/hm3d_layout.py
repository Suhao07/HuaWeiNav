"""Build semantic FloorPlan-compatible layouts from HM3D assets.

The builder may use Habitat NavMesh internally to recover room geometry, but
NavMesh rasters are deliberately not part of the serialized prior-map
contract. The exported bundle contains only the structured semantic layout
and its BEV rendering, which can be reproduced in simulation and on a robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .floorplan_layout import FloorplanLayout
from .hm3d_groundtruth import (
    HM3DGroundTruthBuildConfig,
    HM3DGroundTruthPriorMapBuilder,
    extract_hm3d_mesh_object_bounds,
    open_hm3d_simulator,
)


@dataclass(frozen=True)
class HM3DLayoutBuildResult:
    """Result of converting an HM3D scene to a room-only floorplan layout.

    Args:
        layout: FloorPlan-compatible metric room layout.
        prior_map: Canonical room-only VLN map used to create the layout.
        metadata: JSON-friendly build diagnostics.
    """

    layout: FloorplanLayout
    prior_map: Any
    metadata: dict[str, Any]


class HM3DLayoutBuilder:
    """Construct a FloorPlan-compatible layout from HM3D semantic scene data."""

    @staticmethod
    def _layout_config(config: Optional[HM3DGroundTruthBuildConfig]) -> HM3DGroundTruthBuildConfig:
        """Normalize a general HM3D config into the room-only build contract."""

        base = config or HM3DGroundTruthBuildConfig()
        return HM3DGroundTruthBuildConfig(
            topdown_resolution=base.topdown_resolution,
            floor_height_tolerance=base.floor_height_tolerance,
            min_room_area_m2=base.min_room_area_m2,
            mask_dilation_radius_m=base.mask_dilation_radius_m,
            split_disconnected_components=base.split_disconnected_components,
            include_structural=False,
            include_object_priors=False,
            use_mesh_region_fallback=base.use_mesh_region_fallback,
            max_grid_cells=base.max_grid_cells,
            source="hm3d_floorplan_layout",
        )

    def build_from_sim(
        self,
        sim: Any,
        scene_id: str,
        config: Optional[HM3DGroundTruthBuildConfig] = None,
        *,
        mesh_object_bounds: Optional[dict[str, dict[str, Any]]] = None,
    ) -> HM3DLayoutBuildResult:
        """Build a room-only layout from an existing Habitat-like simulator.

        Args:
            sim: Simulator exposing ``semantic_scene`` and ``pathfinder``.
            scene_id: Stable scene identifier.
            config: Optional geometric configuration.
            mesh_object_bounds: Optional semantic-mesh geometry used to repair
                invalid HM3D region AABBs.

        Returns:
            Layout build result with object priors omitted.
        """

        layout_config = self._layout_config(config)
        result = HM3DGroundTruthPriorMapBuilder().build_from_sim(
            sim,
            scene_id,
            config=layout_config,
            mesh_object_bounds=mesh_object_bounds,
        )
        layout = FloorplanLayout.from_prior_map(result.prior_map)
        metadata = {
            **dict(result.metadata),
            "authority": "semantic_region_geometry",
            "layout_format": "floorplan_vln_compatible",
            "object_instances_omitted": True,
            "coordinate_convention": "habitat_xz_to_floorplan_x_neg_z",
            "semantic_bev_required": True,
            "quality": _layout_quality(result.prior_map),
        }
        layout = FloorplanLayout(
            scene_id=layout.scene_id,
            frame_id=layout.frame_id,
            levels=layout.levels,
            metadata=metadata,
        )
        return HM3DLayoutBuildResult(
            layout=layout,
            prior_map=result.prior_map,
            metadata=metadata,
        )

    def build_from_scene_dir(
        self,
        scene_dir: str | Path,
        *,
        scene_id: str = "",
        config: Optional[HM3DGroundTruthBuildConfig] = None,
    ) -> HM3DLayoutBuildResult:
        """Load an HM3D scene and build its room-only layout.

        Args:
            scene_dir: Directory containing HM3D basis, semantic, and NavMesh assets.
            scene_id: Optional stable scene id override.
            config: Optional geometric build configuration. Object priors are
                disabled by this adapter regardless of the supplied default.

        Returns:
            FloorPlan-compatible layout and its source canonical map.

        Raises:
            RuntimeError: If Habitat-Sim cannot load the scene assets.
            FileNotFoundError: If required HM3D assets are absent.
        """

        with open_hm3d_simulator(scene_dir) as (simulator, _assets):
            mesh_bounds = extract_hm3d_mesh_object_bounds(
                _assets.semantic_glb,
                _assets.semantic_txt,
            )
            return self.build_from_sim(
                simulator,
                scene_id or Path(scene_dir).name.split("-", maxsplit=1)[-1],
                config=config,
                mesh_object_bounds=mesh_bounds,
            )


def build_hm3d_floorplan_layout_from_scene_dir(
    scene_dir: str | Path,
    *,
    scene_id: str = "",
    config: Optional[HM3DGroundTruthBuildConfig] = None,
) -> HM3DLayoutBuildResult:
    """Build a room-only FloorPlan-compatible layout from an HM3D directory."""

    return HM3DLayoutBuilder().build_from_scene_dir(scene_dir, scene_id=scene_id, config=config)


def build_hm3d_floorplan_layout_from_sim(
    sim: Any,
    scene_id: str,
    *,
    config: Optional[HM3DGroundTruthBuildConfig] = None,
) -> HM3DLayoutBuildResult:
    """Build a room-only FloorPlan-compatible layout from a simulator."""

    return HM3DLayoutBuilder().build_from_sim(sim, scene_id, config=config)


def write_hm3d_floorplan_layout(
    result: HM3DLayoutBuildResult,
    output_path: str | Path,
    *,
    quality_output_path: str | Path = "",
) -> dict[str, str]:
    """Write the semantic prior-map bundle.

    Args:
        result: Layout build result.
        output_path: Destination FloorPlan-compatible JSON path.
        quality_output_path: Optional diagnostics JSON path.

    Returns:
        Mapping of artifact names to written paths. The bundle always includes
        the structured layout and a semantic BEV image/SVG; NavMesh ``.npy``
        files are never written.
    """

    output = result.layout.save(output_path)
    paths = {"floorplan": str(output)}
    output_dir = output.parent
    # 中文：BEV 是仿真与实物共用的语义先验接口，必须和 JSON 同源生成。
    # 这里渲染房间边界、拓扑和标签，不把 Habitat 的 NavMesh 真值泄漏到输出。
    from .visualizer import PriorMapFloorPlanVisualizer

    visualizer = PriorMapFloorPlanVisualizer()
    bev_paths = visualizer.write_global_artifacts(
        result.prior_map,
        output_dir,
        stem="prior_map_bev",
    )
    paths.update(
        {
            "prior_map_bev_png": bev_paths["png"],
            "prior_map_bev_svg": bev_paths["svg"],
            "prior_map_bev_markers": bev_paths["markers"],
        }
    )
    if quality_output_path:
        quality = Path(quality_output_path)
        quality.parent.mkdir(parents=True, exist_ok=True)
        quality.write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["quality"] = str(quality)
    manifest_path = output_dir / "prior_map_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scene_id": result.layout.scene_id,
                "source_format": result.metadata.get("source_format", "hm3d_floorplan_layout"),
                "frame_id": result.layout.frame_id,
                "coordinate_convention": result.metadata.get("coordinate_convention"),
                "authority": "semantic_prior_only",
                "artifact_policy": "semantic_bundle_only",
                "object_instances_omitted": True,
                "artifacts": {
                    "floorplan": output.name,
                    "prior_map_bev_png": Path(bev_paths["png"]).name,
                    "prior_map_bev_svg": Path(bev_paths["svg"]).name,
                    "prior_map_bev_markers": Path(bev_paths["markers"]).name,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["manifest"] = str(manifest_path)
    return paths


def _layout_quality(prior_map: Any) -> dict[str, Any]:
    """Summarize semantic-layout coverage without NavMesh rasters.

    The report is intentionally descriptive. It records whether the generated
    layout contains usable room geometry and topology, but it does not assign
    semantic correctness to an HM3D region label.
    """

    rooms = tuple(getattr(prior_map, "rooms", ()) or ())
    rooms_with_boundary = sum(1 for room in rooms if len(getattr(room, "boundary_xy", ()) or ()) >= 3)
    room_count = len(rooms)
    return {
        "room_count": room_count,
        "rooms_with_boundary": rooms_with_boundary,
        "room_boundary_coverage": (rooms_with_boundary / room_count) if room_count else 0.0,
        "room_topology_edge_count": sum(
            1 for edge in (getattr(prior_map, "topology_edges", ()) or ())
            if getattr(edge, "edge_type", "") == "room-room"
        ),
        "semantic_bev_required": True,
        "object_instances_omitted": True,
    }


__all__ = [
    "HM3DLayoutBuildResult",
    "HM3DLayoutBuilder",
    "build_hm3d_floorplan_layout_from_scene_dir",
    "build_hm3d_floorplan_layout_from_sim",
    "write_hm3d_floorplan_layout",
]
