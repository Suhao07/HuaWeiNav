#!/usr/bin/env python3
"""Build a FloorPlan-VLN-compatible room layout from an HM3D scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prior_map.hm3d_groundtruth import HM3DGroundTruthBuildConfig  # noqa: E402
from prior_map.hm3d_layout import (  # noqa: E402
    build_hm3d_floorplan_layout_from_scene_dir,
    write_hm3d_floorplan_layout,
    build_navigation_region_layout,
    open_hm3d_simulator,
)


def parse_args() -> argparse.Namespace:
    """Parse the HM3D floorplan-layout CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert HM3D semantic scene and NavMesh assets into a room-only "
            "semantic floorplan bundle with JSON/BEV outputs. Requires Habitat-Sim."
        )
    )
    parser.add_argument("scene_dir", help="HM3D scene directory containing basis/semantic/navmesh assets.")
    parser.add_argument("--geometry_only", action="store_true", help="Build unknown navigation regions from NavMesh; no semantic labels.")
    parser.add_argument("--floor_height", type=float, help="Habitat y slice; required for geometry-only mode.")
    parser.add_argument("--region_clearance", type=float, default=0.35, help="Additional seed erosion radius in metres.")
    parser.add_argument("--scene_id", default="", help="Optional scene id override.")
    parser.add_argument("--output", required=True, help="FloorPlan-compatible JSON output path.")
    parser.add_argument("--quality_output", default="", help="Optional build-quality JSON output path.")
    parser.add_argument("--topdown_resolution", type=float, default=0.25)
    parser.add_argument("--floor_height_tolerance", type=float, default=1.0)
    parser.add_argument("--min_room_area_m2", type=float, default=0.25)
    parser.add_argument("--mask_dilation_radius_m", type=float, default=0.35)
    parser.add_argument("--max_grid_cells", type=int, default=2_000_000)
    parser.add_argument(
        "--merge_disconnected_components",
        action="store_true",
        help="Keep disconnected components under a semantic region as one room.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and write one HM3D room-layout prior."""

    args = parse_args()
    config = HM3DGroundTruthBuildConfig(
        topdown_resolution=args.topdown_resolution,
        floor_height_tolerance=args.floor_height_tolerance,
        min_room_area_m2=args.min_room_area_m2,
        mask_dilation_radius_m=args.mask_dilation_radius_m,
        split_disconnected_components=not args.merge_disconnected_components,
        include_object_priors=False,
        max_grid_cells=args.max_grid_cells,
    )
    try:
        if args.geometry_only:
            if args.floor_height is None:
                raise ValueError('--geometry_only requires --floor_height')
            with open_hm3d_simulator(args.scene_dir) as (simulator, _assets):
                result = build_navigation_region_layout(simulator,
                    args.scene_id or Path(args.scene_dir).name.split('-', maxsplit=1)[-1],
                    floor_height=args.floor_height, clearance_m=args.region_clearance, config=config)
        else:
            result = build_hm3d_floorplan_layout_from_scene_dir(
                args.scene_dir, scene_id=args.scene_id, config=config)
        paths = write_hm3d_floorplan_layout(
            result,
            args.output,
            quality_output_path=args.quality_output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[hm3d-layout] error: {exc}", file=sys.stderr)
        print("[hm3d-layout] run this command inside the STRIVE Habitat Docker image", file=sys.stderr)
        return 2
    if not result.layout.levels or not any(level.regions for level in result.layout.levels):
        print(
            "[hm3d-layout] error: no room regions were generated; refusing to write an empty layout",
            file=sys.stderr,
        )
        print(
            "[hm3d-layout] inspect quality metadata and verify semantic mesh/NavMesh compatibility",
            file=sys.stderr,
        )
        return 2
    summary = {
        "paths": paths,
        "scene_id": result.layout.scene_id,
        "frame_id": result.layout.frame_id,
        "levels": len(result.layout.levels),
        "rooms": sum(len(level.regions) for level in result.layout.levels),
        "objects_in_layout": 0,
        "authority": result.metadata.get("authority"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
