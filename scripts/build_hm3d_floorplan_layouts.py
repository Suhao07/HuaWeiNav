#!/usr/bin/env python3
"""Batch-build room-only FloorPlan-compatible layouts for HM3D scenes."""

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
)


def parse_args() -> argparse.Namespace:
    """Parse batch layout-builder arguments."""

    parser = argparse.ArgumentParser(description="Batch-build HM3D semantic floorplan bundles.")
    parser.add_argument("scenes_root", help="Directory containing HM3D scene directories.")
    parser.add_argument("--output_root", required=True, help="Directory for generated layouts.")
    parser.add_argument("--scene_id", action="append", default=[], help="Optional scene suffix filter; repeatable.")
    parser.add_argument("--topdown_resolution", type=float, default=0.25)
    parser.add_argument("--floor_height_tolerance", type=float, default=1.0)
    parser.add_argument("--min_room_area_m2", type=float, default=0.25)
    parser.add_argument("--mask_dilation_radius_m", type=float, default=0.35)
    parser.add_argument("--max_grid_cells", type=int, default=2_000_000)
    return parser.parse_args()


def _scene_id(scene_dir: Path) -> str:
    """Infer the HM3D scene id from an annotated directory name."""

    return scene_dir.name.split("-", maxsplit=1)[-1]


def main() -> int:
    """Build all selected scene layouts and write a reproducible manifest."""

    args = parse_args()
    root = Path(args.scenes_root)
    output_root = Path(args.output_root)
    requested = set(str(item) for item in args.scene_id)
    scene_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (not requested or _scene_id(path) in requested)
    )
    config = HM3DGroundTruthBuildConfig(
        topdown_resolution=args.topdown_resolution,
        floor_height_tolerance=args.floor_height_tolerance,
        min_room_area_m2=args.min_room_area_m2,
        mask_dilation_radius_m=args.mask_dilation_radius_m,
        include_object_priors=False,
        max_grid_cells=args.max_grid_cells,
    )
    manifest = {
        "source_root": str(root),
        "output_root": str(output_root),
        "builder": "hm3d_floorplan_layout",
        "object_priors": False,
        "scenes": [],
    }
    for scene_dir in scene_dirs:
        scene_id = _scene_id(scene_dir)
        output_dir = output_root / scene_id
        result = build_hm3d_floorplan_layout_from_scene_dir(scene_dir, scene_id=scene_id, config=config)
        paths = write_hm3d_floorplan_layout(
            result,
            output_dir / "floorplan.json",
            quality_output_path=output_dir / "quality.json",
        )
        manifest["scenes"].append({
            "scene_id": scene_id,
            "scene_dir": str(scene_dir),
            "paths": paths,
            "rooms": sum(len(level.regions) for level in result.layout.levels),
            "metadata": result.metadata,
        })
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "scene_count": len(scene_dirs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
