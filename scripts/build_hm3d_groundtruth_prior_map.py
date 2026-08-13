#!/usr/bin/env python3
"""Build a STRIVE geometric prior map from HM3D Habitat ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prior_map.hm3d_groundtruth import (  # noqa: E402
    HM3DGroundTruthBuildConfig,
    build_hm3d_groundtruth_prior_map_from_scene_dir,
    write_hm3d_groundtruth_prior_map_with_alignment,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Convert an HM3D scene directory into a STRIVE PriorMapData JSON "
            "using Habitat semantic_scene regions/objects and navmesh geometry."
        ),
    )
    parser.add_argument("scene_dir", help="HM3D scene directory containing basis/semantic/navmesh assets.")
    parser.add_argument("--output", required=True, help="Output canonical prior-map JSON path.")
    parser.add_argument("--alignment_output", default="", help="Optional identity alignment JSON output path.")
    parser.add_argument("--scene_id", default="", help="Scene id override. Defaults to scene directory suffix.")
    parser.add_argument("--topdown_resolution", type=float, default=0.25, help="Top-down grid resolution in meters.")
    parser.add_argument(
        "--floor_height_tolerance",
        type=float,
        default=1.0,
        help="Vertical tolerance for floor-aware sampling in meters.",
    )
    parser.add_argument("--min_room_area_m2", type=float, default=0.25, help="Minimum room component area to keep.")
    parser.add_argument(
        "--mask_dilation_radius_m",
        type=float,
        default=0.35,
        help="Mask dilation radius used for room-room contact detection.",
    )
    parser.add_argument(
        "--merge_disconnected_components",
        default=False,
        action="store_true",
        help="Keep disconnected navigable components under one semantic region instead of splitting them.",
    )
    parser.add_argument(
        "--include_structural",
        default=False,
        action="store_true",
        help="Keep structural semantic labels such as wall/floor/window.",
    )
    parser.add_argument("--max_grid_cells", type=int, default=2_000_000, help="Safety cap for grid sampling.")
    parser.add_argument(
        "--layout_only",
        action="store_true",
        help="Emit a room-only canonical map; static semantic objects are omitted.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the HM3D ground-truth prior-map builder.

    Returns:
        Process exit code.
    """

    args = parse_args()
    config = HM3DGroundTruthBuildConfig(
        topdown_resolution=args.topdown_resolution,
        floor_height_tolerance=args.floor_height_tolerance,
        min_room_area_m2=args.min_room_area_m2,
        mask_dilation_radius_m=args.mask_dilation_radius_m,
        split_disconnected_components=not args.merge_disconnected_components,
        include_structural=args.include_structural,
        include_object_priors=not args.layout_only,
        max_grid_cells=args.max_grid_cells,
    )
    result = build_hm3d_groundtruth_prior_map_from_scene_dir(
        args.scene_dir,
        scene_id=args.scene_id,
        config=config,
    )
    paths = write_hm3d_groundtruth_prior_map_with_alignment(
        result,
        args.output,
        alignment_output_path=args.alignment_output,
    )
    label_counts = result.prior_map.metadata.get("label_counts", {})
    top_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    summary = {
        "paths": paths,
        "scene_id": result.prior_map.scene_id,
        "source_format": result.prior_map.source_format,
        "frame_id": result.prior_map.frame_id,
        "room_count": len(result.prior_map.rooms),
        "object_count": len(result.prior_map.objects),
        "topology_edge_count": len(result.prior_map.topology_edges),
        "top_labels": top_labels,
        "alignment": result.alignment.to_dict(),
        "no_episode_goal_positions": result.prior_map.metadata.get("no_episode_goal_positions", False),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
