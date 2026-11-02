#!/usr/bin/env python3
"""Build unknown navigation regions from a portable RGB-D capture, without NavMesh."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.capture_builder import capture_free_cells
from prior_map.hm3d_groundtruth import HM3DGroundTruthBuildConfig
from prior_map.hm3d_layout import build_free_cell_region_layout, write_hm3d_floorplan_layout


def main():
    """Write the same region-layout bundle as the NavMesh geometry-only mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--output", required=True, help="FloorPlan-compatible JSON output path.")
    parser.add_argument("--quality_output", default="")
    parser.add_argument("--robot_height", type=float, required=True)
    parser.add_argument("--robot_radius", type=float, required=True)
    parser.add_argument("--region_clearance", type=float, default=0.35)
    parser.add_argument("--topdown_resolution", type=float, default=0.1)
    parser.add_argument("--floor_height_tolerance", type=float, default=1.0)
    parser.add_argument("--min_room_area_m2", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_depth_m", type=float, default=6.0)
    parser.add_argument("--min_depth_confidence", type=int, choices=(0, 1, 2),
                        help="Sensor confidence required for floor and obstacle points")
    args = parser.parse_args()
    config = HM3DGroundTruthBuildConfig(topdown_resolution=args.topdown_resolution,
        floor_height_tolerance=args.floor_height_tolerance, min_room_area_m2=args.min_room_area_m2)
    cells, floor_height, diagnostics = capture_free_cells(args.capture,
        resolution=args.topdown_resolution, robot_height_m=args.robot_height,
        robot_radius_m=args.robot_radius, stride=args.stride, max_depth_m=args.max_depth_m,
        min_depth_confidence=args.min_depth_confidence)
    scene_id = json.loads((Path(args.capture) / "manifest.json").read_text())["scene_id"]
    result = build_free_cell_region_layout(cells, scene_id, floor_height=floor_height,
        clearance_m=args.region_clearance, config=config, source="rgbd_free_space",
        metadata=diagnostics)
    paths = write_hm3d_floorplan_layout(result, args.output, quality_output_path=args.quality_output)
    print(json.dumps({"paths": paths, **result.metadata}, indent=2))


if __name__ == "__main__":
    main()
