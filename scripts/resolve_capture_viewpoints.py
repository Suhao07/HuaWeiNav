#!/usr/bin/env python3
"""Place historical handheld camera poses on the capture's robot-traversable floor."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.capture_builder import resolve_capture_viewpoints
from prior_map.observations import write_json


def main():
    """Export candidate facts without sending motion commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("capture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot_height", type=float, required=True)
    parser.add_argument("--robot_radius", type=float, required=True)
    parser.add_argument("--resolution", type=float, default=.1)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_depth_m", type=float, default=6.)
    parser.add_argument("--min_depth_confidence", type=int, choices=(0, 1, 2),
                        help="Use the same threshold as build_capture_layout.py")
    parser.add_argument("--object-uid", help="Historical instance ID returned by map query")
    args = parser.parse_args()
    result = resolve_capture_viewpoints(args.bundle, args.capture, resolution=args.resolution,
                                        robot_height_m=args.robot_height, robot_radius_m=args.robot_radius,
                                        stride=args.stride, max_depth_m=args.max_depth_m,
                                        object_uid=args.object_uid,
                                        min_depth_confidence=args.min_depth_confidence)
    write_json(args.output, result)
    print(f"{len(result['viewpoints'])} candidates on traversable floor; {len(result['rejected'])} rejected")


if __name__ == "__main__":
    main()
