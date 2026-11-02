#!/usr/bin/env python3
"""Replay portable captures through STRIVE detection and shared object fusion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.capture_builder import build_capture_map


def main():
    """Build a bounded map using an explicit generic object vocabulary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("output")
    vocabulary = parser.add_mutually_exclusive_group(required=True)
    vocabulary.add_argument("--classes", nargs="+")
    vocabulary.add_argument("--classes-file", type=Path,
                            help="UTF-8 vocabulary, one complete concept per line")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--voxel-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--deferred-association", action="store_true",
                        help="Opt into experimental observation-owned fusion")
    parser.add_argument("--min-depth-confidence", type=int, choices=(0, 1, 2),
                        help="Fuse only depth whose sensor confidence reaches this level")
    args = parser.parse_args()
    association_config = None
    if args.deferred_association:
        from prior_map.association import AssociationConfig
        association_config = AssociationConfig()
    # 多词类别按行读取，与独立建图器使用同一词表，保留词内空格。
    classes = (args.classes if args.classes_file is None else
               [line.strip() for line in args.classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()])
    result = build_capture_map(args.capture, args.output, classes=classes,
        max_frames=args.max_frames, voxel_m=args.voxel_m, max_depth_m=args.max_depth_m, association_config=association_config,
        min_depth_confidence=args.min_depth_confidence)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "capture_manifest")}, indent=2))


if __name__ == "__main__":
    main()
