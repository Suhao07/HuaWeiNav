#!/usr/bin/env python3
"""Build a VLN prior map from HM3D semantic txt annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prior_map.hm3d_semantic import (
    build_hm3d_semantic_prior_map,
    write_hm3d_semantic_prior_map_with_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HM3D *.semantic.txt scene annotations into a STRIVE PriorMapData JSON file.",
    )
    parser.add_argument("semantic_txt_path", help="HM3D scene *.semantic.txt path.")
    parser.add_argument("--output", required=True, help="Output canonical prior-map JSON path.")
    parser.add_argument("--scene_id", default="", help="Scene id override. Defaults to semantic txt basename.")
    parser.add_argument("--include_unknown", default=False, action="store_true", help="Keep unknown labels.")
    parser.add_argument(
        "--include_structural",
        default=False,
        action="store_true",
        help="Keep structural labels such as wall/floor/ceiling.",
    )
    parser.add_argument("--alignment_output", default="", help="Optional alignment JSON output path.")
    parser.add_argument(
        "--alignment_mode",
        choices=("unavailable", "identity"),
        default="unavailable",
        help="Use unavailable unless semantic geometry has been calibrated to the runtime frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_hm3d_semantic_prior_map(
        args.semantic_txt_path,
        scene_id=args.scene_id,
        include_unknown=args.include_unknown,
        include_structural=args.include_structural,
    )
    paths = write_hm3d_semantic_prior_map_with_alignment(
        result,
        args.output,
        alignment_output_path=args.alignment_output,
        alignment_mode=args.alignment_mode,
    )
    label_counts = result.prior_map.metadata.get("label_counts", {})
    top_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    summary = {
        "paths": paths,
        "scene_id": result.prior_map.scene_id,
        "source_format": result.prior_map.source_format,
        "frame_id": result.prior_map.frame_id,
        "loader_source": "canonical_json",
        "instance_count": len(result.prior_map.objects),
        "region_count": len(result.prior_map.rooms),
        "top_labels": top_labels,
        "recommended_prior_map_alignment": (
            str(Path(args.alignment_output)) if args.alignment_output else args.alignment_mode
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
