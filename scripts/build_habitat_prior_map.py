#!/usr/bin/env python3
"""Build a STRIVE prior map from a Habitat ObjectNav dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prior_map.habitat_objectnav import (
    build_habitat_objectnav_prior_map,
    write_prior_map_with_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Habitat ObjectNav JSON/JSON.GZ goals into a STRIVE PriorMapData JSON file.",
    )
    parser.add_argument("dataset_path", help="Habitat ObjectNav dataset JSON or JSON.GZ path.")
    parser.add_argument("--output", required=True, help="Output canonical prior-map JSON path.")
    parser.add_argument("--scene_id", default="", help="Scene id or scene id substring to select.")
    parser.add_argument("--object_category", default="", help="Object category to select, such as tv_monitor.")
    parser.add_argument("--episode_rank", type=int, default=0, help="Rank among matching episodes.")
    parser.add_argument(
        "--include_scene_categories",
        default=False,
        action="store_true",
        help="Include all goal categories for the selected scene instead of only the episode target category.",
    )
    parser.add_argument("--alignment_output", default="", help="Optional alignment JSON output path.")
    parser.add_argument(
        "--alignment_mode",
        choices=("unavailable", "identity"),
        default="unavailable",
        help="Alignment file mode. Use unavailable unless prior/world and runtime frames are calibrated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_habitat_objectnav_prior_map(
        args.dataset_path,
        scene_id=args.scene_id,
        object_category=args.object_category,
        episode_rank=args.episode_rank,
        include_scene_categories=args.include_scene_categories,
    )
    paths = write_prior_map_with_alignment(
        result,
        args.output,
        alignment_output_path=args.alignment_output,
        alignment_mode=args.alignment_mode,
    )
    summary = {
        "paths": paths,
        "scene_id": result.prior_map.scene_id,
        "selected_episode_id": str(result.selected_episode.get("episode_id", "")),
        "selected_object_category": result.prior_map.metadata.get("selected_object_category", ""),
        "goal_count": result.goal_count,
        "source_format": result.prior_map.source_format,
        "frame_id": result.prior_map.frame_id,
        "loader_source": "canonical_json",
        "recommended_prior_map_alignment": (
            str(Path(args.alignment_output)) if args.alignment_output else args.alignment_mode
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
