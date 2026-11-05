"""Render one official IntentionNav start pose into a portable STRIVE capture.

Run with Isaac's python.sh; INTENTIONNAV_ROOT identifies the official environment.
This entry point consumes the start pose only. Target metadata is never exported.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-id", default="SEL_001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    # Isaac's Kit consumes sys.argv as well; only our parser owns these options.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(Path(os.environ["INTENTIONNAV_ROOT"]) / "eval" / "simulator"))
    from iss_env import IsaacSimEnv
    from backends.isaac.capture import capture_frame
    from prior_map.observations import COORDINATES, SCHEMA_VERSION, write_json

    episodes = [json.loads(line) for line in (args.data_root / "dataset/episodes.jsonl").read_text().splitlines()]
    episode = next(item for item in episodes if item["selection_id"] == args.selection_id)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=SCHEMA_VERSION, coordinates=COORDINATES,
                    status="running", source="IntentionNav", selection_id=args.selection_id,
                    scene_id=episode["scene_id"])
    write_json(args.output / "manifest.json", manifest)
    start = time.monotonic()
    env = IsaacSimEnv(gpu=args.gpu)
    try:
        env.load_scene(episode["scene_id"], str(args.data_root / "scenes" / episode["scene_id"] / "start_result_navigation.usd"))
        env.place_agent(episode["start_position"], episode["start_rotation_quat_wxyz"])
        frame = capture_frame(env, args.output, 0)
        manifest.update(status="complete", frame_count=1, elapsed_s=time.monotonic() - start,
                        valid_depth_fraction=frame["valid_depth_fraction"])
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps(manifest), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
