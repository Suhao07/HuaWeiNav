"""Run STRIVE guided navigation against a separately owned Isaac simulator bridge."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9466)
    parser.add_argument("--vlm", default="self_hosted")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-model-calls", type=int, default=60)
    parser.add_argument("--max-moves", type=int, default=30)
    parser.add_argument("--panorama-views", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--decision-owner", choices=("baseline_guided",), default="baseline_guided")
    parser.add_argument("--search-review-moves", type=int, default=10)
    parser.add_argument("--task-spec", type=Path, help="fixed TaskSpec admitted independently of navigation")
    parser.add_argument("--task-spec-sha256", help="expected hash from the independently audited artifact")
    args = parser.parse_args()
    if (args.task_spec is None) != (args.task_spec_sha256 is None):
        parser.error("--task-spec and --task-spec-sha256 must be supplied together")
    from scripts.serve_intentionnav import load_public_episode
    from backends.isaac.runtime import run_episode

    episode = load_public_episode(args.public_episode)
    output = args.output.resolve(strict=True)
    if not output.is_dir():
        parser.error("output must be an existing run directory")
    if (output / "policy_result.json").exists():
        parser.error("output already contains a policy attempt; use a fresh run directory")
    result = run_episode(episode, output, port=args.port, vlm=args.vlm, device=args.device,
                         max_model_calls=args.max_model_calls, max_moves=args.max_moves,
                         panorama_views=args.panorama_views, decision_owner=args.decision_owner,
                         search_review_moves=args.search_review_moves,
                         task_spec_path=args.task_spec, task_spec_sha256=args.task_spec_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
