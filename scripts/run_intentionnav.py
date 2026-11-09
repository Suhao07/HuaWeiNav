"""Run STRIVE full-instruction navigation against an isolated Isaac simulator bridge."""
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
    parser.add_argument("--panorama-views", type=int, default=4,
                        help="number of panoramic headings (>= 2; default: 4)")
    args = parser.parse_args()
    if min(args.max_model_calls, args.max_moves) < 1:
        parser.error("episode budgets must be positive")
    if args.panorama_views < 2:
        parser.error("--panorama-views must be >= 2")
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
                         panorama_views=args.panorama_views)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
