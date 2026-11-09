"""Run one continuous GOAT parent on legacy Habitat with explicit protocol assets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    """Parse explicit inputs, run one parent and print the artifact location/status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-shard', type=Path, required=True, help='Absolute path or relative to data-root')
    parser.add_argument('--parent-index', type=int, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--scenes-root', type=Path, required=True)
    parser.add_argument('--scene', type=Path, help='Explicit local asset for the original scene_id')
    parser.add_argument('--scene-dataset', type=Path, help='Explicit local scene dataset configuration')
    parser.add_argument('--official-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vlm', choices=('self_hosted', 'qwen_local'), required=True)
    parser.add_argument('--max-steps-per-subtask', '--max_steps_per_subtask', type=int, default=500,
                        help='All primitive actions including one reserved terminal-event slot')
    parser.add_argument('--global-budget', type=int, default=5000, help='Includes all SUBTASK_STOP/global STOP events')
    parser.add_argument('--max-model-calls', type=int, default=100)
    parser.add_argument('--image-assets', type=Path,
                        help='JSON keyed by opaque image_ref: path, sha256, scene_id, image_parameters (position/rotation/hfov/image_dimensions)')
    args = parser.parse_args(argv)
    if min(args.max_steps_per_subtask, args.global_budget) < 1 or args.max_model_calls < 0:
        parser.error('Action budgets must be positive; model-call budget must be nonnegative')
    for key in ('data_root', 'scenes_root', 'output', 'scene', 'scene_dataset', 'official_config', 'image_assets'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    from backends.habitat.goat_runtime import run_goat
    result = run_goat(args)
    print(json.dumps({'status': result['status'], 'result': str(args.output / 'result.json')}, ensure_ascii=False))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
