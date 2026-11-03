#!/usr/bin/env python3
"""Run the shared final-verification binding on saved current arrival evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instruction.contracts import TaskSpec
from prior_map.arrival_verification import verify_arrival


def main():
    """Load the explicit task and export verification without issuing task STOP."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('current_bundle', 'compiled_query', 'motion', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--required-stop-distance', type=float, required=True)
    parser.add_argument('--vlm', default='ark')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--candidate-uid')
    parser.add_argument('--observation-id')
    args = vars(parser.parse_args())
    query = args.pop('compiled_query')
    args['task'] = TaskSpec.from_dict(json.loads(query.read_text())['task_spec'])
    report = verify_arrival(**args)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
