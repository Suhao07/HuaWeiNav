#!/usr/bin/env python3
"""Bind a current RGB-D candidate to STRIVE's existing better-view policy."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.prior_approach import resolve_current_approach
from prior_map.observations import write_json


def main():
    """Export current candidate proposals without issuing motion or acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'motion', 'navmesh', 'build_receipt', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--candidate-uid', required=True)
    parser.add_argument('--observation-id', help='Raw observation ID when the candidate has several masks')
    parser.add_argument('--required-stop-distance', type=float, required=True)
    parser.add_argument('--arrival-tolerance', type=float, default=.2)
    args = vars(parser.parse_args())
    output = args.pop('output')
    result = resolve_current_approach(**args)
    write_json(output, result)
    print(f"{len(result['proposals'])} reachable proposals; selected={result['selected'] is not None}")


if __name__ == '__main__':
    main()
