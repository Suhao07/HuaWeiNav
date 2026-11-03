#!/usr/bin/env python3
"""Check historical simulated observation poses against a current NavMesh start."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.prior_viewpoints import resolve_habitat_viewpoints
from prior_map.observations import write_json


def main():
    """Export candidate facts without sending motion commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('navmesh')
    parser.add_argument('--start', nargs=3, type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--build-receipt')
    parser.add_argument('--object-uid', help='Historical instance ID returned by map query')
    args = parser.parse_args()
    result = resolve_habitat_viewpoints(args.bundle, args.navmesh, args.start,
                                       build_receipt=args.build_receipt, object_uid=args.object_uid)
    write_json(args.output, result)
    print(f"{len(result['viewpoints'])} reachable candidates; {len(result['rejected'])} rejected")


if __name__ == '__main__':
    main()
