#!/usr/bin/env python3
"""Build a NavMesh with explicit dimensions and a scene-bound receipt."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.navmesh_build import build_navmesh


def main():
    """Require explicit robot dimensions rather than inheriting guessed settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene')
    parser.add_argument('output')
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--height', type=float, required=True)
    parser.add_argument('--max-climb', type=float, required=True)
    parser.add_argument('--max-slope', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(build_navmesh(args.scene, args.output, radius=args.radius,
        height=args.height, max_climb=args.max_climb, max_slope=args.max_slope), indent=2))


if __name__ == '__main__':
    main()
