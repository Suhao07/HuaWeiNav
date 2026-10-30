#!/usr/bin/env python3
"""Convert a Stray Scanner export into the shared STRIVE RGB-D capture protocol."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.stray_scanner import import_stray_capture


def main():
    """Preserve original data and write a new portable capture directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    parser.add_argument('--every', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(import_stray_capture(args.source, args.output, every=args.every), indent=2))


if __name__ == '__main__':
    main()
