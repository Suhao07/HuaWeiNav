#!/usr/bin/env python3
"""Package the final actual motion frame for shared RGB-D perception replay."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.arrival import extract_arrival_capture


def main():
    """Preserve current frame identity and bind it to the motion receipt hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'reference_bundle', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    extract_arrival_capture(args.motion, args.reference_bundle, args.output)


if __name__ == '__main__':
    main()
