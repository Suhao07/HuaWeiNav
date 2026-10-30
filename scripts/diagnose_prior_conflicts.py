#!/usr/bin/env python3
"""Localize non-edge projection conflicts without removing failed observations."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.geometry_diagnostic import diagnose_interior_conflicts
from prior_map.observations import write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnose_interior_conflicts(args.capture, Path(args.capture) / 'geometry_diagnostic.json')
    write_json(args.output, result)
    print(f"Saved {len(result['conflicts'])} original non-edge conflicts")
