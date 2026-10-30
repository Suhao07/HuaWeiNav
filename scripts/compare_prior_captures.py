#!/usr/bin/env python3
"""Check paired capture controls and report sampled surface coverage."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.capture_comparison import compare_captures
from prior_map.observations import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("output")
    args = parser.parse_args()
    report = compare_captures(args.first, args.second)
    write_json(args.output, report)
    print(report["controls"])
