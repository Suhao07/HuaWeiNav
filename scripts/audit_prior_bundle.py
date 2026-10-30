#!/usr/bin/env python3
"""Report review candidates from the original evidence of a predicted map."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.evidence_audit import audit_bundle
from prior_map.observations import write_json


def main():
    """Write a reproducible evidence audit without changing the map."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_bundle(args.bundle)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "objects"}, indent=2))


if __name__ == "__main__":
    main()
