#!/usr/bin/env python3
"""Create a self-contained evidence-selected map from a complete source bundle."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.compact_bundle import compact_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--views-per-object", type=int, default=3)
    args = parser.parse_args()
    report = compact_bundle(args.source, args.output, views_per_object=args.views_per_object)
    print(json.dumps({key: report[key] for key in ("object_count", "source_observation_count",
        "observation_count", "source_evidence_bytes", "selected_evidence_bytes")}))
