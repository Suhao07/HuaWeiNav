#!/usr/bin/env python3
"""Measure cross-view depth consistency for a saved capture."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.geometry_diagnostic import diagnose_capture

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--tolerance-m", type=float, default=.05)
    a = p.parse_args()
    result = diagnose_capture(a.capture, stride=a.stride, tolerance_m=a.tolerance_m)
    print(json.dumps({"output": str(Path(a.capture) / "geometry_diagnostic.json"),
                      "frames": result["frame_count"], "pairs": len(result["pairs"])}))
