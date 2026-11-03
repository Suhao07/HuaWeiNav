#!/usr/bin/env python3
"""Collect task-independent posed RGB-D using the maintained Habitat image."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.capture import CaptureConfig, collect


def main():
    """Parse explicit scan budgets and run the bounded collector."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_dir")
    p.add_argument("--output", required=True)
    for name, field in CaptureConfig.__dataclass_fields__.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(p.parse_args())
    scene, output = args.pop("scene_dir"), args.pop("output")
    print(json.dumps(collect(scene, output, CaptureConfig(**args)), indent=2))

if __name__ == "__main__":
    main()
