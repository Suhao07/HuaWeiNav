#!/usr/bin/env python3
"""Collect RGB-D along an actually executed, bounded Habitat route."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.habitat.walk_capture import collect_walk

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'scene', 'navmesh', 'build_receipt', 'output'):
        p.add_argument(name)
    p.add_argument('--max-actions', type=int, default=160)
    p.add_argument('--max-positions', type=int, default=3)
    p.add_argument('--capture-profile', help='Frozen baseline sensor/start/action settings for matched comparison')
    a = vars(p.parse_args())
    result = collect_walk(**a)
    print(json.dumps({k: result[k] for k in ('status', 'frame_count', 'action_count', 'actual_path_length_m', 'termination')}))
