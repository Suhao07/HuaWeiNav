#!/usr/bin/env python3
"""Package the final actual motion frame for shared RGB-D perception replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.observations import COORDINATES, SCHEMA_VERSION, write_frame, write_json


def main():
    """Preserve current frame identity and bind it to the motion receipt hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'reference_bundle', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    receipt_bytes = (args.motion / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt['status'] != 'complete' or receipt['motion_status'] != 'reached':
        raise ValueError('Arrival extraction requires a completed motion receipt')
    frame = json.loads((args.motion / 'frames.jsonl').read_text().splitlines()[-1])
    capture = json.loads((args.reference_bundle / 'build_receipt.json').read_text())['capture_manifest']
    args.output.mkdir(parents=True, exist_ok=False)
    write_frame(args.output, frame['frame_id'], np.asarray(Image.open(args.motion / frame['rgb_path'])),
        np.load(args.motion / frame['depth_path']), np.asarray(frame['intrinsics']),
        np.asarray(frame['camera_to_map']), agent_position_habitat=frame['agent_position_habitat'])
    manifest = {key: capture[key] for key in ('scene_id', 'scene_path', 'scene_sha256',
                                            'navmesh_sha256', 'config')}
    manifest.update(schema_version=SCHEMA_VERSION, coordinates=COORDINATES, status='complete',
        mode='historical_viewpoint_arrival', pose_source='simulator_gt', semantic_gt_used=False,
        frame_count=1, source_motion_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        source_frame_id=frame['frame_id'])
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
