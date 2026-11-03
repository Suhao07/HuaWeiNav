"""Package current arrival evidence without changing simulator or task state."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .observations import COORDINATES, SCHEMA_VERSION, write_frame, write_json, load_depth


def extract_arrival_capture(motion, reference_bundle, output):
    """Copy the terminal observed frame and bind it to the reached-motion receipt.

    Args:
        motion: Saved execution directory containing terminal receipt and frames.
        reference_bundle: Historical map supplying scene and sensor provenance.
        output: New single-frame capture directory.
    Returns:
        Capture manifest binding original frame ID, pose and motion hash.
    Raises:
        ValueError: The motion has not completed with a reached terminal state.
    """
    motion, reference_bundle, output = map(Path, (motion, reference_bundle, output))
    receipt_bytes = (motion / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt['status'] != 'complete' or receipt['motion_status'] != 'reached':
        raise ValueError('Arrival extraction requires a completed motion receipt')
    frame = json.loads((motion / 'frames.jsonl').read_text().splitlines()[-1])
    capture = json.loads((reference_bundle / 'build_receipt.json').read_text())['capture_manifest']
    output.mkdir(parents=True, exist_ok=False)
    write_frame(output, frame['frame_id'], np.asarray(Image.open(motion / frame['rgb_path'])),
        load_depth(motion / frame['depth_path']), np.asarray(frame['intrinsics']),
        np.asarray(frame['camera_to_map']), agent_position_habitat=frame['agent_position_habitat'])
    manifest = {key: capture[key] for key in ('scene_id', 'scene_path', 'scene_sha256',
                                            'navmesh_sha256', 'config')}
    manifest.update(schema_version=SCHEMA_VERSION, coordinates=COORDINATES, status='complete',
        mode='historical_viewpoint_arrival', pose_source='simulator_gt', semantic_gt_used=False,
        frame_count=1, source_motion_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        source_frame_id=frame['frame_id'])
    write_json(output / 'manifest.json', manifest)
    return manifest
