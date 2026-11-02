"""Arrival evidence retains the terminal execution frame and receipt binding."""
import hashlib
import json
import numpy as np
import pytest
from prior_map.arrival import extract_arrival_capture
from prior_map.observations import write_frame, read_records, load_depth


def test_arrival_preserves_last_frame_pose_and_motion_identity(tmp_path):
    motion, reference, output = [tmp_path / name for name in ('motion', 'reference', 'arrival')]
    motion.mkdir()
    reference.mkdir()
    receipt = {'status': 'complete', 'motion_status': 'reached'}
    (motion / 'receipt.json').write_text(json.dumps(receipt))
    capture = dict(scene_id='scene', scene_path='/scene.glb', scene_sha256='scene-sha',
                   navmesh_sha256='mesh-sha', config={'width': 3, 'height': 2})
    (reference / 'build_receipt.json').write_text(json.dumps({'capture_manifest': capture}))
    pose = np.eye(4)
    for frame in (0, 35):
        pose[:3, 3] = [frame, 0, 1]
        write_frame(motion, frame, np.full((2, 3, 3), frame, dtype=np.uint8),
                    np.full((2, 3), 1.234567, dtype=np.float32), np.eye(3), pose,
                    agent_position_habitat=[frame, 0, 0])
    manifest = extract_arrival_capture(motion, reference, output)
    row, = read_records(output / 'frames.jsonl')
    assert row['frame_id'] == manifest['source_frame_id'] == 35
    assert row['camera_to_map'] == pose.tolist()
    assert manifest['source_motion_receipt_sha256'] == hashlib.sha256((motion/'receipt.json').read_bytes()).hexdigest()
    assert load_depth(output/row['depth_path']).tobytes() == load_depth(motion/row['depth_path']).tobytes()


@pytest.mark.parametrize('terminal', ['blocked', 'timeout', 'cancelled'])
def test_arrival_rejects_unreached_motion(tmp_path, terminal):
    motion = tmp_path / 'motion'
    motion.mkdir()
    (motion/'receipt.json').write_text(json.dumps({'status':'complete','motion_status':terminal}))
    with pytest.raises(ValueError, match='completed motion receipt'):
        extract_arrival_capture(motion, tmp_path/'unused', tmp_path/'output')
