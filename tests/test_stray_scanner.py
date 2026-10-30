"""Camera convention, video synchronization and calibration contracts."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from prior_map.observations import read_records
from prior_map.stray_scanner import import_stray_capture, stray_camera_to_map


def make_scan(root, *, modern=True, video_frames=2):
    root.mkdir()
    (root / 'depth').mkdir()
    (root / 'confidence').mkdir()
    writer = cv2.VideoWriter(str(root / 'rgb.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(video_frames):
        writer.write(np.full((48, 64, 3), [0, 0, 240], dtype=np.uint8))
    writer.release()
    header = 'timestamp, frame, x, y, z, qx, qy, qz, qw'
    if modern:
        header += ', fx, fy, cx, cy'
    lines = [header]
    for index in range(2):
        line = f'{1.25+index*.05}, {index}, 1, 2, 3, 1, 0, 0, 0'
        if modern:
            line += f', {100+index*20}, 100, 32, 24'
        lines.append(line)
        Image.fromarray(np.full((12, 16), 1234, dtype=np.uint16)).save(root / 'depth' / f'{index:06d}.png')
        Image.fromarray(np.full((12, 16), 2, dtype=np.uint8)).save(root / 'confidence' / f'{index:06d}.png')
    (root / 'odometry.csv').write_text('\n'.join(lines) + '\n')
    np.savetxt(root / 'camera_matrix.csv', [[80, 0, 32], [0, 80, 24], [0, 0, 1]], delimiter=',')


def test_exported_optical_pose_is_not_flipped_twice():
    pose = stray_camera_to_map([1, 2, 3], [1, 0, 0, 0])
    np.testing.assert_allclose(pose[:3, 3], [1, -3, 2])
    np.testing.assert_allclose(pose[:3, :3] @ [0, 0, 1], [0, 1, 0])
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def test_video_decode_per_frame_intrinsics_and_depth_units(tmp_path):
    source, output = tmp_path / 'scan', tmp_path / 'capture'
    make_scan(source)
    result = import_stray_capture(source, output)
    frames = read_records(output / 'frames.jsonl')
    assert result['frame_count'] == 2
    assert result['physical_geometry_verified'] is False
    assert [f['intrinsics'][0][0] for f in frames] == [25, 30]
    np.testing.assert_allclose(np.load(output / frames[0]['depth_path']), 1.234)
    rgb = np.asarray(Image.open(output / frames[0]['rgb_path']))
    assert rgb[..., 0].mean() > 230 and rgb[..., 2].mean() < 5
    assert frames[1]['timestamp'] == 1.3
    assert (output / frames[0]['confidence_ref']).read_bytes() == (source / 'confidence/000000.png').read_bytes()


def test_legacy_intrinsics_and_sampling_preserve_frame_identity(tmp_path):
    source, output = tmp_path / 'scan', tmp_path / 'capture'
    make_scan(source, modern=False)
    result = import_stray_capture(source, output, every=2)
    frames = read_records(output / 'frames.jsonl')
    assert result['source_frame_count'] == 2 and result['frame_count'] == 1
    assert result['intrinsics_source'] == 'session_final_frame_rgb_resolution'
    assert frames[0]['intrinsics'][0][0] == 20


def test_missing_video_frame_records_failure(tmp_path):
    source, output = tmp_path / 'scan', tmp_path / 'capture'
    make_scan(source, video_frames=1)
    with pytest.raises(ValueError, match='Video ended before odometry frame 1'):
        import_stray_capture(source, output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'


def test_distortion_lut_requires_rectification(tmp_path):
    source = tmp_path / 'scan'
    make_scan(source)
    (source / 'distortion').mkdir()
    (source / 'distortion/000000.bin').write_bytes(np.zeros(16, dtype='<f4').tobytes())
    with pytest.raises(NotImplementedError, match='rectify'):
        import_stray_capture(source, tmp_path / 'capture')


def test_missing_confidence_is_explicit_without_inventing_quality(tmp_path):
    source, output = tmp_path / 'scan', tmp_path / 'capture'
    make_scan(source)
    (source / 'confidence/000001.png').unlink()
    result = import_stray_capture(source, output)
    assert result['confidence_frame_count'] == 1
    frames = read_records(output / 'frames.jsonl')
    assert frames[1]['confidence_ref'] is None
    np.testing.assert_allclose(np.load(output / frames[1]['depth_path']), 1.234)
