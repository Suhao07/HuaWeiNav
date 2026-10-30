"""Import official Stray Scanner RGB-D exports into the common capture format.

Verified against strayrobots/scanner ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb.
OdometryEncoder exports q_WC = q_WA * q_AC: optical axes are already applied.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .observations import COORDINATES, SCHEMA_VERSION, write_frame, write_json

# ARKit world is Y-up. Rotate world axes only; exported camera axes are optical.
ARKIT_TO_MAP = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
SCANNER_SOURCE_REVISION = 'ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb'


def stray_camera_to_map(position, quaternion_xyzw):
    """Convert exported camera-to-ARKit pose to right-handed Z-up map pose.

    T_map_camera = T_map_arkit T_arkit_camera; the scanner already multiplied
    its native AR camera orientation by diag(1,-1,-1) before writing the CSV.
    """
    pose = np.eye(4)
    pose[:3, :3] = ARKIT_TO_MAP @ Rotation.from_quat(quaternion_xyzw).as_matrix()
    pose[:3, 3] = ARKIT_TO_MAP @ np.asarray(position)
    return pose


def import_stray_capture(source, output, *, every=1):
    """Decode a scan into aligned RGB and metric depth at native depth resolution.

    Reads headered legacy pose CSVs with session intrinsics, or modern per-frame
    fx/fy/cx/cy. RGB is downsampled to depth size; K_depth = diag(sx,sy,1) K_rgb,
    matching the upstream integration convention. Confidence is copied without
    modifying raw depth. Distortion LUTs require a rectifier before this importer.
    Raises explicit errors for missing frames, mismatched video counts or LUTs.
    This validates format conversion, not physical calibration/relocalization.
    """
    source, output = Path(source), Path(output)
    if every < 1:
        raise ValueError('Frame interval must be positive')
    if any((source / 'distortion').glob('*.bin')):
        raise NotImplementedError('This scan has distortion LUTs; rectify RGB-D before pinhole import')
    with (source / 'odometry.csv').open(newline='') as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fields = set(reader.fieldnames or ())
        required = {'timestamp', 'frame', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'}
        if not required <= fields:
            raise ValueError('Expected headered Stray Scanner pose CSV with timestamp/frame and xyzw pose')
        rows = list(reader)
    calibration_fields = {'fx', 'fy', 'cx', 'cy'}
    if fields & calibration_fields and not calibration_fields <= fields:
        raise ValueError('Per-frame intrinsics require all fx, fy, cx, cy columns')
    per_frame = calibration_fields <= fields
    session_k = None if per_frame else np.loadtxt(source / 'camera_matrix.csv', delimiter=',')
    video = cv2.VideoCapture(str(source / 'rgb.mp4'))
    if not video.isOpened():
        raise ValueError(f'Cannot decode {source / "rgb.mp4"}')
    video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'confidence').mkdir()
    manifest = {'schema_version': SCHEMA_VERSION, 'coordinates': COORDINATES,
        'scene_id': source.name, 'mode': 'handheld_scan', 'pose_source': 'stray_scanner_arkit',
        'semantic_gt_used': False, 'geometry_assisted': False, 'status': 'running',
        'source_format': 'stray_scanner', 'verified_source_revision': SCANNER_SOURCE_REVISION,
        'intrinsics_source': 'per_frame_rgb_resolution' if per_frame else 'session_final_frame_rgb_resolution',
        'rectification_status': 'no_distortion_lut_provided', 'physical_geometry_verified': False,
        'config': {'every': every, 'rgb_resampling': 'area_to_depth_resolution'},
        'odometry_sha256': hashlib.sha256((source / 'odometry.csv').read_bytes()).hexdigest()}
    write_json(output / 'manifest.json', manifest)
    count = 0
    confidence_count = 0
    try:
        if not rows:
            raise ValueError('Scan contains no odometry frames')
        for index, row in enumerate(rows):
            frame_id = int(row['frame'])
            if frame_id != index:
                raise ValueError(f'Scanner saved-frame index {frame_id} differs from video index {index}')
            ok, bgr = video.read()
            if not ok:
                raise ValueError(f'Video ended before odometry frame {frame_id}')
            if index % every:
                continue
            depth_path = source / 'depth' / f'{frame_id:06d}.png'
            depth_mm = np.asarray(Image.open(depth_path))
            if depth_mm.ndim != 2 or depth_mm.dtype not in (np.uint16, np.int32):
                raise ValueError(f'Expected 16-bit millimetre depth PNG: {depth_path}')
            depth = depth_mm.astype(np.float32) / 1000.
            confidence_path = source / 'confidence' / depth_path.name
            confidence_ref = None
            if confidence_path.is_file():
                confidence = np.asarray(Image.open(confidence_path))
                if confidence.shape != depth.shape or not np.isin(confidence, [0, 1, 2]).all():
                    raise ValueError(f'Invalid confidence map: {confidence_path}')
                confidence_ref = f'confidence/{frame_id:06d}.png'
                shutil.copyfile(confidence_path, output / confidence_ref)
                confidence_count += 1
            if per_frame:
                k = np.array([[float(row['fx']), 0, float(row['cx'])],
                              [0, float(row['fy']), float(row['cy'])], [0, 0, 1]])
            else:
                k = session_k.copy()
            scale = np.diag([depth.shape[1] / bgr.shape[1], depth.shape[0] / bgr.shape[0], 1])
            rgb = cv2.cvtColor(cv2.resize(bgr, (depth.shape[1], depth.shape[0]),
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            pose = stray_camera_to_map([float(row[n]) for n in ('x', 'y', 'z')],
                                      [float(row[n]) for n in ('qx', 'qy', 'qz', 'qw')])
            write_frame(output, frame_id, rgb, depth, scale @ k, pose,
                        timestamp=float(row['timestamp']), confidence_ref=confidence_ref,
                        confidence_levels={'low': 0, 'medium': 1, 'high': 2},
                        original_rgb_size_hw=list(bgr.shape[:2]), source_frame_id=frame_id)
            count += 1
        if video.read()[0]:
            raise ValueError('Video contains frames without odometry records')
        manifest.update(status='complete', frame_count=count, source_frame_count=len(rows),
                        confidence_frame_count=confidence_count,
                        coverage_requirement_met=None)
        write_json(output / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', frame_count=count)
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        video.release()
