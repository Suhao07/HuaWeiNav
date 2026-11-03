"""Import official Stray Scanner RGB-D exports into the common capture format.

Verified against strayrobots/scanner ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb.
OdometryEncoder exports q_WC = q_WA * q_AC: optical axes are already applied.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

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


def upright_quarter_turns(poses):
    r"""Session clockwise quarter turns that bring world-up to the image top.

    相机列为地图系下的光学轴，图像上方为 :math:`-y_c`。世界竖直 :math:`e_z` 投影到像平面后
    相对图像上方的角度及会话转数（取众数，整段采集共用同一图像尺寸）：

    .. math::
        \phi_t=\operatorname{atan2}(R_{t,zx},\,-R_{t,zy}),\qquad
        k=\operatorname{mode}_t\big[\operatorname{round}(-\phi_t/90^\circ)\bmod 4\big].
    """
    rotations = np.asarray(poses)[:, :3, :3]
    roll = np.degrees(np.arctan2(rotations[:, 2, 0], -rotations[:, 2, 1]))
    return int(np.bincount(np.round(-roll / 90).astype(int) % 4, minlength=4).argmax())


def turn_clockwise(images, intrinsics, pose, turns):
    r"""Rotate images by clockwise quarter turns while keeping projection exact.

    单次顺时针 :math:`(u',v')=(H-1-v,u)`，新相机轴 :math:`x'=-y,\;y'=x,\;z'=z`：

    .. math::
        K'=\begin{bmatrix}f_y&0&H-1-c_y\\0&f_x&c_x\\0&0&1\end{bmatrix},\qquad
        R'=R\begin{bmatrix}0&1&0\\-1&0&0\\0&0&1\end{bmatrix}.
    """
    pose = pose.copy()
    for _ in range(turns):
        height = images[0].shape[0]
        images = [np.ascontiguousarray(np.rot90(image, k=-1)) for image in images]
        intrinsics = np.array([[intrinsics[1, 1], 0, height - 1 - intrinsics[1, 2]],
                               [0, intrinsics[0, 0], intrinsics[0, 2]], [0, 0, 1]])
        pose[:3, :3] = pose[:3, :3] @ np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
    return images, intrinsics, pose


def import_stray_capture(source, output, *, every=1):
    """Decode a scan into aligned RGB and metric depth at native depth resolution.

    Reads headered legacy pose CSVs with session intrinsics, or modern per-frame
    fx/fy/cx/cy. RGB is downsampled to depth size; K_depth = diag(sx,sy,1) K_rgb,
    matching the upstream integration convention. The session is turned upright
    by quarter turns from gravity; confidence follows the same pixel permutation
    without modifying raw depth values. Distortion LUTs require a rectifier first.
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
    poses = [stray_camera_to_map([float(row[n]) for n in ('x', 'y', 'z')],
                                 [float(row[n]) for n in ('qx', 'qy', 'qz', 'qw')]) for row in rows]
    # 手持竖握时传感器原图侧躺；检测器按正立图像训练，转正只改像素排列，几何精确不变。
    turns = upright_quarter_turns(poses) if poses else 0
    # 上游pts取ARFrame计数且先于自增：首个保存帧pts=-1/60早于startSession(.zero)，
    # 被edit list裁掉，默认解码会让第k帧RGB配上第k+1帧深度/位姿。
    # 忽略edit list按packet显示顺序取回全部保存帧；pts间隙来自编码积压丢帧，不影响顺序对应。
    previous_options = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'ignore_editlist;1'
    try:
        video = cv2.VideoCapture(str(source / 'rgb.mp4'), cv2.CAP_FFMPEG)
    finally:
        if previous_options is None:
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS')
        else:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = previous_options
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
        'config': {'every': every, 'rgb_resampling': 'area_to_depth_resolution',
                   'video_decode': 'ffmpeg_ignore_editlist_packet_order',
                   'upright_quarter_turns_cw': turns},
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
            images = [None, depth]
            if confidence_path.is_file():
                confidence = np.asarray(Image.open(confidence_path))
                if confidence.shape != depth.shape or not np.isin(confidence, [0, 1, 2]).all():
                    raise ValueError(f'Invalid confidence map: {confidence_path}')
                confidence_ref = f'confidence/{frame_id:06d}.png'
                images.append(confidence)
                confidence_count += 1
            if per_frame:
                k = np.array([[float(row['fx']), 0, float(row['cx'])],
                              [0, float(row['fy']), float(row['cy'])], [0, 0, 1]])
            else:
                k = session_k.copy()
            scale = np.diag([depth.shape[1] / bgr.shape[1], depth.shape[0] / bgr.shape[0], 1])
            images[0] = cv2.cvtColor(cv2.resize(bgr, (depth.shape[1], depth.shape[0]),
                                              interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            images, intrinsics, pose = turn_clockwise(images, scale @ k, poses[index], turns)
            if confidence_ref is not None:
                Image.fromarray(images[2]).save(output / confidence_ref)
            write_frame(output, frame_id, images[0], images[1], intrinsics, pose,
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
