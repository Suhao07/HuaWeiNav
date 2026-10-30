"""Portable RGB-D records: metres, Z-up map, OpenCV optical camera axes."""
from pathlib import Path
import json
import gzip
import numpy as np
from PIL import Image

SCHEMA_VERSION = 1
COORDINATES = {
    "unit": "m", "handedness": "right", "up_axis": "z",
    "camera_axes": "x_right_y_down_z_forward", "pose": "camera_to_map",
    "matrix_storage": "row_major", "vector_convention": "column",
}


def write_json(path, value):
    """Write JSON with finite numeric values; propagate serialization failures."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_records(path):
    """Read the ordered JSONL records of a capture or map index."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_depth(path):
    """Read metric NPY depth, optionally wrapped in lossless gzip.

    Both encodings preserve the original dtype, shape and optical-Z values.
    Invalid files propagate their decoding error to the caller.
    """
    path = Path(path)
    if path.name.endswith(".npy.gz"):
        with gzip.open(path, "rb") as stream:
            return np.load(stream, allow_pickle=False)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError(f"Unsupported depth encoding: {path}")


def write_frame(root, frame_id, rgb, depth, intrinsic, camera_to_map, **metadata):
    """Persist an aligned RGB-D frame before any clipping or visualization.

    Depth is float32 optical-axis distance in metres; zero denotes invalid.
    The caller owns synchronization, pose convention and RGB/depth alignment.
    """
    root = Path(root)
    rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=np.float32)
    if rgb.shape[:2] != depth.shape or rgb.dtype != np.uint8:
        raise ValueError("Expected aligned uint8 RGB and HxW metric depth")
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError(f"Invalid depth values in frame {frame_id}")
    pose = np.asarray(camera_to_map, dtype=float)
    if pose.shape != (4, 4) or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("camera_to_map must be a homogeneous 4x4 matrix")
    rot = pose[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rot), 1, atol=1e-5):
        raise ValueError("camera_to_map rotation must be right-handed and orthonormal")
    for folder in ("rgb", "depth"):
        (root / folder).mkdir(exist_ok=True)
    rgb_path, depth_path = f"rgb/{frame_id:06d}.png", f"depth/{frame_id:06d}.npy"
    Image.fromarray(rgb[..., :3]).save(root / rgb_path)
    np.save(root / depth_path, depth, allow_pickle=False)
    record = {"frame_id": frame_id, "rgb_path": rgb_path, "depth_path": depth_path,
              "intrinsics": np.asarray(intrinsic).tolist(), "camera_to_map": pose.tolist(),
              "size_hw": list(depth.shape), "depth_unit": "m", "depth_convention": "optical_z",
              **metadata}
    with (root / "frames.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    return record


def backproject(depth, intrinsic, camera_to_map, stride=1):
    """Project valid pixels: p_map = T_map_camera [z K^-1 (u,v,1), 1]."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0)
    pixels = np.stack([u[valid], v[valid], np.ones(valid.sum())])
    camera = (np.linalg.inv(intrinsic) @ pixels) * z[valid]
    return (camera_to_map[:3, :3] @ camera).T + camera_to_map[:3, 3]


def metric_depth_validity(depth, capture_profile=None):
    """Mark depths that represent measured surfaces rather than clipped bounds.

    For D=clip(Z, z_min, z_max), metric support requires z_min < D < z_max.
    Raw captures without a clipping profile retain their positive depth values.
    """
    valid = depth > 0
    if capture_profile is not None:
        valid &= (depth > capture_profile["depth_min_m"]) & (depth < capture_profile["depth_max_m"])
    return valid
