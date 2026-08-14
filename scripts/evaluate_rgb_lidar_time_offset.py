#!/usr/bin/env python3
"""Estimate RGB--LiDAR temporal offset from an existing ROS 2 bag.

This is a read-only evaluator.  It uses the recorded RGB-aligned depth image,
MID-360 packets, odometry, and the imported rigid transforms.  For a candidate
offset delta = t_rgb - t_lidar, a LiDAR packet is selected near
``t_rgb - delta`` and motion-compensated with odometry before projection into
the depth image.  The score is the robust depth residual at projected points.

The result is an estimate with a scan curve, not an automatic approval.  A
flat curve or too few valid depth correspondences is reported as unidentifiable.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np


def quat_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def pose_matrix(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    out = np.eye(4)
    out[:3, :3] = quat_matrix((q.x, q.y, q.z, q.w))
    out[:3, 3] = [p.x, p.y, p.z]
    return out


def stamp(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    return float(h.stamp.sec) + 1e-9 * float(h.stamp.nanosec)


def interp_pose(times, poses, t):
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return poses[0]
    if i >= len(times):
        return poses[-1]
    a, b = i - 1, i
    u = (t - times[a]) / max(times[b] - times[a], 1e-9)
    out = np.eye(4)
    out[:3, 3] = (1 - u) * poses[a][:3, 3] + u * poses[b][:3, 3]
    # Translation dominates the time-offset residual; nearest rotation avoids
    # introducing a dependency on scipy just for quaternion interpolation.
    out[:3, :3] = poses[a][:3, :3] if u < 0.5 else poses[b][:3, :3]
    return out


def read_bag(path, sample_every=8, points_per_packet=100):
    import rosbag2_py
    from livox_ros_driver2.msg import CustomMsg
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    cameras, lidars, odoms = [], [], []
    camera_i = 0
    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic == "/camera/veocc_d435i/aligned_depth_to_color/image_raw":
            camera_i += 1
            if camera_i % sample_every:
                continue
            msg = deserialize_message(data, Image)
            t = stamp(msg)
            if t is None:
                continue
            dtype = np.uint16 if msg.encoding.lower() in ("16uc1", "mono16") else np.float32
            arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // np.dtype(dtype).itemsize)
            cameras.append((t, arr[:, : msg.width].copy(), msg.encoding))
        elif topic == "/livox/lidar":
            msg = deserialize_message(data, CustomMsg)
            t = stamp(msg)
            if t is None or not msg.points:
                continue
            stride = max(1, len(msg.points) // points_per_packet)
            xyz = np.asarray([(p.x, p.y, p.z) for p in msg.points[::stride]], dtype=np.float64)
            good = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 0.25)
            if np.any(good):
                lidars.append((t, xyz[good]))
        elif topic == "/base_odom":
            msg = deserialize_message(data, Odometry)
            t = stamp(msg)
            if t is not None:
                odoms.append((t, pose_matrix(msg)))
    cameras.sort(key=lambda x: x[0])
    lidars.sort(key=lambda x: x[0])
    odoms.sort(key=lambda x: x[0])
    return cameras, lidars, odoms


def score_offset(cameras, lidars, odom_times, odom_poses, t_cam_from_lidar, t_base_from_lidar, t_base_from_camera, k, offset):
    lidar_times = [x[0] for x in lidars]
    residuals = []
    correspondences = 0
    pairs = 0
    for t_rgb, depth_raw, encoding in cameras:
        target = t_rgb - offset
        j = bisect.bisect_left(lidar_times, target)
        if j >= len(lidars):
            j = len(lidars) - 1
        if j and abs(lidar_times[j - 1] - target) < abs(lidar_times[j] - target):
            j -= 1
        if abs(lidar_times[j] - target) > 0.035:
            continue
        depth = depth_raw.astype(np.float64) * (0.001 if "16" in encoding.lower() else 1.0)
        t_lidar = lidar_times[j]
        world_from_base_l = interp_pose(odom_times, odom_poses, t_lidar)
        world_from_base_c = interp_pose(odom_times, odom_poses, t_rgb)
        cam_from_world = np.linalg.inv(world_from_base_c @ t_base_from_camera)
        world_from_lidar = world_from_base_l @ t_base_from_lidar
        points = lidars[j][1]
        points_h = np.c_[points, np.ones(len(points))]
        camera = (cam_from_world @ world_from_lidar @ points_h.T).T[:, :3]
        valid = camera[:, 2] > 0.15
        camera = camera[valid]
        if len(camera) == 0:
            continue
        pixels = (k @ camera.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        u = np.rint(pixels[:, 0]).astype(int)
        v = np.rint(pixels[:, 1]).astype(int)
        inside = (u >= 1) & (u < depth.shape[1] - 1) & (v >= 1) & (v < depth.shape[0] - 1)
        u, v, z = u[inside], v[inside], camera[inside, 2]
        if len(z) == 0:
            continue
        measured = depth[v, u]
        valid_depth = np.isfinite(measured) & (measured > 0.15) & (measured < 15.0)
        diff = np.abs(z[valid_depth] - measured[valid_depth])
        # Reject obvious foreground/background mismatches while retaining a
        # robust surface alignment statistic.
        diff = diff[diff < 1.5]
        if len(diff):
            residuals.extend(diff.tolist())
            correspondences += len(diff)
            pairs += 1
    if not residuals:
        return {"offset_s": offset, "score_rmse_m": None, "score_median_abs_m": None, "pairs": pairs, "correspondences": 0}
    arr = np.asarray(residuals)
    return {
        "offset_s": float(offset),
        "score_rmse_m": float(np.sqrt(np.mean(arr**2))),
        "score_median_abs_m": float(np.median(arr)),
        "score_p90_abs_m": float(np.percentile(arr, 90)),
        "pairs": pairs,
        "correspondences": int(len(arr)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--extrinsics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--fy", type=float, required=True)
    p.add_argument("--cx", type=float, required=True)
    p.add_argument("--cy", type=float, required=True)
    p.add_argument("--min-offset", type=float, default=-0.20)
    p.add_argument("--max-offset", type=float, default=0.20)
    p.add_argument("--step", type=float, default=0.010)
    args = p.parse_args()
    ext = json.loads(Path(args.extrinsics).read_text())
    t_cam_lidar = np.asarray(ext["T_camera_from_lidar"], dtype=float)
    t_base_lidar = np.asarray(ext["T_base_from_lidar"], dtype=float)
    t_base_camera = np.asarray(ext["T_base_from_camera"], dtype=float)
    cameras, lidars, odoms = read_bag(args.bag)
    odom_times = [x[0] for x in odoms]
    odom_poses = [x[1] for x in odoms]
    k = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=float)
    offsets = np.arange(args.min_offset, args.max_offset + args.step * 0.5, args.step)
    scores = [score_offset(cameras, lidars, odom_times, odom_poses, t_cam_lidar, t_base_lidar, t_base_camera, k, float(x)) for x in offsets]
    usable = [s for s in scores if s["score_median_abs_m"] is not None and s["correspondences"] >= 100]
    usable.sort(key=lambda x: x["score_median_abs_m"])
    best = usable[0] if usable else None
    result = {
        "bag": str(Path(args.bag)),
        "timestamp_basis": "sensor message header stamps; candidate delta=t_rgb-t_lidar",
        "input_counts": {"camera_depth_samples": len(cameras), "lidar_packets": len(lidars), "odom_samples": len(odoms)},
        "scan": scores,
        "best": best,
        "identifiable": bool(best and best["correspondences"] >= 100 and best["score_median_abs_m"] < 0.25),
        "warning": "This is a depth-consistency estimate; approve only after checking the curve, inlier count, and a held-out run.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("input_counts", "best", "identifiable", "warning")}, indent=2))


if __name__ == "__main__":
    main()
