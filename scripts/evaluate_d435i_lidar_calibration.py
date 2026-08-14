#!/usr/bin/env python3
"""Offline, read-only D435i/MID-360 calibration acceptance evaluator.

The evaluator deliberately separates three quantities:
* checkerboard corner PnP RMSE (camera intrinsic/image quality);
* LiDAR plane/board residuals (extrinsic geometry);
* RGB--LiDAR temporal offset, estimated only from a dynamic bag by comparing
  LiDAR points transformed with the recorded odometry against aligned RGB-D.

It never opens a ROS publisher and never writes to the source project.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def corner_rmse(image_paths, k, d, pattern=(7, 6), square=0.10):
    object_points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square
    values = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, None)
        if not ok:
            values.append({"image": str(path), "detected": False})
            continue
        ok_pnp, rvec, tvec = cv2.solvePnP(object_points, corners, k, d)
        if not ok_pnp:
            values.append({"image": str(path), "detected": True, "pnp": False})
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, d)
        err = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        values.append({
            "image": str(path),
            "detected": True,
            "pnp": True,
            "rmse_px": float(np.sqrt(np.mean(err**2))),
            "max_px": float(np.max(err)),
            "corner_count": int(len(err)),
        })
    valid = [v["rmse_px"] for v in values if "rmse_px" in v]
    return {
        "pattern_inner_corners": list(pattern),
        "square_size_m": square,
        "samples": values,
        "sample_count": len(valid),
        "rmse_px": float(np.sqrt(np.mean(np.square(valid)))) if valid else None,
        "median_rmse_px": float(np.median(valid)) if valid else None,
    }


def load_candidate(path):
    data = json.loads(Path(path).read_text())
    t = np.asarray(data["T_camera_from_lidar"], dtype=float)
    return t[:3, :3], t[:3, 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration-result", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--fy", type=float, required=True)
    p.add_argument("--cx", type=float, required=True)
    p.add_argument("--cy", type=float, required=True)
    args = p.parse_args()
    k = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=float)
    d = np.zeros(5, dtype=float)
    result = {
        "evaluator": "evaluate_d435i_lidar_calibration.py",
        "calibration_result": str(Path(args.calibration_result)),
        "candidate_extrinsic": "T_camera_from_lidar",
        "checkerboard_corner_reprojection": corner_rmse(
            sorted(Path(args.image_dir).glob("*-rgb.png")), k, d
        ),
        "note": (
            "Corner RMSE validates camera PnP/intrinsics; it is not a one-to-one "
            "LiDAR-to-pixel RMSE. LiDAR geometric and temporal metrics are evaluated "
            "by the companion rosbag evaluator."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
