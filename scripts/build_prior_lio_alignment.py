#!/usr/bin/env python3
"""Build a prior-map -> Point-LIO alignment from the robot start-marker pose.

The robot base pose at the start marker is measured once in the prior-map
plane. Point-LIO ``camera_init`` is the gravity-aligned lidar pose at LIO start
(mapping session) or the saved-map frame (location mode on that map), so the
alignment is valid only while LIO starts at the marker or localizes on the map
recorded with ``--lio-map``.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.alignment import PriorMapAlignment


def _floats(text, count, name):
    values = [float(item) for item in str(text).split(",")]
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{name} expects {count} comma-separated numbers")
    return values


def lidar_pose_in_prior(base_xy, base_yaw, base_in_lidar):
    """Compose ``T_prior_base * inv(T_lidar_base)`` in the plane.

    Args:
        base_xy: Base position in the prior frame.
        base_yaw: Base forward-axis heading in the prior frame.
        base_in_lidar: ``(x, y, yaw)`` of base in the lidar frame, i.e. the
            ``aft_mapped -> base`` static transform.

    Returns:
        Lidar ``((x, y), yaw)`` in the prior frame.
    """

    tx, ty, phi = base_in_lidar
    # inv(T_lidar_base)：base 系下雷达的位置 = -R(-phi) t。
    lx = -(math.cos(phi) * tx + math.sin(phi) * ty)
    ly = -(-math.sin(phi) * tx + math.cos(phi) * ty)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    return (base_xy[0] + c * lx - s * ly, base_xy[1] + s * lx + c * ly), base_yaw - phi


def build_alignment(args):
    """Return the alignment and receipt payload for parsed CLI arguments."""

    bundle = Path(args.bundle)
    prior_bytes = (bundle / "prior_map.json").read_bytes()
    prior = json.loads(prior_bytes)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    base_xy = _floats(args.base_xy, 2, "--base-xy")
    if args.heading_xy is not None:
        hx, hy = _floats(args.heading_xy, 2, "--heading-xy")
        if math.hypot(hx - base_xy[0], hy - base_xy[1]) < 0.3:
            raise SystemExit("--heading-xy must be at least 0.3 m in front of --base-xy")
        base_yaw = math.atan2(hy - base_xy[1], hx - base_xy[0])
    else:
        base_yaw = math.radians(args.base_yaw_deg)
    base_in_lidar = _floats(args.base_in_lidar, 3, "--base-in-lidar")
    lidar_xy, lidar_yaw = lidar_pose_in_prior(base_xy, base_yaw, base_in_lidar)

    anchor = {"type": "start_marker",
              "valid_when": "LIO (mapping mode) is started with the robot base on the marker"}
    if args.lio_map:
        lio_meta = json.loads(Path(args.lio_map).read_text(encoding="utf-8"))
        pcd = Path(args.lio_map).parent / lio_meta["pcd"]
        sha = hashlib.sha256(pcd.read_bytes()).hexdigest()
        if sha != lio_meta["sha256"]:
            raise SystemExit(f"LIO map sha256 mismatch: {pcd}")
        anchor = {"type": "lio_map", "lio_map_sha256": sha, "lio_map_pcd": str(pcd.resolve()),
                  "valid_when": "LIO runs with LIO_LOCATION_MODE=1 on this PCD; the map was "
                                "recorded from a session started on the marker"}

    alignment = PriorMapAlignment.from_runtime_origin_pose(
        origin_xy_prior=lidar_xy,
        origin_yaw_rad=lidar_yaw,
        prior_frame_id=prior["frame_id"],
        runtime_frame_id=args.runtime_frame,
        confidence=args.confidence,
        diagnostics={
            "method_detail": "start_marker_base_pose",
            "prior_scene_id": prior["scene_id"],
            "prior_map_id": manifest.get("map_id"),
            "prior_map_revision": manifest.get("map_revision"),
            "prior_map_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "base_xy_prior": base_xy,
            "base_yaw_rad_prior": base_yaw,
            "base_in_lidar_xy_yaw": base_in_lidar,
            "anchor": anchor,
        },
    )
    # 自检：标记点处 base 在 runtime 系中应恰为静态 TF 的平移。
    check = alignment.prior_to_runtime(base_xy)
    return alignment, {"base_marker_in_runtime": [check[0], check[1]],
                       "expected": base_in_lidar[:2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", help="Prior bundle directory (prior_map.json + manifest.json)")
    parser.add_argument("--base-xy", required=True, help="Base position on the marker, prior frame, 'x,y' m")
    heading = parser.add_mutually_exclusive_group(required=True)
    heading.add_argument("--base-yaw-deg", type=float, help="Base forward heading in the prior frame")
    heading.add_argument("--heading-xy", help="A prior-frame point straight ahead of the base, 'x,y' m")
    parser.add_argument("--base-in-lidar", default="-0.2,0.0,-1.5708",
                        help="aft_mapped->base static TF 'x,y,yaw_rad' (start_orin_lio_for_strive.sh)")
    parser.add_argument("--lio-map", help="lio_map.json written by save_orin_lio_map.sh")
    parser.add_argument("--runtime-frame", default="camera_init")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    alignment, check = build_alignment(args)
    alignment.save(args.output)
    print(json.dumps({"output": args.output, "rotation_rad": alignment.rotation_rad,
                      "translation_xyz": list(alignment.translation_xyz), **check}, indent=2))


if __name__ == "__main__":
    main()
