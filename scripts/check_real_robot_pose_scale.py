#!/usr/bin/env python3
"""Read-only odometry scale check for semantic-map coordinate acceptance."""

from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


class PoseScaleCheck:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.samples: list[dict[str, float]] = []

    def callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        values = (float(p.x), float(p.y), float(p.z), float(v.x), float(v.y), float(v.z))
        if not all(math.isfinite(x) for x in values):
            self.samples.append({"finite": 0.0})
            return
        self.samples.append(
            {
                "finite": 1.0,
                "max_abs_position_m": max(abs(p.x), abs(p.y), abs(p.z)),
                "position_norm_m": math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z),
                "linear_speed_mps": math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z),
            }
        )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/aft_mapped_to_init")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-abs-m", type=float, default=200.0)
    parser.add_argument("--max-speed-mps", type=float, default=5.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rclpy.init()
    checker = PoseScaleCheck(args.topic)
    node = rclpy.create_node("real_robot_pose_scale_check")
    node.create_subscription(Odometry, args.topic, checker.callback, qos_profile_sensor_data)
    deadline = time.monotonic() + max(args.duration, 0.1)
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    finite = [s for s in checker.samples if s.get("finite") == 1.0]
    max_abs = max((s["max_abs_position_m"] for s in finite), default=None)
    max_speed = max((s["linear_speed_mps"] for s in finite), default=None)
    result = {
        "topic": args.topic,
        "duration_s": args.duration,
        "samples": len(checker.samples),
        "finite_samples": len(finite),
        "max_abs_position_m": max_abs,
        "position_norm_p95_m": percentile([s["position_norm_m"] for s in finite], 0.95),
        "max_linear_speed_mps": max_speed,
        "pass": bool(
            finite
            and max_abs is not None
            and max_speed is not None
            and max_abs <= args.max_abs_m
            and max_speed <= args.max_speed_mps
        ),
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(output + "\n")


if __name__ == "__main__":
    main()
