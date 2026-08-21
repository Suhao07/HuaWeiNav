#!/usr/bin/env python3
"""Run a ROS2 in-process smoke test for the SysNav viewpoint bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tare_planner.msg import ObjectNode, ObjectNodeList, ViewpointPose, ViewpointRep

from strive_sysnav_bringup.viewpoint_bridge_node import SysNavViewpointBridgeNode


class FakeSysNavPublisher(Node):
    """Publish a minimal timestamp-aligned SysNav event sequence."""

    def __init__(self) -> None:
        """Create fake SysNav publishers and a bridge-output subscriber."""

        super().__init__("fake_sysnav_viewpoint_publisher")
        self.odom_publisher = self.create_publisher(Odometry, "/state_estimation", 10)
        self.viewpoint_publisher = self.create_publisher(ViewpointRep, "/viewpoint_rep_header", 10)
        self.object_publisher = self.create_publisher(ObjectNodeList, "/object_nodes_list", 10)
        self.create_subscription(ViewpointPose, "/strive/sysnav/viewpoint_pose", self._on_pose, 10)
        self.latest_pose: ViewpointPose | None = None
        self.publish_count = 0
        self.timer = self.create_timer(0.05, self._publish_inputs)

    def _publish_inputs(self) -> None:
        """Publish one coherent odometry/viewpoint/object sample."""

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.pose.pose.position.x = 1.25
        odom.pose.pose.position.y = -0.75
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.w = 1.0

        viewpoint = ViewpointRep()
        viewpoint.header.stamp = stamp
        viewpoint.header.frame_id = "map"
        viewpoint.viewpoint_id = 7

        object_node = ObjectNode()
        object_node.header.stamp = stamp
        object_node.header.frame_id = "map"
        object_node.object_id = [42]
        object_node.label = "book"
        object_node.position.x = 2.0
        object_node.position.y = -0.5
        object_node.position.z = 0.0
        object_node.status = True
        object_node.viewpoint_id = 7

        object_list = ObjectNodeList()
        object_list.header.stamp = stamp
        object_list.header.frame_id = "map"
        object_list.nodes = [object_node]

        # 中文核心约束：三类输入共享同一个 header.stamp，验证的是实际 ROS
        # topic 的时间关联，不是用测试代码替代 bridge 的 join 逻辑。
        self.odom_publisher.publish(odom)
        self.viewpoint_publisher.publish(viewpoint)
        self.object_publisher.publish(object_list)
        self.publish_count += 1

    def _on_pose(self, msg: ViewpointPose) -> None:
        """Store the latest bridge output for the smoke assertion."""

        self.latest_pose = msg


def _build_parser() -> argparse.ArgumentParser:
    """Build the ROS2 smoke-test argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/strive_sysnav_viewpoint_ros_smoke.json"),
    )
    return parser


def main() -> int:
    """Start the bridge and validate one real ROS2 message round trip."""

    args = _build_parser().parse_args()
    rclpy.init()
    bridge = SysNavViewpointBridgeNode()
    publisher = FakeSysNavPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(bridge)
    executor.add_node(publisher)
    deadline = time.monotonic() + args.timeout_s
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            message = publisher.latest_pose
            if message is None or list(message.observed_object_ids) != [42]:
                continue
            if message.viewpoint_id != 7:
                raise AssertionError(f"unexpected viewpoint_id={message.viewpoint_id}")
            if message.header.frame_id != "map":
                raise AssertionError(f"unexpected frame_id={message.header.frame_id!r}")
            if abs(message.pose.position.x - 1.25) > 1e-6:
                raise AssertionError(f"unexpected x={message.pose.position.x}")
            if abs(message.pose.position.y + 0.75) > 1e-6:
                raise AssertionError(f"unexpected y={message.pose.position.y}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "viewpoint_id": message.viewpoint_id,
                        "frame_id": message.header.frame_id,
                        "stamp": {
                            "sec": message.header.stamp.sec,
                            "nanosec": message.header.stamp.nanosec,
                        },
                        "position": [
                            message.pose.position.x,
                            message.pose.position.y,
                            message.pose.position.z,
                        ],
                        "observed_object_ids": list(message.observed_object_ids),
                        "published_input_cycles": publisher.publish_count,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"SysNav viewpoint ROS smoke passed: {args.output}")
            return 0
        raise RuntimeError("timed out waiting for /strive/sysnav/viewpoint_pose")
    finally:
        executor.remove_node(bridge)
        executor.remove_node(publisher)
        bridge.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
