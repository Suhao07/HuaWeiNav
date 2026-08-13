"""ROS2 probe for replaying sensor bags through the migrated SysNav planner.

The probe owns only a test ``/way_point`` publisher and observes ``/path``.
It deliberately does not start or publish any velocity controller output.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


class LowerBagProbe(Node):
    """Publish one waypoint and verify that localPlanner produces a path.

    The probe distinguishes a valid multi-pose path from the single zero pose
    that SysNav emits when it has no active path or clears a previous goal.
    Sensor receipt counts are recorded so a successful result cannot be
    confused with a planner that ran without bag input.
    """

    def __init__(self) -> None:
        """Create the waypoint publisher and lower-layer subscriptions."""

        super().__init__("strive_lower_bag_probe")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("pointcloud_topic", "/cloud_registered")
        self.declare_parameter("planner_status_topic", "/local_planner/status")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("goal_x", 2.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_z", 0.0)
        self.declare_parameter("goal_delay_s", 0.5)
        self.declare_parameter("timeout_s", 30.0)
        self.declare_parameter("artifact_path", "/tmp/strive_lower_bag_probe.json")

        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.goal = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
            float(self.get_parameter("goal_z").value),
        )
        self.goal_delay_s = max(0.0, float(self.get_parameter("goal_delay_s").value))
        self.timeout_s = max(0.1, float(self.get_parameter("timeout_s").value))
        self.artifact_path = Path(str(self.get_parameter("artifact_path").value))
        self.started_at = time.monotonic()
        self.sensor_started_at: Optional[float] = None
        self.goal_published_at: Optional[float] = None
        self.path_received_at: Optional[float] = None
        self.odom_messages = 0
        self.pointcloud_messages = 0
        self.path_messages = 0
        self.valid_path_messages = 0
        self.max_path_poses = 0
        self.last_status = ""
        self.valid_path = False
        self.finished = False
        self.exit_code = 2

        queue_size = 10
        self._waypoint_pub = self.create_publisher(PointStamped, self.waypoint_topic, queue_size)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            queue_size,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._on_pointcloud,
            queue_size,
        )
        self.create_subscription(
            NavPath,
            str(self.get_parameter("path_topic").value),
            self._on_path,
            queue_size,
        )
        planner_status_topic = str(self.get_parameter("planner_status_topic").value or "")
        if planner_status_topic:
            self.create_subscription(String, planner_status_topic, self._on_status, queue_size)
        self.create_timer(0.05, self._tick)

    def _on_odom(self, _: Odometry) -> None:
        """Record one localization sample from the replayed bag."""

        self.odom_messages += 1
        self.sensor_started_at = self.sensor_started_at or time.monotonic()

    def _on_pointcloud(self, _: PointCloud2) -> None:
        """Record one registered point-cloud sample from the replayed bag."""

        self.pointcloud_messages += 1
        self.sensor_started_at = self.sensor_started_at or time.monotonic()

    def _on_path(self, message: NavPath) -> None:
        """Record whether localPlanner emitted a nontrivial path."""

        self.path_messages += 1
        self.max_path_poses = max(self.max_path_poses, len(message.poses))
        has_motion = len(message.poses) > 1 and any(
            math.hypot(float(pose.pose.position.x), float(pose.pose.position.y)) > 0.1
            for pose in message.poses
        )
        if has_motion:
            self.valid_path = True
            self.valid_path_messages += 1
            self.path_received_at = self.path_received_at or time.monotonic()

    def _on_status(self, message: String) -> None:
        """Record the latest planner state for the replay artifact."""

        self.last_status = str(message.data)

    def _tick(self) -> None:
        """Publish the test goal after replay sensors have become available."""

        now = time.monotonic()
        if self.finished:
            return
        if self.sensor_started_at is not None and self.goal_published_at is None:
            if now - self.sensor_started_at >= self.goal_delay_s:
                self._publish_goal()
        if self.valid_path:
            self.exit_code = 0
            self._finish("valid_path_received")
        elif now - self.started_at >= self.timeout_s:
            self.exit_code = 1
            self._finish("timeout_waiting_for_valid_path")

    def _publish_goal(self) -> None:
        """Publish one map-frame waypoint to the migrated localPlanner."""

        message = PointStamped()
        message.header.frame_id = self.world_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.point.x, message.point.y, message.point.z = self.goal
        self._waypoint_pub.publish(message)
        self.goal_published_at = time.monotonic()

    def _finish(self, reason: str) -> None:
        """Persist replay evidence and stop the probe node."""

        self.finished = True
        payload = {
            "success": self.exit_code == 0,
            "reason": reason,
            "waypoint_topic": self.waypoint_topic,
            "world_frame": self.world_frame,
            "goal": list(self.goal),
            "odom_messages": self.odom_messages,
            "pointcloud_messages": self.pointcloud_messages,
            "path_messages": self.path_messages,
            "valid_path_messages": self.valid_path_messages,
            "max_path_poses": self.max_path_poses,
            "planner_status": self.last_status,
            "elapsed_s": time.monotonic() - self.started_at,
        }
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.get_logger().info(json.dumps(payload, sort_keys=True))


def main(args=None) -> None:
    """Run the lower-stack bag probe and return its acceptance status."""

    rclpy.init(args=args)
    node = LowerBagProbe()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        raise SystemExit(node.exit_code)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
