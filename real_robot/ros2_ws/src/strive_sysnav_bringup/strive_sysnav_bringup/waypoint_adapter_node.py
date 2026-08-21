"""ROS2 node for the configurable STRIVE waypoint format adapter."""

from __future__ import annotations

from typing import Any

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from real_robot.contracts import Pose3D
from real_robot.waypoint_adapter import WaypointAdapterConfig, WaypointAdapterError, WaypointFormatAdapter


class StriveWaypointAdapterNode(Node):
    """Read STRIVE waypoints and optionally emit external-controller arrays."""

    def __init__(self) -> None:
        super().__init__("strive_waypoint_adapter")
        self._declare_parameters()
        config = self._load_config()
        self.adapter = WaypointFormatAdapter(config, now_fn=self._now_seconds)
        self.output_enabled = config.output_enabled
        self.publisher = (
            self.create_publisher(Float32MultiArray, config.output_topic, 10) if self.output_enabled else None
        )
        self.create_subscription(PointStamped, config.input_topic, self._on_waypoint, 10)
        if config.coordinate_mode == "ego_from_odom":
            self.create_subscription(Odometry, config.odom_topic, self._on_odometry, 10)
        self.get_logger().info(
            "waypoint adapter ready: "
            f"input={config.input_topic} output={config.output_topic} "
            f"mode={config.coordinate_mode} output_enabled={config.output_enabled}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("config_path", "")
        self.declare_parameter("input_topic", "/way_point")
        self.declare_parameter("output_topic", "/waypoint")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("input_frame", "map")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("coordinate_mode", "ego_from_odom")
        self.declare_parameter("output_message_type", "std_msgs/msg/Float32MultiArray")
        self.declare_parameter("include_z", False)
        self.declare_parameter("max_input_age_s", 1.0)
        self.declare_parameter("output_enabled", False)
        self.declare_parameter("static_translation_xy_m", [0.0, 0.0])
        self.declare_parameter("static_yaw_rad", 0.0)

    def _load_config(self) -> WaypointAdapterConfig:
        config_path = str(self.get_parameter("config_path").value or "").strip()
        if config_path:
            try:
                return WaypointAdapterConfig.from_yaml(config_path)
            except WaypointAdapterError as exc:
                raise RuntimeError(str(exc)) from exc
        return WaypointAdapterConfig.from_mapping(
            {
                "input_topic": self.get_parameter("input_topic").value,
                "output_topic": self.get_parameter("output_topic").value,
                "odom_topic": self.get_parameter("odom_topic").value,
                "input_frame": self.get_parameter("input_frame").value,
                "output_frame": self.get_parameter("output_frame").value,
                "coordinate_mode": self.get_parameter("coordinate_mode").value,
                "output_message_type": self.get_parameter("output_message_type").value,
                "include_z": self.get_parameter("include_z").value,
                "max_input_age_s": self.get_parameter("max_input_age_s").value,
                "output_enabled": self.get_parameter("output_enabled").value,
                "static_translation_xy_m": self.get_parameter("static_translation_xy_m").value,
                "static_yaw_rad": self.get_parameter("static_yaw_rad").value,
            }
        )

    def _on_odometry(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        self.adapter.update_pose(
            Pose3D(
                position=(float(pose.position.x), float(pose.position.y), float(pose.position.z)),
                orientation_xyzw=(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ),
                frame_id=str(msg.header.frame_id or self.adapter.config.input_frame),
                stamp=self._stamp(msg),
            )
        )

    def _on_waypoint(self, msg: PointStamped) -> None:
        try:
            adapted = self.adapter.convert(
                (msg.point.x, msg.point.y, msg.point.z),
                frame_id=msg.header.frame_id,
                stamp=self._stamp(msg),
                now=self._now_seconds(),
            )
        except WaypointAdapterError as exc:
            self.get_logger().error(f"dropping unsafe waypoint: {exc}")
            return
        if adapted is None:
            self.get_logger().warning("dropping stale or pose-unavailable waypoint")
            return
        if self.publisher is None:
            self.get_logger().info(f"converted waypoint (output disabled): {list(adapted.values)}")
            return
        output = Float32MultiArray()
        output.data = list(adapted.values)
        self.publisher.publish(output)

    def _stamp(self, msg: Any) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9

    def _now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StriveWaypointAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
