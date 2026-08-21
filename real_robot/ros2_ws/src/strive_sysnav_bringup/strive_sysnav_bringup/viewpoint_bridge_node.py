"""Expose SysNav viewpoint IDs as timestamp-aligned executable poses.

SysNav's public ``ViewpointRep`` message intentionally carries only the
viewpoint ID and the state-estimation timestamp.  The migrated runtime needs a
pose to construct a platform-neutral ``MotionGoal``.  This node performs the
smallest possible ROS-side adaptation: it joins the viewpoint record with the
nearest odometry sample and republishes the result without ranking or
replanning viewpoints.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tare_planner.msg import ObjectNodeList, ViewpointPose, ViewpointRep

from real_robot.sysnav_viewpoint_bridge import SysNavViewpointBridgeModel, SysNavViewpointRecord


class SysNavViewpointBridgeNode(Node):
    """Join SysNav viewpoint headers, odometry, and direct object observations."""

    def __init__(self) -> None:
        """Create the bridge subscriptions and the viewpoint-pose publisher."""

        super().__init__("strive_sysnav_viewpoint_bridge")
        self.declare_parameter("viewpoint_topic", "/viewpoint_rep_header")
        self.declare_parameter("object_topic", "/object_nodes_list")
        self.declare_parameter("odom_topic", "/state_estimation")
        self.declare_parameter("output_topic", "/strive/sysnav/viewpoint_pose")
        self.declare_parameter("odom_history_size", 400)
        self.declare_parameter("max_time_offset_s", 0.25)
        self.declare_parameter("queue_size", 10)

        queue_size = int(self.get_parameter("queue_size").value)
        self.model = SysNavViewpointBridgeModel(
            max_time_offset_s=float(self.get_parameter("max_time_offset_s").value),
            odom_history_size=int(self.get_parameter("odom_history_size").value),
        )

        output_topic = str(self.get_parameter("output_topic").value)
        self.publisher = self.create_publisher(ViewpointPose, output_topic, queue_size)
        self.create_subscription(
            ViewpointRep,
            str(self.get_parameter("viewpoint_topic").value),
            self._on_viewpoint,
            queue_size,
        )
        self.create_subscription(
            ObjectNodeList,
            str(self.get_parameter("object_topic").value),
            self._on_objects,
            queue_size,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odometry,
            queue_size,
        )
        self.get_logger().info(
            "SysNav viewpoint bridge ready: "
            f"viewpoint={self.get_parameter('viewpoint_topic').value}, "
            f"odom={self.get_parameter('odom_topic').value}, "
            f"output={output_topic}"
        )

    def _on_odometry(self, msg: Any) -> None:
        """Cache odometry and retry viewpoint records waiting for a pose."""

        self._publish_records(self.model.update_odometry(msg))

    def _on_viewpoint(self, msg: Any) -> None:
        """Cache one SysNav viewpoint header and publish when pose is available."""

        try:
            records = self.model.update_viewpoint(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        self._publish_records(records)

    def _on_objects(self, msg: Any) -> None:
        """Accumulate direct object observations associated with viewpoints."""

        self._publish_records(self.model.update_object_nodes(msg))

    def _publish_records(self, records: Sequence[SysNavViewpointRecord]) -> None:
        """Serialize resolved records to ``tare_planner/ViewpointPose``."""

        for record in records:
            # 中文核心约束：ROS node 只做消息序列化，viewpoint 解析和时间门限在纯模型中完成。
            output = ViewpointPose()
            timestamp_ns = record.timestamp_ns
            if timestamp_ns is None:
                timestamp_ns = int(round(record.timestamp * 1e9))
            output.header.stamp.sec, output.header.stamp.nanosec = divmod(
                int(timestamp_ns), 1_000_000_000
            )
            output.header.frame_id = record.pose.frame_id
            output.viewpoint_id = record.viewpoint_id
            output.pose.position.x = record.pose.position[0]
            output.pose.position.y = record.pose.position[1]
            output.pose.position.z = record.pose.position[2]
            output.pose.orientation.x = record.pose.orientation_xyzw[0]
            output.pose.orientation.y = record.pose.orientation_xyzw[1]
            output.pose.orientation.z = record.pose.orientation_xyzw[2]
            output.pose.orientation.w = record.pose.orientation_xyzw[3]
            output.observed_object_ids = list(record.observed_object_ids)
            self.publisher.publish(output)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Run the SysNav viewpoint bridge ROS node."""

    rclpy.init(args=args)
    node = SysNavViewpointBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
