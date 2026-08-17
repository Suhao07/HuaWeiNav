"""ROS2 HIL lower-layer simulator for the VLN motion action contract.

This node deliberately simulates only SysNav's observable lower-layer topics.
It does not publish ``/cmd_vel`` and therefore cannot move a real chassis.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, String
from strive_motion_msgs.action import ExecuteWaypoint
from strive_motion_msgs.msg import SafetyState


class MotionHilLowerNode(Node):
    """Simulate SysNav lower feedback for one deterministic motion scenario."""

    def __init__(self) -> None:
        """Create topic publishers, waypoint subscriber, and action client."""

        super().__init__("strive_motion_hil_lower")
        self.declare_parameter("scenario", "reached")
        self.declare_parameter("action_name", "/strive/execute_waypoint")
        self.declare_parameter("cancel_after_s", 0.4)
        self.declare_parameter("timeout_s", 1.0)
        self.declare_parameter("native_planner", False)
        self.declare_parameter("native_safety", False)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("artifact_path", "")
        self.scenario = str(self.get_parameter("scenario").value).strip().lower()
        self.native_planner = _as_bool(self.get_parameter("native_planner").value)
        self.native_safety = _as_bool(self.get_parameter("native_safety").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        configured_artifact = str(self.get_parameter("artifact_path").value or "")
        self.artifact_path = Path(configured_artifact) if configured_artifact else None
        self.cancel_after_s = float(self.get_parameter("cancel_after_s").value)
        self.target_x = 2.0
        self.target_y = 0.0
        self.current_x = 0.0
        self.goal_received_at: Optional[float] = None
        self.goal_handle = None
        self.result = None
        self.finished = False
        self._native_path_received = False
        self._native_path_messages = 0
        self._final_cmd_messages = 0
        self._nonzero_final_cmd_messages = 0
        self._max_final_linear_speed = 0.0
        self._max_final_angular_speed = 0.0
        self._latest_final_linear_x = 0.0
        self._last_motion_update_at = time.monotonic()
        self._action_client = ActionClient(self, ExecuteWaypoint, str(self.get_parameter("action_name").value))

        self._odom_pub = self.create_publisher(Odometry, "/aft_mapped_to_init", 10)
        self._path_pub = None if self.native_planner else self.create_publisher(NavPath, "/path", 10)
        self._planner_status_pub = None if self.native_planner else self.create_publisher(String, "/local_planner/status", 10)
        # native_safety 使用 mux 自己发布的安全状态，避免 HIL 的“clear”消息
        # 覆盖 mux 的 HOLD/STALE_INPUT 事实；普通 HIL 仍需模拟该反馈。
        self._safety_pub = None if self.native_safety else self.create_publisher(SafetyState, "/platform/safety_state", 10)
        self._scan_pub = self.create_publisher(PointCloud2, "/hil/registered_scan", 10) if self.native_planner else None
        self._autonomy_enable_pub = self.create_publisher(Bool, "/platform/autonomy_enable", 10) if self.native_safety else None
        self._manual_takeover_pub = self.create_publisher(Bool, "/platform/manual_takeover", 10) if self.native_safety else None
        self._estop_pub = self.create_publisher(Bool, "/platform/estop_active", 10) if self.native_safety else None
        self.create_subscription(PointStamped, "/way_point", self._waypoint_callback, 10)
        if self.native_planner:
            self.create_subscription(NavPath, "/path", self._native_path_callback, 10)
        if self.native_safety:
            self.create_subscription(TwistStamped, self.cmd_vel_topic, self._final_cmd_callback, 10)
        self._timer = self.create_timer(0.05, self._tick)

    def _waypoint_callback(self, message: PointStamped) -> None:
        """Record one map-frame waypoint published by the motion server."""

        self.target_x = float(message.point.x)
        self.target_y = float(message.point.y)
        self.goal_received_at = time.monotonic()

    def _tick(self) -> None:
        """Publish deterministic odom/path/status/safety feedback."""

        now = time.monotonic()
        if self.native_planner:
            self._publish_registered_scan()
        if self.native_safety:
            self._publish_safety_controls()
        self._publish_path()
        if self.scenario == "manual":
            self._publish_safety(SafetyState.MANUAL_TAKEOVER, "manual_takeover")
        elif self.scenario == "stale":
            self._publish_safety(SafetyState.STALE_INPUT, "pointcloud_stale")
        else:
            self._publish_safety(SafetyState.CLEAR, "clear")

        if self.goal_received_at is None:
            self._publish_status("waiting_for_sensor")
            self._publish_odom()
            return

        elapsed = now - self.goal_received_at
        if self.scenario == "reached":
            self.current_x = min(self.target_x, self.current_x + 0.25)
            self._publish_status("tracking")
        elif self.native_safety:
            # 中文说明：native_safety 不再因“收到 path”直接推进位姿；只有
            # SafetyVelocityMux 的最终速度命令才会驱动合成底盘状态，避免把
            # path 生成误当成底盘运动已经发生。
            dt = max(0.0, now - self._last_motion_update_at)
            self._last_motion_update_at = now
            if self._native_path_received and self._latest_final_linear_x > 0.0:
                self.current_x = min(
                    self.target_x,
                    self.current_x + self._latest_final_linear_x * dt,
                )
        elif self.native_planner:
            # The real localPlanner owns /path and /local_planner/status. The
            # HIL node advances pose only after receiving a non-empty path.
            if self._native_path_received:
                self.current_x = min(self.target_x, self.current_x + 0.25)
        elif self.scenario == "blocked":
            self._publish_status("no_feasible_path")
        elif self.scenario == "cancel":
            self._publish_status("tracking")
            if self.goal_handle is not None and elapsed >= self.cancel_after_s:
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
        elif self.scenario == "timeout":
            self._publish_status("tracking")
        elif self.scenario in {"manual", "stale"}:
            self._publish_status("tracking")
        else:
            self._publish_status("no_feasible_path")
        self._publish_odom()

    def _publish_odom(self) -> None:
        """Publish map-frame pose and zero velocity after reaching the target."""

        message = Odometry()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.pose.position.x = self.current_x
        message.pose.pose.position.y = self.current_x * 0.0
        message.pose.pose.orientation.w = 1.0
        if self.native_safety:
            message.twist.twist.linear.x = (
                0.0 if self.current_x >= self.target_x else self._latest_final_linear_x
            )
        else:
            message.twist.twist.linear.x = 0.0 if self.current_x >= self.target_x else 0.25
        self._odom_pub.publish(message)

    def _publish_safety_controls(self) -> None:
        """Enable autonomy and keep manual takeover/estop deasserted in HIL."""

        enable = Bool()
        enable.data = True
        self._autonomy_enable_pub.publish(enable)
        takeover = Bool()
        takeover.data = False
        self._manual_takeover_pub.publish(takeover)
        estop = Bool()
        estop.data = False
        self._estop_pub.publish(estop)

    def _final_cmd_callback(self, message: TwistStamped) -> None:
        """Record final safety-mux output without commanding a chassis."""

        linear = float(message.twist.linear.x)
        lateral = float(message.twist.linear.y)
        angular = float(message.twist.angular.z)
        self._final_cmd_messages += 1
        self._nonzero_final_cmd_messages += int(
            abs(linear) + abs(lateral) + abs(angular) > 1e-4
        )
        self._max_final_linear_speed = max(
            self._max_final_linear_speed, (linear * linear + lateral * lateral) ** 0.5
        )
        self._max_final_angular_speed = max(self._max_final_angular_speed, abs(angular))
        self._latest_final_linear_x = linear

    def _publish_path(self) -> None:
        """Publish a vehicle-frame local path for feedback diagnostics."""

        if self._path_pub is None:
            return

        message = NavPath()
        message.header.frame_id = "vehicle"
        message.header.stamp = self.get_clock().now().to_msg()
        for x in (0.5, 1.0, 1.5):
            pose = message.poses.add() if hasattr(message.poses, "add") else None
            if pose is None:
                from geometry_msgs.msg import PoseStamped

                pose = PoseStamped()
                message.poses.append(pose)
            pose.header.frame_id = "vehicle"
            pose.pose.position.x = x
        self._path_pub.publish(message)

    def _native_path_callback(self, message: NavPath) -> None:
        """Record whether the migrated local planner produced a usable path."""

        valid_path = len(message.poses) > 1 and any(
            abs(float(pose.pose.position.x)) + abs(float(pose.pose.position.y)) > 0.1
            for pose in message.poses
        )
        self._native_path_messages += 1
        # 中文说明：localPlanner 到达后会发布单点零路径清空历史轨迹；验收关心
        # “是否曾经生成过有效路径”，因此该事实只能单调置真，不能被清空消息覆盖。
        self._native_path_received = self._native_path_received or valid_path

    def _publish_registered_scan(self) -> None:
        """Publish one clear scan point for the native local-planner HIL.

        The point is outside the planner crop radius, so the real planner must
        still execute its scan conversion, path scoring, and `/path` publish.
        """

        if self._scan_pub is None:
            return
        import struct

        message = PointCloud2()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.height = 1
        message.width = 1
        message.is_bigendian = False
        message.is_dense = True
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.point_step = 16
        message.row_step = 16
        message.data = list(struct.pack("<ffff", 10.0, 10.0, 0.0, 0.0))
        self._scan_pub.publish(message)

    def _publish_status(self, value: str) -> None:
        """Publish one local-planner status token."""

        if self._planner_status_pub is None:
            return

        message = String()
        message.data = value
        self._planner_status_pub.publish(message)

    def _publish_safety(self, state: int, reason: str) -> None:
        """Publish the final velocity safety state consumed by MotionServer."""

        if self._safety_pub is None:
            return
        message = SafetyState()
        message.state = state
        message.reason_code = reason
        message.autonomy_enabled = state == SafetyState.CLEAR
        message.manual_takeover = state == SafetyState.MANUAL_TAKEOVER
        message.estop_active = False
        self._safety_pub.publish(message)

    def send_goal_and_wait(self) -> int:
        """Send one action goal and return the expected HIL process result."""

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("motion action server is unavailable")
            return 2
        goal = ExecuteWaypoint.Goal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = self.target_x
        goal.target_pose.pose.orientation.w = 1.0
        goal.xy_tolerance_m = 0.35
        goal.timeout_s = float(self.get_parameter("timeout_s").value)
        goal.motion_profile = "go_to_object"
        goal_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        self.goal_handle = goal_future.result()
        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().error("HIL action goal was rejected")
            return 2
        result_future = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.result = result_future.result().result
        outcome = int(self.result.outcome)
        self.get_logger().info(
            f"HIL scenario={self.scenario} outcome={outcome} reason={self.result.reason_code} "
            f"native_path_received={self._native_path_received} "
            f"native_path_messages={self._native_path_messages} "
            f"final_cmd_messages={self._final_cmd_messages} "
            f"nonzero_final_cmd_messages={self._nonzero_final_cmd_messages}"
        )
        self._write_artifact(outcome=outcome)
        if outcome != self._expected_outcome():
            return 1
        if self.native_safety and (
            not self._native_path_received or self._nonzero_final_cmd_messages == 0
        ):
            self.get_logger().error(
                "native_safety did not observe both a migrated path and a non-zero "
                "final SafetyVelocityMux command"
            )
            return 1
        return 0

    def _write_artifact(self, *, outcome: int) -> None:
        """Persist machine-readable evidence for a HIL execution."""

        if self.artifact_path is None:
            return
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": self.scenario,
            "native_planner": self.native_planner,
            "native_safety": self.native_safety,
            "native_path_received": self._native_path_received,
            "native_path_messages": self._native_path_messages,
            "final_cmd_topic": self.cmd_vel_topic if self.native_safety else None,
            "final_cmd_messages": self._final_cmd_messages,
            "nonzero_final_cmd_messages": self._nonzero_final_cmd_messages,
            "max_final_linear_speed_mps": self._max_final_linear_speed,
            "max_final_angular_speed_rps": self._max_final_angular_speed,
            "outcome": outcome,
            "reason_code": getattr(self.result, "reason_code", "") if self.result is not None else "",
        }
        self.artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _expected_outcome(self) -> int:
        """Return the action outcome expected for the selected scenario."""

        return {
            "reached": ExecuteWaypoint.Result.REACHED,
            "native_planner": ExecuteWaypoint.Result.REACHED,
            "native_safety": ExecuteWaypoint.Result.REACHED,
            "blocked": ExecuteWaypoint.Result.BLOCKED,
            "timeout": ExecuteWaypoint.Result.TIMEOUT,
            "cancel": ExecuteWaypoint.Result.PREEMPTED,
            "manual": ExecuteWaypoint.Result.MANUAL_TAKEOVER,
            "stale": ExecuteWaypoint.Result.SAFETY_STOP,
        }.get(self.scenario, ExecuteWaypoint.Result.FAILED)


def main(args: Optional[list[str]] = None) -> None:
    """Run one HIL scenario and return a process status for CI."""

    rclpy.init(args=args)
    node = MotionHilLowerNode()
    try:
        raise SystemExit(node.send_goal_and_wait())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _as_bool(value: object) -> bool:
    """Parse ROS boolean parameters consistently across CLI and launch files."""

    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main(sys.argv[1:])
