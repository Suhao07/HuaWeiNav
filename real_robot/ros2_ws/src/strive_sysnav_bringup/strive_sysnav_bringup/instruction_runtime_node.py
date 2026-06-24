"""ROS2 node for the STRIVE SysNav-backed high-level runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tare_planner.msg import DetectionResult, ObjectNodeList, RoomNodeList

from real_robot.observation_cache import RosObservationCache
from real_robot.contracts import Pose3D
from real_robot.sysnav_ros_adapters import RosNavigationStatusProvider, RosWaypointController
from real_robot.sysnav_runtime import (
    DryRunMotionController,
    FirstObjectSmokePolicy,
    RuntimeDecisionJsonlWriter,
    RuntimeReadiness,
    SysNavInstructionRuntime,
    SysNavSemanticMapBridge,
    WaitInstructionPolicy,
    runtime_decision_to_dict,
)


class StriveInstructionRuntimeNode(Node):
    """Bridge SysNav semantic map topics into STRIVE high-level decisions."""

    def __init__(self) -> None:
        """Create subscriptions, runtime helpers, and the periodic decision timer."""

        super().__init__("strive_instruction_runtime")
        self._declare_parameters()

        self.instruction = str(self.get_parameter("instruction").value or "")
        self.world_frame = str(self.get_parameter("world_frame").value or "map")
        self.dry_run = _param_bool(self.get_parameter("dry_run").value)
        self.require_image = _param_bool(self.get_parameter("require_image").value)
        self.require_pose = _param_bool(self.get_parameter("require_pose").value)
        self.policy_mode = str(self.get_parameter("policy_mode").value or "wait")
        self.prior_map_path = str(self.get_parameter("prior_map_path").value or "")
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self._latest_pose: Optional[Pose3D] = None
        self._latest_image_stamp: Optional[float] = None

        run_directory = Path(str(self.get_parameter("run_directory").value or "/tmp/strive_real_robot_runtime"))
        self._decision_writer = RuntimeDecisionJsonlWriter(run_directory / "runtime_decisions.jsonl")
        image_directory_param = str(self.get_parameter("observation_image_directory").value or "")
        image_directory = Path(image_directory_param) if image_directory_param else run_directory / "observations"
        self.observation_cache = RosObservationCache(
            image_directory=image_directory,
            persist_images=_param_bool(self.get_parameter("persist_observation_images").value),
            rgb_topic=self.image_topic,
            depth_topic=str(self.get_parameter("depth_topic").value or ""),
            pointcloud_topic=str(self.get_parameter("pointcloud_topic").value or ""),
            now_fn=self._now_seconds,
        )
        self.navigation_status_provider = RosNavigationStatusProvider(
            xy_tolerance_m=float(self.get_parameter("xy_goal_tolerance_m").value),
            z_tolerance_m=float(self.get_parameter("z_goal_tolerance_m").value),
            heading_tolerance_rad=None,
            timeout_s=float(self.get_parameter("navigation_timeout_s").value),
            no_progress_timeout_s=float(self.get_parameter("no_progress_timeout_s").value),
            min_progress_delta_m=float(self.get_parameter("min_progress_delta_m").value),
            path_stale_timeout_s=float(self.get_parameter("path_stale_timeout_s").value),
            world_frame=self.world_frame,
            now_fn=self._now_seconds,
        )

        self.semantic_bridge = SysNavSemanticMapBridge(robot_pose_provider=self._current_pose)
        self.high_level_policy = self._build_policy(self.policy_mode)
        self.motion_controller = self._build_motion_controller()
        self.runtime = SysNavInstructionRuntime(
            semantic_map_bridge=self.semantic_bridge,
            high_level_policy=self.high_level_policy,
            motion_controller=self.motion_controller,
            readiness_provider=self._readiness,
            now_fn=self._now_seconds,
        )

        queue_size = int(self.get_parameter("queue_size").value)
        self.create_subscription(
            ObjectNodeList,
            str(self.get_parameter("object_topic").value),
            self.semantic_bridge.update_object_nodes,
            queue_size,
        )
        self.create_subscription(
            RoomNodeList,
            str(self.get_parameter("room_topic").value),
            self.semantic_bridge.update_room_nodes,
            queue_size,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._update_odom,
            queue_size,
        )
        self.create_subscription(
            NavPath,
            str(self.get_parameter("path_topic").value),
            self.navigation_status_provider.update_path,
            queue_size,
        )
        planner_status_topic = str(self.get_parameter("planner_status_topic").value or "")
        if planner_status_topic:
            self.create_subscription(
                String,
                planner_status_topic,
                self.navigation_status_provider.update_local_planner_status,
                queue_size,
            )
        self.create_subscription(
            Image,
            self.image_topic,
            self._update_image,
            queue_size,
        )
        self.create_subscription(
            DetectionResult,
            self.detection_topic,
            self.observation_cache.update_detection_result,
            queue_size,
        )
        depth_topic = str(self.get_parameter("depth_topic").value or "")
        if depth_topic:
            self.create_subscription(
                Image,
                depth_topic,
                self.observation_cache.update_depth_image,
                queue_size,
            )
        pointcloud_topic = str(self.get_parameter("pointcloud_topic").value or "")
        if pointcloud_topic:
            self.create_subscription(
                PointCloud2,
                pointcloud_topic,
                self.observation_cache.update_pointcloud,
                queue_size,
            )

        period_s = float(self.get_parameter("decision_period_s").value)
        self.create_timer(period_s, self._tick)
        self.get_logger().info(
            "STRIVE instruction runtime started: "
            f"dry_run={self.dry_run}, policy_mode={self.policy_mode}, "
            f"run_directory={run_directory}, prior_map_path={self.prior_map_path or '<disabled>'}"
        )

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by the runtime node."""

        self.declare_parameter("instruction", "")
        self.declare_parameter("object_topic", "/object_nodes_list")
        self.declare_parameter("room_topic", "/room_nodes_list")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("planner_status_topic", "")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("detection_topic", "/detection_result")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("pointcloud_topic", "")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("policy_mode", "wait")
        self.declare_parameter("prior_map_path", "")
        self.declare_parameter("run_directory", "/tmp/strive_real_robot_runtime")
        self.declare_parameter("decision_period_s", 1.0)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("require_pose", True)
        self.declare_parameter("require_image", True)
        self.declare_parameter("xy_goal_tolerance_m", 0.35)
        self.declare_parameter("z_goal_tolerance_m", 1.0)
        self.declare_parameter("navigation_timeout_s", 60.0)
        self.declare_parameter("no_progress_timeout_s", 12.0)
        self.declare_parameter("min_progress_delta_m", 0.05)
        self.declare_parameter("path_stale_timeout_s", 5.0)
        self.declare_parameter("persist_observation_images", False)
        self.declare_parameter("observation_image_directory", "")

    def _build_policy(self, policy_mode: str):
        """Build the selected high-level policy implementation."""

        normalized = policy_mode.strip().lower()
        if normalized == "first_object_smoke":
            return FirstObjectSmokePolicy()
        if normalized in {"wait", "disabled", ""}:
            return WaitInstructionPolicy("high-level semantic policy is disabled")
        self.get_logger().warning(f"unknown policy_mode={policy_mode}; falling back to WAIT")
        return WaitInstructionPolicy(f"unknown policy_mode={policy_mode}")

    def _build_motion_controller(self):
        """Build either dry-run or waypoint-publishing motion controller."""

        if self.dry_run:
            return DryRunMotionController()
        return RosWaypointController(
            node=self,
            waypoint_topic=str(self.get_parameter("waypoint_topic").value),
            world_frame=self.world_frame,
            status_provider=self.navigation_status_provider,
        )

    def _update_odom(self, msg: Odometry) -> None:
        """Cache the latest robot pose from odometry."""

        pose = msg.pose.pose
        self._latest_pose = Pose3D(
            position=(float(pose.position.x), float(pose.position.y), float(pose.position.z)),
            orientation_xyzw=(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
            frame_id=msg.header.frame_id or self.world_frame,
            stamp=_stamp_to_seconds(msg.header.stamp),
        )
        self.navigation_status_provider.update_pose(self._latest_pose)
        self.observation_cache.update_pose(self._latest_pose)

    def _update_image(self, msg: Image) -> None:
        """Cache the latest image timestamp as readiness evidence."""

        record = self.observation_cache.update_rgb_image(msg, topic=self.image_topic)
        self._latest_image_stamp = record.timestamp

    def _current_pose(self) -> Pose3D:
        """Return the latest pose or a safe placeholder pose."""

        return self._latest_pose or Pose3D(position=(0.0, 0.0, 0.0), frame_id=self.world_frame)

    def _readiness(self) -> RuntimeReadiness:
        """Return whether required live inputs have arrived."""

        missing = []
        if not self.semantic_bridge.has_object_snapshot():
            missing.append("object_nodes")
        if self.require_pose and self._latest_pose is None:
            missing.append("pose")
        if self.require_image and self._latest_image_stamp is None:
            missing.append("image")
        if missing:
            return RuntimeReadiness(
                ready=False,
                reason="waiting for live inputs: " + ", ".join(missing),
                metadata={
                    "missing": missing,
                    "has_object_nodes": self.semantic_bridge.has_object_snapshot(),
                    "has_pose": self._latest_pose is not None,
                    "has_image": self._latest_image_stamp is not None,
                    "dry_run": self.dry_run,
                },
            )
        return RuntimeReadiness(
            ready=True,
            reason="live inputs ready",
            metadata={
                "has_object_nodes": True,
                "has_pose": self._latest_pose is not None,
                "has_image": self._latest_image_stamp is not None,
                "dry_run": self.dry_run,
            },
        )

    def _tick(self) -> None:
        """Run one high-level runtime step and log the decision."""

        decision = self.runtime.step(self.instruction)
        payload = self._decision_writer.write(decision)
        intent = payload.get("intent", {})
        mode = intent.get("mode", "unknown")
        reason = payload.get("reason", "")
        if mode == "wait":
            self.get_logger().info(f"STRIVE runtime WAIT: {reason}")
        elif self.dry_run:
            self.get_logger().info(f"STRIVE runtime dry-run intent={mode}: {reason}")
        else:
            self.get_logger().info(f"STRIVE runtime dispatched intent={mode}: {reason}")
        self.get_logger().debug(str(runtime_decision_to_dict(decision)))

    def _now_seconds(self) -> float:
        """Return ROS clock time in seconds."""

        msg = self.get_clock().now().to_msg()
        return _stamp_to_seconds(msg)


def _stamp_to_seconds(stamp) -> float:
    """Convert a ROS stamp-like object to seconds."""

    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) / 1e9


def _param_bool(value) -> bool:
    """Return a robust boolean from ROS parameter values or launch strings."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(args: Optional[list[str]] = None) -> None:
    """Run the STRIVE instruction runtime ROS2 node."""

    rclpy.init(args=args)
    node = StriveInstructionRuntimeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
