"""ROS2 node for the VLN SysNav-backed high-level runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tare_planner.msg import DetectionResult, ObjectNodeList, RoomNodeList

from instruction_adapter.compiler import compile_instruction_plan
from planning.semantic_snapshot_context import (
    SemanticMapSnapshotIntentAdapter,
    StaticInstructionPlanProvider,
)
from prior_map.real_robot import PriorMapRealRobotConfig, build_prior_map_real_robot_runtime
from real_robot.observation_cache import ObjectCropEvidenceProvider, RosObservationCache
from real_robot.control.controller_contract import (
    ControllerContractError,
    load_controller_contract,
    validate_controller_contract,
)
from real_robot.contracts import NavigationStatusCode, Pose3D
from real_robot.action_motion_controller import RosActionMotionController
from real_robot.sysnav_ros_adapters import (
    RosNavigationStatusProvider,
    RosWaypointController,
    normalize_ros_topic_name,
    validate_non_velocity_publish_topic,
)
from real_robot.sysnav_runtime import (
    DryRunMotionController,
    FinalInstructionVerifierAdapter,
    FirstObjectSmokePolicy,
    ViewpointEvidenceLoop,
    RuntimeDecisionJsonlWriter,
    RuntimeReadiness,
    SysNavInstructionRuntime,
    SysNavSemanticMapBridge,
    WaitInstructionPolicy,
    runtime_decision_to_dict,
)


class StriveInstructionRuntimeNode(Node):
    """Bridge SysNav semantic map topics into VLN high-level decisions."""

    def __init__(self) -> None:
        """Create subscriptions, runtime helpers, and the periodic decision timer."""

        super().__init__("strive_instruction_runtime")
        self._declare_parameters()

        self.instruction = str(self.get_parameter("instruction").value or "")
        self.world_frame = str(self.get_parameter("world_frame").value or "map")
        # 安全默认：dry_run=true 时只写 runtime_decisions.jsonl，不发布 /way_point。
        self.dry_run = _param_bool(self.get_parameter("dry_run").value)
        self.dry_run_status = str(self.get_parameter("dry_run_status").value or "idle")
        self.require_image = _param_bool(self.get_parameter("require_image").value)
        self.require_pose = _param_bool(self.get_parameter("require_pose").value)
        self.policy_mode = str(self.get_parameter("policy_mode").value or "wait")
        self.dataset_target = str(self.get_parameter("dataset_target").value or "")
        self.instruction_plan_backend = str(self.get_parameter("instruction_plan_backend").value or "llm")
        self.vlm = str(self.get_parameter("vlm").value or "cognav")
        self.enable_final_verifier = _param_bool(self.get_parameter("enable_final_verifier").value)
        self.evidence_mode = str(self.get_parameter("evidence_mode").value or "auto")
        self.prior_map_path = str(self.get_parameter("prior_map_path").value or "")
        self.prior_map_source = str(self.get_parameter("prior_map_source").value or "auto")
        self.prior_map_alignment = str(self.get_parameter("prior_map_alignment").value or "identity")
        self.enable_prior_map_vlm = _param_bool(self.get_parameter("enable_prior_map_vlm").value)
        self.enable_room_semantics = _param_bool(self.get_parameter("enable_room_semantics").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value or "/way_point")
        self.test_waypoint_topic = str(self.get_parameter("test_waypoint_topic").value or "/strive/test_way_point")
        self.motion_backend = str(self.get_parameter("motion_backend").value or "waypoint")
        self.controller_contract_file = str(self.get_parameter("controller_contract_file").value or "")
        self.hold_topic = str(self.get_parameter("hold_topic").value or "")
        self.cancel_topic = str(self.get_parameter("cancel_topic").value or "")
        self.emergency_stop_topic = str(self.get_parameter("emergency_stop_topic").value or "")
        self.allow_emergency_stop_publish = _param_bool(self.get_parameter("allow_emergency_stop_publish").value)
        self.lower_controller_enabled = _param_bool(self.get_parameter("lower_controller_enabled").value)
        self._validate_motion_safety()
        self._latest_pose: Optional[Pose3D] = None
        self._latest_image_stamp: Optional[float] = None

        run_directory = Path(str(self.get_parameter("run_directory").value or "/tmp/strive_real_robot_runtime"))
        self._decision_writer = RuntimeDecisionJsonlWriter(run_directory / "runtime_decisions.jsonl")
        self.prior_map_runtime = build_prior_map_real_robot_runtime(
            PriorMapRealRobotConfig(
                prior_map_path=self.prior_map_path,
                prior_map_source=self.prior_map_source,
                prior_map_alignment=self.prior_map_alignment,
                enable_high_level_vlm=self.enable_prior_map_vlm,
                vlm=self.vlm,
                high_level_interval=max(1, int(self.get_parameter("prior_map_vlm_interval").value)),
                room_semantic_interval=max(1, int(self.get_parameter("room_semantic_interval").value)),
                enable_room_semantics=self.enable_room_semantics,
                run_directory=str(run_directory),
            )
        )
        if self.prior_map_runtime is not None:
            self.get_logger().info(
                "prior map enabled for real-robot runtime: "
                f"scene={self.prior_map_runtime.base_map.scene_id}, "
                f"source={self.prior_map_runtime.base_map.source_format}, "
                f"alignment={self.prior_map_runtime.alignment.diagnostics_payload()}"
            )
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
        self.semantic_bridge = SysNavSemanticMapBridge(robot_pose_provider=self._current_pose)
        # 中文注释：SysNav RoomNode 只有 mask 消息，RGB 由独立相机 topic 提供；
        # 在 bridge 层注入两个证据 provider，才能复现“RGB + room mask”分类输入。
        self.semantic_bridge.room_adapter.rgb_path_provider = self.observation_cache.latest_rgb_visual_path
        self.semantic_bridge.room_adapter.room_mask_path_provider = self.observation_cache.persist_room_mask
        self.navigation_status_provider = RosNavigationStatusProvider(
            xy_tolerance_m=float(self.get_parameter("xy_goal_tolerance_m").value),
            z_tolerance_m=float(self.get_parameter("z_goal_tolerance_m").value),
            heading_tolerance_rad=None,
            timeout_s=float(self.get_parameter("navigation_timeout_s").value),
            no_progress_timeout_s=float(self.get_parameter("no_progress_timeout_s").value),
            min_progress_delta_m=float(self.get_parameter("min_progress_delta_m").value),
            path_stale_timeout_s=float(self.get_parameter("path_stale_timeout_s").value),
            velocity_tolerance_mps=float(self.get_parameter("velocity_tolerance_mps").value),
            stable_reach_time_s=float(self.get_parameter("stable_reach_time_s").value),
            world_frame=self.world_frame,
            now_fn=self._now_seconds,
        )

        self.high_level_policy = self._build_policy(self.policy_mode)
        self.motion_controller = self._build_motion_controller()
        # evidence loop 只在 semantic_snapshot 模式下启用；wait/smoke 不需要 verifier 链路。
        self.viewpoint_evidence_loop = self._build_viewpoint_evidence_loop()
        self.runtime = SysNavInstructionRuntime(
            semantic_map_bridge=self.semantic_bridge,
            high_level_policy=self.high_level_policy,
            motion_controller=self.motion_controller,
            viewpoint_evidence_loop=self.viewpoint_evidence_loop,
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
            f"lower_controller_enabled={self.lower_controller_enabled}, waypoint_topic={self.waypoint_topic}, "
            f"run_directory={run_directory}, prior_map_path={self.prior_map_path or '<disabled>'}, "
            f"prior_map_source={self.prior_map_source}, prior_map_alignment={self.prior_map_alignment}"
        )

    def _declare_parameters(self) -> None:
        """Declare ROS parameters used by the runtime node."""

        self.declare_parameter("instruction", "")
        self.declare_parameter("object_topic", "/object_nodes_list")
        self.declare_parameter("room_topic", "/room_nodes_list")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("planner_status_topic", "/local_planner/status")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("detection_topic", "/detection_result")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("pointcloud_topic", "")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("test_waypoint_topic", "/strive/test_way_point")
        self.declare_parameter("motion_backend", "waypoint")
        self.declare_parameter("motion_action_name", "/strive/execute_waypoint")
        self.declare_parameter("controller_contract_file", "")
        self.declare_parameter("hold_topic", "")
        self.declare_parameter("cancel_topic", "")
        self.declare_parameter("emergency_stop_topic", "")
        self.declare_parameter("allow_emergency_stop_publish", False)
        self.declare_parameter("lower_controller_enabled", False)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("policy_mode", "wait")
        self.declare_parameter("dataset_target", "")
        self.declare_parameter("instruction_plan_backend", "llm")
        self.declare_parameter("vlm", "cognav")
        self.declare_parameter("enable_final_verifier", False)
        self.declare_parameter("evidence_mode", "auto")
        self.declare_parameter("prior_map_path", "")
        self.declare_parameter("prior_map_source", "auto")
        self.declare_parameter("prior_map_alignment", "identity")
        self.declare_parameter("enable_prior_map_vlm", False)
        self.declare_parameter("prior_map_vlm_interval", 10)
        self.declare_parameter("room_semantic_interval", 10)
        self.declare_parameter("enable_room_semantics", False)
        self.declare_parameter("run_directory", "/tmp/strive_real_robot_runtime")
        self.declare_parameter("decision_period_s", 1.0)
        self.declare_parameter("queue_size", 10)
        # ``use_sim_time`` is declared by rclpy's Node constructor.  Declaring
        # it again makes the physical-runtime node fail before it can write a
        # dry-run decision on ROS 2 Humble.
        self.declare_parameter("dry_run", True)
        self.declare_parameter("dry_run_status", "idle")
        self.declare_parameter("require_pose", True)
        self.declare_parameter("require_image", True)
        self.declare_parameter("xy_goal_tolerance_m", 0.35)
        self.declare_parameter("z_goal_tolerance_m", 1.0)
        self.declare_parameter("navigation_timeout_s", 60.0)
        self.declare_parameter("no_progress_timeout_s", 12.0)
        self.declare_parameter("min_progress_delta_m", 0.05)
        self.declare_parameter("path_stale_timeout_s", 5.0)
        self.declare_parameter("velocity_tolerance_mps", 0.08)
        self.declare_parameter("stable_reach_time_s", 0.2)
        self.declare_parameter("persist_observation_images", False)
        self.declare_parameter("observation_image_directory", "")

    def _build_policy(self, policy_mode: str):
        """Build the selected high-level policy implementation."""

        normalized = policy_mode.strip().lower()
        if normalized == "first_object_smoke":
            return FirstObjectSmokePolicy()
        if normalized in {"semantic_snapshot", "semantic_map_snapshot", "instruction", "instruction_plan"}:
            if not self.instruction.strip():
                return WaitInstructionPolicy("semantic_snapshot policy requires non-empty instruction")
            # InstructionPlan 只在显式 semantic_snapshot 模式编译，避免 wait/smoke 启动时触发 LLM。
            plan = compile_instruction_plan(
                raw_instruction=self.instruction,
                dataset_target=self.dataset_target,
                backend=self.instruction_plan_backend,
                vlm=self.vlm,
            )
            self.get_logger().info(
                "compiled InstructionPlan for semantic_snapshot policy: "
                f"valid={plan.valid}, targets={plan.target_detector_prompts}, "
                f"backend={self.instruction_plan_backend}, vlm={self.vlm}"
            )
            return SemanticMapSnapshotIntentAdapter(
                StaticInstructionPlanProvider(plan),
                prior_context_provider=self._prior_map_context_provider,
                vlm=self.vlm,
            )
        if normalized in {"wait", "disabled", ""}:
            return WaitInstructionPolicy("high-level semantic policy is disabled")
        self.get_logger().warning(f"unknown policy_mode={policy_mode}; falling back to WAIT")
        return WaitInstructionPolicy(f"unknown policy_mode={policy_mode}")

    def _build_motion_controller(self):
        """Build either dry-run or waypoint-publishing motion controller."""

        if self.dry_run:
            # dry_run_status 可模拟 reached/blocked 等状态，用于不接底盘时测试 verifier 分支。
            return DryRunMotionController(status_code=_navigation_status_code(self.dry_run_status))
        if self.motion_backend.strip().lower() == "action":
            # Action backend 不直接发布 /way_point；SysNavMotionServer 才是唯一 topic owner。
            return RosActionMotionController(node=self, action_name=str(self.get_parameter("motion_action_name").value))
        return RosWaypointController(
            node=self,
            waypoint_topic=self.waypoint_topic,
            world_frame=self.world_frame,
            status_provider=self.navigation_status_provider,
            hold_topic=self.hold_topic,
            cancel_topic=self.cancel_topic,
            emergency_stop_topic=self.emergency_stop_topic,
            allow_emergency_stop_publish=self.allow_emergency_stop_publish,
        )

    def _build_viewpoint_evidence_loop(self):
        """Build reached-view evidence and optional final verifier loop."""

        if self.policy_mode.strip().lower() not in {
            "semantic_snapshot",
            "semantic_map_snapshot",
            "instruction",
            "instruction_plan",
        }:
            return None
        # ObjectCropEvidenceProvider 复用 observation cache，不直接订阅 ROS topic。
        evidence_provider = ObjectCropEvidenceProvider(
            observation_provider=self.observation_cache.latest_observation,
            object_provider=lambda: self.semantic_bridge.build_snapshot(timestamp=self._now_seconds()),
            detection_provider=lambda: self.observation_cache.latest_detection_frame,
            mode=self.evidence_mode,
            now_fn=self._now_seconds,
        )
        final_verifier = FinalInstructionVerifierAdapter(vlm=self.vlm) if self.enable_final_verifier else None
        if final_verifier is None:
            # 可先跑 evidence loop dry-run，不给 STOP 权限；打开 verifier 前先确认证据可用。
            self.get_logger().warning(
                "semantic_snapshot evidence loop is enabled without final verifier; "
                "reached goals will not produce STOP"
            )
        return ViewpointEvidenceLoop(
            motion_controller=self.motion_controller,
            evidence_provider=evidence_provider,
            final_verifier=final_verifier,
            now_fn=self._now_seconds,
        )

    def _prior_map_context_provider(self, snapshot, plan, step: int) -> dict:
        """Return prior-map context kwargs for semantic snapshot planning."""

        if self.prior_map_runtime is None:
            return {}
        return self.prior_map_runtime.update_and_query(snapshot=snapshot, plan=plan, step=step)

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
            # readiness gate 是最后一道保护：输入未齐时 runtime 不会调用 policy 或发布 waypoint。
            return RuntimeReadiness(
                ready=False,
                reason="waiting for live inputs: " + ", ".join(missing),
                metadata={
                    "missing": missing,
                    "has_object_nodes": self.semantic_bridge.has_object_snapshot(),
                    "has_pose": self._latest_pose is not None,
                    "has_image": self._latest_image_stamp is not None,
                    "dry_run": self.dry_run,
                    "runtime_safety": self._runtime_safety_metadata(),
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
                "runtime_safety": self._runtime_safety_metadata(),
            },
        )

    def _tick(self) -> None:
        """Run one high-level runtime step and log the decision."""

        decision = self.runtime.step(self.instruction)
        # 每条决策日志都携带控制边界，便于复盘时确认有没有启用真实底盘链路。
        decision.metadata.setdefault("runtime_safety", self._runtime_safety_metadata())
        decision.lower_planner_state.setdefault("runtime_safety", self._runtime_safety_metadata())
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

    def _validate_motion_safety(self) -> None:
        """Validate live motion parameters before any publisher is created."""

        self.waypoint_topic = validate_non_velocity_publish_topic(self.waypoint_topic, role="waypoint")
        self.test_waypoint_topic = validate_non_velocity_publish_topic(
            self.test_waypoint_topic,
            role="test_waypoint",
        )
        if self.hold_topic:
            self.hold_topic = validate_non_velocity_publish_topic(self.hold_topic, role="hold")
        if self.cancel_topic:
            self.cancel_topic = validate_non_velocity_publish_topic(self.cancel_topic, role="cancel")
        if self.emergency_stop_topic:
            self.emergency_stop_topic = validate_non_velocity_publish_topic(
                self.emergency_stop_topic,
                role="emergency_stop",
            )

        if self.dry_run:
            return
        if self.lower_controller_enabled:
            if not self.controller_contract_file:
                raise RuntimeError(
                    "Refusing live waypoint publication: controller_contract_file is required"
                )
            try:
                contract = load_controller_contract(self.controller_contract_file)
                validate_controller_contract(
                    contract,
                    waypoint_topic=self.waypoint_topic,
                    world_frame=self.world_frame,
                    action_name=str(self.get_parameter("motion_action_name").value or "/strive/execute_waypoint"),
                )
            except ControllerContractError as exc:
                raise RuntimeError(f"Refusing live waypoint publication: {exc}") from exc
            return
        if _same_topic(self.waypoint_topic, self.test_waypoint_topic):
            self.get_logger().warning(
                "lower_controller_enabled=false; publishing only to configured test waypoint topic "
                f"{self.test_waypoint_topic}"
            )
            return
        raise RuntimeError(
            "Refusing live waypoint publication: dry_run=false requires lower_controller_enabled=true "
            f"or waypoint_topic:={self.test_waypoint_topic}"
        )

    def _runtime_safety_metadata(self) -> dict:
        """Return JSON-friendly safety boundary metadata for decision logs."""

        motion_output = "dry_run"
        if not self.dry_run:
            motion_output = "lower_controller" if self.lower_controller_enabled else "test_waypoint"
        return {
            "motion_output": motion_output,
            "dry_run": self.dry_run,
            "lower_controller_enabled": self.lower_controller_enabled,
            "waypoint_topic": self.waypoint_topic,
            "test_waypoint_topic": self.test_waypoint_topic,
            "hold_topic": self.hold_topic,
            "cancel_topic": self.cancel_topic,
            "emergency_stop_topic": self.emergency_stop_topic,
            "allow_emergency_stop_publish": self.allow_emergency_stop_publish,
            "controller_contract_file": self.controller_contract_file or None,
        }


def _stamp_to_seconds(stamp) -> float:
    """Convert a ROS stamp-like object to seconds."""

    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) / 1e9


def _param_bool(value) -> bool:
    """Return a robust boolean from ROS parameter values or launch strings."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _navigation_status_code(value) -> NavigationStatusCode:
    """Return a navigation status code from a launch parameter string."""

    try:
        return NavigationStatusCode(str(value or "").strip().lower())
    except ValueError:
        return NavigationStatusCode.IDLE


def _same_topic(left: str, right: str) -> bool:
    """Return whether two ROS topic strings refer to the same normalized topic."""

    return normalize_ros_topic_name(left) == normalize_ros_topic_name(right)


def main(args: Optional[list[str]] = None) -> None:
    """Run the VLN instruction runtime ROS2 node."""

    rclpy.init(args=args)
    node = StriveInstructionRuntimeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
