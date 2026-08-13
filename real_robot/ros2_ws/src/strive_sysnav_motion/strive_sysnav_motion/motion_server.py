"""Task-level ROS2 action server around the native SysNav waypoint chain.

The server is the single owner of ``/way_point`` for an action-backed runtime.
SysNav remains responsible for local path generation and path following; this
node adds goal lifecycle, cancellation, feedback, timeout, and reason codes.
"""

from __future__ import annotations

import threading
import time
import math
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from strive_motion_msgs.action import ExecuteWaypoint
from strive_motion_msgs.action import AlignView
from strive_motion_msgs.msg import SafetyState as SafetyStateMessage

from real_robot.contracts import MotionGoal, MotionReasonCode, NavigationStatusCode
from real_robot.control.controller_contract import (
    ControllerContractError,
    load_controller_contract,
    validate_controller_contract,
)
from real_robot.ros_motion_action import motion_goal_from_action_goal
from strive_sysnav_motion.ros_messages import replace_message
from real_robot.sysnav_ros_adapters import RosNavigationStatusProvider, RosWaypointController


class SysNavMotionServer(Node):
    """Expose SysNav waypoint execution through ``ExecuteWaypoint``."""

    def __init__(self) -> None:
        """Create the action server and subscribe to lower-layer state."""

        super().__init__("sysnav_motion_server")
        self.declare_parameter("action_name", "/strive/execute_waypoint")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("planner_status_topic", "/local_planner/status")
        self.declare_parameter("hold_topic", "/platform/safe_hold")
        self.declare_parameter("cancel_topic", "/local_planner/cancel")
        self.declare_parameter("safety_state_topic", "/platform/safety_state")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("xy_goal_tolerance_m", 0.35)
        self.declare_parameter("z_goal_tolerance_m", 1.0)
        self.declare_parameter("navigation_timeout_s", 60.0)
        self.declare_parameter("no_progress_timeout_s", 12.0)
        self.declare_parameter("min_progress_delta_m", 0.05)
        self.declare_parameter("path_stale_timeout_s", 5.0)
        self.declare_parameter("velocity_tolerance_mps", 0.08)
        self.declare_parameter("stable_reach_time_s", 0.2)
        self.declare_parameter("allow_look_at", False)
        self.declare_parameter("alignment_action_name", "/strive/align_view")
        self.declare_parameter("alignment_server_wait_timeout_s", 0.5)
        self.declare_parameter("tf_lookup_timeout_s", 0.2)
        self.declare_parameter("controller_contract_file", "")
        self.declare_parameter("require_controller_contract", True)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.controller_contract_file = str(self.get_parameter("controller_contract_file").value or "")
        self.require_controller_contract = _as_bool(self.get_parameter("require_controller_contract").value)
        self._validate_controller_contract()
        self.allow_look_at = _as_bool(self.get_parameter("allow_look_at").value)
        self.alignment_action_name = str(self.get_parameter("alignment_action_name").value)
        self.tf_lookup_timeout_s = max(0.0, float(self.get_parameter("tf_lookup_timeout_s").value))
        self._alignment_client = (
            ActionClient(self, AlignView, self.alignment_action_name)
            if self.allow_look_at and self.alignment_action_name
            else None
        )
        self._tf_buffer = None
        self._tf_listener = None
        try:
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        except ImportError:
            # The action server still works in minimal offline ROS images; a
            # non-native frame then fails explicitly during goal execution.
            self.get_logger().warning("tf2_ros unavailable; only native goal frames are executable")
        self.status_provider = RosNavigationStatusProvider(
            xy_tolerance_m=float(self.get_parameter("xy_goal_tolerance_m").value),
            z_tolerance_m=float(self.get_parameter("z_goal_tolerance_m").value),
            timeout_s=float(self.get_parameter("navigation_timeout_s").value),
            no_progress_timeout_s=float(self.get_parameter("no_progress_timeout_s").value),
            min_progress_delta_m=float(self.get_parameter("min_progress_delta_m").value),
            path_stale_timeout_s=float(self.get_parameter("path_stale_timeout_s").value),
            velocity_tolerance_mps=float(self.get_parameter("velocity_tolerance_mps").value),
            stable_reach_time_s=float(self.get_parameter("stable_reach_time_s").value),
            world_frame=self.world_frame,
            now_fn=time.monotonic,
        )
        self.controller = RosWaypointController(
            node=self,
            waypoint_topic=str(self.get_parameter("waypoint_topic").value),
            world_frame=self.world_frame,
            status_provider=self.status_provider,
            hold_topic=str(self.get_parameter("hold_topic").value or ""),
            cancel_topic=str(self.get_parameter("cancel_topic").value or ""),
        )
        self.status_provider.create_ros_subscriptions(
            self,
            odom_topic=str(self.get_parameter("odom_topic").value),
            path_topic=str(self.get_parameter("path_topic").value),
            planner_status_topic=str(self.get_parameter("planner_status_topic").value or ""),
        )
        self.create_subscription(
            SafetyStateMessage,
            str(self.get_parameter("safety_state_topic").value or ""),
            self.status_provider.update_safety_state,
            10,
        )
        self._callback_group = ReentrantCallbackGroup()
        self._active_goal_lock = threading.Lock()
        self._active_goal_handle: Optional[Any] = None
        self._server = ActionServer(
            self,
            ExecuteWaypoint,
            str(self.get_parameter("action_name").value),
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "SysNav motion action server ready: "
            f"action={self.get_parameter('action_name').value}, "
            f"waypoint={self.get_parameter('waypoint_topic').value}, "
            "controller ownership remains below this node"
        )

    def _validate_controller_contract(self) -> None:
        """Fail closed before creating a live motion action server."""

        if not self.require_controller_contract:
            return
        try:
            contract = load_controller_contract(self.controller_contract_file)
            validate_controller_contract(
                contract,
                waypoint_topic=str(self.get_parameter("waypoint_topic").value),
                world_frame=self.world_frame,
                action_name=str(self.get_parameter("action_name").value),
            )
        except ControllerContractError as exc:
            raise RuntimeError(f"motion server contract gate failed: {exc}") from exc

    def _goal_callback(self, goal_request: Any) -> GoalResponse:
        """Accept one finite, frame-consistent goal when the server is idle."""

        target_pose = getattr(goal_request, "target_pose", None)
        if target_pose is None or getattr(target_pose, "header", None) is None:
            self.get_logger().warning("rejecting action goal without target_pose")
            return GoalResponse.REJECT
        frame_id = str(getattr(target_pose.header, "frame_id", "") or "")
        if not frame_id:
            self.get_logger().warning("rejecting action goal with empty target frame")
            return GoalResponse.REJECT
        position = getattr(target_pose, "pose", None)
        position = getattr(position, "position", None)
        coordinates = (getattr(position, "x", None), getattr(position, "y", None), getattr(position, "z", None))
        if any(value is None or not math.isfinite(float(value)) for value in coordinates):
            self.get_logger().warning("rejecting action goal with non-finite target coordinates")
            return GoalResponse.REJECT
        if (
            float(getattr(goal_request, "xy_tolerance_m", 0.0)) < 0
            or float(getattr(goal_request, "yaw_tolerance_rad", 0.0)) < 0
            or float(getattr(goal_request, "timeout_s", 0.0)) < 0
        ):
            self.get_logger().warning("rejecting action goal with negative tolerance or timeout")
            return GoalResponse.REJECT
        if bool(getattr(goal_request, "has_look_at", False)) and not self.allow_look_at:
            self.get_logger().warning("rejecting look_at goal: alignment backend is not approved")
            return GoalResponse.REJECT
        with self._active_goal_lock:
            if self._active_goal_handle is not None:
                self.get_logger().warning("rejecting action goal while another goal is active")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle: Any) -> CancelResponse:
        """Accept cancellation so the execute callback can issue safe hold."""

        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle: Any) -> Any:
        """Execute one goal, publish feedback, and return one terminal result."""

        with self._active_goal_lock:
            if self._active_goal_handle is not None:
                result = ExecuteWaypoint.Result()
                return self._finish_abort(goal_handle, result, MotionReasonCode.CONTROLLER_FAULT, "another motion goal is active")
            self._active_goal_handle = goal_handle

        try:
            request = goal_handle.request
            normalized_request = self._normalize_goal_request(request)
            if normalized_request is None:
                result = ExecuteWaypoint.Result()
                self._fill_result(
                    result,
                    NavigationStatusCode.LOCALIZATION_LOST,
                    MotionReasonCode.LOCALIZATION_LOST,
                    "unable to transform goal into the configured world frame",
                )
                goal_handle.abort()
                return result
            motion_goal = motion_goal_from_action_goal(normalized_request, self.world_frame)
            if motion_goal.look_at is not None and not self.allow_look_at:
                return self._finish_abort(goal_handle, ExecuteWaypoint.Result(), MotionReasonCode.LOOK_AT_UNSUPPORTED, "look_at backend is not approved")

            goal_id = self.controller.send_goal(motion_goal)
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.controller.cancel(goal_id)
                    result = ExecuteWaypoint.Result()
                    self._fill_result(result, NavigationStatusCode.PREEMPTED, MotionReasonCode.CANCELLED, "goal cancelled")
                    goal_handle.canceled()
                    return result

                status = self.controller.poll_status(goal_id)
                self._publish_feedback(goal_handle, status)
                if status.is_terminal():
                    if status.status == NavigationStatusCode.REACHED and motion_goal.look_at is not None:
                        alignment = self._align_view(goal_handle, normalized_request, status)
                        if alignment[0] != AlignView.Result.ALIGNED:
                            return self._finish_alignment_failure(goal_handle, alignment)
                        return self._finish_from_status(
                            goal_handle,
                            status,
                            view_aligned=True,
                            alignment_reason=alignment[1],
                        )
                    return self._finish_from_status(goal_handle, status)
                time.sleep(0.05)

            self.controller.hold()
            result = ExecuteWaypoint.Result()
            self._fill_result(result, NavigationStatusCode.FAILED, MotionReasonCode.CONTROLLER_FAULT, "ROS shutdown during motion")
            goal_handle.abort()
            return result
        finally:
            with self._active_goal_lock:
                self._active_goal_handle = None

    def _finish_from_status(
        self,
        goal_handle: Any,
        status: Any,
        *,
        view_aligned: bool = False,
        alignment_reason: str = "",
    ) -> Any:
        """Map a platform-neutral status to an action result."""

        result = ExecuteWaypoint.Result()
        reason = status.reason_code if isinstance(status.reason_code, MotionReasonCode) else MotionReasonCode.NONE
        if status.status == NavigationStatusCode.REACHED:
            outcome = ExecuteWaypoint.Result.REACHED
            reason = MotionReasonCode.GOAL_REACHED
            goal_handle.succeed()
        elif status.status == NavigationStatusCode.BLOCKED:
            outcome = ExecuteWaypoint.Result.BLOCKED
            reason = reason if reason != MotionReasonCode.NONE else MotionReasonCode.NO_PROGRESS
            goal_handle.abort()
        elif status.status == NavigationStatusCode.TIMEOUT:
            outcome = ExecuteWaypoint.Result.TIMEOUT
            reason = MotionReasonCode.GOAL_TIMEOUT
            goal_handle.abort()
        elif status.status == NavigationStatusCode.PREEMPTED:
            outcome = ExecuteWaypoint.Result.PREEMPTED
            reason = MotionReasonCode.CANCELLED
            goal_handle.canceled()
        elif status.status == NavigationStatusCode.SAFETY_STOP:
            outcome = ExecuteWaypoint.Result.SAFETY_STOP
            reason = reason if reason != MotionReasonCode.NONE else MotionReasonCode.ESTOP_ACTIVE
            goal_handle.abort()
        elif status.status == NavigationStatusCode.MANUAL_TAKEOVER:
            outcome = ExecuteWaypoint.Result.MANUAL_TAKEOVER
            reason = MotionReasonCode.MANUAL_TAKEOVER
            goal_handle.abort()
        elif status.status == NavigationStatusCode.LOCALIZATION_LOST:
            outcome = ExecuteWaypoint.Result.LOCALIZATION_LOST
            reason = MotionReasonCode.LOCALIZATION_LOST
            goal_handle.abort()
        else:
            outcome = ExecuteWaypoint.Result.FAILED
            reason = MotionReasonCode.CONTROLLER_FAULT
            goal_handle.abort()
        self._fill_result(
            result,
            status.status,
            reason,
            status.message,
            outcome=outcome,
            status_obj=status,
            view_aligned=view_aligned,
            alignment_reason=alignment_reason,
        )
        return result

    def _normalize_goal_request(self, request: Any) -> Optional[Any]:
        """Transform target and look-at messages into ``world_frame``.

        TF conversion is deliberately performed at the motion boundary. The
        semantic planner therefore remains frame-agnostic, while the lower
        SysNav planner receives one explicit map-frame waypoint.
        """

        target_pose = getattr(request, "target_pose", None)
        if target_pose is None:
            return None
        transformed_pose = self._transform_pose(target_pose)
        if transformed_pose is None:
            return None
        transformed_look_at = None
        if bool(getattr(request, "has_look_at", False)):
            transformed_look_at = self._transform_point(getattr(request, "look_at", None))
            if transformed_look_at is None:
                return None
        normalized = replace_message(request)
        normalized.target_pose = transformed_pose
        if transformed_look_at is not None:
            normalized.look_at = transformed_look_at
        return normalized

    def _transform_pose(self, pose: Any) -> Optional[Any]:
        """Return a PoseStamped in the configured world frame."""

        frame_id = str(getattr(getattr(pose, "header", None), "frame_id", "") or "")
        if frame_id == self.world_frame:
            return pose
        if self._tf_buffer is None:
            return None
        try:
            from rclpy.duration import Duration

            return self._tf_buffer.transform(
                pose,
                self.world_frame,
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except Exception as exc:  # pragma: no cover - depends on live TF graph
            self.get_logger().warning(f"goal pose TF transform failed: {exc}")
            return None

    def _transform_point(self, point: Any) -> Optional[Any]:
        """Return a PointStamped in the configured world frame."""

        if point is None:
            return None
        frame_id = str(getattr(getattr(point, "header", None), "frame_id", "") or "")
        if frame_id == self.world_frame:
            return point
        if self._tf_buffer is None:
            return None
        try:
            from rclpy.duration import Duration

            return self._tf_buffer.transform(
                point,
                self.world_frame,
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )
        except Exception as exc:  # pragma: no cover - depends on live TF graph
            self.get_logger().warning(f"look_at TF transform failed: {exc}")
            return None

    def _align_view(self, goal_handle: Any, request: Any, status: Any) -> tuple[int, str]:
        """Run the external view-alignment action after position is reached."""

        if self._alignment_client is None:
            return ExecuteWaypoint.Result.FAILED, MotionReasonCode.VIEW_ALIGNMENT_UNAVAILABLE.value
        if not self._alignment_client.wait_for_server(
            timeout_sec=float(self.get_parameter("alignment_server_wait_timeout_s").value)
        ):
            return ExecuteWaypoint.Result.FAILED, MotionReasonCode.VIEW_ALIGNMENT_UNAVAILABLE.value

        # 位置到达后，视角对齐是一个独立的异步子任务。显式发布该状态，
        # 让上层能够区分“还在移动”和“正在对齐相机/云台”。
        alignment_feedback = ExecuteWaypoint.Feedback()
        alignment_feedback.state = ExecuteWaypoint.Feedback.ALIGNING
        alignment_feedback.reason_code = MotionReasonCode.NONE.value
        alignment_feedback.safety_state = str(status.safety_state or "unknown")
        alignment_feedback.distance_remaining_m = float(status.distance_to_goal or 0.0)
        alignment_feedback.path_length_remaining_m = float(status.path_length_remaining or 0.0)
        alignment_feedback.progress = float(status.progress or 0.0)
        alignment_feedback.path_valid = bool((status.metadata or {}).get("path_available", False))
        alignment_feedback.settled_at_goal = True
        if status.current_pose is not None:
            alignment_feedback.current_pose = _pose_stamped_from_pose(status.current_pose)
        goal_handle.publish_feedback(alignment_feedback)

        request_msg = AlignView.Goal()
        request_msg.target_pose = request.target_pose
        request_msg.has_look_at = bool(getattr(request, "has_look_at", False))
        request_msg.look_at = request.look_at
        request_msg.yaw_tolerance_rad = float(getattr(request, "yaw_tolerance_rad", 0.0))
        request_msg.timeout_s = float(getattr(request, "timeout_s", 0.0))
        request_msg.target_object_uid = str(getattr(request, "target_object_uid", "") or "")
        request_msg.anchor_object_uid = str(getattr(request, "anchor_object_uid", "") or "")
        event = threading.Event()
        result_box: dict[str, Any] = {}
        goal_future = self._alignment_client.send_goal_async(request_msg)

        def on_goal_response(future: Any) -> None:
            try:
                handle = future.result()
                result_box["handle"] = handle
                if not handle.accepted:
                    result_box["outcome"] = AlignView.Result.FAILED
                    result_box["reason"] = MotionReasonCode.VIEW_ALIGNMENT_FAILED.value
                    event.set()
                    return
                handle.get_result_async().add_done_callback(on_result)
            except Exception as exc:  # pragma: no cover - live ROS transport
                result_box["outcome"] = AlignView.Result.FAILED
                result_box["reason"] = str(exc)
                event.set()

        def on_result(future: Any) -> None:
            try:
                wrapped = future.result()
                result = wrapped.result
                result_box["outcome"] = int(result.outcome)
                result_box["reason"] = str(result.reason_code or result.message or "")
            except Exception as exc:  # pragma: no cover - live ROS transport
                result_box["outcome"] = AlignView.Result.FAILED
                result_box["reason"] = str(exc)
            event.set()

        goal_future.add_done_callback(on_goal_response)
        timeout_s = max(0.0, float(getattr(request, "timeout_s", 0.0) or 0.0))
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while not event.wait(0.05):
            if goal_handle.is_cancel_requested:
                handle = result_box.get("handle")
                if handle is not None:
                    handle.cancel_goal_async()
                return ExecuteWaypoint.Result.PREEMPTED, MotionReasonCode.CANCELLED.value
            if deadline is not None and time.monotonic() >= deadline:
                handle = result_box.get("handle")
                if handle is not None:
                    handle.cancel_goal_async()
                return ExecuteWaypoint.Result.TIMEOUT, MotionReasonCode.GOAL_TIMEOUT.value
        outcome = int(result_box.get("outcome", AlignView.Result.FAILED))
        reason = str(result_box.get("reason", ""))
        return outcome, reason

    def _finish_alignment_failure(self, goal_handle: Any, alignment: tuple[int, str]) -> Any:
        """Map an alignment action terminal result into ExecuteWaypoint."""

        outcome, reason = alignment
        result = ExecuteWaypoint.Result()
        if outcome == AlignView.Result.PREEMPTED:
            goal_handle.canceled()
            status = NavigationStatusCode.PREEMPTED
            reason_code = MotionReasonCode.CANCELLED
        elif outcome == AlignView.Result.TIMEOUT:
            goal_handle.abort()
            status = NavigationStatusCode.TIMEOUT
            reason_code = MotionReasonCode.GOAL_TIMEOUT
        elif outcome == AlignView.Result.BLOCKED:
            goal_handle.abort()
            status = NavigationStatusCode.BLOCKED
            reason_code = MotionReasonCode.VIEW_ALIGNMENT_FAILED
        else:
            goal_handle.abort()
            status = NavigationStatusCode.FAILED
            reason_code = MotionReasonCode.VIEW_ALIGNMENT_FAILED
        self._fill_result(
            result,
            status,
            reason_code,
            reason or "view alignment failed",
            outcome={
                NavigationStatusCode.PREEMPTED: ExecuteWaypoint.Result.PREEMPTED,
                NavigationStatusCode.TIMEOUT: ExecuteWaypoint.Result.TIMEOUT,
                NavigationStatusCode.BLOCKED: ExecuteWaypoint.Result.BLOCKED,
            }.get(status, ExecuteWaypoint.Result.FAILED),
            view_aligned=False,
            alignment_reason=reason,
        )
        return result

    def _finish_abort(self, goal_handle: Any, result: Any, reason: MotionReasonCode, message: str) -> Any:
        """Abort an invalid action request with a stable reason code."""

        self._fill_result(result, NavigationStatusCode.FAILED, reason, message)
        goal_handle.abort()
        return result

    def _publish_feedback(self, goal_handle: Any, status: Any) -> None:
        """Publish status feedback without exposing ROS internals upstream."""

        feedback = ExecuteWaypoint.Feedback()
        if status.status == NavigationStatusCode.QUEUED:
            feedback.state = ExecuteWaypoint.Feedback.PLANNING
        elif str(status.safety_state or "").lower() == "hold":
            feedback.state = ExecuteWaypoint.Feedback.HOLDING
        elif str(status.safety_state or "").lower() in {"manual_takeover", "estop", "stale_input", "controller_fault"}:
            feedback.state = (
                ExecuteWaypoint.Feedback.MANUAL_TAKEOVER
                if str(status.safety_state).lower() == "manual_takeover"
                else ExecuteWaypoint.Feedback.SAFETY_STOP
            )
        else:
            feedback.state = ExecuteWaypoint.Feedback.TRACKING
        feedback.distance_remaining_m = float(status.distance_to_goal or 0.0)
        feedback.path_length_remaining_m = float(status.path_length_remaining or 0.0)
        feedback.progress = float(status.progress or 0.0)
        feedback.path_valid = bool(status.metadata.get("path_available", False))
        feedback.safety_state = str(status.safety_state or "unknown")
        feedback.reason_code = _reason_value(status.reason_code)
        feedback.speed_mps = float((status.metadata or {}).get("speed_mps", 0.0))
        feedback.settled_at_goal = bool(
            status.status == NavigationStatusCode.REACHED
            or (status.metadata or {}).get("settled_since") is not None
            and (status.metadata or {}).get("speed_mps", 0.0) <= 0.08
        )
        if status.current_pose is not None:
            feedback.current_pose = _pose_stamped_from_pose(status.current_pose)
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _fill_result(
        result: Any,
        status: NavigationStatusCode,
        reason: MotionReasonCode,
        message: str,
        *,
        outcome: Optional[int] = None,
        status_obj: Any = None,
        view_aligned: bool = False,
        alignment_reason: str = "",
    ) -> None:
        """Populate a result message with outcome and final pose."""

        if outcome is None:
            outcome = {
                NavigationStatusCode.REACHED: ExecuteWaypoint.Result.REACHED,
                NavigationStatusCode.BLOCKED: ExecuteWaypoint.Result.BLOCKED,
                NavigationStatusCode.TIMEOUT: ExecuteWaypoint.Result.TIMEOUT,
                NavigationStatusCode.PREEMPTED: ExecuteWaypoint.Result.PREEMPTED,
                NavigationStatusCode.SAFETY_STOP: ExecuteWaypoint.Result.SAFETY_STOP,
                NavigationStatusCode.MANUAL_TAKEOVER: ExecuteWaypoint.Result.MANUAL_TAKEOVER,
                NavigationStatusCode.LOCALIZATION_LOST: ExecuteWaypoint.Result.LOCALIZATION_LOST,
            }.get(status, ExecuteWaypoint.Result.FAILED)
        result.outcome = int(outcome)
        result.reason_code = reason.value
        result.message = str(message or "")
        result.view_aligned = bool(view_aligned)
        result.alignment_reason = str(alignment_reason or "")
        result.travelled_distance_m = float((status_obj.metadata or {}).get("travelled_distance_m", 0.0)) if status_obj else 0.0
        if status_obj is not None and status_obj.current_pose is not None:
            result.final_pose = _pose_stamped_from_pose(status_obj.current_pose)


def _pose_stamped_from_pose(pose: Any) -> PoseStamped:
    """Convert a platform-neutral pose to a ROS PoseStamped message."""

    msg = PoseStamped()
    msg.header.frame_id = pose.frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose.position
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = pose.orientation_xyzw
    return msg


def _reason_value(reason: Any) -> str:
    """Return a string reason code from enum or string input."""

    return reason.value if isinstance(reason, MotionReasonCode) else str(reason or MotionReasonCode.NONE.value)


def _as_bool(value: Any) -> bool:
    """Parse ROS bool/string parameters."""

    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(args: Optional[list[str]] = None) -> None:
    """Run the SysNav task-level motion action server."""

    rclpy.init(args=args)
    node = SysNavMotionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.controller.hold()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
