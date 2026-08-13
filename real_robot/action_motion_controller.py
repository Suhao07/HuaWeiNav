"""ROS2 Action client backend for the platform-neutral motion protocol."""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional

from real_robot.contracts import MotionGoal, MotionReasonCode, NavigationStatus, NavigationStatusCode, Pose3D


class RosActionMotionController:
    """Adapt ``ExecuteWaypoint`` Action feedback to ``MotionControllerProtocol``.

    The adapter never publishes ``/way_point`` or ``/cmd_vel``.  A separate
    ``SysNavMotionServer`` owns the action server and the native SysNav topic.
    """

    def __init__(
        self,
        node: Any,
        action_name: str = "/strive/execute_waypoint",
        action_type: Optional[Any] = None,
        server_wait_timeout_s: float = 0.5,
    ) -> None:
        """Initialize a non-blocking ROS2 action client.

        Args:
            node: Existing ROS2 node used for logging and callback execution.
            action_name: Fully qualified ExecuteWaypoint action name.
            action_type: Optional injected action type for tests.
            server_wait_timeout_s: Bounded wait used when the runtime submits a
                goal while the motion server is still starting.
        """

        try:
            from rclpy.action import ActionClient
            from strive_motion_msgs.action import ExecuteWaypoint
        except ImportError as exc:
            raise RuntimeError("ROS2 and strive_motion_msgs are required for RosActionMotionController") from exc
        self.node = node
        self._action_type = action_type or ExecuteWaypoint
        self._client = ActionClient(node, self._action_type, action_name)
        self.server_wait_timeout_s = max(0.0, float(server_wait_timeout_s))
        self._lock = threading.RLock()
        self._records: Dict[str, Dict[str, Any]] = {}

    def send_goal(self, goal: MotionGoal) -> str:
        """Submit a goal asynchronously and return a local tracking id."""

        if not goal.requires_motion():
            goal_id = f"action_goal:{uuid.uuid4().hex}"
            self._records[goal_id] = {"goal": goal, "status": NavigationStatus(NavigationStatusCode.IDLE, goal_id=goal_id)}
            return goal_id
        goal_id = f"action_goal:{uuid.uuid4().hex}"
        request = self._goal_message(goal)
        with self._lock:
            self._records[goal_id] = {
                "goal": goal,
                "status": NavigationStatus(NavigationStatusCode.QUEUED, goal_id=goal_id, message="action goal queued"),
                "server_handle": None,
            }
        if not self._client.wait_for_server(timeout_sec=self.server_wait_timeout_s):
            self._update(goal_id, NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message="motion action server unavailable", reason_code=MotionReasonCode.CONTROLLER_FAULT))
            return goal_id
        future = self._client.send_goal_async(request, feedback_callback=lambda msg: self._feedback_callback(goal_id, msg))
        future.add_done_callback(lambda done: self._goal_response_callback(goal_id, done))
        return goal_id

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return the latest feedback/result snapshot for one local goal id."""

        with self._lock:
            record = self._records.get(goal_id)
            if record is None:
                return NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message="unknown action goal", reason_code=MotionReasonCode.UNKNOWN_GOAL)
            return record["status"]

    def cancel(self, goal_id: Optional[str] = None) -> None:
        """Request cancellation of one goal or all locally tracked goals."""

        ids = [goal_id] if goal_id else list(self._records)
        for current_id in ids:
            record = self._records.get(current_id)
            if not record:
                continue
            handle = record.get("server_handle")
            if handle is not None:
                handle.cancel_goal_async()

    def hold(self) -> None:
        """Cancel active action goals and rely on the server's safe hold path."""

        self.cancel()

    def _goal_message(self, goal: MotionGoal) -> Any:
        """Build an ExecuteWaypoint goal message from a MotionGoal."""

        request = self._action_type.Goal()
        request.target_pose = _pose_stamped(goal.goal_pose)
        request.has_look_at = goal.look_at is not None
        if goal.look_at is not None:
            request.look_at.header.frame_id = goal.goal_pose.frame_id if goal.goal_pose else "map"
            request.look_at.point.x, request.look_at.point.y, request.look_at.point.z = goal.look_at
        request.xy_tolerance_m = float(goal.tolerance.get("xy_goal_tolerance_m", 0.35))
        request.yaw_tolerance_rad = float(goal.tolerance.get("yaw_tolerance_rad", 0.0))
        request.timeout_s = float(goal.metadata.get("timeout_s") or 0.0)
        request.motion_profile = str(goal.metadata.get("motion_profile", goal.mode.value))
        request.target_object_uid = str(goal.target_object_uid or "")
        request.anchor_object_uid = str(goal.anchor_object_uid or "")
        request.relation_edge_id = str(goal.relation_edge_id or "")
        return request

    def _goal_response_callback(self, goal_id: str, future: Any) -> None:
        """Store the accepted server handle and register its result callback."""

        try:
            handle = future.result()
        except Exception as exc:  # pragma: no cover - ROS transport failure
            self._update(goal_id, NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message=str(exc), reason_code=MotionReasonCode.CONTROLLER_FAULT))
            return
        if not handle.accepted:
            self._update(goal_id, NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message="motion action rejected", reason_code=MotionReasonCode.INVALID_GOAL))
            return
        with self._lock:
            if goal_id in self._records:
                self._records[goal_id]["server_handle"] = handle
                self._records[goal_id]["status"] = NavigationStatus(NavigationStatusCode.RUNNING, goal_id=goal_id, message="motion action accepted")
        handle.get_result_async().add_done_callback(lambda done: self._result_callback(goal_id, done))

    def _feedback_callback(self, goal_id: str, feedback_msg: Any) -> None:
        """Translate action feedback into the shared navigation status."""

        feedback = getattr(feedback_msg, "feedback", feedback_msg)
        state = int(getattr(feedback, "state", 1))
        if state in {0, 1, 2, 5}:
            status = NavigationStatusCode.RUNNING
            reason = MotionReasonCode.NONE
        elif state == 4:
            status = NavigationStatusCode.MANUAL_TAKEOVER
            reason = MotionReasonCode.MANUAL_TAKEOVER
        else:
            status = NavigationStatusCode.SAFETY_STOP
            reason = MotionReasonCode.COMMAND_STALE
        self._update(
            goal_id,
            NavigationStatus(
                status,
                goal_id=goal_id,
                distance_to_goal=float(getattr(feedback, "distance_remaining_m", 0.0)),
                path_length_remaining=float(getattr(feedback, "path_length_remaining_m", 0.0)),
                progress=float(getattr(feedback, "progress", 0.0)),
                message="motion action feedback",
                safety_state=str(getattr(feedback, "safety_state", "unknown")),
                reason_code=reason,
                metadata={
                    "path_available": bool(getattr(feedback, "path_valid", False)),
                    "speed_mps": float(getattr(feedback, "speed_mps", 0.0)),
                    "settled_at_goal": bool(getattr(feedback, "settled_at_goal", False)),
                    "view_aligned": bool(getattr(feedback, "view_aligned", False)),
                    "alignment_reason": str(getattr(feedback, "alignment_reason", "") or ""),
                },
            ),
        )

    def _result_callback(self, goal_id: str, future: Any) -> None:
        """Translate the action result into a terminal navigation status."""

        try:
            wrapped = future.result()
            result = wrapped.result
        except Exception as exc:  # pragma: no cover - ROS transport failure
            self._update(goal_id, NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message=str(exc), reason_code=MotionReasonCode.CONTROLLER_FAULT))
            return
        mapping = {
            0: NavigationStatusCode.REACHED,
            1: NavigationStatusCode.BLOCKED,
            2: NavigationStatusCode.TIMEOUT,
            3: NavigationStatusCode.PREEMPTED,
            5: NavigationStatusCode.SAFETY_STOP,
            6: NavigationStatusCode.MANUAL_TAKEOVER,
            7: NavigationStatusCode.LOCALIZATION_LOST,
        }
        status = mapping.get(int(result.outcome), NavigationStatusCode.FAILED)
        try:
            reason = MotionReasonCode(str(result.reason_code))
        except ValueError:
            reason = MotionReasonCode.CONTROLLER_FAULT if status == NavigationStatusCode.FAILED else MotionReasonCode.NONE
        pose = _pose_from_pose_stamped(getattr(result, "final_pose", None))
        self._update(
            goal_id,
            NavigationStatus(
                status,
                goal_id=goal_id,
                current_pose=pose,
                message=str(result.message),
                reason_code=reason,
                metadata={
                    "travelled_distance_m": float(result.travelled_distance_m),
                    "view_aligned": bool(getattr(result, "view_aligned", False)),
                    "alignment_reason": str(getattr(result, "alignment_reason", "") or ""),
                },
            ),
        )

    def _update(self, goal_id: str, status: NavigationStatus) -> None:
        """Update a local status record if its action id is still active."""

        with self._lock:
            if goal_id in self._records:
                self._records[goal_id]["status"] = status


def _pose_stamped(pose: Optional[Pose3D]) -> Any:
    """Build a geometry PoseStamped message."""

    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    if pose is None:
        return msg
    msg.header.frame_id = pose.frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose.position
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = pose.orientation_xyzw
    return msg


def _pose_from_pose_stamped(msg: Any) -> Optional[Pose3D]:
    """Convert an optional ROS PoseStamped result into Pose3D."""

    if msg is None or not hasattr(msg, "pose"):
        return None
    return Pose3D(
        position=(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
        orientation_xyzw=(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
        frame_id=msg.header.frame_id,
    )
