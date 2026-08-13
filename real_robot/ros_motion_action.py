"""ROS-independent helpers for the STRIVE motion action boundary.

The ROS node uses these helpers to convert an action goal into the existing
platform-neutral ``MotionGoal`` contract.  Keeping conversion here makes the
task protocol testable without importing ``rclpy``.
"""

from __future__ import annotations

from typing import Any

from real_robot.contracts import MotionGoal, MotionGoalMode, Pose3D


def pose3d_from_pose_stamped(msg: Any, default_frame: str = "map") -> Pose3D:
    """Convert a ROS-like ``PoseStamped`` message into ``Pose3D``."""

    header = getattr(msg, "header", None)
    pose = getattr(msg, "pose", None)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    return Pose3D(
        position=(float(getattr(position, "x", 0.0)), float(getattr(position, "y", 0.0)), float(getattr(position, "z", 0.0))),
        orientation_xyzw=(
            float(getattr(orientation, "x", 0.0)),
            float(getattr(orientation, "y", 0.0)),
            float(getattr(orientation, "z", 0.0)),
            float(getattr(orientation, "w", 1.0)),
        ),
        frame_id=str(getattr(header, "frame_id", "") or default_frame),
    )


def motion_goal_from_action_goal(goal_msg: Any, default_frame: str = "map") -> MotionGoal:
    """Convert ``ExecuteWaypoint.Goal`` into a STRIVE ``MotionGoal``."""

    target_pose = pose3d_from_pose_stamped(getattr(goal_msg, "target_pose"), default_frame)
    look_at = None
    if bool(getattr(goal_msg, "has_look_at", False)):
        point = getattr(getattr(goal_msg, "look_at", None), "point", None)
        if point is not None:
            look_at = (float(point.x), float(point.y), float(point.z))
    motion_profile = str(getattr(goal_msg, "motion_profile", "") or "").strip().lower()
    try:
        mode = MotionGoalMode(motion_profile)
    except ValueError:
        mode = MotionGoalMode.IMPROVE_VIEW if look_at is not None else MotionGoalMode.GO_TO_OBJECT
    return MotionGoal(
        mode=mode,
        goal_pose=target_pose,
        look_at=look_at,
        target_object_uid=str(getattr(goal_msg, "target_object_uid", "") or "") or None,
        anchor_object_uid=str(getattr(goal_msg, "anchor_object_uid", "") or "") or None,
        relation_edge_id=str(getattr(goal_msg, "relation_edge_id", "") or "") or None,
        tolerance={
            "xy_goal_tolerance_m": float(getattr(goal_msg, "xy_tolerance_m", 0.35)),
            "yaw_tolerance_rad": float(getattr(goal_msg, "yaw_tolerance_rad", 0.0)),
        },
        reason=str(getattr(goal_msg, "motion_profile", "") or "execute_waypoint"),
        metadata={
            "motion_profile": str(getattr(goal_msg, "motion_profile", "") or "default"),
            "timeout_s": float(getattr(goal_msg, "timeout_s", 0.0) or 0.0) or None,
        },
    )
