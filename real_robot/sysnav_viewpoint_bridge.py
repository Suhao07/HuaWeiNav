"""Pure data model for joining SysNav viewpoint records with odometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from real_robot.contracts import Pose3D


@dataclass(frozen=True)
class SysNavViewpointRecord:
    """Resolved viewpoint record ready for ROS serialization.

    Args:
        viewpoint_id: Stable integer ID assigned by SysNav.
        timestamp: SysNav viewpoint header timestamp.
        pose: Robot pose sampled in the viewpoint frame.
        observed_object_ids: SysNav object IDs directly observed at this view.
        timestamp_ns: Original viewpoint timestamp in integer nanoseconds.
    """

    viewpoint_id: int
    timestamp: float
    pose: Pose3D
    observed_object_ids: Tuple[int, ...] = ()
    timestamp_ns: Optional[int] = None


class SysNavViewpointBridgeModel:
    """Join ``ViewpointRep`` headers, odometry, and direct object updates.

    SysNav creates a viewpoint representation at the current robot position
    and publishes the state-estimation timestamp in ``ViewpointRep.header``.
    This model uses the nearest odometry sample within a configured time
    tolerance. It never creates a pose from an object centroid or a room
    centroid.
    """

    def __init__(self, max_time_offset_s: float = 0.25, odom_history_size: int = 400) -> None:
        """Initialize the timestamp joiner.

        Args:
            max_time_offset_s: Maximum allowed viewpoint/odometry time gap.
            odom_history_size: Number of recent odometry samples to retain.

        Raises:
            ValueError: If either retention or tolerance is negative/invalid.
        """

        if max_time_offset_s < 0.0:
            raise ValueError("max_time_offset_s must be non-negative")
        if odom_history_size <= 0:
            raise ValueError("odom_history_size must be positive")
        self.max_time_offset_s = float(max_time_offset_s)
        self._odom_history: deque[Tuple[int, Any]] = deque(maxlen=int(odom_history_size))
        self._viewpoint_messages: Dict[int, Any] = {}
        self._object_ids_by_viewpoint: Dict[int, set[int]] = {}
        self._pending_viewpoint_ids: set[int] = set()
        self._emitted_signatures: Dict[int, Tuple[int, Tuple[int, ...]]] = {}

    def update_odometry(self, msg: Any) -> Tuple[SysNavViewpointRecord, ...]:
        """Add odometry and return viewpoint records newly made resolvable.

        Args:
            msg: ROS-like ``nav_msgs/Odometry`` message.

        Returns:
            Newly resolved viewpoint records.
        """

        self._odom_history.append((_stamp_ns(msg), msg))
        return self._resolve_pending()

    def update_viewpoint(self, msg: Any) -> Tuple[SysNavViewpointRecord, ...]:
        """Add one SysNav viewpoint header and resolve it when possible.

        Args:
            msg: ROS-like ``tare_planner/ViewpointRep`` message.

        Returns:
            A one-element tuple when the pose is available, otherwise empty.

        Raises:
            ValueError: If the viewpoint ID is negative.
        """

        viewpoint_id = int(getattr(msg, "viewpoint_id", -1))
        if viewpoint_id < 0:
            raise ValueError("SysNav viewpoint ID must be non-negative")
        self._viewpoint_messages[viewpoint_id] = msg
        self._pending_viewpoint_ids.add(viewpoint_id)
        return self._resolve_pending()

    def update_object_nodes(self, msg: Any) -> Tuple[SysNavViewpointRecord, ...]:
        """Accumulate direct object observations and refresh affected views.

        Args:
            msg: ROS-like ``tare_planner/ObjectNodeList`` message.

        Returns:
            Updated viewpoint records whose observed-object set changed.
        """

        changed: set[int] = set()
        for node in getattr(msg, "nodes", ()):
            viewpoint_id = int(getattr(node, "viewpoint_id", -1))
            if viewpoint_id < 0:
                continue
            object_ids = {int(value) for value in getattr(node, "object_id", ())}
            if not object_ids:
                continue
            current = self._object_ids_by_viewpoint.setdefault(viewpoint_id, set())
            before = set(current)
            current.update(object_ids)
            if current != before:
                changed.add(viewpoint_id)
                self._pending_viewpoint_ids.add(viewpoint_id)
        return self._resolve_pending(only=changed)

    def _resolve_pending(self, only: Optional[set[int]] = None) -> Tuple[SysNavViewpointRecord, ...]:
        """Resolve pending IDs against the closest valid odometry samples."""

        candidate_ids = set(self._pending_viewpoint_ids if only is None else only)
        resolved = []
        for viewpoint_id in sorted(candidate_ids):
            msg = self._viewpoint_messages.get(viewpoint_id)
            if msg is None or not self._odom_history:
                continue
            viewpoint_stamp_ns = _stamp_ns(msg)
            odom_stamp_ns, odom_msg = min(
                self._odom_history,
                key=lambda item: abs(item[0] - viewpoint_stamp_ns),
            )
            if abs(odom_stamp_ns - viewpoint_stamp_ns) > int(self.max_time_offset_s * 1e9):
                continue

            viewpoint_frame = _frame_id(msg)
            odom_frame = _frame_id(odom_msg)
            if viewpoint_frame and odom_frame and viewpoint_frame != odom_frame:
                continue

            object_ids = tuple(sorted(self._object_ids_by_viewpoint.get(viewpoint_id, set())))
            signature = (viewpoint_stamp_ns, object_ids)
            if self._emitted_signatures.get(viewpoint_id) == signature:
                self._pending_viewpoint_ids.discard(viewpoint_id)
                continue

            pose = _pose_from_odometry(odom_msg, viewpoint_frame or odom_frame or "map")
            resolved.append(
                SysNavViewpointRecord(
                    viewpoint_id=viewpoint_id,
                    timestamp=viewpoint_stamp_ns / 1e9,
                    pose=pose,
                    observed_object_ids=object_ids,
                    timestamp_ns=viewpoint_stamp_ns,
                )
            )
            self._emitted_signatures[viewpoint_id] = signature
            self._pending_viewpoint_ids.discard(viewpoint_id)
        return tuple(resolved)


def _stamp(msg: Any) -> float:
    """Return seconds from a ROS-like header."""

    return _stamp_ns(msg) / 1e9


def _stamp_ns(msg: Any) -> int:
    """Return the exact integer nanosecond timestamp from a ROS-like header."""

    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    return sec * 1_000_000_000 + nanosec


def _frame_id(msg: Any) -> str:
    """Return a normalized frame ID from a ROS-like message."""

    return str(getattr(getattr(msg, "header", None), "frame_id", "") or "")


def _pose_from_odometry(msg: Any, default_frame: str) -> Pose3D:
    """Convert a ROS-like odometry message to the platform-neutral pose."""

    pose_with_covariance = getattr(msg, "pose", None)
    pose = getattr(pose_with_covariance, "pose", pose_with_covariance)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    return Pose3D(
        position=(float(position.x), float(position.y), float(position.z)),
        orientation_xyzw=(
            float(getattr(orientation, "x", 0.0)),
            float(getattr(orientation, "y", 0.0)),
            float(getattr(orientation, "z", 0.0)),
            float(getattr(orientation, "w", 1.0)),
        ),
        frame_id=_frame_id(msg) or default_frame,
        stamp=_stamp(msg),
    )
