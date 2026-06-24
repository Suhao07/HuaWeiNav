"""Adapters for reusing SysNav ROS detection, mapping, and waypoint topics.

The adapters in this module are intentionally thin. They translate SysNav ROS
messages into STRIVE real-robot contracts and translate STRIVE motion goals
back to SysNav's `/way_point` interface. ROS message imports are lazy so this
module remains importable in unit tests and offline analysis environments.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from real_robot.contracts import (
    BBox2D,
    DetectionFrame,
    FrontierSnapshot,
    MotionGoal,
    MotionGoalMode,
    NavigationStatus,
    NavigationStatusCode,
    ObjectNodeSnapshot,
    Pose3D,
    RoomSnapshot,
    SemanticMapSnapshot,
)
from real_robot.detector_vocabulary import (
    DetectorVocabulary,
    merge_label_provenance,
    vocabulary_context,
)


@dataclass(frozen=True)
class SysNavTopicConfig:
    """Topic names used by the first SysNav-backed STRIVE real-robot runtime."""

    camera_image: str = "/camera/image"
    detection_result: str = "/detection_result"
    object_nodes_list: str = "/object_nodes_list"
    room_nodes_list: str = "/room_nodes_list"
    waypoint: str = "/way_point"
    odometry: str = "/aft_mapped_to_init"
    path: str = "/path"
    world_frame: str = "map"


class RosDetectionResultAdapter:
    """Convert SysNav ``tare_planner/DetectionResult`` messages to STRIVE detections."""

    def __init__(
        self,
        topic: str = SysNavTopicConfig.detection_result,
        detector_vocabulary: Optional[DetectorVocabulary] = None,
    ) -> None:
        self.topic = topic
        self.detector_vocabulary = detector_vocabulary

    def from_msg(self, msg: Any, image_ref: Optional[str] = None) -> DetectionFrame:
        """Return a platform-neutral detection frame for one SysNav message."""

        # 核心：adapter 只做 ROS msg -> contract 的字段规范化，不做类别归一化或目标判断。
        boxes = _boxes_from_parallel_arrays(
            _as_sequence(getattr(msg, "x1", ())),
            _as_sequence(getattr(msg, "y1", ())),
            _as_sequence(getattr(msg, "x2", ())),
            _as_sequence(getattr(msg, "y2", ())),
        )
        labels = tuple(str(label) for label in _as_sequence(getattr(msg, "label", ())))
        confidences = tuple(float(conf) for conf in _as_sequence(getattr(msg, "confidence", ())))
        track_ids = tuple(str(track_id) for track_id in _as_sequence(getattr(msg, "track_id", ())))
        stamp = _stamp_from_header(getattr(msg, "header", None), default=0.0)
        label_provenance = tuple(_label_provenance(self.detector_vocabulary, label) for label in labels)

        inline_image = getattr(msg, "image", None)
        metadata = {
            "ros_topic": self.topic,
            "frame_id": _frame_id_from_header(getattr(msg, "header", None)),
            # 内联图像只记录摘要；真实图像落盘/缓存由 observation 或 runtime 层负责。
            "image": _image_summary(inline_image),
            "detector_vocabulary": vocabulary_context(self.detector_vocabulary),
            "label_provenance": label_provenance,
        }

        return DetectionFrame(
            timestamp=stamp,
            image_ref=image_ref or f"ros://{self.topic}/image/{stamp:.9f}",
            boxes_xyxy=boxes,
            labels=labels,
            confidences=confidences,
            track_ids=track_ids,
            source="sysnav_detection_result",
            metadata=metadata,
        )


class RosObjectNodeAdapter:
    """Convert SysNav ``ObjectNode`` and ``ObjectNodeList`` messages."""

    def __init__(
        self,
        topic: str = SysNavTopicConfig.object_nodes_list,
        detector_vocabulary: Optional[DetectorVocabulary] = None,
    ) -> None:
        self.topic = topic
        self.detector_vocabulary = detector_vocabulary

    def from_msg(self, msg: Any) -> ObjectNodeSnapshot:
        """Return a STRIVE object snapshot for one SysNav object node."""

        # 核心：SysNav object_id 是运行时对象身份，应作为 STRIVE ledger/cache 的主键来源。
        object_ids = tuple(int(obj_id) for obj_id in _as_sequence(getattr(msg, "object_id", ())))
        position = _point_to_vector3(getattr(msg, "position", None))
        # bbox3d 是几何证据，不在 adapter 层解释“是不是目标”或“关系是否成立”。
        bbox3d = tuple(
            point
            for point in (_point_to_vector3(point_msg) for point_msg in _as_sequence(getattr(msg, "bbox3d", ())))
            if point is not None
        )
        bbox_center, bbox_extent = _bbox3d_center_extent(bbox3d)
        viewpoint_id = getattr(msg, "viewpoint_id", None)
        visible_viewpoints = (str(viewpoint_id),) if viewpoint_id is not None and int(viewpoint_id) >= 0 else ()
        raw_label = str(getattr(msg, "label", ""))
        uid = _object_uid(object_ids, raw_label, position)

        metadata = {
            "ros_topic": self.topic,
            "frame_id": _frame_id_from_header(getattr(msg, "header", None)),
            # 保留 SysNav 原始 id，便于回放时对齐 SysNav 日志和 STRIVE 决策日志。
            "sysnav_object_ids": object_ids,
            "status": bool(getattr(msg, "status", False)),
            "is_asked_vlm": bool(getattr(msg, "is_asked_vlm", False)),
            "viewpoint_id": viewpoint_id,
            "bbox3d_corners": bbox3d,
            "cloud_present": getattr(msg, "cloud", None) is not None,
            "detector_vocabulary": vocabulary_context(self.detector_vocabulary),
        }
        metadata = merge_label_provenance(metadata, _label_provenance(self.detector_vocabulary, raw_label))

        return ObjectNodeSnapshot(
            uid=uid,
            label=raw_label,
            position=position,
            confidence=1.0 if bool(getattr(msg, "status", False)) else 0.0,
            bbox3d_center=bbox_center,
            bbox3d_extent=bbox_extent,
            image_ref=_none_if_empty(getattr(msg, "img_path", None)),
            pointcloud_ref=f"ros://{self.topic}/{uid}/cloud" if getattr(msg, "cloud", None) is not None else None,
            visible_viewpoints=visible_viewpoints,
            track_ids=tuple(str(obj_id) for obj_id in object_ids),
            verified_state="active" if bool(getattr(msg, "status", False)) else "inactive",
            metadata=metadata,
        )

    def from_list_msg(self, msg: Any) -> Tuple[ObjectNodeSnapshot, ...]:
        """Return all object snapshots in one SysNav object list message."""

        return tuple(self.from_msg(node) for node in _as_sequence(getattr(msg, "nodes", ())))


class RosRoomNodeAdapter:
    """Convert SysNav ``RoomNode`` and ``RoomNodeList`` messages."""

    def __init__(self, topic: str = SysNavTopicConfig.room_nodes_list) -> None:
        self.topic = topic

    def from_msg(self, msg: Any) -> RoomSnapshot:
        """Return a STRIVE room snapshot for one SysNav room node."""

        room_id = int(getattr(msg, "id", -1))
        polygon_points = _polygon_points(getattr(msg, "polygon", None))
        metadata = {
            "ros_topic": self.topic,
            "show_id": getattr(msg, "show_id", None),
            "is_connected": bool(getattr(msg, "is_connected", False)),
            "area": float(getattr(msg, "area", 0.0)),
            "polygon_point_count": len(polygon_points),
            "room_mask_present": getattr(msg, "room_mask", None) is not None,
        }

        return RoomSnapshot(
            uid=f"sysnav_room:{room_id}",
            # SysNav RoomNode 不直接给语义房间名；room label 仍由 STRIVE/VLM room policy 推断。
            label=None,
            centroid=_point_to_vector3(getattr(msg, "centroid", None)),
            neighbors=tuple(f"sysnav_room:{int(neighbor)}" for neighbor in _as_sequence(getattr(msg, "neighbors", ()))),
            image_ref=f"ros://{self.topic}/room_mask/{room_id}"
            if getattr(msg, "room_mask", None) is not None
            else None,
            explored=bool(getattr(msg, "is_connected", False)),
            metadata=metadata,
        )

    def from_list_msg(self, msg: Any) -> Tuple[RoomSnapshot, ...]:
        """Return all room snapshots in one SysNav room list message."""

        return tuple(self.from_msg(node) for node in _as_sequence(getattr(msg, "nodes", ())))


@dataclass
class _GoalProgressState:
    goal_id: str
    started_at: float
    last_progress_at: float
    initial_distance_m: Optional[float] = None
    best_distance_m: Optional[float] = None
    preempted: bool = False
    progress_samples: list[Dict[str, float]] = field(default_factory=list)


class RosNavigationStatusProvider:
    """Infer `NavigationStatus` from odometry, path, and planner status topics.

    The provider does not publish control commands and does not own path
    planning. It is a read-only progress monitor that can be passed to
    `RosWaypointController(status_provider=...)`.
    """

    def __init__(
        self,
        xy_tolerance_m: float = 0.35,
        z_tolerance_m: Optional[float] = 1.0,
        heading_tolerance_rad: Optional[float] = None,
        timeout_s: float = 60.0,
        no_progress_timeout_s: float = 12.0,
        min_progress_delta_m: float = 0.05,
        path_stale_timeout_s: float = 5.0,
        max_progress_samples: int = 20,
        world_frame: str = SysNavTopicConfig.world_frame,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the status provider.

        Args:
            xy_tolerance_m: Goal reached threshold in the horizontal plane.
            z_tolerance_m: Optional vertical threshold. Set to `None` to ignore
                z. The default is permissive because SysNav waypoints may carry
                a ground-relative z offset while odometry z stays near zero.
            heading_tolerance_rad: Optional heading threshold. The first
                `/way_point` interface carries only a point, so heading is
                ignored unless a future controller supplies it explicitly.
            timeout_s: Maximum wall/ROS time for one goal attempt.
            no_progress_timeout_s: Time without meaningful distance decrease
                before reporting `BLOCKED`.
            min_progress_delta_m: Minimum distance improvement considered real
                progress.
            path_stale_timeout_s: Age after which a cached path is treated as
                stale for metadata.
            max_progress_samples: Number of recent distance samples to keep.
            world_frame: Default frame for pose messages without header frame.
            now_fn: Time source used for elapsed/progress calculations.
        """

        self.xy_tolerance_m = float(xy_tolerance_m)
        self.z_tolerance_m = None if z_tolerance_m is None else float(z_tolerance_m)
        self.heading_tolerance_rad = heading_tolerance_rad
        self.timeout_s = float(timeout_s)
        self.no_progress_timeout_s = float(no_progress_timeout_s)
        self.min_progress_delta_m = float(min_progress_delta_m)
        self.path_stale_timeout_s = float(path_stale_timeout_s)
        self.max_progress_samples = int(max_progress_samples)
        self.world_frame = world_frame
        self.now_fn = now_fn

        self.latest_pose: Optional[Pose3D] = None
        self.latest_path_points: Tuple[Tuple[float, float, float], ...] = ()
        self.latest_path_stamp: Optional[float] = None
        self.latest_path_received_at: Optional[float] = None
        self.latest_planner_status: Optional[Dict[str, Any]] = None
        self._goal_states: Dict[str, _GoalProgressState] = {}

    def update_odometry(self, msg: Any) -> None:
        """Cache the latest odometry pose from a ROS-like message."""

        self.latest_pose = _pose3d_from_odometry_msg(msg, default_frame=self.world_frame)

    def update_pose(self, pose: Pose3D) -> None:
        """Cache a platform-neutral pose for offline replay or tests."""

        self.latest_pose = pose

    def update_path(self, msg: Any) -> None:
        """Cache the latest local planner path from a ROS-like Path message."""

        self.latest_path_points = _path_points_from_msg(msg)
        self.latest_path_stamp = _stamp_from_header(getattr(msg, "header", None), default=self.now_fn())
        self.latest_path_received_at = self.now_fn()

    def update_local_planner_status(self, msg: Any) -> None:
        """Cache a generic local planner status message.

        String-like values such as `blocked`, `timeout`, `preempted`, and
        `reached` are mapped into terminal navigation states. Boolean `False`
        means no executable path is currently available.
        """

        value = getattr(msg, "data", msg)
        self.latest_planner_status = {
            "raw": value,
            "text": str(value).strip().lower(),
            "received_at": self.now_fn(),
        }

    def create_ros_subscriptions(
        self,
        node: Any,
        odometry_type: Optional[Any] = None,
        path_type: Optional[Any] = None,
        planner_status_type: Optional[Any] = None,
        odom_topic: str = SysNavTopicConfig.odometry,
        path_topic: str = SysNavTopicConfig.path,
        planner_status_topic: str = "",
        queue_size: int = 10,
    ) -> Dict[str, Any]:
        """Create ROS subscriptions and return their handles.

        ROS message types are injectable to keep unit tests independent from a
        ROS installation.
        """

        subscriptions = {
            "odometry": node.create_subscription(
                odometry_type or _import_odometry_type(),
                odom_topic,
                self.update_odometry,
                queue_size,
            ),
            "path": node.create_subscription(
                path_type or _import_path_type(),
                path_topic,
                self.update_path,
                queue_size,
            ),
        }
        if planner_status_topic:
            subscriptions["planner_status"] = node.create_subscription(
                planner_status_type or _import_string_type(),
                planner_status_topic,
                self.update_local_planner_status,
                queue_size,
            )
        return subscriptions

    def cancel(self, goal_id: Optional[str] = None) -> None:
        """Mark one active goal, or all goals, as preempted."""

        if goal_id is None:
            for state in self._goal_states.values():
                state.preempted = True
            return
        state = self._goal_states.get(goal_id)
        if state is not None:
            state.preempted = True

    def __call__(self, goal_id: str, goal: MotionGoal) -> NavigationStatus:
        """Return the latest status for one active motion goal."""

        now = self.now_fn()
        state = self._goal_states.get(goal_id)
        if state is None:
            state = _GoalProgressState(goal_id=goal_id, started_at=now, last_progress_at=now)
            self._goal_states[goal_id] = state

        if state.preempted:
            return NavigationStatus(
                NavigationStatusCode.PREEMPTED,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                stamp=now,
                message="goal preempted by STRIVE waypoint controller",
                metadata=self._metadata(goal, state, now),
            )

        if not goal.requires_motion():
            status = NavigationStatusCode.REACHED if goal.mode == MotionGoalMode.STOP else NavigationStatusCode.IDLE
            return NavigationStatus(
                status,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                stamp=now,
                message=f"{goal.mode.value} does not require motion",
                metadata=self._metadata(goal, state, now),
            )

        if goal.goal_pose is None:
            return NavigationStatus(
                NavigationStatusCode.FAILED,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                stamp=now,
                message="motion goal has no goal_pose",
                metadata=self._metadata(goal, state, now),
            )

        if self.latest_pose is None:
            return NavigationStatus(
                NavigationStatusCode.QUEUED,
                goal_id=goal_id,
                stamp=now,
                message="waiting for odometry before evaluating navigation status",
                metadata=self._metadata(goal, state, now),
            )

        distances = self._distances_to_goal(goal)
        self._record_progress_sample(state, now, distances["distance_3d_m"])

        if state.initial_distance_m is None:
            state.initial_distance_m = distances["distance_3d_m"]
            state.best_distance_m = distances["distance_3d_m"]
        elif state.best_distance_m is None or distances["distance_3d_m"] < state.best_distance_m - self.min_progress_delta_m:
            state.best_distance_m = distances["distance_3d_m"]
            state.last_progress_at = now

        path_length = self._path_length_remaining()
        metadata = self._metadata(goal, state, now, distances=distances, path_length_remaining=path_length)
        progress = self._progress_fraction(state, distances["distance_3d_m"])

        if self._is_reached(distances):
            return NavigationStatus(
                NavigationStatusCode.REACHED,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                distance_to_goal=distances["distance_3d_m"],
                path_length_remaining=path_length,
                progress=1.0,
                stamp=now,
                message="goal reached within configured tolerance",
                metadata=metadata,
            )

        planner_status = self._fresh_planner_status_code(now)
        if planner_status in {
            NavigationStatusCode.REACHED,
            NavigationStatusCode.BLOCKED,
            NavigationStatusCode.TIMEOUT,
            NavigationStatusCode.PREEMPTED,
            NavigationStatusCode.FAILED,
        }:
            return NavigationStatus(
                planner_status,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                distance_to_goal=distances["distance_3d_m"],
                path_length_remaining=path_length,
                progress=progress,
                stamp=now,
                message=f"local planner reported {planner_status.value}",
                metadata=metadata,
            )

        elapsed_s = now - state.started_at
        if elapsed_s >= self.timeout_s:
            return NavigationStatus(
                NavigationStatusCode.TIMEOUT,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                distance_to_goal=distances["distance_3d_m"],
                path_length_remaining=path_length,
                progress=progress,
                stamp=now,
                message="navigation goal timed out before reaching target",
                metadata=metadata,
            )

        no_progress_elapsed_s = now - state.last_progress_at
        if no_progress_elapsed_s >= self.no_progress_timeout_s:
            return NavigationStatus(
                NavigationStatusCode.BLOCKED,
                goal_id=goal_id,
                current_pose=self.latest_pose,
                distance_to_goal=distances["distance_3d_m"],
                path_length_remaining=path_length,
                progress=progress,
                stamp=now,
                message="navigation made no measurable progress",
                metadata=metadata,
            )

        return NavigationStatus(
            NavigationStatusCode.RUNNING,
            goal_id=goal_id,
            current_pose=self.latest_pose,
            distance_to_goal=distances["distance_3d_m"],
            path_length_remaining=path_length,
            progress=progress,
            stamp=now,
            message="navigation goal is running",
            metadata=metadata,
        )

    def _distances_to_goal(self, goal: MotionGoal) -> Dict[str, float]:
        """Return x/y/z/3-D distance facts from latest pose to goal."""

        assert self.latest_pose is not None
        assert goal.goal_pose is not None
        current = self.latest_pose.position
        target = goal.goal_pose.position
        dx = float(target[0] - current[0])
        dy = float(target[1] - current[1])
        dz = float(target[2] - current[2])
        return {
            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
            "xy_distance_m": math.hypot(dx, dy),
            "z_distance_m": abs(dz),
            "distance_3d_m": math.sqrt(dx * dx + dy * dy + dz * dz),
        }

    def _is_reached(self, distances: Dict[str, float]) -> bool:
        """Return whether distance facts satisfy configured reach thresholds."""

        if distances["xy_distance_m"] > self.xy_tolerance_m:
            return False
        if self.z_tolerance_m is not None and distances["z_distance_m"] > self.z_tolerance_m:
            return False
        return True

    def _path_length_remaining(self) -> Optional[float]:
        """Return path length from current pose through cached path points."""

        if self.latest_pose is None or not self.latest_path_points:
            return None
        return _path_length((self.latest_pose.position, *self.latest_path_points))

    def _record_progress_sample(self, state: _GoalProgressState, stamp: float, distance_m: float) -> None:
        """Append one bounded progress sample."""

        state.progress_samples.append({"stamp": float(stamp), "distance_3d_m": float(distance_m)})
        if len(state.progress_samples) > self.max_progress_samples:
            del state.progress_samples[: len(state.progress_samples) - self.max_progress_samples]

    def _progress_fraction(self, state: _GoalProgressState, current_distance_m: float) -> Optional[float]:
        """Return normalized progress based on initial distance."""

        if state.initial_distance_m is None or state.initial_distance_m <= 1e-6:
            return None
        return max(0.0, min(1.0, (state.initial_distance_m - current_distance_m) / state.initial_distance_m))

    def _metadata(
        self,
        goal: MotionGoal,
        state: _GoalProgressState,
        now: float,
        distances: Optional[Dict[str, float]] = None,
        path_length_remaining: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return detailed status diagnostics for logs and JSONL output."""

        path_age_s = None if self.latest_path_received_at is None else max(0.0, now - self.latest_path_received_at)
        path_available = bool(self.latest_path_points) and (
            path_age_s is None or path_age_s <= self.path_stale_timeout_s
        )
        planner_status_age_s = (
            None
            if self.latest_planner_status is None
            else max(0.0, now - float(self.latest_planner_status.get("received_at", now)))
        )
        return {
            "goal_mode": goal.mode.value,
            "goal_frame_id": goal.goal_pose.frame_id if goal.goal_pose else None,
            "elapsed_s": max(0.0, now - state.started_at),
            "timeout_s": self.timeout_s,
            "no_progress_timeout_s": self.no_progress_timeout_s,
            "no_progress_elapsed_s": max(0.0, now - state.last_progress_at),
            "min_progress_delta_m": self.min_progress_delta_m,
            "xy_tolerance_m": self.xy_tolerance_m,
            "z_tolerance_m": self.z_tolerance_m,
            "heading_tolerance_rad": self.heading_tolerance_rad,
            "heading_checked": self.heading_tolerance_rad is not None,
            "initial_distance_m": state.initial_distance_m,
            "best_distance_m": state.best_distance_m,
            "distance": distances or {},
            "path_available": path_available,
            "path_pose_count": len(self.latest_path_points),
            "path_age_s": path_age_s,
            "path_length_remaining": path_length_remaining,
            "planner_status": self.latest_planner_status,
            "planner_status_age_s": planner_status_age_s,
            "planner_status_fresh": planner_status_age_s is not None and planner_status_age_s <= self.path_stale_timeout_s,
            "progress_samples": list(state.progress_samples),
        }

    def _fresh_planner_status_code(self, now: float) -> Optional[NavigationStatusCode]:
        """Return a planner status only when the cached status is fresh."""

        if self.latest_planner_status is None:
            return None
        received_at = float(self.latest_planner_status.get("received_at", now))
        if now - received_at > self.path_stale_timeout_s:
            return None
        return _planner_status_code(self.latest_planner_status)


class RosWaypointController:
    """Publish STRIVE motion goals to SysNav's ``/way_point`` interface."""

    def __init__(
        self,
        node: Any,
        waypoint_topic: str = SysNavTopicConfig.waypoint,
        world_frame: str = SysNavTopicConfig.world_frame,
        publisher: Optional[Any] = None,
        point_stamped_type: Optional[Any] = None,
        status_provider: Optional[Callable[[str, MotionGoal], NavigationStatus]] = None,
        queue_size: int = 10,
    ) -> None:
        self.node = node
        self.waypoint_topic = waypoint_topic
        self.world_frame = world_frame
        self.status_provider = status_provider
        self._last_goal_id: Optional[str] = None
        self._last_goal: Optional[MotionGoal] = None
        self._last_status = NavigationStatus(NavigationStatusCode.IDLE, message="no goal submitted")

        if publisher is not None:
            # 测试或离线回放可以注入 fake publisher，避免 contract 测试依赖 ROS2 runtime。
            self.publisher = publisher
            self._point_stamped_type = point_stamped_type
        else:
            # 核心：ROS message 类型延迟导入，保证非 ROS 环境仍可 import adapter 做离线分析。
            self._point_stamped_type = point_stamped_type or _import_point_stamped_type()
            self.publisher = node.create_publisher(self._point_stamped_type, waypoint_topic, queue_size)

    def send_goal(self, goal: MotionGoal) -> str:
        """Submit one STRIVE motion goal to SysNav and return a stable goal id."""

        goal_id = f"sysnav_goal:{uuid.uuid4().hex}"
        self._last_goal_id = goal_id
        self._last_goal = goal

        if not goal.requires_motion():
            # STOP/WAIT 是高层状态，不应伪造成 /way_point，否则会触发下层无意义移动。
            status = NavigationStatusCode.REACHED if goal.mode == MotionGoalMode.STOP else NavigationStatusCode.IDLE
            self._last_status = NavigationStatus(status, goal_id=goal_id, message=f"{goal.mode.value} does not require motion")
            return goal_id

        point_msg = self._make_point_stamped(goal)
        # 核心：STRIVE 只发布 waypoint，局部避障、速度控制和急停继续由 SysNav 下层负责。
        self.publisher.publish(point_msg)
        self._last_status = NavigationStatus(
            NavigationStatusCode.RUNNING,
            goal_id=goal_id,
            distance_to_goal=None,
            message=f"published {goal.mode.value} to {self.waypoint_topic}",
            metadata={
                "waypoint_topic": self.waypoint_topic,
                "target_object_uid": goal.target_object_uid,
                "anchor_object_uid": goal.anchor_object_uid,
                "relation_edge_id": goal.relation_edge_id,
            },
        )
        return goal_id

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return the latest status for the submitted goal."""

        if self.status_provider is not None and self._last_goal is not None:
            # live robot 可接入 odom/path/progress monitor；adapter 本身不推断可达性。
            return self.status_provider(goal_id, self._last_goal)
        if goal_id != self._last_goal_id:
            return NavigationStatus(NavigationStatusCode.FAILED, goal_id=goal_id, message="unknown goal id")
        return self._last_status

    def cancel(self, goal_id: Optional[str] = None) -> None:
        """Mark the active goal as preempted.

        SysNav's existing `/way_point` interface has no universal cancel topic, so
        platform-specific stop/cancel wiring should be added by a subclass.
        """

        target_goal_id = goal_id or self._last_goal_id
        if self.status_provider is not None and hasattr(self.status_provider, "cancel"):
            self.status_provider.cancel(target_goal_id)
        self._last_status = NavigationStatus(
            NavigationStatusCode.PREEMPTED,
            goal_id=target_goal_id,
            message="goal cancelled by STRIVE bridge",
        )

    def hold(self) -> None:
        """Request a safe hold at the bridge level.

        The first implementation only updates bridge state. A live robot adapter
        should override this method to publish SysNav's platform-specific stop
        or hold signal.
        """

        self._last_status = NavigationStatus(
            NavigationStatusCode.IDLE,
            goal_id=self._last_goal_id,
            message="safe hold requested at STRIVE bridge",
        )

    def _make_point_stamped(self, goal: MotionGoal) -> Any:
        """Build a ``geometry_msgs/PointStamped`` compatible message."""

        if goal.goal_pose is None:
            raise ValueError("MotionGoal.goal_pose is required for SysNav waypoint publication")

        msg = self._point_stamped_type()
        # SysNav /way_point 只消费三维点；朝向/look_at 由后续 controller 或证据采集层处理。
        msg.header.frame_id = goal.goal_pose.frame_id or self.world_frame
        _set_stamp_now(msg, self.node)
        msg.point.x = float(goal.goal_pose.position[0])
        msg.point.y = float(goal.goal_pose.position[1])
        msg.point.z = float(goal.goal_pose.position[2])
        return msg


def build_semantic_map_snapshot(
    object_list_msg: Any,
    room_list_msg: Optional[Any],
    robot_pose: Pose3D,
    timestamp: Optional[float] = None,
    object_adapter: Optional[RosObjectNodeAdapter] = None,
    room_adapter: Optional[RosRoomNodeAdapter] = None,
) -> SemanticMapSnapshot:
    """Build a STRIVE map snapshot from SysNav object and room list messages."""

    object_adapter = object_adapter or RosObjectNodeAdapter()
    room_adapter = room_adapter or RosRoomNodeAdapter()
    objects = object_adapter.from_list_msg(object_list_msg)
    rooms = room_adapter.from_list_msg(room_list_msg) if room_list_msg is not None else ()

    # 核心：SysNav 继续负责 detector/mapping，STRIVE 只消费只读 snapshot 做语义规划。
    # 这里不能反向修改 SysNav object/room 状态，否则会破坏两个系统的职责边界。
    return SemanticMapSnapshot(
        timestamp=timestamp if timestamp is not None else _stamp_from_header(getattr(object_list_msg, "header", None), default=time.time()),
        robot_pose=robot_pose,
        objects=objects,
        rooms=rooms,
        frontiers=_frontiers_from_rooms(rooms),
        source="sysnav_ros",
        metadata={
            "object_count": len(objects),
            "room_count": len(rooms),
            "sysnav_topics": SysNavTopicConfig().__dict__,
            "detector_vocabulary": vocabulary_context(getattr(object_adapter, "detector_vocabulary", None)),
        },
    )


def _frontiers_from_rooms(rooms: Iterable[RoomSnapshot]) -> Tuple[FrontierSnapshot, ...]:
    """Expose room centroids as coarse exploration references for the first bridge."""

    frontiers = []
    for room in rooms:
        if room.centroid is None:
            continue
        # 第一版只把 room centroid 暴露为粗粒度参考点；真正 frontier 仍来自 SysNav planner。
        frontiers.append(
            FrontierSnapshot(
                uid=f"{room.uid}:centroid",
                position=room.centroid,
                room_id=room.uid,
                metadata={"source": "sysnav_room_centroid"},
            )
        )
    return tuple(frontiers)


def _boxes_from_parallel_arrays(
    x1_values: Sequence[Any],
    y1_values: Sequence[Any],
    x2_values: Sequence[Any],
    y2_values: Sequence[Any],
) -> Tuple[BBox2D, ...]:
    """Return bbox tuples after validating SysNav parallel arrays."""

    # SysNav DetectionResult 使用并行数组；进入 STRIVE 前必须先保证长度一致。
    lengths = {len(x1_values), len(y1_values), len(x2_values), len(y2_values)}
    if len(lengths) != 1:
        raise ValueError("SysNav DetectionResult bbox arrays must have the same length")
    return tuple(
        (float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in zip(x1_values, y1_values, x2_values, y2_values)
    )


def _as_sequence(value: Any) -> Tuple[Any, ...]:
    """Return ROS array fields as immutable Python tuples."""

    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _stamp_from_header(header: Any, default: float) -> float:
    """Return a float timestamp from a ROS-like header."""

    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return float(default)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    if sec is None:
        return float(default)
    return float(sec) + float(nanosec) / 1e9


def _frame_id_from_header(header: Any) -> Optional[str]:
    """Return frame id from a ROS-like header."""

    frame_id = getattr(header, "frame_id", None)
    return str(frame_id) if frame_id else None


def _image_summary(image_msg: Any) -> Dict[str, Any]:
    """Return lightweight metadata for an inline ROS image message."""

    if image_msg is None:
        return {"present": False}
    return {
        "present": True,
        "height": getattr(image_msg, "height", None),
        "width": getattr(image_msg, "width", None),
        "encoding": getattr(image_msg, "encoding", None),
        "step": getattr(image_msg, "step", None),
    }


def _point_to_vector3(point: Any) -> Optional[Tuple[float, float, float]]:
    """Convert a ROS-like point to a vector tuple."""

    if point is None:
        return None
    if not all(hasattr(point, field) for field in ("x", "y", "z")):
        return None
    return (float(point.x), float(point.y), float(point.z))


def _bbox3d_center_extent(points: Tuple[Tuple[float, float, float], ...]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """Compute bbox center and extent from SysNav 3-D corner points."""

    if not points:
        return None, None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    min_corner = (min(xs), min(ys), min(zs))
    max_corner = (max(xs), max(ys), max(zs))
    center = tuple((lo + hi) / 2.0 for lo, hi in zip(min_corner, max_corner))
    extent = tuple(hi - lo for lo, hi in zip(min_corner, max_corner))
    return center, extent


def _object_uid(object_ids: Tuple[int, ...], label: Any, position: Optional[Tuple[float, float, float]]) -> str:
    """Return a stable STRIVE uid for one SysNav object node."""

    # object_id 优先级最高；无 id 时才退化到 label+position，避免同类物体互相污染 ledger。
    if object_ids:
        return "sysnav_object:" + ":".join(str(obj_id) for obj_id in object_ids)
    if position is not None:
        return f"sysnav_object:{label}:{position[0]:.3f}:{position[1]:.3f}:{position[2]:.3f}"
    return f"sysnav_object:{label}:unknown"


def _label_provenance(vocabulary: Optional[DetectorVocabulary], raw_label: str) -> Dict[str, Any]:
    """Return detector label provenance for one raw SysNav label."""

    if vocabulary is None:
        return {
            "raw_detector_label": raw_label,
            "known_in_detector_vocabulary": None,
            "detector_name": None,
            "config_path": None,
        }
    return vocabulary.provenance_for(raw_label)


def _none_if_empty(value: Any) -> Optional[str]:
    """Return None for empty strings or missing values."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _polygon_points(polygon_stamped: Any) -> Tuple[Tuple[float, float, float], ...]:
    """Return polygon points from a ROS ``PolygonStamped``-like object."""

    polygon = getattr(polygon_stamped, "polygon", None)
    points = getattr(polygon, "points", None)
    if points is None:
        return ()
    return tuple(point for point in (_point_to_vector3(point_msg) for point_msg in points) if point is not None)


def _pose3d_from_odometry_msg(msg: Any, default_frame: str) -> Pose3D:
    """Return `Pose3D` from a ROS-like `nav_msgs/Odometry` message."""

    pose_with_covariance = getattr(msg, "pose", None)
    pose_msg = getattr(pose_with_covariance, "pose", pose_with_covariance)
    return _pose3d_from_pose_msg(pose_msg, getattr(msg, "header", None), default_frame)


def _pose3d_from_pose_stamped_msg(msg: Any, default_frame: str) -> Pose3D:
    """Return `Pose3D` from a ROS-like `geometry_msgs/PoseStamped` message."""

    return _pose3d_from_pose_msg(getattr(msg, "pose", None), getattr(msg, "header", None), default_frame)


def _pose3d_from_pose_msg(pose_msg: Any, header: Any, default_frame: str) -> Pose3D:
    """Return `Pose3D` from a ROS-like pose and header."""

    position = _point_to_vector3(getattr(pose_msg, "position", None)) or (0.0, 0.0, 0.0)
    return Pose3D(
        position=position,
        orientation_xyzw=_orientation_to_xyzw(getattr(pose_msg, "orientation", None)),
        frame_id=_frame_id_from_header(header) or default_frame,
        stamp=_stamp_from_header(header, default=0.0),
    )


def _orientation_to_xyzw(orientation: Any) -> Tuple[float, float, float, float]:
    """Return a quaternion tuple from a ROS-like orientation."""

    if orientation is None:
        return (0.0, 0.0, 0.0, 1.0)
    return (
        float(getattr(orientation, "x", 0.0)),
        float(getattr(orientation, "y", 0.0)),
        float(getattr(orientation, "z", 0.0)),
        float(getattr(orientation, "w", 1.0)),
    )


def _path_points_from_msg(msg: Any) -> Tuple[Tuple[float, float, float], ...]:
    """Return path pose positions from a ROS-like `nav_msgs/Path` message."""

    points = []
    for pose_stamped in _as_sequence(getattr(msg, "poses", ())):
        pose = _pose3d_from_pose_stamped_msg(pose_stamped, default_frame=_frame_id_from_header(getattr(msg, "header", None)) or SysNavTopicConfig.world_frame)
        points.append(pose.position)
    return tuple(points)


def _path_length(points: Sequence[Tuple[float, float, float]]) -> float:
    """Return cumulative Euclidean length for a sequence of 3-D points."""

    if len(points) < 2:
        return 0.0
    return sum(_euclidean_distance(prev, curr) for prev, curr in zip(points[:-1], points[1:]))


def _euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Return Euclidean distance between two 3-D points."""

    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dz = float(a[2] - b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _planner_status_code(status: Optional[Dict[str, Any]]) -> Optional[NavigationStatusCode]:
    """Map a generic local planner status record to a navigation status code."""

    if status is None:
        return None
    raw = status.get("raw")
    if isinstance(raw, bool):
        return None if raw else NavigationStatusCode.BLOCKED

    text = str(status.get("text", "")).strip().lower()
    if text in {"", "running", "active", "ok", "true", "path", "path_available"}:
        return None
    if text in {"reached", "success", "succeeded", "done"}:
        return NavigationStatusCode.REACHED
    if text in {"blocked", "no_path", "no path", "path_blocked", "stuck"}:
        return NavigationStatusCode.BLOCKED
    if text in {"timeout", "timed_out", "time out"}:
        return NavigationStatusCode.TIMEOUT
    if text in {"preempted", "cancelled", "canceled", "preempt"}:
        return NavigationStatusCode.PREEMPTED
    if text in {"failed", "failure", "error"}:
        return NavigationStatusCode.FAILED
    if text == "false":
        return NavigationStatusCode.BLOCKED
    return None


def _import_point_stamped_type() -> Any:
    """Import ``geometry_msgs.msg.PointStamped`` lazily."""

    try:
        from geometry_msgs.msg import PointStamped
    except ImportError as exc:
        raise RuntimeError(
            "geometry_msgs is required for RosWaypointController without an injected point_stamped_type"
        ) from exc
    return PointStamped


def _import_odometry_type() -> Any:
    """Import ``nav_msgs.msg.Odometry`` lazily."""

    try:
        from nav_msgs.msg import Odometry
    except ImportError as exc:
        raise RuntimeError("nav_msgs is required for ROS odometry subscriptions") from exc
    return Odometry


def _import_path_type() -> Any:
    """Import ``nav_msgs.msg.Path`` lazily."""

    try:
        from nav_msgs.msg import Path
    except ImportError as exc:
        raise RuntimeError("nav_msgs is required for ROS path subscriptions") from exc
    return Path


def _import_string_type() -> Any:
    """Import ``std_msgs.msg.String`` lazily."""

    try:
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("std_msgs is required for ROS planner status subscriptions") from exc
    return String


def _set_stamp_now(point_stamped_msg: Any, node: Any) -> None:
    """Set header stamp when the provided ROS node exposes a clock."""

    get_clock = getattr(node, "get_clock", None)
    if get_clock is None:
        return
    try:
        point_stamped_msg.header.stamp = get_clock().now().to_msg()
    except AttributeError:
        return
