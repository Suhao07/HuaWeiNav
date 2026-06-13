"""Adapters for reusing SysNav ROS detection, mapping, and waypoint topics.

The adapters in this module are intentionally thin. They translate SysNav ROS
messages into STRIVE real-robot contracts and translate STRIVE motion goals
back to SysNav's `/way_point` interface. ROS message imports are lazy so this
module remains importable in unit tests and offline analysis environments.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
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


def _import_point_stamped_type() -> Any:
    """Import ``geometry_msgs.msg.PointStamped`` lazily."""

    try:
        from geometry_msgs.msg import PointStamped
    except ImportError as exc:
        raise RuntimeError(
            "geometry_msgs is required for RosWaypointController without an injected point_stamped_type"
        ) from exc
    return PointStamped


def _set_stamp_now(point_stamped_msg: Any, node: Any) -> None:
    """Set header stamp when the provided ROS node exposes a clock."""

    get_clock = getattr(node, "get_clock", None)
    if get_clock is None:
        return
    try:
        point_stamped_msg.header.stamp = get_clock().now().to_msg()
    except AttributeError:
        return
