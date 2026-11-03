"""Backend-independent sensor and semantic scene value objects.

These contracts have no dependency on ROS, Habitat, model clients or controllers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

Vector3 = Tuple[float, float, float]
QuaternionXYZW = Tuple[float, float, float, float]
BBox2D = Tuple[float, float, float, float]

class CameraModel(str, Enum):
    """Camera projection model consumed by real-robot adapters."""

    PANORAMA = "panorama"
    PINHOLE = "pinhole"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pose3D:
    """Rigid pose in a named coordinate frame."""

    position: Vector3
    orientation_xyzw: QuaternionXYZW = (0.0, 0.0, 0.0, 1.0)
    frame_id: str = "map"
    stamp: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly pose representation."""

        return {
            "position": list(self.position),
            "orientation_xyzw": list(self.orientation_xyzw),
            "frame_id": self.frame_id,
            "stamp": self.stamp,
        }


@dataclass(frozen=True)
class CameraFrame:
    """RGB/RGB-D camera evidence without binding to a concrete image type."""

    # 核心：contract 层只保存引用和标定信息，图像数组由 adapter/runtime 缓存管理。
    image_ref: str
    camera_model: CameraModel = CameraModel.UNKNOWN
    timestamp: Optional[float] = None
    frame_id: str = "camera"
    rgb_shape: Optional[Tuple[int, int, int]] = None
    depth_ref: Optional[str] = None
    depth_valid_mask_ref: Optional[str] = None
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    extrinsics: Optional[Pose3D] = None
    fov: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealObservation:
    """Synchronized sensor snapshot delivered to VLN real-robot runtime."""

    # 核心：一次 observation 必须绑定同一个 robot_pose，避免 RGB/LiDAR/odom 时序漂移。
    timestamp: float
    robot_pose: Pose3D
    camera_frames: Tuple[CameraFrame, ...] = ()
    pointcloud_ref: Optional[str] = None
    pointcloud_frame_id: Optional[str] = None
    odom_frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def primary_camera(self) -> Optional[CameraFrame]:
        """Return the first camera frame, if the adapter produced one."""

        return self.camera_frames[0] if self.camera_frames else None


@dataclass(frozen=True)
class DetectionFrame:
    """Detector output aligned with one camera frame."""

    timestamp: float
    image_ref: str
    boxes_xyxy: Tuple[BBox2D, ...] = ()
    labels: Tuple[str, ...] = ()
    confidences: Tuple[float, ...] = ()
    track_ids: Tuple[str, ...] = ()
    masks_ref: Tuple[str, ...] = ()
    source: str = "detector"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate parallel detector arrays early."""

        # 检测输出进入 mapper 前先做结构校验，避免后续 bbox/label 错位污染对象图。
        n = len(self.boxes_xyxy)
        if len(self.labels) != n or len(self.confidences) != n:
            raise ValueError("boxes_xyxy, labels, and confidences must have the same length")
        if self.track_ids and len(self.track_ids) != n:
            raise ValueError("track_ids must be empty or match boxes_xyxy length")
        if self.masks_ref and len(self.masks_ref) != n:
            raise ValueError("masks_ref must be empty or match boxes_xyxy length")
        for box in self.boxes_xyxy:
            _validate_bbox_xyxy(box)

    def is_empty(self) -> bool:
        """Return whether this frame contains no detections."""

        return not self.boxes_xyxy


@dataclass(frozen=True)
class ObjectNodeSnapshot:
    """Stable object node snapshot exposed by mapping or detection adapters."""

    uid: str
    label: str
    position: Optional[Vector3] = None
    confidence: float = 0.0
    bbox2d_xyxy: Optional[BBox2D] = None
    bbox3d_center: Optional[Vector3] = None
    bbox3d_extent: Optional[Vector3] = None
    room_id: Optional[str] = None
    image_ref: Optional[str] = None
    pointcloud_ref: Optional[str] = None
    visible_viewpoints: Tuple[str, ...] = ()
    track_ids: Tuple[str, ...] = ()
    verified_state: str = "unverified"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optional 2-D evidence."""

        if self.bbox2d_xyxy is not None:
            _validate_bbox_xyxy(self.bbox2d_xyxy)

    def stable_key(self) -> str:
        """Return the strongest available identity key for cache lookup."""

        # 核心：真实机器人上的 detector track 会漂移，优先使用 mapper uid。
        if self.uid:
            return self.uid
        if self.track_ids:
            return ":".join(self.track_ids)
        return f"{self.label}:{self.position}"

    def concept_payload(self) -> Dict[str, Any]:
        """Return a compact payload for concept grounding prompts."""

        # 只给 concept matcher 必要字段；大图、点云等重证据通过 image_ref/pointcloud_ref 查取。
        return {
            "uid": self.uid,
            "label": self.label,
            "position": self.position,
            "confidence": self.confidence,
            "room_id": self.room_id,
            "bbox2d_xyxy": self.bbox2d_xyxy,
            "visible_viewpoints": list(self.visible_viewpoints),
            "verified_state": self.verified_state,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ViewpointSnapshot:
    """Discrete candidate viewpoint maintained by the high-level map."""

    uid: str
    pose: Pose3D
    room_id: Optional[str] = None
    visible_objects: Tuple[str, ...] = ()
    image_ref: Optional[str] = None
    explored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontierSnapshot:
    """Exploration frontier candidate represented in robot coordinates."""

    uid: str
    position: Vector3
    room_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoomSnapshot:
    """Room-level semantic and topological summary."""

    uid: str
    label: Optional[str] = None
    centroid: Optional[Vector3] = None
    neighbors: Tuple[str, ...] = ()
    objects: Tuple[str, ...] = ()
    frontiers: Tuple[str, ...] = ()
    image_ref: Optional[str] = None
    explored: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticMapSnapshot:
    """Map snapshot consumed by high-level planning and verification."""

    timestamp: float
    robot_pose: Pose3D
    objects: Tuple[ObjectNodeSnapshot, ...] = ()
    rooms: Tuple[RoomSnapshot, ...] = ()
    viewpoints: Tuple[ViewpointSnapshot, ...] = ()
    frontiers: Tuple[FrontierSnapshot, ...] = ()
    source: str = "real_robot"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def object_by_uid(self, uid: str) -> Optional[ObjectNodeSnapshot]:
        """Return one object snapshot by uid."""

        return next((obj for obj in self.objects if obj.uid == uid), None)

    def room_by_uid(self, uid: str) -> Optional[RoomSnapshot]:
        """Return one room snapshot by uid."""

        return next((room for room in self.rooms if room.uid == uid), None)


def _validate_bbox_xyxy(box: BBox2D) -> None:
    """Validate a 2-D bounding box in ``x1, y1, x2, y2`` format."""

    if len(box) != 4:
        raise ValueError("bbox must contain exactly four values")
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
