"""Platform-neutral contracts for STRIVE real-robot execution.

This module defines value objects exchanged between high-level STRIVE semantic
planning and platform-specific robot adapters. It intentionally avoids ROS,
Habitat, numpy, OpenCV, or detector imports so the same contracts can be used
by live robots, bag replay, simulation bridges, and unit tests.
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


class MotionGoalMode(str, Enum):
    """High-level motion intent independent of a concrete robot API."""

    EXPLORE = "explore"
    GO_TO_FRONTIER = "go_to_frontier"
    GO_TO_OBJECT = "go_to_object"
    GO_TO_ANCHOR = "go_to_anchor"
    IMPROVE_VIEW = "improve_view"
    VERIFY_TARGET = "verify_target"
    VERIFY_RELATION = "verify_relation"
    WAIT = "wait"
    STOP = "stop"


class NavigationStatusCode(str, Enum):
    """Status reported by a lower-level navigation controller."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    REACHED = "reached"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    PREEMPTED = "preempted"
    FAILED = "failed"


class EvidenceSource(str, Enum):
    """Origin of a visual or geometric evidence record."""

    LIVE_SENSOR = "live_sensor"
    BAG_REPLAY = "bag_replay"
    PROJECTED_OBJECT = "projected_object"
    VIEWPOINT_CAPTURE = "viewpoint_capture"
    EXTERNAL = "external"


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
    """Synchronized sensor snapshot delivered to STRIVE real-robot runtime."""

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


@dataclass(frozen=True)
class NavigationIntent:
    """Planner-owned intent before it is translated to a robot motion goal."""

    mode: MotionGoalMode
    goal_pose: Optional[Pose3D] = None
    target_object_uid: Optional[str] = None
    anchor_object_uid: Optional[str] = None
    relation_edge_id: Optional[str] = None
    stop_allowed: bool = False
    priority: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_motion_goal(self) -> "MotionGoal":
        """Convert the high-level intent to a lower-level motion request."""

        return MotionGoal(
            mode=self.mode,
            goal_pose=self.goal_pose,
            target_object_uid=self.target_object_uid,
            anchor_object_uid=self.anchor_object_uid,
            relation_edge_id=self.relation_edge_id,
            reason=self.reason,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class MotionGoal:
    """Motion request consumed by a NavigationBridge or MotionController."""

    mode: MotionGoalMode
    goal_pose: Optional[Pose3D] = None
    look_at: Optional[Vector3] = None
    target_object_uid: Optional[str] = None
    anchor_object_uid: Optional[str] = None
    relation_edge_id: Optional[str] = None
    tolerance: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def requires_motion(self) -> bool:
        """Return whether this goal should be sent to the motion layer."""

        return self.mode not in {MotionGoalMode.STOP, MotionGoalMode.WAIT} and self.goal_pose is not None


@dataclass(frozen=True)
class ViewpointGoal:
    """Viewpoint-specific goal used for evidence acquisition."""

    pose: Pose3D
    purpose: MotionGoalMode = MotionGoalMode.IMPROVE_VIEW
    look_at: Optional[Vector3] = None
    target_object_uid: Optional[str] = None
    anchor_object_uid: Optional[str] = None
    relation_edge_id: Optional[str] = None
    evidence_requirements: Dict[str, Any] = field(default_factory=dict)
    tolerance: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_motion_goal(self) -> MotionGoal:
        """Return the executable motion goal for this viewpoint."""

        return MotionGoal(
            mode=self.purpose,
            goal_pose=self.pose,
            look_at=self.look_at,
            target_object_uid=self.target_object_uid,
            anchor_object_uid=self.anchor_object_uid,
            relation_edge_id=self.relation_edge_id,
            tolerance=dict(self.tolerance),
            reason=self.reason,
            metadata={
                **self.metadata,
                "evidence_requirements": self.evidence_requirements,
            },
        )


@dataclass(frozen=True)
class NavigationStatus:
    """Asynchronous execution status from the platform motion layer."""

    status: NavigationStatusCode
    goal_id: Optional[str] = None
    current_pose: Optional[Pose3D] = None
    distance_to_goal: Optional[float] = None
    path_length_remaining: Optional[float] = None
    progress: Optional[float] = None
    stamp: Optional[float] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Return whether the controller finished this goal attempt."""

        return self.status in {
            NavigationStatusCode.REACHED,
            NavigationStatusCode.BLOCKED,
            NavigationStatusCode.TIMEOUT,
            NavigationStatusCode.PREEMPTED,
            NavigationStatusCode.FAILED,
        }

    def succeeded(self) -> bool:
        """Return whether the controller reports goal reachability success."""

        return self.status == NavigationStatusCode.REACHED


@dataclass(frozen=True)
class ViewEvidence:
    """Visual evidence captured after reaching or sampling a viewpoint."""

    source: EvidenceSource
    timestamp: float
    pose: Optional[Pose3D] = None
    image_ref: Optional[str] = None
    camera_model: CameraModel = CameraModel.UNKNOWN
    bbox_xyxy: Optional[BBox2D] = None
    target_object_uid: Optional[str] = None
    anchor_object_uid: Optional[str] = None
    relation_edge_id: Optional[str] = None
    quality: Dict[str, Any] = field(default_factory=dict)
    verifier_payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optional bbox evidence."""

        if self.bbox_xyxy is not None:
            _validate_bbox_xyxy(self.bbox_xyxy)

    def for_verifier(self) -> Dict[str, Any]:
        """Return a prompt/verifier-friendly evidence dictionary."""

        return {
            "source": self.source.value,
            "timestamp": self.timestamp,
            "pose": self.pose.as_dict() if self.pose else None,
            "image_ref": self.image_ref,
            "camera_model": self.camera_model.value,
            "bbox_xyxy": self.bbox_xyxy,
            "target_object_uid": self.target_object_uid,
            "anchor_object_uid": self.anchor_object_uid,
            "relation_edge_id": self.relation_edge_id,
            "quality": self.quality,
            "verifier_payload": self.verifier_payload,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ViewpointResult:
    """Outcome of executing one viewpoint goal and collecting evidence."""

    goal: ViewpointGoal
    status: NavigationStatus
    evidence: Optional[ViewEvidence] = None
    final_pose: Optional[Pose3D] = None
    path_length: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeDecision:
    """Single decision record emitted by the real-robot runtime loop."""

    timestamp: float
    intent: NavigationIntent
    motion_goal: Optional[MotionGoal] = None
    navigation_status: Optional[NavigationStatus] = None
    accepted_candidate_uid: Optional[str] = None
    accepted_relation_edge_id: Optional[str] = None
    verifier_decision: Dict[str, Any] = field(default_factory=dict)
    lower_planner_state: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _validate_bbox_xyxy(box: BBox2D) -> None:
    """Validate a 2-D bounding box in ``x1, y1, x2, y2`` format."""

    if len(box) != 4:
        raise ValueError("bbox must contain exactly four values")
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox must satisfy x2 >= x1 and y2 >= y1")
