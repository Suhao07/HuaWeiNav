"""Platform-neutral contracts for VLN real-robot execution.

This module defines value objects exchanged between high-level VLN semantic
planning and platform-specific robot adapters. It intentionally avoids ROS,
Habitat, numpy, OpenCV, or detector imports so the same contracts can be used
by live robots, bag replay, simulation bridges, and unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from motion.contracts import (MotionGoalMode, NavigationStatusCode, MotionReasonCode,
                              NavigationIntent, MotionGoal, NavigationStatus)
from typing import Any, Dict, Optional, Tuple


from scene.contracts import (
    CameraFrame, CameraModel, DetectionFrame, FrontierSnapshot, ObjectNodeSnapshot, Pose3D, RealObservation, RoomSnapshot, SemanticMapSnapshot, ViewpointSnapshot, _validate_bbox_xyxy,
)


Vector3 = Tuple[float, float, float]
QuaternionXYZW = Tuple[float, float, float, float]
BBox2D = Tuple[float, float, float, float]










class EvidenceSource(str, Enum):
    """Origin of a visual or geometric evidence record."""

    LIVE_SENSOR = "live_sensor"
    BAG_REPLAY = "bag_replay"
    PROJECTED_OBJECT = "projected_object"
    VIEWPOINT_CAPTURE = "viewpoint_capture"
    EXTERNAL = "external"
























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

        # 视角目标携带 evidence_requirements，底层只执行 pose，证据是否足够由 verifier 判断。
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

        # 核心：verifier 消费结构化证据，不直接依赖 ROS msg 或相机驱动对象。
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
