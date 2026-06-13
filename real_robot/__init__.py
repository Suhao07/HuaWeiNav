"""Real-robot interfaces for STRIVE.

The package starts with platform-independent contracts. Platform adapters
should live in sibling modules and translate ROS/SysNav/Habitat-specific data
into these contracts before invoking STRIVE planning or verification logic.
"""

from real_robot.contracts import (
    BBox2D,
    CameraFrame,
    CameraModel,
    DetectionFrame,
    EvidenceSource,
    FrontierSnapshot,
    MotionGoal,
    MotionGoalMode,
    NavigationIntent,
    NavigationStatus,
    NavigationStatusCode,
    ObjectNodeSnapshot,
    Pose3D,
    QuaternionXYZW,
    RealObservation,
    RoomSnapshot,
    RuntimeDecision,
    SemanticMapSnapshot,
    Vector3,
    ViewEvidence,
    ViewpointGoal,
    ViewpointResult,
    ViewpointSnapshot,
)

__all__ = [
    "BBox2D",
    "CameraFrame",
    "CameraModel",
    "DetectionFrame",
    "EvidenceSource",
    "FrontierSnapshot",
    "MotionGoal",
    "MotionGoalMode",
    "NavigationIntent",
    "NavigationStatus",
    "NavigationStatusCode",
    "ObjectNodeSnapshot",
    "Pose3D",
    "QuaternionXYZW",
    "RealObservation",
    "RoomSnapshot",
    "RuntimeDecision",
    "SemanticMapSnapshot",
    "Vector3",
    "ViewEvidence",
    "ViewpointGoal",
    "ViewpointResult",
    "ViewpointSnapshot",
]
