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
from real_robot.detector_vocabulary import (
    DetectorLabelEntry,
    DetectorVocabulary,
    DetectorVocabularyAdapter,
    merge_label_provenance,
    vocabulary_context,
)
from real_robot.sysnav_ros_adapters import (
    RosDetectionResultAdapter,
    RosObjectNodeAdapter,
    RosRoomNodeAdapter,
    RosWaypointController,
    SysNavTopicConfig,
    build_semantic_map_snapshot,
)
from real_robot.sysnav_runtime import (
    LatestObservationEvidenceProvider,
    SysNavInstructionRuntime,
    SysNavSemanticMapBridge,
    ViewpointEvidenceLoop,
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
    "DetectorLabelEntry",
    "DetectorVocabulary",
    "DetectorVocabularyAdapter",
    "merge_label_provenance",
    "vocabulary_context",
    "RosDetectionResultAdapter",
    "RosObjectNodeAdapter",
    "RosRoomNodeAdapter",
    "RosWaypointController",
    "SysNavTopicConfig",
    "build_semantic_map_snapshot",
    "LatestObservationEvidenceProvider",
    "SysNavInstructionRuntime",
    "SysNavSemanticMapBridge",
    "ViewpointEvidenceLoop",
]
