"""Backend-neutral motion commands and execution feedback."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from scene.contracts import Pose3D, Vector3

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
    SAFETY_STOP = "safety_stop"
    MANUAL_TAKEOVER = "manual_takeover"
    LOCALIZATION_LOST = "localization_lost"



class MotionReasonCode(str, Enum):
    """Stable reason codes for motion-task results and safety transitions."""

    NONE = "none"
    GOAL_REACHED = "goal_reached"
    NO_FEASIBLE_PATH = "no_feasible_path"
    NO_PROGRESS = "no_progress"
    GOAL_TIMEOUT = "goal_timeout"
    ODOMETRY_STALE = "odometry_stale"
    POINTCLOUD_STALE = "pointcloud_stale"
    LOCALIZATION_LOST = "localization_lost"
    MANUAL_TAKEOVER = "manual_takeover"
    ESTOP_ACTIVE = "estop_active"
    CONTROLLER_FAULT = "controller_fault"
    COMMAND_STALE = "command_stale"
    CANCELLED = "cancelled"
    UNKNOWN_GOAL = "unknown_goal"
    INVALID_GOAL = "invalid_goal"
    LOOK_AT_UNSUPPORTED = "look_at_unsupported"
    VIEW_ALIGNMENT_UNAVAILABLE = "view_alignment_unavailable"
    VIEW_ALIGNMENT_FAILED = "view_alignment_failed"



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

        # 核心：planner 输出的是语义意图，bridge 才负责翻译成具体平台可执行的 motion goal。
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
    action: Optional[int] = None

    def requires_motion(self) -> bool:
        """Return whether this goal should be sent to the motion layer."""

        # STOP/WAIT 是高层控制状态，不应该被误发布成底层 waypoint。
        return (self.action is not None and self.action != 0) or (
            self.mode not in {MotionGoalMode.STOP, MotionGoalMode.WAIT} and self.goal_pose is not None
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
    reason_code: MotionReasonCode = MotionReasonCode.NONE
    safety_state: str = "unknown"
    current_velocity: Optional[Vector3] = None

    def is_terminal(self) -> bool:
        """Return whether the controller finished this goal attempt."""

        # 真实机器人是异步执行；terminal 表示这次 motion attempt 已有明确结果。
        return self.status in {
            NavigationStatusCode.REACHED,
            NavigationStatusCode.BLOCKED,
            NavigationStatusCode.TIMEOUT,
            NavigationStatusCode.PREEMPTED,
            NavigationStatusCode.FAILED,
            NavigationStatusCode.SAFETY_STOP,
            NavigationStatusCode.MANUAL_TAKEOVER,
            NavigationStatusCode.LOCALIZATION_LOST,
        }

    def succeeded(self) -> bool:
        """Return whether the controller reports goal reachability success."""

        return self.status == NavigationStatusCode.REACHED
