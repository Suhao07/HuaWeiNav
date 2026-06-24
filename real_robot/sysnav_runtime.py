"""Runtime helpers for the first SysNav-backed STRIVE real-robot loop.

The classes here compose the lower-level ROS adapters with STRIVE planning
interfaces. They do not own detection or mapping models; SysNav continues to
publish object/room nodes, while STRIVE consumes snapshots and emits waypoint
goals through the motion bridge.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from real_robot.contracts import (
    CameraModel,
    EvidenceSource,
    MotionGoal,
    MotionGoalMode,
    NavigationIntent,
    NavigationStatus,
    NavigationStatusCode,
    Pose3D,
    RuntimeDecision,
    SemanticMapSnapshot,
    ViewEvidence,
    ViewpointGoal,
    ViewpointResult,
)
from real_robot.sysnav_ros_adapters import (
    RosObjectNodeAdapter,
    RosRoomNodeAdapter,
    build_semantic_map_snapshot,
)


class MotionControllerProtocol(Protocol):
    """Minimal motion bridge used by runtime controllers."""

    def send_goal(self, goal: MotionGoal) -> str:
        """Submit one motion goal and return a platform goal id."""

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return current lower-level execution status."""


class EvidenceProviderProtocol(Protocol):
    """Evidence acquisition hook for reached viewpoint goals."""

    def capture(self, goal: ViewpointGoal, status: NavigationStatus) -> ViewEvidence:
        """Capture current RGB/crop/pose evidence for verifier use."""


class FinalVerifierProtocol(Protocol):
    """Verifier hook decoupled from concrete VLM implementation."""

    def verify(self, evidence: ViewEvidence, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return final verifier decision for the captured evidence."""


class InstructionPolicyProtocol(Protocol):
    """High-level STRIVE policy interface for real-robot snapshots."""

    def decide(self, snapshot: SemanticMapSnapshot, instruction: Optional[str] = None) -> NavigationIntent:
        """Return the next semantic navigation intent."""


@dataclass(frozen=True)
class RuntimeReadiness:
    """Readiness state for a live real-robot runtime tick.

    Args:
        ready: Whether the runtime has enough synchronized inputs to make a
            high-level decision.
        reason: Human-readable reason used when the runtime must wait.
        metadata: JSON-friendly diagnostics such as topic availability.
    """

    ready: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly readiness record."""

        return {
            "ready": self.ready,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class DryRunMotionController:
    """Motion controller that records goals without publishing robot commands.

    This controller is used by live ROS dry-run and bag replay. It preserves the
    `MotionGoal -> NavigationStatus` contract while guaranteeing that no
    waypoint is sent to the lower planner.
    """

    def __init__(self, status_code: NavigationStatusCode = NavigationStatusCode.IDLE) -> None:
        """Initialize the dry-run controller.

        Args:
            status_code: Status returned for motion-requiring goals. WAIT and
                STOP keep their non-motion status semantics.
        """

        self.status_code = status_code
        self.goals: list[MotionGoal] = []
        self._statuses: Dict[str, NavigationStatus] = {}

    def send_goal(self, goal: MotionGoal) -> str:
        """Record one goal and return a dry-run goal id.

        Args:
            goal: Motion request produced by STRIVE high-level policy.

        Returns:
            Stable dry-run goal id for this recorded request.
        """

        goal_id = f"dry_run_goal:{uuid.uuid4().hex}"
        self.goals.append(goal)
        if goal.mode == MotionGoalMode.STOP:
            status = NavigationStatusCode.REACHED
        elif not goal.requires_motion():
            status = NavigationStatusCode.IDLE
        else:
            status = self.status_code
        self._statuses[goal_id] = NavigationStatus(
            status=status,
            goal_id=goal_id,
            message=f"dry-run recorded {goal.mode.value}; no waypoint published",
            metadata={
                "dry_run": True,
                "requires_motion": goal.requires_motion(),
                "target_object_uid": goal.target_object_uid,
                "anchor_object_uid": goal.anchor_object_uid,
            },
        )
        return goal_id

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return the dry-run status for a recorded goal id.

        Args:
            goal_id: Goal id returned by `send_goal`.

        Returns:
            Recorded `NavigationStatus`, or FAILED for unknown goal ids.
        """

        return self._statuses.get(
            goal_id,
            NavigationStatus(
                NavigationStatusCode.FAILED,
                goal_id=goal_id,
                message="unknown dry-run goal id",
                metadata={"dry_run": True},
            ),
        )


class WaitInstructionPolicy:
    """Conservative policy that always asks the runtime to wait."""

    def __init__(self, reason: str = "no high-level policy configured") -> None:
        """Initialize the policy.

        Args:
            reason: Reason attached to each WAIT intent.
        """

        self.reason = reason

    def decide(self, snapshot: SemanticMapSnapshot, instruction: Optional[str] = None) -> NavigationIntent:
        """Return a WAIT intent with snapshot diagnostics.

        Args:
            snapshot: Current semantic map snapshot.
            instruction: Optional raw instruction.

        Returns:
            WAIT `NavigationIntent`.
        """

        return NavigationIntent(
            mode=MotionGoalMode.WAIT,
            reason=self.reason,
            metadata={
                "instruction": instruction or "",
                "object_count": len(snapshot.objects),
                "room_count": len(snapshot.rooms),
                "policy": "wait",
            },
        )


class FirstObjectSmokePolicy:
    """Smoke-test policy that targets the first object with a 3-D position.

    This policy is only for runtime wiring validation. It does not interpret
    natural language and must not be used as the final semantic navigation
    policy.
    """

    def decide(self, snapshot: SemanticMapSnapshot, instruction: Optional[str] = None) -> NavigationIntent:
        """Return a GO_TO_OBJECT intent for the first positioned object.

        Args:
            snapshot: Current semantic map snapshot.
            instruction: Optional raw instruction, recorded only for debug.

        Returns:
            `NavigationIntent` for a smoke-test object, or WAIT when no object
            has a usable position.
        """

        for obj in snapshot.objects:
            if obj.position is None:
                continue
            return NavigationIntent(
                mode=MotionGoalMode.GO_TO_OBJECT,
                goal_pose=Pose3D(
                    position=obj.position,
                    frame_id=snapshot.robot_pose.frame_id,
                    stamp=snapshot.timestamp,
                ),
                target_object_uid=obj.uid,
                reason=f"smoke policy selected first positioned object: {obj.label}",
                metadata={
                    "instruction": instruction or "",
                    "policy": "first_object_smoke",
                    "object_label": obj.label,
                },
            )
        return NavigationIntent(
            mode=MotionGoalMode.WAIT,
            reason="no positioned object available for first_object_smoke policy",
            metadata={
                "instruction": instruction or "",
                "policy": "first_object_smoke",
                "object_count": len(snapshot.objects),
            },
        )


@dataclass
class SysNavSemanticMapBridge:
    """Cache SysNav object/room node topics and expose STRIVE map snapshots."""

    robot_pose_provider: Callable[[], Pose3D]
    object_adapter: RosObjectNodeAdapter = field(default_factory=RosObjectNodeAdapter)
    room_adapter: RosRoomNodeAdapter = field(default_factory=RosRoomNodeAdapter)
    latest_object_list_msg: Optional[Any] = None
    latest_room_list_msg: Optional[Any] = None

    def update_object_nodes(self, msg: Any) -> None:
        """Store the latest SysNav ``/object_nodes_list`` message."""

        # 核心：SysNav semantic_mapping_node 是对象图唯一写入方，STRIVE runtime 只缓存只读消息。
        self.latest_object_list_msg = msg

    def update_room_nodes(self, msg: Any) -> None:
        """Store the latest SysNav ``/room_nodes_list`` message."""

        self.latest_room_list_msg = msg

    def has_object_snapshot(self) -> bool:
        """Return whether at least one object list message has arrived."""

        return self.latest_object_list_msg is not None

    def build_snapshot(self, timestamp: Optional[float] = None) -> Optional[SemanticMapSnapshot]:
        """Build a STRIVE semantic map snapshot from cached SysNav messages."""

        if self.latest_object_list_msg is None:
            return None
        return build_semantic_map_snapshot(
            object_list_msg=self.latest_object_list_msg,
            room_list_msg=self.latest_room_list_msg,
            robot_pose=self.robot_pose_provider(),
            timestamp=timestamp,
            object_adapter=self.object_adapter,
            room_adapter=self.room_adapter,
        )

    def create_ros_subscriptions(
        self,
        node: Any,
        object_node_list_type: Any,
        room_node_list_type: Any,
        object_topic: str = "/object_nodes_list",
        room_topic: str = "/room_nodes_list",
        queue_size: int = 10,
    ) -> Dict[str, Any]:
        """Register ROS subscriptions on a provided node and return handles."""

        # ROS 类型由调用方注入，避免非 ROS 环境 import runtime 时失败。
        return {
            "object_nodes": node.create_subscription(
                object_node_list_type,
                object_topic,
                self.update_object_nodes,
                queue_size,
            ),
            "room_nodes": node.create_subscription(
                room_node_list_type,
                room_topic,
                self.update_room_nodes,
                queue_size,
            ),
        }


@dataclass
class SysNavInstructionRuntime:
    """Dispatch STRIVE navigation intents to SysNav waypoint execution."""

    semantic_map_bridge: SysNavSemanticMapBridge
    high_level_policy: InstructionPolicyProtocol
    motion_controller: MotionControllerProtocol
    now_fn: Callable[[], float] = time.time
    readiness_provider: Optional[Callable[[], RuntimeReadiness]] = None

    def step(self, instruction: Optional[str] = None) -> RuntimeDecision:
        """Run one real-robot high-level decision step."""

        if self.readiness_provider is not None:
            readiness = self.readiness_provider()
            if not readiness.ready:
                wait_intent = NavigationIntent(
                    mode=MotionGoalMode.WAIT,
                    reason=readiness.reason or "waiting for live runtime inputs",
                    metadata={"readiness": readiness.as_dict()},
                )
                return RuntimeDecision(
                    timestamp=self.now_fn(),
                    intent=wait_intent,
                    lower_planner_state={"readiness": readiness.as_dict()},
                    reason=wait_intent.reason,
                )

        snapshot = self.semantic_map_bridge.build_snapshot(timestamp=self.now_fn())
        if snapshot is None:
            wait_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="waiting for SysNav /object_nodes_list",
            )
            return RuntimeDecision(
                timestamp=self.now_fn(),
                intent=wait_intent,
                reason="waiting for SysNav semantic map",
            )

        intent = self.high_level_policy.decide(snapshot, instruction)
        motion_goal = intent.to_motion_goal()
        goal_id = self.motion_controller.send_goal(motion_goal)
        status = self.motion_controller.poll_status(goal_id)

        # 核心：STRIVE 输出 NavigationIntent，RosWaypointController 负责变成 /way_point。
        return RuntimeDecision(
            timestamp=snapshot.timestamp,
            intent=intent,
            motion_goal=motion_goal,
            navigation_status=status,
            lower_planner_state={
                "goal_id": goal_id,
                "snapshot_source": snapshot.source,
                "object_count": len(snapshot.objects),
                "room_count": len(snapshot.rooms),
            },
            reason=intent.reason,
        )


@dataclass
class ViewpointEvidenceLoop:
    """Execute a viewpoint goal and verify evidence after the robot reaches it."""

    motion_controller: MotionControllerProtocol
    evidence_provider: EvidenceProviderProtocol
    final_verifier: Optional[FinalVerifierProtocol] = None
    poll_interval_s: float = 0.2
    timeout_s: float = 30.0
    now_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep

    def run(self, goal: ViewpointGoal, context: Optional[Dict[str, Any]] = None) -> ViewpointResult:
        """Execute ``ViewpointGoal -> /way_point -> wait reached -> evidence -> verifier``."""

        started_at = self.now_fn()
        motion_goal = goal.as_motion_goal()
        goal_id = self.motion_controller.send_goal(motion_goal)
        status = self.motion_controller.poll_status(goal_id)
        poll_count = 1

        while not status.is_terminal():
            elapsed = self.now_fn() - started_at
            if elapsed >= self.timeout_s:
                status = NavigationStatus(
                    NavigationStatusCode.TIMEOUT,
                    goal_id=goal_id,
                    message="viewpoint execution timed out before evidence acquisition",
                    metadata={"elapsed_s": elapsed, "poll_count": poll_count},
                )
                break
            self.sleep_fn(self.poll_interval_s)
            status = self.motion_controller.poll_status(goal_id)
            poll_count += 1

        if not status.succeeded():
            # 核心：只有运动层确认到达后才采集 final verifier 证据；blocked/timeout 不能伪造成功视角。
            return ViewpointResult(
                goal=goal,
                status=status,
                reason=f"viewpoint motion did not reach target: {status.status.value}",
                metadata={"goal_id": goal_id, "poll_count": poll_count},
            )

        evidence = self.evidence_provider.capture(goal, status)
        verifier_decision: Dict[str, Any] = {}
        if self.final_verifier is not None:
            # VLM 只评估当前证据是否满足任务；物理到达状态来自 motion_controller。
            verifier_decision = self.final_verifier.verify(
                evidence,
                {
                    **(context or {}),
                    "goal_id": goal_id,
                    "motion_status": status.status.value,
                    "viewpoint_goal": _viewpoint_goal_summary(goal),
                },
            )

        return ViewpointResult(
            goal=goal,
            status=status,
            evidence=evidence,
            final_pose=status.current_pose,
            path_length=status.metadata.get("path_length") if isinstance(status.metadata, dict) else None,
            reason=str(verifier_decision.get("reason", status.message)),
            metadata={
                "goal_id": goal_id,
                "poll_count": poll_count,
                "verifier_decision": verifier_decision,
            },
        )


@dataclass
class LatestObservationEvidenceProvider:
    """Build viewpoint evidence from the latest cached observation and object crop."""

    observation_provider: Callable[[], Optional[Any]]
    crop_provider: Optional[Callable[[ViewpointGoal, Any], Dict[str, Any]]] = None

    def capture(self, goal: ViewpointGoal, status: NavigationStatus) -> ViewEvidence:
        """Capture the current RGB/crop reference after a viewpoint is reached."""

        observation = self.observation_provider()
        camera = observation.primary_camera() if observation is not None and hasattr(observation, "primary_camera") else None
        crop_payload: Dict[str, Any] = {}
        if self.crop_provider is not None:
            crop_payload = self.crop_provider(goal, observation) or {}

        # 核心：evidence 只记录引用和结构化质量信息，图像裁剪本身由 provider/runtime 管理。
        return ViewEvidence(
            source=EvidenceSource.VIEWPOINT_CAPTURE,
            timestamp=float(getattr(observation, "timestamp", time.time()) if observation is not None else time.time()),
            pose=status.current_pose or getattr(observation, "robot_pose", None),
            image_ref=crop_payload.get("image_ref") or getattr(camera, "image_ref", None),
            camera_model=_coerce_camera_model(getattr(camera, "camera_model", None) or crop_payload.get("camera_model")),
            bbox_xyxy=crop_payload.get("bbox_xyxy"),
            target_object_uid=goal.target_object_uid,
            anchor_object_uid=goal.anchor_object_uid,
            relation_edge_id=goal.relation_edge_id,
            quality=dict(crop_payload.get("quality") or {}),
            verifier_payload=dict(crop_payload.get("verifier_payload") or {}),
            metadata={
                "motion_status": status.status.value,
                "evidence_requirements": goal.evidence_requirements,
                "observation_available": observation is not None,
                **dict(crop_payload.get("metadata") or {}),
            },
        )


class RuntimeDecisionJsonlWriter:
    """Append real-robot runtime decisions to a JSONL log file."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the writer and create parent directories.

        Args:
            path: Destination JSONL file path.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: RuntimeDecision) -> Dict[str, Any]:
        """Append one decision to disk.

        Args:
            decision: Runtime decision emitted by `SysNavInstructionRuntime`.

        Returns:
            JSON-friendly dictionary that was written.
        """

        payload = runtime_decision_to_dict(decision)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload


def runtime_decision_to_dict(decision: RuntimeDecision) -> Dict[str, Any]:
    """Return a JSON-friendly runtime decision dictionary.

    Args:
        decision: Runtime decision emitted by the real-robot loop.

    Returns:
        JSON-serializable dictionary.
    """

    return _json_ready(decision)


def _json_ready(value: Any) -> Any:
    """Convert dataclasses, enums, tuples, and paths into JSON-ready values."""

    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _viewpoint_goal_summary(goal: ViewpointGoal) -> Dict[str, Any]:
    """Return a compact JSON-friendly viewpoint goal summary."""

    return {
        "purpose": goal.purpose.value,
        "pose": goal.pose.as_dict(),
        "look_at": goal.look_at,
        "target_object_uid": goal.target_object_uid,
        "anchor_object_uid": goal.anchor_object_uid,
        "relation_edge_id": goal.relation_edge_id,
        "evidence_requirements": goal.evidence_requirements,
    }


def _coerce_camera_model(value: Any) -> CameraModel:
    """Return a valid camera model enum for evidence payloads."""

    if isinstance(value, CameraModel):
        return value
    if value is None:
        return CameraModel.UNKNOWN
    try:
        return CameraModel(str(value))
    except ValueError:
        return CameraModel.UNKNOWN
