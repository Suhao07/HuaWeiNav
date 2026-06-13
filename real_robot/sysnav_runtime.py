"""Runtime helpers for the first SysNav-backed STRIVE real-robot loop.

The classes here compose the lower-level ROS adapters with STRIVE planning
interfaces. They do not own detection or mapping models; SysNav continues to
publish object/room nodes, while STRIVE consumes snapshots and emits waypoint
goals through the motion bridge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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

    def step(self, instruction: Optional[str] = None) -> RuntimeDecision:
        """Run one real-robot high-level decision step."""

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
