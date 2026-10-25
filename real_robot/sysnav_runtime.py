"""Runtime helpers for the first SysNav-backed VLN real-robot loop.

The classes here compose the lower-level ROS adapters with VLN planning
interfaces. They do not own detection or mapping models; SysNav continues to
publish object/room nodes, while VLN consumes snapshots and emits waypoint
goals through the motion bridge.
"""

from __future__ import annotations

import time
from concurrent.futures import Executor, Future
from scene.dependencies import object_dependencies
from verification.images import image_dependencies_valid
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
from real_robot.sysnav_goal_resolver import GoalResolver, PreResolvedGoalResolver
from motion.supervisor import MotionSupervisor, ActiveGoalPoll, motion_goal_signature
from backends.ros.scene import SysNavSemanticMapBridge
from backends.ros.smoke import DryRunMotionController, FirstObjectSmokePolicy
from harness.trace import RuntimeDecisionJsonlWriter, runtime_decision_to_dict, runtime_viewpoint_result_to_dict


class MotionControllerProtocol(Protocol):
    """Minimal motion bridge used by runtime controllers."""

    def send_goal(self, goal: MotionGoal) -> str:
        """Submit one motion goal and return a platform goal id."""

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return current lower-level execution status."""

    def cancel(self, goal_id: Optional[str] = None) -> None:
        """Cancel one active goal, or all goals when no id is provided."""

    def hold(self) -> None:
        """Request a platform-safe hold without publishing velocity directly."""


class EvidenceProviderProtocol(Protocol):
    """Evidence acquisition hook for reached viewpoint goals."""

    def capture(self, goal: ViewpointGoal, status: NavigationStatus) -> ViewEvidence:
        """Capture current RGB/crop/pose evidence for verifier use."""


class FinalVerifierProtocol(Protocol):
    """Verifier hook decoupled from concrete VLM implementation."""

    def verify(self, evidence: ViewEvidence, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return final verifier decision for the captured evidence."""


class InstructionPolicyProtocol(Protocol):
    """High-level VLN policy interface for real-robot snapshots."""

    def decide(self, snapshot: SemanticMapSnapshot, instruction: Optional[str] = None) -> NavigationIntent:
        """Return the next semantic navigation intent."""

    def task_complete(self) -> bool:
        """Read completion from the domain task owner."""


class FinalInstructionVerifierAdapter:
    """Thin adapter from `ViewEvidence` to existing `FinalInstructionVerifier`."""

    def __init__(self, verifier: Optional[Any] = None, vlm: str = "cognav",
                 evidence_dir: str = "outputs/evidence") -> None:
        """Initialize the adapter.

        Args:
            verifier: Optional existing verifier instance.
            vlm: VLM provider used when constructing the default verifier.
            evidence_dir: Run-owned directory retaining model image inputs.
        """

        if verifier is None:
            from instruction_adapter.verifier import FinalInstructionVerifier

            verifier = FinalInstructionVerifier(vlm=vlm, evidence_dir=evidence_dir)
        self.verifier = verifier

    def verify(self, evidence: ViewEvidence, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run existing final instruction verification on reached evidence."""

        candidate = context.get("candidate_instance") or context.get("candidate")
        if isinstance(candidate, dict):
            from instruction_adapter.verifier import CandidateInstance

            allowed = {
                "uid",
                "detector_label",
                "canonical_label",
                "centroid",
                "bbox_2d",
                "confidence",
                "step",
            }
            candidate = CandidateInstance(**{key: candidate[key] for key in allowed if key in candidate})
        if candidate is None:
            # 没有 candidate 就不能让 verifier “猜测”成功目标，直接转成 hard reject 风格结果。
            return {
                "satisfied": False,
                "decision": "reject_candidate",
                "reason": "final verifier skipped: missing candidate_instance",
                "diagnostics": {"missing_candidate": True},
            }

        plan = context.get("instruction_plan")
        raw_instruction = str(context.get("raw_instruction", "") or getattr(plan, "raw_instruction", ""))
        evidence_payload = evidence.for_verifier()
        # ViewEvidence 是平台无关证据；额外 verifier facts 只能通过 context 追加，不读 ROS msg。
        evidence_payload.update(dict(context.get("verifier_evidence") or {}))
        result = self.verifier.verify(
            raw_instruction=raw_instruction,
            instruction_plan=plan,
            candidate=candidate,
            evidence=evidence_payload,
        )
        return result.as_dict() if hasattr(result, "as_dict") else dict(result or {})


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


@dataclass
class RealInstructionRuntimeState:
    """Track observation attempts and pending verification between timer ticks.

    Motion ownership and terminal feedback live in MotionSupervisor.
    """

    pending_verification: Optional[ActiveGoalPoll] = None
    attempted_viewpoint_ids: set[str] = field(default_factory=set)
    attempt_context_signature: Optional[tuple[str, ...]] = None

    def prepare_attempt_context(self, intent: NavigationIntent) -> None:
        """Reset viewpoint attempts when the semantic candidate changes.

        Args:
            intent: New semantic intent produced by the VLN policy.
        """

        signature = (
            str(intent.target_object_uid or ""),
            str(intent.anchor_object_uid or ""),
            str(intent.relation_edge_id or ""),
            str(intent.metadata.get("candidate_uid", "") or ""),
        )
        if self.attempt_context_signature != signature:
            self.attempt_context_signature = signature
            self.attempted_viewpoint_ids.clear()

    def record_viewpoint_attempt(self, motion_goal: MotionGoal) -> None:
        """Record the resolved viewpoint used by one motion attempt."""

        viewpoint_id = str(motion_goal.metadata.get("viewpoint_uid", "") or "").strip()
        if viewpoint_id:
            self.attempted_viewpoint_ids.add(viewpoint_id)



class WaitInstructionPolicy:
    """Conservative policy that always asks the runtime to wait."""

    def __init__(self, reason: str = "no high-level policy configured") -> None:
        """Initialize the policy.

        Args:
            reason: Reason attached to each WAIT intent.
        """

        self.reason = reason

    def task_complete(self) -> bool:
        """An unconfigured task remains pending."""
        return False

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


@dataclass
class SysNavInstructionRuntime:
    """Dispatch VLN navigation intents to SysNav waypoint execution."""

    semantic_map_bridge: SysNavSemanticMapBridge
    high_level_policy: InstructionPolicyProtocol
    motion_controller: MotionControllerProtocol
    now_fn: Callable[[], float] = time.time
    readiness_provider: Optional[Callable[[], RuntimeReadiness]] = None
    viewpoint_evidence_loop: Optional[Any] = None
    goal_resolver: GoalResolver = field(default_factory=PreResolvedGoalResolver)
    state: RealInstructionRuntimeState = field(default_factory=RealInstructionRuntimeState)
    motion: MotionSupervisor = field(init=False)
    inference_executor: Optional[Executor] = None
    _decision_future: Optional[Future] = field(default=None, init=False)
    _verification_future: Optional[Future] = field(default=None, init=False)
    _decision_scope: tuple = field(default=(), init=False)
    _verification_scope: tuple = field(default=(), init=False)
    _current_instruction: Optional[str] = field(default=None, init=False)
    _input_revision: int = field(default=0, init=False)
    _decision_snapshot: Optional[SemanticMapSnapshot] = field(default=None, init=False)
    _verification_dependencies: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Share one motion owner between runtime and viewpoint execution."""
        self.motion = MotionSupervisor(self.motion_controller)
        if self.viewpoint_evidence_loop is not None:
            self.viewpoint_evidence_loop.motion = self.motion

    def clear_hold(self) -> None:
        """Resume high-level decisions after an external safety reset.

        The runtime never clears HOLD from ordinary sensor updates. The safety
        owner or an explicitly authorized operator must call this method after
        the underlying stop condition has been resolved.
        """

        # 中文核心边界：安全恢复必须由安全所有者显式确认，VLN 不根据一帧正常观测自动复位。
        self.motion.resume()

    def _inference_scope(self) -> tuple:
        """Bind slow proposals to the instruction and current control owner."""
        return self._input_revision, self.motion.control_epoch

    def _decide(self, snapshot, instruction):
        """Poll background policy work; inline execution supports deterministic replay."""
        if self.inference_executor is None:
            return self.high_level_policy.decide(snapshot, instruction)
        if self._decision_future is None:
            self._decision_scope = self._inference_scope()
            self._decision_snapshot = snapshot
            self._decision_future = self.inference_executor.submit(
                self.high_level_policy.decide, snapshot, instruction,
            )
        if not self._decision_future.done():
            return None
        future, self._decision_future = self._decision_future, None
        intent = future.result()  # 保留模型/策略异常；由运行入口记录完整调用栈。
        uids = (intent.target_object_uid, intent.anchor_object_uid)
        observed = object_dependencies(self._decision_snapshot, *uids)
        current = object_dependencies(snapshot, *uids)
        if self._decision_scope == self._inference_scope() and observed == current:
            return intent
        return None

    def _verify_reached_goal(
        self,
        active_goal: ActiveGoalPoll,
        snapshot: SemanticMapSnapshot,
    ) -> tuple[Optional[ViewpointResult], Dict[str, Any], Optional[NavigationIntent]]:
        """Run evidence/verifier hooks for a reached motion goal."""

        if not active_goal.status.succeeded() or self.viewpoint_evidence_loop is None:
            # 只有 lower planner 明确 REACHED 后才采集 evidence；blocked/timeout 不能伪造到达视角。
            return None, {}, None
        goal = _viewpoint_goal_from_motion_goal(active_goal.motion_goal)
        context = _verification_context_from_intent(active_goal.intent, active_goal.motion_goal)
        dependencies = object_dependencies(
            snapshot, active_goal.motion_goal.target_object_uid, active_goal.motion_goal.anchor_object_uid,
        )
        context["verifier_evidence"] = {"dependencies": dependencies}
        if self.inference_executor is None:
            viewpoint_result = self.viewpoint_evidence_loop.verify_reached(
                goal, active_goal.status, context=context,
            )
        else:
            if self._verification_future is None:
                self._verification_scope = self._inference_scope()
                self._verification_dependencies = dependencies
                self._verification_future = self.inference_executor.submit(
                    self.viewpoint_evidence_loop.verify_reached,
                    goal, active_goal.status, context=context,
                )
            if not self._verification_future.done():
                return ViewpointResult(
                    goal=goal, status=active_goal.status,
                    reason="verification inference pending",
                    metadata={"evidence_pending": True, "inference_pending": True},
                ), {}, None
            future, self._verification_future = self._verification_future, None
            viewpoint_result = future.result()
            if (self._verification_scope != self._inference_scope()
                    or self._verification_dependencies != dependencies):
                return ViewpointResult(
                    goal=goal, status=active_goal.status,
                    reason="verification context or object geometry changed",
                    metadata={"stale_inference": True},
                ), {}, None
        if bool((viewpoint_result.metadata or {}).get("evidence_pending")):
            # 到达事件已经成立，但传感器尚未产生运动后的新证据；保持同一
            # verification transaction，不能回到 policy 重新发布 waypoint。
            return viewpoint_result, {}, None
        verifier_decision = dict((viewpoint_result.metadata or {}).get("verifier_decision") or {})
        if _policy_handles_viewpoint_result(self.high_level_policy, viewpoint_result, active_goal.intent):
            # policy adapter 写回 ledger/execution state 后，要清掉 completed suppression，让下一轮重新决策。
            self.motion.clear_completed()
        if not _verifier_accepts(verifier_decision):
            return viewpoint_result, verifier_decision, None
        if not self.high_level_policy.task_complete():
            return viewpoint_result, verifier_decision, None
        # 目标验证通过后，由任务 reducer 的数量与顺序规则授予 STOP。
        stop_intent = NavigationIntent(
            mode=MotionGoalMode.STOP,
            stop_allowed=True,
            reason=str(verifier_decision.get("reason", "final verifier accepted reached viewpoint")),
            metadata={
                "source": "final_verifier",
                "verifier_decision": verifier_decision,
                "target_object_uid": active_goal.motion_goal.target_object_uid,
                "anchor_object_uid": active_goal.motion_goal.anchor_object_uid,
                "relation_edge_id": active_goal.motion_goal.relation_edge_id,
                "candidate_uid": active_goal.intent.metadata.get("candidate_uid"),
            },
        )
        self.motion.clear_completed()
        return viewpoint_result, verifier_decision, stop_intent

    def step(self, instruction: Optional[str] = None) -> RuntimeDecision:
        """Run one real-robot high-level decision step."""

        if instruction != self._current_instruction:
            self._input_revision += 1
            self._current_instruction = instruction
        if self.readiness_provider is not None:
            readiness = self.readiness_provider()
            if not readiness.ready:
                # 输入不完整时只返回 WAIT，绝不调用 policy 或 motion controller。
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
            # object snapshot 是语义规划的最低输入；没有对象图时不应下发任何 waypoint。
            wait_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="waiting for SysNav /object_nodes_list",
            )
            return RuntimeDecision(
                timestamp=self.now_fn(),
                intent=wait_intent,
                reason="waiting for SysNav semantic map",
            )

        if self.motion.held:
            hold_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="runtime is held by lower-level safety state",
                metadata={"halted_status": runtime_decision_to_dict(
                    RuntimeDecision(
                        timestamp=snapshot.timestamp,
                        intent=NavigationIntent(mode=MotionGoalMode.WAIT),
                        navigation_status=self.motion.halted_status,
                    )
                ).get("navigation_status")},
            )
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=hold_intent,
                navigation_status=self.motion.halted_status,
                lower_planner_state={"runtime_state": "HOLD"},
                reason=hold_intent.reason,
            )

        if self.state.pending_verification is not None:
            pending = self.state.pending_verification
            viewpoint_result, verifier_decision, stop_intent = self._verify_reached_goal(pending, snapshot)
            if not bool((viewpoint_result.metadata if viewpoint_result else {}).get("evidence_pending")):
                self.state.pending_verification = None
            if stop_intent is not None:
                return RuntimeDecision(
                    timestamp=snapshot.timestamp,
                    intent=stop_intent,
                    navigation_status=pending.status,
                    accepted_candidate_uid=str(
                        pending.intent.metadata.get("candidate_uid")
                        or pending.motion_goal.target_object_uid
                        or pending.motion_goal.anchor_object_uid
                        or ""
                    ),
                    accepted_relation_edge_id=pending.motion_goal.relation_edge_id,
                    verifier_decision=verifier_decision,
                    lower_planner_state={"runtime_state": "STOP", "goal_id": pending.status.goal_id},
                    reason=stop_intent.reason,
                    metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
                )
            verification_pending = self.state.pending_verification is not None
            phase = "VERIFYING" if verification_pending else "IDLE"
            reason = "waiting for post-reach evidence" if verification_pending else viewpoint_result.reason
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=NavigationIntent(
                    mode=MotionGoalMode.WAIT,
                    reason=reason,
                    metadata={"runtime_state": phase},
                ),
                navigation_status=pending.status,
                verifier_decision=verifier_decision,
                lower_planner_state={"runtime_state": phase, "goal_id": pending.status.goal_id},
                reason=reason,
                metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
            )

        active_goal = self.motion.poll_active()
        if active_goal is not None:
            if active_goal.status.is_terminal():
                self.state.record_viewpoint_attempt(active_goal.motion_goal)
                if active_goal.status.status in {
                    NavigationStatusCode.SAFETY_STOP,
                    NavigationStatusCode.MANUAL_TAKEOVER,
                    NavigationStatusCode.LOCALIZATION_LOST,
                }:
                    self.motion.hold(active_goal.status)
                elif active_goal.status.succeeded() and self.viewpoint_evidence_loop is not None:
                    self.state.pending_verification = active_goal
            # 有 active goal 时本轮只跟踪执行状态，不重复调用高层 policy 生成新 waypoint。
            viewpoint_result, verifier_decision, stop_intent = self._verify_reached_goal(active_goal, snapshot)
            if active_goal.status.succeeded() and self.viewpoint_evidence_loop is not None and bool(
                (viewpoint_result.metadata if viewpoint_result else {}).get("evidence_pending")
            ):
                self.state.pending_verification = active_goal
            else:
                self.state.pending_verification = None
            if stop_intent is not None:
                return RuntimeDecision(
                    timestamp=snapshot.timestamp,
                    intent=stop_intent,
                    motion_goal=None,
                    navigation_status=active_goal.status,
                    accepted_candidate_uid=str(
                        active_goal.intent.metadata.get("candidate_uid")
                        or active_goal.motion_goal.target_object_uid
                        or active_goal.motion_goal.anchor_object_uid
                        or ""
                    ),
                    accepted_relation_edge_id=active_goal.motion_goal.relation_edge_id,
                    verifier_decision=verifier_decision,
                    lower_planner_state={
                        "tracking_active_goal": False,
                        "goal_id": active_goal.status.goal_id,
                        "snapshot_source": snapshot.source,
                        "object_count": len(snapshot.objects),
                        "room_count": len(snapshot.rooms),
                    },
                    reason=stop_intent.reason,
                    metadata={
                        "viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)
                        if viewpoint_result is not None
                        else {},
                    },
                )
            if bool((viewpoint_result.metadata if viewpoint_result else {}).get("evidence_pending")):
                return RuntimeDecision(
                    timestamp=snapshot.timestamp,
                    intent=NavigationIntent(
                        mode=MotionGoalMode.WAIT,
                        reason="waiting for post-reach evidence",
                        metadata={"runtime_state": "VERIFYING"},
                    ),
                    navigation_status=active_goal.status,
                    verifier_decision=verifier_decision,
                    lower_planner_state={
                        "runtime_state": "VERIFYING",
                        "goal_id": active_goal.status.goal_id,
                    },
                    reason="waiting for post-reach evidence",
                    metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
                )
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=active_goal.intent,
                motion_goal=active_goal.motion_goal,
                navigation_status=active_goal.status,
                verifier_decision=verifier_decision,
                lower_planner_state={
                    "tracking_active_goal": active_goal.tracking_active,
                    "goal_id": active_goal.status.goal_id,
                    "snapshot_source": snapshot.source,
                    "object_count": len(snapshot.objects),
                    "room_count": len(snapshot.rooms),
                },
                reason=active_goal.status.message or active_goal.intent.reason,
                metadata={
                    "viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)
                    if viewpoint_result is not None
                    else {},
                },
            )

        intent = self._decide(snapshot, instruction)
        if intent is None:
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=NavigationIntent(mode=MotionGoalMode.WAIT, reason="policy inference pending"),
                lower_planner_state={"runtime_state": "THINKING"},
                reason="policy inference pending",
            )
        self.state.prepare_attempt_context(intent)
        motion_goal = self.goal_resolver.resolve(
            intent,
            snapshot,
            set(self.state.attempted_viewpoint_ids),
        )
        if intent.mode not in {MotionGoalMode.WAIT, MotionGoalMode.STOP} and motion_goal is None:
            wait_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="no SysNav executable viewpoint available",
                metadata={
                    **dict(intent.metadata or {}),
                    "runtime_state": "IDLE",
                    "viewpoint_provider_unavailable": True,
                },
            )
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=wait_intent,
                lower_planner_state={"runtime_state": "IDLE", "viewpoint_provider_unavailable": True},
                reason=wait_intent.reason,
            )
        if motion_goal is None:
            motion_goal = intent.to_motion_goal()
        if intent.mode == MotionGoalMode.WAIT:
            # WAIT 是高层语义等待状态，不进入 motion controller，避免生成无意义 goal_id。
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=intent,
                lower_planner_state={
                    "snapshot_source": snapshot.source,
                    "object_count": len(snapshot.objects),
                    "room_count": len(snapshot.rooms),
                    "policy_wait": True,
                },
                reason=intent.reason,
            )
        if self.motion.should_suppress_completed(motion_goal):
            # 同一目标已到达但 verifier/policy 尚未推进时，避免 timer tick 反复发布同一个 waypoint。
            wait_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="motion goal already reached; waiting for verifier or policy state update",
                metadata={
                    "completed_goal_signature": self.motion.last_completed_signature,
                    "last_terminal_status": runtime_decision_to_dict(
                        RuntimeDecision(
                            timestamp=snapshot.timestamp,
                            intent=intent,
                            navigation_status=self.motion.last_terminal_status,
                        )
                    ).get("navigation_status"),
                },
            )
            return RuntimeDecision(
                timestamp=snapshot.timestamp,
                intent=wait_intent,
                lower_planner_state={
                    "snapshot_source": snapshot.source,
                    "object_count": len(snapshot.objects),
                    "room_count": len(snapshot.rooms),
                    "suppressed_completed_goal": True,
                },
                reason=wait_intent.reason,
            )
        goal_id = self.motion.submit(intent, motion_goal)
        self.state.record_viewpoint_attempt(motion_goal)
        status = self.motion.poll_active().status
        viewpoint_result: Optional[ViewpointResult] = None
        verifier_decision: Dict[str, Any] = {}
        if motion_goal.requires_motion() and status.status in {
            NavigationStatusCode.QUEUED,
            NavigationStatusCode.RUNNING,
        }:
            # 异步执行目标进入 active state，后续 tick 只 poll status。
            pass  # MotionSupervisor 已持有运行中的目标。
        elif motion_goal.requires_motion() and status.is_terminal():
            # 有些 fake/dry-run 或快速 planner 会第一次 poll 就 terminal，这条路径也要走 verifier。
            immediate_terminal = ActiveGoalPoll(
                status=status,
                motion_goal=motion_goal,
                intent=intent,
                tracking_active=False,
            )
            self.state.record_viewpoint_attempt(motion_goal)
            if status.status in {
                NavigationStatusCode.SAFETY_STOP,
                NavigationStatusCode.MANUAL_TAKEOVER,
                NavigationStatusCode.LOCALIZATION_LOST,
            }:
                self.motion.hold(status)
            if status.succeeded() and self.viewpoint_evidence_loop is not None:
                self.state.pending_verification = immediate_terminal
            viewpoint_result, verifier_decision, stop_intent = self._verify_reached_goal(
                immediate_terminal, snapshot
            )
            if not bool((viewpoint_result.metadata if viewpoint_result else {}).get("evidence_pending")):
                self.state.pending_verification = None
            if stop_intent is not None:
                return RuntimeDecision(
                    timestamp=snapshot.timestamp,
                    intent=stop_intent,
                    motion_goal=None,
                    navigation_status=status,
                    accepted_candidate_uid=str(
                        intent.metadata.get("candidate_uid")
                        or motion_goal.target_object_uid
                        or motion_goal.anchor_object_uid
                        or ""
                    ),
                    accepted_relation_edge_id=motion_goal.relation_edge_id,
                    verifier_decision=verifier_decision,
                    lower_planner_state={
                        "tracking_active_goal": False,
                        "goal_id": status.goal_id,
                        "snapshot_source": snapshot.source,
                        "object_count": len(snapshot.objects),
                        "room_count": len(snapshot.rooms),
                    },
                    reason=stop_intent.reason,
                    metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
                )
            if bool((viewpoint_result.metadata if viewpoint_result else {}).get("evidence_pending")):
                return RuntimeDecision(
                    timestamp=snapshot.timestamp,
                    intent=NavigationIntent(
                        mode=MotionGoalMode.WAIT,
                        reason="waiting for post-reach evidence",
                        metadata={"runtime_state": "VERIFYING"},
                    ),
                    motion_goal=None,
                    navigation_status=status,
                    verifier_decision=verifier_decision,
                    lower_planner_state={
                        "runtime_state": "VERIFYING",
                        "goal_id": status.goal_id,
                    },
                    reason="waiting for post-reach evidence",
                    metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
                )

        # 核心：VLN 输出 NavigationIntent，RosWaypointController 负责变成 /way_point。
        return RuntimeDecision(
            timestamp=snapshot.timestamp,
            intent=intent,
            motion_goal=motion_goal,
            navigation_status=status,
            verifier_decision=verifier_decision,
            lower_planner_state={
                "goal_id": goal_id,
                "snapshot_source": snapshot.source,
                "object_count": len(snapshot.objects),
                "room_count": len(snapshot.rooms),
            },
            reason=intent.reason,
            metadata={"viewpoint_result": runtime_viewpoint_result_to_dict(viewpoint_result)},
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
    motion: MotionSupervisor = field(init=False)

    def __post_init__(self):
        """Create an owner for standalone use; runtime supplies its shared owner."""
        self.motion = MotionSupervisor(self.motion_controller)

    def run(self, goal: ViewpointGoal, context: Optional[Dict[str, Any]] = None) -> ViewpointResult:
        """Execute ``ViewpointGoal -> /way_point -> wait reached -> evidence -> verifier``."""

        started_at = self.now_fn()
        motion_goal = goal.as_motion_goal()
        intent = NavigationIntent(mode=motion_goal.mode, goal_pose=motion_goal.goal_pose)
        goal_id = self.motion.submit(intent, motion_goal)
        status = self.motion.poll_active().status
        poll_count = 1

        while not status.is_terminal():
            elapsed = self.now_fn() - started_at
            if elapsed >= self.timeout_s:
                self.motion.cancel()
                status = NavigationStatus(
                    NavigationStatusCode.TIMEOUT,
                    goal_id=goal_id,
                    message="viewpoint execution timed out before evidence acquisition",
                    metadata={"elapsed_s": elapsed, "poll_count": poll_count},
                )
                break
            self.sleep_fn(self.poll_interval_s)
            status = self.motion.poll_active().status
            poll_count += 1

        result = self.verify_reached(goal, status, context=context)
        return ViewpointResult(
            goal=result.goal,
            status=result.status,
            evidence=result.evidence,
            final_pose=result.final_pose,
            path_length=result.path_length,
            reason=result.reason,
            metadata={
                **dict(result.metadata or {}),
                "goal_id": goal_id,
                "poll_count": poll_count,
            },
        )

    def verify_reached(
        self,
        goal: ViewpointGoal,
        status: NavigationStatus,
        context: Optional[Dict[str, Any]] = None,
    ) -> ViewpointResult:
        """Capture evidence and run verifier after motion already reached.

        Args:
            goal: Reached viewpoint goal.
            status: Navigation status for the reached goal.
            context: Optional verifier context.

        Returns:
            Viewpoint result. Evidence is only captured when status succeeded.
        """

        if not status.succeeded():
            # 核心：只有运动层确认到达后才采集 final verifier 证据；blocked/timeout 不能伪造成功视角。
            return ViewpointResult(
                goal=goal,
                status=status,
                reason=f"viewpoint motion did not reach target: {status.status.value}",
                metadata={"goal_id": status.goal_id},
            )

        capture_after = getattr(self.evidence_provider, "capture_after", None)
        evidence = (
            capture_after(goal, status)
            if callable(capture_after)
            else self.evidence_provider.capture(goal, status)
        )
        if evidence is None:
            return ViewpointResult(
                goal=goal,
                status=status,
                reason="waiting for post-reach evidence",
                metadata={"goal_id": status.goal_id, "evidence_pending": True},
            )
        verifier_decision: Dict[str, Any] = {}
        if self.final_verifier is not None:
            # VLM 只评估当前证据是否满足任务；物理到达状态来自 motion_controller。
            # 这里不读取 ROS topic；provider 已经把当前观测封装成 ViewEvidence。
            verifier_decision = self.final_verifier.verify(
                evidence,
                {
                    **(context or {}),
                    "goal_id": status.goal_id,
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
                "goal_id": status.goal_id,
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

    def capture_after(self, goal: ViewpointGoal, status: NavigationStatus) -> Optional[ViewEvidence]:
        """Capture only when the latest observation is newer than the reach event."""

        observation = self.observation_provider()
        if status.stamp is not None:
            if observation is None or float(getattr(observation, "timestamp", 0.0)) <= float(status.stamp):
                return None
        return self.capture(goal, status)


def _viewpoint_goal_from_motion_goal(goal: MotionGoal) -> ViewpointGoal:
    """Convert a reached motion goal into a verifier viewpoint goal."""

    return ViewpointGoal(
        pose=goal.goal_pose or Pose3D(position=(0.0, 0.0, 0.0)),
        purpose=goal.mode if goal.mode in {
            MotionGoalMode.GO_TO_OBJECT,
            MotionGoalMode.GO_TO_ANCHOR,
            MotionGoalMode.IMPROVE_VIEW,
            MotionGoalMode.VERIFY_TARGET,
            MotionGoalMode.VERIFY_RELATION,
        } else MotionGoalMode.VERIFY_TARGET,
        look_at=goal.look_at,
        target_object_uid=goal.target_object_uid,
        anchor_object_uid=goal.anchor_object_uid,
        relation_edge_id=goal.relation_edge_id,
        tolerance=dict(goal.tolerance),
        reason=goal.reason,
        metadata=dict(goal.metadata),
    )


def _verification_context_from_intent(intent: NavigationIntent, goal: MotionGoal) -> Dict[str, Any]:
    """Build final verifier context from the active intent metadata."""

    metadata = dict(intent.metadata or {})
    return {
        "raw_instruction": metadata.get("raw_instruction", ""),
        "instruction_plan": metadata.get("instruction_plan"),
        "candidate_instance": metadata.get("candidate_instance"),
        "candidate_uid": metadata.get("candidate_uid"),
        "selected_snapshot_uid": metadata.get("selected_snapshot_uid"),
        "selection_role": metadata.get("selection_role"),
        "target_object_uid": goal.target_object_uid,
        "anchor_object_uid": goal.anchor_object_uid,
    }


def _policy_handles_viewpoint_result(policy: Any, result: ViewpointResult, intent: NavigationIntent) -> bool:
    """Notify policy adapters about verifier feedback when supported."""

    handler = getattr(policy, "handle_viewpoint_result", None)
    if handler is None:
        return False
    return bool(handler(result, intent))


def _verifier_accepts(payload: Dict[str, Any]) -> bool:
    """Read explicit semantic and stop-view confirmation from the verifier."""

    if not payload:
        return False
    return (
        payload.get("satisfied") is True and payload.get("decision") == "accept"
        and payload.get("semantic_satisfied") is True
        and payload.get("view_sufficient_for_stop") is True
        and image_dependencies_valid(payload.get("dependencies", {}))
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
