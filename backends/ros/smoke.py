"""Recording-only controller and deterministic policies for ROS wiring checks."""
from __future__ import annotations
import uuid
from typing import Dict, Optional
from real_robot.contracts import (
    MotionGoal, MotionGoalMode, NavigationIntent, NavigationStatus,
    NavigationStatusCode, SemanticMapSnapshot, Pose3D,
)

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
            goal: Motion request produced by VLN high-level policy.

        Returns:
            Stable dry-run goal id for this recorded request.
        """

        goal_id = f"dry_run_goal:{uuid.uuid4().hex}"
        # dry-run 记录完整 MotionGoal，方便 JSONL 复盘，但绝不触发 RosWaypointController。
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

    def cancel(self, goal_id: Optional[str] = None) -> None:
        """Mark one recorded goal, or all recorded goals, as preempted.

        Args:
            goal_id: Optional dry-run goal id. ``None`` selects every record.
        """

        selected_ids = [goal_id] if goal_id is not None else list(self._statuses)
        for selected_id in selected_ids:
            if selected_id not in self._statuses:
                continue
            self._statuses[selected_id] = NavigationStatus(
                NavigationStatusCode.PREEMPTED,
                goal_id=selected_id,
                message="dry-run goal cancelled",
                metadata={"dry_run": True},
            )

    def hold(self) -> None:
        """Place all non-terminal dry-run goals into a safe idle state."""

        for goal_id, status in list(self._statuses.items()):
            if status.is_terminal():
                continue
            self._statuses[goal_id] = NavigationStatus(
                NavigationStatusCode.IDLE,
                goal_id=goal_id,
                message="dry-run safe hold requested",
                metadata={"dry_run": True},
            )


class FirstObjectSmokePolicy:
    """Smoke-test policy that targets the first object with a 3-D position.

    This policy is only for runtime wiring validation. It does not interpret
    natural language and must not be used as the final semantic navigation
    policy.
    """

    def task_complete(self) -> bool:
        """Smoke navigation reports execution evidence only."""
        return False

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
