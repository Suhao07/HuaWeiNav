"""One motion owner for asynchronous navigation backends.

提交占有执行槽位，控制端终态释放槽位：|M_active(robot)| <= 1。
请求超时保留占有权，查询和取消用于确认实际执行状态。
"""
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from typing import Any, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from motion.contracts import MotionGoal, NavigationIntent, NavigationStatus

@dataclass(frozen=True)
class ActiveGoalPoll:
    """Status snapshot for the currently tracked lower-level goal."""

    status: NavigationStatus
    motion_goal: MotionGoal
    intent: NavigationIntent
    tracking_active: bool


def motion_goal_signature(goal: MotionGoal) -> tuple[Any, ...]:
    """Return a stable identity for suppressing repeated goal dispatches."""

    pose = goal.goal_pose
    position = None
    orientation = None
    frame_id = ""
    if pose is not None:
        position = tuple(round(float(value), 3) for value in pose.position)
        orientation = tuple(round(float(value), 4) for value in pose.orientation_xyzw)
        frame_id = pose.frame_id
    tolerance = tuple(sorted((str(key), round(float(value), 3)) for key, value in goal.tolerance.items()))
    return (
        goal.mode.value,
        position,
        orientation,
        frame_id,
        goal.target_object_uid or "",
        goal.anchor_object_uid or "",
        goal.relation_edge_id or "",
        tolerance,
        goal.action,
    )



class MotionSupervisor:
    """Serialize motion ownership while preserving the backend's feedback.

    Args:
        controller: Backend implementing send_goal, poll_status, cancel and hold.
    """
    def __init__(self, controller):
        self.controller = controller
        self._lock = RLock()
        self.active_goal_id = None
        self.active_motion_goal = None
        self.active_intent = None
        self.active_request_id = None
        self.phase = "idle"
        self.control_epoch = 0
        self.held = False
        self.halted_status = None
        self.last_completed_signature = None
        self.last_terminal_status = None
        self._requests = {}

    def submit(self, intent, goal, *, request_id=None, control_epoch=None):
        """Reserve ownership, dispatch once, and return the backend goal ID.

        Args:
            intent: Semantic context retained for completion feedback.
            goal: Backend motion goal.
            request_id: Stable ID reused by a caller retrying the same request.
            control_epoch: Control version observed by the proposal producer.

        Raises:
            RuntimeError: Ownership is held or the proposal uses an old epoch.
            ValueError: A request ID is reused for another goal.
            Exception: Original backend submission error; ownership stays unknown.
        """
        request_id = request_id or str(uuid4())
        signature = motion_goal_signature(goal)
        with self._lock:
            if request_id in self._requests:
                previous_signature, goal_id = self._requests[request_id]
                if previous_signature != signature:
                    raise ValueError("Request ID already belongs to another goal")
                if goal_id is None:
                    raise RuntimeError("Submission outcome is unknown; reconcile backend state")
                return goal_id
            if control_epoch is not None and control_epoch != self.control_epoch:
                raise RuntimeError("Motion proposal has an outdated control epoch")
            if self.held:
                raise RuntimeError("Motion is held; explicit resume is required")
            if self.active_request_id is not None:
                raise RuntimeError("Motion ownership is held by an existing request")
            self.active_request_id = request_id
            self.active_motion_goal = goal
            self.active_intent = intent
            self.phase = "submitting"
            self._requests[request_id] = (signature, None)
            try:
                goal_id = self.controller.send_goal(goal)
            except Exception:
                self.phase = "unknown"
                raise
            self.active_goal_id = goal_id
            self._requests[request_id] = (signature, goal_id)
            self.phase = "accepted"
            return goal_id

    def poll_active(self):
        """Query the owned goal and release ownership on backend termination."""
        with self._lock:
            if self.active_request_id is None:
                return None
            if self.active_goal_id is None:
                raise RuntimeError("Submission outcome is unknown; reconcile backend state")
            goal, intent = self.active_motion_goal, self.active_intent
            try:
                status = self.controller.poll_status(self.active_goal_id)
            except Exception:
                self.phase = "unknown"
                raise
            # STOP 也以执行端回执为准；请求类型本身不能证明机器人已停下。
            terminal = status.is_terminal()
            if terminal:
                self.last_completed_signature = motion_goal_signature(goal)
                self.last_terminal_status = status
                self.active_request_id = None
                self.active_goal_id = None
                self.active_motion_goal = None
                self.active_intent = None
                self.phase = "terminated"
            elif self.phase != "cancel_requested":
                self.phase = "running"
            return ActiveGoalPoll(status, goal, intent, not terminal)

    def reconcile(self, request_id, backend_goal_id):
        """Bind a backend-confirmed goal ID after a submission response was lost."""
        with self._lock:
            if request_id != self.active_request_id or self.phase != "unknown":
                raise ValueError("Reconciliation requires the active unknown request")
            signature, _ = self._requests[request_id]
            self._requests[request_id] = (signature, backend_goal_id)
            self.active_goal_id = backend_goal_id
            self.phase = "accepted"

    def cancel(self):
        """Request cancellation and retain ownership until terminal feedback."""
        with self._lock:
            self.control_epoch += 1
            if self.active_request_id is None:
                return
            self.phase = "cancel_requested"
            self.controller.cancel(self.active_goal_id)

    def hold(self, status=None):
        """Request backend HOLD and invalidate outstanding motion proposals."""
        with self._lock:
            self.control_epoch += 1
            self.held = True
            self.halted_status = status
            self.controller.hold()

    def resume(self):
        """Release the supervisor's HOLD after the operator resolves its cause."""
        with self._lock:
            self.held = False
            self.halted_status = None
            self.control_epoch += 1

    def should_suppress_completed(self, goal):
        """Suppress an unchanged goal whose backend has already reported arrival."""
        return (goal.requires_motion()
                and self.last_completed_signature == motion_goal_signature(goal)
                and self.last_terminal_status is not None
                and self.last_terminal_status.succeeded())

    def clear_completed(self):
        """Allow a new semantic decision to dispatch a previously reached goal."""
        self.last_completed_signature = None
        self.last_terminal_status = None
