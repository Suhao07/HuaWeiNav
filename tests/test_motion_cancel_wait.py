"""Offline cancellation receipts, deadlines and uncertain ownership."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from motion.contracts import MotionGoalMode, NavigationIntent, NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from scene.contracts import Pose3D


@pytest.fixture
def clock(monkeypatch):
    import motion.supervisor as module
    state = NS(now=0.0, sleeps=[])

    def sleep(delay):
        state.sleeps.append(delay)
        state.now += delay

    monkeypatch.setattr(module, "monotonic", lambda: state.now)
    monkeypatch.setattr(module, "sleep", sleep)
    return state


def owner_with_statuses(*codes, mode=MotionGoalMode.IMPROVE_VIEW):
    controller = NS(send_goal=Mock(return_value="backend-goal"), cancel=Mock(), hold=Mock(),
                    poll_status=Mock(side_effect=[NavigationStatus(code, goal_id="backend-goal")
                                                 for code in codes]))
    owner = MotionSupervisor(controller)
    intent = NavigationIntent(mode, goal_pose=Pose3D((1., 0., 0.)))
    owner.submit(intent, intent.to_motion_goal(), request_id="owned")
    return owner


@pytest.mark.parametrize("terminal", [NavigationStatusCode.PREEMPTED, NavigationStatusCode.REACHED,
                                      NavigationStatusCode.FAILED, NavigationStatusCode.SAFETY_STOP])
def test_cancel_wait_releases_only_for_backend_terminal(clock, terminal):
    owner = owner_with_statuses(NavigationStatusCode.RUNNING, NavigationStatusCode.RUNNING, terminal)
    result = owner.cancel_and_wait(timeout=1.0, poll_interval=0.1)
    assert result.status.status is terminal
    assert result.status.succeeded() is (terminal is NavigationStatusCode.REACHED)
    assert not result.tracking_active and owner.active_request_id is None
    assert owner.last_terminal_status is result.status
    assert owner.control_epoch == 1
    assert clock.sleeps == [0.1, 0.1]
    owner.controller.cancel.assert_called_once_with("backend-goal")
    owner.controller.send_goal.assert_called_once()
    owner.controller.hold.assert_not_called()


def test_timeout_keeps_owner_and_can_later_observe_real_terminal(clock):
    owner = owner_with_statuses(*([NavigationStatusCode.RUNNING] * 4), NavigationStatusCode.PREEMPTED)
    goal, intent = owner.active_motion_goal, owner.active_intent
    with pytest.raises(TimeoutError, match="owned"):
        owner.cancel_and_wait(timeout=0.25, poll_interval=0.1)
    assert clock.now == pytest.approx(0.25)
    assert clock.sleeps == pytest.approx([0.1, 0.1, 0.05])
    assert owner.active_request_id == "owned" and owner.active_goal_id == "backend-goal"
    assert owner.active_motion_goal is goal and owner.active_intent is intent
    assert owner.phase == "cancel_requested" and owner.last_terminal_status is None
    with pytest.raises(RuntimeError, match="ownership"):
        owner.submit(intent, goal, request_id="replacement")
    result = owner.poll_active()
    assert result.status.status is NavigationStatusCode.PREEMPTED
    assert owner.active_request_id is None


@pytest.mark.parametrize("operation", ["cancel", "poll_status"])
def test_backend_errors_are_preserved_without_releasing_ownership(clock, operation):
    owner = owner_with_statuses(NavigationStatusCode.RUNNING)
    error = OSError("backend unavailable")
    getattr(owner.controller, operation).side_effect = error
    with pytest.raises(OSError) as raised:
        owner.cancel_and_wait(timeout=0.1)
    assert raised.value is error
    assert owner.active_request_id == "owned"
    assert owner.last_terminal_status is None
    assert clock.sleeps == []


def test_unknown_submission_is_cancelled_but_not_declared_complete(clock):
    controller = NS(send_goal=Mock(side_effect=TimeoutError("lost reply")), cancel=Mock(), poll_status=Mock())
    owner = MotionSupervisor(controller)
    intent = NavigationIntent(MotionGoalMode.STOP)
    with pytest.raises(TimeoutError, match="lost reply"):
        owner.submit(intent, intent.to_motion_goal(), request_id="unknown")
    with pytest.raises(RuntimeError, match="unknown"):
        owner.cancel_and_wait(timeout=0)
    controller.cancel.assert_called_once_with(None)
    controller.poll_status.assert_not_called()
    assert owner.active_request_id == "unknown" and owner.last_terminal_status is None
    owner.reconcile("unknown", "confirmed")
    assert owner.active_goal_id == "confirmed"


@pytest.mark.parametrize("terminal", [NavigationStatusCode.PREEMPTED, NavigationStatusCode.REACHED])
def test_stop_uses_real_terminal_feedback_not_request_type(clock, terminal):
    owner = owner_with_statuses(NavigationStatusCode.RUNNING, terminal, mode=MotionGoalMode.STOP)
    result = owner.cancel_and_wait(timeout=1.0)
    assert result.status.status is terminal
    assert result.status.succeeded() is (terminal is NavigationStatusCode.REACHED)
    assert owner.controller.poll_status.call_count == 2
    owner.controller.send_goal.assert_called_once()


def test_zero_timeout_polls_once_without_sleep(clock):
    owner = owner_with_statuses(NavigationStatusCode.RUNNING)
    with pytest.raises(TimeoutError):
        owner.cancel_and_wait(timeout=0)
    owner.controller.poll_status.assert_called_once_with("backend-goal")
    assert clock.sleeps == [] and owner.active_request_id == "owned"


def test_idle_cleanup_does_not_cancel_or_invalidate_epoch(clock):
    controller = NS(cancel=Mock(), poll_status=Mock())
    owner = MotionSupervisor(controller)
    assert owner.cancel_and_wait() is None
    assert owner.control_epoch == 0
    controller.cancel.assert_not_called()
    controller.poll_status.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"timeout": -1}, {"timeout": float("inf")},
                                    {"timeout": float("nan")}, {"poll_interval": 0},
                                    {"poll_interval": -1}, {"poll_interval": float("inf")}])
def test_invalid_deadlines_fail_before_backend_calls(kwargs):
    owner = owner_with_statuses(NavigationStatusCode.RUNNING)
    with pytest.raises(ValueError):
        owner.cancel_and_wait(**kwargs)
    owner.controller.cancel.assert_not_called()
    assert owner.active_request_id == "owned" and owner.control_epoch == 0
