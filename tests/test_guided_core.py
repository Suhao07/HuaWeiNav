"""Shared guided execution lifecycle checks without simulator dependencies."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from motion.contracts import MotionGoalMode, NavigationIntent, NavigationStatus, NavigationStatusCode
from motion.supervisor import MotionSupervisor
from navigation.guided import run_guided_loop
from scene.contracts import Pose3D


def loop_fixture(*, code=NavigationStatusCode.REACHED, frontier=False):
    events = []
    controller = NS(send_goal=Mock(return_value="goal"),
                    poll_status=Mock(return_value=NavigationStatus(code)), cancel=Mock())
    motion = MotionSupervisor(controller)
    metadata = {"task_revision": 1, "candidate_evidence_fingerprint": "e1"}
    if frontier:
        metadata["node_index"] = 0
    intent = NavigationIntent(MotionGoalMode.GO_TO_FRONTIER if frontier else MotionGoalMode.IMPROVE_VIEW,
                              goal_pose=Pose3D((1., 0., 0.)), target_object_uid=None if frontier else "u",
                              metadata=metadata)
    motion.submit(intent, intent.to_motion_goal())
    guided = NS(tools=NS(motion=motion), motion_finished=Mock(), record=Mock(),
                select=Mock(return_value=False), verify=Mock(return_value="verified_finish"),
                recover=Mock(return_value="search_exhausted"),
                observation_updated=Mock(side_effect=lambda: events.append("evidence")))
    execution = NS(episode_idx=42, initialize=Mock(), limit_reason=Mock(return_value=None),
                   advance=Mock(), after_motion=Mock(),
                   frontier_arrived=Mock(side_effect=lambda f: events.append("observe")),
                   frontier_observed=Mock(side_effect=lambda f: events.append("mark")),
                   object_arrived=Mock(), finish_success=Mock())
    return guided, execution, controller, events


def test_loop_frontier_observe_refresh_then_mark():
    guided, execution, controller, events = loop_fixture(frontier=True)
    assert run_guided_loop(guided, execution) == "search_exhausted"
    assert events == ["observe", "evidence", "mark"]
    execution.advance.assert_called_once_with()
    guided.select.assert_called_once_with(42)
    guided.verify.assert_not_called()
    execution.finish_success.assert_not_called()
    controller.cancel.assert_not_called()


def test_loop_blocked_recovers_without_observation_or_verification():
    guided, execution, _, _ = loop_fixture(code=NavigationStatusCode.BLOCKED)
    assert run_guided_loop(guided, execution) == "search_exhausted"
    assert guided.recover.call_args.args == (guided.recovery_context, 42)
    assert guided.recovery_context["status"] == "blocked"
    guided.motion_finished.assert_called_once()
    guided.verify.assert_not_called()
    execution.frontier_arrived.assert_not_called()
    execution.finish_success.assert_not_called()


@pytest.mark.parametrize("code", [NavigationStatusCode.REACHED, NavigationStatusCode.RUNNING])
def test_last_budget_accepted_motion_runs_before_decision_limit(code):
    guided, execution, controller, _ = loop_fixture(code=code)
    controller.poll_status.side_effect = [NavigationStatus(code), NavigationStatus(NavigationStatusCode.REACHED)]
    execution.limit_reason.return_value = "decision_budget_exhausted"
    assert run_guided_loop(guided, execution) == "verified_finish"
    assert execution.advance.call_count == (2 if code is NavigationStatusCode.RUNNING else 1)
    guided.verify.assert_called_once_with((1., 0., 0.))
    execution.finish_success.assert_called_once_with()
    controller.cancel.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["initialize", "advance", "after_motion", "limit_reason", "select"])
def test_loop_errors_always_cancel_owned_motion_without_masking_original(failure_stage):
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.RUNNING)
    error = ValueError("original execution error")
    if failure_stage == "select":
        motion = guided.tools.motion
        intent, goal = motion.active_intent, motion.active_motion_goal
        controller.poll_status.return_value = NavigationStatus(NavigationStatusCode.PREEMPTED)
        motion.poll_active()

        def submit_then_fail(index):
            motion.submit(intent, goal)
            raise error

        guided.select.side_effect = submit_then_fail
    else:
        getattr(execution, failure_stage).side_effect = error
    controller.cancel.side_effect = lambda goal_id: setattr(
        controller.poll_status, "return_value", NavigationStatus(NavigationStatusCode.PREEMPTED))
    with pytest.raises(ValueError) as raised:
        run_guided_loop(guided, execution)
    assert raised.value is error
    controller.cancel.assert_called_once_with("goal")
    assert guided.tools.motion.active_request_id is None
    execution.finish_success.assert_not_called()


def test_loop_limit_cancels_without_successful_stop():
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.PREEMPTED)
    execution.limit_reason.return_value = "episode_limit"
    assert run_guided_loop(guided, execution) == "episode_limit"
    controller.cancel.assert_called_once_with("goal")
    execution.advance.assert_not_called()
    execution.finish_success.assert_not_called()
    assert guided.tools.motion.last_terminal_status.status is NavigationStatusCode.PREEMPTED


@pytest.mark.parametrize("original_failure", [False, True])
def test_cleanup_timeout_is_visible_and_original_error_remains_primary(monkeypatch, original_failure):
    guided, execution, controller, _ = loop_fixture(code=NavigationStatusCode.RUNNING)
    # Use the real bounded cancellation path, but a zero deadline to keep the test offline and immediate.
    cancel = guided.tools.motion.cancel_and_wait
    monkeypatch.setattr(guided.tools.motion, "cancel_and_wait", lambda: cancel(timeout=0))
    original = ValueError("original")
    if original_failure:
        execution.initialize.side_effect = original
    else:
        execution.limit_reason.return_value = "episode_limit"
    with pytest.raises(ValueError if original_failure else TimeoutError) as raised:
        run_guided_loop(guided, execution)
    if original_failure:
        assert raised.value is original
        assert isinstance(raised.value.__cause__, TimeoutError)
    assert guided.tools.motion.active_request_id is not None
    assert guided.tools.motion.last_terminal_status is None
    controller.cancel.assert_called_once_with("goal")
    execution.finish_success.assert_not_called()
