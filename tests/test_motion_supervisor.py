"""Exercise actual ownership transitions across retry, cancel and uncertainty."""
import pytest

from motion.supervisor import MotionSupervisor
from real_robot.contracts import MotionGoal, MotionGoalMode, NavigationIntent, NavigationStatus, NavigationStatusCode, Pose3D


class Controller:
    def __init__(self):
        self.sent = []
        self.cancelled = []
        self.held = False
        self.status = NavigationStatusCode.RUNNING
        self.send_error = None

    def send_goal(self, goal):
        self.sent.append(goal)
        if self.send_error:
            raise self.send_error
        return "goal-1"

    def poll_status(self, goal_id):
        return NavigationStatus(self.status, goal_id=goal_id)

    def cancel(self, goal_id):
        self.cancelled.append(goal_id)

    def hold(self):
        self.held = True


def request():
    pose = Pose3D((1.0, 0.0, 0.0))
    return (NavigationIntent(MotionGoalMode.GO_TO_OBJECT, goal_pose=pose),
            MotionGoal(MotionGoalMode.GO_TO_OBJECT, goal_pose=pose))


def test_retry_dispatches_once_and_cancel_waits_for_backend_terminal():
    controller = Controller()
    owner = MotionSupervisor(controller)
    intent, goal = request()
    assert owner.submit(intent, goal, request_id="r1") == "goal-1"
    assert owner.submit(intent, goal, request_id="r1") == "goal-1"
    assert len(controller.sent) == 1
    with pytest.raises(RuntimeError, match="ownership"):
        owner.submit(intent, goal, request_id="r2")
    owner.cancel()
    assert controller.cancelled == ["goal-1"]
    assert owner.poll_active().tracking_active
    assert owner.phase == "cancel_requested"
    with pytest.raises(RuntimeError, match="ownership"):
        owner.submit(intent, goal, request_id="r2")
    controller.status = NavigationStatusCode.PREEMPTED
    assert not owner.poll_active().tracking_active
    assert owner.submit(intent, goal, request_id="r2") == "goal-1"


def test_lost_submission_response_requires_reconciliation():
    controller = Controller()
    failure = TimeoutError("response lost after dispatch")
    controller.send_error = failure
    owner = MotionSupervisor(controller)
    intent, goal = request()
    with pytest.raises(TimeoutError) as error:
        owner.submit(intent, goal, request_id="r1")
    assert error.value is failure
    with pytest.raises(RuntimeError, match="unknown"):
        owner.submit(intent, goal, request_id="r1")
    assert len(controller.sent) == 1
    owner.reconcile("r1", "backend-confirmed-id")
    assert owner.poll_active().status.goal_id == "backend-confirmed-id"


def test_hold_and_control_epoch_govern_new_submissions():
    owner = MotionSupervisor(Controller())
    intent, goal = request()
    owner.hold()
    assert owner.controller.held
    with pytest.raises(RuntimeError, match="held"):
        owner.submit(intent, goal)
    owner.resume()
    with pytest.raises(RuntimeError, match="epoch"):
        owner.submit(intent, goal, control_epoch=0)
    owner.submit(intent, goal, control_epoch=owner.control_epoch)


def test_stop_request_keeps_ownership_until_backend_confirmation():
    controller = Controller()
    owner = MotionSupervisor(controller)
    stop = NavigationIntent(MotionGoalMode.STOP)
    owner.submit(stop, stop.to_motion_goal())
    assert owner.poll_active().tracking_active
    intent, goal = request()
    with pytest.raises(RuntimeError, match="ownership"):
        owner.submit(intent, goal)
    controller.status = NavigationStatusCode.REACHED
    assert not owner.poll_active().tracking_active
    owner.submit(intent, goal)
