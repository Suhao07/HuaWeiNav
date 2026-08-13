from types import SimpleNamespace

from real_robot.contracts import MotionGoal, MotionGoalMode, NavigationStatusCode, Pose3D
from real_robot.sysnav_ros_adapters import RosNavigationStatusProvider


def _goal():
    return MotionGoal(
        mode=MotionGoalMode.GO_TO_OBJECT,
        goal_pose=Pose3D(position=(2.0, 0.0, 0.0)),
    )


def test_safety_stop_is_terminal_even_before_odometry_arrives() -> None:
    provider = RosNavigationStatusProvider(now_fn=lambda: 1.0)
    provider.update_safety_state(SimpleNamespace(state=3, reason_code="estop_active"))

    status = provider("goal-estop", _goal())

    assert status.status == NavigationStatusCode.SAFETY_STOP
    assert status.safety_state == "estop"


def test_safety_hold_waits_for_enable_instead_of_failing_goal() -> None:
    provider = RosNavigationStatusProvider(now_fn=lambda: 1.0)
    provider.update_safety_state(SimpleNamespace(state=1, autonomy_enabled=False))

    status = provider("goal-hold", _goal())

    assert status.status == NavigationStatusCode.QUEUED


def test_safety_hold_obeys_motion_timeout() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(timeout_s=2.0, now_fn=lambda: clock["t"])
    provider.update_safety_state(SimpleNamespace(state=1, autonomy_enabled=False))
    provider.update_pose(Pose3D(position=(0.0, 0.0, 0.0)))

    assert provider("goal-hold-timeout", _goal()).status == NavigationStatusCode.QUEUED
    clock["t"] = 3.0
    status = provider("goal-hold-timeout", _goal())

    assert status.status == NavigationStatusCode.TIMEOUT
