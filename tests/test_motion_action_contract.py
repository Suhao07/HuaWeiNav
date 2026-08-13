from types import SimpleNamespace

from real_robot.ros_motion_action import motion_goal_from_action_goal
from real_robot.contracts import MotionGoalMode


def _pose_stamped(frame_id="map"):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def test_action_goal_conversion_preserves_task_identity_and_tolerances() -> None:
    request = SimpleNamespace(
        target_pose=_pose_stamped(),
        look_at=SimpleNamespace(point=SimpleNamespace(x=2.0, y=2.0, z=1.0)),
        has_look_at=True,
        xy_tolerance_m=0.3,
        yaw_tolerance_rad=0.2,
        timeout_s=30.0,
        motion_profile="approach_target",
        target_object_uid="book:1",
        anchor_object_uid="shelf:2",
        relation_edge_id="book:1:on:shelf:2",
    )

    goal = motion_goal_from_action_goal(request)

    assert goal.mode == MotionGoalMode.IMPROVE_VIEW
    assert goal.target_object_uid == "book:1"
    assert goal.anchor_object_uid == "shelf:2"
    assert goal.relation_edge_id == "book:1:on:shelf:2"
    assert goal.tolerance["xy_goal_tolerance_m"] == 0.3
    assert goal.metadata["timeout_s"] == 30.0


def test_action_motion_profile_selects_a_declared_motion_mode() -> None:
    request = SimpleNamespace(
        target_pose=_pose_stamped(),
        look_at=SimpleNamespace(point=SimpleNamespace(x=2.0, y=2.0, z=1.0)),
        has_look_at=True,
        xy_tolerance_m=0.3,
        yaw_tolerance_rad=0.2,
        timeout_s=30.0,
        motion_profile="verify_relation",
        target_object_uid="book:1",
        anchor_object_uid="shelf:2",
        relation_edge_id="book:1:on:shelf:2",
    )

    goal = motion_goal_from_action_goal(request)

    assert goal.mode == MotionGoalMode.VERIFY_RELATION
