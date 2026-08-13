from types import SimpleNamespace

import pytest

from real_robot.contracts import MotionGoal, MotionGoalMode, NavigationStatusCode, Pose3D
from real_robot.detector_vocabulary import DetectorVocabularyAdapter
from real_robot.sysnav_ros_adapters import (
    RosDetectionResultAdapter,
    RosNavigationStatusProvider,
    RosObjectNodeAdapter,
    RosRoomNodeAdapter,
    RosWaypointController,
    build_semantic_map_snapshot,
)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _header(sec=1, nanosec=250_000_000, frame_id="map"):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _orientation(x=0.0, y=0.0, z=0.0, w=1.0):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def _pose_msg(x, y, z):
    return SimpleNamespace(position=_point(x, y, z), orientation=_orientation())


def _odom_msg(x, y, z, sec=1):
    return SimpleNamespace(
        header=_header(sec=sec),
        pose=SimpleNamespace(pose=_pose_msg(x, y, z)),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )
        ),
    )


def _path_msg(*points):
    return SimpleNamespace(
        header=_header(sec=1),
        poses=[
            SimpleNamespace(header=_header(sec=1), pose=_pose_msg(point[0], point[1], point[2]))
            for point in points
        ],
    )


def _vocabulary(tmp_path):
    config = tmp_path / "objects.yaml"
    config.write_text(
        """
prompts:
  book:
    prompts:
      - book
    is_instance: true
  garbage_bin:
    prompts:
      - trash can
    is_instance: true
  cabinet:
    prompts:
      - cabinet
    is_instance: true
""",
        encoding="utf-8",
    )
    return DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config), detector_name="sysnav_test_detector")


def test_detection_result_adapter_maps_sysnav_message(tmp_path) -> None:
    msg = SimpleNamespace(
        header=_header(),
        track_id=[7, 8],
        x1=[10.0, 20.0],
        y1=[11.0, 21.0],
        x2=[30.0, 40.0],
        y2=[31.0, 41.0],
        label=["book", "shelf"],
        confidence=[0.8, 0.9],
        image=SimpleNamespace(height=480, width=640, encoding="bgr8", step=1920),
    )

    frame = RosDetectionResultAdapter(detector_vocabulary=_vocabulary(tmp_path)).from_msg(msg)

    assert frame.timestamp == 1.25
    assert frame.boxes_xyxy == ((10.0, 11.0, 30.0, 31.0), (20.0, 21.0, 40.0, 41.0))
    assert frame.labels == ("book", "shelf")
    assert frame.track_ids == ("7", "8")
    assert frame.metadata["image"]["encoding"] == "bgr8"
    assert frame.metadata["detector_vocabulary"]["detector_name"] == "sysnav_test_detector"
    assert frame.metadata["label_provenance"][0]["canonical_label"] == "book"
    assert frame.metadata["label_provenance"][1]["known_in_detector_vocabulary"] is False


def test_object_node_adapter_maps_identity_geometry_and_evidence(tmp_path) -> None:
    bbox = (
        _point(0, 0, 0),
        _point(2, 0, 0),
        _point(0, 4, 0),
        _point(2, 4, 0),
        _point(0, 0, 6),
        _point(2, 0, 6),
        _point(0, 4, 6),
        _point(2, 4, 6),
    )
    msg = SimpleNamespace(
        header=_header(),
        object_id=[42],
        label="cabinet",
        position=_point(1.0, 2.0, 0.5),
        bbox3d=bbox,
        cloud=object(),
        status=True,
        img_path="/tmp/cabinet.npy",
        is_asked_vlm=True,
        viewpoint_id=3,
    )

    snapshot = RosObjectNodeAdapter(detector_vocabulary=_vocabulary(tmp_path)).from_msg(msg)

    assert snapshot.uid == "sysnav_object:42"
    assert snapshot.label == "cabinet"
    assert snapshot.position == (1.0, 2.0, 0.5)
    assert snapshot.bbox3d_center == (1.0, 2.0, 3.0)
    assert snapshot.bbox3d_extent == (2.0, 4.0, 6.0)
    assert snapshot.visible_viewpoints == ("3",)
    assert snapshot.verified_state == "active"
    assert snapshot.metadata["label_provenance"]["raw_detector_label"] == "cabinet"
    assert snapshot.metadata["label_provenance"]["canonical_label"] == "cabinet"


def test_room_node_adapter_maps_topology_summary() -> None:
    msg = SimpleNamespace(
        id=5,
        show_id=2,
        centroid=_point(1.0, 2.0, 0.0),
        neighbors=[4, 6],
        is_connected=True,
        area=12.5,
        room_mask=object(),
        polygon=SimpleNamespace(polygon=SimpleNamespace(points=[_point(0, 0, 0), _point(1, 0, 0)])),
    )

    room = RosRoomNodeAdapter().from_msg(msg)

    assert room.uid == "sysnav_room:5"
    assert room.centroid == (1.0, 2.0, 0.0)
    assert room.neighbors == ("sysnav_room:4", "sysnav_room:6")
    assert room.explored is True
    assert room.metadata["area"] == 12.5
    assert room.metadata["polygon_point_count"] == 2


def test_build_semantic_map_snapshot_uses_sysnav_object_and_room_lists(tmp_path) -> None:
    object_msg = SimpleNamespace(
        header=_header(sec=2),
        nodes=[
            SimpleNamespace(
                header=_header(sec=2),
                object_id=[1],
                label="chair",
                position=_point(0.0, 1.0, 0.0),
                bbox3d=[],
                cloud=None,
                status=True,
                img_path="",
                is_asked_vlm=False,
                viewpoint_id=-1,
            )
        ],
    )
    room_msg = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id=9,
                show_id=9,
                centroid=_point(5.0, 0.0, 0.0),
                neighbors=[],
                is_connected=False,
                area=1.0,
                room_mask=None,
                polygon=None,
            )
        ]
    )

    snapshot = build_semantic_map_snapshot(
        object_list_msg=object_msg,
        room_list_msg=room_msg,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0)),
        object_adapter=RosObjectNodeAdapter(detector_vocabulary=_vocabulary(tmp_path)),
    )

    assert snapshot.source == "sysnav_ros"
    assert snapshot.object_by_uid("sysnav_object:1").label == "chair"
    assert snapshot.room_by_uid("sysnav_room:9").centroid == (5.0, 0.0, 0.0)
    assert snapshot.frontiers[0].room_id == "sysnav_room:9"
    assert snapshot.metadata["detector_vocabulary"]["detector_name"] == "sysnav_test_detector"


class FakePointStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.point = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class FakeEmpty:
    pass


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


def test_waypoint_controller_publishes_sysnav_waypoint() -> None:
    publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=publisher,
        point_stamped_type=FakePointStamped,
    )
    goal = MotionGoal(
        mode=MotionGoalMode.GO_TO_OBJECT,
        goal_pose=Pose3D(position=(1.0, 2.0, -0.8), frame_id="map"),
        target_object_uid="sysnav_object:1",
    )

    goal_id = controller.send_goal(goal)
    status = controller.poll_status(goal_id)

    assert len(publisher.published) == 1
    assert publisher.published[0].point.x == 1.0
    assert publisher.published[0].point.y == 2.0
    assert publisher.published[0].point.z == -0.8
    assert status.status == NavigationStatusCode.RUNNING
    assert status.metadata["target_object_uid"] == "sysnav_object:1"


def test_waypoint_controller_does_not_publish_stop_goal() -> None:
    publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=publisher,
        point_stamped_type=FakePointStamped,
    )
    goal = MotionGoal(mode=MotionGoalMode.STOP)

    goal_id = controller.send_goal(goal)

    assert publisher.published == []
    assert controller.poll_status(goal_id).status == NavigationStatusCode.REACHED


def test_waypoint_controller_rejects_direct_cmd_vel_topics() -> None:
    with pytest.raises(ValueError, match="direct velocity"):
        RosWaypointController(
            node=SimpleNamespace(),
            waypoint_topic="/cmd_vel",
            publisher=FakePublisher(),
            point_stamped_type=FakePointStamped,
        )

    with pytest.raises(ValueError, match="direct velocity"):
        RosWaypointController(
            node=SimpleNamespace(),
            publisher=FakePublisher(),
            point_stamped_type=FakePointStamped,
            hold_topic="/robot/cmd_vel",
            hold_publisher=FakePublisher(),
            empty_type=FakeEmpty,
        )


def test_waypoint_controller_rejects_untransformed_goal_frame() -> None:
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=FakePublisher(),
        point_stamped_type=FakePointStamped,
        world_frame="map",
    )
    goal = MotionGoal(
        mode=MotionGoalMode.GO_TO_OBJECT,
        goal_pose=Pose3D(position=(1.0, 0.0, 0.0), frame_id="odom"),
    )

    with pytest.raises(ValueError, match="does not match configured world frame"):
        controller.send_goal(goal)


def test_waypoint_controller_hold_publishes_hold_without_emergency_by_default() -> None:
    waypoint_publisher = FakePublisher()
    hold_publisher = FakePublisher()
    emergency_publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=waypoint_publisher,
        point_stamped_type=FakePointStamped,
        hold_topic="/platform/safe_hold",
        emergency_stop_topic="/platform/emergency_stop",
        hold_publisher=hold_publisher,
        emergency_stop_publisher=emergency_publisher,
        empty_type=FakeEmpty,
    )
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    goal_id = controller.send_goal(goal)
    controller.hold()
    status = controller.poll_status(goal_id)

    assert len(hold_publisher.published) == 1
    assert emergency_publisher.published == []
    assert status.status == NavigationStatusCode.IDLE
    assert status.metadata["hold_signal_published"] is True
    assert status.metadata["emergency_stop_publish_allowed"] is False
    assert status.metadata["emergency_stop_published"] is False


def test_waypoint_controller_hold_can_publish_emergency_when_explicitly_allowed() -> None:
    hold_publisher = FakePublisher()
    emergency_publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=FakePublisher(),
        point_stamped_type=FakePointStamped,
        hold_topic="/platform/safe_hold",
        emergency_stop_topic="/platform/emergency_stop",
        allow_emergency_stop_publish=True,
        hold_publisher=hold_publisher,
        emergency_stop_publisher=emergency_publisher,
        empty_type=FakeEmpty,
    )
    goal_id = controller.send_goal(
        MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))
    )

    controller.hold()
    status = controller.poll_status(goal_id)

    assert len(hold_publisher.published) == 1
    assert len(emergency_publisher.published) == 1
    assert status.metadata["emergency_stop_published"] is True


def test_waypoint_controller_cancel_publishes_cancel_signal() -> None:
    cancel_publisher = FakePublisher()
    hold_publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=FakePublisher(),
        point_stamped_type=FakePointStamped,
        cancel_topic="/local_planner/cancel",
        hold_topic="/platform/safe_hold",
        cancel_publisher=cancel_publisher,
        hold_publisher=hold_publisher,
        empty_type=FakeEmpty,
    )
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    goal_id = controller.send_goal(goal)
    controller.cancel(goal_id)
    status = controller.poll_status(goal_id)

    assert len(cancel_publisher.published) == 1
    # Planner cancellation and the independent safety hold are both required.
    assert len(hold_publisher.published) == 1
    assert status.status == NavigationStatusCode.PREEMPTED
    assert status.metadata["cancel_signal_published"] is True
    assert status.metadata["hold_signal_published"] is True
    assert status.metadata["hold_fallback_published"] is False


def test_waypoint_controller_cancel_uses_hold_fallback_without_cancel_topic() -> None:
    hold_publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=FakePublisher(),
        point_stamped_type=FakePointStamped,
        hold_topic="/platform/safe_hold",
        hold_publisher=hold_publisher,
        empty_type=FakeEmpty,
    )
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    goal_id = controller.send_goal(goal)
    controller.cancel(goal_id)
    status = controller.poll_status(goal_id)

    assert len(hold_publisher.published) == 1
    assert status.status == NavigationStatusCode.PREEMPTED
    assert status.metadata["cancel_signal_published"] is False
    assert status.metadata["hold_fallback_published"] is True


def test_navigation_status_provider_reports_running_and_reached_from_odom_and_path() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(
        xy_tolerance_m=0.25,
        z_tolerance_m=1.0,
        timeout_s=30.0,
        no_progress_timeout_s=10.0,
        now_fn=lambda: clock["t"],
    )
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    provider.update_path(_path_msg((0.5, 0.0, 0.0), (1.0, 0.0, 0.0)))
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    running = provider("goal-1", goal)

    assert running.status == NavigationStatusCode.RUNNING
    assert running.distance_to_goal == 1.0
    assert running.path_length_remaining == 1.0
    assert running.metadata["path_frame_id"] == "map"
    assert running.metadata["path_available"] is True
    assert running.metadata["heading_checked"] is False

    clock["t"] = 1.0
    provider.update_odometry(_odom_msg(0.8, 0.0, 0.0, sec=2))
    reached = provider("goal-1", goal)

    assert reached.status == NavigationStatusCode.REACHED
    assert reached.distance_to_goal < 0.25
    assert reached.progress == 1.0


def test_navigation_status_provider_reports_timeout_before_no_progress() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(
        xy_tolerance_m=0.1,
        timeout_s=2.0,
        no_progress_timeout_s=30.0,
        now_fn=lambda: clock["t"],
    )
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(2.0, 0.0, 0.0)))

    assert provider("goal-timeout", goal).status == NavigationStatusCode.RUNNING

    clock["t"] = 3.0
    status = provider("goal-timeout", goal)

    assert status.status == NavigationStatusCode.TIMEOUT
    assert status.metadata["elapsed_s"] == 3.0


def test_navigation_status_provider_reports_blocked_on_no_progress() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(
        xy_tolerance_m=0.1,
        timeout_s=30.0,
        no_progress_timeout_s=2.0,
        min_progress_delta_m=0.05,
        now_fn=lambda: clock["t"],
    )
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(2.0, 0.0, 0.0)))

    assert provider("goal-blocked", goal).status == NavigationStatusCode.RUNNING

    clock["t"] = 3.0
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0, sec=4))
    status = provider("goal-blocked", goal)

    assert status.status == NavigationStatusCode.BLOCKED
    assert status.metadata["no_progress_elapsed_s"] == 3.0
    assert status.metadata["progress_samples"][-1]["distance_3d_m"] == 2.0


def test_navigation_status_provider_respects_local_planner_blocked_status() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(path_stale_timeout_s=5.0, now_fn=lambda: clock["t"])
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(2.0, 0.0, 0.0)))
    assert provider("goal-planner-blocked", goal).status == NavigationStatusCode.RUNNING
    clock["t"] = 1.0
    provider.update_local_planner_status(SimpleNamespace(data="blocked"))

    status = provider("goal-planner-blocked", goal)

    assert status.status == NavigationStatusCode.BLOCKED
    assert status.metadata["planner_status"]["text"] == "blocked"
    assert status.metadata["planner_status_fresh"] is True

    clock["t"] = 6.1
    stale = provider("goal-planner-stale", goal)

    assert stale.status == NavigationStatusCode.RUNNING
    assert stale.metadata["planner_status_fresh"] is False


def test_navigation_status_provider_maps_planner_no_feasible_path_reason() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(now_fn=lambda: clock["t"])
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(2.0, 0.0, 0.0)))
    assert provider("goal-no-path", goal).status == NavigationStatusCode.RUNNING
    clock["t"] = 1.0
    provider.update_local_planner_status(SimpleNamespace(data="no_feasible_path"))

    status = provider("goal-no-path", goal)

    assert status.status == NavigationStatusCode.BLOCKED
    assert status.reason_code.value == "no_feasible_path"


def test_waypoint_controller_starts_planner_generation_before_publish() -> None:
    clock = {"t": 0.0}
    provider = RosNavigationStatusProvider(now_fn=lambda: clock["t"])
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    provider.update_local_planner_status(SimpleNamespace(data="tracking"))
    publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=publisher,
        point_stamped_type=FakePointStamped,
        status_provider=provider,
    )
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    goal_id = controller.send_goal(goal)
    clock["t"] = 0.1
    provider.update_local_planner_status(SimpleNamespace(data="no_feasible_path"))

    assert controller.poll_status(goal_id).status == NavigationStatusCode.BLOCKED


def test_waypoint_controller_cancel_marks_status_provider_preempted() -> None:
    provider = RosNavigationStatusProvider(now_fn=lambda: 0.0)
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    publisher = FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=publisher,
        point_stamped_type=FakePointStamped,
        status_provider=provider,
    )
    goal = MotionGoal(mode=MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D(position=(1.0, 0.0, 0.0)))

    goal_id = controller.send_goal(goal)
    assert controller.poll_status(goal_id).status == NavigationStatusCode.RUNNING

    controller.cancel(goal_id)

    assert controller.poll_status(goal_id).status == NavigationStatusCode.PREEMPTED
