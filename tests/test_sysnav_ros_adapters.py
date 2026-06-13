from types import SimpleNamespace

from real_robot.contracts import MotionGoal, MotionGoalMode, NavigationStatusCode, Pose3D
from real_robot.sysnav_ros_adapters import (
    RosDetectionResultAdapter,
    RosObjectNodeAdapter,
    RosRoomNodeAdapter,
    RosWaypointController,
    build_semantic_map_snapshot,
)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _header(sec=1, nanosec=250_000_000, frame_id="map"):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def test_detection_result_adapter_maps_sysnav_message() -> None:
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

    frame = RosDetectionResultAdapter().from_msg(msg)

    assert frame.timestamp == 1.25
    assert frame.boxes_xyxy == ((10.0, 11.0, 30.0, 31.0), (20.0, 21.0, 40.0, 41.0))
    assert frame.labels == ("book", "shelf")
    assert frame.track_ids == ("7", "8")
    assert frame.metadata["image"]["encoding"] == "bgr8"


def test_object_node_adapter_maps_identity_geometry_and_evidence() -> None:
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

    snapshot = RosObjectNodeAdapter().from_msg(msg)

    assert snapshot.uid == "sysnav_object:42"
    assert snapshot.label == "cabinet"
    assert snapshot.position == (1.0, 2.0, 0.5)
    assert snapshot.bbox3d_center == (1.0, 2.0, 3.0)
    assert snapshot.bbox3d_extent == (2.0, 4.0, 6.0)
    assert snapshot.visible_viewpoints == ("3",)
    assert snapshot.verified_state == "active"


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


def test_build_semantic_map_snapshot_uses_sysnav_object_and_room_lists() -> None:
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
    )

    assert snapshot.source == "sysnav_ros"
    assert snapshot.object_by_uid("sysnav_object:1").label == "chair"
    assert snapshot.room_by_uid("sysnav_room:9").centroid == (5.0, 0.0, 0.0)
    assert snapshot.frontiers[0].room_id == "sysnav_room:9"


class FakePointStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.point = SimpleNamespace(x=0.0, y=0.0, z=0.0)


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
