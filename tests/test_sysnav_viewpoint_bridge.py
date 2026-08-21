from types import SimpleNamespace

import pytest

from real_robot.sysnav_viewpoint_bridge import SysNavViewpointBridgeModel


def _header(sec: int, nanosec: int = 0, frame_id: str = "map"):
    return SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
        frame_id=frame_id,
    )


def _odom(sec: int, x: float, y: float, frame_id: str = "map"):
    return SimpleNamespace(
        header=_header(sec, frame_id=frame_id),
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
    )


def _viewpoint(viewpoint_id: int, sec: int, frame_id: str = "map"):
    return SimpleNamespace(header=_header(sec, frame_id=frame_id), viewpoint_id=viewpoint_id)


def _objects(viewpoint_id: int, *object_ids: int):
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(viewpoint_id=viewpoint_id, object_id=list(object_ids)),
        ]
    )


def test_hil_join_uses_timestamp_aligned_odom_and_direct_object_observation() -> None:
    model = SysNavViewpointBridgeModel(max_time_offset_s=0.1)

    model.update_odometry(_odom(10, 1.0, 2.0))
    model.update_object_nodes(_objects(3, 42, 43))
    records = model.update_viewpoint(_viewpoint(3, 10))

    assert len(records) == 1
    assert records[0].viewpoint_id == 3
    assert records[0].timestamp_ns == 10_000_000_000
    assert records[0].pose.position == (1.0, 2.0, 0.0)
    assert records[0].observed_object_ids == (42, 43)


def test_hil_join_waits_for_fresh_odom_instead_of_using_stale_pose() -> None:
    model = SysNavViewpointBridgeModel(max_time_offset_s=0.1)

    model.update_odometry(_odom(1, 1.0, 2.0))
    assert model.update_viewpoint(_viewpoint(3, 10)) == ()
    assert model.update_odometry(_odom(10, 5.0, 6.0))[0].pose.position == (5.0, 6.0, 0.0)


def test_hil_join_preserves_subsecond_timestamp_exactly() -> None:
    model = SysNavViewpointBridgeModel(max_time_offset_s=0.001)
    odom = _odom(10, 1.0, 2.0)
    viewpoint = _viewpoint(3, 10)
    odom.header.stamp.nanosec = 123_456_789
    viewpoint.header.stamp.nanosec = 123_456_789

    model.update_odometry(odom)
    records = model.update_viewpoint(viewpoint)

    assert records[0].timestamp_ns == 10_123_456_789
    assert records[0].timestamp == pytest.approx(10.123456789)


def test_hil_join_rejects_frame_mismatch_without_transform_guess() -> None:
    model = SysNavViewpointBridgeModel(max_time_offset_s=0.1)

    model.update_odometry(_odom(10, 1.0, 2.0, frame_id="odom"))
    assert model.update_viewpoint(_viewpoint(3, 10, frame_id="map")) == ()


def test_hil_join_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        SysNavViewpointBridgeModel(max_time_offset_s=-1.0)
    with pytest.raises(ValueError):
        SysNavViewpointBridgeModel(odom_history_size=0)
