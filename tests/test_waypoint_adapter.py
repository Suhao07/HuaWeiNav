from math import sqrt

import pytest

from real_robot.contracts import Pose3D
from real_robot.waypoint_adapter import WaypointAdapterConfig, WaypointAdapterError, WaypointFormatAdapter


def test_identity_adapter_converts_xy_and_drops_z_by_default() -> None:
    adapter = WaypointFormatAdapter(WaypointAdapterConfig(coordinate_mode="identity"), now_fn=lambda: 10.0)
    result = adapter.convert((1.25, -2.5, 0.7), frame_id="map", stamp=9.5, now=10.0)
    assert result is not None
    assert result.values == (1.25, -2.5)


def test_ego_from_odom_rotates_world_point_into_robot_frame() -> None:
    half = sqrt(0.5)
    adapter = WaypointFormatAdapter(WaypointAdapterConfig(coordinate_mode="ego_from_odom"), now_fn=lambda: 10.0)
    adapter.update_pose(Pose3D(position=(1.0, 2.0, 0.0), orientation_xyzw=(0.0, 0.0, half, half), frame_id="map"))
    result = adapter.convert((2.0, 2.0, 0.0), frame_id="map", stamp=9.8, now=10.0)
    assert result is not None
    assert result.values[0] == pytest.approx(0.0, abs=1e-6)
    assert result.values[1] == pytest.approx(-1.0, abs=1e-6)


def test_stale_waypoint_is_dropped_without_output() -> None:
    adapter = WaypointFormatAdapter(WaypointAdapterConfig(coordinate_mode="identity", max_input_age_s=1.0), now_fn=lambda: 10.0)
    assert adapter.convert((1.0, 2.0), frame_id="map", stamp=8.9, now=10.0) is None


def test_frame_mismatch_and_direct_velocity_topic_are_rejected() -> None:
    with pytest.raises(WaypointAdapterError, match="frame mismatch"):
        WaypointFormatAdapter(WaypointAdapterConfig(coordinate_mode="identity")).convert((1.0, 2.0), frame_id="odom")
    with pytest.raises(WaypointAdapterError, match="direct velocity"):
        WaypointAdapterConfig(output_topic="/cmd_vel")


def test_static_se2_and_optional_z_are_configurable() -> None:
    adapter = WaypointFormatAdapter(WaypointAdapterConfig(coordinate_mode="static_se2", static_translation_xy_m=(0.5, -0.25), static_yaw_rad=1.5707963267948966, include_z=True))
    result = adapter.convert((1.0, 0.0, 0.75), frame_id="map")
    assert result is not None
    assert result.values == pytest.approx((0.5, 0.75, 0.75))
