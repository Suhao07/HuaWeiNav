"""Launch the single-owner safety velocity mux."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return the safety mux launch description."""

    names = {
        "autonomy_cmd_topic": "/cmd_vel/autonomy",
        "manual_cmd_topic": "/cmd_vel/manual",
        "output_cmd_topic": "/cmd_vel",
        "autonomy_enable_topic": "/platform/autonomy_enable",
        "manual_takeover_topic": "/platform/manual_takeover",
        "estop_topic": "/platform/estop_active",
        "estop_reset_topic": "/platform/estop_reset",
        "hold_topic": "/platform/safe_hold",
        "safety_state_topic": "/platform/safety_state",
        "odom_topic": "/aft_mapped_to_init",
        "pointcloud_topic": "/cloud_registered",
        "require_sensor_freshness": "true",
        "sensor_watchdog_timeout_s": "0.5",
        "max_linear_speed_mps": "0.5",
        "max_angular_speed_rps": "1.0",
        "max_linear_accel_mps2": "0.5",
        "max_angular_accel_rps2": "1.0",
        "command_watchdog_timeout_s": "0.25",
    }
    declarations = [DeclareLaunchArgument(name, default_value=value) for name, value in names.items()]
    params = {name: LaunchConfiguration(name) for name in names}
    return LaunchDescription(
        declarations
        + [
            Node(
                package="strive_sysnav_motion",
                executable="safety_velocity_mux",
                name="strive_safety_velocity_mux",
                output="screen",
                parameters=[params],
            )
        ]
    )
