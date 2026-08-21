"""Launch the configurable STRIVE waypoint format adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return a launch description with safe output disabled by default."""

    names = (
        ("config_path", ""),
        ("input_topic", "/way_point"),
        ("output_topic", "/waypoint"),
        ("odom_topic", "/aft_mapped_to_init"),
        ("input_frame", "map"),
        ("output_frame", "base_link"),
        ("coordinate_mode", "ego_from_odom"),
        ("output_message_type", "std_msgs/msg/Float32MultiArray"),
        ("include_z", "false"),
        ("max_input_age_s", "1.0"),
        ("output_enabled", "false"),
        ("static_translation_xy_m", "[0.0, 0.0]"),
        ("static_yaw_rad", "0.0"),
    )
    arguments = [DeclareLaunchArgument(name, default_value=value) for name, value in names]
    parameters = {name: LaunchConfiguration(name) for name, _ in names}
    return LaunchDescription(
        arguments
        + [
            Node(
                package="strive_sysnav_bringup",
                executable="strive_waypoint_adapter",
                name="strive_waypoint_adapter",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
