"""Launch the minimal SysNav viewpoint-pose bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return a launch description for timestamp-aligned viewpoint poses."""

    names = (
        ("viewpoint_topic", "/viewpoint_rep_header"),
        ("object_topic", "/object_nodes_list"),
        ("odom_topic", "/state_estimation"),
        ("output_topic", "/strive/sysnav/viewpoint_pose"),
        ("odom_history_size", "400"),
        ("max_time_offset_s", "0.25"),
        ("queue_size", "10"),
    )
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in names]
        + [
            Node(
                package="strive_sysnav_bringup",
                executable="strive_sysnav_viewpoint_bridge",
                name="strive_sysnav_viewpoint_bridge",
                output="screen",
                parameters=[{name: LaunchConfiguration(name) for name, _ in names}],
            )
        ]
    )
