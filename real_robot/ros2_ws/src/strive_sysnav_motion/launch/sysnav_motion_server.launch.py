"""Launch the task-level SysNav motion action server."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return the guarded motion server launch description."""

    names = {
        "action_name": "/strive/execute_waypoint",
        "waypoint_topic": "/way_point",
        "odom_topic": "/aft_mapped_to_init",
        "path_topic": "/path",
        "planner_status_topic": "/local_planner/status",
        "hold_topic": "/platform/safe_hold",
        "cancel_topic": "/local_planner/cancel",
        "safety_state_topic": "/platform/safety_state",
        "world_frame": "map",
        "velocity_tolerance_mps": "0.08",
        "stable_reach_time_s": "0.2",
        "allow_look_at": "false",
        "alignment_action_name": "/strive/align_view",
        "alignment_server_wait_timeout_s": "0.5",
        "tf_lookup_timeout_s": "0.2",
        "controller_contract_file": "",
        "require_controller_contract": "true",
    }
    declarations = [DeclareLaunchArgument(name, default_value=value) for name, value in names.items()]
    params = {name: LaunchConfiguration(name) for name in names}
    return LaunchDescription(
        declarations
        + [
            Node(
                package="strive_sysnav_motion",
                executable="sysnav_motion_server",
                name="sysnav_motion_server",
                output="screen",
                parameters=[params],
            )
        ]
    )
