"""Launch the HuaweiNav to AgileX rosbridge boundary."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return the bridge, telemetry and remote mux mode node."""

    defaults = {
        "input_cmd_topic": "/cmd_vel",
        "uri": "ws://192.168.1.102:9090",
        "interface": "",
        "remote_control_enabled": "false",
        "controller_contract_file": "",
        "waypoint_topic": "/way_point",
        "world_frame": "map",
        "action_name": "/strive/execute_waypoint",
        "auto_topic": "/navflow_cmd_vel",
        "manual_topic": "/run_management/virtual_joy",
        "autonomy_enable_topic": "/platform/autonomy_enable",
        "manual_takeover_topic": "/platform/manual_takeover",
        "chassis_ready_topic": "/platform/chassis_ready",
        "command_timeout_s": "1.0",
        "max_linear_mps": "0.5",
        "max_angular_rps": "0.8",
    }
    declarations = [DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()]
    node = Node(
        package="strive_sysnav_motion",
        executable="agilex_platform",
        name="strive_agilex_platform",
        output="screen",
        parameters=[{name: LaunchConfiguration(name) for name in defaults}],
    )
    return LaunchDescription(declarations + [node])
