"""Launch the STRIVE high-level instruction runtime node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Return launch description for the high-level runtime node."""

    return LaunchDescription(
        [
            DeclareLaunchArgument("instruction", default_value=""),
            DeclareLaunchArgument("object_topic", default_value="/object_nodes_list"),
            DeclareLaunchArgument("room_topic", default_value="/room_nodes_list"),
            DeclareLaunchArgument("odom_topic", default_value="/aft_mapped_to_init"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image"),
            DeclareLaunchArgument("waypoint_topic", default_value="/way_point"),
            DeclareLaunchArgument("world_frame", default_value="map"),
            DeclareLaunchArgument("policy_mode", default_value="wait"),
            DeclareLaunchArgument("prior_map_path", default_value=""),
            DeclareLaunchArgument("run_directory", default_value="/tmp/strive_real_robot_runtime"),
            DeclareLaunchArgument("decision_period_s", default_value="1.0"),
            DeclareLaunchArgument("queue_size", default_value="10"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("require_pose", default_value="true"),
            DeclareLaunchArgument("require_image", default_value="true"),
            Node(
                package="strive_sysnav_bringup",
                executable="strive_instruction_runtime",
                name="strive_instruction_runtime",
                output="screen",
                parameters=[
                    {
                        "instruction": LaunchConfiguration("instruction"),
                        "object_topic": LaunchConfiguration("object_topic"),
                        "room_topic": LaunchConfiguration("room_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "waypoint_topic": LaunchConfiguration("waypoint_topic"),
                        "world_frame": LaunchConfiguration("world_frame"),
                        "policy_mode": LaunchConfiguration("policy_mode"),
                        "prior_map_path": LaunchConfiguration("prior_map_path"),
                        "run_directory": LaunchConfiguration("run_directory"),
                        "decision_period_s": LaunchConfiguration("decision_period_s"),
                        "queue_size": LaunchConfiguration("queue_size"),
                        "dry_run": LaunchConfiguration("dry_run"),
                        "require_pose": LaunchConfiguration("require_pose"),
                        "require_image": LaunchConfiguration("require_image"),
                    }
                ],
            ),
        ]
    )
