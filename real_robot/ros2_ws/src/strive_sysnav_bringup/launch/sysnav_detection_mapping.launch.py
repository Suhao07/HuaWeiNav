import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _non_empty(value):
    return value is not None and str(value).strip() != ""


def launch_setup(context, *args, **kwargs):
    semantic_share = get_package_share_directory("semantic_mapping")
    mapping_config = LaunchConfiguration("mapping_config").perform(context)
    object_file = LaunchConfiguration("object_file").perform(context)
    detector_model_path = LaunchConfiguration("detector_model_path").perform(context)
    detector_model_type = LaunchConfiguration("detector_model_type").perform(context)
    sam2_checkpoint = LaunchConfiguration("sam2_checkpoint").perform(context)
    platform = LaunchConfiguration("platform").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("1", "true", "yes", "on")

    if not _non_empty(mapping_config):
        mapping_config = os.path.join(semantic_share, "mapping_mecanum_real.yaml")
    if not _non_empty(object_file):
        object_file = os.path.join(semantic_share, "config", "objects.yaml")

    detection_parameters = [
        mapping_config,
        {
            "platform": platform,
            "use_sim_time": use_sim_time,
            "object_file": object_file,
            "detector_model_type": detector_model_type,
        },
    ]
    if _non_empty(detector_model_path):
        detection_parameters[1]["detector_model_path"] = detector_model_path

    mapping_parameters = [
        mapping_config,
        {
            "platform": platform,
            "use_sim_time": use_sim_time,
            "object_file": object_file,
        },
    ]

    actions = []
    if _non_empty(sam2_checkpoint):
        # 核心：SAM2 checkpoint 是部署资产，不写死在代码里，由 launch/env 显式传入。
        actions.append(SetEnvironmentVariable("SYSNAV_SAM2_CHECKPOINT", sam2_checkpoint))

    actions.extend(
        [
            Node(
                package="semantic_mapping",
                executable="detection_node",
                name="detection_node",
                output="screen",
                parameters=detection_parameters,
            ),
            Node(
                package="semantic_mapping",
                executable="semantic_mapping_node",
                name="semantic_mapping_node",
                output="screen",
                parameters=mapping_parameters,
            ),
        ]
    )
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("platform", default_value="mecanum"),
            DeclareLaunchArgument("mapping_config", default_value=""),
            DeclareLaunchArgument("object_file", default_value=""),
            DeclareLaunchArgument("detector_model_type", default_value="yoloe"),
            DeclareLaunchArgument("detector_model_path", default_value=""),
            DeclareLaunchArgument("sam2_checkpoint", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
