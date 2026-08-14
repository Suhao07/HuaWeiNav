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
    detector_device = LaunchConfiguration("detector_device").perform(context)
    detector_imgsz = int(LaunchConfiguration("detector_imgsz").perform(context))
    sam2_checkpoint = LaunchConfiguration("sam2_checkpoint").perform(context)
    platform = LaunchConfiguration("platform").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("1", "true", "yes", "on")
    camera_topic = LaunchConfiguration("camera_topic").perform(context)
    cloud_topic = LaunchConfiguration("cloud_topic").perform(context)
    odom_topic = LaunchConfiguration("odom_topic").perform(context)
    viewpoint_topic = LaunchConfiguration("viewpoint_topic").perform(context)
    detection_topic = LaunchConfiguration("detection_topic").perform(context)
    object_nodes_topic = LaunchConfiguration("object_nodes_topic").perform(context)
    projection_config = LaunchConfiguration("projection_config").perform(context)
    start_semantic_mapping = LaunchConfiguration("start_semantic_mapping").perform(context).lower() in ("1", "true", "yes", "on")
    start_usb_cam = LaunchConfiguration("start_usb_cam").perform(context).lower() in ("1", "true", "yes", "on")
    usb_video_device = LaunchConfiguration("usb_video_device").perform(context)
    usb_image_width = int(LaunchConfiguration("usb_image_width").perform(context))
    usb_image_height = int(LaunchConfiguration("usb_image_height").perform(context))
    usb_pixel_format = LaunchConfiguration("usb_pixel_format").perform(context)
    usb_framerate = float(LaunchConfiguration("usb_framerate").perform(context))
    usb_camera_info_url = LaunchConfiguration("usb_camera_info_url").perform(context)

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
            "device": detector_device,
            "detector_imgsz": detector_imgsz,
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
    if _non_empty(projection_config):
        mapping_parameters[1]["projection_config"] = projection_config

    actions = []
    if _non_empty(sam2_checkpoint):
        # 核心：SAM2 checkpoint 是部署资产，不写死在代码里，由 launch/env 显式传入。
        actions.append(SetEnvironmentVariable("SYSNAV_SAM2_CHECKPOINT", sam2_checkpoint))

    actions.extend(
        [
            *(
                [
                    Node(
                        package="usb_cam",
                        executable="usb_cam_node_exe",
                        name="strive_usb_cam",
                        output="screen",
                        parameters=[
                            {
                                "video_device": usb_video_device,
                                "image_width": usb_image_width,
                                "image_height": usb_image_height,
                                "pixel_format": usb_pixel_format,
                                "framerate": usb_framerate,
                                **(
                                    {"camera_info_url": usb_camera_info_url}
                                    if _non_empty(usb_camera_info_url)
                                    else {}
                                ),
                            }
                        ],
                        remappings=[
                            ("image_raw", camera_topic),
                            ("/image_raw", camera_topic),
                        ],
                    )
                ]
                if start_usb_cam
                else []
            ),
            Node(
                package="semantic_mapping",
                executable="detection_node",
                name="detection_node",
                output="screen",
                parameters=detection_parameters,
                remappings=[
                    ("/camera/image", camera_topic),
                    ("/detection_result", detection_topic),
                ],
            ),
            *(
                [
                    Node(
                        package="semantic_mapping",
                        executable="semantic_mapping_node",
                        name="semantic_mapping_node",
                        output="screen",
                        parameters=mapping_parameters,
                        remappings=[
                            ("/registered_scan", cloud_topic),
                            ("/state_estimation", odom_topic),
                            ("/viewpoint_rep_header", viewpoint_topic),
                            ("/detection_result", detection_topic),
                            ("/object_nodes_list", object_nodes_topic),
                        ],
                    )
                ]
                if start_semantic_mapping
                else []
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
            DeclareLaunchArgument("detector_device", default_value="cuda:0"),
            DeclareLaunchArgument("detector_imgsz", default_value="640"),
            DeclareLaunchArgument("sam2_checkpoint", default_value=""),
            DeclareLaunchArgument("camera_topic", default_value="/camera/image"),
            DeclareLaunchArgument("cloud_topic", default_value="/registered_scan"),
            DeclareLaunchArgument("odom_topic", default_value="/state_estimation"),
            DeclareLaunchArgument("viewpoint_topic", default_value="/viewpoint_rep_header"),
            DeclareLaunchArgument("detection_topic", default_value="/detection_result"),
            DeclareLaunchArgument("object_nodes_topic", default_value="/object_nodes_list"),
            DeclareLaunchArgument("projection_config", default_value=""),
            DeclareLaunchArgument("start_semantic_mapping", default_value="true"),
            DeclareLaunchArgument("start_usb_cam", default_value="false"),
            DeclareLaunchArgument("usb_video_device", default_value="/dev/video0"),
            DeclareLaunchArgument("usb_image_width", default_value="1280"),
            DeclareLaunchArgument("usb_image_height", default_value="720"),
            DeclareLaunchArgument("usb_pixel_format", default_value="yuyv"),
            DeclareLaunchArgument("usb_framerate", default_value="30.0"),
            DeclareLaunchArgument("usb_camera_info_url", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
