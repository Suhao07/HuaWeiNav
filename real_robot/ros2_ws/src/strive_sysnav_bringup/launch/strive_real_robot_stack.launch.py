"""Bring up the complete VLN real-robot perception and motion stack.

The lower controller is intentionally conditional.  A default launch only
starts perception and the high-level runtime in dry-run mode; production
motion requires an explicit lower-stack flag and the shared controller
contract validated by every motion boundary.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _arg(name: str, default: str) -> DeclareLaunchArgument:
    """Create one documented stack launch argument."""

    return DeclareLaunchArgument(name, default_value=default)


def generate_launch_description() -> LaunchDescription:
    """Return the guarded perception, planning, and runtime launch graph."""

    # Keep the interface explicit: a robot profile can remap sensor and
    # control topics without editing any package source.
    defaults = {
        "enable_lower_stack": "false",
        "platform": "mecanum",
        "start_semantic_mapping": "true",
        "start_viewpoint_bridge": "true",
        "start_usb_cam": "false",
        "camera_topic": "/camera/image",
        "cloud_topic": "/cloud_registered",
        "odom_topic": "/aft_mapped_to_init",
        "viewpoint_topic": "/viewpoint_rep_header",
        "viewpoint_pose_topic": "/strive/sysnav/viewpoint_pose",
        "detection_topic": "/detection_result",
        "object_nodes_topic": "/object_nodes_list",
        "object_topic": "/object_nodes_list",
        "room_topic": "/room_nodes_list",
        "mapping_config": "",
        "projection_config": "",
        "detector_model_type": "yoloe",
        "detector_model_path": "",
        "sam2_checkpoint": "",
        "usb_video_device": "/dev/video0",
        "usb_image_width": "1280",
        "usb_image_height": "720",
        "usb_pixel_format": "yuyv",
        "usb_framerate": "30.0",
        "usb_camera_info_url": "",
        "instruction": "",
        "dataset_target": "",
        "policy_mode": "wait",
        "instruction_plan_backend": "rules",
        "vlm": "cognav",
        "motion_backend": "action",
        "motion_action_name": "/strive/execute_waypoint",
        "controller_contract_file": "",
        "dry_run": "true",
        "dry_run_status": "idle",
        "lower_controller_enabled": "false",
        "waypoint_topic": "/way_point",
        "test_waypoint_topic": "/strive/test_way_point",
        "hold_topic": "/platform/safe_hold",
        "cancel_topic": "/local_planner/cancel",
        "emergency_stop_topic": "",
        "allow_emergency_stop_publish": "false",
        "enable_final_verifier": "false",
        "evidence_mode": "auto",
        "prior_map_path": "",
        "prior_map_source": "auto",
        "prior_map_alignment": "identity",
        "enable_prior_map_vlm": "false",
        "enable_room_semantics": "false",
        "run_directory": "/tmp/strive_real_robot_runtime",
        "decision_period_s": "1.0",
        "use_sim_time": "false",
        "require_pose": "true",
        "require_image": "true",
        "pointcloud_topic": "/cloud_registered",
        "depth_topic": "",
        "path_topic": "/path",
        "planner_status_topic": "/local_planner/status",
        "world_frame": "map",
        "xy_goal_tolerance_m": "0.35",
        "z_goal_tolerance_m": "1.0",
        "navigation_timeout_s": "60.0",
        "no_progress_timeout_s": "12.0",
        "min_progress_delta_m": "0.05",
        "path_stale_timeout_s": "5.0",
        "velocity_tolerance_mps": "0.08",
        "stable_reach_time_s": "0.2",
        "persist_observation_images": "false",
        "observation_image_directory": "",
        "terrain_map_topic": "/terrain_map",
        "autonomy_cmd_topic": "/cmd_vel/autonomy",
        "manual_cmd_topic": "/cmd_vel/manual",
        "output_cmd_topic": "/cmd_vel",
        "autonomy_enable_topic": "/platform/autonomy_enable",
        "manual_takeover_topic": "/platform/manual_takeover",
        "estop_topic": "/platform/estop_active",
        "estop_reset_topic": "/platform/estop_reset",
        "safety_state_topic": "/platform/safety_state",
        "sensor_watchdog_timeout_s": "0.5",
        "max_speed": "0.5",
        "max_angular_speed": "1.0",
        "max_linear_accel": "0.5",
        "max_angular_accel": "1.0",
        "command_watchdog_timeout": "0.25",
        "autonomy_speed": "0.3",
        "start_safety_mux": "true",
        "require_controller_contract": "true",
        "allow_look_at": "false",
        "alignment_action_name": "/strive/align_view",
        "alignment_server_wait_timeout_s": "0.5",
        "tf_lookup_timeout_s": "0.2",
    }
    declarations = [_arg(name, value) for name, value in defaults.items()]

    detection_arguments = {
        "platform": LaunchConfiguration("platform"),
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "camera_topic": LaunchConfiguration("camera_topic"),
        "cloud_topic": LaunchConfiguration("cloud_topic"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "viewpoint_topic": LaunchConfiguration("viewpoint_topic"),
        "detection_topic": LaunchConfiguration("detection_topic"),
        "object_nodes_topic": LaunchConfiguration("object_nodes_topic"),
        "mapping_config": LaunchConfiguration("mapping_config"),
        "projection_config": LaunchConfiguration("projection_config"),
        "start_semantic_mapping": LaunchConfiguration("start_semantic_mapping"),
        "start_usb_cam": LaunchConfiguration("start_usb_cam"),
        "detector_model_type": LaunchConfiguration("detector_model_type"),
        "detector_model_path": LaunchConfiguration("detector_model_path"),
        "sam2_checkpoint": LaunchConfiguration("sam2_checkpoint"),
        "usb_video_device": LaunchConfiguration("usb_video_device"),
        "usb_image_width": LaunchConfiguration("usb_image_width"),
        "usb_image_height": LaunchConfiguration("usb_image_height"),
        "usb_pixel_format": LaunchConfiguration("usb_pixel_format"),
        "usb_framerate": LaunchConfiguration("usb_framerate"),
        "usb_camera_info_url": LaunchConfiguration("usb_camera_info_url"),
    }

    lower_arguments = {
        "cloud_topic": LaunchConfiguration("cloud_topic"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "waypoint_topic": LaunchConfiguration("waypoint_topic"),
        "path_topic": LaunchConfiguration("path_topic"),
        "terrain_map_topic": LaunchConfiguration("terrain_map_topic"),
        "autonomy_cmd_topic": LaunchConfiguration("autonomy_cmd_topic"),
        "manual_cmd_topic": LaunchConfiguration("manual_cmd_topic"),
        "output_cmd_topic": LaunchConfiguration("output_cmd_topic"),
        "autonomy_enable_topic": LaunchConfiguration("autonomy_enable_topic"),
        "manual_takeover_topic": LaunchConfiguration("manual_takeover_topic"),
        "estop_topic": LaunchConfiguration("estop_topic"),
        "estop_reset_topic": LaunchConfiguration("estop_reset_topic"),
        "hold_topic": LaunchConfiguration("hold_topic"),
        "cancel_topic": LaunchConfiguration("cancel_topic"),
        "planner_status_topic": LaunchConfiguration("planner_status_topic"),
        "safety_state_topic": LaunchConfiguration("safety_state_topic"),
        "action_name": LaunchConfiguration("motion_action_name"),
        "allow_look_at": LaunchConfiguration("allow_look_at"),
        "alignment_action_name": LaunchConfiguration("alignment_action_name"),
        "alignment_server_wait_timeout_s": LaunchConfiguration("alignment_server_wait_timeout_s"),
        "tf_lookup_timeout_s": LaunchConfiguration("tf_lookup_timeout_s"),
        "controller_contract_file": LaunchConfiguration("controller_contract_file"),
        "require_controller_contract": LaunchConfiguration("require_controller_contract"),
        "sensor_watchdog_timeout_s": LaunchConfiguration("sensor_watchdog_timeout_s"),
        "world_frame": LaunchConfiguration("world_frame"),
        "xy_goal_tolerance_m": LaunchConfiguration("xy_goal_tolerance_m"),
        "z_goal_tolerance_m": LaunchConfiguration("z_goal_tolerance_m"),
        "navigation_timeout_s": LaunchConfiguration("navigation_timeout_s"),
        "no_progress_timeout_s": LaunchConfiguration("no_progress_timeout_s"),
        "min_progress_delta_m": LaunchConfiguration("min_progress_delta_m"),
        "path_stale_timeout_s": LaunchConfiguration("path_stale_timeout_s"),
        "velocity_tolerance_mps": LaunchConfiguration("velocity_tolerance_mps"),
        "stable_reach_time_s": LaunchConfiguration("stable_reach_time_s"),
        "max_speed": LaunchConfiguration("max_speed"),
        "max_angular_speed": LaunchConfiguration("max_angular_speed"),
        "max_linear_accel": LaunchConfiguration("max_linear_accel"),
        "max_angular_accel": LaunchConfiguration("max_angular_accel"),
        "command_watchdog_timeout": LaunchConfiguration("command_watchdog_timeout"),
        "autonomy_speed": LaunchConfiguration("autonomy_speed"),
        "start_safety_mux": LaunchConfiguration("start_safety_mux"),
    }

    runtime_arguments = {
        "instruction": LaunchConfiguration("instruction"),
        "object_topic": LaunchConfiguration("object_topic"),
        "room_topic": LaunchConfiguration("room_topic"),
        "viewpoint_pose_topic": LaunchConfiguration("viewpoint_pose_topic"),
        "odom_topic": LaunchConfiguration("odom_topic"),
        "path_topic": LaunchConfiguration("path_topic"),
        "planner_status_topic": LaunchConfiguration("planner_status_topic"),
        "image_topic": LaunchConfiguration("camera_topic"),
        "detection_topic": LaunchConfiguration("detection_topic"),
        "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
        "depth_topic": LaunchConfiguration("depth_topic"),
        "waypoint_topic": LaunchConfiguration("waypoint_topic"),
        "test_waypoint_topic": LaunchConfiguration("test_waypoint_topic"),
        "motion_backend": LaunchConfiguration("motion_backend"),
        "motion_action_name": LaunchConfiguration("motion_action_name"),
        "controller_contract_file": LaunchConfiguration("controller_contract_file"),
        "hold_topic": LaunchConfiguration("hold_topic"),
        "cancel_topic": LaunchConfiguration("cancel_topic"),
        "emergency_stop_topic": LaunchConfiguration("emergency_stop_topic"),
        "allow_emergency_stop_publish": LaunchConfiguration("allow_emergency_stop_publish"),
        "lower_controller_enabled": LaunchConfiguration("lower_controller_enabled"),
        "world_frame": LaunchConfiguration("world_frame"),
        "policy_mode": LaunchConfiguration("policy_mode"),
        "dataset_target": LaunchConfiguration("dataset_target"),
        "instruction_plan_backend": LaunchConfiguration("instruction_plan_backend"),
        "vlm": LaunchConfiguration("vlm"),
        "enable_final_verifier": LaunchConfiguration("enable_final_verifier"),
        "evidence_mode": LaunchConfiguration("evidence_mode"),
        "prior_map_path": LaunchConfiguration("prior_map_path"),
        "prior_map_source": LaunchConfiguration("prior_map_source"),
        "prior_map_alignment": LaunchConfiguration("prior_map_alignment"),
        "enable_prior_map_vlm": LaunchConfiguration("enable_prior_map_vlm"),
        "enable_room_semantics": LaunchConfiguration("enable_room_semantics"),
        "run_directory": LaunchConfiguration("run_directory"),
        "decision_period_s": LaunchConfiguration("decision_period_s"),
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "dry_run": LaunchConfiguration("dry_run"),
        "dry_run_status": LaunchConfiguration("dry_run_status"),
        "require_pose": LaunchConfiguration("require_pose"),
        "require_image": LaunchConfiguration("require_image"),
        "xy_goal_tolerance_m": LaunchConfiguration("xy_goal_tolerance_m"),
        "z_goal_tolerance_m": LaunchConfiguration("z_goal_tolerance_m"),
        "navigation_timeout_s": LaunchConfiguration("navigation_timeout_s"),
        "no_progress_timeout_s": LaunchConfiguration("no_progress_timeout_s"),
        "min_progress_delta_m": LaunchConfiguration("min_progress_delta_m"),
        "path_stale_timeout_s": LaunchConfiguration("path_stale_timeout_s"),
        "velocity_tolerance_mps": LaunchConfiguration("velocity_tolerance_mps"),
        "stable_reach_time_s": LaunchConfiguration("stable_reach_time_s"),
        "persist_observation_images": LaunchConfiguration("persist_observation_images"),
        "observation_image_directory": LaunchConfiguration("observation_image_directory"),
    }

    detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("strive_sysnav_bringup"), "launch", "sysnav_detection_mapping.launch.py"]
            )
        ),
        launch_arguments=detection_arguments.items(),
    )
    viewpoint_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("strive_sysnav_bringup"), "launch", "sysnav_viewpoint_bridge.launch.py"]
            )
        ),
        launch_arguments={
            "viewpoint_topic": LaunchConfiguration("viewpoint_topic"),
            "object_topic": LaunchConfiguration("object_nodes_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "output_topic": LaunchConfiguration("viewpoint_pose_topic"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_viewpoint_bridge")),
    )
    lower_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("strive_sysnav_motion"), "launch", "sysnav_lower_stack.launch.py"]
            )
        ),
        launch_arguments=lower_arguments.items(),
        condition=IfCondition(LaunchConfiguration("enable_lower_stack")),
    )
    runtime_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("strive_sysnav_bringup"), "launch", "strive_instruction_runtime.launch.py"]
            )
        ),
        launch_arguments=runtime_arguments.items(),
    )

    # 中文说明：lower stack 只有显式 enable 才存在；高层 runtime 始终运行，
    # 但默认 dry-run，避免“启动了感知”被误解为“已经获得底盘控制权”。
    return LaunchDescription(declarations + [detection_launch, viewpoint_bridge_launch, lower_launch, runtime_launch])
