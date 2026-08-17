"""Launch the migrated SysNav local planning stack behind VLN safety gates."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Return the lower-stack launch description with explicit topic remaps."""

    args = {
        "cloud_topic": "/cloud_registered",
        "odom_topic": "/aft_mapped_to_init",
        "waypoint_topic": "/way_point",
        "path_topic": "/path",
        "terrain_map_topic": "/terrain_map",
        "autonomy_cmd_topic": "/cmd_vel/autonomy",
        "manual_cmd_topic": "/cmd_vel/manual",
        "output_cmd_topic": "/cmd_vel",
        "autonomy_enable_topic": "/platform/autonomy_enable",
        "manual_takeover_topic": "/platform/manual_takeover",
        "estop_topic": "/platform/estop_active",
        "estop_reset_topic": "/platform/estop_reset",
        "hold_topic": "/platform/safe_hold",
        "cancel_topic": "/local_planner/cancel",
        "planner_status_topic": "/local_planner/status",
        "action_name": "/strive/execute_waypoint",
        "safety_state_topic": "/platform/safety_state",
        "allow_look_at": "false",
        "alignment_action_name": "/strive/align_view",
        "alignment_server_wait_timeout_s": "0.5",
        "tf_lookup_timeout_s": "0.2",
        "controller_contract_file": "",
        "require_controller_contract": "true",
        "sensor_watchdog_timeout_s": "0.5",
        "world_frame": "map",
        "xy_goal_tolerance_m": "0.35",
        "z_goal_tolerance_m": "1.0",
        "navigation_timeout_s": "60.0",
        "no_progress_timeout_s": "12.0",
        "min_progress_delta_m": "0.05",
        "path_stale_timeout_s": "5.0",
        "velocity_tolerance_mps": "0.08",
        "stable_reach_time_s": "0.2",
        "max_speed": "0.5",
        "max_angular_speed": "1.0",
        "max_linear_accel": "0.5",
        "max_angular_accel": "1.0",
        "command_watchdog_timeout": "0.25",
        "autonomy_speed": "0.3",
        "start_safety_mux": "true",
    }
    declarations = [DeclareLaunchArgument(name, default_value=value) for name, value in args.items()]
    cloud = LaunchConfiguration("cloud_topic")
    odom = LaunchConfiguration("odom_topic")
    waypoint = LaunchConfiguration("waypoint_topic")
    path = LaunchConfiguration("path_topic")
    terrain_map = LaunchConfiguration("terrain_map_topic")

    terrain = Node(
        package="terrain_analysis",
        executable="terrainAnalysis",
        name="terrainAnalysis",
        output="screen",
        remappings=[
            ("/state_estimation", odom),
            ("/registered_scan", cloud),
            ("/terrain_map", terrain_map),
        ],
    )
    local_planner = Node(
        package="local_planner",
        executable="localPlanner",
        name="localPlanner",
        output="screen",
        parameters=[
            {"pathFolder": PathJoinSubstitution([FindPackageShare("local_planner"), "paths"])},
            {"useTerrainAnalysis": True},
            {"maxSpeed": LaunchConfiguration("max_speed")},
            {"cancel_topic": LaunchConfiguration("cancel_topic")},
            {"status_topic": LaunchConfiguration("planner_status_topic")},
        ],
        remappings=[
            ("/state_estimation", odom),
            ("/registered_scan", cloud),
            ("/terrain_map", terrain_map),
            ("/way_point", waypoint),
            ("/path", path),
        ],
    )
    path_follower = Node(
        package="local_planner",
        executable="pathFollower",
        name="pathFollower",
        output="screen",
        parameters=[
            {"autonomyMode": True},
            {"autonomySpeed": LaunchConfiguration("autonomy_speed")},
            {"maxSpeed": LaunchConfiguration("max_speed")},
            {"manual_cmd_topic": LaunchConfiguration("manual_cmd_topic")},
        ],
        remappings=[
            ("/state_estimation", odom),
            ("/path", path),
            ("/cmd_vel/autonomy", LaunchConfiguration("autonomy_cmd_topic")),
        ],
    )
    motion_server = Node(
        package="strive_sysnav_motion",
        executable="sysnav_motion_server",
        name="sysnav_motion_server",
        output="screen",
        parameters=[
            {"world_frame": LaunchConfiguration("world_frame")},
            {"waypoint_topic": waypoint},
            {"odom_topic": odom},
            {"path_topic": path},
            {"planner_status_topic": LaunchConfiguration("planner_status_topic")},
            {"hold_topic": LaunchConfiguration("hold_topic")},
            {"cancel_topic": LaunchConfiguration("cancel_topic")},
            {"safety_state_topic": LaunchConfiguration("safety_state_topic")},
            {"xy_goal_tolerance_m": LaunchConfiguration("xy_goal_tolerance_m")},
            {"z_goal_tolerance_m": LaunchConfiguration("z_goal_tolerance_m")},
            {"navigation_timeout_s": LaunchConfiguration("navigation_timeout_s")},
            {"no_progress_timeout_s": LaunchConfiguration("no_progress_timeout_s")},
            {"min_progress_delta_m": LaunchConfiguration("min_progress_delta_m")},
            {"path_stale_timeout_s": LaunchConfiguration("path_stale_timeout_s")},
            {"velocity_tolerance_mps": LaunchConfiguration("velocity_tolerance_mps")},
            {"stable_reach_time_s": LaunchConfiguration("stable_reach_time_s")},
            {"allow_look_at": LaunchConfiguration("allow_look_at")},
            {"alignment_action_name": LaunchConfiguration("alignment_action_name")},
            {"alignment_server_wait_timeout_s": LaunchConfiguration("alignment_server_wait_timeout_s")},
            {"tf_lookup_timeout_s": LaunchConfiguration("tf_lookup_timeout_s")},
            {"action_name": LaunchConfiguration("action_name")},
            {"controller_contract_file": LaunchConfiguration("controller_contract_file")},
            {"require_controller_contract": LaunchConfiguration("require_controller_contract")},
        ],
    )
    safety_mux = Node(
        package="strive_sysnav_motion",
        executable="safety_velocity_mux",
        name="strive_safety_velocity_mux",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_safety_mux")),
        parameters=[
            {"autonomy_cmd_topic": LaunchConfiguration("autonomy_cmd_topic")},
            {"manual_cmd_topic": LaunchConfiguration("manual_cmd_topic")},
            {"output_cmd_topic": LaunchConfiguration("output_cmd_topic")},
            {"autonomy_enable_topic": LaunchConfiguration("autonomy_enable_topic")},
            {"manual_takeover_topic": LaunchConfiguration("manual_takeover_topic")},
            {"estop_topic": LaunchConfiguration("estop_topic")},
            {"estop_reset_topic": LaunchConfiguration("estop_reset_topic")},
            {"hold_topic": LaunchConfiguration("hold_topic")},
            {"safety_state_topic": LaunchConfiguration("safety_state_topic")},
            {"controller_contract_file": LaunchConfiguration("controller_contract_file")},
            {"require_controller_contract": LaunchConfiguration("require_controller_contract")},
            {"waypoint_topic": waypoint},
            {"world_frame": LaunchConfiguration("world_frame")},
            {"action_name": LaunchConfiguration("action_name")},
            {"planner_status_topic": LaunchConfiguration("planner_status_topic")},
            {"odom_topic": odom},
            {"pointcloud_topic": cloud},
            {"require_sensor_freshness": True},
            {"sensor_watchdog_timeout_s": LaunchConfiguration("sensor_watchdog_timeout_s")},
            {"max_linear_speed_mps": LaunchConfiguration("max_speed")},
            {"max_angular_speed_rps": LaunchConfiguration("max_angular_speed")},
            {"max_linear_accel_mps2": LaunchConfiguration("max_linear_accel")},
            {"max_angular_accel_rps2": LaunchConfiguration("max_angular_accel")},
            {"command_watchdog_timeout_s": LaunchConfiguration("command_watchdog_timeout")},
            {"start_autonomy_enabled": False},
        ],
    )
    # 安全 mux 可以显式关闭用于底层 planner 离线调试；真实运行默认开启，
    # 保证 /cmd_vel 只有一个最终 owner，并以 HOLD 状态启动。
    return LaunchDescription(declarations + [terrain, local_planner, path_follower, motion_server, safety_mux])
