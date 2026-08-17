from pathlib import Path


def test_vendored_sysnav_path_follower_has_no_direct_chassis_output() -> None:
    source = Path("real_robot/ros2_ws/src/local_planner/src/pathFollower.cpp").read_text(encoding="utf-8")

    assert '"/cmd_vel/autonomy"' in source
    assert '"/cmd_vel"' not in source.replace('"/cmd_vel/autonomy"', "")
    assert "serial::" not in source
    assert "motorCtrSerial" not in source


def test_lower_stack_documents_single_safety_mux_owner() -> None:
    source = Path("real_robot/ros2_ws/src/SYSNAV_MIGRATION.md").read_text(encoding="utf-8")

    assert "/cmd_vel/autonomy" in source
    assert "SafetyVelocityMux" in source
    assert "single-owner" in source


def test_lower_stack_does_not_pass_cancel_to_the_velocity_mux() -> None:
    source = Path(
        "real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py"
    ).read_text(encoding="utf-8")

    mux_block = source.split("safety_mux =", 1)[1]
    assert '"cancel_topic"' not in mux_block


def test_real_robot_image_builds_the_task_motion_overlay() -> None:
    dockerfile = Path("docker/Dockerfile.real_robot").read_text(encoding="utf-8")

    for package in ("terrain_analysis", "local_planner", "strive_motion_msgs", "strive_sysnav_motion"):
        assert package in dockerfile


def test_real_robot_image_keeps_sysnav_path_assets_in_context() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "!real_robot/ros2_ws/src/local_planner/paths/*.ply" in dockerignore


def test_local_planner_cancel_clears_the_path_follower_input() -> None:
    source = Path("real_robot/ros2_ws/src/local_planner/src/localPlanner.cpp").read_text(encoding="utf-8")

    assert "std_msgs::msg::Empty" in source
    assert '"/local_planner/cancel"' in source
    assert "cancelRequested" in source
    assert "path.poses.resize(1)" in source


def test_local_planner_publishes_explicit_status_and_path_follower_splits_manual_cmd() -> None:
    planner = Path("real_robot/ros2_ws/src/local_planner/src/localPlanner.cpp").read_text(encoding="utf-8")
    follower = Path("real_robot/ros2_ws/src/local_planner/src/pathFollower.cpp").read_text(encoding="utf-8")

    assert "std_msgs::msg::String" in planner
    assert '"no_feasible_path"' in planner
    assert '"tracking"' in planner
    assert '"manual_cmd_topic"' in follower
    assert 'create_publisher<geometry_msgs::msg::TwistStamped>(manualCmdTopic' in follower
    assert 'manual_cmd.header.stamp = nh->now();' in follower


def test_motion_action_has_explicit_view_alignment_contract() -> None:
    execute_action = Path(
        "real_robot/ros2_ws/src/strive_motion_msgs/action/ExecuteWaypoint.action"
    ).read_text(encoding="utf-8")
    align_action = Path(
        "real_robot/ros2_ws/src/strive_motion_msgs/action/AlignView.action"
    ).read_text(encoding="utf-8")
    hil = Path(
        "real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/motion_hil.py"
    ).read_text(encoding="utf-8")

    assert "bool view_aligned" in execute_action
    assert "uint8 ALIGNED=0" in align_action
    assert "motion_hil" in Path("real_robot/ros2_ws/src/strive_sysnav_motion/setup.py").read_text(encoding="utf-8")
    # HIL only publishes lower-layer state. The production safety mux remains
    # the sole owner of the base velocity command.
    assert "create_publisher(TwistStamped" not in hil
    assert "native_planner" in hil
    assert "/hil/registered_scan" in hil
    assert "native_path_received" in hil
    assert "native_path_messages" in hil
    assert "native_safety" in hil
    assert "nonzero_final_cmd_messages" in hil
    assert "artifact_path" in hil
    assert "latest_final_linear_x" in hil
    assert "create_subscription(TwistStamped" in hil


def test_bag_replay_has_topic_gate_and_deterministic_runtime_boundary() -> None:
    source = Path("scripts/run_real_robot_bag_replay.sh").read_text(encoding="utf-8")

    assert "BAG_REQUIRED_TOPICS" in source
    assert "bag_info.txt" in source
    assert "check_required_topics" in source
    assert "BAG_RUNTIME_GRACE_S" in source
    assert "BAG_REQUIRE_RUNTIME_DECISION" in source
    assert "waypoint_to_path_acceptance=native_planner_hil_only" in source


def test_safety_mux_receives_the_same_contract_gate_as_motion_server() -> None:
    source = Path(
        "real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py"
    ).read_text(encoding="utf-8")

    safety_block = source.split("safety_mux =", 1)[1]
    assert '"controller_contract_file"' in safety_block
    assert '"require_controller_contract"' in safety_block
    assert '"waypoint_topic"' in safety_block
    assert '"planner_status_topic"' in safety_block
    assert '"action_name"' in safety_block


def test_lower_bag_probe_is_path_only_and_has_sensor_evidence() -> None:
    source = Path(
        "real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/lower_bag_probe.py"
    ).read_text(encoding="utf-8")
    setup = Path("real_robot/ros2_ws/src/strive_sysnav_motion/setup.py").read_text(encoding="utf-8")

    assert "lower_bag_probe" in setup
    assert "valid_path_messages" in source
    assert "pointcloud_messages" in source
    assert '"/cmd_vel"' not in source


def test_lower_planner_bag_replay_starts_no_velocity_controller() -> None:
    source = Path("scripts/run_lower_planner_bag_replay.sh").read_text(encoding="utf-8")

    assert "localPlanner" in source
    assert "lower_bag_probe" in source
    assert "ros2 run local_planner pathFollower" not in source
    assert "ros2 run strive_sysnav_motion safety_velocity_mux" not in source
    assert "cmd_vel_started=false" in source
    assert "LOWER_BAG_REQUIRED_TOPICS" in source


def test_synthetic_lower_bag_smoke_is_explicitly_non_hardware() -> None:
    generator = Path("scripts/generate_synthetic_lower_planner_bag.py").read_text(encoding="utf-8")
    smoke = Path("scripts/run_lower_planner_bag_smoke.sh").read_text(encoding="utf-8")

    assert "rosbag2_py.SequentialWriter" in generator
    assert "/aft_mapped_to_init" in generator
    assert "/cloud_registered" in generator
    assert "synthetic" in generator.lower()
    assert "run_lower_planner_bag_replay.sh" in smoke
    assert "LOWER_BAG_REQUIRED_TOPICS" in smoke


def test_production_aggregate_launch_is_explicit_and_action_backed() -> None:
    launch = Path(
        "real_robot/ros2_ws/src/strive_sysnav_bringup/launch/strive_real_robot_stack.launch.py"
    ).read_text(encoding="utf-8")
    runner = Path("scripts/run_sysnav_detection_mapping.sh").read_text(encoding="utf-8")

    assert '"enable_lower_stack": "false"' in launch
    assert '"motion_backend": "action"' in launch
    assert "sysnav_lower_stack.launch.py" in launch
    assert "strive_instruction_runtime.launch.py" in launch
    assert "START_LOWER_STACK" in runner
    assert "strive_real_robot_stack.launch.py" in runner
    assert "START_LOWER_STACK=1" in runner
    assert 'STRIVE_DRY_RUN:-true' in runner
    assert 'STRIVE_LOWER_CONTROLLER_ENABLED:-false' in runner
    assert "CONTROL_CONTRACT_FILE" in runner
    assert 'runtime_vlm="${STRIVE_VLM:-${STRIVE_LLM_CLIENT:-${LLM_PROVIDER:-cognav}}}"' in runner


def test_native_safety_hil_starts_the_complete_lower_motion_chain() -> None:
    runner = Path("scripts/run_motion_hil.sh").read_text(encoding="utf-8")

    assert "native_safety" in runner
    assert "pathFollower" in runner
    assert "safety_velocity_mux" in runner
    assert "cmd_vel/autonomy" in runner
    assert "nonzero_final_cmd_messages" not in runner
