#!/usr/bin/env python3
"""Run task-free baseline patrol and save portable, unprocessed RGB-D frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    """Use a frozen Habitat dataset for scene/start selection, excluding goals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Frozen Habitat episode file providing scene and start pose")
    parser.add_argument("output")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    import habitat
    import numpy as np
    import quaternion
    import random
    import torch
    from config_utils import hm3d_config
    from constants import categories
    from benchmark.episode_selection import select_episode
    from mapping_utils.transform import habitat_camera_intrinsic
    from mapper_with_process_obs import Instruct_Mapper
    from objnav_agent_with_process_obs import HM3D_Objnav_Agent
    from backends.habitat.capture import WORLD_TO_MAP, OPTICAL_TO_GL
    from backends.habitat.patrol_runtime import run_patrol
    from prior_map.observations import SCHEMA_VERSION, COORDINATES, write_frame, write_json

    if args.max_steps < 13 or args.episode_index < 0:
        parser.error("Use at least 13 observation steps and a nonnegative episode index")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    capture = output / "capture"
    capture.mkdir()
    config = hm3d_config(dataset_path=str(Path(args.dataset).resolve()), episodes=args.episode_index + 1)
    # 当前维护环境使用 Habitat 0.2；采集禁用任务指标和目标传感器。
    config.defrost()
    config.TASK.MEASUREMENTS = []
    config.TASK.SENSORS = []
    config.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    config.SEED = args.seed
    config.freeze()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (output / "habitat_config.yaml").write_text(config.dump())
    intrinsic = habitat_camera_intrinsic(config)
    manifest = {"schema_version": SCHEMA_VERSION, "coordinates": COORDINATES,
        "status": "running", "mode": "baseline_patrol", "pose_source": "habitat_gt_camera_pose",
        "semantic_gt_used": False, "start_source": "frozen_episode_start",
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "episode_index": args.episode_index, "max_steps": args.max_steps, "seed": args.seed,
        "collector": "STRIVE_room_first_distance_policy", "frame_count": 0}
    write_json(capture / "manifest.json", manifest)
    started = time.monotonic()
    with habitat.Env(config) as env:
        env.seed(args.seed)
        select_episode(env, args.episode_index)
        def record(agent):
            state = env.sim.get_agent_state()
            camera = state.sensor_states["rgb"]
            pose = np.eye(4)
            pose[:3, :3] = WORLD_TO_MAP @ quaternion.as_rotation_matrix(camera.rotation) @ OPTICAL_TO_GL
            pose[:3, 3] = WORLD_TO_MAP @ camera.position
            frame_id = agent.episode_steps - 1
            write_frame(capture, frame_id, agent.obs["rgb"][..., :3],
                agent.obs["depth"].squeeze(-1), intrinsic, pose, sim_step=frame_id,
                agent_position_habitat=state.position.tolist())
            manifest["frame_count"] += 1
        mapper = Instruct_Mapper(intrinsic, voxel_dimension=[500, 500, 30],
            save_dir=str(output), categories=categories, no_gpt_seg=True, env=env)
        agent = HM3D_Objnav_Agent(env, mapper, str(output), gpt_relocate=False,
                                 observation_recorder=record)
        agent.reset(0, patrol=True)
        scene = Path(env.current_episode.scene_id)
        rgb = config.SIMULATOR.RGB_SENSOR
        depth = config.SIMULATOR.DEPTH_SENSOR
        state = env.sim.get_agent_state()
        agent_config = env.sim.sim_config.agents[0]
        # 导出实际运行网格供对照加载；这里只记录运行配置，不冒充重建参数回执。
        mesh = output / "runtime.navmesh"
        env.sim.pathfinder.save_nav_mesh(str(mesh))
        scene_sha = hashlib.sha256(scene.read_bytes()).hexdigest()
        mesh_sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
        settings = {"agent_radius": agent_config.radius, "agent_height": agent_config.height}
        write_json(output / "runtime_navmesh_receipt.json", {
            "status": "complete", "scene_sha256": scene_sha, "navmesh_sha256": mesh_sha,
            "settings": settings, "scope": "exported_active_mesh_with_runtime_agent_dimensions"})
        profile = {"scene_sha256": scene_sha, "navmesh_sha256": mesh_sha,
            "width": rgb.WIDTH, "height": rgb.HEIGHT, "hfov": rgb.HFOV,
            "sensor_height": rgb.POSITION[1], "depth_min_m": depth.MIN_DEPTH,
            "depth_max_m": depth.MAX_DEPTH, "intrinsics": intrinsic.tolist(),
            "start_position_habitat": state.position.tolist(),
            "start_rotation_wxyz": quaternion.as_float_array(state.rotation).tolist(),
            "move_forward_m": agent_config.action_space[1].actuation.amount,
            "turn_degrees": agent_config.action_space[2].actuation.amount,
            "scan_action": "turn_right",
            "arrival_radius_m": agent.planner._goal_radius,
            "max_actions": args.max_steps - 1, "seed": args.seed}
        write_json(output / "capture_profile.json", profile)
        manifest.update(scene_id=scene.stem, scene_path=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
            config={"width": rgb.WIDTH, "height": rgb.HEIGHT, "hfov": rgb.HFOV,
                    "sensor_height": rgb.POSITION[1]},
            start_position_habitat=state.position.tolist(),
            capture_profile=profile)
        write_json(capture / "manifest.json", manifest)
        receipt = run_patrol(agent, episode_idx=0, max_steps=args.max_steps)
        manifest.update(status="complete", termination=receipt["termination"],
                        elapsed_seconds=time.monotonic() - started,
                        actual_path_length_m=receipt["path_length_m"])
        write_json(capture / "manifest.json", manifest)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
