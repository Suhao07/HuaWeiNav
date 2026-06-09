import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import habitat
from loguru import logger
from matplotlib.pyplot import savefig
from tqdm import tqdm

from config_utils import hm3d_config, mp3d_config
from constants import *
from cv_utils.gpt_utils import ask_gpt_similar_objects
#指令解析模块
from instruction_adapter import (
    StriveInstructionParser,
    extract_dataset_target,
    render_instruction_context,
)
from mapper_with_process_obs import Instruct_Mapper
from mapping_utils.transform import habitat_camera_intrinsic
from objnav_agent_with_process_obs import HM3D_Objnav_Agent

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def _set_next_episode_by_index(env, index):
    # 新版 Habitat iterator 提供 set_next_episode_by_index；
    # 旧版没有该方法，只能重排 episodes 并重建内部 iterator。
    iterator = env.episode_iterator
    if hasattr(iterator, "set_next_episode_by_index"):
        iterator.set_next_episode_by_index(index)
        return

    episodes = list(iterator.episodes)
    if not episodes:
        raise IndexError("No episodes are available in the Habitat iterator.")
    index = index % len(episodes)
    iterator.episodes = episodes[index:] + episodes[:index]
    if hasattr(iterator, "_iterator"):
        iterator._iterator = iter(iterator.episodes)
    if hasattr(iterator, "_prev_scene_id"):
        iterator._prev_scene_id = None


def _filter_episodes(env, scene_id_contains=None, object_category=None):
    if not scene_id_contains and not object_category:
        return

    iterator = env.episode_iterator
    episodes = list(iterator.episodes)
    filtered = []
    for episode in episodes:
        scene_id = str(getattr(episode, "scene_id", ""))
        category = str(getattr(episode, "object_category", ""))
        if scene_id_contains and scene_id_contains not in scene_id:
            continue
        if object_category and category.lower() != object_category.lower():
            continue
        filtered.append(episode)

    if not filtered:
        raise ValueError(
            f"No episodes matched scene_id_contains={scene_id_contains!r}, "
            f"object_category={object_category!r}."
        )

    # 直接替换 iterator 的 episode 列表，比依赖不同 Habitat 版本的 index 语义更稳定。
    iterator.episodes = filtered
    if hasattr(iterator, "_iterator"):
        iterator._iterator = iter(iterator.episodes)
    if hasattr(iterator, "_prev_scene_id"):
        iterator._prev_scene_id = None
    logger.info(
        "Filtered episodes: {} matched scene_id_contains={!r}, object_category={!r}",
        len(filtered),
        scene_id_contains,
        object_category,
    )


def _candidate_content_files(data_root, stage, scene_id):
    scene_file = f"{scene_id}.json.gz"
    explicit = os.getenv("STRIVE_SCENE_OBJECT_DATASET")
    if explicit:
        yield Path(explicit)

    preferred = [
        f"datasets/objectnav/hm3d_ovon/v1/val_seen_complex_balanced_2k/content/{scene_file}",
        f"datasets/objectnav/hm3d_ovon/v1/val_seen_complex_instruction/content/{scene_file}",
        f"datasets/objectnav/hm3d_ovon/v1/val_seen_instruction_balanced_3k/content/{scene_file}",
        f"datasets/objectnav/hm3d_ovon/v1/val_seen_instruction_2k/content/{scene_file}",
        f"datasets/objectnav/hm3d/v1/{stage}_instruction/content/{scene_file}",
        f"objectgoal_hm3d_ovon/val_seen/content/{scene_file}",
        f"objectgoal_hm3d/{stage}/content/{scene_file}",
        f"objectgoal_hm3d_custom/{stage}/content/{scene_file}",
        f"objectgoal_hm3d/{stage}_mini/content/{scene_file}",
    ]
    for rel in preferred:
        yield Path(data_root) / rel


def _episode_category(episode):
    return str(episode.get("object_category", episode.get("object_category_name", ""))).lower()


def _prepare_scene_object_dataset(args, stage="val"):
    if not args.scene_id or not args.object_category:
        return

    data_root = os.getenv("HM3D_DATA_PATH")
    if not data_root:
        raise ValueError("HM3D_DATA_PATH is not set; cannot locate scene content files.")

    matched_source = None
    matched_episodes = None
    target_category = args.object_category.lower()
    for path in _candidate_content_files(data_root, stage, args.scene_id):
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            dataset = json.load(f)
        episodes = [
            episode for episode in dataset.get("episodes", [])
            if _episode_category(episode) == target_category
        ]
        if episodes:
            matched_source = path
            matched_episodes = episodes
            break

    if not matched_episodes:
        searched = [str(path) for path in _candidate_content_files(data_root, stage, args.scene_id)]
        raise ValueError(
            f"No episode matched scene_id={args.scene_id!r}, object_category={args.object_category!r}. "
            f"Searched: {searched}"
        )

    if args.episode_rank < 0 or args.episode_rank >= len(matched_episodes):
        raise IndexError(
            f"episode_rank={args.episode_rank} is out of range; "
            f"{len(matched_episodes)} matched episodes are available."
        )

    with gzip.open(matched_source, "rt", encoding="utf-8") as f:
        filtered_dataset = json.load(f)
    filtered_dataset["episodes"] = [matched_episodes[args.episode_rank]]

    output_dir = Path(args.filtered_dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{_safe_name(args.scene_id)}_{_safe_name(args.object_category)}_rank{args.episode_rank}.json.gz"
    )
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(filtered_dataset, f)

    os.environ["HM3D_DATASET_PATH"] = str(output_path.resolve())
    args.dataset_episodes = 1
    args.eval_episodes = 1
    args.start_episode = 0
    logger.info(
        "Prepared filtered dataset: source={}, output={}, matched_count={}, episode_id={}",
        matched_source,
        output_path,
        len(matched_episodes),
        filtered_dataset["episodes"][0].get("episode_id"),
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--dataset_episodes", type=int, default=None)
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument("--scene_id_contains", type=str, default=None)
    parser.add_argument("--object_category", type=str, default=None)
    parser.add_argument("--episode_rank", type=int, default=0)
    parser.add_argument("--filtered_dataset_dir", type=str, default="logs/datasets")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--mapper_resolution", type=float, default=0.05)
    parser.add_argument("--grid_resolution", type=float, default=0.1)
    parser.add_argument("--grid_size", type=int, default=500)
    parser.add_argument("--grid_height", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="default")
    parser.add_argument("--not_do_seg", default=True, action="store_false")
    parser.add_argument("--no_gpt_seg", default=True, action="store_false")
    parser.add_argument("--relocate", default=False, action="store_true")
    parser.add_argument("--no_gpt_relocate", default=False, action="store_true")
    parser.add_argument("--vlm", type=str, default="cognav")
    parser.add_argument("--custom_instruction", type=str, default="")
    parser.add_argument("--enable_instruction_adapter", default=False, action="store_true")
    parser.add_argument("--instruction_adapter_backend", type=str, default="llm")
    parser.add_argument("--instruction_adapter_strict_classes", default=False, action="store_true")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    args.save_dir = "logs/" + args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    # 通过 scene_id/object_category 指定任务时，先生成单 episode 数据集。
    # 这样 Habitat 只会看到一个 episode，避免不同版本 iterator 的采样顺序差异。
    _prepare_scene_object_dataset(args, stage="val")

    dataset_episodes = args.dataset_episodes or max(args.eval_episodes, args.start_episode + 1)
    habitat_config = hm3d_config(stage='val', episodes=dataset_episodes)
    habitat_env = habitat.Env(habitat_config)
    _filter_episodes(habitat_env, args.scene_id_contains, args.object_category)
    habitat_mapper = Instruct_Mapper(
        habitat_camera_intrinsic(habitat_config),
        pcd_resolution=args.mapper_resolution,
        grid_resolution=args.grid_resolution,
        voxel_dimension=[args.grid_size, args.grid_size, args.grid_height],
        save_dir=args.save_dir,
        categories=categories,
        no_gpt_seg=args.no_gpt_seg,
        env=habitat_env,
        vlm=args.vlm)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,
                                      habitat_mapper,
                                      save_dir=args.save_dir,
                                      do_seg=args.not_do_seg,
                                      relocate=args.relocate,
                                      gpt_relocate=not args.no_gpt_relocate,
                                      vlm=args.vlm)
    evaluation_metrics = []
    instruction_parser = StriveInstructionParser(
        backend=args.instruction_adapter_backend,
        strict_available_classes=args.instruction_adapter_strict_classes,
        vlm=args.vlm,
    )

    start_idx = args.start_episode
    if hasattr(habitat_agent.env.episode_iterator, "set_next_episode_by_index"):
        # 新版 iterator 需要先 reset 一次完成内部状态初始化。
        habitat_agent.env.reset()

    for i in tqdm(range(start_idx, args.eval_episodes)):
        # try:
        # logger_path = f"./{args.save_dir}/episode-{i}/log.txt"
        # logger.add(logger_path, mode="w")
        logger.info(f"Processing episode: i = {i}")

        _set_next_episode_by_index(habitat_agent.env, i)
        habitat_agent.reset(i)

        target = habitat_agent.instruct_goal
        dataset_target = extract_dataset_target(target)
        if args.enable_instruction_adapter or str(args.custom_instruction or "").strip():
            current_episode = getattr(habitat_agent.env, "current_episode", None)
            custom_instruction = str(args.custom_instruction or "").strip()
            episode_info = {} if custom_instruction else (getattr(current_episode, "info", None) or {})
            raw_instruction = (
                custom_instruction
                or str(episode_info.get("instruction", "")).strip()
                or target
            )
            instruction_plan = instruction_parser.parse_plan(
                raw_instruction=raw_instruction,
                dataset_target=dataset_target,
                available_classes=habitat_mapper.object_perceiver.classes,
                episode_info=episode_info,
            )
            instruction_spec = instruction_plan.to_legacy_spec()
            habitat_mapper.instruction_plan = instruction_plan
            habitat_mapper.instruction_spec = instruction_spec
            habitat_mapper.instruction_execution_state = None
            habitat_mapper.instruction_constraint_evaluator.ensure_state(habitat_mapper, instruction_plan)
            habitat_mapper.target_list = list(
                instruction_spec.target_detector_prompts
                or [instruction_spec.canonical_target or dataset_target]
            )
            habitat_mapper.target = instruction_spec.canonical_target or habitat_mapper.target_list[0]
            habitat_mapper.target_aliases = list(instruction_spec.target_match_terms)
            habitat_agent.instruct_goal = render_instruction_context(instruction_plan)

            instruction_dir = os.path.join(args.save_dir, f"episode-{i}", "instruction_adapter")
            os.makedirs(instruction_dir, exist_ok=True)
            with open(os.path.join(instruction_dir, "plan.json"), "w", encoding="utf-8") as f:
                json.dump(instruction_plan.as_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
            with open(os.path.join(instruction_dir, "spec.json"), "w", encoding="utf-8") as f:
                json.dump(instruction_spec.as_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            habitat_mapper.instruction_plan = None
            habitat_mapper.instruction_spec = None
            habitat_mapper.instruction_execution_state = None
            habitat_mapper.target = dataset_target

            # 先用 LLM 扩展目标同义/相关类别，再初始化 GroundingDINO+SAM 的文本提示。
            habitat_mapper.target_list = ask_gpt_similar_objects(
                habitat_mapper.object_perceiver.classes,
                habitat_mapper.target,
                args.vlm
            )
            habitat_mapper.target_aliases = list(habitat_mapper.target_list)

        logger.info(f"Target: {habitat_mapper.target}")
        logger.info(f"Target list: {habitat_mapper.target_list}")
        if habitat_mapper.instruction_spec is not None:
            logger.info(f"Instruction spec: {habitat_mapper.instruction_spec.as_dict()}")

        habitat_mapper.object_perceiver.sam.initialize(habitat_mapper.target)

        if args.relocate:
            habitat_agent.make_plan_mod_relocate(idx=i)
        else:
            habitat_agent.make_plan_mod_no_relocate(idx=i)

        flag = True
        while flag and not habitat_env.episode_over and habitat_agent.episode_steps < args.max_steps:
            flag = habitat_agent.step_mod(idx=i)

        habitat_agent.save_trajectory(f"./{args.save_dir}/episode-{i}/")
        evaluation_metrics.append({
            'Episode': i,
            'success': habitat_agent.metrics['success'],
            'spl': habitat_agent.metrics['spl'],
            'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
            'Episode Steps': habitat_agent.episode_steps,
            'start and goal distance': habitat_agent.start_end_episode_distance,
            'travel distance': habitat_agent.travel_distance,
            'Found Goal': habitat_agent.found_goal,
            'End': 1,
            'object_goal': habitat_agent.instruct_goal,
        })
        write_metrics(evaluation_metrics, path=f"./{args.save_dir}/metrics.csv")
        logger.info('\n')

        # except Exception as e:
        #     logger.exception(f"Error occurred in episode {i}: {e} \n")

        #     evaluation_metrics.append({
        #         'Episode': i,
        #         'success': habitat_agent.metrics['success'],
        #         'spl': habitat_agent.metrics['spl'],
        #         'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
        #         'Episode Steps': habitat_agent.episode_steps,
        #         'start and goal distance': habitat_agent.start_end_episode_distance,
        #         'travel distance': habitat_agent.travel_distance,
        #         'Found Goal': habitat_agent.found_goal,
        #         'End': 0,
        #         'object_goal': habitat_agent.instruct_goal,
        #     })
        #     write_metrics(evaluation_metrics, path=f"./{args.save_dir}/metrics.csv")
        #     logger.info('\n')
