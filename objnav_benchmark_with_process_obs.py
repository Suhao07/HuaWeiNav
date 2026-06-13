import argparse
import csv
import json
import os
import shutil
import time

import habitat
from loguru import logger
from tqdm import tqdm

from benchmark import available_benchmarks, get_provider
from constants import *
#指令解析模块
from instruction_adapter import (
    StriveInstructionParser,
    extract_dataset_target,
    render_instruction_context,
)
from llm_utils.lvlm_call_tracker import counts_compact, reset_counts, set_trace_dir
from mapper_with_process_obs import Instruct_Mapper
from mapping_utils.transform import habitat_camera_intrinsic
from objnav_agent_with_process_obs import HM3D_Objnav_Agent

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="auto",
        choices=["auto"] + available_benchmarks(),
        help="Benchmark provider. Use auto for backwards-compatible HM3D/OVON smoke tests.",
    )
    parser.add_argument(
        "--benchmark_split",
        type=str,
        default=None,
        help="Explicit provider split, e.g. val_seen_instruction_balanced_3k for HM3D-OVON.",
    )
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--success_distance", type=float, default=None)
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
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--clean_save_dir", default=False, action="store_true")
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


def _install_instruction_plan(
    *,
    habitat_mapper,
    habitat_agent,
    instruction_parser,
  raw_instruction,
    dataset_target,
    episode_info,
    episode_dir,
    source,
):
    """Compile and install the canonical object/instruction plan for one episode.

    Benchmark ObjectNav and free-form instruction navigation intentionally share
    this runtime contract. The only difference is the prompt source: benchmark
    mode uses a narrow object prompt, while custom instruction mode may use the
    user instruction or dataset metadata.
    """

    instruction_plan = instruction_parser.parse_plan(
        raw_instruction=raw_instruction,
        dataset_target=dataset_target,
        available_classes=habitat_mapper.object_perceiver.classes,
        episode_info=episode_info,
    )
    instruction_plan.diagnostics.setdefault("source", [])
    instruction_plan.diagnostics["runtime_source"] = source
    instruction_spec = instruction_plan.to_legacy_spec()

    habitat_mapper.instruction_plan = instruction_plan
    habitat_mapper.instruction_spec = instruction_spec
    habitat_mapper.instruction_execution_state = None
    habitat_mapper.instruction_constraint_evaluator.ensure_state(habitat_mapper, instruction_plan)

    # Perception can use both terminal detector terms and non-terminal concept
    # detector terms. Stop remains controlled by terminal target + verifier.
    habitat_mapper.target_list = list(
        instruction_spec.target_detector_prompts
        or [instruction_spec.canonical_target or dataset_target]
    )
    perception_terms = list(habitat_mapper.target_list)
    for concept in getattr(instruction_plan, "concept_queries", []) or []:
        perception_terms.extend(list(getattr(concept, "detector_terms", []) or []))
    habitat_mapper.perception_target_list = list(dict.fromkeys([term for term in perception_terms if term]))
    habitat_mapper.target = instruction_spec.canonical_target or habitat_mapper.target_list[0]
    habitat_mapper.target_aliases = list(instruction_spec.target_match_terms)
    habitat_agent.instruct_goal = render_instruction_context(instruction_plan)

    instruction_dir = os.path.join(episode_dir, "instruction_adapter")
    os.makedirs(instruction_dir, exist_ok=True)
    with open(os.path.join(instruction_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(instruction_plan.as_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(instruction_dir, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(instruction_spec.as_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
    return instruction_plan, instruction_spec


if __name__ == "__main__":
    args = get_args()
    args.save_dir = "logs/" + args.save_dir
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    if args.clean_save_dir and os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "clean_save_dir": bool(args.clean_save_dir),
                "run_id": run_id,
                "save_dir": args.save_dir,
                "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    benchmark_provider = get_provider(args)
    benchmark_spec = benchmark_provider.prepare(args)
    with open(os.path.join(args.save_dir, "benchmark_spec.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_spec.as_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Benchmark spec: {}", benchmark_spec.as_dict())

    dataset_episodes = args.dataset_episodes or max(args.eval_episodes, args.start_episode + 1)
    habitat_config = benchmark_provider.make_config(benchmark_spec, episodes=dataset_episodes)
    habitat_env = habitat.Env(habitat_config)
    if not benchmark_spec.is_filtered:
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
    habitat_agent.run_id = run_id
    # Provider-specific success distance must reach the agent as well as Habitat.
    # 否则 final-stop verifier 和靠近控制会沿用 HM3D 默认 1.0m，
    # 与 OVON 等 benchmark 的 1.5m 成功半径不一致。
    habitat_agent.success_distance = benchmark_spec.success_distance
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
        episode_dir = os.path.join(args.save_dir, f"episode-{i}")
        reset_counts()
        set_trace_dir(os.path.join(episode_dir, "lvlm_calls"))

        target = habitat_agent.instruct_goal
        dataset_target = extract_dataset_target(target)
        current_episode = getattr(habitat_agent.env, "current_episode", None)
        custom_instruction = str(args.custom_instruction or "").strip()
        if custom_instruction:
            raw_instruction = custom_instruction
            episode_info = {}
            plan_source = "custom_instruction"
        elif args.enable_instruction_adapter:
            episode_info = getattr(current_episode, "info", None) or {}
            raw_instruction = str(episode_info.get("instruction", "")).strip() or target
            plan_source = "episode_instruction"
        else:
            # benchmark 默认也走同一套目标指令链路。这里故意不读取
            # episode.info 的复杂自然语言，只生成 ObjectNav 的窄目标 prompt；
            # 这样 check_again、ledger、view-control 都与指令模式复用，
            # 但 benchmark 语义仍是“找到这个类别”。
            raw_instruction = target
            episode_info = {}
            plan_source = "benchmark_object_goal"

        instruction_plan, instruction_spec = _install_instruction_plan(
            habitat_mapper=habitat_mapper,
            habitat_agent=habitat_agent,
            instruction_parser=instruction_parser,
            raw_instruction=raw_instruction,
            dataset_target=dataset_target,
            episode_info=episode_info,
            episode_dir=episode_dir,
            source=plan_source,
        )

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
            # success/spl 是 Habitat 原始目标类别指标；instruction_success
            # 是自然语言指令 verifier 的终止结果，两者不能混读。
            'Episode': i,
            'run_id': run_id,
            'benchmark': benchmark_spec.benchmark,
            'benchmark_split': benchmark_spec.split,
            'benchmark_dataset_path': benchmark_spec.dataset_path,
            'benchmark_source_file': benchmark_spec.source_file,
            'benchmark_source_episode_id': benchmark_spec.source_episode_id,
            'benchmark_success_distance': benchmark_spec.success_distance,
            'success': habitat_agent.metrics['success'],
            'spl': habitat_agent.metrics['spl'],
            'distance_to_goal': habitat_agent.metrics['distance_to_goal'],
            'Episode Steps': habitat_agent.episode_steps,
            'start and goal distance': habitat_agent.start_end_episode_distance,
            'travel distance': habitat_agent.travel_distance,
            'Found Goal': habitat_agent.found_goal,
            'instruction_success': habitat_agent.instruction_success,
            'instruction_decision': habitat_agent.instruction_decision,
            'instruction_accept_step': habitat_agent.instruction_accept_step,
            'final_stop_success': habitat_agent.final_stop_success,
            'final_stop_decision': habitat_agent.final_stop_decision,
            'final_stop_accept_step': habitat_agent.final_stop_accept_step,
            'final_stop_mode': habitat_agent.final_stop_mode,
            'accepted_candidate_uid': habitat_agent.accepted_candidate_uid,
            'accepted_relation_edge': json.dumps(habitat_agent.accepted_relation_edge, ensure_ascii=False, sort_keys=True),
            'accepted_distance_to_target': habitat_agent.accepted_distance_to_target,
            'accepted_distance_source': habitat_agent.accepted_distance_source,
            'lvlm_call_count_by_type': counts_compact(),
            # 兼容前期文档/脚本中的拼写；规范字段是 lvlm_call_count_by_type。
            'lvml_call_count_by_type': counts_compact(),
            'End': 1,
            'object_goal': (
                getattr(getattr(habitat_mapper, "instruction_plan", None), "raw_instruction", "")
                or habitat_agent.instruct_goal
            ),
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
