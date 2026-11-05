"""GOAT description subtasks evaluated independently from their parent start pose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import BenchmarkSpec
from .hm3d_objectnav import HM3DInstructionProvider


def language_tasks(data, *, split, source_file):
    """Preserve description, target and parent/subtask IDs without composite history.

    Only description tasks enter this project subset. Starting every task from
    the parent start pose defines a derived benchmark, not GOAT's sequential run.
    """
    goals = {g['object_id']: g for group in data['goals'].values() for g in group}
    result = []
    for parent in data['episodes']:
        scene_path = Path(parent['scene_id'])
        scene_name = scene_path.parent.name
        for index, task in enumerate(parent['tasks']):
            if task[1] != 'description':
                continue
            goal = goals[task[2]]
            instruction = goal['lang_desc']
            if len(instruction.split(' ')) > 55:  # Same description-length filter as the official loader.
                continue
            result.append(dict(
                episode_id=f"{split}:{scene_name}:{parent['episode_id']}:{index}",
                scene_id=f"hm3d_v0.2/val/{scene_name}/{scene_path.name}",
                start_position=parent['start_position'], start_rotation=parent['start_rotation'],
                object_category=task[0], goals=[{key: goal[key] for key in
                    ('object_id', 'object_category', 'position', 'view_points')}],
                info=dict(instruction=instruction, source_file=source_file,
                    source_episode_id=str(parent['episode_id']), source_subtask_index=index,
                    protocol='goat_language_single_parent_start_v1')))
    return result


def freeze_subset(episodes, count, seed):
    """Deduplicate identical navigation tasks, then select a fixed hash-ranked sample.

    Identity includes scene, start pose, target instance and instruction; repeated
    mentions inside one composite episode never increase the test denominator.
    """
    unique = {}
    for ep in episodes:
        key = json.dumps([ep['scene_id'], ep['start_position'], ep['start_rotation'],
                          ep['goals'][0]['object_id'], ep['info']['instruction']], sort_keys=True)
        unique.setdefault(key, ep)
    ranked = sorted(unique.items(), key=lambda item: hashlib.sha256(f'{seed}:{item[0]}'.encode()).hexdigest())
    if len(ranked) < count:
        raise ValueError(f'Requested {count} unique tasks, available {len(ranked)}')
    return [ep for _, ep in ranked[:count]], len(unique)


class GOATLanguageProvider(HM3DInstructionProvider):
    """Use project navigation with GOAT's STOP and 0.25m viewpoint success criterion."""
    name = 'goat_language'

    def prepare(self, args):
        spec = super().prepare(args)
        if args.success_distance is not None and args.success_distance != 0.25:
            raise ValueError('goat_language uses the fixed 0.25m viewpoint criterion')
        data = spec.as_dict()
        data.update(benchmark=self.name, success_distance=0.25)
        data['provenance'].update(protocol='goat_language_single_parent_start_v1',
            task_file_sha256=hashlib.sha256(Path(spec.dataset_path).read_bytes()).hexdigest(),
            success_rule='STOP and geodesic_to_viewpoint_lt_0.25')
        return BenchmarkSpec(**data)

    def make_config(self, spec, episodes):
        from . import instance_goal_dataset
        from config_utils import hm3d_config, _write_context
        cfg = hm3d_config(stage=spec.split, episodes=episodes,
                          dataset_path=spec.dataset_path, success_distance=0.25)
        with _write_context(cfg):
            cfg.DATASET.TYPE = 'InstanceGoalNav-v1'
            cfg.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = 'VIEW_POINTS'
        return cfg
