"""LangMap single-goal inputs and official terminal metrics, independent of policy."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from .contracts import BenchmarkSpec
from .hm3d_objectnav import HM3DObjectNavProvider

EXCLUDED_IDS = frozenset(json.loads(Path(__file__).with_name('langmap_exclusions.json').read_text()))


def convert_scene(data, scene_name, style='concise'):
    """Convert public instructions and retain each task's exact evaluator goal set.

    Ground-truth object IDs/positions stay in goals; only instruction text enters
    the language compiler. Both description styles share geometry and task IDs.
    """
    objects = {g['object_id']: g for g in data['goals']}
    episodes = []
    skipped = []
    for level in ('instance', 'region', 'room', 'object'):
        for task in data[f'episodes_by_{level}_level']:
            uid = f"{scene_name}_{level}_{task['episode_id']}"
            if uid in EXCLUDED_IDS:
                skipped.append({'id': uid, 'reason': 'official_blacklist'})
                continue
            targets = [objects[k] for k in task['target_object_ids']]
            category = targets[0]['object_category'] if level == 'instance' else task['object_category']
            if level == 'instance':
                text = f"Find the {objects[task['instance_id']]['annot_unique_' + style + '_description']}"
            elif level == 'region':
                region = data['region_annotation'][task['region_id']]
                text = f"Find the {category} in the {region['region_category'].lower()} that has {region[style + '_description']}"
            elif level == 'room':
                text = f"Find the {category} in the {task['room_name']}"
            else:
                text = f"Find the {category}."
            if level in ('room', 'region') and 'tie:' in text.lower():
                text += ", 'Tie: A & B' means the area spans both rooms A and B"
            pos, rot = task['start_position'], task['start_rotation']
            if not targets or not all(math.isfinite(v) for v in pos + rot) or abs(sum(v*v for v in rot)**0.5 - 1) > 0.1:
                skipped.append({'id': uid, 'reason': 'invalid_start_or_empty_goals'})
                continue
            goals = []
            for obj in targets:
                # 官方无视点时使用对象位置；保留这条评测语义，不向导航器泄漏。
                views = obj['view_points'] or [{'agent_state': {'position': obj['position'], 'rotation': [0, 0, 0, 1]}, 'iou': 0.0}]
                goals.append({k: obj[k] for k in ('object_id', 'object_category', 'position')})
                goals[-1]['view_points'] = views
            episodes.append(dict(episode_id=uid,
                scene_id=f"hm3d_v0.2/val/{scene_name}/{scene_name.split('-', 1)[1]}.basis.glb",
                start_position=pos, start_rotation=rot, object_category=category, goals=goals,
                info={'instruction': text, 'navigation_type': level, 'description_style': style}))
    return episodes, skipped


def terminal_metrics(initial_distance, final_distance, path_length):
    r"""Official final-state criterion: S=1[d_G(x_T,V)<0.25], SPL=S L*/max(L*,L).

    STOP is not required. Unreachable targets count as failures. L is the sum of
    consecutive simulator-position displacements, including every action.
    """
    success = float(final_distance < 0.25)
    denominator = max(initial_distance, path_length)
    spl = (initial_distance / denominator if denominator else 1.0) if success and math.isfinite(initial_distance) else 0.0
    return dict(success=success, spl=spl, distance_to_goal=final_distance,
                start_distance=initial_distance, path_length=path_length)


class LangMapProvider(HM3DObjectNavProvider):
    """Reuse HM3D execution with task-specific goals and a separate evaluator."""
    name = 'langmap'

    def prepare(self, args):
        """Materialize single-goal tasks from an official scene file or directory."""
        source = Path(args.dataset_path)
        files = sorted(source.glob('*.json.gz')) if source.is_dir() else [source]
        episodes, excluded, hashes = [], [], {}
        for file in files:
            scene = file.name.removesuffix('.json.gz')
            if args.scene_id and args.scene_id not in scene:
                continue
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            with gzip.open(file, 'rt') as stream:
                converted, skipped = convert_scene(json.load(stream), scene)
            episodes.extend(converted)
            excluded.extend(skipped)
        if args.object_category:
            episodes = [e for e in episodes if e['object_category'] == args.object_category]
        if not episodes:
            raise ValueError('No LangMap tasks matched the explicit input selection')
        categories = {c: i for i, c in enumerate(sorted({e['object_category'] for e in episodes}))}
        output = Path(args.save_dir) / 'langmap_tasks.json.gz'
        with gzip.open(output, 'wt') as stream:
            json.dump(dict(episodes=episodes, category_to_task_category_id=categories,
                           category_to_scene_annotation_category_id=categories), stream)
        return BenchmarkSpec(benchmark=self.name, split='val', dataset_path=str(output),
            source_file=str(source), success_distance=0.25, is_filtered=True,
            provenance={'input_sha256': hashes, 'description_style': 'concise',
                        'official_code_revision': '3f2cd6e37a7f06818f515b480b0e81293a8269af',
                        'tasks': len(episodes), 'excluded': excluded,
                        'navmesh': 'stretch', 'success_rule': 'final_geodesic_to_viewpoint_lt_0.25'})

    def make_config(self, spec, episodes):
        """Register LangMap dataset, simulator and metrics in the existing environment."""
        from . import langmap_habitat, instance_goal_dataset  # Register the selected backend.
        from config_utils import hm3d_config, _write_context
        from habitat.config import Config
        cfg = hm3d_config(stage=spec.split, episodes=episodes,
                          dataset_path=spec.dataset_path, success_distance=0.25)
        with _write_context(cfg):
            cfg.DATASET.TYPE = 'InstanceGoalNav-v1'
            cfg.SIMULATOR.TYPE = 'LangMapSim-v0'
            cfg.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = 'VIEW_POINTS'
            cfg.TASK.LANGMAP_EVALUATION = Config({'TYPE': 'LangMapEvaluation'})
            cfg.TASK.MEASUREMENTS = list(cfg.TASK.MEASUREMENTS) + ['LANGMAP_EVALUATION']
        Path(spec.dataset_path).with_name('habitat_config.yaml').write_text(cfg.dump())
        return cfg
