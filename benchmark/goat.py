"""Original GOAT continuous episodes and offline metrics; no policy or Habitat imports.

Pinned to GOAT-Bench 74c41d19d4a4c3608d1575b512087b5a529aee0e. This is NOT
``goat_language`` (independent, deduplicated tasks) and is not a runnable provider.
Only GOATPublicGoal belongs on the policy boundary. Raw episodes, target views,
image camera poses and evaluator distances remain driver/evaluator-only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

UPSTREAM_REVISION = '74c41d19d4a4c3608d1575b512087b5a529aee0e'
UPSTREAM_SHA256 = {
    'nav.py': '22cbd28d95a7fd47eb04cf327db58c2dbfd4852ac5b1a50f2c986f880557bf3f',
    'goat_task.py': 'ca2d0b17caa47e52bae133285fbeed5bed4e68080d4f3cff91473981fe01404e',
    'goat_dataset.py': '81f2d639dcea3f1a97fcae9a693adfb6358dd5962f079274b8a6e871b014e02d',
    'goat_stretch_hm3d.yaml': '35a2fe5e19fa9cd37c48e69593c6e4906d9dee8f06de363e4750b84feb9b1136',
}
MODALITIES = ('object', 'image', 'description')


@dataclass(frozen=True)
class GOATPublicGoal:
    """Policy-visible goal, with no target identity or geometry.

    Args:
        modality: Official object, description or image task type.
        category: Category for object tasks only.
        description: Unmodified language instruction for description tasks only.
        image_ref: Opaque asset reference for image tasks, NOT an existing file.
            The driver must resolve it to real reference pixels before inference.
    """

    modality: str
    category: Optional[str] = None
    description: Optional[str] = None
    image_ref: Optional[str] = None


@dataclass(frozen=True)
class GOATSubtask:
    """Ordered driver-private subtask; do not serialize this into policy prompts.

    Args:
        source_subtask_index: Index before description filtering.
        source_task: Original tuple, including object_id and image index.
        public_goal: The ONLY policy-facing part of this record.
        evaluator_goals: Full original goal dictionaries, including child-category
            goals for object tasks, or matching instance goals for other tasks.
            Treat these shared dataset dictionaries as read-only.
    """

    source_subtask_index: int
    source_task: tuple
    public_goal: GOATPublicGoal
    evaluator_goals: tuple[dict[str, Any], ...] = field(repr=False)

    @property
    def object_id(self):
        """Return the original instance ID (None for category tasks)."""
        return self.source_task[2]

    @property
    def image_index(self) -> Optional[int]:
        """Return the original zero-based image_goals index, or None."""
        return self.source_task[3] if self.public_goal.modality == 'image' else None

    @property
    def image_parameters(self) -> Optional[dict[str, Any]]:
        """Return driver-only selected camera parameters, never policy text.

        Preserves position, rotation, hfov, image_dimensions and coverage fields
        exactly. All alternative cameras remain in evaluator_goals.image_goals.
        """
        if self.image_index is None:
            return None
        return self.evaluator_goals[0]['image_goals'][self.image_index]

    @property
    def view_positions(self) -> tuple:
        """Return private geodesic endpoints; no object-center fallback is valid."""
        return tuple(tuple(view['agent_state']['position'])
                     for goal in self.evaluator_goals for view in goal['view_points'])


@dataclass(frozen=True)
class GOATEpisode:
    """One continuous parent episode, not a collection of independent resets.

    Args:
        split, source_file, source_episode_index: Source shard identity and row.
        source_episode: Entire unmodified parent JSON, including original ID,
            start pose, tasks and scene_dataset_config. Treat as read-only.
        subtasks: Filtered execution sequence; repetitions are significant.
        filtered_subtask_indices: Only descriptions exceeding the official limit.
    """

    split: str
    source_file: str
    source_episode_index: int
    source_episode: dict[str, Any] = field(repr=False)
    subtasks: tuple[GOATSubtask, ...]
    filtered_subtask_indices: tuple[int, ...] = ()

    @property
    def source_episode_id(self):
        """Return the original ID without upstream's per-shard renumbering."""
        return self.source_episode['episode_id']

    @property
    def scene_id(self) -> str:
        """Return the original scene path; asset remapping belongs to the driver."""
        return self.source_episode['scene_id']

    @property
    def start_position(self) -> tuple:
        """Return the parent start position, used only once per continuous run."""
        return tuple(self.source_episode['start_position'])

    @property
    def start_rotation(self) -> tuple:
        """Return the parent start quaternion in original coefficient order."""
        return tuple(self.source_episode['start_rotation'])


def parse_goat_episodes(data, *, split: str, source_file: str) -> list[GOATEpisode]:
    """Parse official scene JSON without mutating inputs or deduplicating tasks.

    Args:
        data: GOAT JSON with episodes and scene/category-keyed goals.
        split: Source split label, retained without rewriting scene paths.
        source_file: Source shard identity (use an absolute path for real data).
    Returns:
        Parents in source order, including parents with no remaining subtasks.
    Raises:
        ValueError: A task has an unsupported modality.

    Description filtering uses len(text.split(' ')) <= 55, before child expansion.
    Unlike upstream's loader bug, tasks AND goals are filtered together. Child
    groups are expanded once per subtask without its in-place list accumulation;
    the official endpoint set is unchanged. Missing child groups are ignored as
    upstream does. Legacy single-goal/dedup_goals input is not this protocol.
    """
    result = []
    for row, parent in enumerate(data['episodes']):
        scene = Path(parent['scene_id']).name
        subtasks, filtered = [], []
        for index, task in enumerate(parent['tasks']):
            category, modality, object_id = task[:3]
            if modality not in MODALITIES:
                raise ValueError(f'Unsupported GOAT modality: {modality}')
            base = data['goals'][f'{scene}_{category}']
            if modality == 'description':
                instance = next(g for g in base if g['object_id'] == object_id)
                if len(instance['lang_desc'].split(' ')) > 55:
                    filtered.append(index)
                    continue
            goals = list(base)
            for child in base[0]['children_object_categories']:
                goals.extend(data['goals'].get(f'{scene}_{child}', ()))
            if modality != 'object':
                goals = [g for g in goals if g['object_id'] == object_id]

            if modality == 'object':
                public = GOATPublicGoal(modality, category=category)
            elif modality == 'description':
                public = GOATPublicGoal(modality, description=goals[0]['lang_desc'])
            else:
                # Reference identity does not disclose instance names/camera poses.
                identity = json.dumps([parent['scene_id'], object_id, task[3]])
                reference = 'goat-image:' + hashlib.sha256(identity.encode()).hexdigest()
                public = GOATPublicGoal(modality, image_ref=reference)
            subtasks.append(GOATSubtask(index, tuple(task), public, tuple(goals)))
        result.append(GOATEpisode(split, str(source_file), row, parent,
                                  tuple(subtasks), tuple(filtered)))
    return result


def load_goat_episodes(source_file, *, split: str) -> list[GOATEpisode]:
    """Read one explicit official JSON/JSON.GZ shard, not a derived task subset.

    Args:
        source_file: Scene shard path; split index files usually have no episodes.
        split: Source split label.
    Returns:
        Parsed parents. Enumerate content shards explicitly to load a whole split.
    """
    path = Path(source_file).resolve()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return parse_goat_episodes(json.load(stream), split=split, source_file=str(path))


class GOATEvaluator:
    r"""Deterministic event-level transcription of pinned GoatSuccess/GoatSPL.

    Args:
        episode: Parsed continuous parent; no start-pose reset between subtasks.
        initial_distance: Actual initial geodesic distance to the first subtask's
            union of view_positions on the official Stretch navmesh, in meters.
    Raises:
        ValueError: No executable subtasks (upstream metrics are undefined).

    S_i = 1[subtask_stop_i and d_previous < 0.25].
    SPL_i = S_i * L_i / max(L_i, P_i), with L_i measured at that segment's
    ACTUAL start, and P_i the sum of per-action 3D Euclidean displacements.
    Unstopped/unvisited segments remain zero in the full episode denominator.
    Upstream's undefined SPL cases are not repaired: a Python-float zero-start,
    zero-path segment raises ZeroDivisionError, and inf/inf produces NaN. No
    artificial perfect SPL or finite-distance fallback is introduced.
    """

    def __init__(self, episode: GOATEpisode, initial_distance: float):
        if not episode.subtasks:
            raise ValueError('GOAT scoring requires at least one retained subtask')
        self.episode = episode
        self.active_subtask_index = 0
        self.ended = False
        self._distance = initial_distance
        self._start_distance = initial_distance
        self._path_length = 0.0
        self._success = [0.0] * len(episode.subtasks)
        self._spl = [0.0] * len(episode.subtasks)
        self._task_success = False

    def step(self, *, distance_to_goal: Optional[float], path_delta: float,
             subtask_stop: bool = False, stop_called: bool = False,
             episode_ended: bool = False) -> dict:
        """Consume one actual driver action, in upstream measurement order.

        Args:
            distance_to_goal: Post-action geodesic distance to the active goal.
                On subtask_stop, supply the NEXT goal's distance at the SAME pose;
                success uses the previous step's stored OLD-goal distance. Can be
                None only on the final subtask_stop. The stop action must not move.
            path_delta: Actual ||position_after - position_before||_2 in meters,
                including every low-level action, NOT a planned path or geodesic.
            subtask_stop: Whether THIS action was SubtaskStopAction, not the sticky
                is_sub_task_stop_called flag. Advances even when success is zero.
            stop_called: Actual global task.is_stop_called; does not advance tasks.
            episode_ended: Driver termination/timeout, not a substitute for STOP.
        Returns:
            Metric snapshot. Terminal snapshots are frozen; a trailing global
            STOP cannot upgrade task_success after the final subtask_stop.
        Raises:
            ValueError: Missing distance for a non-final-subtask-stop event.
        """
        if self.ended:
            return self.metrics()
        index = self.active_subtask_index
        final_switch = subtask_stop and index + 1 == len(self.episode.subtasks)
        if distance_to_goal is None and not final_switch:
            raise ValueError('A post-action active-goal distance is required')
        self._path_length += path_delta
        if subtask_stop:
            self._success[index] = float(self._distance < 0.25)
            denominator = max(self._start_distance, self._path_length)
            ratio = self._start_distance / denominator
            self._spl[index] = self._success[index] * ratio
            self.active_subtask_index += 1
            # Includes failed stops: the next segment begins here, not at GT.
            self._start_distance = distance_to_goal
            self._path_length = 0.0
        self._task_success = all(self._success) and stop_called
        if not final_switch:
            self._distance = distance_to_goal
        self.ended = final_switch or stop_called or episode_ended
        return self.metrics()

    def metrics(self) -> dict:
        """Return upstream success/SPL fields plus explicitly derived modal SPL.

        composite_success means every segment succeeded; task_success additionally
        needs global STOP in that SAME update. Ordinary upstream completion ends
        automatically on the last subtask_stop, so task_success stays False.
        object/image/description_success include absent modalities as zero.
        Upstream does not expose modal SPL (its expansion is commented out);
        the three *_spl fields here are means of its spl_by_subtask by modality.
        """
        count = len(self.episode.subtasks)
        result = dict(task_success=bool(self._task_success),
                      composite_success=all(self._success),
                      partial_success=sum(self._success) / count,
                      subtask_success=list(self._success),
                      composite_spl=sum(self._spl) / count,
                      spl_by_subtask=list(self._spl))
        for modality in MODALITIES:
            indices = [i for i, task in enumerate(self.episode.subtasks)
                       if task.public_goal.modality == modality]
            result[f'{modality}_success'] = (
                sum(self._success[i] for i in indices) / len(indices) if indices else 0.0)
            result[f'{modality}_spl'] = (
                sum(self._spl[i] for i in indices) / len(indices) if indices else 0.0)
        return result
