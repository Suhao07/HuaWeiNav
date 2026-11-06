"""Continuous GOAT protocol tests; the oracle executes pinned upstream class bodies."""
import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.goat import (
    GOATEvaluator, UPSTREAM_SHA256, load_goat_episodes, parse_goat_episodes,
)


def dataset():
    def goal(object_id, category, description):
        return dict(object_id=object_id, object_category=category,
                    position=[10, 1, 3], children_object_categories=[],
                    lang_desc=description,
                    view_points=[{'agent_state': {'position': [9, 0, 3],
                                                 'rotation': [0, 0, 0, 1]}, 'iou': 0.7}],
                    image_goals=[dict(position=[8, 1.3, 2], rotation=[0, 0, 0, 1],
                                      hfov=fov, image_dimensions=[512, 512],
                                      frame_coverage=0.3, object_coverage=0.8)
                                 for fov in (70, 90)])
    chair = goal('chair_1', 'chair', ' '.join(['word'] * 55))
    chair['children_object_categories'] = ['stool', 'missing_child']
    return dict(goals={
        'scene.basis.glb_chair': [chair, goal('chair_2', 'chair', ' '.join(['word'] * 56))],
        'scene.basis.glb_stool': [goal('stool_1', 'stool', 'a stool')],
        # An identically named instance in another scene must never be selected.
        'other.basis.glb_chair': [goal('chair_1', 'chair', 'other scene chair')],
    }, episodes=[dict(episode_id='original-17', scene_id='hm3d/val//scene/scene.basis.glb',
                      scene_dataset_config='hm3d.scene_dataset_config.json',
                      start_position=[1, 2, 3], start_rotation=[0, 0, 0, 1],
                      tasks=[['chair', 'object', None], ['chair', 'description', 'chair_1'],
                             ['chair', 'image', 'chair_2', 1], ['chair', 'object', None]])])


def parse(data):
    return parse_goat_episodes(data, split='val_unseen', source_file='/dataset/scene.json.gz')[0]


def test_original_sequence_duplicates_children_filter_and_private_images():
    data = dataset()
    tasks = data['episodes'][0]['tasks']
    tasks[1:1] = [tasks[0][:]]
    tasks[3:3] = [tasks[2][:], ['chair', 'description', 'chair_2']]
    tasks.insert(6, tasks[5][:])
    original = deepcopy(data)
    episode = parse(data)
    assert data == original  # No accumulated child-goal extensions across repeats.
    assert episode.source_episode == original['episodes'][0]
    assert episode.source_episode_id == 'original-17'
    assert episode.source_episode_index == 0
    assert episode.source_file == '/dataset/scene.json.gz'
    assert episode.scene_id == original['episodes'][0]['scene_id']
    assert episode.start_position == (1, 2, 3)
    assert episode.start_rotation == (0, 0, 0, 1)
    assert episode.filtered_subtask_indices == (4,)
    assert [s.source_subtask_index for s in episode.subtasks] == [0, 1, 2, 3, 5, 6, 7]
    assert [s.source_task for s in episode.subtasks] == [tuple(t) for i, t in enumerate(tasks) if i != 4]
    assert [s.public_goal.modality for s in episode.subtasks] == [
        'object', 'object', 'description', 'description', 'image', 'image', 'object']
    for i in (0, 1, 6):
        assert [g['object_id'] for g in episode.subtasks[i].evaluator_goals] == [
            'chair_1', 'chair_2', 'stool_1']
    description = episode.subtasks[2]
    assert description.public_goal.description == ' '.join(['word'] * 55)
    assert len(description.evaluator_goals) == 1
    image = episode.subtasks[4]
    assert image.object_id == 'chair_2'  # Long description did NOT filter this image.
    assert image.image_index == 1
    assert image.image_parameters == data['goals']['scene.basis.glb_chair'][1]['image_goals'][1]
    assert image.evaluator_goals[0]['image_goals'] == data['goals']['scene.basis.glb_chair'][1]['image_goals']
    assert image.view_positions == ((9, 0, 3),)
    assert image.public_goal == episode.subtasks[5].public_goal
    assert asdict(image.public_goal) == dict(modality='image', category=None,
                                            description=None, image_ref=image.public_goal.image_ref)
    assert 'chair_2' not in image.public_goal.image_ref
    for subtask in episode.subtasks:
        assert 'position' not in json.dumps(asdict(subtask.public_goal))
    # No deduplication of parents, no rewriting their IDs, no shared-list growth.
    data['episodes'].append(deepcopy(data['episodes'][0]))
    episodes = parse_goat_episodes(data, split='val_unseen', source_file='/dataset/scene.json.gz')
    assert [e.source_episode_index for e in episodes] == [0, 1]
    assert episodes[0].subtasks == episodes[1].subtasks


@pytest.mark.parametrize('suffix', ['.json', '.json.gz'])
def test_load_explicit_original_shard(tmp_path, suffix):
    path = tmp_path / ('scene' + suffix)
    opener = gzip.open if suffix.endswith('.gz') else open
    with opener(path, 'wt') as stream:
        json.dump(dataset(), stream)
    episode = load_goat_episodes(path, split='val_unseen')[0]
    assert episode.source_file == str(path.resolve())
    assert episode.source_episode == dataset()['episodes'][0]
    assert parse_goat_episodes({'episodes': []}, split='val', source_file='index') == []


def test_literal_space_description_filter_keeps_empty_parent_identity():
    data = dataset()
    data['goals']['scene.basis.glb_chair'][0]['lang_desc'] = 'word' + ' ' * 55
    data['episodes'][0]['tasks'] = [['chair', 'description', 'chair_1']]
    episode = parse(data)
    assert episode.filtered_subtask_indices == (0,)
    assert episode.subtasks == ()
    assert episode.source_episode_id == 'original-17'
    with pytest.raises(ValueError, match='retained subtask'):
        GOATEvaluator(episode, 1.0)


@pytest.fixture
def upstream():
    """Replace only Habitat plumbing, not upstream measurement/action algorithms.

    Source fingerprints prevent silent oracle drift. An ordinary STRIVE checkout
    without the external pinned sources skips parity tests explicitly; the pure
    parsing tests still run. No Habitat modules, simulator or model are started.
    """
    root = Path(__file__).resolve().parents[2] / 'GOAT-Bench-upstream'
    if not (root / 'nav.py').is_file():
        pytest.skip('Pinned GOAT-Bench-upstream sources required for parity oracle')
    for name, expected in UPSTREAM_SHA256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected

    class Measure:
        def __init__(self, **kwargs):
            self.uuid = self._get_uuid()
            self._metric = None

        def get_metric(self):
            return self._metric

    class SimulatorTaskAction:
        def __init__(self, sim):
            self._sim = sim

    scope = dict(np=np, defaultdict=defaultdict, Measure=Measure,
                 NavigationTask=object, SimulatorTaskAction=SimulatorTaskAction)
    for name, wanted in (
        ('goat_task.py', {'GoatTask', 'SubtaskStopAction'}),
        ('nav.py', {'GoatDistanceToGoal', 'GoatSuccess', 'GoatSPL'}),
    ):
        tree = ast.parse((root / name).read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
        assert {node.name for node in classes} == wanted
        for node in classes:
            node.decorator_list = []  # Only remove Habitat registration.
        module = ast.Module(body=ast.parse('from __future__ import annotations').body + classes,
                            type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(root / name), 'exec'), scope)

    class Oracle:
        def __init__(self, episode, initial, *, source_goals=None, geodesic_distance=None):
            self.position = np.zeros(3)
            self.distance = initial
            self.sim = SimpleNamespace(
                get_agent_state=lambda: SimpleNamespace(position=self.position.copy()),
                geodesic_distance=geodesic_distance or (lambda *args: self.distance),
                get_observations_at=lambda: {},
            )
            self.episode = SimpleNamespace(
                tasks=[s.source_task for s in episode.subtasks],
                goals=source_goals if source_goals is not None else [s.evaluator_goals for s in episode.subtasks],
                scene_id=episode.scene_id, episode_id=episode.source_episode_id,
            )
            self.task = scope['GoatTask']()
            self.task.is_stop_called = False
            self.stop_action = scope['SubtaskStopAction'](self.sim)
            self.stop_action.reset(self.task)
            config = SimpleNamespace(success_distance=0.25, distance_to='VIEW_POINTS')
            self.measures = {scope[name].cls_uuid: scope[name](self.sim, config)
                             for name in ('GoatDistanceToGoal', 'GoatSuccess', 'GoatSPL')}
            self.task.measurements = SimpleNamespace(measures=self.measures,
                                                     check_measure_dependencies=lambda *args: None)
            for measure in self.measures.values():
                measure.reset_metric(self.episode, self.task)

        def step(self, *, distance_to_goal, position, subtask_stop=False, stop_called=False):
            self.position = np.asarray(position, dtype=float)
            self.distance = distance_to_goal
            if subtask_stop:
                self.stop_action.step(self.task)  # Actual upstream unconditional advance.
            self.task.last_action = self.stop_action if subtask_stop else None
            self.task.is_stop_called = stop_called
            for measure in self.measures.values():
                measure.update_metric(self.episode, self.task)
            return self.metrics()

        def metrics(self):
            return deepcopy({**self.measures['success'].get_metric(),
                             **self.measures['spl'].get_metric()})

    return Oracle


def assert_parity(actual, expected):
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, nan_ok=True), name


def test_actual_upstream_failed_stop_switch_distance_and_path_reset(upstream):
    episode = parse(dataset())
    evaluator, oracle = GOATEvaluator(episode, 4.0), upstream(episode, 4.0)
    assert_parity(evaluator.metrics(), oracle.metrics())
    # (new-goal distance after action, x position, is_subtask_stop)
    events = [(0.25, 6, False), (0.1, 6, True), (6.0, 6, True),
              (0.1, 18, False), (4.0, 18, True), (0.249, 26, False), (None, 26, True)]
    previous = 0
    for distance, x, subtask_stop in events:
        actual = evaluator.step(distance_to_goal=distance, path_delta=abs(x - previous),
                                subtask_stop=subtask_stop)
        expected = oracle.step(distance_to_goal=distance, position=[x, 0, 0],
                               subtask_stop=subtask_stop)
        assert_parity(actual, expected)
        assert evaluator.active_subtask_index == oracle.task.active_subtask_idx
        previous = x
    assert actual['subtask_success'] == [0, 1, 1, 1]
    assert actual['spl_by_subtask'] == [0, 1, 0.5, 0.5]
    assert actual['partial_success'] == 0.75
    assert actual['composite_spl'] == 0.5
    assert actual['object_success'] == 0.5
    assert actual['object_spl'] == 0.25
    assert actual['image_spl'] == 0.5
    assert actual['description_spl'] == 1.0
    assert not actual['composite_success']
    assert evaluator.ended


@pytest.mark.parametrize('modality', ['object', 'description', 'image'])
def test_actual_upstream_wrong_instance_only_satisfies_category_goal(upstream, modality):
    from benchmark.goat_environment import GOATEnvironment

    data = dataset()
    target = data['goals']['scene.basis.glb_chair'][1]
    target['lang_desc'] = 'the second chair'
    target['view_points'][0]['agent_state']['position'] = [19., 0., 3.]
    task = ['chair', modality, None if modality == 'object' else 'chair_2']
    if modality == 'image':
        task.append(0)
    data['episodes'][0].update(tasks=[task], start_position=[0., 0., 0.])
    source_goals = deepcopy(data['goals']['scene.basis.glb_chair'] +
                            data['goals']['scene.basis.glb_stool'] if modality == 'object' else [target])
    expected_endpoints = [view['agent_state']['position'] for goal in source_goals
                          for view in goal['view_points']]
    episode = parse(data)
    wrong_instance_view = np.array([9., 0., 3.])
    state = SimpleNamespace(position=np.zeros(3), rotation=[0., 0., 0., 1.])
    queried = []

    def distance(position, endpoints, *args):
        return float(np.linalg.norm(np.asarray(endpoints) - position, axis=1).min())

    def environment_distance(position, endpoints):
        queried.append(endpoints)
        return distance(position, endpoints)

    def move(action):
        assert action == 1
        state.position = wrong_instance_view.copy()
        return {}

    base = SimpleNamespace(episode_over=False, reset=lambda: {}, step=move,
                           sim=SimpleNamespace(get_agent_state=lambda: state,
                                               geodesic_distance=environment_distance,
                                               get_observations_at=lambda: {}))
    events = []
    env = GOATEnvironment(base, episode, max_steps_per_subtask=3, global_budget=10, record=events.append)
    oracle = upstream(episode, None, source_goals=[source_goals], geodesic_distance=distance)
    env.reset()
    assert_parity(env.evaluator.metrics(), oracle.metrics())
    env.step(1)
    assert_parity(env.evaluator.metrics(),
                  oracle.step(distance_to_goal=None, position=wrong_instance_view))
    env.subtask_stop('wrong_instance_probe')
    actual = env.evaluator.metrics()
    expected = oracle.step(distance_to_goal=None, position=wrong_instance_view, subtask_stop=True)
    assert_parity(actual, expected)
    assert len(queried) == 2
    for endpoints in queried:
        np.testing.assert_array_equal(endpoints, expected_endpoints)
    assert actual['subtask_success'] == [float(modality == 'object')]
    assert env.evaluator.ended and env.index == oracle.task.active_subtask_idx == 1
    assert events[-1]['action'] == 'SUBTASK_STOP' and events[-1]['path_delta'] == 0.


@pytest.mark.parametrize('global_stop_at_final', [False, True])
def test_actual_upstream_composite_vs_task_success_and_final_freeze(upstream, global_stop_at_final):
    episode = parse(dataset())
    evaluator, oracle = GOATEvaluator(episode, 0.1), upstream(episode, 0.1)
    for i in range(len(episode.subtasks)):
        final = i == len(episode.subtasks) - 1
        distance = None if final else 0.1
        stop = final and global_stop_at_final
        actual = evaluator.step(distance_to_goal=distance, path_delta=0,
                                subtask_stop=True, stop_called=stop)
        expected = oracle.step(distance_to_goal=distance, position=[0, 0, 0],
                               subtask_stop=True, stop_called=stop)
        assert_parity(actual, expected)
    assert actual['composite_success'] is True
    assert actual['task_success'] is global_stop_at_final
    assert actual['composite_spl'] == 1.0
    assert_parity(evaluator.step(distance_to_goal=None, path_delta=0, stop_called=True),
                  oracle.step(distance_to_goal=None, position=[0, 0, 0], stop_called=True))
    assert evaluator.metrics() == actual


@pytest.mark.parametrize('global_stop', [False, True])
def test_actual_upstream_unstopped_and_unvisited_segments_remain_zero(upstream, global_stop):
    episode = parse(dataset())
    evaluator, oracle = GOATEvaluator(episode, 2.0), upstream(episode, 2.0)
    actual = evaluator.step(distance_to_goal=0.1, path_delta=3,
                            stop_called=global_stop, episode_ended=True)
    expected = oracle.step(distance_to_goal=0.1, position=[1, 2, 2], stop_called=global_stop)
    assert_parity(actual, expected)
    assert actual['subtask_success'] == [0, 0, 0, 0]
    assert evaluator.active_subtask_index == 0
    assert evaluator.ended


@pytest.mark.parametrize('initial', [0.0, float('inf')])
def test_actual_upstream_undefined_spl_is_not_replaced_with_perfect_score(upstream, initial):
    parent = parse(dataset())
    episode = replace(parent, subtasks=parent.subtasks[:1])
    evaluator, oracle = GOATEvaluator(episode, initial), upstream(episode, initial)
    if initial == 0:
        with pytest.raises(ZeroDivisionError):
            oracle.step(distance_to_goal=None, position=[0, 0, 0], subtask_stop=True)
        with pytest.raises(ZeroDivisionError):
            evaluator.step(distance_to_goal=None, path_delta=0, subtask_stop=True)
    else:
        assert_parity(evaluator.step(distance_to_goal=None, path_delta=0, subtask_stop=True),
                      oracle.step(distance_to_goal=None, position=[0, 0, 0], subtask_stop=True))


def continuous_agent(episode, events, *, segment_budget=4, global_budget=30):
    from unittest.mock import Mock
    from backends.habitat.motion import HabitatExecution
    from benchmark.goat_environment import GOATEnvironment
    from instruction_adapter.view_control import ViewControlState

    state = SimpleNamespace(position=np.array(episode.start_position, dtype=float),
                            rotation=list(episode.start_rotation))
    base = SimpleNamespace(episode_over=False, reset_count=0, actions=[],
                           current_episode=SimpleNamespace(object_category='object', goals=[], info={}))
    distances = iter([0.1, 4.0, 0.1, 0.1])
    current_distance = [2.0]
    queried = []
    def distance(position, endpoints):
        queried.append((list(position), endpoints))
        return current_distance[0]
    base.sim = SimpleNamespace(get_agent_state=lambda: state,
                               geodesic_distance=distance, get_observations_at=lambda: {})
    def reset():
        base.reset_count += 1
        return {}
    def step(action):
        base.actions.append(action)
        if action == 0:
            base.episode_over = True
        elif action == 1:
            state.position += [1., 2., 2.]
            current_distance[0] = next(distances)
        elif action == 2:
            state.rotation = [0., 1., 0., 0.]
        return {}
    base.reset, base.step, base.close = reset, step, Mock()
    env = GOATEnvironment(base, episode, max_steps_per_subtask=segment_budget,
                          global_budget=global_budget, record=events.append)
    agent = SimpleNamespace(env=env, mapper=SimpleNamespace(nodes=[object()], traj=[0], update_iterations=2,
        current_position=np.array([4., 5., 6.])), episode_steps=19, current_node_idx=7,
        travel_distance=8., position_trajectory=[object()], reset_debug_probes=Mock(),
        execution=HabitatExecution(env), view_control_state=ViewControlState(),
        final_instruction_verifier=SimpleNamespace(evidence_store=Mock()))
    agent.reset = lambda index: env.reset()
    return agent, base, queried


def test_continuous_driver_success_failed_budget_and_same_pose_switch(monkeypatch, tmp_path):
    import backends.habitat.goat_runtime as runtime
    episode, events, installs = parse(dataset()), [], []
    agent, base, queried = continuous_agent(episode, events)
    mapper, nodes, trajectory = agent.mapper, agent.mapper.nodes, agent.position_trajectory
    def install(agent, public_goal, **kwargs):
        installs.append((public_goal, kwargs['revision'], agent.env.pose()))
        assert agent.episode_steps == 19 and agent.current_node_idx == 7
        assert agent.travel_distance == 8. and agent.mapper is mapper
        assert agent.mapper.nodes is nodes and agent.position_trajectory is trajectory
        assert agent.object_final is None and not agent.found_goal
        agent.task = SimpleNamespace(spec=SimpleNamespace(revision=kwargs['revision']))
    def guided(agent, **kwargs):
        assert kwargs['stop_on_success'] is False
        revision = agent.task.spec.revision
        # Repeated guided IDs must not return stale execution receipts across tasks.
        agent.execution.step(1, request_id='guided-1')
        agent.object_final, agent.found_goal = object(), True
        if revision == 2:
            # An unguarded nested skill is cut off at the public env.step boundary.
            for _ in range(10):
                agent.execution.step(2)
        return {'reason': 'verified_finish', 'task_complete': True}
    monkeypatch.setattr(runtime, 'install_public_goal', install)
    monkeypatch.setattr(runtime, '_guided_run', guided)
    result = runtime.run_continuous_parent(agent, episode, parser=None, output=tmp_path, references={})
    assert result['status'] == 'complete', result
    assert result['parent_reset_count'] == base.reset_count == 1
    assert base.actions == [1, 1, 2, 2, 1, 1]
    for row in result['subtasks']:
        assert row['goal_switch'] == dict(mapper_preserved=True, nodes_preserved=True,
            map_updates_before=2, map_updates_after=2, node_count=1, pose_after_install=row['start_pose'])
    assert [i[1] for i in installs] == [1, 2, 3, 4]
    assert [i[0] for i in installs] == [s.public_goal for s in episode.subtasks]
    assert result['metrics']['subtask_success'] == [1., 0., 1., 1.]
    assert not result['metrics']['task_success']
    assert result['subtasks'][1]['reason'] == 'subtask_budget'
    for old, new in zip(result['subtasks'], result['subtasks'][1:]):
        assert old['end_pose'] == new['start_pose']
    stops = [e for e in events if e['action'] == 'SUBTASK_STOP']
    assert len(stops) == 4 and all(e['before'] == e['after'] and e['path_delta'] == 0 for e in stops)
    assert [e['path_delta'] for e in events if e['action'] == 'MOVE_FORWARD'] == [3.] * 4
    assert events[2]['after']['rotation_xyzw'] == [0., 0., 0., 1.]
    assert events[3]['after']['rotation_xyzw'] == [0., 1., 0., 0.]
    assert queried[0][1] == list(episode.subtasks[0].view_positions)
    assert queried[2] == (stops[0]['after']['position'], list(episode.subtasks[1].view_positions))
    assert agent.final_instruction_verifier.evidence_store.reset.call_count == 4
    assert np.isnan(agent.env.get_metrics()['distance_to_goal'])
    agent.reset_debug_probes.assert_not_called()
    with pytest.raises(RuntimeError, match='one parent reset'):
        agent.env.reset()


def test_global_budget_emits_actual_stop_without_visiting_remaining_goals(monkeypatch, tmp_path):
    import backends.habitat.goat_runtime as runtime
    episode, events = parse(dataset()), []
    agent, base, _ = continuous_agent(episode, events, global_budget=3)
    monkeypatch.setattr(runtime, 'install_public_goal', lambda *a, **kw: None)
    def nested_scan(agent, **kwargs):
        for _ in range(12):
            agent.execution.step(2)
    monkeypatch.setattr(runtime, '_guided_run', nested_scan)
    result = runtime.run_continuous_parent(agent, episode, parser=None, output=tmp_path, references={})
    assert result['status'] == 'complete'
    assert [e['action'] for e in events] == ['TURN_LEFT', 'TURN_LEFT', 'STOP']
    assert base.actions == [2, 2, 0] and base.reset_count == 1
    assert result['subtasks'][0]['reason'] == result['termination_reason'] == 'global_budget'
    assert len(result['subtasks']) == 4
    assert all(row['reason'] == 'not_executed' and row['start_pose'] is None and row['stop'] is None
               for row in result['subtasks'][1:])
    assert result['metrics']['subtask_success'] == [0.] * 4
    assert result['metrics']['spl_by_subtask'] == [0.] * 4


@pytest.mark.parametrize('strict_available_classes', [False, True])
def test_public_install_uses_original_installer_and_revision_scoped_reference(
        monkeypatch, tmp_path, strict_available_classes):
    import sys
    import backends.habitat.goat_runtime as runtime
    import perception.detector_adapter as detector_adapter
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from task.session import TaskSession
    from scene.contracts import SemanticMapSnapshot, Pose3D
    from verification.contracts import VerificationResult
    from unittest.mock import Mock

    seen, available = [], ['chair', 'lamp']
    parser = SimpleNamespace(vlm='self_hosted', strict_available_classes=strict_available_classes,
        parse_plan=Mock(side_effect=lambda **kwargs: runtime_plan(
            TaskSpec(kwargs['raw_instruction'], (TargetSpec('target', 'lamp'),)))))
    ground = Mock(wraps=detector_adapter.ground_plan)
    monkeypatch.setattr(detector_adapter, 'ground_plan', ground)
    def installer(**kwargs):
        assert kwargs['episode_info'] == {} and kwargs['dataset_target'] in ('chair', '')
        seen.append({key: kwargs[key] for key in ('raw_instruction', 'dataset_target', 'episode_info')})
        if kwargs['source'] == 'benchmark_object_goal':
            plan = runtime_plan(TaskSpec('Find chair', (TargetSpec('target', 'chair'),)))
        else:
            plan = kwargs['instruction_parser'].parse_plan(raw_instruction=kwargs['raw_instruction'],
                dataset_target='', available_classes=available, episode_info={})
        agent = kwargs['habitat_agent']
        agent.task = TaskSession(snapshot=SemanticMapSnapshot(0., Pose3D((0., 0., 0.))), plan=plan)
        (Path(kwargs['episode_dir']) / 'instruction_adapter').mkdir(exist_ok=True)
        return plan, None
    monkeypatch.setitem(sys.modules, 'objnav_benchmark_with_process_obs',
                        SimpleNamespace(_install_instruction_plan=installer))
    describe = Mock(return_value={'category': 'lamp', 'appearance': 'blue glass'})
    monkeypatch.setattr(runtime, '_describe_reference', describe)
    initialize = Mock()
    agent = SimpleNamespace(mapper=SimpleNamespace(target='lamp', object_perceiver=SimpleNamespace(
        sam=SimpleNamespace(initialize=initialize))))
    episode = parse(dataset())
    image = episode.subtasks[2].public_goal
    runtime.install_public_goal(agent, image, parser=parser, revision=3,
                                directory=tmp_path, reference='/real/reference.png')
    parser.parse_plan.assert_not_called()
    ground.assert_called_once()
    assert ground.call_args.kwargs == dict(available_classes=available, vlm='self_hosted',
        strict_available_classes=strict_available_classes, use_legacy_similarity_fallback=False)
    assert agent.task.spec.revision == agent.task.verification_ledger.task_revision == 3
    assert agent.task.spec.source == 'goat_image'
    assert len(agent.task.spec.targets) == len(agent.task.plan.targets) == 1
    assert not agent.task.spec.constraints and not agent.task.plan.constraints
    attributes = {'appearance': 'blue glass', 'goal_reference_image': '/real/reference.png'}
    assert agent.task.spec.targets[0].name == agent.task.plan.terminal_targets[0].name == 'lamp'
    assert agent.task.spec.targets[0].terminal
    assert agent.task.spec.targets[0].attributes == attributes
    assert agent.task.plan.terminal_targets[0].attributes == attributes
    assert agent.task.plan.terminal_targets[0].source == 'goat_image'
    assert agent.task.plan.target_detector_prompts == ['lamp']
    assert seen[0]['raw_instruction'] == (
        'Find the same lamp shown in the reference image. Visible appearance: blue glass')
    assert json.loads((tmp_path / 'instruction_adapter' / 'task_spec.json').read_text()) == agent.task.spec.as_dict()
    describe.assert_called_once_with('/real/reference.png', 'self_hosted')

    old_task = agent.task
    old_spec, old_plan = old_task.spec.as_dict(), old_task.plan.as_dict()
    old_task.progress.completed = True
    old_ledger = old_task.verification_ledger
    record = old_ledger.put(old_task.spec.raw_instruction, 'candidate',
        VerificationResult(False, 'reject_candidate', task_revision=3))
    runtime.install_public_goal(agent, image, parser=parser, revision=4,
                                directory=tmp_path, reference='/real/next-reference.png')
    parser.parse_plan.assert_not_called()
    assert ground.call_count == 2 and describe.call_count == 2
    assert 'chair' not in json.dumps(seen) and 'image_parameters' not in json.dumps(seen)
    assert agent.task.spec.revision == agent.task.verification_ledger.task_revision == 4
    new_attributes = {**attributes, 'goal_reference_image': '/real/next-reference.png'}
    assert agent.task.spec.targets[0].attributes == new_attributes
    assert agent.task.plan.terminal_targets[0].attributes == new_attributes
    assert agent.task.verification_ledger is not old_ledger
    assert not agent.task.verification_ledger.records and not agent.task.verification_ledger.history
    assert agent.task.verification_ledger.get(agent.task.spec.raw_instruction, 'candidate') is None
    agent.task.verification_ledger.put(agent.task.spec.raw_instruction, 'candidate',
        VerificationResult(False, 'need_better_view', task_revision=4))
    assert old_ledger.task_revision == 3 and old_ledger.history == [record]
    assert old_ledger.get(old_task.spec.raw_instruction, 'candidate') is record
    assert old_task.spec.as_dict() == old_spec and old_task.plan.as_dict() == old_plan
    assert old_task.progress.completed and not agent.task.progress.completed

    runtime.install_public_goal(agent, episode.subtasks[0].public_goal,
                                parser=parser, revision=5, directory=tmp_path)
    parser.parse_plan.assert_not_called()
    assert agent.task.spec.revision == agent.task.verification_ledger.task_revision == 5
    assert agent.task.spec.targets[0].name == 'chair'
    assert not agent.task.spec.targets[0].attributes
    assert not agent.task.progress.completed
    runtime.install_public_goal(agent, episode.subtasks[1].public_goal,
                                parser=parser, revision=6, directory=tmp_path)
    parser.parse_plan.assert_called_once_with(raw_instruction=episode.subtasks[1].public_goal.description,
        dataset_target='', available_classes=available, episode_info={})
    assert agent.task.spec.revision == agent.task.verification_ledger.task_revision == 6
    assert not agent.task.spec.targets[0].attributes
    assert ground.call_count == describe.call_count == 2
    assert initialize.call_count == 4


def test_reference_assets_keep_original_pixels_and_require_camera_binding(tmp_path, monkeypatch):
    from PIL import Image
    from backends.habitat.goat_runtime import prepare_reference_images, GOATUnavailable
    episode = parse(dataset())
    with pytest.raises(GOATUnavailable, match='unverified'):
        prepare_reference_images(episode, asset_map=None, data_root=tmp_path, output=tmp_path)
    path = tmp_path / 'reference.png'
    Image.new('RGB', (512, 512), (30, 60, 90)).save(path)
    task = episode.subtasks[2]
    camera = {key: task.image_parameters[key] for key in ('position', 'rotation', 'hfov', 'image_dimensions')}
    entry = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 scene_id=episode.scene_id, image_parameters=camera)
    mapping = tmp_path / 'assets.json'
    mapping.write_text(json.dumps({task.public_goal.image_ref: entry}))
    refs, records = prepare_reference_images(episode, asset_map=mapping, data_root=tmp_path, output=tmp_path)
    assert Path(refs[task.public_goal.image_ref]).read_bytes() == path.read_bytes()
    assert records[0]['sha256'] == entry['sha256']
    import base64
    from unittest.mock import Mock
    import llm_utils.cognav_llm_adapter as adapter
    from backends.habitat.goat_runtime import _describe_reference
    request = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        parsed=SimpleNamespace(category='  lamp\n', appearance=' blue glass\t')))]))
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=request))))
    monkeypatch.setattr(adapter, 'get_client_and_model', lambda vlm: (client, 'Qwen-local'))
    monkeypatch.delenv('LLM_OFFLINE', raising=False)
    image_goal = _describe_reference(refs[task.public_goal.image_ref], 'self_hosted')
    assert image_goal == {'category': 'lamp', 'appearance': 'blue glass'}
    content = request.call_args.kwargs['messages'][0]['content']
    assert base64.b64decode(content[1]['image_url']['url'].split(',')[1]) == path.read_bytes()
    entry['image_parameters']['hfov'] = 42
    mapping.write_text(json.dumps({task.public_goal.image_ref: entry}))
    with pytest.raises(GOATUnavailable, match='camera parameters'):
        prepare_reference_images(episode, asset_map=mapping, data_root=tmp_path, output=tmp_path)


@pytest.mark.parametrize('distance', [0., float('inf')])
def test_driver_undefined_spl_keeps_stop_and_reports_unavailable_score(tmp_path, monkeypatch, distance):
    import backends.habitat.goat_runtime as runtime
    parent = parse(dataset())
    episode = replace(parent, subtasks=parent.subtasks[:1])
    events = []
    agent, base, _ = continuous_agent(episode, events)
    base.sim.geodesic_distance = lambda *args: distance
    monkeypatch.setattr(runtime, 'install_public_goal', lambda *a, **kw: None)
    monkeypatch.setattr(runtime, '_guided_run', lambda *a, **kw: {'reason': 'search_exhausted'})
    result = runtime.run_continuous_parent(agent, episode, parser=None, output=tmp_path, references={})
    value = runtime._write_json(tmp_path / 'result.json', result)
    assert events[0]['action'] == value['subtasks'][0]['stop'] == 'SUBTASK_STOP'
    if distance == 0:
        assert value['status'] == 'infrastructure_error' and value['metrics'] is None
        assert value['error']['type'] == 'ZeroDivisionError'
    else:
        assert value['status'] == 'complete' and value['metrics']['composite_spl'] is None
        assert any(note['value'] == 'nan' for note in value['nonfinite_values'])
    assert json.loads((tmp_path / 'result.json').read_text()) == value


@pytest.mark.parametrize('bad_action', [False, True])
def test_legacy_carrier_recomputes_owned_stretch_navmesh(monkeypatch, tmp_path, bad_action):
    import sys
    from unittest.mock import Mock
    from benchmark.goat_environment import GOATEnvironment, make_legacy_environment

    cfg = SimpleNamespace(defrost=Mock(), freeze=Mock(), DATASET=SimpleNamespace(),
        ENVIRONMENT=SimpleNamespace(ITERATOR_OPTIONS=SimpleNamespace()), TASK=SimpleNamespace(),
        SIMULATOR=SimpleNamespace(SCENE_DATASET='', HABITAT_SIM_V0=SimpleNamespace(),
            AGENT_0=SimpleNamespace(), RGB_SENSOR=SimpleNamespace(), DEPTH_SENSOR=SimpleNamespace()))
    profile = dict(height=1.41, radius=.17, forward_step_size=.25, turn_angle=30.,
                   tilt_angle=30., image_dimensions=[240, 320], hfov=90., camera_position=[0., 1.31, 0.])
    amounts = [0., .25, 30., 30., 30., 30.]
    actions = {key: SimpleNamespace(name=name.lower(), actuation=SimpleNamespace(amount=amounts[key]))
               for key, name in GOATEnvironment.ACTIONS.items()}
    if bad_action:
        actions[4].actuation.amount = 15.
    original_pathfinder = Mock()
    actual_sim = SimpleNamespace(pathfinder=original_pathfinder,
                                 sim_config=SimpleNamespace(agents=[SimpleNamespace(action_space=actions)],
                                     sim_cfg=SimpleNamespace(scene_dataset_config_file=str(tmp_path / 'scenes.json'))))
    new_pathfinder = SimpleNamespace(is_loaded=True, is_navigable=Mock(return_value=True))
    def save(path):
        Path(path).write_bytes(b'recomputed-stretch')
        return True
    new_pathfinder.save_nav_mesh = Mock(side_effect=save)
    class Settings:
        def set_defaults(self):
            self.agent_height, self.agent_radius = 1.5, .1
            self.agent_max_climb, self.cell_height = .2, .2
            self.cell_size, self.filter_ledge_spans = .05, True
    def recompute(pathfinder, settings, *, include_static_objects):
        assert pathfinder is new_pathfinder is actual_sim.pathfinder
        assert settings.agent_height == 1.41 and settings.agent_radius == .17
        assert settings.agent_max_climb == settings.cell_height == .2
        assert settings.cell_size == .05 and settings.filter_ledge_spans
        assert include_static_objects is False
        return True
    actual_sim.recompute_navmesh = Mock(side_effect=recompute)
    habitat_sim = SimpleNamespace(__version__='0.2.4', NavMeshSettings=Settings,
                                  PathFinder=Mock(return_value=new_pathfinder))
    base = SimpleNamespace(sim=actual_sim, reset=Mock())
    habitat = SimpleNamespace(__version__='0.2.1', Env=Mock(return_value=base))
    dataset_factory = Mock()
    for name, module in {
        'habitat': habitat, 'habitat_sim': habitat_sim,
        'habitat.config.default': SimpleNamespace(get_config=lambda: cfg),
        'habitat.sims.habitat_simulator.actions': SimpleNamespace(
            HabitatSimActions=SimpleNamespace(**{name: key for key, name in GOATEnvironment.ACTIONS.items()})),
        'benchmark.instance_goal_dataset': SimpleNamespace(InstanceGoalDataset=dataset_factory),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    episode = parse(dataset())
    source_mesh = tmp_path / 'scene.basis.navmesh'
    source_mesh.write_bytes(b'original-basis')
    result_base, result_cfg, after_reset = make_legacy_environment(
        episode, scene=tmp_path / 'scene.basis.glb', scene_dataset=tmp_path / 'scenes.json',
        scenes_root=tmp_path, profile=profile, global_budget=100, output=tmp_path)
    assert result_base is base and result_cfg is cfg
    habitat.Env.assert_called_once_with(config=cfg, dataset=dataset_factory.return_value)
    base.reset.assert_not_called()
    actual_sim.recompute_navmesh.assert_not_called()
    carrier = json.loads(dataset_factory.return_value.from_json.call_args.args[0])['episodes'][0]
    assert carrier['start_position'] == list(episode.start_position)
    assert carrier['start_rotation'] == list(episode.start_rotation)
    assert carrier['scene_dataset_config'] == str(tmp_path / 'scenes.json')
    assert cfg.ENVIRONMENT.MAX_EPISODE_STEPS == 101
    assert cfg.SIMULATOR.RGB_SENSOR.HEIGHT == 240 and cfg.SIMULATOR.RGB_SENSOR.WIDTH == 320
    assert cfg.SIMULATOR.ACTION_SPACE_CONFIG == 'v1'
    assert cfg.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING is False
    if bad_action:
        with pytest.raises(RuntimeError, match='Effective simulator action'):
            after_reset(actual_sim)
        actual_sim.recompute_navmesh.assert_not_called()
        return
    receipt = after_reset(actual_sim)
    assert receipt['habitat_lab_version'] == '0.2.1' and receipt['habitat_sim_version'] == '0.2.4'
    assert receipt['scope'] == 'legacy_habitat_compat_not_official_version_equivalence'
    assert receipt['recomputed'] and receipt['is_loaded'] and receipt['start_navigable']
    assert receipt['settings']['agent_height'] == 1.41
    assert receipt['settings']['agent_radius'] == .17 and receipt['settings']['cell_size'] == .05
    assert receipt['include_static_objects'] is False
    assert receipt['path'] == str(tmp_path / 'stretch.navmesh')
    assert receipt['sha256'] == hashlib.sha256(b'recomputed-stretch').hexdigest()
    assert receipt['actions'][3] == dict(id=4, name='look_up', amount=30.)
    json.dumps(receipt)
    original_pathfinder.load_nav_mesh.assert_not_called()
    new_pathfinder.is_navigable.assert_called_once_with(episode.start_position)
    actual_sim.recompute_navmesh.assert_called_once()
    assert source_mesh.read_bytes() == b'original-basis'


@pytest.mark.parametrize('failure', [None, 'draw', 'restore'])
def test_reference_render_world_camera_and_full_pose_restore(monkeypatch, tmp_path, failure):
    import sys
    from unittest.mock import Mock
    from PIL import Image
    from benchmark.goat_environment import render_reference_image

    original = SimpleNamespace(position=np.array([1., 2., 3.]), rotation=np.array([0., 1., 0., 0.]),
        sensor_states={
            'rgb': SimpleNamespace(position=np.array([1., 3.31, 3.]),
                                   rotation=np.array([.5, 0., 0., np.sqrt(.75)])),
            'depth': SimpleNamespace(position=np.array([1.1, 3.31, 3.]),
                                     rotation=np.array([-.5, 0., 0., np.sqrt(.75)]))})
    uuid = 'goat_reference'
    events = []
    agent = SimpleNamespace(state=deepcopy(original), _sensors={'rgb': object(), 'depth': object()},
                            scene_node=object(), agent_config=SimpleNamespace(
                                sensor_specifications=[SimpleNamespace(uuid=k) for k in original.sensor_states]))
    agent.get_state = lambda: deepcopy(agent.state)
    def set_state(state, *, reset_sensors, infer_sensor_states):
        assert reset_sensors is False and infer_sensor_states is False
        events.append('set_state')
        agent.state = deepcopy(state)
        if failure == 'restore' and uuid not in state.sensor_states:
            agent.state.sensor_states['rgb'].rotation = np.array([0., 0., 0., 1.])
    agent.set_state = Mock(side_effect=set_state)
    rgba = np.arange(4 * 7 * 4, dtype=np.uint8).reshape(4, 7, 4)
    sensor = SimpleNamespace(get_observation=Mock(return_value=rgba), close=Mock())
    parameters = dict(position=[8., 1.3, 2.], rotation=[.5, 0., 0., np.sqrt(.75)],
                      hfov=73., image_dimensions=[4, 7], frame_coverage=.3)
    def draw():
        events.append('draw')
        pose = agent.state.sensor_states[uuid]
        np.testing.assert_allclose(pose.position, parameters['position'])  # No robot height added.
        np.testing.assert_allclose(pose.rotation, parameters['rotation'])
        for key, old_pose in original.sensor_states.items():
            np.testing.assert_array_equal(agent.state.sensor_states[key].rotation, old_pose.rotation)
        if failure == 'draw':
            raise RuntimeError('draw failed')
    sensor.draw_observation = Mock(side_effect=draw)
    sim = SimpleNamespace(_sensors={'rgb': object(), 'depth': object()}, step=Mock(),
                          get_agent=Mock(return_value=agent), get_sensor_observations=Mock())
    def add(spec):
        events.append('add')
        assert spec.uuid == uuid and spec.sensor_type == 'COLOR' and spec.sensor_subtype == 'PINHOLE'
        assert spec.resolution == [4, 7] and spec.hfov == 73.
        sensor._spec = deepcopy(spec)
        node = Mock()
        node.parent.absolute_transformation.return_value.inverted.return_value.transform_point.return_value = [7., -.7, -1.]
        sim._sensors[uuid], agent._sensors[uuid] = sensor, SimpleNamespace(node=node)
        agent.agent_config.sensor_specifications.append(spec)
        agent.state.sensor_states[uuid] = SimpleNamespace(position=np.zeros(3), rotation=[0., 0., 0., 1.])
    sim.add_sensor = Mock(side_effect=add)
    delete = Mock()
    quaternion = Mock(side_effect=lambda xyzw: np.asarray(xyzw, dtype=float))
    for name, module in {
        'habitat_sim': SimpleNamespace(CameraSensorSpec=SimpleNamespace,
            SensorType=SimpleNamespace(COLOR='COLOR'), SensorSubType=SimpleNamespace(PINHOLE='PINHOLE')),
        'habitat_sim._ext': SimpleNamespace(habitat_sim_bindings=SimpleNamespace(
            SensorFactory=SimpleNamespace(delete_subtree_sensor=delete))),
        'habitat_sim.agent.agent': SimpleNamespace(SixDOFPose=SimpleNamespace),
        'habitat_sim.utils.common': SimpleNamespace(quat_from_coeffs=quaternion),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / 'reference.png'
    if failure:
        with pytest.raises(RuntimeError, match='draw failed' if failure == 'draw' else 'full agent/sensor pose'):
            render_reference_image(sim, parameters, path)
        assert not path.exists()
    else:
        receipt = render_reference_image(sim, parameters, path)
        np.testing.assert_array_equal(np.asarray(Image.open(path)), rgba[:, :, :3])
        assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt['pose_restored'] is True
        assert receipt['image_parameters'] == {k: parameters[k] for k in
                                               ('position', 'rotation', 'hfov', 'image_dimensions')}
        assert receipt['actual_sensor'] == dict(uuid=uuid, sensor_type='COLOR', sensor_subtype='PINHOLE',
            resolution=[4, 7], hfov=73., position=parameters['position'], rotation=parameters['rotation'])
        json.dumps(receipt)
    assert events == ['add', 'set_state', 'draw', 'set_state']
    assert agent.set_state.call_count == 2
    quaternion.assert_called_once_with(parameters['rotation'])
    sim.step.assert_not_called()
    sim.get_sensor_observations.assert_not_called()
    sensor.close.assert_called_once()
    delete.assert_called_once_with(agent.scene_node, uuid)
    assert set(sim._sensors) == set(agent._sensors) == {'rgb', 'depth'}
    assert [s.uuid for s in agent.agent_config.sensor_specifications] == ['rgb', 'depth']
    np.testing.assert_array_equal(agent.state.position, original.position)
    np.testing.assert_array_equal(agent.state.rotation, original.rotation)
    if failure != 'restore':
        for key, pose in original.sensor_states.items():
            np.testing.assert_array_equal(agent.state.sensor_states[key].position, pose.position)
            np.testing.assert_array_equal(agent.state.sensor_states[key].rotation, pose.rotation)


@pytest.mark.parametrize('height,width', [(480, 640), (640, 360)])
def test_centered_camera_depth_bounds_preserve_image_shape(height, width):
    from mapping_utils.preprocess import calc_depth_img, preprocess_depth

    intrinsic = np.array([[320., 0., (width - 1) / 2],
                          [0., 320., (height - 1) / 2], [0., 0., 1.]])
    lower, upper = calc_depth_img(intrinsic, .51, 4.9)
    assert lower.shape == upper.shape == (height, width)
    depth = np.full((height, width), 2., dtype=np.float32)
    np.testing.assert_array_equal(preprocess_depth(depth, lower, upper), depth)
