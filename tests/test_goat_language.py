"""Preserve GOAT provenance and the independent-task denominator."""
import pytest
from benchmark.goat_language import language_tasks, freeze_subset


def test_language_subset_preserves_targets_and_removes_repeated_subtasks():
    goal=dict(object_id='chair_1',object_category='chair',position=[1,0,0],
              view_points=[],lang_desc='the red chair next to the table')
    parent=dict(episode_id=7,scene_id='hm3d/val//00802-wcojb4TFT35/wcojb4TFT35.basis.glb',
                start_position=[0,0,0],start_rotation=[0,0,0,1],
                tasks=[['chair','object',None],['chair','description','chair_1'],
                       ['chair','description','chair_1']])
    tasks=language_tasks({'goals':{'chair':[goal]},'episodes':[parent]},split='val_unseen',source_file='source.gz')
    assert len(tasks)==2
    assert tasks[0]['info']['source_subtask_index']==1
    assert tasks[0]['info']['instruction']==goal['lang_desc']
    assert tasks[0]['goals'][0]['object_id']=='chair_1'
    assert tasks[0]['start_position']==parent['start_position']
    selected,unique=freeze_subset(tasks,1,20261105)
    assert unique==1
    assert selected[0]['episode_id'].endswith(':7:1')
    with pytest.raises(ValueError,match='available 1'):
        freeze_subset(tasks,2,20261105)


@pytest.mark.parametrize('invalid', [None, 'identity', 'instruction', 'protocol', 'nonfinite', 'unfinished',
                                     'independent_identity', 'independent_protocol'])
@pytest.mark.parametrize('with_trace', [False, True])
def test_language_launcher_binds_frozen_task_and_score(tmp_path, invalid, with_trace, evaluation_env):
    import csv
    import gzip
    import json
    from scripts.launch_goat_episode import language_inputs, language_score
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    row = dict(episode_id='val:scene:3:2', scene_id='scene/a.glb',
               start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
               object_category='chair', info=dict(instruction='the red chair',
               protocol='goat_language_single_parent_start_v1'))
    source = tmp_path / 'tasks.json.gz'
    with gzip.open(source, 'wt') as stream:
        json.dump({'episodes': [row]}, stream)
    argv, identity, instruction = language_inputs(source, 0, tmp_path, tmp_path, 500, 60)
    assert argv[argv.index('--benchmark') + 1] == 'goat_language'
    assert argv[argv.index('--eval_episodes') + 1] == '1'
    assert instruction == row['info']['instruction']
    assert json.loads((tmp_path / 'runtime_episodes.json').read_text()) == {'runtime_episodes': [identity]}
    folder = tmp_path / 'navigation/task'
    episode = folder / 'episode-0'
    (episode / 'harness').mkdir(parents=True)
    actual = dict(identity)
    if invalid == 'identity':
        actual['episode_id'] = 'wrong'
    (episode / 'episode_identity.json').write_text(json.dumps(actual))
    (episode / 'harness/termination.json').write_text(json.dumps({'reason': 'episode_limit'}))
    if with_trace:
        for key, value in identity.items():
            setattr(evaluation_env.current_episode, key, value)
        trace = HabitatEvaluationTrace(evaluation_env, episode)
        evaluation_env.moves = [[2, 1, 0]]
        trace.step(0)
    metrics = dict(Episode='0', object_goal=instruction, benchmark='goat_language',
                   benchmark_success_distance='0.25', success='0', spl='0', distance_to_goal='2', End='1')
    if invalid == 'instruction':
        metrics['object_goal'] = 'the blue chair'
    if invalid == 'protocol':
        metrics['benchmark_success_distance'] = '1.0'
    if invalid == 'nonfinite':
        metrics['spl'] = 'nan'
    if invalid == 'unfinished':
        metrics['End'] = '0'
    with (folder / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    independent = score_habitat_evaluation(episode)
    if invalid == 'independent_identity':
        independent['provenance']['episode_identity'] = dict(identity, episode_id='wrong')
    if invalid == 'independent_protocol':
        independent['provenance']['schema'] = 'wrong'
    if invalid not in (None, 'nonfinite'):
        with pytest.raises(ValueError):
            language_score(tmp_path, 0, identity, instruction, independent)
    else:
        score = language_score(tmp_path, 0, identity, instruction, independent)
        if invalid == 'nonfinite':
            assert not score['aggregate_ready'] and score['metrics'] is None
            assert score['reason'] == 'nonfinite_runtime_metrics'
        else:
            assert score['aggregate_ready'] and score['metrics'] == {'SR': 0, 'SPL': 0, 'distance_to_goal': 2}
        assert score['raw_metrics'] == metrics
        json.dumps(score, allow_nan=False)
        assert score['protocol'] == 'goat_language_single_parent_start_v1'
        if with_trace:
            assert score['independent_metrics']['status'] == 'available'
            assert score['independent_metrics']['SR'] == score['independent_metrics']['SPL'] == 1
            independent['provenance'].pop('episode_identity')
            with pytest.raises(ValueError, match='Independent evaluation episode identity differs'):
                language_score(tmp_path, 0, identity, instruction, independent)
        else:
            assert score['independent_metrics']['status'] == 'unavailable'
            assert 'SR' not in score['independent_metrics']
            assert 'SPL' not in score['independent_metrics']


@pytest.mark.parametrize('outcome', ['policy_error', 'success', 'scoring_error', 'nonfinite_json', 'overflow_json'])
@pytest.mark.parametrize('observation_mode', [None, 'multi'])
def test_language_launcher_resolves_symlinked_data_mount(tmp_path, monkeypatch, evaluation_env, outcome, observation_mode):
    import csv
    import gzip
    import json
    import subprocess
    import sys
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    from scripts import launch_goat_episode as launcher

    data = tmp_path / 'data'
    data.mkdir()
    alias = tmp_path / 'data-link'
    alias.symlink_to(data, target_is_directory=True)
    source = tmp_path / 'source'
    source.mkdir()
    task = dict(episode_id='val:scene:1:0', scene_id='scene/a.glb',
                start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1], object_category='chair',
                info=dict(instruction='the red chair', protocol='goat_language_single_parent_start_v1'))
    with gzip.open(data / 'tasks.json.gz', 'wt') as stream:
        json.dump({'episodes': [task]}, stream)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['launcher', '--output', str(output), '--data-root', str(alias),
        '--language-task-file', str(alias / 'tasks.json.gz'), '--language-task-index', '0'] + (
            ['--observation-mode', observation_mode] if observation_mode else []))
    for name in ('STRIVE_ROOT', 'HF_HOME_HOST', 'SAM_CHECKPOINT', 'GROUNDING_DINO_CHECKPOINT'):
        monkeypatch.setenv(name, str(source))
    monkeypatch.setenv('VLN_LVLM_API_KEY', 'test-only')
    fake_habitat = sys.modules['habitat_sim']
    monkeypatch.setitem(sys.modules, 'habitat_sim', None)  # The production host has no Habitat.
    episode = output / 'navigation/task/episode-0'
    commands = []
    policy_commands = []

    def run_policy(command, **kwargs):
        policy_commands.append(command)
        if outcome == 'policy_error':
            return subprocess.CompletedProcess(command, 1)
        identity = {key: task[key] for key in
                    ('episode_id', 'start_position', 'start_rotation', 'object_category')}
        identity['scene_id'] = str(data / 'scene_datasets' / task['scene_id'])
        (episode / 'harness').mkdir(parents=True)
        (episode / 'episode_identity.json').write_text(json.dumps(identity))
        (episode / 'harness/termination.json').write_text(json.dumps({'reason': 'episode_limit'}))
        metrics = dict(Episode='0', object_goal='the red chair', benchmark='goat_language',
                       benchmark_success_distance='.25', success='1', spl='nan', distance_to_goal='0', End='1')
        with (episode.parent / 'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        with monkeypatch.context() as container:
            container.setitem(sys.modules, 'habitat_sim', fake_habitat)
            for key, value in identity.items():
                setattr(evaluation_env.current_episode, key, value)
            trace = HabitatEvaluationTrace(evaluation_env, episode)
            evaluation_env.moves = [[2, 1, 0]]
            trace.step(0)
        monkeypatch.delenv('VLN_LVLM_API_KEY')  # Scoring must not re-read policy credentials.
        return subprocess.CompletedProcess(command, 0)

    def check_output(command, **kwargs):
        commands.append(command)
        assert kwargs == {'text': True}
        if command[:3] == ['docker', 'image', 'inspect']:
            return 'sha256:test\n'
        assert sys.modules['habitat_sim'] is None
        if outcome == 'scoring_error':
            raise subprocess.CalledProcessError(1, command)
        with monkeypatch.context() as container:
            container.setitem(sys.modules, 'habitat_sim', fake_habitat)
            independent = score_habitat_evaluation(episode)
        if outcome in ('nonfinite_json', 'overflow_json'):
            independent['provenance']['episode_info']['bad'] = 'NONFINITE'
            return json.dumps(independent).replace('"NONFINITE"',
                'NaN' if outcome == 'nonfinite_json' else '1e999')
        return json.dumps(independent, allow_nan=False)

    monkeypatch.setattr(launcher.subprocess, 'check_output', check_output)
    monkeypatch.setattr(launcher.subprocess, 'run', run_policy)
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == (1 if outcome == 'policy_error' else 0)
    assert len(policy_commands) == 1
    command = policy_commands[0]
    assert command[command.index('--observation-mode') + 1] == (observation_mode or 'single')
    receipt = json.loads((output / 'execution.json').read_text())
    assert receipt['observation_mode'] == (observation_mode or 'single')
    assert len(commands) == (1 if outcome == 'policy_error' else 2)
    if outcome != 'policy_error':
        assert commands[-1] == [
            'docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'MAGNUM_LOG=quiet', '-e', 'HABITAT_SIM_LOG=quiet',
            '-v', f'{source}:/workspace/STRIVE:ro', '-v', f'{output}:{output}:ro',
            '-w', '/workspace/STRIVE', 'sha256:test', '-m', 'benchmark.habitat_evaluation', str(episode),
        ]
        result = json.loads((output / 'evaluation.json').read_text())
        assert not result['aggregate_ready'] and result['metrics'] is None
        if outcome == 'success':
            assert result['raw_metrics']['spl'] == 'nan'
            assert result['independent_metrics']['SR'] == result['independent_metrics']['SPL'] == 1
            assert result['independent_metrics']['provenance']['evaluation_directory'] == str(episode / 'evaluation')
        else:
            assert 'independent_metrics' not in result
            assert result['unavailable_metrics']['navigation']
    execution = json.loads((output / 'execution.json').read_text())
    assert f'{data}:{data}:ro' in execution['command']
    assert execution['environment']['HM3D_DATA_PATH'] == str(data)
    assert execution['environment']['VLN_LVLM_TRANSPORT_RETRIES'] == '0'
    assert 'VLN_LVLM_TRANSPORT_RETRIES' in execution['command']
    argv = execution['command']
    assert argv[argv.index('--dataset_path') + 1] == str(data / 'tasks.json.gz')
    assert execution['language_task']['identity']['scene_id'] == str(data / 'scene_datasets/scene/a.glb')


@pytest.fixture
def evaluation_env(monkeypatch):
    """Provide a base, distinct camera, and mesh snapshots reloadable by independent pathfinders."""
    import json
    import math
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS

    class Pathfinder:
        instances = []
        nav_mesh_settings = NS(agent_radius=0.10000000149, agent_height=1.5,
                               cell_height=0.20000000298, filter_low_hanging_obstacles=True)

        def __init__(self):
            self.queries = []
            self.distances = {}
            self.is_loaded = False
            self.loaded_from = None
            self.instances.append(self)

        def save_nav_mesh(self, path):
            """Freeze the fake mesh's geometry, not any recorder-written distance receipts."""
            Path(path).write_text(json.dumps([
                [start, end, distance if math.isfinite(distance) else None]
                for (start, end), distance in self.distances.items()], allow_nan=False))
            return True

        def load_nav_mesh(self, path):
            """Recreate an independent mesh with its own overrides and Euclidean path queries."""
            self.distances = {(tuple(start), tuple(end)): distance if distance is not None else math.inf
                              for start, end, distance in json.loads(Path(path).read_text())}
            self.is_loaded = True
            self.loaded_from = Path(path)
            return True

        def find_path(self, query):
            """Fill a shortest-path request without consulting benchmark metrics or receipts."""
            assert self.is_loaded
            start, end = list(query.requested_start), list(query.requested_end)
            self.queries.append((start, end))
            query.geodesic_distance = self.distances.get((tuple(start), tuple(end)), math.dist(start, end))
            query.points = [start, end]
            return query.geodesic_distance != float('inf')

    class Env:
        def __init__(self):
            self.position = [0., 1., 0.]
            self.moves = []
            self.actions = []
            self.sim = NS(pathfinder=Pathfinder(), get_agent_state=self.state)
            self.sim.pathfinder.is_loaded = True
            self.current_episode = NS(
                episode_id='test:3:2', scene_id='scene/a.glb', object_category='chair',
                start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
                info={'instruction': 'the red chair', 'source_subtask_index': 2},
                goals=[NS(object_id='chair_1', view_points=[
                    NS(agent_state=NS(position=[8, 1, 0])), NS(agent_state=NS(position=[9, 1, 0]))]),
                    NS(object_id='chair_2', view_points=[NS(agent_state=NS(position=[2, 1, 0]))])])

        def state(self):
            """Expose a mutable base position and a deliberately different camera."""
            return NS(position=self.position, rotation=NS(imag=[0, 0, 0], real=1.),
                      sensor_states={'rgb': NS(position=[100, 100, 100])})

        def step(self, action):
            """Apply one scripted base displacement and retain the observation."""
            self.actions.append(action)
            if self.moves:
                self.position[:] = self.moves.pop(0)
            self.observation = {'action': action, 'step': len(self.actions)}
            return self.observation

        def get_metrics(self):
            """Fail if independent evaluation tries to transcribe benchmark metrics."""
            raise AssertionError('Benchmark metrics must not enter independent evaluation')

    monkeypatch.setitem(sys.modules, 'habitat_sim', NS(PathFinder=Pathfinder, ShortestPath=NS, __version__='fake'))
    return Env()


def test_language_trace_records_owned_steps_base_poses_and_all_viewpoint_minimum(tmp_path, evaluation_env, monkeypatch):
    """The immutable GT and full 3D step displacements determine independent SPL."""
    import json
    import math
    from backends.habitat.motion import HabitatExecution
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    env = evaluation_env
    execution = HabitatExecution(env)
    trace = HabitatEvaluationTrace(env, tmp_path, provenance={'task_file_sha256': 'frozen-input'})
    execution.controller.evaluation_trace = trace
    reference = json.loads((tmp_path / 'evaluation/reference.json').read_text())
    assert env.actions == []
    assert reference['episode_identity']['start_position'] == [0, 0, 0]
    assert reference['initial_base_pose'] == {'position': [0, 1, 0], 'rotation': [0, 0, 0, 1]}
    assert reference['reference_distance'] == 2
    assert [path['distance'] for path in reference['viewpoint_paths']] == [8, 9, 2]
    assert reference['viewpoint_paths'][-1]['path_points'] == [[0, 1, 0], [2, 1, 0]]
    assert reference['navmesh']['settings']['agent_radius'] == 0.10000000149
    assert reference['navmesh']['settings']['filter_low_hanging_obstacles'] is True
    assert reference['input_provenance'] == {'task_file_sha256': 'frozen-input'}
    env.current_episode.goals[-1].view_points[0].agent_state.position[:] = [999, 999, 999]
    env.moves = [[0, 3, 0], [2, 1, 0], [2, 1, .1]]
    observation = execution.step(1, request_id='one')
    assert observation is env.observation
    assert execution.step(1, request_id='one') is observation
    rows = [json.loads(line) for line in (tmp_path / 'evaluation/trajectory.jsonl').read_text().splitlines()]
    assert rows == [{'step': 0, 'action': 1, 'before': [0, 1, 0], 'after': [0, 3, 0]}]
    execution.step(1)
    execution.step(0)
    rows = [json.loads(line) for line in (tmp_path / 'evaluation/trajectory.jsonl').read_text().splitlines()]
    assert env.actions == [1, 1, 0]
    assert [row['step'] for row in rows] == [0, 1, 2]
    assert rows[-1]['before'] == [2, 1, 0] and rows[-1]['after'] == [2, 1, .1]
    assert rows[-1]['final_distance'] == .1
    assert rows[-1]['final_viewpoint_paths'][-1]['path_points'] == [[2, 1, .1], [2, 1, 0]]
    assert env.sim.pathfinder.queries == []
    assert trace.pathfinder is not env.sim.pathfinder
    assert len(trace.pathfinder.queries) == 6
    assert trace.pathfinder.loaded_from == tmp_path / 'evaluation/runtime.navmesh'
    def no_live_access(*args, **kwargs):
        raise AssertionError('Offline scoring must not access the simulator or recorder pathfinder')
    monkeypatch.setattr(env, 'step', no_live_access)
    monkeypatch.setattr(env.sim, 'get_agent_state', no_live_access)
    monkeypatch.setattr(trace.pathfinder, 'find_path', no_live_access)
    score = score_habitat_evaluation(tmp_path, expected_identity=reference['episode_identity'])
    offline = type(trace.pathfinder).instances[-1]
    assert offline is not trace.pathfinder and offline is not env.sim.pathfinder
    assert offline.loaded_from == tmp_path / 'evaluation/runtime.navmesh'
    assert len(offline.queries) == 6
    assert env.actions == [1, 1, 0]
    assert score['status'] == 'available' and score['SR'] == 1
    assert score['path_length'] == pytest.approx(2 + math.sqrt(8) + .1)
    assert score['SPL'] == pytest.approx(2 / (2 + math.sqrt(8) + .1))
    assert score['provenance']['navmesh']['sha256'] == reference['navmesh']['sha256']


def test_language_live_mesh_mutation_does_not_change_frozen_queries(tmp_path, evaluation_env, monkeypatch):
    """In-place runtime mesh edits cannot change either recorder or offline distances."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    env = evaluation_env
    live = env.sim.pathfinder
    live.distances[((0, 1, 0), (2, 1, 0))] = 4.0
    trace = HabitatEvaluationTrace(env, tmp_path)
    for start in ((0, 1, 0), (2, 1, 0)):
        for end in ((8, 1, 0), (9, 1, 0), (2, 1, 0)):
            live.distances[(start, end)] = float('inf')
    def no_live_query(*args):
        raise AssertionError('Frozen evaluation must not query the runtime pathfinder')
    monkeypatch.setattr(live, 'find_path', no_live_query)
    env.moves = [[2, 1, 0]]
    trace.step(0)
    terminal = json.loads((tmp_path / 'evaluation/trajectory.jsonl').read_text())
    assert terminal['final_distance'] == 0
    score = score_habitat_evaluation(tmp_path)
    assert score['reference_distance'] == 4.0
    assert score['distance_to_goal'] == 0
    assert score['SR'] == score['SPL'] == 1
    assert live.queries == [] and env.actions == [0]


@pytest.mark.parametrize('phase', ['initial', 'terminal'])
@pytest.mark.parametrize('tampering', ['minimum_distance', 'other_distance', 'unreachable', 'reachable'])
def test_language_independent_recomputes_each_distance_receipt(tmp_path, evaluation_env, phase, tampering):
    """Even internally consistent distance and reachability receipts must match the saved mesh."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    env = evaluation_env
    if tampering == 'reachable':
        for start in ((0, 1, 0), (2, 1, 0)):
            env.sim.pathfinder.distances[(start, (8, 1, 0))] = float('inf')
    trace = HabitatEvaluationTrace(env, tmp_path)
    env.moves = [[2, 1, 0]]
    trace.step(0)
    path = tmp_path / 'evaluation' / ('reference.json' if phase == 'initial' else 'trajectory.jsonl')
    record = json.loads(path.read_text())
    paths = record['viewpoint_paths' if phase == 'initial' else 'final_viewpoint_paths']
    if tampering == 'minimum_distance':
        paths[-1]['distance'] += .01
    elif tampering == 'other_distance':
        paths[0]['distance'] += 1
    elif tampering == 'unreachable':
        paths[0].update(distance=None, path_points=[])
    else:
        paths[0].update(distance=8, path_points=[[0, 1, 0], [8, 1, 0]])
    record['reference_distance' if phase == 'initial' else 'final_distance'] = min(
        item['distance'] for item in paths if item['distance'] is not None)
    path.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='differs from frozen NavMesh'):
        score_habitat_evaluation(tmp_path)


def test_language_distance_tolerance_never_changes_recomputed_score(tmp_path, evaluation_env):
    """Receipt float noise is tolerated, but the success boundary uses fresh mesh distances."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    evaluation_env.moves = [[2, 1, .24999999]]
    trace.step(0)
    path = tmp_path / 'evaluation/trajectory.jsonl'
    terminal = json.loads(path.read_text())
    terminal['final_viewpoint_paths'][-1]['distance'] = terminal['final_distance'] = .25000001
    path.write_text(json.dumps(terminal) + '\n')
    path = tmp_path / 'evaluation/reference.json'
    reference = json.loads(path.read_text())
    reference['viewpoint_paths'][-1]['distance'] = reference['reference_distance'] = 2.0000001
    path.write_text(json.dumps(reference))
    score = score_habitat_evaluation(tmp_path)
    assert score['distance_to_goal'] == .24999999
    assert score['reference_distance'] == 2
    assert score['SR'] == 1


@pytest.mark.parametrize('metadata', ['episode_info', 'input_provenance', 'navmesh_settings'])
@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999'])
def test_language_nonfinite_reference_metadata_rejected_before_availability(
        tmp_path, evaluation_env, metadata, token):
    """Nonfinite provenance must fail parsing, not leak into an available, unserializable score."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation

    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    trace.step(0)
    path = tmp_path / 'evaluation/reference.json'
    reference = json.loads(path.read_text())
    target = reference['navmesh']['settings'] if metadata == 'navmesh_settings' else reference[metadata]
    target['nested'] = {'value': 'NONFINITE'}
    path.write_text(json.dumps(reference).replace('"NONFINITE"', token))
    instances = len(type(trace.pathfinder).instances)
    with pytest.raises(ValueError, match='finite'):
        score_habitat_evaluation(tmp_path)
    assert len(type(trace.pathfinder).instances) == instances


def test_language_evaluator_module_cli_emits_strict_json(tmp_path, evaluation_env, monkeypatch, capsys):
    """The module CLI performs offline mesh scoring with a single JSON stdout result."""
    import runpy
    import sys
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, strict_json_loads

    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    evaluation_env.moves = [[2, 1, 0]]
    trace.step(0)
    monkeypatch.delitem(sys.modules, 'benchmark.habitat_evaluation')
    monkeypatch.setattr(sys, 'argv', ['benchmark.habitat_evaluation', str(tmp_path)])
    runpy.run_module('benchmark.habitat_evaluation', run_name='__main__')
    result = strict_json_loads(capsys.readouterr().out)
    assert result['status'] == 'available' and result['SR'] == result['SPL'] == 1
    assert result['provenance']['evaluation_directory'] == str(tmp_path / 'evaluation')
    assert evaluation_env.actions == [0]


@pytest.mark.parametrize('benchmark', ['goat_language', 'hm3d_objectnav', 'langmap'])
def test_language_recorder_attaches_only_at_language_episode_boundary(tmp_path, evaluation_env, benchmark):
    """Exercise the isolated entrypoint hook without importing navigation or models."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    from backends.habitat.motion import HabitatExecution

    source = Path(__file__).resolve().parents[1] / 'objnav_benchmark_with_process_obs.py'
    tree = ast.parse(source.read_text())
    hook = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                and any(isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Attribute) and target.attr == 'evaluation_trace'
                    for target in child.targets) for child in node.body))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    reset = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'reset')
    install = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == '_install_instruction_plan')
    identity_write = next(node for node in calls if isinstance(node.func, ast.Attribute)
                          and node.func.attr == 'dump' and node.args
                          and isinstance(node.args[0], ast.Name) and node.args[0].id == 'actual_episode')
    assert reset.lineno < identity_write.lineno < hook.lineno < install.lineno
    agent = NS(env=evaluation_env, execution=HabitatExecution(evaluation_env))
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), 'exec'),
         {'benchmark_spec': NS(benchmark=benchmark, provenance={}),
          'habitat_agent': agent, 'episode_dir': str(tmp_path)})
    assert (agent.execution.controller.evaluation_trace is not None) == (benchmark == 'goat_language')
    assert (tmp_path / 'evaluation').exists() == (benchmark == 'goat_language')
    assert evaluation_env.actions == []


def test_language_reference_ignores_only_unreachable_viewpoints(tmp_path, evaluation_env):
    """An unreachable view must not hide reachable views in other goal sets."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    evaluation_env.sim.pathfinder.distances[((0, 1, 0), (8, 1, 0))] = float('inf')
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    reference = json.loads((tmp_path / 'evaluation/reference.json').read_text())
    assert [path['distance'] for path in reference['viewpoint_paths']] == [None, 9, 2]
    assert reference['reference_distance'] == 2
    evaluation_env.moves = [[2, 1, 0]]
    trace.step(0)
    assert score_habitat_evaluation(tmp_path)['SPL'] == 1


@pytest.mark.parametrize('distance,success', [(.25, 0), (.24999999999999997, 1),
                                             (.25000000000000006, 0), (.2500000298, 0)])
def test_language_independent_strict_distance_boundary(tmp_path, evaluation_env, distance, success):
    """Preserve raw floating point distances without rounding the strict threshold."""
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    evaluation_env.moves = [[2, 1, distance]]
    trace.step(0)
    score = score_habitat_evaluation(tmp_path)
    assert score['distance_to_goal'] == distance
    assert score['SR'] == success
    assert (score['SPL'] > 0) == bool(success)


@pytest.mark.parametrize('movement,spl', [(0, 1), (.2, 0)])
def test_language_independent_zero_reference_distance(tmp_path, evaluation_env, movement, spl):
    """A zero-length reference handles both stationary and moving successful STOP."""
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    env = evaluation_env
    env.current_episode.goals[-1].view_points[0].agent_state.position = list(env.position)
    trace = HabitatEvaluationTrace(env, tmp_path)
    env.moves = [[0, 1, movement]]
    trace.step(0)
    score = score_habitat_evaluation(tmp_path)
    assert score['reference_distance'] == 0 and score['path_length'] == movement
    assert score['SR'] == 1 and score['SPL'] == spl


@pytest.mark.parametrize('actions', [[], [1]])
def test_language_independent_requires_real_terminal_stop(tmp_path, evaluation_env, actions):
    """Timeout or policy completion cannot substitute for an actual STOP action."""
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    for action in actions:
        trace.step(action)
    evaluation_env.episode_over = True
    evaluation_env.task_complete = True
    score = score_habitat_evaluation(tmp_path)
    assert score['status'] == 'unavailable' and score['reason'] == 'missing_terminal_STOP'
    assert 'SR' not in score and 'SPL' not in score
    assert evaluation_env.actions == actions


@pytest.mark.parametrize('missing', ['reference.json', 'trajectory.jsonl', 'runtime.navmesh', 'final_distance'])
def test_language_independent_missing_evidence_unavailable(tmp_path, evaluation_env, missing):
    """Missing evidence never becomes a fabricated zero or success metric."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    trace.step(0)
    if missing == 'final_distance':
        path = tmp_path / 'evaluation/trajectory.jsonl'
        record = json.loads(path.read_text())
        del record['final_distance']
        path.write_text(json.dumps(record) + '\n')
    else:
        (tmp_path / 'evaluation' / missing).unlink()
    score = score_habitat_evaluation(tmp_path)
    assert score['status'] == 'unavailable'
    assert 'SR' not in score and 'SPL' not in score


@pytest.mark.parametrize('broken', ['first_step', 'step_gap', 'initial', 'continuity', 'hash',
                                   'after_stop', 'nonfinite_position', 'minimum', 'viewpoints', 'identity'])
def test_language_independent_rejects_corrupt_evidence(tmp_path, evaluation_env, broken):
    """Reject altered mesh, incomplete GT and discontinuous or nonfinite base traces."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    trace.step(1)
    trace.step(0)
    path = tmp_path / 'evaluation/trajectory.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if broken == 'first_step':
        rows[0]['step'] = 1
    elif broken == 'step_gap':
        rows[1]['step'] = 2
    elif broken == 'initial':
        rows[0]['before'][0] += 1
    elif broken == 'continuity':
        rows[1]['before'][0] += 1
    elif broken == 'hash':
        (tmp_path / 'evaluation/runtime.navmesh').write_bytes(b'different mesh')
    elif broken == 'after_stop':
        rows.append(dict(rows[-1], step=2))
    elif broken == 'nonfinite_position':
        rows[-1]['after'][1] = float('nan')
    elif broken == 'minimum':
        rows[-1]['final_distance'] = .1
    elif broken == 'viewpoints':
        rows[-1]['final_viewpoint_paths'].pop()
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        score_habitat_evaluation(tmp_path, expected_identity={} if broken == 'identity' else None)


@pytest.mark.parametrize('phase', ['initial', 'stop'])
def test_language_unreachable_viewpoints_use_null_not_nan(tmp_path, evaluation_env, phase):
    """Explicit unreachable path queries stay null, not nonfinite JSON numbers."""
    import json
    from benchmark.habitat_evaluation import HabitatEvaluationTrace, score_habitat_evaluation
    env = evaluation_env
    stop_position = list(env.position) if phase == 'initial' else [2, 1, 0]
    for goal in env.current_episode.goals:
        for view in goal.view_points:
            env.sim.pathfinder.distances[(tuple(stop_position), tuple(view.agent_state.position))] = float('inf')
    trace = HabitatEvaluationTrace(env, tmp_path)
    env.moves = [stop_position]
    trace.step(0)
    record = json.loads((tmp_path / 'evaluation/trajectory.jsonl').read_text())
    assert record['final_distance'] is None
    assert all(path['distance'] is None and path['path_points'] == [] for path in record['final_viewpoint_paths'])
    score = score_habitat_evaluation(tmp_path)
    if phase == 'initial':
        assert score['status'] == 'unavailable' and 'SPL' not in score
    else:
        assert score['status'] == 'available' and score['SR'] == score['SPL'] == 0


@pytest.mark.parametrize('failure', ['save', 'simulator', 'query', 'append', 'nonfinite_query', 'nonfinite_base'])
def test_language_trace_propagates_io_and_simulation_errors(tmp_path, evaluation_env, monkeypatch, failure):
    """Errors remain visible without retrying steps or fabricating terminal evidence."""
    from pathlib import Path
    from benchmark.habitat_evaluation import HabitatEvaluationTrace
    env = evaluation_env
    def fail(*args, **kwargs):
        raise OSError('injected failure')
    if failure == 'save':
        monkeypatch.setattr(env.sim.pathfinder, 'save_nav_mesh', fail)
        with pytest.raises(OSError, match='injected failure'):
            HabitatEvaluationTrace(env, tmp_path)
        assert env.actions == []
        return
    trace = HabitatEvaluationTrace(env, tmp_path)
    if failure == 'simulator':
        monkeypatch.setattr(env, 'step', fail)
    elif failure == 'query':
        monkeypatch.setattr(trace.pathfinder, 'find_path', fail)
    elif failure == 'append':
        original_open = Path.open
        def open_path(path, mode='r', *args, **kwargs):
            if path.name == 'trajectory.jsonl' and mode == 'a':
                fail()
            return original_open(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', open_path)
    elif failure == 'nonfinite_query':
        trace.pathfinder.distances[((0, 1, 0), (8, 1, 0))] = float('nan')
    elif failure == 'nonfinite_base':
        env.position[0] = float('nan')
    with pytest.raises((OSError, ValueError)):
        trace.step(0)
    assert env.actions == ([] if failure in ('simulator', 'nonfinite_base') else [0])
    assert (tmp_path / 'evaluation/trajectory.jsonl').read_text() == ''


@pytest.mark.parametrize('phase', ['initial', 'between', 'after_stop'])
def test_language_recorder_rejects_discontinuity_without_extra_steps(tmp_path, evaluation_env, phase):
    """Unrecorded motion or a second STOP cannot be silently repaired."""
    from benchmark.habitat_evaluation import HabitatEvaluationTrace
    trace = HabitatEvaluationTrace(evaluation_env, tmp_path)
    if phase == 'between':
        trace.step(1)
    elif phase == 'after_stop':
        trace.step(0)
    evaluation_env.position[0] += 1
    previous_actions = list(evaluation_env.actions)
    with pytest.raises(ValueError):
        trace.step(0)
    assert evaluation_env.actions == previous_actions
