"""Offline artifact audit; STRIVE must be on PYTHONPATH. Never starts a simulator."""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


TOL = 1e-5
MODALITIES = ('object', 'image', 'description')
ACTIONS = ('STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    def invalid(value):
        raise ValueError(f'{path}: non-standard JSON constant {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_sha(path, expected, report):
    actual = _sha(path)
    _require(actual == expected, f'SHA256 mismatch: {path}; expected {expected}, got {actual}')
    report['sha256'][str(path)] = actual
    return actual


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f'{label}: invalid integer {value!r}')
    return value


def _number(value, label):
    _require(not isinstance(value, (bool, str)) and value is not None,
             f'{label}: missing/non-numeric value {value!r}')
    value = float(value)
    _require(math.isfinite(value), f'{label}: nonfinite value {value}; no fallback permitted')
    return value


def _pose(value, label):
    position = np.asarray(value['position'], dtype=float)
    rotation = np.asarray(value['rotation_xyzw'], dtype=float)
    _require(position.shape == (3,) and rotation.shape == (4,), f'{label}: invalid pose shape')
    _require(np.isfinite(position).all() and np.isfinite(rotation).all(), f'{label}: nonfinite pose')
    _require(abs(float(np.linalg.norm(rotation)) - 1) < 1e-4, f'{label}: quaternion is not unit length')
    return position, rotation


def _same_pose(left, right, label):
    p, q = _pose(left, label)
    r, s = _pose(right, label)
    _require(np.allclose(p, r, rtol=0, atol=TOL), f'{label}: position discontinuity')
    _require(np.allclose(q, s, rtol=0, atol=TOL) or np.allclose(q, -s, rtol=0, atol=TOL),
             f'{label}: rotation discontinuity')


def _compare(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and actual.keys() == expected.keys(), f'{label}: metric keys differ')
        for key in expected:
            _compare(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f'{label}: lengths differ')
        for index, (a, b) in enumerate(zip(actual, expected)):
            _compare(a, b, f'{label}[{index}]')
    elif isinstance(expected, bool):
        _require(type(actual) is bool and actual == expected, f'{label}: {actual!r} != {expected!r}')
    else:
        a, b = _number(actual, label), _number(expected, label)
        _require(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f'{label}: recorded {a} != oracle {b}')


def _image_dimensions(path):
    from PIL import Image

    with Image.open(path) as image:
        _require(image.mode == 'RGB' and image.format in ('PNG', 'JPEG'), f'{path}: expected RGB PNG/JPEG')
        dimensions, image_format = [image.height, image.width], image.format
        image.verify()
        return dimensions, image_format


def _images(directory, episode, records, report):
    expected = {}
    for index, subtask in enumerate(episode.subtasks):
        if subtask.public_goal.modality == 'image':
            ref = subtask.public_goal.image_ref
            camera = {key: subtask.image_parameters[key]
                      for key in ('position', 'rotation', 'hfov', 'image_dimensions')}
            if ref in expected:
                _require(expected[ref]['camera'] == camera, f'Conflicting camera for image reference {ref}')
                expected[ref]['indices'].append(index)
            else:
                expected[ref] = dict(camera=camera, indices=[index])
    _require(len(records) == len(expected), 'Image receipt count differs from unique source image references')
    seen = set()
    report['images'] = []
    for record in records:
        ref = record['image_ref']
        _require(ref in expected and ref not in seen, f'Unexpected/duplicate image receipt: {ref}')
        seen.add(ref)
        camera = expected[ref]['camera']
        _require(record['image_parameters'] == camera, f'{ref}: original camera differs from source')
        receipt_path = Path(record['path'])
        _require(receipt_path.parent.name == 'reference_images' and receipt_path.stem == ref.split(':')[-1],
                 f'{ref}: unexpected reference image path')
        path = directory / 'reference_images' / receipt_path.name
        digest = _verify_sha(path, record['sha256'], report)
        dimensions, image_format = _image_dimensions(path)
        _require(dimensions == camera['image_dimensions'], f'{ref}: pixel dimensions differ from camera')
        if record.get('mode') == 'explicit_pre_rendered_asset_no_live_pose_or_sensor_mutation':
            restoration = 'not_applicable_pre_rendered_asset; no live sensor receipt'
        else:
            sensor = record['actual_sensor']
            _require(sensor['uuid'] == 'goat_reference' and sensor['sensor_type'].endswith('COLOR')
                     and sensor['sensor_subtype'].endswith('PINHOLE'), f'{ref}: wrong actual sensor')
            _require(sensor['resolution'] == dimensions, f'{ref}: actual sensor resolution differs')
            _require(math.isclose(_number(sensor['hfov'], ref), _number(camera['hfov'], ref),
                                 rel_tol=0, abs_tol=TOL), f'{ref}: actual HFOV differs')
            _same_pose(dict(position=sensor['position'], rotation_xyzw=sensor['rotation']),
                       dict(position=camera['position'], rotation_xyzw=camera['rotation']), f'{ref}: camera')
            _require(record['pose_restored'] is True, f'{ref}: missing/failed restoration receipt')
            restoration = 'pose_restored=true receipt; before/after sensor poses are not logged'
        report['images'].append(dict(image_ref=ref, subtask_indices=expected[ref]['indices'],
                                     path=str(path), sha256=digest, dimensions_hw=dimensions,
                                     format=image_format, camera_parameters_match=True,
                                     restoration=restoration))


def _lifecycle(goat_module, directory, result, report):
    root = Path(goat_module.__file__).resolve().parents[1]
    environment = root / 'benchmark/goat_environment.py'
    runtime = root / 'backends/habitat/goat_runtime.py'
    tree = ast.parse(environment.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GOATEnvironment')
    reset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'reset')
    body = reset.body[1:] if ast.get_docstring(reset) is not None else reset.body
    guard = ast.parse('if self._reset_called:\n    raise RuntimeError("Continuous GOAT permits only one parent reset")').body[0]
    assignment = ast.parse('self._reset_called = True').body[0]
    increment = ast.parse('self.parent_reset_count += 1').body[0]
    base_reset = ast.parse('self._observation = self._base.reset()').body[0]
    _require(len(body) >= 4 and all(ast.dump(a) == ast.dump(b)
                                  for a, b in zip(body[:4], (guard, assignment, increment, base_reset))),
             'GOATEnvironment.reset one-parent guard/counter differs from audited code')
    runtime_tree = ast.parse(runtime.read_text(encoding='utf-8'))
    driver = next(node for node in runtime_tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'run_continuous_parent')
    resets = [node for node in ast.walk(driver) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == 'agent' and node.func.attr == 'reset']
    _require(len(resets) == 1 and not any(resets[0] in list(ast.walk(node))
                                        for node in ast.walk(driver) if isinstance(node, (ast.For, ast.While))),
             'Continuous driver must call agent.reset once outside subtask loops')
    paths = (Path(goat_module.__file__).resolve(), environment, runtime)
    receipt = directory.parent / 'execution.json'
    bound = receipt.is_file()
    recorded = _json(receipt)['source_sha256'] if bound else {}
    fingerprints = {}
    for path in paths:
        key = path.relative_to(root).as_posix()
        fingerprints[key] = _verify_sha(path, recorded[key], report) if bound else _sha(path)
    parent_reset_count = _integer(result['parent_reset_count'], 'parent_reset_count')
    _require(parent_reset_count == 1, 'Parent must reset exactly once')
    switches = []
    for index, row in enumerate(result['subtasks']):
        # Unvisited rows and global-budget segments without installation have no receipt.
        if row['start_pose'] is None or 'goal_switch' not in row:
            continue
        switch = row['goal_switch']
        label = f'Subtask {index}: goal_switch'
        _require(switch['mapper_preserved'] is True and switch['nodes_preserved'] is True,
                 f'{label}: mapper/nodes identity was not preserved')
        before = _integer(switch['map_updates_before'], f'{label}.map_updates_before')
        after = _integer(switch['map_updates_after'], f'{label}.map_updates_after')
        _require(before == after, f'{label}: goal installation changed map updates')
        _integer(switch['node_count'], f'{label}.node_count')
        _same_pose(switch['pose_after_install'], row['start_pose'], f'{label}: installation -> start')
        switches.append(dict(index=index, **switch, pose_matches_start=True))
    images = {item['image_ref']: item for item in report['images']}
    consumed = {}
    for response_path in sorted((directory / 'lvlm_calls').glob('*_goat_reference_image.json')):
        response = _json(response_path)
        request_path = response_path.parent / f"request_{response['metadata']['request_id']}.json"
        request = _json(request_path)
        _require('qwen' in request['model'].lower(), 'Image goal was not sent to configured Qwen')
        for message in request['messages']:
            if not isinstance(message['content'], list):
                continue
            for block in message['content']:
                if block['type'] == 'image_url':
                    uri = block['image_url']['url']
                    _require(uri.startswith('data:image/'), 'Image request lacks retained pixels')
                    digest = hashlib.sha256(base64.b64decode(uri.split(',', 1)[1], validate=True)).hexdigest()
                    consumed.setdefault(digest, []).append(dict(
                        request_id=request['request_id'], model=request['model'],
                        request_sha256=_sha(request_path), response_sha256=_sha(response_path)))
    bindings = []
    for switch in switches:
        index = switch['index']
        task = _json(directory / f'subtask-{index:03d}' / 'instruction_adapter/task_spec.json')
        _require(task['revision'] == index + 1, f'Subtask {index}: installed revision differs')
        goal = result['public_goals'][index]
        if goal['modality'] != 'image':
            continue
        reference = images[goal['image_ref']]
        targets = [target for target in task['targets'] if target['terminal']]
        _require(len(targets) == 1 and task['source'] == 'goat_image', 'Image task is not a structured public goal')
        path = Path(targets[0]['attributes']['goal_reference_image'])
        _require(path.name == Path(reference['path']).name, f'Subtask {index}: wrong reference binding')
        _require(reference['sha256'] in consumed, f'Subtask {index}: reference pixels were not consumed')
        bindings.append(dict(index=index, revision=task['revision'], image_ref=goal['image_ref'],
                             sha256=reference['sha256'], requests=consumed[reference['sha256']]))
    report['image_goal_bindings'] = bindings
    report['lifecycle'] = dict(
        reset_guard_source=str(environment), reset_guard_line=reset.lineno,
        second_env_reset_raises=True, driver_reset_call_sites=1,
        inspected_source_sha256=fingerprints, execution_source_receipt_verified=bound,
        parent_reset_count=parent_reset_count, goal_switch_receipts_verified=switches,
        continuity_scope='runtime reset/map receipts, logged action poses and same-pose switches; '
                         'not simulator dynamics replay')


class _Measure:
    def __init__(self, **kwargs):
        self.uuid, self._metric = self._get_uuid(), None

    def get_metric(self):
        return self._metric


class _TaskAction:
    def __init__(self, sim):
        self._sim = sim


def _upstream(root, pins, report):
    scope = dict(np=np, defaultdict=defaultdict, Measure=_Measure,
                 NavigationTask=object, SimulatorTaskAction=_TaskAction)
    for filename, wanted in (
        ('goat_task.py', {'GoatTask', 'SubtaskStopAction'}),
        ('nav.py', {'GoatDistanceToGoal', 'GoatSuccess', 'GoatSPL'}),
    ):
        path = root / filename
        _verify_sha(path, pins[filename], report)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in wanted]
        _require({node.name for node in classes} == wanted, f'{path}: oracle classes missing')
        for node in classes:
            node.decorator_list = []
        module = ast.Module(body=ast.parse('from __future__ import annotations').body + classes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope


class _Oracle:
    def __init__(self, classes, episode, position, pathfinder, habitat_sim):
        self.position = np.asarray(position, dtype=float)
        self.pathfinder, self.habitat_sim = pathfinder, habitat_sim
        self.endpoints = [np.asarray(s.view_positions, dtype=np.float32) for s in episode.subtasks]
        for index, points in enumerate(self.endpoints):
            _require(points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == 3
                     and np.isfinite(points).all(), f'Subtask {index}: no valid view_positions union')
        self.queries = 0
        sim = SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=self.position.copy()),
                              geodesic_distance=self._distance, get_observations_at=lambda: {})
        self.episode = SimpleNamespace(tasks=[s.source_task for s in episode.subtasks],
                                       goals=[s.evaluator_goals for s in episode.subtasks],
                                       scene_id=episode.scene_id, episode_id=episode.source_episode_id)
        self.task = classes['GoatTask']()
        self.task.is_stop_called = False
        self.stop_action = classes['SubtaskStopAction'](sim)
        self.stop_action.reset(self.task)
        config = SimpleNamespace(success_distance=0.25, distance_to='VIEW_POINTS')
        self.measures = {classes[name].cls_uuid: classes[name](sim, config)
                         for name in ('GoatDistanceToGoal', 'GoatSuccess', 'GoatSPL')}
        self.task.measurements = SimpleNamespace(measures=self.measures, check_measure_dependencies=lambda *args: None)
        for measure in self.measures.values():
            measure.reset_metric(self.episode, self.task)

    def _distance(self, position, viewpoints, episode):
        index = self.task.active_subtask_idx
        endpoints = self.endpoints[index]
        _require(np.array_equal(np.asarray(viewpoints, dtype=np.float32), endpoints),
                 f'Subtask {index}: upstream endpoints differ from parsed view_positions')
        query = self.habitat_sim.MultiGoalShortestPath()
        query.requested_start = np.asarray(position, dtype=np.float32)
        query.requested_ends = endpoints
        found = self.pathfinder.find_path(query)
        distance = float(query.geodesic_distance)
        self.queries += 1
        _require(found and math.isfinite(distance) and distance >= 0,
                 f'Subtask {index}, query {self.queries}: unreachable/invalid geodesic ({distance}); no fallback')
        return distance

    def step(self, event):
        self.position = np.asarray(event['after']['position'], dtype=float)
        switched = event['action'] == 'SUBTASK_STOP'
        if switched:
            self.stop_action.step(self.task)
        self.task.last_action = self.stop_action if switched else None
        self.task.is_stop_called = event['action'] == 'STOP'
        for measure in self.measures.values():
            measure.update_metric(self.episode, self.task)

    def distance(self):
        return float(self.measures['distance_to_goal'].get_metric()['distance_to_target'])

    def scores(self):
        metrics = {**self.measures['success'].get_metric(), **self.measures['spl'].get_metric()}
        return {key: ([float(v) for v in value] if isinstance(value, list)
                      else bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
                for key, value in metrics.items()}


def _replay(directory, result, episode, oracle, report):
    rows, tasks = result['subtasks'], episode.subtasks
    _require(len(rows) == len(tasks), 'Subtasks were dropped or deduplicated')
    for index, (row, task) in enumerate(zip(rows, tasks)):
        _require(row['index'] == index and row['source_subtask_index'] == task.source_subtask_index
                 and row['revision'] == index + 1, f'Subtask {index}: source order/revision differs')
    budget = result['budgets']
    global_limit = _integer(budget['global_budget'], 'global_budget', 1)
    segment_limit = _integer(budget['max_steps_per_subtask'], 'max_steps_per_subtask', 1)
    previous = dict(position=list(episode.start_position), rotation_xyzw=list(episode.start_rotation))
    active = segment_steps = total = 0
    ended = False
    counts, modal = Counter(), {modality: Counter() for modality in MODALITIES}
    path_length = segment_path = 0.0
    summaries = []
    with (directory / 'actions.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            _require(line.strip(), f'actions.jsonl:{line_number}: empty record')
            event = json.loads(line)
            label = f'action {line_number}'
            _require(not ended and active < len(tasks), f'{label}: action after parent termination')
            _require('error' not in event, f'{label}: recorded action error')
            action = event['action']
            _require(action in (*ACTIONS, 'SUBTASK_STOP'), f'{label}: unsupported action {action}')
            total += 1
            segment_steps += 1
            _require(_integer(event['step'], label, 1) == total
                     and _integer(event['subtask_index'], label) == active
                     and _integer(event['segment_step'], label, 1) == segment_steps,
                     f'{label}: broken step/subtask/segment sequence')
            _require(total <= global_limit and segment_steps <= segment_limit, f'{label}: budget exceeded')
            if action not in ('STOP', 'SUBTASK_STOP'):
                _require(total < global_limit and segment_steps < segment_limit, f'{label}: consumed reserved stop slot')
            if action == 'SUBTASK_STOP':
                _require(total < global_limit, f'{label}: global boundary requires STOP, not SUBTASK_STOP')
            _same_pose(previous, event['before'], f'{label}: previous after -> before')
            before, _ = _pose(event['before'], label)
            after, _ = _pose(event['after'], label)
            delta = float(np.linalg.norm(after - before))
            logged_delta = _number(event['path_delta'], f'{label}: path_delta')
            _require(logged_delta >= 0 and math.isclose(delta, logged_delta, rel_tol=1e-6, abs_tol=1e-7),
                     f'{label}: path_delta does not equal actual 3D displacement')
            if action != 'MOVE_FORWARD' and delta > TOL:
                report.setdefault('nonforward_displacements', []).append(
                    dict(step=total, action=action, path_delta=delta, included_in_path=True))
            if action not in ('TURN_LEFT', 'TURN_RIGHT'):
                _same_pose(dict(position=event['after']['position'], rotation_xyzw=event['before']['rotation_xyzw']),
                           event['after'], f'{label}: unexpected base rotation')
            row = rows[active]
            if segment_steps == 1:
                _require(row['start_action'] == total - 1, f'{label}: wrong segment start action')
                _same_pose(row['start_pose'], event['before'], f'{label}: segment start')
                summaries.append(dict(index=active, source_subtask_index=tasks[active].source_subtask_index,
                                      modality=tasks[active].public_goal.modality, start_action=total - 1,
                                      start_geodesic=oracle.distance()))
            terminal = action in ('STOP', 'SUBTASK_STOP')
            if terminal:
                _same_pose(event['before'], event['after'], f'{label}: stop must retain full pose')
                _require(delta == 0, f'{label}: stop moved')
                _require(row['end_action'] == total and row['stop'] == action, f'{label}: segment end action/stop differs')
                _same_pose(row['end_pose'], event['after'], f'{label}: segment end')
                if action == 'SUBTASK_STOP':
                    _require(isinstance(row['reason'], str) and row['reason'] == event['reason'], f'{label}: stop reason differs')
                    if row['reason'] == 'subtask_budget':
                        _require(segment_steps == segment_limit, f'{label}: premature subtask_budget')
                else:
                    _require(row['reason'] in ('global_budget', 'policy_global_stop'), f'{label}: invalid global STOP reason')
                    if row['reason'] == 'global_budget':
                        _require(total == global_limit, f'{label}: premature global_budget')
                _require(_json(directory / f'subtask-{active:03d}' / 'segment.json') == row,
                         f'{label}: segment.json differs from result.json')
                summaries[-1]['stop_old_goal_geodesic'] = oracle.distance()
            path_length += delta
            segment_path += delta
            counts[action] += 1
            modal[tasks[active].public_goal.modality][action] += 1
            oracle.step(event)
            scores = oracle.scores()
            _require(all(math.isfinite(float(item)) for value in scores.values()
                         for item in (value if isinstance(value, list) else [value])),
                     f'{label}: undefined upstream score; no zero substitution permitted')
            if terminal:
                summaries[-1].update(end_action=total, action_count=segment_steps, path_length=segment_path,
                                     stop=action, reason=row['reason'], success=scores['subtask_success'][active],
                                     spl=scores['spl_by_subtask'][active])
                if action == 'SUBTASK_STOP':
                    active += 1
                    segment_steps, segment_path = 0, 0.0
                ended = action == 'STOP' or active == len(tasks)
            _require(oracle.task.active_subtask_idx == active, f'{label}: upstream subtask advance differs')
            previous = event['after']
    _require(total > 0 and ended, 'Action log is empty or truncated before terminal stop')
    _require(_integer(result['actual_action_count'], 'actual_action_count', 1) == total, 'Actual action count differs')
    for index in range(len(summaries), len(rows)):
        row = rows[index]
        _require(row['reason'] == 'not_executed' and all(row[key] is None
                 for key in ('start_pose', 'end_pose', 'start_action', 'end_action', 'stop')),
                 f'Subtask {index}: unvisited row claims execution')
    expected_reason = summaries[-1]['reason'] if counts['STOP'] else 'final_subtask_stop'
    _require(result['termination_reason'] == expected_reason, 'Parent termination reason differs')
    report['actions'] = dict(total=total, by_action={key: counts[key] for key in (*ACTIONS, 'SUBTASK_STOP')},
                             by_modality={key: dict(value) for key, value in modal.items()},
                             actual_path_length=path_length, budgets=budget, budgets_verified=True,
                             geodesic_queries=oracle.queries, upstream_updates=total)
    report['subtasks'] = summaries
    report['lifecycle'].update(logged_pose_chain_continuous=True, same_pose_subtask_switches=counts['SUBTASK_STOP'])
    upstream = oracle.scores()
    derived = {}
    for modality in MODALITIES:
        values = [upstream['spl_by_subtask'][i] for i, task in enumerate(tasks) if task.public_goal.modality == modality]
        derived[f'{modality}_spl'] = sum(values) / len(values) if values else 0.0
    report['scores'] = dict(recorded=result['metrics'], upstream=upstream, derived_modal_spl=derived,
                            modal_spl_definition='mean upstream spl_by_subtask; absent modality=0 by protocol')
    _compare(result['metrics'], {**upstream, **derived}, 'result.metrics')
    report['scores']['match'] = True


def _audit(args, report):
    global np
    sys.dont_write_bytecode = True
    import numpy as np
    import habitat_sim
    import benchmark.goat as goat

    directory = args.episode_dir.resolve(strict=True)
    result_path = directory / 'result.json'
    result = _json(result_path)
    report['sha256'][str(result_path)] = _sha(result_path)
    _require(result['status'] == 'complete', f"Episode is not complete: {result['status']}; {result.get('error')}")
    _require(result.get('aggregate_ready') is True, f"Scores are unavailable: {result.get('unavailable_metrics')}")
    _verify_sha(directory / 'actions.jsonl', result['actions_sha256'], report)
    source = (args.source_shard or Path(result['source_file'])).resolve(strict=True)
    _verify_sha(source, result['source_sha256'], report)
    episodes = goat.load_goat_episodes(source, split=result['split'])
    parent = _integer(result['parent_index'], 'parent_index')
    _require(parent < len(episodes), 'parent_index outside source shard')
    episode = episodes[parent]
    _require(episode.subtasks, 'Source parent has no retained subtasks')
    _require(episode.source_episode_id == result['source_episode_id'], 'Source parent episode ID differs')
    _require(list(episode.filtered_subtask_indices) == result['filtered_subtask_indices'], 'Filtered source indices differ')
    _require([asdict(s.public_goal) for s in episode.subtasks] == result['public_goals'], 'Public goal sequence differs (including repetitions)')
    repeated_tasks, repeated_targets = defaultdict(list), defaultdict(list)
    for subtask in episode.subtasks:
        index = subtask.source_subtask_index
        _require(tuple(episode.source_episode['tasks'][index]) == subtask.source_task, 'Source task identity changed')
        repeated_tasks[json.dumps(subtask.source_task)].append(index)
        repeated_targets[json.dumps((subtask.source_task[0], subtask.object_id))].append(index)
    _require([s.source_subtask_index for s in episode.subtasks] == [
        i for i in range(len(episode.source_episode['tasks'])) if i not in episode.filtered_subtask_indices],
        'Source sequence was reordered or deduplicated')
    report['source'] = dict(source_file_recorded=result['source_file'], source_file_used=str(source),
                            split=result['split'], parent_index=parent, source_episode_id=episode.source_episode_id,
                            source_subtask_indices=[s.source_subtask_index for s in episode.subtasks],
                            filtered_subtask_indices=list(episode.filtered_subtask_indices),
                            repeated_task_source_indices=[v for v in repeated_tasks.values() if len(v) > 1],
                            repeated_target_source_indices=[v for v in repeated_targets.values() if len(v) > 1],
                            subtasks_by_modality=dict(Counter(s.public_goal.modality for s in episode.subtasks)))
    resources = result['resources']
    _require(result['upstream_revision'] == resources['upstream_revision'] == goat.UPSTREAM_REVISION, 'Upstream revision differs')
    _require(resources['config_sha256'] == goat.UPSTREAM_SHA256['goat_stretch_hm3d.yaml'], 'Stretch config hash differs')
    _require(resources['source_scene_id'] == episode.scene_id, 'Scene identity differs from source parent')
    for name in ('scene', 'scene_dataset'):
        _verify_sha(Path(resources[name]).resolve(strict=True), resources[f'{name}_sha256'], report)
    _compare(resources['profile'], dict(height=1.41, radius=0.17, camera_position=[0, 1.31, 0],
                                       hfov=42, image_dimensions=[640, 360], turn_angle=30, tilt_angle=30,
                                       forward_step_size=0.25, allow_sliding=False), 'Stretch profile')
    nav = resources['navmesh']
    _require(nav['recomputed'] is True and nav['is_loaded'] is True and nav['start_navigable'] is True
             and nav['include_static_objects'] is False, 'Navmesh preparation receipt failed')
    for key, expected in dict(agent_height=1.41, agent_radius=0.17, agent_max_climb=0.20, cell_height=0.20).items():
        _compare(nav['settings'][key], expected, f'navmesh.settings.{key}')
    _require(nav['actions'] == [dict(id=i, name=ACTIONS[i].lower(), amount=0.25 if i == 1 else 30)
                                for i in range(1, 6)], 'Effective simulator action receipt differs')
    _require(Path(nav['path']).name == 'stretch.navmesh', 'Unexpected saved navmesh filename')
    nav_path = directory / 'stretch.navmesh'
    _verify_sha(nav_path, nav['sha256'], report)
    pathfinder = habitat_sim.PathFinder()
    _require(pathfinder.load_nav_mesh(str(nav_path)) and pathfinder.is_loaded, 'Cannot load saved Stretch navmesh')
    _require(pathfinder.is_navigable(episode.start_position), 'Parent start is not navigable on saved navmesh')
    report['navmesh'] = dict(path_recorded=nav['path'], path_used=str(nav_path),
                             saved_habitat_sim_version=nav['habitat_sim_version'],
                             audit_habitat_sim_version=habitat_sim.__version__, recomputed_by_audit=False)
    _images(directory, episode, result['images'], report)
    _lifecycle(goat, directory, result, report)
    classes = _upstream(args.upstream_root.resolve(strict=True), goat.UPSTREAM_SHA256, report)
    with (directory / 'actions.jsonl').open(encoding='utf-8') as stream:
        first = json.loads(stream.readline())
    _same_pose(first['before'], dict(position=episode.start_position, rotation_xyzw=episode.start_rotation),
               'Actual initial pose -> source parent start')
    oracle = _Oracle(classes, episode, first['before']['position'], pathfinder, habitat_sim)
    _replay(directory, result, episode, oracle, report)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-dir', type=Path, required=True)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New audit JSON; existing paths are never overwritten')
    parser.add_argument('--source-shard', type=Path, help='Explicit local replacement for result.source_file (same SHA256)')
    args = parser.parse_args()
    output = args.output.absolute()
    report = dict(status='failed', scope='offline real-artifact audit; no simulator, policy or remote calls',
                  episode_dir=str(args.episode_dir.resolve()), sha256={},
                  tolerances=dict(pose_absolute=TOL, score_absolute=1e-6, score_relative=1e-6),
                  evidence_limits=['Reset/map checks verify runtime receipts, not simulator dynamics replay.',
                                   'Image checks verify headers, hashes and camera/restoration receipts, not rendering.',
                                   'Ground truth is used only in this private offline audit; no policy is called.'])
    try:
        _require(output.suffix.lower() == '.json', '--output must be a new .json path')
        with output.open('x', encoding='utf-8') as stream:
            try:
                _audit(args, report)
                report['status'] = 'passed'
            except Exception as exc:
                report['error'] = dict(type=type(exc).__name__, message=str(exc))
                print(f'AUDIT ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except Exception as exc:
        print(f'AUDIT ERROR: cannot create audit JSON {output}: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f"{report['status']}: {output}")
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(_main())
