"""One-parent GOAT driver; policy receives only public goals and real image assets."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from benchmark.goat import UPSTREAM_REVISION, UPSTREAM_SHA256, load_goat_episodes
from benchmark.goat_environment import (
    GOATEnvironment, SegmentBoundary, make_legacy_environment, render_reference_image,
)


class GOATUnavailable(RuntimeError):
    """Required protocol assets or exact legacy capabilities are unavailable."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    notes = []

    def clean(item, key=''):
        if isinstance(item, float) and not math.isfinite(item):
            notes.append({'field': key, 'value': str(item), 'encoding': 'null (not a finite score)'})
            return None
        if isinstance(item, dict):
            return {k: clean(v, f'{key}.{k}'.lstrip('.')) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v, f'{key}[{i}]') for i, v in enumerate(item)]
        return item

    result = clean(value)
    if notes:
        result['nonfinite_values'] = notes
        missing = {item['field']: 'undefined upstream metric' for item in notes
                   if item['field'].startswith('metrics.')}
        if missing:
            result.update(aggregate_ready=False, unavailable_metrics=missing)
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


def _resources(args, episode):
    import yaml

    if _sha256(args.official_config) != UPSTREAM_SHA256['goat_stretch_hm3d.yaml']:
        raise GOATUnavailable('Pinned official Stretch YAML hash differs')
    sim = yaml.safe_load(args.official_config.read_text())['habitat']['simulator']
    agent = sim['agents']['main_agent']
    camera = agent['sim_sensors']['rgb_sensor']
    profile = dict(height=agent['height'], radius=agent['radius'],
                   camera_position=camera['position'], hfov=camera['hfov'],
                   image_dimensions=[camera['height'], camera['width']],
                   turn_angle=sim['turn_angle'], tilt_angle=sim['tilt_angle'],
                   forward_step_size=0.25, allow_sliding=sim['habitat_sim_v0']['allow_sliding'])
    scene = (args.scene or args.scenes_root / episode.scene_id).resolve(strict=True)
    scene_dataset = (args.scene_dataset or
                     args.scenes_root / episode.source_episode['scene_dataset_config']).resolve(strict=True)
    return scene, scene_dataset, profile, {
        'protocol': 'goat_stretch_hm3d', 'upstream_revision': UPSTREAM_REVISION,
        'config_sha256': UPSTREAM_SHA256['goat_stretch_hm3d.yaml'],
        'source_scene_id': episode.scene_id, 'scene': str(scene), 'scene_sha256': _sha256(scene),
        'scene_dataset': str(scene_dataset), 'scene_dataset_sha256': _sha256(scene_dataset),
        'profile': profile, 'navmesh_mode': 'recompute_with_official_stretch_settings',
        'scope': 'legacy_habitat_compat_not_official_version_equivalence',
        'depth': 'aligned RGB-D policy sensor; not an official GOAT observation sensor',
    }


def prepare_reference_images(episode, *, asset_map, data_root, output):
    """Require scene/camera/hash-bound pre-rendered pixels; never move the live simulator."""
    images = [subtask for subtask in episode.subtasks if subtask.public_goal.modality == 'image']
    if not images:
        return {}, []
    if asset_map is None:
        raise GOATUnavailable('Legacy exact pose/FOV/resolution rendering is unverified; --image-assets must map opaque image_ref to pre-rendered path, sha256, scene_id and image_parameters')
    from PIL import Image

    entries = json.loads(Path(asset_map).read_text())
    references, records = {}, []
    for subtask in images:
        ref = subtask.public_goal.image_ref
        if ref in references:
            continue
        entry = entries.get(ref)
        if entry is None:
            raise GOATUnavailable(f'Missing pre-rendered image asset for {ref}')
        camera = {key: subtask.image_parameters[key] for key in
                  ('position', 'rotation', 'hfov', 'image_dimensions')}
        if entry.get('scene_id') != episode.scene_id or entry.get('image_parameters') != camera:
            raise GOATUnavailable('Reference image scene/camera parameters differ from GOAT')
        path = (Path(data_root) / entry['path']).resolve(strict=True)
        if _sha256(path) != entry.get('sha256'):
            raise GOATUnavailable('Reference image hash mismatch')
        with Image.open(path) as image:
            if image.format not in ('PNG', 'JPEG') or image.mode != 'RGB':
                raise GOATUnavailable('Reference must contain original RGB PNG/JPEG pixels')
            if [image.height, image.width] != camera['image_dimensions']:
                raise GOATUnavailable('Reference image dimensions differ from GOAT [height, width]')
            suffix = '.png' if image.format == 'PNG' else '.jpg'
        directory = Path(output) / 'reference_images'
        directory.mkdir(exist_ok=True)
        target = directory / (ref.split(':')[-1] + suffix)
        shutil.copyfile(path, target)
        references[ref] = str(target.resolve())
        records.append(dict(image_ref=ref, path=str(target.resolve()), sha256=entry['sha256'],
                            source_path=str(path), image_parameters=camera,
                            mode='explicit_pre_rendered_asset_no_live_pose_or_sensor_mutation'))
    return references, records


def _describe_reference(path, vlm):
    from llm_utils.cognav_llm_adapter import get_client_and_model
    from pydantic import BaseModel, Field

    class ImageGoalDescription(BaseModel):
        category: str = Field(min_length=1)
        appearance: str = Field(min_length=1)

    if vlm not in ('self_hosted', 'qwen_local') or os.getenv('LLM_OFFLINE', '0').lower() in ('1', 'true'):
        raise GOATUnavailable('Image goals require real self_hosted Qwen image inference, not offline fallback')
    client, model = get_client_and_model(vlm)
    if 'qwen' not in model.lower():
        raise GOATUnavailable('Image goals require an explicitly configured self_hosted Qwen model')
    image = Path(path)
    mime = 'image/png' if image.suffix == '.png' else 'image/jpeg'
    encoded = base64.b64encode(image.read_bytes()).decode('ascii')
    response = client.beta.chat.completions.parse(
        model=model, response_format=ImageGoalDescription, trace_label='goat_reference_image',
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe the main navigation target in this reference image. Return only its category and brief visible appearance. Do not invent hidden attributes, object IDs, location coordinates or a route. The robot must find this same instance.'},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}},
        ]}])
    answer = response.choices[0].message.parsed
    if answer is None or not answer.category.strip() or not answer.appearance.strip():
        raise GOATUnavailable('Qwen could not describe the actual reference image')
    return {'category': answer.category.strip(), 'appearance': answer.appearance.strip()}


def _stamp_plan(plan, revision, reference):
    from instruction.compiler import task_from_parsed_plan

    data = deepcopy(plan.diagnostics.get('task_spec')) or task_from_parsed_plan(plan).as_dict()
    data['revision'] = revision
    if reference:
        for target in plan.terminal_targets:
            target.attributes = {**(target.attributes or {}), 'goal_reference_image': reference}
        for target in data['targets']:
            if target['terminal']:
                target['attributes'] = {**(target.get('attributes') or {}), 'goal_reference_image': reference}
    plan.diagnostics['task_spec'] = data
    return plan


class _PublicGoalParser:
    def __init__(self, parser, revision, reference, image_goal=None):
        self.parser, self.revision, self.reference = parser, revision, reference
        self.vlm, self.strict_available_classes = parser.vlm, parser.strict_available_classes
        self.image_goal = image_goal

    def parse_plan(self, **kwargs):
        if self.image_goal is not None:
            # Qwen already supplied structured semantics; do not parse them as language again.
            from instruction.contracts import TaskSpec, TargetSpec
            from instruction_adapter.task_compat import runtime_plan
            from perception.detector_adapter import ground_plan

            task = TaskSpec(
                raw_instruction=kwargs['raw_instruction'],
                targets=(TargetSpec('target', self.image_goal['category'], terminal=True,
                                    attributes={'appearance': self.image_goal['appearance'],
                                                'goal_reference_image': self.reference}),),
                source='goat_image', revision=self.revision)
            plan = ground_plan(
                runtime_plan(task), available_classes=kwargs.get('available_classes'),
                vlm=self.vlm, strict_available_classes=self.strict_available_classes,
                use_legacy_similarity_fallback=False)
        else:
            plan = self.parser.parse_plan(**kwargs)
        return _stamp_plan(plan, self.revision, self.reference)


def install_public_goal(agent, public_goal, *, parser, revision, directory, reference=None):
    """Install public object, language or structured image goals without private task fields."""
    from objnav_benchmark_with_process_obs import _install_instruction_plan

    image_goal = None
    if public_goal.modality == 'image':
        if reference is None:
            raise GOATUnavailable('Opaque image_ref is not an image path')
        image_goal = _describe_reference(reference, parser.vlm)
        instruction = (f"Find the same {image_goal['category']} shown in the reference image. "
                       f"Visible appearance: {image_goal['appearance']}")
    else:
        instruction = public_goal.description or f'Find the {public_goal.category}.'
    plan, _ = _install_instruction_plan(
        habitat_mapper=agent.mapper, habitat_agent=agent,
        instruction_parser=_PublicGoalParser(parser, revision, reference, image_goal=image_goal),
        raw_instruction=instruction,
        dataset_target=public_goal.category if public_goal.modality == 'object' else '',
        episode_info={}, episode_dir=str(directory),
        source='benchmark_object_goal' if public_goal.modality == 'object' else 'goat_public_goal')
    # The object-goal installer bypasses parse_plan; apply its newer revision legally.
    if agent.task.spec.revision != revision:
        agent.task.install_plan(_stamp_plan(plan, revision, reference))
    _write_json(Path(directory) / 'instruction_adapter' / 'plan.json', plan.as_dict())
    _write_json(Path(directory) / 'instruction_adapter' / 'task_spec.json', agent.task.spec.as_dict())
    agent.mapper.object_perceiver.sam.initialize(agent.mapper.target)


def clear_subtask_state(agent):
    """Clear target-local execution state, retaining the map, node, counters and trajectory."""
    from backends.habitat.motion import HabitatExecution

    agent.execution.motion.cancel()
    agent.execution.motion.poll_active()
    # Scope guided-N request IDs to this segment without resetting the environment.
    agent.execution = HabitatExecution(agent.env)
    agent.view_control_state.reset()
    agent.final_instruction_verifier.evidence_store.reset()
    for field in ('selected_candidate', 'object_final', 'room_final', 'waypoint', 'waypoint_final',
                  'confirmed_target_waypoint', 'found_goal_position', 'check_again_postion',
                  'last_check_again_evidence', 'current_surface_evidence', 'instruction_accept_step',
                  'final_stop_accept_step', 'accepted_distance_to_target'):
        setattr(agent, field, None)
    for field in ('found_goal', 'need_check_again', 'just_come_back',
                  'final_instruction_accepted_this_step', 'instruction_success', 'final_stop_success'):
        setattr(agent, field, False)
    for field in ('last_check_again_image_path', 'instruction_decision', 'final_stop_decision',
                  'final_stop_mode', 'accepted_candidate_uid', 'accepted_distance_source'):
        setattr(agent, field, '')
    agent.accepted_relation_edge = {}
    agent.final_verifier_attempt_counter = {}
    agent.best_distance, agent.path, agent.path_index = 2.0, [], 0
    agent.path_node_idx = []


def _guided_run(*args, **kwargs):
    from backends.habitat.guided_runtime import run_guided_episode
    return run_guided_episode(*args, **kwargs)


def run_continuous_parent(agent, episode, *, parser, output, references, max_model_calls=100):
    """Reset once, install public revisions, and emit driver-owned stop events."""
    env = agent.env
    result = dict(status='running', metrics=None, subtasks=[
        dict(index=index, source_subtask_index=subtask.source_subtask_index,
             revision=index + 1, start_pose=None, end_pose=None, start_action=None,
             end_action=None, reason='not_executed', stop=None)
        for index, subtask in enumerate(episode.subtasks)])
    row = None
    try:
        agent.reset(episode.source_episode_index)
        for index, subtask in enumerate(episode.subtasks):
            if env.evaluator.ended:
                break
            row = result['subtasks'][index]
            row.update(start_pose=env.pose(), start_action=env.total_steps, reason=None)
            directory = Path(output) / f'subtask-{index:03d}'
            directory.mkdir(exist_ok=True)
            if env.boundary_reason != 'global_budget':
                mapper, nodes = agent.mapper, agent.mapper.nodes
                updates = mapper.update_iterations
                clear_subtask_state(agent)
                install_public_goal(agent, subtask.public_goal, parser=parser,
                                    revision=index + 1, directory=directory,
                                    reference=references.get(subtask.public_goal.image_ref))
                row['goal_switch'] = dict(
                    mapper_preserved=agent.mapper is mapper,
                    nodes_preserved=agent.mapper.nodes is nodes,
                    map_updates_before=updates, map_updates_after=agent.mapper.update_iterations,
                    node_count=len(agent.mapper.nodes), pose_after_install=env.pose())
                try:
                    termination = _guided_run(
                        agent, episode_idx=episode.source_episode_index, max_steps=math.inf,
                        max_model_calls=max_model_calls, stop_on_success=False,
                        output_dir=directory / 'harness')
                except SegmentBoundary as exc:
                    from backends.habitat.motion import HabitatExecution
                    # Admission rejection executed nothing; discard its unassigned receipt.
                    agent.execution.controller.cancel()
                    agent.execution = HabitatExecution(env)
                    termination = {'reason': str(exc), 'task_complete': False}
                finally:
                    agent.execution.motion.cancel()
                    if agent.execution.motion.active_goal_id is not None:
                        agent.execution.motion.poll_active()
                row['termination'] = termination
            else:
                termination = {'reason': 'global_budget'}
            if env.evaluator.ended:
                row.update(reason='policy_global_stop', stop='STOP')
            elif env.boundary_reason == 'global_budget':
                agent.obs = env.step(0)
                row.update(reason='global_budget', stop='STOP')
            else:
                reason = ('verified_finish' if termination['reason'] == 'verified_finish'
                          else env.boundary_reason or termination['reason'])
                agent.obs = env.subtask_stop(reason)
                row.update(reason=reason, stop='SUBTASK_STOP')
            row.update(end_pose=env.pose(), end_action=env.total_steps)
            _write_json(directory / 'segment.json', row)
        result.update(status='complete', metrics=env.evaluator.metrics(),
                      termination_reason=row['reason'] if row['stop'] == 'STOP' else 'final_subtask_stop')
    except Exception as exc:
        traceback.print_exc()
        result.update(status='unavailable' if isinstance(exc, GOATUnavailable) else 'infrastructure_error',
                      error={'type': type(exc).__name__, 'message': str(exc)})
        if row is not None:
            row.update(reason=result['status'], end_pose=env.pose(), end_action=env.total_steps)
            event = env.last_event
            if event and event['step'] > row['start_action'] and event['action'] in ('STOP', 'SUBTASK_STOP'):
                row['stop'] = event['action']
            _write_json(directory / 'segment.json', row)
        if env.evaluator is not None:
            result['partial_metrics_not_final'] = env.evaluator.metrics()
    result['actual_action_count'] = env.total_steps
    result['parent_reset_count'] = env.parent_reset_count
    return result


def run_goat(args):
    """Load exactly one parent, validate resources and execute the existing navigation stack."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory for one GOAT parent')
    result = dict(status='unavailable', metrics=None, source_sha256=None, public_goals=[])
    env = None
    try:
        source = (args.data_root / args.source_shard).resolve(strict=True)
        result['source_sha256'] = _sha256(source)
        episodes = load_goat_episodes(source, split=args.split)
        if args.parent_index < 0 or args.parent_index >= len(episodes):
            raise GOATUnavailable('parent-index is outside the explicit source shard')
        episode = episodes[args.parent_index]
        result.update(source_file=str(source), split=episode.split, parent_index=args.parent_index,
                      source_episode_id=episode.source_episode_id,
                      filtered_subtask_indices=list(episode.filtered_subtask_indices),
                      public_goals=[asdict(s.public_goal) for s in episode.subtasks],
                      upstream_revision=UPSTREAM_REVISION, vlm=args.vlm,
                      budgets=dict(max_steps_per_subtask=args.max_steps_per_subtask,
                                   global_budget=args.global_budget,
                                   accounting='all env.step actions plus SUBTASK_STOP; last slot reserved for stop',
                                   episode_over='wrapper segment boundary released on advance; carrier reset once'))
        if not episode.subtasks:
            raise GOATUnavailable('Parent has no retained GOAT subtasks')
        scene, scene_dataset, profile, resources = _resources(args, episode)
        result['resources'] = resources
        references, images = ({}, []) if args.image_assets is None else prepare_reference_images(
            episode, asset_map=args.image_assets, data_root=args.data_root, output=output)
        result.update(images=images, status='ready')
        _write_json(output / 'manifest.json', result)

        def record(event):
            with (output / 'actions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')

        base, cfg, prepare_navmesh = make_legacy_environment(
            episode, scene=scene, scene_dataset=scene_dataset, scenes_root=args.scenes_root,
            profile=profile, global_budget=args.global_budget, output=output)

        def after_reset(simulator):
            resources['navmesh'] = prepare_navmesh(simulator)
            if args.image_assets is None:
                directory = output / 'reference_images'
                directory.mkdir(exist_ok=True)
                for subtask in episode.subtasks:
                    ref = subtask.public_goal.image_ref
                    if subtask.public_goal.modality != 'image' or ref in references:
                        continue
                    path = directory / (ref.split(':')[-1] + '.png')
                    rendered = render_reference_image(simulator, subtask.image_parameters, path)
                    references[ref] = str(path.resolve())
                    images.append(dict(image_ref=ref, **rendered))
            _write_json(output / 'manifest.json', result)

        env = GOATEnvironment(base, episode, max_steps_per_subtask=args.max_steps_per_subtask,
                              global_budget=args.global_budget, record=record, after_reset=after_reset)
        from objnav_benchmark_with_process_obs import HM3D_Objnav_Agent, Instruct_Mapper, StriveInstructionParser
        from constants import categories
        from mapping_utils.transform import habitat_camera_intrinsic
        from llm_utils.lvlm_call_tracker import counts, reset_counts, set_trace_dir

        reset_counts()
        set_trace_dir(str(output / 'lvlm_calls'))
        resources['mapper_height_bounds'] = dict(
            floor_height=-profile['camera_position'][1],
            ceiling_height=profile['height'] - profile['camera_position'][1])
        mapper = Instruct_Mapper(habitat_camera_intrinsic(cfg), pcd_resolution=0.05,
                                 grid_resolution=0.1, voxel_dimension=[500, 500, 30],
                                 **resources['mapper_height_bounds'],
                                 save_dir=str(output), categories=categories, no_gpt_seg=True,
                                 env=env, vlm=args.vlm)
        agent = HM3D_Objnav_Agent(env, mapper, save_dir=str(output), vlm=args.vlm)
        parser = StriveInstructionParser(backend='llm', strict_available_classes=False, vlm=args.vlm)
        result.update(run_continuous_parent(agent, episode, parser=parser, output=output,
                                           references=references, max_model_calls=args.max_model_calls))
        result['model_usage'] = counts()
        if result['actual_action_count']:
            result['actions_sha256'] = _sha256(output / 'actions.jsonl')
    except Exception as exc:
        traceback.print_exc()
        result.update(status='unavailable' if isinstance(exc, (GOATUnavailable, FileNotFoundError, ImportError)) else 'infrastructure_error',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result.update(status='infrastructure_error', close_error=str(exc))
    result['aggregate_ready'] = result['status'] == 'complete'
    return _write_json(output / 'result.json', result)
