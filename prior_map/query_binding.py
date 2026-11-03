"""Bind a saved retrieval result to the map used for historical-view execution."""
import hashlib
from pathlib import Path
from .bundle import load_bundle
from .contracts import STRUCTURAL_LABELS


def query_execution_task(query, explicit_task=None):
    """Resolve the single-target motion diagnostic's immutable task once.

    Compiled receipts own the task semantics; a separate task file must match
    that revision exactly. Concept-only receipts can bind an explicit task for
    the same concept. The raw language in ``concept`` is never a detector label
    when a compiled TaskSpec is present.
    """
    from instruction.contracts import TaskSpec, TargetSpec
    if query['task_spec'] is not None:
        task = TaskSpec.from_dict(query['task_spec'])
        if explicit_task is not None and task != explicit_task:
            raise ValueError('Verification task differs from compiled query task')
    else:
        task = explicit_task or TaskSpec(query['concept'],
            (TargetSpec('query', query['concept']),), source='explicit_concept_diagnostic')
        if len(task.targets) == 1 and task.targets[0].name != query['concept']:
            raise ValueError('Verification target differs from queried concept')
    if (len(task.targets) != 1 or not task.targets[0].terminal
            or task.targets[0].min_count != 1 or task.constraints):
        raise ValueError('Motion diagnostic supports one unconstrained object target')
    return task


def current_concept_candidate(directory, concept):
    """Reuse concept ranking on current detections; return a hypothesis, not identity.

    This diagnostic supports one concept target. Historical UID and category
    never overwrite a current instance or count as semantic verification.
    """
    from instruction_adapter.contracts import InstructionPlan, TargetQuery
    from .alignment import PriorMapAlignment
    from .memory import PriorMapMemory
    from .query import PriorMapQueryService
    prior = load_bundle(directory)['prior_map']
    memory = PriorMapMemory(base_map=prior, alignment=PriorMapAlignment.identity(
        prior_frame_id=prior.frame_id, runtime_frame_id=prior.frame_id))
    plan = InstructionPlan(raw_instruction=concept,
        targets=[TargetQuery(id='query', name=concept, detector_terms=[concept])], valid=True)
    rankings = PriorMapQueryService().query(plan, {}, memory).object_rankings
    matches = [r for r in rankings if r.metadata['score_components']['concept_relevance'] > 0]
    if not matches:
        raise ValueError('Current perception has no matching concept hypothesis')
    return max(matches, key=lambda r: r.score).object_uid


def queried_object_uid(query, directory):
    """Resolve the top ranked hypothesis after checking identity and revision.

    A legacy map can share its map_id with another bundle. The canonical map
    hash also binds the receipt to its actual object identities and geometry.
    This is hypothesis selection, with no task-progress or verification effect.
    Raises ValueError for stale/map-mismatched receipts or an empty retrieval.
    """
    root = Path(directory)
    bundle = load_bundle(root)
    expected = {'map_id': bundle['manifest']['map_id'],
                'map_revision': bundle['manifest']['map_revision'],
                'prior_map_sha256': hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest()}
    for field, value in expected.items():
        if query[field] != value:
            raise ValueError(f'Query receipt differs from current map: {field}')
    if not query['matches']:
        raise ValueError('Query returned no historical object hypothesis')
    best = max(query['matches'], key=lambda item: item['ranking']['score'])
    uid = best['ranking']['object_uid']
    if uid not in {obj.uid for obj in bundle['prior_map'].objects}:
        raise ValueError(f'Query references unknown historical object: {uid}')
    return uid


def query_evidence_observation_ids(query, directory):
    """Bind the selected hypothesis to observations matching the query vocabulary.

    Reuses the query service's text relevance, not a new semantic verifier.
    Unmatched observations stay in the map as conflicting/context evidence.
    A detection-label match remains a hypothesis requiring current verification.
    """
    from .query import _extract_target_terms, _best_match_score, _normalize
    uid = queried_object_uid(query, directory)
    terms = [_normalize(term) for term in _extract_target_terms(query['runtime_plan'])]
    observations = load_bundle(directory)['observations']
    selected = [o['observation_id'] for o in observations
                if uid in o['object_uids'] and
                _best_match_score([_normalize(o['detector_label'])], terms) > 0]
    if not selected:
        raise ValueError('Selected hypothesis has no query-matching observation evidence')
    return selected


def current_concept_observation(directory, concept):
    """Select one raw observation for the current ranked concept hypothesis.

    Reuse label relevance, then detector confidence, with observation ID as a
    deterministic tie-break. This selects evidence, not a verified identity.
    """
    from .query import _best_match_score, _normalize
    uid = current_concept_candidate(directory, concept)
    candidates = []
    for observation in load_bundle(directory)['observations']:
        if uid not in observation['object_uids']:
            continue
        relevance = _best_match_score([_normalize(observation['detector_label'])], [_normalize(concept)])
        if relevance > 0:
            candidates.append((relevance, observation['detector_score'], observation['observation_id']))
    if not candidates:
        raise ValueError('Current candidate has no concept-matching observation')
    best = sorted(candidates, key=lambda row: (-row[0], -row[1], row[2]))[0]
    return dict(candidate_uid=uid, observation_id=best[2])


def historical_object_evidence(directory, plan, result, output):
    """Build one historical reference for the top ranked object's matching observation.

    Reuse query vocabulary matching, then detector confidence and observation ID.
    The annotated image identifies the raw detection, not the fused object's whole
    extent. Historical evidence can guide search; current verification owns STOP.
    """
    from PIL import Image, ImageDraw
    from .query import _extract_target_terms, _normalize, _best_match_score
    from .multimodal import PriorMapMultimodalContext
    root = Path(directory)
    bundle = load_bundle(root)
    if not result.object_rankings:
        return ()
    uid = result.object_rankings[0].object_uid
    terms = [_normalize(term) for term in _extract_target_terms(plan)]
    matches = []
    for obs in bundle['observations']:
        relevance = _best_match_score([_normalize(obs['detector_label'])], terms)
        if uid in obs['object_uids'] and relevance > 0:
            matches.append((relevance, obs['detector_score'], obs['observation_id'], obs))
    if not matches:
        return ()
    obs = sorted(matches, key=lambda row: (-row[0], -row[1], row[2]))[0][3]
    image = Image.open(root / obs['image_ref']).convert('RGB')
    ImageDraw.Draw(image).rectangle(obs['bbox_xyxy'], outline='red', width=2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'historical-object.png'
    image.save(path)
    metadata = dict(map_id=bundle['manifest']['map_id'],
        map_revision=bundle['manifest']['map_revision'], object_uid=uid,
        observation_id=obs['observation_id'], detector_label=obs['detector_label'],
        bbox_xyxy=obs['bbox_xyxy'], mask_sha256=hashlib.sha256((root/obs['mask_ref']).read_bytes()).hexdigest(),
        prior_map_sha256=hashlib.sha256((root/'prior_map.json').read_bytes()).hexdigest())
    return (PriorMapMultimodalContext(image_path=str(path.resolve()),
        image_role='historical_object_reference', map_frame_id=bundle['prior_map'].frame_id,
        text_context='Red box: historical detection hypothesis. Use for search ranking only; '
                     'current observations must establish identity, visibility and task completion.',
        metadata=metadata),)


def historical_region_observations(directory, rooms):
    """Summarize predicted labels observed from each uniquely located camera.

    Args:
        directory: Existing portable map bundle with observations.jsonl.
        rooms: Canonical geometric regions in the bundle coordinate frame.
    Returns:
        Region summaries with distinct-frame label counts and evidence references.

    N(r,c)=|{frame: camera(frame) in r and label c was detected}|, c ∉ STRUCTURAL_LABELS.
    同帧重复框只计一次；冲突类别各自保留。观察来源表达可见语义，
    不把自由空间多边形当成家具归属边界，也不宣称独立对象数量。
    墙、地面、门窗等构件几乎每帧可见，不区分区域，只从摘要隐去；
    frame_ids 仍计全部观察帧，地图与observations.jsonl保持不变。
    """
    import json
    from collections import defaultdict
    from matplotlib.path import Path as Polygon

    path = Path(directory) / 'observations.jsonl'
    content = path.read_bytes()
    polygons = [(room.uid, Polygon(room.boundary_xy)) for room in rooms if len(room.boundary_xy) >= 3]
    frames, label_frames, examples = defaultdict(set), defaultdict(lambda: defaultdict(set)), defaultdict(dict)
    for line in content.decode('utf-8').splitlines():
        observation = json.loads(line)
        pose = observation['camera_to_map']
        matches = [uid for uid, polygon in polygons if polygon.contains_point((pose[0][3], pose[1][3]))]
        if len(matches) != 1:
            continue
        uid, frame, label = matches[0], observation['frame_id'], observation['detector_label']
        frames[uid].add(frame)
        if label in STRUCTURAL_LABELS:
            continue
        label_frames[uid][label].add(frame)
        examples[uid].setdefault(label, observation['observation_id'])
    return {uid: {'meaning': 'predicted_labels_observed_from_region',
                  'frame_ids': sorted(frames[uid]),
                  'label_frame_counts': {label: len(ids) for label, ids in sorted(counts.items())},
                  'example_observation_ids': examples[uid],
                  'observations_sha256': hashlib.sha256(content).hexdigest()}
            for uid, counts in label_frames.items()}
