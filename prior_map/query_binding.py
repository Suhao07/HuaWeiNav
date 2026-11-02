"""Bind a saved retrieval result to the map used for historical-view execution."""
import hashlib
from pathlib import Path
from .bundle import load_bundle


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
