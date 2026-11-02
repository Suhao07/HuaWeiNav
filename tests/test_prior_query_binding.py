"""Retrieval identity is checked before a historical object reaches motion."""
import hashlib
import pytest
from prior_map.bundle import load_bundle
from prior_map.query_binding import queried_object_uid
from prior_map.query_binding import current_concept_candidate
from prior_map.query_binding import query_execution_task
from instruction.contracts import TaskSpec, TargetSpec
from test_prior_compaction import _source


def _receipt(root):
    manifest = load_bundle(root)['manifest']
    return {'map_id': manifest['map_id'], 'map_revision': manifest['map_revision'],
        'prior_map_sha256': hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest(),
        'matches': [{'ranking': {'object_uid': 'instance', 'score': 1.}}]}


def test_query_survives_bundle_relocation(tmp_path):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    moved = tmp_path / 'moved'
    root.rename(moved)
    assert queried_object_uid(receipt, moved) == 'instance'


def test_current_concept_requires_current_matching_evidence(tmp_path):
    root = tmp_path / 'current'
    _source(root)
    assert current_concept_candidate(root, 'chair') == 'instance'
    with pytest.raises(ValueError, match='no matching concept'):
        current_concept_candidate(root, 'refrigerator')


def test_compiled_language_uses_target_for_current_query(tmp_path):
    root = tmp_path / 'current'
    _source(root)
    task = TaskSpec('Find a comfortable chair to sit on.',
                    (TargetSpec('target', 'chair', attributes={'color': 'brown'}),), revision=3)
    query = {'concept': task.raw_instruction, 'task_spec': task.as_dict()}
    resolved = query_execution_task(query, task)
    assert resolved == task
    assert current_concept_candidate(root, resolved.targets[0].name) == 'instance'


def test_compiled_query_rejects_a_different_task_revision():
    task = TaskSpec('Find a chair.', (TargetSpec('target', 'chair'),), revision=2)
    changed = TaskSpec('Find a chair.', task.targets, revision=3)
    with pytest.raises(ValueError, match='differs from compiled query'):
        query_execution_task({'concept': task.raw_instruction, 'task_spec': task.as_dict()}, changed)


def test_concept_query_binds_matching_explicit_task():
    query = {'concept': 'chair', 'task_spec': None}
    task = TaskSpec('Find a chair.', (TargetSpec('target', 'chair'),))
    assert query_execution_task(query, task) == task
    assert query_execution_task(query).targets[0].name == 'chair'
    with pytest.raises(ValueError, match='differs from queried concept'):
        query_execution_task(query, TaskSpec('Find a bed.', (TargetSpec('target', 'bed'),)))


def test_multi_target_contract_is_rejected_before_motion():
    task = TaskSpec('Find a chair and a table.',
                    (TargetSpec('chair', 'chair'), TargetSpec('table', 'table')))
    with pytest.raises(ValueError, match='one unconstrained object target'):
        query_execution_task({'concept': task.raw_instruction, 'task_spec': task.as_dict()})


@pytest.mark.parametrize('field', ['map_id', 'map_revision', 'prior_map_sha256'])
def test_query_rejects_mismatched_map_before_motion(tmp_path, field):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    receipt[field] = 'different'
    with pytest.raises(ValueError, match=field):
        queried_object_uid(receipt, root)


def test_query_requires_a_known_hypothesis(tmp_path):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    receipt['matches'][0]['ranking']['object_uid'] = 'missing'
    with pytest.raises(ValueError, match='unknown historical object'):
        queried_object_uid(receipt, root)
    receipt['matches'] = []
    with pytest.raises(ValueError, match='no historical object'):
        queried_object_uid(receipt, root)


def test_query_evidence_keeps_matching_observations_and_preserves_conflicts(tmp_path):
    from prior_map.query_binding import query_evidence_observation_ids
    root = tmp_path / 'source'
    _source(root)
    query = _receipt(root)
    query['runtime_plan'] = {'targets': [{'name': 'chair', 'terminal': True, 'detector_terms': ['chair']}]}
    assert query_evidence_observation_ids(query, root) == ['o0', 'o1', 'o2', 'o3']
    assert len(load_bundle(root)['observations']) == 5
    query['runtime_plan']['targets'][0]['detector_terms'] = ['table']
    query['runtime_plan']['targets'][0]['name'] = 'table'
    assert query_evidence_observation_ids(query, root) == ['conflict']


def test_query_evidence_requires_actual_matching_detector_observation(tmp_path):
    from prior_map.query_binding import query_evidence_observation_ids
    root = tmp_path / 'source'
    _source(root)
    query = _receipt(root)
    query['runtime_plan'] = {'targets': [{'name': 'piano', 'terminal': True}]}
    with pytest.raises(ValueError, match='no query-matching observation'):
        query_evidence_observation_ids(query, root)


def test_current_observation_selection_uses_raw_concept_then_confidence(tmp_path):
    from prior_map.query_binding import current_concept_observation
    root=tmp_path/'source'
    _source(root)
    assert current_concept_observation(root,'chair') == {'candidate_uid':'instance','observation_id':'o0'}


def test_current_observation_requires_raw_support_for_fused_label(tmp_path):
    import json
    from prior_map.query_binding import current_concept_observation
    root=tmp_path/'source'
    _source(root)
    records=load_bundle(root)['observations']
    for record in records:
        record['detector_label']='table'
    (root/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='no concept-matching observation'):
        current_concept_observation(root,'chair')
