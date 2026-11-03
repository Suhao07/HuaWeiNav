"""Category observations preserve uncertainty about historical instance identity."""
from types import SimpleNamespace

import pytest

from instruction_adapter.contracts import InstructionPlan, TargetQuery
from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import PriorMapData, PriorObject
from prior_map.memory import PriorMapMemory
from prior_map.query import PriorMapQueryService


def _memory(exact=True):
    return PriorMapMemory(base_map=PriorMapData(scene_id='room', objects=tuple(
        PriorObject(uid=uid, label='chair', exact=exact, position_xyz=position)
        for uid, position in [('left', (0., 0., 0.)), ('right', (5., 0., 0.))])),
        alignment=PriorMapAlignment.identity())


@pytest.mark.parametrize('backend', ['mapper', 'snapshot'])
@pytest.mark.parametrize('exact', [True, False])
def test_category_observation_preserves_unresolved_identity(backend, exact):
    memory = _memory(exact)
    context = SimpleNamespace(objects=[{'uid': 'new-chair', 'label': 'chair', 'confidence': .9}])
    record = (memory.update_from_mapper(context, 1) if backend == 'mapper'
              else memory.update_from_snapshot(context))
    assert record.observed_object_uids == ('new-chair',)
    assert all(state.observation_count == 0 for state in memory.object_states.values())
    assert all(state.matched_runtime_uid is None for state in memory.object_states.values())


def test_live_category_preserves_instance_query_uncertainty():
    memory = _memory()
    plan = InstructionPlan(raw_instruction='chair',
        targets=[TargetQuery(id='target', name='chair')], valid=True)
    context = {'objects': [{'uid': 'new-chair', 'label': 'chair'}]}
    result = PriorMapQueryService().query(plan, context, memory)
    assert all(r.metadata['score_components']['live_match'] == 0 for r in result.object_rankings)


def test_verified_correspondence_updates_the_bound_instance():
    memory = _memory()
    memory.mark_object_verified('right', 'new-chair', 1)
    memory.update_from_mapper(SimpleNamespace(objects=[
        {'uid': 'new-chair', 'label': 'chair', 'confidence': .9}]), 2)
    assert memory.object_states['right'].observation_count == 1
    assert memory.object_states['left'].observation_count == 0


def test_same_persisted_uid_updates_instance():
    memory = _memory()
    memory.update_from_mapper(SimpleNamespace(objects=[
        {'uid': 'right', 'label': 'chair', 'confidence': .9}]), 2)
    assert memory.object_states['right'].observation_count == 1
    assert memory.object_states['left'].observation_count == 0
