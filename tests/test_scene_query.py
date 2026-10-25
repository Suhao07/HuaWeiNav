"""Queries preserve observations and share identity across backend exports."""
from copy import deepcopy
from types import SimpleNamespace

from backends.habitat.scene import snapshot_from_mapper
from instruction_adapter.contracts import InstructionPlan, TargetQuery
from instruction_adapter.verifier import candidate_from_object
from planning.semantic_snapshot_context import select_target_candidate_from_snapshot
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot


def test_query_preserves_native_snapshot_and_distinct_uids(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    objects = (ObjectNodeSnapshot("cup-a", "cup", (1., 0., 0.), 0.8),
               ObjectNodeSnapshot("cup-b", "cup", (1., 0., 0.), 0.9))
    snapshot = SemanticMapSnapshot(1, Pose3D((0., 0., 0.)), objects=objects)
    before = deepcopy(snapshot)
    plan = InstructionPlan("cup", targets=[TargetQuery("cup", "cup")], valid=True)
    selection = select_target_candidate_from_snapshot(snapshot, plan=plan)
    assert snapshot == before
    assert selection.result.obj is objects[1]
    assert candidate_from_object(objects[0]).uid != candidate_from_object(objects[1]).uid
    assert isinstance(selection.result.obj.confidence, float)
    assert not hasattr(selection.result.obj, "tag")


def test_habitat_export_and_native_input_select_same_object(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    mapper = SimpleNamespace(current_position=[0, 0, 0], objects=[
        SimpleNamespace(uid="cup-a", tag="cup", position=[1, 0, 0], confidence=0.9,
                        bbox=[1, 2, 3, 4]),
        SimpleNamespace(uid="cup-b", tag="cup", position=[2, 0, 0], confidence=0.5,
                        bbox=[1, 2, 3, 4]),
    ])
    plan = InstructionPlan("cup", targets=[TargetQuery("cup", "cup")], valid=True)
    exported, lookup = snapshot_from_mapper(mapper, step=3)
    native = SemanticMapSnapshot(3, Pose3D((0, 0, 0)), objects=exported.objects)
    a = select_target_candidate_from_snapshot(exported, plan=plan)
    b = select_target_candidate_from_snapshot(native, plan=plan)
    assert a.selected_snapshot_uid == b.selected_snapshot_uid == "cup-a"
    assert lookup[a.selected_snapshot_uid] is mapper.objects[0]
    assert mapper.objects[0].tag == "cup"
