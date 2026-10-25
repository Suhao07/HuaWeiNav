"""Dependency checks track relevant geometry, not unrelated scene updates."""
from dataclasses import replace
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot
from scene.dependencies import object_dependencies


def test_dependency_scope_keeps_unrelated_observations_independent():
    cup = ObjectNodeSnapshot("cup-1", "cup", position=(1.0, 0.0, 0.0), confidence=0.8)
    snapshot = SemanticMapSnapshot(1.0, Pose3D((0.0, 0.0, 0.0)), objects=(cup,))
    original = object_dependencies(snapshot, "cup-1")
    updated = replace(snapshot, timestamp=2.0, robot_pose=Pose3D((0.1, 0.0, 0.0)),
                      objects=(replace(cup, confidence=0.9), ObjectNodeSnapshot("chair", "chair")))
    assert object_dependencies(updated, "cup-1") == original
    moved = replace(snapshot, objects=(replace(cup, position=(2.0, 0.0, 0.0)),))
    assert object_dependencies(moved, "cup-1") != original
    assert object_dependencies(replace(snapshot, objects=()), "cup-1") != original
    reset = replace(snapshot, metadata={"frame_revision": 2})
    assert object_dependencies(reset, "cup-1") != original
