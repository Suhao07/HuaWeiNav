"""Exercise the NumPy point-cloud to serializable scene boundary."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np

from backends.habitat.scene import snapshot_from_mapper
from scene.dependencies import object_dependencies


def test_float32_point_cloud_exports_json_geometry_and_dependency_hash():
    points = np.array([[1, 2, 3], [3, 6, 9]], dtype=np.float32)
    positions = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: points))
    obj = SimpleNamespace(uid="toilet-1", tag="toilet", position=points.mean(axis=0),
                          confidence=np.float32(0.9),
                          pcd=SimpleNamespace(point=SimpleNamespace(positions=positions)))
    mapper = SimpleNamespace(objects=[obj], current_position=np.zeros(3, dtype=np.float32))
    snapshot, _ = snapshot_from_mapper(mapper, step=155)
    payload = json.loads(json.dumps(asdict(snapshot)))
    assert payload["objects"][0]["bbox3d_center"] == [2, 4, 6]
    assert payload["objects"][0]["bbox3d_extent"] == [2, 4, 6]
    assert len(object_dependencies(snapshot, "toilet-1")["object:toilet-1"]) == 64


def test_native_instance_identity_survives_geometry_and_label_updates():
    from mapping_utils.representation import ObjectNode
    points = np.array([[1, 2, 3], [3, 6, 9]], dtype=np.float32)
    positions = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: points))
    pcd = SimpleNamespace(point=SimpleNamespace(positions=positions))
    obj = ObjectNode(pcd, pcd, "tv_monitor", 0.9, points.mean(axis=0), 2, None, None)
    mapper = SimpleNamespace(objects=[obj], current_position=np.zeros(3, dtype=np.float32))
    before, _ = snapshot_from_mapper(mapper)
    uid = before.objects[0].uid
    before_dependencies = object_dependencies(before, uid)

    # 模拟 mapper 对已关联实例融合新观测，而不是创建另一物理实例。
    points[1] += 0.5
    obj.position = points.mean(axis=0)
    obj.tag = "tv monitor"
    after, lookup = snapshot_from_mapper(mapper)
    assert after.objects[0].uid == uid
    assert lookup[uid] is obj
    assert object_dependencies(after, uid) != before_dependencies
    another = ObjectNode(pcd, pcd, obj.tag, 0.9, obj.position, 2, None, None)
    assert another.uid != uid
