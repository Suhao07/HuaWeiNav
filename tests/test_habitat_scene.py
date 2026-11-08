"""Exercise the NumPy point-cloud to serializable scene boundary."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from backends.habitat.scene import current_observation, snapshot_from_mapper
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


@pytest.fixture
def projected_scene(monkeypatch, tmp_path):
    """Mock the projection boundary, not visibility logic or Open3D internals."""
    geometry = ModuleType("mapping_utils.geometry")
    projector = Mock(side_effect=lambda pcd, *_: tuple(pcd.T))
    geometry.project_to_camera = projector
    monkeypatch.setitem(sys.modules, "mapping_utils.geometry", geometry)

    def build(projections, measured_depth=1.):
        objects = [SimpleNamespace(
            uid=uid, pcd=np.asarray(points, dtype=float).reshape(-1, 3),
            # Historical detector regions must not imply current visibility.
            bbox=[0, 0, 8, 6], rgb=np.zeros((6, 8, 3), dtype=np.uint8),
        ) for uid, points in projections.items()]
        mapper = SimpleNamespace(
            objects=objects, camera_intrinsic=np.diag([2., 3., 1.]),
            current_position=np.array([1., 2., 3.]),
            current_rotation=np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
        )
        return SimpleNamespace(
            mapper=mapper, obs={"depth": np.full((6, 8, 1), measured_depth)},
            episode_samples=3, episode_steps=17, save_dir=str(tmp_path),
        )

    return build, projector


def test_current_observation_binds_visible_regions_to_saved_episode_frame(projected_scene):
    from PIL import Image

    build, projector = projected_scene
    agent = build({
        "visible": [(1, 1, 1.), (3, 1, 1.), (1, 3, 1.), (7, 5, 2.), (-1, 0, 1.)],
        "occluded": [(1, 1, 2.), (3, 1, 2.), (1, 3, 2.)],
        "offscreen": [(8, 1, 1.), (8, 2, 1.), (8, 3, 1.)],
        "empty": [],
    })
    saved = Path(agent.save_dir) / "episode-2" / "rgb" / "monitor-rgb_17.jpg"
    saved.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(saved)

    frame = current_observation(agent)

    assert frame["observed_object_uids"] == ["visible"]
    assert frame["object_boxes"] == {"visible": [1, 1, 4, 4]}
    assert frame["id"] == "2:17" and frame["step"] == 17
    assert frame["image_ref"] == str(saved.resolve()) and saved.is_file()
    assert frame["position"] == agent.mapper.current_position.tolist()
    assert frame["rotation"] == agent.mapper.current_rotation.tolist()
    assert frame["source"] == "depth_validated_map_projection"
    assert frame["depth_tolerance_m"] == .2
    json.dumps(frame)
    assert projector.call_count == len(agent.mapper.objects)
    for call, obj in zip(projector.call_args_list, agent.mapper.objects):
        assert call.args[0] is obj.pcd
        assert call.args[1] is agent.mapper.camera_intrinsic
        assert call.args[2] is agent.mapper.current_position
        assert call.args[3] is agent.mapper.current_rotation


@pytest.mark.parametrize("points", [
    pytest.param([], id="empty-projection"),
    pytest.param([(-1, 0, 1.), (-1, 1, 1.), (-1, 2, 1.)], id="left"),
    pytest.param([(8, 0, 1.), (8, 1, 1.), (8, 2, 1.)], id="right"),
    pytest.param([(0, -1, 1.), (1, -1, 1.), (2, -1, 1.)], id="above"),
    pytest.param([(0, 6, 1.), (1, 6, 1.), (2, 6, 1.)], id="below"),
    pytest.param([(1, 1, 1.)] * 5, id="one-distinct-pixel"),
    pytest.param([(1, 1, 1.), (2, 2, 1.)] * 3, id="two-distinct-pixels"),
    pytest.param([(1, 1, 1.), (2, 2, 1.), (3, 3, 2.)], id="third-pixel-occluded"),
    pytest.param([(1, 1, 1.2001), (2, 2, 1.2001), (3, 3, 1.2001)], id="behind-surface"),
    pytest.param([(1, 1, .7999), (2, 2, .7999), (3, 3, .7999)], id="in-front-of-surface"),
])
def test_current_observation_rejects_insufficient_visible_pixels(projected_scene, points):
    build, _ = projected_scene
    frame = current_observation(build({"stale-object": points}))
    assert frame["observed_object_uids"] == []
    assert frame["object_boxes"] == {}


@pytest.mark.parametrize("measured_depth", [0., -1., float("nan"), float("inf")])
def test_current_observation_rejects_invalid_sensor_depth(projected_scene, measured_depth):
    build, _ = projected_scene
    agent = build({"object": [(1, 1, .1), (2, 2, .1), (3, 3, .1)]}, measured_depth)
    frame = current_observation(agent)
    assert frame["observed_object_uids"] == [] and frame["object_boxes"] == {}


@pytest.mark.parametrize("projected_depth,measured_depth", [(.25, .05), (.05, .25)])
def test_current_observation_includes_exact_depth_tolerance(projected_scene, projected_depth, measured_depth):
    build, _ = projected_scene
    # These values subtract to exactly the float used for the 0.2 m threshold.
    assert abs(projected_depth - measured_depth) == .2
    agent = build({"object": [(0, 0, projected_depth), (7, 0, projected_depth),
                              (7, 5, projected_depth)]}, measured_depth)
    frame = current_observation(agent)
    assert frame["observed_object_uids"] == ["object"]
    assert frame["object_boxes"] == {"object": [0, 0, 8, 6]}


def test_current_observation_recomputes_visibility_from_current_depth(projected_scene):
    build, _ = projected_scene
    agent = build({"object": [(1, 1, 1.), (2, 2, 1.), (3, 3, 1.)]})
    previous = current_observation(agent)
    agent.episode_steps += 1
    agent.obs["depth"] = np.full((6, 8, 1), .5)
    current = current_observation(agent)
    assert previous["observed_object_uids"] == ["object"]
    assert current["observed_object_uids"] == [] and current["object_boxes"] == {}
    assert current["id"] == "2:18" and current["step"] == 18
    assert Path(current["image_ref"]).name == "monitor-rgb_18.jpg"
    assert current["image_ref"] != previous["image_ref"]
