import ast
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from backends.habitat.geometry import HabitatMapperQueries
from backends.isaac.environment import MAX_BODY_BYTES
from backends.isaac.mapper import IsaacMapper, IsaacMapperQueries, camera_inputs
from mapping_utils.preprocess import calc_depth_img
from test_isaac_environment import FakeEnv, FakeMap, bridge_at, http_request


class MemoryClient:
    def __init__(self, bridge):
        self.bridge = bridge
        self.commands = []

    def command(self, action, **fields):
        request = {"action": action, **fields}
        payload = json.dumps(request, allow_nan=False).encode()
        assert len(payload) <= MAX_BODY_BYTES
        self.commands.append(request)
        status, result, _ = http_request(self.bridge, payload)
        assert status == 200, result
        return result


def geometry_functions():
    path = Path(__file__).resolve().parents[1] / "mapping_utils/geometry.py"
    names = {"get_pointcloud_from_depth", "get_pointcloud_from_depth_mask", "translate_to_world"}
    tree = ast.parse(path.read_text())
    scope = {"np": np}
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name in names], type_ignores=[]), str(path), "exec"), scope)
    return scope


def frame_for(shape=(48, 64), center=(32., 24.), yaw=.7):
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    right = np.cross(forward, [0, 0, 1])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, [0, 0, -1], forward))
    transform[:3, 3] = [2., -3., 1.5]
    return {"size_hw": list(shape), "intrinsics": [[40., 0., center[0]], [0., 40., center[1]], [0., 0., 1.]],
            "camera_to_map": transform.tolist(), "depth_unit": "m", "depth_convention": "optical_z"}


@pytest.mark.parametrize("center", [(31.5, 23.5), (32., 24.), (20.3, 17.2)])
@pytest.mark.parametrize("yaw", [0., .7, -2.9])
def test_camera_backprojection_matches_optical_geometry(center, yaw):
    frame = frame_for(center=center, yaw=yaw)
    before = copy.deepcopy(frame)
    intrinsic, position, rotation, shape = camera_inputs(frame)
    depth = np.linspace(.6, 4., np.prod(shape)).reshape(shape)
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    functions = geometry_functions()
    points, _ = functions["get_pointcloud_from_depth"](rgb, depth, intrinsic)
    masked = functions["get_pointcloud_from_depth_mask"](depth, np.ones(shape), intrinsic)
    np.testing.assert_array_equal(points, masked)
    world = functions["translate_to_world"](points, position, rotation)
    v, u = np.indices(shape)
    optical = np.column_stack((((u - intrinsic[0, 2]) * depth / intrinsic[0, 0]).ravel(),
                               ((v - intrinsic[1, 2]) * depth / intrinsic[1, 1]).ravel(), depth.ravel()))
    camera = np.asarray(frame["camera_to_map"])
    np.testing.assert_allclose(world, optical @ camera[:3, :3].T + camera[:3, 3], atol=1e-12)
    if center == (31.5, 23.5):
        np.testing.assert_array_equal(points[:, 1],
            ((shape[0] - 1 - v - intrinsic[1, 2]) * depth / intrinsic[1, 1]).ravel())
    assert frame == before
    lower, upper = calc_depth_img(intrinsic, .51, 4.9, image_shape=shape)
    assert lower.shape == upper.shape == shape
    assert np.all(lower == np.float32(.51)) and np.all(upper == np.float32(4.9))


def test_habitat_depth_dimensions_unchanged_by_default():
    intrinsic = np.array(frame_for(center=(31.5, 23.5))["intrinsics"])
    default = calc_depth_img(intrinsic, .51, 4.9)
    explicit = calc_depth_img(intrinsic, .51, 4.9, image_shape=(48, 64))
    for left, right in zip(default, explicit):
        np.testing.assert_array_equal(left, right)


def test_habitat_queries_keep_native_height_and_current_body(monkeypatch):
    points, paths, distances = [], [], []
    monkeypatch.setitem(sys.modules, "habitat_sim", NS(ShortestPath=NS))
    body = NS(position=np.array([3., .2, -4.]))
    sim = NS(is_navigable=lambda p: points.append(p) or p[1] == .08,
             pathfinder=NS(find_path=lambda p: paths.append(p) or False),
             get_agent_state=lambda: body,
             geodesic_distance=lambda start, end: distances.append((list(start), end)) or math.inf)
    mapper = NS(env=NS(sim=sim))
    query = HabitatMapperQueries(mapper)
    values = np.array([[1., 2., .08], [1., 2., 5.]])
    before = values.copy()
    assert query.navigable_mask(values).tolist() == [True, False]
    assert points == [[1., .08, 2.], [1., 5., 2.]]
    assert not query.path_exists(values[0], values[1])
    assert paths[0].requested_start == points[0] and paths[0].requested_end == points[1]
    assert query.distance_from_agent(values[1]) == math.inf
    assert distances == [([3., .2, -4.], [1., .2, 2.])]
    body.position = np.array([5., .3, -6.])
    query.distance_from_agent(values[1])
    assert distances[-1] == ([5., .3, -6.], [1., .3, 2.])
    np.testing.assert_array_equal(values, before)
    assert query.navigable_mask(np.empty((0, 3))).shape == (0,)


def test_habitat_node_distance_projection_remains_distinct_from_body_distance():
    calls = []
    sim = NS(geodesic_distance=lambda a, b: calls.append((list(a), list(b))) or 3.,
             get_agent_state=lambda: NS(position=[8., 2.4, -3.]))
    mapper = NS(initial_position=np.array([1., 2., 3.]), env=NS(sim=sim))
    query = HabitatMapperQueries(mapper)
    start = np.array([4., 5., 6.])
    ends = np.array([[7., 8., 9.], [10., 11., 12.]])
    before = start.copy(), ends.copy()
    assert query.node_distances(start, ends) == [3., 3.]
    assert calls == [([4., 2.12, 5.], [7., 2.12, 8.]), ([4., 2.12, 5.], [10., 2.12, 11.])]
    query.distances_from_agent(ends)
    assert calls[-2:] == [([8., 2.4, -3.], [7., 2.4, 8.]), ([8., 2.4, -3.], [10., 2.4, 11.])]
    np.testing.assert_array_equal(start, before[0])
    np.testing.assert_array_equal(ends, before[1])


def test_topology_blocking_uses_injected_queries_without_simulator():
    path = Path(__file__).resolve().parents[1] / "mapping/topology_graph.py"
    tree = ast.parse(path.read_text())
    scope = {"np": np, "Any": object}
    nodes_to_load = [node for node in tree.body
                     if isinstance(node, ast.ImportFrom) and node.module == "__future__"
                     or isinstance(node, ast.FunctionDef) and node.name == "_blocking_node"]
    exec(compile(ast.Module(body=nodes_to_load, type_ignores=[]), str(path), "exec"), scope)
    calls = []
    nodes = [NS(idx=0, position=np.array([0., 0., 0.])), NS(idx=1, position=np.array([2., 1., 0.]))]
    mapper = NS(nodes=nodes, current_node_idx=0, initial_position=np.array([4., 5., 1.5]),
                geometry_queries=NS(node_distances=lambda start, ends: calls.append((start, ends)) or [2., .5]))
    assert scope["_blocking_node"](mapper, np.array([2.2, 1., 0.]), block_current=False) is nodes[1]
    np.testing.assert_array_equal(calls[0][0], [6.2, 6., 1.5])
    np.testing.assert_array_equal(calls[0][1], [[4., 5., 1.5], [6., 6., 1.5]])


def test_isaac_queries_use_same_planner_and_batch_http(tmp_path):
    grid = np.ones((5, 9), dtype=np.uint8)
    grid[:, 4] = 0
    bridge = bridge_at(tmp_path, FakeMap(grid, np.arange(9), np.arange(5)), FakeEnv((1, 2, 1.5)))
    client = MemoryClient(bridge)
    query = IsaacMapperQueries(client, [1., 2., 1.5])
    trace, explored = bridge._trajectory.read_bytes(), list(bridge.planner.wm.explored_positions)
    points = np.array([[1, 2, -5], [7, 2, 10], [4, 2, 1.5], [100, 2, 1.5]] * 200, dtype=float)
    assert query.navigable_mask(points).tolist() == [True, True, False, False] * 200
    assert [len(row['positions']) for row in client.commands] == [256, 256, 256, 32]
    assert not query.path_exists([1, 2, 0], [7, 2, 0])
    assert query.distance_from_agent([7, 2, 0]) == math.inf
    query.camera_position = np.array([6., 2., 9.])
    assert query.distance_from_agent([7, 2, -3]) == 1.
    assert query.path_exists([6, 2, -100], [6, 2, 100])
    point = [6.123456789012345, 2.123456789012345, 1.5]
    pairs = [{"start": point, "end": point} for _ in range(257)]
    rows = query.paths(pairs)
    assert all(row["path_length"] == 0. for row in rows)
    assert [len(row["pairs"]) for row in client.commands[-5:]] == [64, 64, 64, 64, 1]
    assert bridge._trajectory.read_bytes() == trace
    assert bridge.planner.wm.explored_positions == explored and not bridge.env.calls
    assert query.navigable_mask(np.empty((0, 3))).tolist() == [] and query.paths([]) == []


@pytest.mark.parametrize("mutation", ["domain", "count", "echo", "bool", "path", "length"])
def test_untrusted_query_responses_are_rejected(mutation):
    def respond(action, **kwargs):
        if action == "walkability_xy":
            row = {"position": [1., 2.], "walkable": True}
        else:
            row = {"start": [1., 2.], "end": [2., 2.], "path": [[1., 2.], [2., 2.]], "path_length": 1.}
        result = {"geometry_domain": "freemap_xy", "results": [row]}
        if mutation == "domain": result["geometry_domain"] = "navmesh"
        if mutation == "count": result["results"] = []
        if mutation == "echo": row["position"] = [2., 2.]
        if mutation == "bool": row["walkable"] = 1
        if mutation == "path": row["path"][0] = [1.1, 2.]
        if mutation == "length": row["path_length"] = 2.
        return result
    query = IsaacMapperQueries(NS(command=respond), [1, 2, 1.5])
    with pytest.raises(ValueError):
        if mutation in {"path", "length"}:
            query.path_exists([1, 2, 1.5], [2, 2, 1.5])
        else:
            query.navigable_mask([[1, 2, 1.5]])


@pytest.mark.parametrize("field,value", [("depth_convention", "range"), ("depth_unit", "mm"),
                                         ("size_hw", [0, 64])])
def test_camera_contract_rejects_invalid_inputs(field, value):
    frame = frame_for()
    frame[field] = value
    with pytest.raises(ValueError):
        camera_inputs(frame)


def test_mapper_receives_true_pose_dimensions_and_explicit_height_band(monkeypatch, tmp_path):
    from PIL import Image
    frame = frame_for()
    frame.update(rgb_path="rgb.png", depth_path="depth.npy")
    packet = {"frame": frame, "capture_root": str(tmp_path)}
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    depth = np.ones((48, 64), dtype=np.float32)
    np.save(tmp_path / "depth.npy", depth)
    seen = {}
    class Mapper:
        def __init__(self, intrinsic, **kwargs): seen.update(intrinsic=intrinsic, kwargs=kwargs)
        def initialize(self, position, rotation, env): seen.update(initial=(position, rotation, env))
        def update(self, *args): seen['update'] = args
    monkeypatch.setitem(sys.modules, "mapper_with_process_obs", NS(Instruct_Mapper=Mapper))
    bridge = IsaacMapper(packet, NS(), perceiver=object(), save_dir=tmp_path, device="cpu:0",
                         floor_world_z=.08, ceiling_world_z=1.68)
    kwargs = seen["kwargs"]
    assert kwargs["floor_height"] == pytest.approx(-1.42)
    assert kwargs["ceiling_height"] == pytest.approx(.18)
    assert kwargs["image_shape"] == (48, 64)
    np.testing.assert_array_equal(seen["initial"][0], [2., -3., 1.5])
    bridge.update(packet, episode_idx=0, episode_step=1, on_node_flag=False, current_node_idx=0)
    np.testing.assert_array_equal(seen["update"][1], depth)
    np.testing.assert_array_equal(seen["update"][2], [2., -3., 1.5])
    np.testing.assert_array_equal(np.load(tmp_path / "depth.npy"), depth)
    frame["intrinsics"][0][0] += 1
    with pytest.raises(ValueError, match="calibration changed"):
        bridge.update(packet, episode_idx=0, episode_step=2, on_node_flag=False, current_node_idx=0)


@pytest.mark.parametrize('points', [[[1, 2], [3, 4], [5, 6]], [1, 2, 3], np.zeros((2, 3, 3))])
def test_query_does_not_reshape_malformed_point_batches(points):
    from unittest.mock import Mock
    client = NS(command=Mock(side_effect=AssertionError('No malformed requests')))
    query = IsaacMapperQueries(client, [0, 0, 1.5])
    with pytest.raises(ValueError, match='N by 3'):
        query.navigable_mask(points)
    client.command.assert_not_called()
    assert query.navigable_mask([]).tolist() == []


def test_original_graph_exports_world_goals_without_mutating_local_nodes(monkeypatch, tmp_path):
    from planning import exploration_policy as policy
    from scene.contracts import ObjectNodeSnapshot
    from unittest.mock import Mock

    grid = np.ones((5, 9), dtype=np.uint8)
    grid[:, 4] = 0
    bridge = bridge_at(tmp_path, FakeMap(grid, np.arange(9), np.arange(5)), FakeEnv((1, 1, 1.5)))
    client = MemoryClient(bridge)
    adapter = object.__new__(IsaacMapper)
    adapter.queries = IsaacMapperQueries(client, [1., 1., 1.5])
    adapter._distances, adapter.candidates = {}, []
    nodes = [NS(idx=i, position=np.array(p, dtype=float), state=int(i == 0), room_idx=int(i == 2),
                has_frontier=i != 0, frontier_idxs=np.array([[i, 0]]), objects=[0] if i < 2 else [])
             for i, p in enumerate([(0, 0, 0), (1, 0, 0), (5, 0, 0)])]
    rooms = [NS(nodes=nodes[:2], state=0), NS(nodes=nodes[2:], state=0)]
    mapper = NS(initial_position=np.array([1., 1., 1.5]), current_position=np.zeros(3),
        current_node_idx=0, floor_height=-1.42, nodes=nodes, room_nodes=rooms,
        objects=[NS(uid='observed:1')], geometry_queries=adapter.queries, traj=[0],
        get_edges=lambda: [(0, 1), (1, 2)])
    adapter.mapper = mapper
    reset = Mock()
    monkeypatch.setattr(policy, '_reset_process_pointclouds', reset)
    mapper.explore_in_room = lambda room: policy.explore_in_room(mapper, room)
    mapper.explore_after_fully_explored = lambda: policy.explore_after_fully_explored(mapper)
    mapper.find_closet_viewpoint_in_room = lambda room: policy.find_closest_viewpoint_in_room(mapper, room)
    before = np.array([n.position for n in nodes])
    trace = bridge._trajectory.read_bytes()
    adapter.refresh(client, {'position': [1., 1., 1.5]})
    assert [row['node_idx'] for row in adapter.candidates] == [1]
    selected, source, candidates = adapter.select(choose_room=Mock(side_effect=AssertionError('Local first')))
    assert selected.idx == 1 and selected.position == (2., 1., 1.5) and source == 'current_room'
    reset.assert_called_once_with(mapper)
    assert candidates[0]['raw_distance_m'] == 1.
    fields = adapter.snapshot_fields(objects=(ObjectNodeSnapshot('observed:1', 'chair'),))
    assert fields['viewpoints'][1].pose.position == (2., 1., 1.5)
    assert fields['frontiers'][1].metadata['reachable'] is False
    assert fields['frontiers'][1].metadata['path_length'] is None
    assert fields['rooms'][0].objects == ('observed:1',)
    assert fields['rooms'][0].neighbors == ('room:1',)
    assert fields['rooms'][0].centroid[2] == pytest.approx(.08)
    np.testing.assert_array_equal([n.position for n in nodes], before)
    assert bridge._trajectory.read_bytes() == trace
    assert nodes[1].state == 0
    adapter.arrive(1)
    assert mapper.current_node_idx == 1 and mapper.traj == [0, 1] and nodes[1].state == 0
    def visit(node):
        node.state, node.has_frontier = 1, False
    mapper.change_state = visit
    adapter.mark_observed(1)
    assert nodes[1].state == 1 and not adapter.candidates
    nodes[2].has_frontier = False
    assert adapter.select()[0] is None


@pytest.mark.parametrize('semantic', [False, True])
def test_unreachable_local_frontier_does_not_block_reachable_other_room(monkeypatch, tmp_path, semantic):
    from planning import exploration_policy as policy
    from unittest.mock import Mock

    grid = np.ones((5, 9), dtype=np.uint8)
    grid[:, 4] = 0
    bridge = bridge_at(tmp_path, FakeMap(grid, np.arange(9), np.arange(5)), FakeEnv((1, 1, 1.5)))
    client = MemoryClient(bridge)
    adapter = object.__new__(IsaacMapper)
    adapter.queries = IsaacMapperQueries(client, [1., 1., 1.5])
    adapter._distances, adapter.candidates = {}, []
    nodes = [NS(idx=i, position=np.array(p, dtype=float), state=int(i == 0), room_idx=int(i == 2),
                has_frontier=i != 0, frontier_idxs=np.array([[i, 0]]), objects=[])
             for i, p in enumerate([(0, 0, 0), (5, 0, 0), (1, 0, 0)])]
    rooms = [NS(nodes=nodes[:2], state=0), NS(nodes=nodes[2:], state=0)]
    mapper = NS(initial_position=np.array([1., 1., 1.5]), current_position=np.zeros(3),
        current_node_idx=0, nodes=nodes, room_nodes=rooms, geometry_queries=adapter.queries)
    adapter.mapper = mapper
    reset = Mock()
    monkeypatch.setattr(policy, '_reset_process_pointclouds', reset)
    mapper.explore_in_room = lambda room: policy.explore_in_room(mapper, room)
    mapper.explore_after_fully_explored = lambda: policy.explore_after_fully_explored(mapper)
    mapper.find_closet_viewpoint_in_room = lambda room: policy.find_closest_viewpoint_in_room(mapper, room)
    adapter.refresh(client, {'position': [1., 1., 1.5]})
    choose = Mock(side_effect=lambda available: available[0]) if semantic else None
    selected, source, candidates = adapter.select(choose_room=choose)
    assert selected is not None and selected.idx == 2
    assert selected.position == (2., 1., 1.5)
    assert source == ('room_semantic_policy' if semantic else 'room_distance_policy')
    assert [row['node_idx'] for row in candidates] == [2]
    reset.assert_called_once_with(mapper)
    if semantic:
        choose.assert_called_once()
        assert [room.room_idx for room in choose.call_args.args[0]] == [1]
    assert policy.find_closest_nodes(mapper, [nodes[1]]) is None
    assert policy.find_closest_nodes(mapper, [nodes[1], nodes[2]]) is nodes[2]


def test_observation_batch_uses_original_clouds_and_finishes_once(monkeypatch):
    from unittest.mock import Mock
    cloud = NS(voxel_down_sample=Mock(return_value='merged_cloud'))
    monkeypatch.setitem(sys.modules, 'mapping_utils.geometry', NS(gpu_merge_pointcloud=Mock(return_value=cloud)))
    adapter = object.__new__(IsaacMapper)
    adapter._step, adapter._arrival_node = 0, None
    adapter._distances, adapter.candidates = {1: 2.}, [1]
    adapter.mapper = NS(current_obj_indices=[2], current_node_idx=0, current_pcd='current',
        current_navigable_pcd='nav', pcd_resolution=.05,
        process_nav_pcd=NS(is_empty=lambda: False), traversable_pcd_all=NS(is_empty=lambda: False),
        get_nodes=Mock(), update_obj=Mock())
    adapter.update = Mock()
    adapter.begin_observation()
    assert adapter.mapper.current_obj_indices == []
    packet = {'capture_root': '.', 'frame': {'rgb_path': 'rgb.png'}, 'pose': {'yaw': .3}}
    adapter.add(packet)
    adapter.mapper.current_obj_indices.append(4)
    adapter.add(packet)
    assert not adapter.candidates and not adapter._distances
    assert adapter.finish_observation() is True
    args = adapter.mapper.get_nodes.call_args.args
    assert args[0] == ['merged_cloud', 'merged_cloud'] and args[2:] == (None, 0, 2)
    np.testing.assert_array_equal(args[1], [.3, .3])
    adapter.mapper.update_obj.assert_called_once_with(0, [4])
    assert adapter.finish_observation() is False
    assert adapter.mapper.get_nodes.call_count == 1


@pytest.mark.parametrize('confidence', [.9, np.float32(.9), np.array(.9)])
def test_mapper_serializes_scalar_confidence(monkeypatch, confidence):
    from mapping import map_serializer
    monkeypatch.setattr(map_serializer, '_object_size', lambda obj: np.ones(3))
    mapper = NS(objects=[NS(confidence=confidence, tag='chair', position=np.array([1., 2., .3]))], room_nodes=[])
    result = map_serializer.to_json_save_node_info(mapper)
    assert result['objects'][0]['confidence'] == .9
    assert result['objects'][0]['position'] == [1., 2., .3]
    json.dumps(result, allow_nan=False)
