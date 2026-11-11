"""Regression contracts for unbounded ray lookup and empty obstacle evidence."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from matplotlib.path import Path as Polygon
from bresenham import bresenham

ROOT = Path(__file__).resolve().parents[1]


def production_functions():
    """Load geometry methods without starting detectors or Habitat."""
    namespace = dict(np=np, Path=Polygon, bresenham=bresenham)
    tree = ast.parse((ROOT / 'mapping_utils/projection.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('translate_point_to_grid', 'translate_single_grid_to_point')]
    mapper = ast.parse((ROOT / 'mapper_with_process_obs.py').read_text())
    methods = next(n for n in mapper.body if isinstance(n, ast.ClassDef)).body
    nodes.extend(n for n in methods if isinstance(n, ast.FunctionDef) and n.name in
                 ('calculate_intersections', 'get_closest_disances_and_points',
                  '_visibility_cells', '_visibility_grid_points', 'is_visible',
                  'is_visible_in_nav_map', 'update_obj_node', 'update_edges'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'production_geometry', 'exec'), namespace)
    return namespace


def test_ray_query_is_translation_equivariant_across_both_map_edges():
    """Crossing either old boundary preserves physical ray endpoints."""
    ns = production_functions()
    mapper = SimpleNamespace(grid_resolution=0.25, voxel_dimension=[500, 500, 20])
    x, y = np.meshgrid(np.arange(-4, 5) * .25, np.arange(-4, 5) * .25)
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    origin = np.zeros(3)
    fn = ns['calculate_intersections']
    reference = fn(mapper, points, origin, num_rays=16, max_distance=2)
    for offset in (np.array([150., 175., 0]), np.array([-150., -175., 0])):
        actual = fn(mapper, points + offset, origin + offset, num_rays=16, max_distance=2)
        np.testing.assert_allclose(actual - offset[:2], reference, atol=1e-10)


def test_empty_obstacle_evidence_preserves_observed_ground_endpoints():
    """An empty obstacle band still permits ground-supported ray updates."""
    fn = production_functions()['get_closest_disances_and_points']
    angles = np.linspace(.1, 1.4, 200)
    points = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(len(angles))])
    distances, endpoints = fn(None, points, np.empty((0, 3)), np.zeros(3), max_distance=1.8)
    assert len(endpoints) > 0
    assert np.isfinite(distances).all()
    np.testing.assert_allclose(np.linalg.norm(endpoints, axis=1), 1)


def test_visibility_matches_dense_dilation_and_crosses_map_edges():
    """Preserve in-window obstacle dilation and visibility under translation."""
    import cv2
    ns = production_functions()
    mapper = SimpleNamespace(grid_resolution=.25, voxel_dimension=[500, 500, 20])
    mapper._visibility_cells = MethodType(ns['_visibility_cells'], mapper)
    mapper._visibility_grid_points = MethodType(ns['_visibility_grid_points'], mapper)
    x, y = np.meshgrid(np.arange(-4, 5)*.25, np.arange(-4, 5)*.25)
    ground = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    obstacles = np.array([[0., 0., 0.]])
    grid = ns['translate_point_to_grid']
    cells = grid(ground, .25, mapper.voxel_dimension)[:, :2]
    blocked = grid(obstacles, .25, mapper.voxel_dimension)[:, :2]
    dense = np.zeros((500, 500)); dense[cells[:, 0], cells[:, 1]] = 1
    obstacle_map = np.zeros_like(dense); obstacle_map[blocked[:, 0], blocked[:, 1]] = 1
    dilated = cv2.dilate(obstacle_map, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    expected = set(map(tuple, np.argwhere(cv2.bitwise_and(dense, cv2.bitwise_not(dilated)) != 0)))
    assert mapper._visibility_cells(ground, obstacles) == expected
    for offset in (np.zeros(3), np.array([150., 175., 0]), np.array([-150., -175., 0])):
        left = np.array([-1., 0., 0.]) + offset
        right = np.array([1., 0., 0.]) + offset
        assert not ns['is_visible'](mapper, left, right, ground+offset, obstacles+offset)
        assert ns['is_visible'](mapper, left, right, ground+offset, np.empty((0, 3)))
        assert ns['is_visible_in_nav_map'](mapper, left, right, ground+offset)


def _legacy_grid(mapper, points):
    return (np.floor(np.asarray(points) / mapper.grid_resolution).astype(int)
            + np.asarray(mapper.voxel_dimension) // 2 - 1)[..., :2]


def _legacy_pair(mapper, point1, point2, ground, obstacles):
    """Old per-pair XY norm/first argmin, dilation and endpoint exclusion."""
    nearest = [ground[np.argmin(np.linalg.norm(ground[:, :2] - point[:2], axis=1))]
               for point in (point1, point2)]
    start, end = [_legacy_grid(mapper, point) for point in nearest]
    blocked = {(x + dx, y + dy) for x, y in _legacy_grid(mapper, obstacles)
               for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1))}
    cells = set(map(tuple, _legacy_grid(mapper, ground))) - blocked
    ray = list(bresenham(*start, *end))[1:-1]
    return tuple((*start, *end)), sum(cell not in cells for cell in ray)


class _Positions:
    """Shared-memory CPU tensor stand-in counting actual cloud conversions."""

    def __init__(self, values, counts):
        self.values, self.counts = values, counts

    def __getitem__(self, key):
        return self.values[key]

    def cpu(self):
        self.counts['cpu'] += 1
        return self

    def numpy(self):
        self.counts['numpy'] += 1
        return self.values


class _Cloud:
    def __init__(self, values, counts=None):
        self.counts = counts if counts is not None else dict(cpu=0, numpy=0)
        self.point = SimpleNamespace(positions=_Positions(values, self.counts))

    def select_by_index(self, indices):
        return _Cloud(self.point.positions.values[indices], self.counts)


def _mapper(ns, ground, node_points, object_points=()):
    mapper = SimpleNamespace(grid_resolution=.25, voxel_dimension=[500, 500, 20],
                             floor_height=-1.42, navigable_pcd=_Cloud(ground),
                             nodes=[SimpleNamespace(idx=i, position=p.copy())
                                    for i, p in enumerate(node_points)],
                             objects=[SimpleNamespace(position=p.copy()) for p in object_points],
                             update_obj=Mock())
    for name in ('_visibility_cells', '_visibility_grid_points'):
        setattr(mapper, name, Mock(wraps=MethodType(ns[name], mapper)))
    return mapper


def _scene(offset):
    offset = np.asarray(offset)
    x, y = np.meshgrid(np.arange(-12, 13)*.25, np.arange(-4, 5)*.25)
    ground = np.column_stack((x.ravel(), y.ravel(), np.linspace(-1.5, -1.3, x.size))) + offset
    nodes = np.array([[-2., 0., 2.4], [2., 0., -.7], [-2., .75, 1.1],
                      [2., .75, .6], [-1.875, 0., 9.]]) + offset
    objects = np.array([[-1.75, 0., .34], [1.75, 0., .8], [1.875, .75, 7.],
                        [-2.125, .75, -.3], [-1.75, 0., 99.]]) + offset
    # A thick wall yields both blocked (>5 cells) and unobstructed pairs.
    x, y = np.meshgrid(np.arange(-3, 4)*.25, np.arange(-4, 5)*.25)
    obstacles = np.vstack((np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))),
                           [-2., 0., 0.])) + offset
    # The last obstacle blocks both ends of an adjacent node/object ray, not its empty interior.
    return ground, nodes, objects, obstacles


@pytest.mark.parametrize('offset', [(0., 0., 0.), (150., 175., 0.), (-150., -175., 0.)])
def test_visibility_grid_points_keeps_first_xy_tie_and_input_heights(offset):
    """Batch snapping is exact, including ties, negative grids and empty input."""
    ns = production_functions()
    offset = np.asarray(offset)
    ground = np.array([[.25, 0., 80.], [0., 0., -9.], [-.25, -.25, 3.]]) + offset
    points = np.array([[.125, 0., -9.], [-.25, -.25, -30.], [.125, 0., 99.]]) + offset
    before = ground.copy(), points.copy()
    mapper = _mapper(ns, ground, [])
    ns['translate_point_to_grid'] = Mock(wraps=ns['translate_point_to_grid'])
    actual = mapper._visibility_grid_points(points, ground)
    np.testing.assert_array_equal(actual, _legacy_grid(mapper, ground[[0, 2, 0]]))
    ns['translate_point_to_grid'].assert_called_once()
    np.testing.assert_array_equal(ns['translate_point_to_grid'].call_args.args[0], ground[[0, 2, 0]])
    # Reordering equidistant samples changes the selected endpoint, not its height.
    reversed_ground = ground[[1, 0, 2]]
    np.testing.assert_array_equal(mapper._visibility_grid_points(points[:1], reversed_ground),
                                  _legacy_grid(mapper, reversed_ground[:1]))
    ns['translate_point_to_grid'].reset_mock()
    for empty in ([], np.empty((0, 3))):
        assert mapper._visibility_grid_points(empty, ground).shape == (0, 2)
    assert ns['translate_point_to_grid'].call_count == 2
    np.testing.assert_array_equal(ground, before[0])
    np.testing.assert_array_equal(points, before[1])


@pytest.mark.parametrize('offset', [(0., 0., 0.), (150., 175., 0.), (-150., -175., 0.)])
def test_visibility_each_pair_matches_legacy_math_without_mutation(offset):
    ns = production_functions()
    ground, nodes, objects, obstacles = _scene(offset)
    mapper = _mapper(ns, ground, nodes, objects)
    before = [array.copy() for array in (ground, nodes, objects, obstacles)]
    outcomes = []
    for node in nodes:
        for obj in objects:
            _, blocked = _legacy_pair(mapper, node, obj, ground, obstacles)
            actual = ns['is_visible'](mapper, node, obj, ground, obstacles)
            assert actual == (blocked == 0)
            outcomes.append(actual)
    assert any(outcomes) and not all(outcomes)
    assert mapper._visibility_grid_points.call_count == len(nodes) * len(objects)
    for call, (node, obj) in zip(mapper._visibility_grid_points.call_args_list,
                                 ((node, obj) for node in nodes for obj in objects)):
        np.testing.assert_array_equal(call.args[0], [node, obj])
    for actual, original in zip((ground, nodes, objects, obstacles), before):
        np.testing.assert_array_equal(actual, original)
    # Both blocked endpoints are deliberately ignored, even for an empty interior.
    pair = ground[[0, 1]]
    assert ns['is_visible'](mapper, pair[0], pair[1], ground, pair)
    assert ns['is_visible'](mapper, pair[0], pair[0], ground, pair)


@pytest.mark.parametrize('offset', [(0., 0., 0.), (150., 175., 0.), (-150., -175., 0.)])
@pytest.mark.parametrize('object_count', [0, 1, 5])
def test_object_node_round_matches_pairs_and_builds_geometry_once(monkeypatch, offset, object_count):
    """No node*object CPU copies or grids, and no stale visibility next round."""
    ns = production_functions()
    ground, nodes, objects, wall = _scene(offset)
    mapper = _mapper(ns, ground, nodes, objects[:object_count])
    mapper.useful_pcd = _Cloud(np.empty((0, 3)))
    positions = [item.position for item in mapper.nodes + mapper.objects]
    before = [point.copy() for point in positions]
    ground_before = ground.copy()
    ns['bresenham'] = Mock(wraps=bresenham)
    ns['translate_point_to_grid'] = Mock(wraps=ns['translate_point_to_grid'])
    rounds = []
    for round_idx, obstacles in enumerate((wall, np.empty((0, 3))), 1):
        # Strict height-band filtering still excludes evidence at/below the cutoff.
        low = np.array([[-1.5, 0., mapper.floor_height + .88],
                        [-1.25, 0., mapper.floor_height + .5]]) + np.asarray(offset)
        useful = np.vstack((obstacles, low))
        mapper.useful_pcd.point.positions.values = useful
        useful_before = useful.copy()
        expected, rays = [], []
        for node in mapper.nodes:
            visible = []
            for index, obj in enumerate(mapper.objects):
                ray, blocked = _legacy_pair(mapper, node.position, obj.position, ground, obstacles)
                rays.append(ray)
                if not blocked:
                    visible.append(index)
            expected.append((node.idx, visible))
        mapper.update_obj.reset_mock()
        mapper._visibility_cells.reset_mock()
        mapper._visibility_grid_points.reset_mock()
        ns['translate_point_to_grid'].reset_mock()
        ns['bresenham'].reset_mock()
        with monkeypatch.context() as patch:
            norm = Mock(wraps=np.linalg.norm)
            patch.setattr(np.linalg, 'norm', norm)
            ns['update_obj_node'](mapper)
            assert norm.call_count == len(nodes) + object_count
        actual = [call.args for call in mapper.update_obj.call_args_list]
        assert actual == expected  # Includes empty updates and original node/object order.
        rounds.append(actual)
        assert [call.args for call in ns['bresenham'].call_args_list] == rays
        assert mapper._visibility_cells.call_count == 1
        assert mapper._visibility_grid_points.call_count == 2
        assert ns['translate_point_to_grid'].call_count == 4
        assert mapper.navigable_pcd.counts == dict(cpu=round_idx, numpy=round_idx)
        assert mapper.useful_pcd.counts == dict(cpu=round_idx, numpy=round_idx)
        np.testing.assert_array_equal(useful, useful_before)
    if object_count:
        assert rounds[0] != rounds[1]
    for item, position, original in zip(mapper.nodes + mapper.objects, positions, before):
        assert item.position is position
        np.testing.assert_array_equal(position, original)
    np.testing.assert_array_equal(ground, ground_before)


def _graph(mapper, edges):
    """Adjacency with real connectivity checks and an ordered mutation trace."""
    mapper.neighbors = {node.idx: [] for node in mapper.nodes}
    mapper.events = []

    def add(i, j):
        mapper.events.append(('add', i, j))
        mapper.neighbors[i].append(j)
        mapper.neighbors[j].append(i)

    def remove(i, j):
        mapper.events.append(('remove', i, j))
        mapper.neighbors[i].remove(j)
        mapper.neighbors[j].remove(i)

    def connected(i, j):
        seen, pending = set(), [i]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(mapper.neighbors[current])
        result = j in seen
        mapper.events.append(('connected', i, j, result))
        return result

    mapper.add_edge, mapper.remove_edge, mapper.check_connected = add, remove, connected
    mapper.get_edges = lambda: sorted((i, j) for i, adjacent in mapper.neighbors.items()
                                      for j in adjacent if i < j)
    for edge in edges:
        add(*edge)
    mapper.events.clear()


def _legacy_edges(mapper, ground, obstacles):
    rays, counts = [], []
    for i, node in enumerate(mapper.nodes):
        for other in mapper.nodes[i + 1:]:
            j = other.idx
            ray, blocked = _legacy_pair(mapper, node.position, other.position, ground, obstacles)
            rays.append(ray)
            counts.append(blocked)
            if blocked > 5:
                if i in mapper.neighbors[j]:
                    mapper.remove_edge(i, j)
                    if not mapper.check_connected(i, j):
                        mapper.add_edge(i, j)
            elif i not in mapper.neighbors[j]:
                mapper.add_edge(i, j)
    return rays, counts


def _stub_edge_export(ns, mapper):
    # Exercise the whole method, but never write PLY files or create directories.
    mapper.save_dir, mapper.episode_idx, mapper.update_iterations = '/unused', 0, 0
    ns['os'] = SimpleNamespace(makedirs=Mock())
    ns['o3d'] = SimpleNamespace(
        geometry=SimpleNamespace(LineSet=lambda: SimpleNamespace(paint_uniform_color=Mock())),
        utility=SimpleNamespace(Vector3dVector=np.asarray, Vector2iVector=np.asarray),
        io=SimpleNamespace(write_line_set=Mock()))


@pytest.mark.parametrize('offset', [(0., 0., 0.), (150., 175., 0.), (-150., -175., 0.)])
@pytest.mark.parametrize('node_count', [2, 5])
def test_edges_match_legacy_order_connectivity_and_fresh_round_geometry(monkeypatch, offset, node_count):
    ns = production_functions()
    ground, nodes, _, wall = _scene(offset)
    nodes = nodes[:node_count]
    mapper = _mapper(ns, ground, nodes)
    reference = _mapper(ns, ground, nodes)
    mapper.obstacle_pcd = _Cloud(np.empty((0, 3)))
    _graph(mapper, [])
    _graph(reference, [])
    _stub_edge_export(ns, mapper)
    ns['bresenham'] = Mock(wraps=bresenham)
    ns['translate_point_to_grid'] = Mock(wraps=ns['translate_point_to_grid'])
    positions = [node.position for node in mapper.nodes]
    ground_before, wall_before = ground.copy(), wall.copy()
    edge_rounds = []
    for round_idx, obstacles in enumerate((np.empty((0, 3)), wall), 1):
        mapper.obstacle_pcd.point.positions.values = obstacles
        mapper.events.clear()
        reference.events.clear()
        rays, counts = _legacy_edges(reference, ground, obstacles)
        mapper._visibility_cells.reset_mock()
        mapper._visibility_grid_points.reset_mock()
        ns['bresenham'].reset_mock()
        ns['translate_point_to_grid'].reset_mock()
        with monkeypatch.context() as patch:
            norm = Mock(wraps=np.linalg.norm)
            patch.setattr(np.linalg, 'norm', norm)
            ns['update_edges'](mapper)
            assert norm.call_count == node_count
        assert mapper.get_edges() == reference.get_edges()
        assert mapper.events == reference.events
        assert [call.args for call in ns['bresenham'].call_args_list] == rays
        assert mapper._visibility_cells.call_count == 1
        assert mapper._visibility_grid_points.call_count == 1
        assert ns['translate_point_to_grid'].call_count == 3
        assert mapper.navigable_pcd.counts == dict(cpu=round_idx, numpy=round_idx)
        assert mapper.obstacle_pcd.counts == dict(cpu=round_idx, numpy=round_idx)
        edge_rounds.append(mapper.get_edges())
        if round_idx == 2:
            assert max(counts) > 5
            assert any(event[0] == 'remove' for event in mapper.events)
            assert any(event[0] == 'connected' and not event[3] for event in mapper.events)
    if node_count > 2:
        assert edge_rounds[0] != edge_rounds[1]
    for node, position, original in zip(mapper.nodes, positions, nodes):
        assert node.position is position
        np.testing.assert_array_equal(position, original)
    np.testing.assert_array_equal(ground, ground_before)
    np.testing.assert_array_equal(wall, wall_before)


@pytest.mark.parametrize('blocked_count', [5, 6])
@pytest.mark.parametrize('existing_edge', [False, True])
def test_edges_keep_strict_five_cell_threshold_and_restore_bridge(blocked_count, existing_edge):
    ns = production_functions()
    ground = np.column_stack((np.arange(8)*.25, np.zeros(8), np.full(8, -1.42)))
    nodes = ground[[0, 7]].copy()
    nodes[:, 2] = [.4, 2.3]
    # With 5 blocked interior cells, the endpoint is blocked too: it must not count.
    obstacles = ground[[1, 3, 6] if blocked_count == 5 else [1, 3, 5]].copy()
    mapper = _mapper(ns, ground, nodes)
    mapper.obstacle_pcd = _Cloud(obstacles)
    _graph(mapper, [(0, 1)] if existing_edge else [])
    _stub_edge_export(ns, mapper)
    assert _legacy_pair(mapper, *nodes, ground, obstacles)[1] == blocked_count
    ns['update_edges'](mapper)
    assert mapper.get_edges() == ([(0, 1)] if blocked_count == 5 or existing_edge else [])
    if blocked_count == 6 and existing_edge:
        assert mapper.events == [('remove', 0, 1), ('connected', 0, 1, False), ('add', 0, 1)]
    else:
        assert mapper.events == ([('add', 0, 1)] if blocked_count == 5 and not existing_edge else [])
    np.testing.assert_array_equal([node.position for node in mapper.nodes], nodes)
