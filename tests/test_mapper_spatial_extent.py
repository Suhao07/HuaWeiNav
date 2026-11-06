"""Regression contracts for unbounded ray lookup and empty obstacle evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
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
                  '_visibility_cells', 'is_visible', 'is_visible_in_nav_map'))
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
    from types import MethodType
    ns = production_functions()
    mapper = SimpleNamespace(grid_resolution=.25, voxel_dimension=[500, 500, 20])
    mapper._visibility_cells = MethodType(ns['_visibility_cells'], mapper)
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
