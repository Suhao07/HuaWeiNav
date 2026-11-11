from types import SimpleNamespace

import numpy as np
import pytest

from mapping.frontier_extractor import adaptive_intersection_distance, append_traversable_candidate


def test_adaptive_intersection_distance_uses_default_without_frontiers():
    """Empty frontier input should keep the conservative default radius."""

    radius, nearest = adaptive_intersection_distance([], np.array([0.0, 0.0, 0.0]))

    assert radius == 2.5
    assert nearest is None


def test_adaptive_intersection_distance_scales_nearest_frontier():
    """Nearest frontier distance controls the local visibility radius."""

    clusters = [
        np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        np.array([[0.0, 2.0, 0.0]]),
    ]

    radius, nearest = adaptive_intersection_distance(
        clusters,
        np.array([0.0, 0.0, 0.0]),
        default_distance=2.5,
        scale=1.2,
    )

    assert nearest == 1.0
    assert radius == 1.2


def test_adaptive_intersection_distance_caps_at_default():
    """Far frontiers should not expand the local crop beyond the default."""

    clusters = [np.array([[10.0, 0.0, 0.0]])]

    radius, nearest = adaptive_intersection_distance(
        clusters,
        np.array([0.0, 0.0, 0.0]),
        default_distance=2.5,
        scale=1.2,
    )

    assert nearest == 10.0
    assert radius == 2.5


@pytest.mark.parametrize('shared_memory', [False, True])
def test_candidate_camera_height_preserves_traversable_cloud(shared_memory):
    points = np.array([[1., 2., -1.42], [3., 4., -1.42]])
    before = points.copy()
    chosen = points[0] if shared_memory else points[0].copy()
    mapper = SimpleNamespace(traversable_pcd=points, current_position=np.array([0., 0., 0.]),
        find_closest_point_in_pc=lambda center, cloud: chosen,
        check_traversability=lambda start, end: np.array_equal(end, before[0]))
    candidates = []
    cells = np.array([[1, 2]])
    append_traversable_candidate(mapper, candidates, center=np.array([1., 2., 0.]),
        has_frontier=True, frontier_idxs=cells, current_position=np.zeros(3))
    assert len(candidates) == 1
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(candidates[0][0], [1., 2., 0.])
    assert not np.shares_memory(candidates[0][0], points)
    assert candidates[0][1] is True and candidates[0][2] is cells


@pytest.mark.parametrize('device', ['CPU:0', 'CUDA:0'])
@pytest.mark.parametrize('values', [[], [-1, -1], [0, -1, 2, 0, 2, 1]])
def test_mapper_dbscan_labels_are_dense_integers(device, values):
    import ast
    from pathlib import Path
    from unittest.mock import Mock

    o3d = pytest.importorskip('open3d')
    if device == 'CUDA:0' and not o3d.core.cuda.is_available():
        pytest.skip('Open3D CUDA unavailable')
    root = Path(__file__).parents[1]
    expected = np.asarray(values, dtype=np.int32)
    tensor = o3d.core.Tensor(expected, device=o3d.core.Device(device))
    cloud = Mock()
    cloud.cluster_dbscan.return_value = tensor
    scope = dict(np=np, pcd=cloud, pcd_removed=cloud, frontiers_pcd=cloud)
    mapper = ast.parse((root / 'mapper_with_process_obs.py').read_text())
    assignments = [n for n in ast.walk(mapper) if isinstance(n, ast.Assign)
                   and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                           and child.func.attr == 'cluster_dbscan' for child in ast.walk(n.value))]
    assert len(assignments) == 4
    for node in assignments:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'mapper_labels', 'exec'), scope)
        assert scope['labels'].dtype == np.int32
        np.testing.assert_array_equal(scope['labels'], expected)
    clusterer = ast.parse((root / 'mapping/frontier_clusterer.py').read_text())
    maxima = [n for tree in (mapper, clusterer) for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'max_label'
                                                   for t in n.targets)
              and any(isinstance(child, ast.Name) and child.id == 'labels' for child in ast.walk(n.value))]
    assert len(maxima) == 3
    for node in maxima:
        if not values and not isinstance(node.value, ast.IfExp):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'mapper_label_max', 'exec'), scope)
        assert scope['max_label'] == (max(values) if values else -1)
        for label in range(scope['max_label'] + 1):
            np.testing.assert_array_equal((scope['labels'] == label).nonzero()[0],
                                          np.flatnonzero(expected == label))
