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
