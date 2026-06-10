import numpy as np

from mapping.frontier_extractor import adaptive_intersection_distance


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
