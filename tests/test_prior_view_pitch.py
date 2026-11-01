"""Camera geometry for the prior-map executor's target-facing pitch."""
import numpy as np
from navigation_core.view_geometry import rotation_towards


def test_horizontal_default_preserves_baseline_rotation():
    np.testing.assert_allclose(rotation_towards(np.array([0., 0.]), np.array([1., 0.])),
                               [[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]])


def test_pitched_camera_faces_target_in_mapper_coordinates():
    position = np.array([2., -3., 1.4])
    for target in [np.array([4., 1., .4]), np.array([-1., 2., 2.])]:
        direction = target-position
        pitch = np.arctan2(direction[2], np.linalg.norm(direction[:2]))
        rotation = rotation_towards(position[:2], target[:2], pitch=pitch)
        np.testing.assert_allclose(-rotation[:, 2], direction/np.linalg.norm(direction))
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-15)
        # Mapper (x, z, y) is reflected relative to Habitat; preserve that convention.
        assert np.isclose(np.linalg.det(rotation), -1.)
