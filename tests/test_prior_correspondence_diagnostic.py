"""Spatial statistics distinguish partial coverage from cross-instance identity."""
import numpy as np
import pytest
from scripts.diagnose_prior_correspondence import overlap
from scripts.diagnose_prior_correspondence import evidence_clouds
from types import SimpleNamespace
from PIL import Image


def test_partial_surface_has_asymmetric_support():
    historical = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    current = historical[:1]
    scores = overlap(current, historical, .1)
    assert scores['current_support'] == 1.
    assert scores['historical_support'] == pytest.approx(1/3)
    assert scores['pooled_support'] == .5


def test_separated_surfaces_have_zero_support():
    scores = overlap(np.array([[0., 0., 0.]]), np.array([[1., 0., 0.]]), .1)
    assert scores['pooled_support'] == 0.
    assert scores['current_median_distance_m'] == 1.


def test_common_rigid_transform_preserves_overlap():
    current = np.array([[0., 0., 0.], [1., 0., 0.]])
    historical = np.array([[0., 0., .05], [1., 0., .05], [2., 0., 0.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    offset = np.array([3., 4., 2.])
    original = overlap(current, historical, .1)
    moved = overlap(current @ rotation.T + offset, historical @ rotation.T + offset, .1)
    assert moved == pytest.approx(original)


def test_observation_mode_preserves_masks_inside_one_fused_instance(tmp_path):
    np.save(tmp_path / 'depth.npy', np.ones((1, 2)))
    observations = []
    for i, mask in enumerate([[[255, 0]], [[0, 255]]]):
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(tmp_path / f'{i}.png')
        observations.append({'observation_id': f'obs{i}', 'object_uids': ['merged'],
            'depth_ref': 'depth.npy', 'mask_ref': f'{i}.png',
            'intrinsics': np.eye(3), 'camera_to_map': np.eye(4)})
    bundle = {'prior_map': SimpleNamespace(objects=[SimpleNamespace(uid='merged')]),
              'observations': observations}
    merged = evidence_clouds(tmp_path, bundle, {}, .05, 6.)
    separate = evidence_clouds(tmp_path, bundle, {}, .05, 6., by_observation=True)
    assert set(separate) == {'obs0', 'obs1'}
    assert len(merged['merged']) == 2
    assert overlap(separate['obs0'], separate['obs1'], .1)['current_support'] == 0.
