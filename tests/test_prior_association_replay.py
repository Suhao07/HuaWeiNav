"""Association diagnostics must replay the source geometry and evidence scope."""
import numpy as np
import pytest
from PIL import Image

from prior_map.observations import load_fusion_observation
from scripts.diagnose_prior_associations import replay_geometry


def receipt(**overrides):
    return dict(voxel_m=.05, max_depth_m=6., capture_manifest={}, **overrides)


def test_compacted_evidence_cannot_reproduce_source_association():
    with pytest.raises(ValueError, match='complete source'):
        replay_geometry(receipt(compaction_receipt='compaction_receipt.json'))


def test_unrecognized_policy_is_not_silently_unfiltered():
    with pytest.raises(ValueError, match='Unknown depth confidence'):
        replay_geometry(receipt(confidence_policy='new_policy'))


def test_replay_discards_low_confidence_surface_from_recorded_depth(tmp_path):
    # A low-confidence foreground occupies most pixels. Ignoring confidence
    # makes clustering prefer it over the smaller valid background surface.
    depth = np.ones((30, 30), dtype=np.float32)
    depth[:, 27:] = 2.
    confidence = np.zeros((30, 30), dtype=np.uint8)
    confidence[:, 27:] = 1
    np.save(tmp_path / 'depth.npy', depth)
    Image.fromarray(confidence).save(tmp_path / 'confidence.png')
    Image.fromarray(np.full((30, 30), 255, np.uint8)).save(tmp_path / 'mask.png')
    Image.fromarray(np.zeros((30, 30, 3), np.uint8)).save(tmp_path / 'rgb.png')
    record = dict(observation_id='obs', depth_ref='depth.npy', mask_ref='mask.png',
                  confidence_ref='confidence.png', image_ref='rgb.png',
                  intrinsics=[[50., 0., 0.], [0., 50., 0.], [0., 0., 1.]],
                  camera_to_map=np.eye(4).tolist(), detector_label='chair',
                  detector_score=.7, bbox_xyxy=[0, 0, 30, 30])
    geometry = replay_geometry(receipt(confidence_policy='fused_depth_requires_confidence_ge_1'))
    entity, _ = load_fusion_observation(tmp_path, record, **geometry)
    points = entity.pcd.point.positions.cpu().numpy()
    assert len(points) >= 10
    np.testing.assert_allclose(points[:, 2], 2.)
    geometry['min_confidence'] = None
    unfiltered, _ = load_fusion_observation(tmp_path, record, **geometry)
    assert np.any(unfiltered.pcd.point.positions.cpu().numpy()[:, 2] == 1.)
