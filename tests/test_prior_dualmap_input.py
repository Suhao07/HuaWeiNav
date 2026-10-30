"""Check the independent builder receives the exact shared observations."""
import json
import numpy as np
import pytest
from prior_map.observations import write_frame
from prior_map.dualmap_input import iter_dualmap_inputs


def test_dualmap_unknown_does_not_alias_last_concrete_category():
    from prior_map.dualmap_input import dualmap_vocabulary
    assert dualmap_vocabulary(['chair', 'shelf']) == ['chair', 'shelf', 'unknown']
    assert dualmap_vocabulary(['chair', 'shelf', 'unknown']) == ['chair', 'shelf', 'unknown']


def test_dualmap_clip_endpoints_follow_shared_metric_support(tmp_path):
    from prior_map.observations import metric_depth_validity
    depth = np.array([[0., .5, .51, 4.9, 5.]], dtype=np.float32)
    profile = {'depth_min_m': .5, 'depth_max_m': 5.}
    write_frame(tmp_path, 0, np.zeros((1, 5, 3), dtype=np.uint8), depth, np.eye(3), np.eye(4))
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'capture_profile': profile}))
    row = next(iter_dualmap_inputs(tmp_path))
    np.testing.assert_array_equal(row['depth'][..., 0], np.where(metric_depth_validity(depth, profile), depth, 0))
    np.testing.assert_array_equal(row['depth'][..., 0], [[0., 0., depth[0, 2], depth[0, 3], 0.]])
    np.testing.assert_array_equal(np.load(tmp_path / 'depth/000000.npy'), depth)


def test_dualmap_input_preserves_rgb_depth_and_world_pose(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 251
    depth = np.array([[0, 1.234567, 2], [3, 4, 5]], dtype=np.float32)
    k = np.array([[160., 0, 1], [0, 160., .5], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    write_frame(tmp_path, 5, rgb, depth, k, pose)
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    rows = list(iter_dualmap_inputs(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row['color_name'] == '5'
    assert 'time_stamp' not in row
    np.testing.assert_array_equal(row['color'], rgb)
    np.testing.assert_array_equal(row['depth'][..., 0], depth)
    np.testing.assert_array_equal(row['pose'], pose)
    np.testing.assert_array_equal(row['intrinsics'], k)


@pytest.mark.parametrize('frame_stride', [1, 2])
def test_dualmap_export_keeps_instance_evidence_after_relocation(tmp_path, frame_stride):
    from types import SimpleNamespace
    from prior_map.dualmap_export import export_dualmap_bundle
    from prior_map.bundle import load_bundle
    source = tmp_path / 'capture'
    source.mkdir()
    rgb = np.full((3, 4, 3), 70, dtype=np.uint8)
    k, pose = np.eye(3), np.eye(4)
    frame_ids = (10, 25) if frame_stride == 1 else (10, 17, 25)
    for frame_id in frame_ids:
        write_frame(source, frame_id, rgb, np.ones((3, 4), dtype=np.float32), k, pose)
    (source / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    inputs = list(iter_dualmap_inputs(source, frame_stride=frame_stride))
    assert [row['color_name'] for row in inputs] == ['10', '25']
    assert [row['idx'] for row in inputs] == [0, 1]
    mask = np.zeros((3, 4), dtype=bool)
    mask[:, :2] = True
    obs = SimpleNamespace(idx=1, mask=mask, xyxy=np.array([0, 0, 2, 3]), class_id=0, conf=.8)
    obj = SimpleNamespace(uid='instance-a', class_id=0, max_prob=.9,
        pcd=SimpleNamespace(points=np.array([[1., 2., 3.], [2., 2., 3.]])), observations=[obs])
    root = tmp_path / 'bundle'
    built = export_dualmap_bundle(source, root, [obj], ['chair'], scene_id='scene', frame_stride=frame_stride)
    second = export_dualmap_bundle(source, tmp_path / 'another' / 'bundle',
        [obj], ['chair'], scene_id='scene', frame_stride=frame_stride)
    assert built['manifest']['map_id'] != second['manifest']['map_id']
    moved = tmp_path / 'moved'
    root.rename(moved)
    result = load_bundle(moved)
    assert result['manifest']['map_id'] == built['manifest']['map_id']
    assert result['prior_map'].objects[0].uid == 'instance-a'
    assert result['prior_map'].objects[0].position_xyz == (1.5, 2., 3.)
    evidence = result['observations'][0]
    assert evidence['frame_id'] == 25
    assert evidence['object_uids'] == ['instance-a']
    from PIL import Image
    np.testing.assert_array_equal(np.asarray(Image.open(moved / evidence['mask_ref'])) > 0, mask)
    assert result['viewpoints'][0]['observation_ids'] == [evidence['observation_id']]
