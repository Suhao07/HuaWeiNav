"""Geometric and portable-IO contracts for observation-based priors."""
import json
import numpy as np
import pytest
from prior_map.observations import write_frame, read_records, backproject
from prior_map.bundle import write_bundle, load_bundle
from prior_map.contracts import PriorMapData, PriorObject
from backends.habitat.capture import WORLD_TO_MAP, OPTICAL_TO_GL, select_positions


def test_panorama_action_count_matches_runtime_turn_increment():
    from backends.habitat.walk_capture import panorama_action_count
    assert panorama_action_count(10) == 36
    assert panorama_action_count(30) == 12
    for turn in (0, -10, 361, 17):
        with pytest.raises(ValueError):
            panorama_action_count(turn)


def test_metric_depth_roundtrip_and_projection(tmp_path):
    depth = np.array([[0., 1.123456], [2., 3.]], dtype=np.float32)
    write_frame(tmp_path, 0, np.zeros((2, 2, 3), dtype=np.uint8), depth, np.eye(3), np.eye(4))
    restored = np.load(tmp_path / "depth/000000.npy")
    assert np.array_equal(depth, restored)
    assert len(read_records(tmp_path / "frames.jsonl")) == 1
    points = backproject(restored, np.eye(3), np.eye(4))
    np.testing.assert_allclose(points, [[1.123456, 0, 1.123456], [0, 2, 2], [3, 3, 3]])


def test_right_handed_camera_and_spread():
    assert np.linalg.det(WORLD_TO_MAP @ OPTICAL_TO_GL) == 1
    np.testing.assert_equal(WORLD_TO_MAP @ [0, 1, 0], [0, 0, 1])
    np.testing.assert_equal(WORLD_TO_MAP @ OPTICAL_TO_GL @ [0, 0, 1], [0, 1, 0])
    chosen = select_positions(np.array([[0., 0, 0], [.1, 0, 0], [2., 0, 0]]), .5, 3)
    assert len(chosen) == 2


def test_bundle_relocatable_and_bad_reference(tmp_path):
    root = tmp_path / "map"
    root.mkdir()
    (root / "image.png").write_bytes(b"evidence")
    prior = PriorMapData(scene_id="scene", frame_id="map_z_up", objects=(PriorObject(uid="o1", label="chair", position_xyz=(1, 2, 3)),))
    obs = [{"observation_id": "f1", "object_uids": ["o1"], "image_ref": "image.png", "camera_to_map": np.eye(4).tolist(), "intrinsics": np.eye(3).tolist()}]
    views = [{"uid": "v1", "kind": "observation_pose", "observation_ids": ["f1"], "pose_in_map": np.eye(4).tolist()}]
    write_bundle(root, map_id="map1", revision=1, prior_map=prior, observations=obs, viewpoints=views)
    moved = tmp_path / "moved"
    root.rename(moved)
    assert load_bundle(moved)["prior_map"].objects[0].position_xyz == (1, 2, 3)
    obs[0]["object_uids"] = ["absent"]
    (moved / "observations.jsonl").write_text(json.dumps(obs[0]) + "\n")
    with pytest.raises(ValueError, match="Unknown objects"):
        load_bundle(moved)


def test_wrong_pose_has_front_surface_conflicts(tmp_path):
    from prior_map.geometry_diagnostic import diagnose_capture
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.full((32, 32), 2., dtype=np.float32)
    k = np.array([[30., 0, 15.5], [0, 30., 15.5], [0, 0, 1]])
    write_frame(tmp_path, 0, rgb, depth, k, np.eye(4))
    wrong = np.eye(4)
    wrong[2, 3] = .3
    write_frame(tmp_path, 1, rgb, depth, k, wrong)
    pair = diagnose_capture(tmp_path, stride=2)['pairs'][0]
    assert pair['front_conflict_count'] == pair['projected_count']
    assert pair['consistent_fraction_all_projected'] == 0


def test_depth_edge_stratification_preserves_interior_pose_error(tmp_path):
    from prior_map.geometry_diagnostic import depth_edge_mask, diagnose_capture
    depth = np.full((32, 32), 2., dtype=np.float32)
    depth[:, 16:] = 3.
    edge = depth_edge_mask(depth, .1)
    assert edge[:, 15:17].all()
    assert edge.sum() == 64
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    k = np.array([[30., 0, 15.5], [0, 30., 15.5], [0, 0, 1]])
    depth[:] = 2.
    write_frame(tmp_path, 0, rgb, depth, k, np.eye(4))
    wrong = np.eye(4)
    wrong[2, 3] = .3
    write_frame(tmp_path, 1, rgb, depth, k, wrong)
    pair = diagnose_capture(tmp_path, stride=2)['pairs'][0]
    assert pair['front_conflict_near_depth_edge'] == 0
    assert pair['front_conflict_away_from_depth_edge'] == pair['projected_count']


def test_evidence_audit_flags_coview_disjoint_masks_without_pruning(tmp_path):
    from PIL import Image
    from prior_map.evidence_audit import audit_bundle
    root = tmp_path / 'audit'
    root.mkdir()
    masks = [np.pad(np.ones((2, 2), dtype=np.uint8), ((0, 2), (0, 2))),
             np.pad(np.ones((2, 2), dtype=np.uint8), ((2, 0), (2, 0)))]
    observations = []
    for index, mask in enumerate(masks):
        Image.fromarray(mask * 255).save(root / f'{index}.png')
        observations.append({'observation_id': str(index), 'frame_id': 0,
            'object_uids': ['cabinet1'], 'mask_ref': f'{index}.png',
            'detector_label': ['cabinet', 'shelf'][index],
            'camera_to_map': np.eye(4).tolist(), 'intrinsics': np.eye(3).tolist()})
    prior = PriorMapData(scene_id='scene', frame_id='map_z_up',
        objects=(PriorObject(uid='cabinet1', label='cabinet', exact=True),))
    write_bundle(root, map_id='map', revision=1, prior_map=prior, observations=observations, viewpoints=[])
    before = (root / 'observations.jsonl').read_bytes()
    report = audit_bundle(root)
    assert report['objects_with_multiple_labels'] == 1
    assert report['objects_with_disjoint_same_frame_masks'] == 1
    assert report['objects_with_multiple_frames'] == 0
    assert (root / 'observations.jsonl').read_bytes() == before


def test_capture_selection_uses_path_distance_across_walls():
    from backends.habitat.capture import order_positions
    points = np.array([[0., 0, 0], [.1, 0, 0], [2., 0, 0]])
    # A 与 B 欧氏相邻，但隔墙路径长；A 与 C 在同一侧。
    costs = np.array([[0., 10., 2.], [10., 0., 8.], [2., 8., 0.]])
    def path_distance(a, b):
        i = np.flatnonzero(np.all(points == a, axis=1))[0]
        j = np.flatnonzero(np.all(points == b, axis=1))[0]
        return costs[i, j]
    selected = select_positions(points, .5, 2, path_distance)
    np.testing.assert_equal(selected, points[:2])
    ordered = order_positions(list(points), path_distance)
    np.testing.assert_equal(ordered, points[[0, 2, 1]])


def test_capture_plans_for_probes_excluded_by_camera_clearance():
    candidates = np.array([[0., 0, 0], [.8, 0, 0]])
    probes = np.array([[-.5, 0, 0], [.1, 0, 0], [1.3, 0, 0]])
    selected = select_positions(candidates, 1., 2, coverage_points=probes)
    assert len(selected) == 2
    distances = np.linalg.norm(np.array(selected)[:, None] - probes[None], axis=2)
    assert (distances.min(axis=0) <= 1.).all()
    # 两个候选自身距离小于半径；仍需第二个位置覆盖边界需求点。


def test_semantic_audit_uses_full_mask_majority_and_retains_conflicting_votes():
    from prior_map.semantic_audit import summarize_semantic_evidence
    rows = [dict(object_uids=['a'], detector_label='cabinet', dominant_gt_label='door',
                 dominant_gt_id=gt, dominant_fraction_full_mask=fraction)
            for gt, fraction in [(1, .9), (2, .8), (3, .4)]]
    report = summarize_semantic_evidence(rows)
    assert report['strong_mask_count'] == 2
    assert report['predicted_instances_with_conflicting_gt_votes'] == 1
    assert report['instance_votes']['a'] == {1: 1, 2: 1}


def test_historical_camera_pose_resolves_base_height_and_heading():
    from backends.habitat.prior_viewpoints import camera_to_base_pose
    camera = np.eye(4)
    camera[:3, :3] = WORLD_TO_MAP @ OPTICAL_TO_GL
    camera[:3, 3] = [2., 3., 2.25]
    base = camera_to_base_pose(camera, 1.25)
    np.testing.assert_allclose(base[:3, 3], [2., 3., 1.])
    np.testing.assert_allclose(base[:3, 0], [0., 1., 0.])
    assert np.linalg.det(base[:3, :3]) == pytest.approx(1)


def test_navmesh_receipt_binds_scene_and_generated_geometry(tmp_path):
    from backends.habitat.navmesh_build import verify_build_receipt
    receipt = tmp_path / 'build_receipt.json'
    record = {'status': 'complete', 'navmesh_sha256': 'mesh-a',
              'scene_sha256': 'scene-a', 'settings': {'agent_radius': .1}}
    receipt.write_text(json.dumps(record))
    assert verify_build_receipt(receipt, 'mesh-a', 'scene-a') == record['settings']
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_build_receipt(receipt, 'mesh-b', 'scene-a')
    with pytest.raises(ValueError, match='scene differs'):
        verify_build_receipt(receipt, 'mesh-a', 'scene-b')
    record['status'] = 'running'
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='incomplete'):
        verify_build_receipt(receipt, 'mesh-a', 'scene-a')


def test_inverse_depth_sampling_preserves_planar_geometry_and_pose_bias():
    from prior_map.geometry_diagnostic import interpolate_inverse_depth
    v, u = np.mgrid[0:8, 0:8]
    depth = 1 / (.5 + .01 * u + .02 * v)
    uv = np.array([[2.35, 4.1], [5.7, 1.25]])
    expected = 1 / (.5 + .01 * uv[:, 0] + .02 * uv[:, 1])
    sampled = interpolate_inverse_depth(depth, uv)
    np.testing.assert_allclose(sampled, expected, rtol=1e-12)
    # 插值仅修正取样；真实相机 z 偏差仍保留在残差中。
    np.testing.assert_allclose(expected - .3 - sampled, [-.3, -.3])
