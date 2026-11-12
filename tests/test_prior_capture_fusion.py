"""Integration contracts using the maintained detector/point-cloud environment."""
import numpy as np
import unittest

try:
    import open3d as o3d
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Requires maintained STRIVE perception environment") from exc
from mapping.object_fusion import associate_object_entities
from mapping_utils.geometry import gpu_pointcloud_from_array
from mapping_utils.representation import ObjectNode


def word_tokenizer(captions, return_tensors):
    """Count BERT-style tokens as whitespace words plus [CLS]/[SEP]."""
    from types import SimpleNamespace
    return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(captions[0].split()) + 2)))


def _entity(offset=0., score=.7):
    points = np.array([[x, y, z] for x in (0., .03) for y in (0., .03)
                       for z in (0., .03)], dtype=np.float32) + offset
    cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 100), o3d.core.Device("CPU:0"))
    return ObjectNode(cloud, cloud, "chair", score, points.mean(axis=0), 8, None, [0, 0, 1, 1])


class CaptureFusionTest(unittest.TestCase):
    """Exercise shared fusion and task-free detector contracts."""

    def test_empty_target_refinement_requires_no_task_context(self):
        from navigation.bbox_refinement import refine_candidates
        # object() 没有 task 或 mapper；空候选应直接得到空证据集合。
        self.assertEqual(refine_candidates(object(), []), {})

    def test_patrol_reset_records_observation_without_task_metrics(self):
        from objnav_agent_with_process_obs import HM3D_Objnav_Agent
        from types import SimpleNamespace
        from unittest.mock import Mock
        from tempfile import TemporaryDirectory
        camera = SimpleNamespace(position=np.zeros(3), rotation=None)
        env = SimpleNamespace(
            reset=lambda: {"rgb": np.zeros((20, 20, 3), dtype=np.uint8),
                           "depth": np.ones((20, 20, 1), dtype=np.float32)},
            sim=SimpleNamespace(get_agent_state=lambda: SimpleNamespace(sensor_states={"rgb": camera})),
            get_metrics=Mock(side_effect=AssertionError("Patrol accessed task metrics")))
        agent = HM3D_Objnav_Agent.__new__(HM3D_Objnav_Agent)
        agent.env = env
        agent.mapper = SimpleNamespace(initialize=Mock(), update=Mock(),
            object_perceiver=SimpleNamespace(sam=SimpleNamespace(initialize=Mock())))
        agent.final_instruction_verifier = SimpleNamespace(evidence_store=SimpleNamespace(reset=Mock()))
        agent.view_control_state = SimpleNamespace(reset=Mock())
        agent.observation_recorder = Mock()
        with TemporaryDirectory() as directory:
            agent.save_dir = directory
            agent.reset(0, patrol=True)
        env.get_metrics.assert_not_called()
        agent.observation_recorder.assert_called_once_with(agent)
        agent.mapper.object_perceiver.sam.initialize.assert_called_once_with(None)
        self.assertIsNone(agent.task)
        self.assertIsNone(agent.mapper.target)
        self.assertEqual(agent.instruct_goal, "")
        self.assertEqual(agent.episode_steps, 1)

    def test_viewpoint_visibility_uses_actual_image_dimensions(self):
        from types import SimpleNamespace
        from planning.viewpoint_policy import build_check_again_viewpoints
        pixels = np.array([[80, 60], [240, 180], [80, 180], [240, 60],
                           [80, 60], [240, 180], [400, 120], [400, 120]])
        points = np.column_stack((np.full(8, 2.), 1 + (pixels[:, 0] - 160) / 100,
                                  -(pixels[:, 1] - 120) / 100))
        obj = SimpleNamespace(position=np.array([2., 1., 0.]), bbox=[80, 60, 240, 180],
                              pcd=SimpleNamespace(point=SimpleNamespace(positions=torch.tensor(points))))
        proposals, _ = build_check_again_viewpoints(object_node=obj,
            camera_intrinsic=np.array([[100., 0., 160.], [0., 100., 121.], [0., 0., 1.]]),
            mapper_initial_position=np.zeros(3), habitat_path_points=[[1, 0, 1], [1, 0, 1]], target_height=0,
            success_distance=2., stop_criterion=1., image_width=320, image_height=240)
        assert proposals
        for proposal in proposals:
            self.assertAlmostEqual(proposal['visible_ratio'], 6/8)
            self.assertAlmostEqual(proposal['center_score'], 1.)

    def test_mapper_initializes_with_injected_perceiver_without_loading_models(self):
        from pathlib import Path
        import subprocess
        import sys
        code = '''
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'cv_utils.sam', 'config_utils'} or fullname.split('.')[0] in {'mmdet', 'mmengine', 'segment_anything'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockModels())
import numpy as np
import quaternion
from cv_utils.image_perceiver import ImagePerceiver
from mapper_with_process_obs import Instruct_Mapper
perceiver = ImagePerceiver(classes=['chair'], device='cpu:0')
intrinsic = np.array([[40., 0., 31.5], [0., 40., 23.5], [0., 0., 1.]])
mapper = Instruct_Mapper(intrinsic, perceiver=perceiver, device='cpu:0', floor_height=-1.5)
mapper.initialize(np.array([2., 1.5, -3.]), quaternion.quaternion(1., 0., 0., 0.), object())
assert mapper.object_perceiver is perceiver
assert perceiver.classes == ['chair']
np.testing.assert_array_equal(mapper.initial_position, [2., -3., 1.5])
np.testing.assert_array_equal(mapper.current_position, [0., 0., 0.])
assert mapper.scene_pcd.is_empty() and mapper.navigable_pcd.is_empty()
assert mapper.node_cnt == 1 and mapper.current_node_idx == 0
assert mapper.lower_img.shape == (48, 64)
assert mapper.trav_map.device.type == 'cpu'
assert 'cv_utils.sam' not in sys.modules and 'config_utils' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_mapper_perceiver_keeps_constructor_arguments(self):
        from unittest.mock import patch, sentinel
        import mapper_with_process_obs as module
        categories = [{'name': 'chair'}, {'name': 'custom category'}]
        expected = ['chair', 'custom category']
        expected.extend(name for name in module.INTEREST_OBJECTS if name not in expected)
        intrinsic = np.array([[40., 0., 31.5], [0., 40., 23.5], [0., 0., 1.]])
        with patch.object(module, 'MMDINOSAM_Perceiver', return_value=sentinel.perceiver) as factory:
            mapper = module.Instruct_Mapper(intrinsic, categories=categories,
                                            no_gpt_seg=True, device='cpu:0')
        factory.assert_called_once_with(classes=expected, no_gpt_seg=True, device='cpu:0')
        self.assertIs(mapper.object_perceiver, sentinel.perceiver)
        self.assertEqual(categories, [{'name': 'chair'}, {'name': 'custom category'}])

    def test_room_segmentation_uses_configured_height_band(self):
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from mapping import room_segmenter
        points = np.array([[0., 0., -1.5], [.2, 0., -1.25], [.4, 0., .4],
                           [.6, 0., .6]], dtype=np.float32)
        cloud = gpu_pointcloud_from_array(points, np.zeros_like(points), o3d.core.Device('CPU:0'))
        floor = cloud.select_by_index([0])
        node = SimpleNamespace(idx=0, room_idx=-1, has_frontier=False)
        with TemporaryDirectory() as directory:
            mapper = SimpleNamespace(scene_pcd=cloud, navigable_pcd=floor, nodes=[node],
                floor_height=-1.5, ceiling_height=.5, save_dir=directory,
                episode_idx=0, update_iterations=1, grid_resolution=.1,
                voxel_dimension=np.array([500, 500, 20]), current_node_idx=0,
                current_global_frontier_map_idxs=[], get_nodes_positions=lambda: np.zeros((1, 3)),
                find_closet_viewpoint_in_room=lambda room: None)
            with patch.object(room_segmenter, 'project_room', wraps=room_segmenter.project_room) as project:
                room_segmenter.segment_room(mapper, 1)
            obstacle_points = project.call_args_list[0].args[0]
            footprint_points = project.call_args_list[1].args[0]
            self.assertEqual(len(obstacle_points), 2)
            np.testing.assert_allclose(sorted(obstacle_points[:, 2]), [-1.25, .4])
            np.testing.assert_allclose(sorted(footprint_points[:, 2]), [-1.5, -1.5, -1.25, .4])
            self.assertEqual(node.room_idx, 0)
            self.assertEqual(mapper.room_nodes[0].nodes, [node])
        np.testing.assert_array_equal(cloud.point.positions.numpy(), points)

    def test_task_free_perceiver_keeps_vocabulary_and_empty_target_group(self):
        from cv_utils.image_perceiver import ImagePerceiver
        from unittest.mock import Mock
        perceiver = ImagePerceiver(classes=["chair"])
        masks = torch.ones((1, 20, 20), dtype=torch.bool)
        perceiver.segment = Mock(return_value=(torch.tensor([[1., 1., 15., 15.]]),
            torch.tensor([.8]), np.array(["chair"]), masks,
            torch.empty((0, 4)), torch.empty(0), torch.empty((0, 20, 20))))
        result = perceiver.perceive(np.zeros((20, 20, 3), dtype=np.uint8), target=None)
        self.assertEqual(result[0].tolist(), ["chair"])
        self.assertEqual(result[5].shape, (0, 4))
        self.assertEqual(result[6].shape, (0, 20, 20))

    def test_perceiver_empty_observation_and_broken_contract_are_distinct(self):
        from cv_utils.image_perceiver import ImagePerceiver
        from unittest.mock import Mock
        perceiver = ImagePerceiver(classes=["chair"])
        empty = (torch.empty((0, 4)), torch.empty(0), np.array([], dtype=str),
                 torch.empty((0, 20, 20)), torch.empty((0, 4)), torch.empty(0),
                 torch.empty((0, 20, 20)))
        perceiver.segment = Mock(return_value=empty)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        self.assertTrue(all(len(value) == 0 for value in perceiver.perceive(image)))
        # 一个类别对应两个框属于协议错误，必须让采集器停止并报告失败。
        broken = list(empty)
        broken[0] = torch.tensor([[1., 1., 15., 15.], [2., 2., 16., 16.]])
        broken[1] = torch.tensor([.8, .7])
        broken[2] = np.array(["chair"])
        broken[3] = torch.ones((2, 20, 20), dtype=torch.bool)
        perceiver.segment.return_value = tuple(broken)
        with self.assertRaises(IndexError):
            perceiver.perceive(image)

    def test_unknown_refinement_keeps_geometry_without_category_posterior(self):
        from types import SimpleNamespace
        from navigation.bbox_refinement import apply_refined_tag
        from mapping.object_fusion import label_probabilities
        from backends.isaac.perception import IsaacObservationMap
        obj = _entity()
        points = obj.pcd.point.positions.cpu().numpy().copy()
        apply_refined_tag(SimpleNamespace(mapper=SimpleNamespace(target="display")), obj,
                          "unknown", np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(float(obj.confidence), 0.)
        self.assertEqual(label_probabilities(obj), {})
        observation_map = IsaacObservationMap.__new__(IsaacObservationMap)
        observation_map.mapper = SimpleNamespace(objects=[obj], initial_position=np.zeros(3))
        observation_map.latest = {}
        published, = observation_map.objects()
        self.assertEqual(published.uid, obj.uid)
        self.assertEqual(published.metadata["label_probabilities"], {})
        np.testing.assert_array_equal(observation_map.points(obj.uid), points)

    def test_zero_weight_votes_do_not_inflate_positive_evidence(self):
        from copy import deepcopy
        from mapping.object_fusion import accumulate_label_vote, label_probabilities
        for tag in ("chair", "table"):
            for n, confidence in ((8, 0.), (0, .9)):
                obj = _entity()
                before = deepcopy((obj.num_list, obj.conf_list, obj.tag, obj.confidence))
                accumulate_label_vote(obj, tag, n, confidence)
                self.assertEqual((obj.num_list, obj.conf_list, obj.tag, obj.confidence), before)
                self.assertEqual(label_probabilities(obj), {"chair": 1.})
        for tag in ("chair", "table"):
            obj = _entity(score=0.)
            accumulate_label_vote(obj, tag, 8, 0.)
            self.assertEqual(label_probabilities(obj), {})
            accumulate_label_vote(obj, tag, 8, .9)
            self.assertEqual(obj.tag, tag)
            self.assertEqual(obj.num_list[tag], 8)
            self.assertEqual(obj.confidence, .9)
            self.assertEqual(label_probabilities(obj), {tag: 1.})

    def test_target_alias_keeps_observed_votes_through_background_fusion(self):
        from mapping.object_fusion import label_probabilities
        first, observed = _entity(), _entity(.01, .9)
        refs, _ = associate_object_entities([first], [observed], target="seat",
                                             is_target_tag=lambda tag: tag == "chair")
        self.assertIs(refs[0], first)
        self.assertEqual(first.tag, "seat")
        self.assertEqual(first.num_list, {"seat": 8})
        self.assertEqual(first.conf_list, {"seat": .9})
        background = _entity(.01, .2)
        refs, indices = associate_object_entities(refs, [background], target="seat",
                                                  is_target_tag=lambda tag: False)
        self.assertEqual(indices, [0])
        self.assertEqual(first.num_list, {"seat": 8, "chair": 8})
        self.assertEqual(first.tag, "seat")
        self.assertAlmostEqual(label_probabilities(first)["seat"], .81 / (.81 + .04))

    def test_unknown_observations_still_merge_geometrically(self):
        from mapping.object_fusion import label_probabilities
        first, same = _entity(score=0.), _entity(.01, 0.)
        refs, indices = associate_object_entities([], [first, same], target="", is_target_tag=lambda tag: False)
        self.assertEqual(indices, [0, 0])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].uid, first.uid)
        self.assertEqual(refs[0].pcd.point.positions.shape[0], 16)
        self.assertEqual(label_probabilities(refs[0]), {})

    def test_task_free_fusion_keeps_identity_and_evidence_indices(self):
        first, same, separate = _entity(), _entity(.01, .9), _entity(2.)
        refs, indices = associate_object_entities([], [first, same, separate],
                                                  target="", is_target_tag=lambda tag: False)
        assert indices == [0, 0, 1]
        assert [e.uid for e in refs] == [first.uid, separate.uid]
        assert refs[0].pcd.point.positions.shape[0] == 16
        expected = (.7 ** 3 + .9 ** 3) / (.7 ** 2 + .9 ** 2)
        self.assertAlmostEqual(refs[0].confidence, expected)

    def test_duplicate_surfaces_merge_but_contained_part_stays(self):
        from mapping.object_fusion import label_probabilities, merge_duplicate_instances
        def plane(size, copies=1):
            grid = np.arange(0., size, .05, dtype=np.float32)
            points = np.array([[x, y, 0.] for x in grid for y in grid] * copies, dtype=np.float32)
            cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 100), o3d.core.Device("CPU:0"))
            return ObjectNode(cloud, cloud, "table", .7, points.mean(axis=0), len(points), None, [0, 0, 1, 1])
        # 重复观察叠加的点不影响表面覆盖；局部部件仅单向包含，不合并。
        whole, duplicate, part = plane(1., copies=3), plane(1.), plane(.2)
        duplicate.num_list["desk"], duplicate.conf_list["desk"] = 100, .5
        survivors, merged = merge_duplicate_instances([whole, part, duplicate], .05)
        self.assertEqual([e.uid for e in survivors], [whole.uid, part.uid])
        self.assertEqual(merged, {duplicate.uid: whole.uid})
        # 被并实例的少数类别票随合并保留：p(l)∝c_l²n_l。
        self.assertEqual(whole.num_list, {"table": 1600, "desk": 100})
        p = label_probabilities(whole)
        self.assertAlmostEqual(p["desk"], .25 * 100 / (.49 * 1600 + .25 * 100))
        self.assertEqual(whole.tag, "table")


    def test_detector_without_target_only_queries_vocabulary(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        from types import SimpleNamespace
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.sam_predictor = SimpleNamespace(set_image=lambda image: None)
        detector.tokenizer, detector.max_text_tokens = word_tokenizer, 256
        detector.classes = ["chair", "table", "door", "window"]
        detector.initialize(target=None)
        detector.BOX_THRESHOLD = .2
        calls = []
        def predict(image, prompt):
            calls.append(prompt)
            return {"bboxes": [], "scores": [], "labels": np.array([], dtype=int)}
        detector._run_grounding = predict
        output = detector.segment(np.zeros((32, 32, 3), dtype=np.uint8), target=None)
        assert calls == detector.TEXT_PROMPTs
        assert output[0].shape == (0, 4)
        assert output[3].shape == (0, 32, 32)

    def test_target_detection_gets_masks_when_vocabulary_is_empty(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        from cv_utils.image_perceiver import ImagePerceiver
        from types import SimpleNamespace
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.sam_predictor = SimpleNamespace(set_image=lambda image: None)
        detector.tokenizer, detector.max_text_tokens = word_tokenizer, 256
        detector.classes = ["table"]
        detector.initialize(target="chair")
        detector.BOX_THRESHOLD, detector.NMS_IOU_THRESHOLD = .2, .8
        target_box = [[2., 2., 20., 20.]]
        detector._run_grounding = lambda image, prompt: (
            {"bboxes": target_box, "scores": [.9], "labels": np.array([0])} if prompt == "chair"
            else {"bboxes": [], "scores": [], "labels": np.array([], dtype=int)})
        detector._predict_masks = lambda image, boxes: torch.ones((len(boxes), 32, 32), dtype=torch.bool)
        output = detector.segment(np.zeros((32, 32, 3), dtype=np.uint8), target="chair")
        assert output[0].shape == (0, 4) and output[3].shape == (0, 32, 32)
        assert output[4].shape == (1, 4) and output[6].shape == (1, 32, 32)
        perceiver = ImagePerceiver.__new__(ImagePerceiver)
        perceiver.segment = lambda *args: output
        result = perceiver.perceive(np.zeros((32, 32, 3), dtype=np.uint8), target="chair")
        assert list(result[4]) == ["chair"] and result[6].shape == (1, 32, 32)

    def test_single_concept_vocabulary_has_one_nonempty_prompt(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.tokenizer, detector.max_text_tokens = word_tokenizer, 256
        detector.classes = ["coffee machine"]
        detector.initialize(target=None)
        assert detector.TEXT_PROMPTs == ["coffee machine ."]

    def test_grounding_receives_bgr_while_rgb_input_stays_unchanged(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.TEXT_THRESHOLD = .25
        received = []
        def infer(**kwargs):
            received.append(kwargs["inputs"])
            return {"predictions": [{"bboxes": []}]}
        detector.grounding_model = infer
        rgb = np.array([[[255, 17, 0]]], dtype=np.uint8)
        detector._run_grounding(rgb, "chair")
        np.testing.assert_equal(received[0], [[[0, 17, 255]]])
        np.testing.assert_equal(rgb, [[[255, 17, 0]]])
        assert received[0].flags.c_contiguous

    def test_indexed_nearest_distance_matches_dense_reference(self):
        from mapping.object_fusion import nearest_distances
        from mapping_utils.geometry import pointcloud_distance
        rng = np.random.default_rng(17)
        a = rng.normal(size=(51, 3)).astype(np.float32)
        b = rng.normal(size=(79, 3)).astype(np.float32)
        source = gpu_pointcloud_from_array(a, np.zeros(a.shape), o3d.core.Device('CPU:0'))
        reference = gpu_pointcloud_from_array(b, np.zeros(b.shape), o3d.core.Device('CPU:0'))
        indexed = nearest_distances(source, reference)
        dense = pointcloud_distance(source, reference)
        np.testing.assert_allclose(indexed.numpy(), dense.numpy(), atol=2e-6, rtol=1e-5)
        np.testing.assert_equal(indexed.numpy() < .1, dense.numpy() < .1)

    def test_aabb_gate_rejects_separate_clouds_and_keeps_overlap(self):
        from mapping.object_fusion import clouds_may_overlap
        assert clouds_may_overlap(_entity().pcd, _entity(.01).pcd)
        assert not clouds_may_overlap(_entity().pcd, _entity(2.).pcd)
