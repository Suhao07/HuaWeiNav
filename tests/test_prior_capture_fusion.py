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
        from unittest.mock import patch
        from planning.viewpoint_policy import build_check_again_viewpoints
        projected = np.array([[80, 240, 400], [60, 180, 120], [1, 1, 1]])
        with patch('planning.viewpoint_policy.project_to_camera', return_value=projected):
            proposals, _ = build_check_again_viewpoints(object_node=_entity(),
                camera_intrinsic=np.eye(3), mapper_initial_position=np.zeros(3),
                habitat_path_points=[[1, 0, 1], [.5, 0, 1]], target_height=0,
                success_distance=1., stop_criterion=.7, image_width=320, image_height=240)
        assert proposals
        for proposal in proposals:
            self.assertAlmostEqual(proposal['visible_ratio'], 2/3)
            self.assertAlmostEqual(proposal['center_score'], 1.)

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

    def test_task_free_fusion_keeps_identity_and_evidence_indices(self):
        first, same, separate = _entity(), _entity(.01, .9), _entity(2.)
        refs, indices = associate_object_entities([], [first, same, separate],
                                                  target="", is_target_tag=lambda tag: False)
        assert indices == [0, 0, 1]
        assert [e.uid for e in refs] == [first.uid, separate.uid]
        assert refs[0].pcd.point.positions.shape[0] == 16
        expected = (.7 ** 3 + .9 ** 3) / (.7 ** 2 + .9 ** 2)
        self.assertAlmostEqual(refs[0].confidence, expected)


    def test_detector_without_target_only_queries_vocabulary(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        from types import SimpleNamespace
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.sam_predictor = SimpleNamespace(set_image=lambda image: None)
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

    def test_single_concept_vocabulary_has_one_nonempty_prompt(self):
        from cv_utils.sam import MMDINO_Grounded_SAM
        detector = MMDINO_Grounded_SAM.__new__(MMDINO_Grounded_SAM)
        detector.classes = ["coffee machine"]
        detector.initialize(target=None)
        assert detector.TEXT_PROMPTs == ["coffee machine ."]
        assert detector.split == 1

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
