"""Camera pitch geometry for viewpoint scoring and recheck execution."""
from importlib.util import find_spec
from types import SimpleNamespace as NS
import unittest

import numpy as np
from navigation_core.view_geometry import camera_pitch, pitch_rotation, rotation_towards


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
        assert np.isclose(camera_pitch(rotation), pitch)


def _low_object():
    # 低矮物体：顶面 h=-0.4 横跨 0.6~1.2 m，前沿立面在 0.6 m 处向下延伸。
    points = np.array([[d, 0., -.4] for d in np.arange(.6, 1.25, .1)]
                      + [[.6, 0., h] for h in (-.8, -.6)])
    positions = NS(cpu=lambda: NS(numpy=lambda: points), shape=points.shape)
    return NS(pcd=NS(point=NS(positions=positions)), bbox=[0, 0, 100, 100])


@unittest.skipUnless(find_spec("open3d") and find_spec("quaternion"),
                     "Requires maintained STRIVE perception environment")
class RecheckPitchTest(unittest.TestCase):
    """Viewpoint stop distance and tilt execution share one camera model."""

    def test_stop_distance_uses_in_frame_support_and_look_down_recovers_near_surface(self):
        from planning.viewpoint_policy import score_object_view
        intrinsic = np.array([[388., 0., 320.], [0., 388., 240.], [0., 0., 1.]])
        level = rotation_towards(np.zeros(2), np.array([1., 0.]))
        views = {pitch: score_object_view(
            object_node=_low_object(), camera_intrinsic=intrinsic, position=np.zeros(3),
            rotation=level @ pitch_rotation(pitch), success_distance=.65)
            for pitch in (0., -np.radians(30))}
        # 水平视角看不到 0.6 m 处的近端表面，停止距离只能取画面内最近点。
        self.assertAlmostEqual(views[0.][0].distance_to_target, .7)
        self.assertFalse(views[0.][1])
        self.assertAlmostEqual(views[-np.radians(30)][0].distance_to_target, .6)
        self.assertTrue(views[-np.radians(30)][1])

    def _clouds(self, points):
        import open3d as o3d
        device = o3d.core.Device("CPU:0")
        tensor = o3d.t.geometry.PointCloud(device)
        tensor.point.positions = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float64,
                                                device=device)
        legacy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return (("tensor", tensor), ("legacy", legacy))

    def test_projection_indices_preserve_source_order_and_default_api(self):
        from mapping_utils.geometry import project_to_camera
        position = np.array([1., 2., 3.])
        tiny_depth = np.spacing(position[2])
        points = position + np.array([[0., 0., 1.], [1., .5, -2.], [0., 0., 0.],
                                      [-1., -.5, -1.], [0., 0., -4.], [0., 0., 2.],
                                      [0., 0., -tiny_depth]])
        intrinsic = np.array([[80., 0., 160.], [0., 80., 120.], [0., 0., 1.]])
        for kind, cloud in self._clouds(points):
            with self.subTest(cloud=kind):
                default = project_to_camera(cloud, intrinsic, position, np.eye(3))
                explicit = project_to_camera(cloud, intrinsic, position, np.eye(3),
                                             return_indices=False)
                indexed = project_to_camera(cloud, intrinsic, position, np.eye(3),
                                            return_indices=True)
                self.assertEqual(len(default), 3)
                self.assertEqual(len(explicit), 3)
                self.assertEqual(len(indexed), 4)
                self.assertEqual(indexed[3].dtype, np.dtype(np.intp))
                np.testing.assert_array_equal(indexed[3], [1, 3, 4, 6])
                expected = ([200, 80, 160, 160], [99, 159, 119, 119], [2, 1, 4, tiny_depth])
                for actual, values in zip(default, expected):
                    np.testing.assert_array_equal(actual, values)
                for actual, unchanged, with_indices in zip(default, explicit, indexed[:3]):
                    np.testing.assert_array_equal(actual, unchanged)
                    np.testing.assert_array_equal(actual, with_indices)

    def test_projection_without_front_points_returns_empty_arrays(self):
        from mapping_utils.geometry import project_to_camera
        for case, points in (("empty", np.empty((0, 3))),
                             ("behind", np.array([[0., 0., 1.], [1., 2., 3.]])),
                             ("camera_plane", np.array([[0., 0., 0.], [1., 2., 0.]]))):
            for kind, cloud in self._clouds(points):
                with self.subTest(case=case, cloud=kind):
                    default = project_to_camera(cloud, np.eye(3), np.zeros(3), np.eye(3))
                    indexed = project_to_camera(cloud, np.eye(3), np.zeros(3), np.eye(3),
                                                return_indices=True)
                    self.assertEqual(len(default), 3)
                    self.assertEqual(len(indexed), 4)
                    self.assertEqual(indexed[3].dtype, np.dtype(np.intp))
                    for value in (*default, *indexed):
                        self.assertEqual(value.shape, (0,))

    def test_view_without_visible_support_returns_none(self):
        from planning.viewpoint_policy import score_object_view
        intrinsic = np.array([[80., 0., 160.], [0., 80., 120.], [0., 0., 1.]])
        for case, points in (("empty", np.empty((0, 3))),
                             ("behind", np.array([[0., 0., 1.], [1., 2., 3.]])),
                             ("camera_plane", np.array([[0., 0., 0.], [1., 2., 0.]])),
                             ("out_of_frame", np.array([[10., 0., -1.], [0., 10., -1.]]))):
            with self.subTest(case=case):
                cloud = self._clouds(points)[0][1]
                self.assertIsNone(score_object_view(
                    object_node=NS(pcd=cloud, bbox=[0, 0, 100, 100]),
                    camera_intrinsic=intrinsic, position=np.zeros(3), rotation=np.eye(3),
                    success_distance=2., image_width=320, image_height=240))

    def test_near_plane_disagreement_keeps_visible_source_points_and_strict_stop(self):
        from mapping_utils.geometry import project_to_camera
        from navigation_core.view_geometry import score_projection
        from planning.viewpoint_policy import score_object_view
        rotation = rotation_towards([0, 0], [1, 1])
        position = np.array([-1.2465937, 3.876936, -.8])
        points = np.array([position + .1 * rotation[:, 2],
                           position - 2 * rotation[:, 2],
                           position - .1 * rotation[:, 0],
                           position - .25 * rotation[:, 2] + rotation[:, 1],
                           position - 3 * rotation[:, 2]])
        intrinsic = np.array([[80., 0., 160.5], [0., 80., 120.5], [0., 0., 1.]])
        cloud = self._clouds(points)[0][1]
        x, y, depth, indices = project_to_camera(cloud, intrinsic, position, rotation,
                                                return_indices=True)
        # The batched inverse excludes point 2; the independent dot product includes it.
        self.assertLess(((points - position) @ rotation[:, 2])[2], 0.)
        np.testing.assert_array_equal(indices, [1, 3, 4])
        np.testing.assert_allclose(depth, [2., .25, 3.])
        np.testing.assert_array_equal(x, [160, 160, 160])
        np.testing.assert_array_equal(y, [119, -200, 119])
        visible_points = points[[1, 4]]
        distance = np.min(np.linalg.norm(visible_points[:, :2] - position[:2], axis=1))
        self.assertAlmostEqual(distance, 2.)
        for tolerance in (0., .25):
            boundary = distance + tolerance
            for threshold, feasible in ((np.nextafter(boundary, -np.inf), False),
                                        (boundary, True)):
                with self.subTest(tolerance=tolerance, threshold=threshold):
                    expected = score_projection(
                        camera_points=np.array([[160, 119], [160, 119]]), all_point_count=3,
                        object_positions=visible_points, viewpoint_position=position,
                        original_bbox_area=10000., original_bbox_length=100.,
                        original_bbox_width=100., success_distance=threshold,
                        image_width=320, image_height=240)
                    view = score_object_view(
                        object_node=NS(pcd=cloud, bbox=[0, 0, 100, 100]),
                        camera_intrinsic=intrinsic, position=position, rotation=rotation,
                        success_distance=threshold, arrival_tolerance=tolerance,
                        image_width=320, image_height=240)
                    self.assertEqual(view, (expected, feasible))
                    self.assertAlmostEqual(view[0].visible_ratio, 2/3)

    def _agent(self, action_space):
        agent = NS(mapper=NS(current_rotation=rotation_towards(np.zeros(2), np.array([1., 0.]))),
                   on_node_flag=False, update_trajectory=lambda flag: None,
                   env=NS(episode_over=False, sim=NS(sim_config=NS(agents=[NS(action_space=action_space)]))))
        actions = []

        def step(action):
            actions.append(action)
            delta = np.radians(30) if action_space[action].name == "look_up" else -np.radians(30)
            agent.mapper.current_rotation = agent.mapper.current_rotation @ pitch_rotation(delta)

        agent.execution = NS(step=step)
        return agent, actions

    def test_tilt_camera_steps_between_level_and_look_down(self):
        from navigation.goal_approach_controller import recheck_pitches, tilt_camera
        agent, actions = self._agent({4: NS(name="look_up", actuation=NS(amount=30.)),
                                      5: NS(name="look_down", actuation=NS(amount=30.))})
        np.testing.assert_allclose(recheck_pitches(agent), (0., -np.radians(30)))
        self.assertTrue(tilt_camera(agent, -np.radians(30)))
        self.assertEqual(actions, [5])
        self.assertTrue(tilt_camera(agent, 0.))
        self.assertTrue(tilt_camera(agent, 0.))
        self.assertEqual(actions, [5, 4])

    def test_fixed_pitch_platform_keeps_mounted_view(self):
        from navigation.goal_approach_controller import recheck_pitches, tilt_camera
        agent, actions = self._agent({0: NS(name="stop", actuation=None),
                                      2: NS(name="turn_left", actuation=NS(amount=30.))})
        agent.mapper.current_rotation = agent.mapper.current_rotation @ pitch_rotation(-np.radians(15))
        np.testing.assert_allclose(recheck_pitches(agent), (-np.radians(15),))
        self.assertTrue(tilt_camera(agent, 0.))
        self.assertEqual(actions, [])
