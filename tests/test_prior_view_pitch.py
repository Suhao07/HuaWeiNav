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
