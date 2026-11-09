"""Camera pitch geometry for viewpoint scoring and recheck execution."""
import ast
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import pytest
from navigation_core.view_geometry import (
    camera_pitch, habitat_path_to_strive_points, interpolate_polyline,
    optical_rotation_towards, pitch_rotation, rank_view_proposals,
    rotation_towards, score_points_view, score_projection, truncate_at_stop_radius,
)
from planning.viewpoint_policy import build_check_again_viewpoints, build_view_proposals, score_object_view


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


@pytest.fixture
def legacy_projection():
    """Run the unchanged historical projector without importing perception packages."""
    path = Path(__file__).resolve().parents[1] / "mapping_utils/geometry.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "project_to_camera")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["project_to_camera"]


def _object(points, position=None, bbox=(100., 80., 430., 370.)):
    points = np.asarray(points)
    return NS(pcd=NS(point=NS(positions=NS(
        cpu=lambda: NS(numpy=lambda: points), shape=points.shape))),
        bbox=bbox, position=np.zeros(3) if position is None else position)


def _legacy_view(project, obj, intrinsic, position, rotation, distance, rho=0., size=(640, 480)):
    """Freeze pre-extraction score_object_view's projection/support contract."""
    points = obj.pcd.point.positions.cpu().numpy()
    projected = np.asarray(project(obj.pcd, intrinsic, position, rotation))
    if projected.ndim < 2 or projected.shape[1] == 0:
        return None
    pixels = np.asarray(projected[:2].T, dtype=np.int32)
    in_frame = ((pixels[:, 0] >= 0) & (pixels[:, 0] < size[0])
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < size[1]))
    front = (points - position) @ rotation[:, 2] < 0
    bbox = obj.bbox
    quality = score_projection(
        camera_points=pixels[in_frame], all_point_count=projected.shape[1],
        object_positions=points[front][in_frame], viewpoint_position=position,
        original_bbox_area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
        original_bbox_length=bbox[2]-bbox[0], original_bbox_width=bbox[3]-bbox[1],
        success_distance=distance, image_width=size[0], image_height=size[1])
    return None if quality is None else (quality, quality.distance_to_target + rho <= distance)


def _assert_quality(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual[1] == expected[1]
    for key, value in asdict(expected[0]).items():
        assert getattr(actual[0], key) == pytest.approx(value, abs=1e-12, rel=1e-12)


def _optical_inputs(obj, intrinsic, position, rotation):
    """Explicit caller conversion: mapper reflection, camera axes, legacy cy-1."""
    reflection = np.diag([1., -1., 1.])
    pose = np.eye(4)
    pose[:3, :3] = reflection @ rotation @ np.diag([1., -1., -1.])
    pose[:3, 3] = reflection @ position
    k = intrinsic.copy()
    k[1, 2] -= 1
    return dict(object_points=obj.pcd.point.positions.cpu().numpy() @ reflection,
                original_bbox=obj.bbox, camera_intrinsic=k, camera_to_map=pose)


@pytest.mark.parametrize("pitch", [0., -.53, .41])
@pytest.mark.parametrize("yaw", [0., .73, -2.1])
@pytest.mark.parametrize("size", [(640, 480), (320, 240)])
def test_habitat_and_optical_scores_match_original_projector(legacy_projection, pitch, yaw, size):
    rng = np.random.default_rng(912)
    position = np.array([1.75, -2.5, .8])
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    rotation = rotation_towards(position[:2], position[:2] + heading, pitch=pitch)
    camera_points = rng.uniform([-3., -2., -4.], [3., 2., 2.], size=(300, 3))
    obj = _object(camera_points @ rotation.T + position)
    intrinsic = np.array([[321.7, 0., size[0]/2], [0., 309.2, size[1]/2], [0., 0., 1.]])
    expected = _legacy_view(legacy_projection, obj, intrinsic, position, rotation, 1.3, .15, size)
    assert expected is not None
    _assert_quality(score_object_view(
        object_node=obj, camera_intrinsic=intrinsic, position=position, rotation=rotation,
        success_distance=1.3, arrival_tolerance=.15, image_width=size[0], image_height=size[1]), expected)
    args = _optical_inputs(obj, intrinsic, position, rotation)
    _assert_quality(score_points_view(**args, success_distance=1.3, arrival_tolerance=.15,
                                    image_width=size[0], image_height=size[1]), expected)
    assert np.linalg.det(args["camera_to_map"][:3, :3]) == pytest.approx(1.)


@pytest.mark.parametrize("points", [[], [[1., 0., 0.]], [[1., 0., -1.]], [[100., 100., 1.]]])
def test_empty_behind_on_plane_or_out_of_frame_has_no_view(legacy_projection, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    intrinsic = np.array([[100., 0., 320.], [0., 100., 240.], [0., 0., 1.]])
    obj = _object(points)
    # This legacy basis becomes optical identity after both explicit conversions.
    rotation = np.diag([1., 1., -1.])
    assert _legacy_view(legacy_projection, obj, intrinsic, np.zeros(3), rotation, 2.) is None
    assert score_object_view(object_node=obj, camera_intrinsic=intrinsic, position=np.zeros(3),
                             rotation=rotation, success_distance=2.) is None
    assert score_points_view(**_optical_inputs(obj, intrinsic, np.zeros(3), rotation),
                             success_distance=2.) is None


@pytest.mark.parametrize("rho,feasible", [(.25-1e-9, True), (.25, True), (.25+1e-9, False)])
def test_stop_budget_rho_boundary_uses_visible_xy_support(legacy_projection, rho, feasible):
    obj = _object([[.75, 0., 0.], [1.5, 0., 0.], [.1, 0., 8.], [-.1, 0., 0.]])
    rotation = rotation_towards(np.zeros(2), np.array([1., 0.]))
    intrinsic = np.array([[100., 0., 320.], [0., 100., 240.], [0., 0., 1.]])
    expected = _legacy_view(legacy_projection, obj, intrinsic, np.zeros(3), rotation, 1., rho)
    assert expected[0].distance_to_target == .75
    assert expected[0].visible_ratio == pytest.approx(2/3)  # behind excluded, out-of-frame retained in denominator
    assert expected[1] == feasible
    _assert_quality(score_object_view(object_node=obj, camera_intrinsic=intrinsic,
        position=np.zeros(3), rotation=rotation, success_distance=1., arrival_tolerance=rho), expected)
    _assert_quality(score_points_view(**_optical_inputs(obj, intrinsic, np.zeros(3), rotation),
                                     success_distance=1., arrival_tolerance=rho), expected)


def _legacy_proposals(project, obj, intrinsic, origin, raw_path, height, pitches, align_pitch):
    path = habitat_path_to_strive_points(raw_path, mapper_initial_position=origin, target_height=height)
    samples = truncate_at_stop_radius(interpolate_polyline(path, spacing=.25),
        object_positions=obj.pcd.point.positions.cpu().numpy(), success_distance=1., stop_criterion=.7)[::-1]
    proposals = []
    for point in samples:
        angles = (float(np.arctan2(obj.position[2]-point[2],
                                  np.linalg.norm(obj.position[:2]-point[:2]))),) if align_pitch else pitches
        scored = []
        for angle in angles:
            rotation = rotation_towards(point[:2], obj.position[:2], pitch=angle)
            if rotation is None:
                break
            view = _legacy_view(project, obj, intrinsic, point, rotation, 1., .15)
            if view is not None:
                scored.append((view[1], view[0].score, angle, view[0]))
        if scored:
            stop, _, pitch, quality = max(scored, key=lambda row: row[:2])
            proposals.append(dict(asdict(quality), position=point, pose=point.tolist(), pitch=pitch,
                                  travel_distance=float(np.linalg.norm(point[:2]-path[0, :2])),
                                  stop_feasible=stop))
    return proposals, samples


@pytest.mark.parametrize("pitches,align_pitch", [((0.,), False), ((0., -.5, .3), False), ((0.,), True)])
def test_shared_builder_matches_habitat_sampling_scoring_and_pitch(legacy_projection, pitches, align_pitch):
    points = np.array([[x, y, z] for x in (1.3, 1.45) for y in (.4, .55) for z in (-.6, -.3, 0.)])
    obj = _object(points, points.mean(axis=0))
    intrinsic = np.array([[388., 0., 320.], [0., 388., 240.], [0., 0., 1.]])
    origin = np.array([7., -3., 1.5])
    local_path = np.array([[-1., -1., .8], [.3, -.4, 1.], [1.1, .45, 1.2], [1.4, .5, 1.3], [3., 2., 2.]])
    raw_path = (local_path + origin)[:, [0, 2, 1]]
    expected, expected_path = _legacy_proposals(legacy_projection, obj, intrinsic, origin, raw_path,
                                               .2, pitches, align_pitch)
    assert expected
    actual, actual_path = build_check_again_viewpoints(
        object_node=obj, camera_intrinsic=intrinsic, mapper_initial_position=origin,
        habitat_path_points=raw_path, target_height=.2, success_distance=1., stop_criterion=.7,
        arrival_tolerance=.15, pitches=pitches, align_pitch=align_pitch)
    np.testing.assert_allclose(actual_path, expected_path, atol=1e-14)
    reflection = np.diag([1., -1., 1.])
    common_path = habitat_path_to_strive_points(raw_path, mapper_initial_position=origin, target_height=.2) @ reflection
    before_path = common_path.copy()
    standard_k = intrinsic.copy()
    standard_k[1, 2] -= 1
    pure, pure_path = build_view_proposals(object_points=points @ reflection,
        object_position=reflection @ obj.position, original_bbox=obj.bbox, camera_intrinsic=standard_k,
        path_points=common_path, target_height=.2, success_distance=1., stop_criterion=.7,
        arrival_tolerance=.15, pitches=pitches, align_pitch=align_pitch)
    np.testing.assert_array_equal(common_path, before_path)
    np.testing.assert_allclose(np.asarray(pure_path) @ reflection, expected_path, atol=1e-14)
    assert len(actual) == len(pure) == len(expected)
    for wrapped, generic, reference in zip(actual, pure, expected):
        generic = {**generic, "position": reflection @ generic["position"],
                   "pose": (reflection @ np.asarray(generic["pose"])).tolist()}
        assert wrapped.keys() == generic.keys()
        for key, value in reference.items():
            assert wrapped[key] == pytest.approx(value, abs=1e-12, rel=1e-12)
            assert generic[key] == pytest.approx(value, abs=1e-12, rel=1e-12)
        if not align_pitch:
            assert wrapped["pitch"] in pitches
    assert max(np.linalg.norm(b-a) for a, b in zip(actual_path, actual_path[1:])) <= .25 + 1e-12
    assert actual_path[0][0] < 3.  # truncation excludes the path tail after the final in-radius sample


def test_path_conversion_is_separate_and_builder_overrides_only_camera_height():
    raw = np.array([[5., 10., -3.], [6., 11., -2.]])
    converted = habitat_path_to_strive_points(raw, mapper_initial_position=np.array([5., -3., 1.]),
                                            target_height=.4)
    np.testing.assert_array_equal(converted, [[0., 0., .4], [1., 1., .4]])
    assert habitat_path_to_strive_points([], mapper_initial_position=np.zeros(3), target_height=0.).shape == (0, 3)
    assert habitat_path_to_strive_points([[1, 2]], mapper_initial_position=np.zeros(3), target_height=0.).shape == (0, 3)
    path = np.array([[7., -2., 9.], [7.1, -2., 10.]])
    before = path.copy()
    args = dict(object_points=np.empty((0, 3)), object_position=np.zeros(3), original_bbox=[0, 0, 1, 1],
                camera_intrinsic=np.eye(3), target_height=.6, success_distance=1., stop_criterion=.7)
    proposals, samples = build_view_proposals(**args, path_points=path)
    assert proposals == []
    np.testing.assert_allclose(samples, [[7.1, -2., .6], [7., -2., .6]])
    np.testing.assert_array_equal(path, before)
    for short in ([], [[1., 2., 3.]]):
        assert build_view_proposals(**args, path_points=short) == ([], [])
    for malformed in ([1., 2., 3.], [[1., 2.]]):
        with pytest.raises(ValueError, match="path_points"):
            build_view_proposals(**args, path_points=malformed)


def test_builder_does_not_invent_unexecutable_pitch():
    obj = _low_object()
    points = obj.pcd.point.positions.cpu().numpy()
    intrinsic = np.array([[388., 0., 320.], [0., 388., 239.], [0., 0., 1.]])
    args = dict(object_points=points, object_position=points.mean(axis=0), original_bbox=obj.bbox,
                camera_intrinsic=intrinsic, path_points=np.zeros((2, 3)), target_height=0.,
                success_distance=.65, stop_criterion=.7)
    level, _ = build_view_proposals(**args)
    down, _ = build_view_proposals(**args, pitches=(0., -np.pi/6))
    assert level and down
    assert all(p["pitch"] == 0. and not p["stop_feasible"] for p in level)
    assert all(p["pitch"] == -np.pi/6 and p["stop_feasible"] for p in down)
    assert build_view_proposals(**args, pitches=())[0] == []
    aligned, _ = build_view_proposals(**args, align_pitch=True)
    expected = np.arctan2(args["object_position"][2], np.linalg.norm(args["object_position"][:2]))
    assert aligned and all(p["pitch"] == pytest.approx(expected) for p in aligned)
    for pitch in (0., -.5, .4):
        rotation = optical_rotation_towards(np.zeros(2), np.array([1., 0.]), pitch=pitch)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(rotation[:, 2], [np.cos(pitch), 0., np.sin(pitch)])
        assert np.linalg.det(rotation) == pytest.approx(1.)
    assert optical_rotation_towards(np.zeros(2), np.zeros(2)) is None


def test_ranking_preserves_terminal_cost_approach_quality_ties_and_rho_motion():
    def proposal(uid, score, travel, stop):
        return dict(uid=uid, score=score, travel_distance=travel, stop_feasible=stop)
    proposals = [proposal("far-quality", .9, 3., True), proposal("near", .7, 1., True),
                 proposal("near-quality", .8, 1., True), proposal("intermediate", .99, .25, False),
                 proposal("far-intermediate", .99, 2., False), proposal("unreachable", 1., np.inf, True),
                 proposal("invalid", 1., np.nan, True), proposal("equal-first", .4, 2., False),
                 proposal("equal-last", .4, 2., False)]
    before = [p.copy() for p in proposals]
    assert [p["uid"] for p in rank_view_proposals(proposals)] == [
        "near-quality", "near", "far-quality", "intermediate", "far-intermediate", "equal-first", "equal-last"]
    assert [p["uid"] for p in rank_view_proposals(proposals, approach=True)] == [
        "far-quality", "near-quality", "near"]
    assert proposals == before
    assert rank_view_proposals([], approach=True) == []
    assert rank_view_proposals([proposal("not-terminal", 1., 0., False)], approach=True) == []
    # The active controller's strict motion filter remains a separate executor budget.
    moving = [proposal(str(distance), .9, distance, True) for distance in (.25-1e-9, .25, .25+1e-9)]
    ranked = rank_view_proposals([p for p in moving if p["travel_distance"] > .25])
    assert len(ranked) == 1 and ranked[0] is moving[-1]


def test_shared_entry_imports_without_simulator_or_perception_dependencies():
    import subprocess
    import sys
    code = '''
import importlib.abc
import sys
class BlockPerception(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'open3d', 'torch', 'quaternion', 'habitat_sim', 'mapping_utils'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockPerception())
from planning.viewpoint_policy import build_view_proposals, score_points_view, rank_view_proposals
'''
    subprocess.run([sys.executable, "-B", "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
