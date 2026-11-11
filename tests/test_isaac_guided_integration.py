"""Offline Isaac adapter integration with the real shared policy and task reducer."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from backends.isaac import runtime
from harness.tools import NavigationTools
from instruction.contracts import TaskSpec, TargetSpec
from instruction_adapter.view_control import ViewControlState
from motion.contracts import NavigationStatusCode
from navigation.guided import GuidedPolicy, run_guided_loop
from scene.contracts import ObjectNodeSnapshot, Pose3D, ViewpointSnapshot
from verification.contracts import VerificationResult

from test_isaac_runtime import (FakeClient, FakeExplorationMap, FrameClient, _agent,
                                _episode_replay, _frame_packet, _session, _surface_map,
                                _tools, offline_only, projection_fixture)


class PathClient(FakeClient):
    """Answer only explicit endpoint path queries, never synthetic ring requests."""

    def __init__(self, *, reachable=True):
        super().__init__()
        self.reachable = reachable
        self.path = [[0., 0., 1.5], [1., 0., 1.5], [1., 1., 1.5],
                     [3., 1., 1.5], [3., 0., 1.5]]
        self.path_queries = []

    @staticmethod
    def cost(position):
        # Deliberately not straight-line displacement. One physically reachable
        # proposal is within the active-view movement tolerance and must be filtered.
        return 0.1 if position[0] == 3. else 10. + position[0] + 2. * position[1]

    def command(self, action, **arguments):
        assert action == "paths", "View queries must not move, observe or ask for rings"
        assert set(arguments) == {"positions"}
        assert 0 < len(arguments["positions"]) <= 256
        self.commands.append((action, arguments))
        self.path_queries.append(arguments["positions"])
        if len(self.path_queries) == 1:
            if self.reachable:
                self.path[-1] = list(arguments["positions"][1])
            return [{"position": point, "path": self.path if self.reachable and i == 1 else [],
                     "path_length": 5. if self.reachable and i == 1 else None}
                    for i, point in enumerate(arguments["positions"])]
        return [{"position": point, "path": [self.pose["position"], point],
                 "path_length": self.cost(point)} for point in arguments["positions"]]


@pytest.mark.parametrize("fault", ["missing", "extra", "mismatched", "nan", "negative", "empty", "nonfinite", "shape"])
def test_candidate_path_responses_cannot_supply_uncorrelated_or_invalid_costs(tmp_path, fault):
    positions = [[1., 0., 1.5]]
    row = {"position": positions[0], "path": [[0., 0., 1.5], positions[0]], "path_length": 1.}
    if fault == "mismatched":
        row["position"] = [2., 0., 1.5]
    elif fault == "nan":
        row["path_length"] = float("nan")
    elif fault == "negative":
        row["path_length"] = -1.
    elif fault == "empty":
        row["path"] = []
    elif fault == "nonfinite":
        row["path"][0][0] = float("inf")
    elif fault == "shape":
        row["path"] = [[0., 0.], [1., 0.]]
    rows = [] if fault == "missing" else [row, row] if fault == "extra" else [row]
    client = FakeClient()
    client.command = Mock(return_value=rows)
    tools = _tools(_session(), client, tmp_path)
    if fault in {"missing", "extra", "mismatched"}:
        with pytest.raises(ValueError, match="paths"):
            tools._query_paths(positions)
    else:
        assert tools._query_paths(positions)[0]["path_length"] is None
    client.command.assert_called_once_with("paths", positions=positions)
    assert tools.motion.active_request_id is None


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("large_cloud", [False, True])
def test_query_viewpoints_uses_shared_path_samples_and_geodesic_costs(monkeypatch, tmp_path, active, large_cloud):
    from navigation_core.view_geometry import rank_view_proposals
    from planning.viewpoint_policy import build_view_proposals

    client = PathClient()
    packet = _frame_packet(tmp_path, 1, np.ones((20, 20)))
    packet["frame"]["intrinsics"] = [[12., 0., 10.], [0., 12., 10.], [0., 0., 1.]]
    yy, zz = np.meshgrid(np.linspace(-.35, .35, 5), np.linspace(1.15, 1.85, 5))
    points = np.column_stack((np.full(25, 4.), yy.ravel(), zz.ravel()))
    if large_cloud:
        points = np.vstack((points - [3., 0., 0.], points, points + [3., 0., 0.]))
    obj = ObjectNodeSnapshot("obj", "chair", (4., 0., 1.5), bbox2d_xyxy=(7, 7, 13, 13))
    session = _session()
    retained = ViewpointSnapshot("unrelated-view", Pose3D((8., 8., 1.5)), visible_objects=("other",))
    stale = ViewpointSnapshot("old-view", Pose3D((6., 6., 1.5)), visible_objects=("obj",))
    session.snapshot = replace(session.snapshot, objects=(obj,), viewpoints=(retained, stale))
    exploration = FakeExplorationMap()
    exploration.approaches = [(3., 0., 1.5), (2., 2., 1.5)]
    observation_map = SimpleNamespace(points=Mock(return_value=points))
    tools = _tools(session, client, tmp_path, observation_map, exploration)
    tools.last_packet = packet
    state = tools.view_states.setdefault((1, "obj"), ViewControlState())
    if active:
        state.start("obj", {"required_stop_distance": 2.}, {})
    builder = Mock(wraps=build_view_proposals)
    ranker = Mock(wraps=rank_view_proposals)
    monkeypatch.setattr(runtime, "build_view_proposals", builder)
    monkeypatch.setattr(runtime, "rank_view_proposals", ranker)
    views = tools.query_viewpoints("obj")
    assert views
    assert exploration.approach_queries == []
    surface_x = 1. if large_cloud else 4.
    np.testing.assert_allclose(client.path_queries[0],
        [[surface_x - .1 * i, 0., 1.5] for i in range(8)])
    observation_map.points.assert_called_once_with("obj")
    builder.assert_called_once()
    args = builder.call_args.kwargs
    np.testing.assert_array_equal(args["path_points"], client.path)
    np.testing.assert_array_equal(args["object_points"], points)
    np.testing.assert_array_equal(args["stop_reference_position"], obj.position)
    assert (args["target_height"], args["success_distance"], args["stop_criterion"],
            args["arrival_tolerance"], args["pitches"]) == (1.5, 2., 1., .25, (0.,))
    ranker.assert_called_once()
    proposed, sampled_path = build_view_proposals(**args)
    queried = [tuple(p) for batch in client.path_queries[1:] for p in batch]
    assert queried == [tuple(p["pose"]) for p in proposed]
    assert all(position in {tuple(point) for point in sampled_path} for position in queried)
    expected = []
    for proposal in proposed:
        proposal["travel_distance"] = client.cost(proposal["pose"])
        if not active or proposal["travel_distance"] > tools.arrival_tolerance:
            expected.append(proposal)
    expected = rank_view_proposals(expected)
    assert [v["pose"]["position"] for v in views] == [tuple(p["pose"]) for p in expected]
    assert [v["metadata"]["path_length"] for v in views] == [p["travel_distance"] for p in expected]
    assert sum(v["metadata"]["recommended"] for v in views) >= 1
    assert all(v["metadata"]["source"] == "shared_path_sampled_geometry" for v in views)
    assert all(v["metadata"]["visibility"] == "predicted_requires_fresh_verification" for v in views)
    for view in views:
        metadata = view["metadata"]
        distance = float(np.linalg.norm(np.asarray(obj.position)[:2] - view["pose"]["position"][:2]))
        assert metadata["distance_to_target"] == pytest.approx(distance)
        assert metadata["stop_feasible"] == (distance + tools.arrival_tolerance <= 2.)
        assert metadata["distance_source"] == "observed_instance_position_xy"
        assert metadata["stop_reference_position"] == list(obj.position)
        assert metadata["observation_identity"] == {
            "candidate_uid": "obj", "task_revision": 1, "observation_id": "1"}
        assert metadata["visible_surface_distance_source"] == "projected_visible_surface_xy"
    if large_cloud:
        early = next(v["metadata"] for v in views if v["pose"]["position"][0] == 0.)
        assert early["visible_surface_distance"] < 2. < early["distance_to_target"]
        assert not early["stop_feasible"] and not early["recommended"]
    assert session.snapshot.viewpoints[0] is retained
    assert stale not in session.snapshot.viewpoints
    assert tools.motion.active_request_id is None and tools.motion.controller.moves == 0
    assert state.last_selected_index is None and state.attempts == []
    json.dumps(views, allow_nan=False)

    agent, parse_call = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.selected = {"candidate_uid": "obj", "target_id": "t1", "role": "primary"}
    chosen = next(v for v in views if v["metadata"]["recommended"])
    guided.submit(chosen["pose"]["position"], candidate_uid="obj", source="baseline_viewpoint")
    assert state.last_selected_index is not None
    assert tuple(state.proposals[state.last_selected_index].pose) == chosen["pose"]["position"]
    assert state.attempts == [] and guided.candidate_motion_goals == {}
    assert tools.motion.active_intent.goal_pose.position == chosen["pose"]["position"]
    assert tools.motion.controller.moves == 0
    parse_call.assert_not_called()


@pytest.mark.parametrize("offset", [(0., 0., 0.), (12., -7., 2.)])
def test_surface_backoff_uses_three_dimensions_and_first_reachable_path(monkeypatch, tmp_path, offset):
    offset = np.asarray(offset)
    client = FakeClient()
    client.pose["position"] = (np.array([0., 0., 1.5]) + offset).tolist()
    points = np.array([[.1, .1, 20.], [3., 4., 1.5]]) + offset
    original = points.copy()
    obj = ObjectNodeSnapshot("obj", "chair", tuple(np.array([.1, .1, 1.5]) + offset),
        bbox2d_xyxy=(1, 1, 3, 3))
    session = _session()
    session.snapshot = replace(session.snapshot, objects=(obj,))
    tools = _tools(session, client, tmp_path, SimpleNamespace(points=lambda uid: points))
    tools.last_packet = _frame_packet(tmp_path, 1, np.ones((10, 10)))
    rows = []

    def paths(action, **arguments):
        assert action == "paths"
        for i, target in enumerate(arguments["positions"]):
            rows.append({"position": target,
                "path": [client.pose["position"], target] if i >= 2 else [],
                "path_length": 10. - i if i >= 2 else None})
        return rows

    client.command = paths
    builder = Mock(return_value=([], []))
    monkeypatch.setattr(runtime, "build_view_proposals", builder)
    assert tools.query_viewpoints("obj") == []
    np.testing.assert_allclose([r["position"] for r in rows],
        [np.array([3. - .06 * i, 4. - .08 * i, 1.5]) + offset for i in range(8)])
    np.testing.assert_allclose(builder.call_args.kwargs["path_points"], rows[2]["path"])
    np.testing.assert_array_equal(points, original)
    assert tools.exploration_map.approach_queries == []
    assert tools.motion.controller.goals == {}


@pytest.mark.parametrize("distance", [0., .05, .25])
def test_surface_backoff_near_camera_remains_finite(monkeypatch, tmp_path, distance):
    client = PathClient(reachable=False)
    points = np.array([[distance, 0., 1.5]])
    obj = ObjectNodeSnapshot("obj", "chair", tuple(points[0]), bbox2d_xyxy=(1, 1, 3, 3))
    session = _session()
    session.snapshot = replace(session.snapshot, objects=(obj,))
    tools = _tools(session, client, tmp_path, SimpleNamespace(points=lambda uid: points))
    tools.last_packet = _frame_packet(tmp_path, 1, np.ones((10, 10)))
    assert tools.query_viewpoints("obj") == []
    np.testing.assert_allclose(client.path_queries[0],
        [[distance - .1 * i, 0., 1.5] for i in range(max(1, int(distance / .1)))])
    np.testing.assert_array_equal(points, [[distance, 0., 1.5]])


@pytest.fixture
def relation_view_scene(tmp_path):
    client = FakeClient()
    client.reachable = True
    near, joint = (3., 0., 1.5), (1., -2., 1.5)
    def paths(action, **arguments):
        assert action == "paths"
        client.commands.append((action, arguments))
        return [{"position": p, "path": [client.pose["position"], p] if client.reachable else [],
                 "path_length": float(np.linalg.norm(np.asarray(p)[:2])) if client.reachable else None}
                for p in arguments["positions"]]
    client.command = paths
    packet = _frame_packet(tmp_path, 1, np.ones((100, 100)))
    intrinsic = np.array([[80., 0., 50.], [0., 80., 50.], [0., 0., 1.]])
    packet["frame"]["intrinsics"] = intrinsic.tolist()
    yy, zz = np.meshgrid(np.linspace(-.1, .1, 5), np.linspace(1.4, 1.6, 5))
    cloud = np.column_stack((np.full(25, 4.), yy.ravel(), zz.ravel()))
    anchor_cloud = cloud + [0., 2., 0.]
    obj = ObjectNodeSnapshot("obj", "chair", (4., 0., 1.5), bbox2d_xyxy=(40, 40, 60, 60))
    anchor = ObjectNodeSnapshot("anchor", "table", (4., 2., 1.5), bbox2d_xyxy=(40, 40, 60, 60))
    session = _session(spec=TaskSpec("Find the chair near the table.", (
        TargetSpec("t1", "chair"), TargetSpec("t2", "table", role="anchor", terminal=False))))
    session.snapshot = replace(session.snapshot, objects=(obj, anchor))
    exploration = FakeExplorationMap()
    exploration.approaches = [near, joint]
    clouds = {"obj": cloud, "anchor": anchor_cloud}
    tools = _tools(session, client, tmp_path, SimpleNamespace(points=clouds.__getitem__), exploration)
    tools.last_packet = packet
    agent, parse = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.selected = {"candidate_uid": "obj", "target_id": "t1", "role": "primary"}
    pending = [{"predicate_id": "c0", "candidate_uid": "obj", "anchor_uid": "anchor",
                "anchor_concept_id": "t2", "status": "insufficient_evidence"}]
    guided.relation_outcomes[(1, "obj")] = pending
    return SimpleNamespace(guided=guided, clouds=clouds, near=near, joint=joint,
                           intrinsic=intrinsic, parse=parse, pending=pending, obj=obj)


@pytest.mark.parametrize("anchor_state", ["visible", "out_of_view", "missing", "unreachable"])
def test_relation_feedback_requires_predicted_common_view(relation_view_scene, anchor_state):
    from navigation_core.view_geometry import optical_rotation_towards, score_points_view

    scene = relation_view_scene
    guided, obj = scene.guided, scene.obj
    tools, session = guided.tools, guided.tools.session
    if anchor_state == "out_of_view":
        scene.clouds["anchor"] += [0., 0., 20.]
    elif anchor_state == "missing":
        session.snapshot = replace(session.snapshot, objects=(obj,))
    elif anchor_state == "unreachable":
        tools.client.reachable = False
    views = guided.feedback_views("obj")
    if anchor_state != "visible":
        assert views == []
    else:
        assert views and any(v.pose.position == scene.joint for v in views)
        assert all(v.pose.position != scene.near for v in views)
        for view in views:
            pose = np.eye(4)
            pose[:3, :3] = optical_rotation_towards(view.pose.position[:2], obj.position[:2])
            pose[:3, 3] = view.pose.position
            for points in scene.clouds.values():
                assert score_points_view(object_points=points, original_bbox=obj.bbox2d_xyxy,
                    camera_intrinsic=scene.intrinsic, camera_to_map=pose, success_distance=2.,
                    image_width=100, image_height=100) is not None
            assert view.metadata["covisible_pairs"] == [{"predicate_id": "c0", "anchor_uid": "anchor"}]
            assert view.metadata["visibility"] == "predicted_requires_fresh_verification"
        assert sum(v.metadata["recommended"] for v in views) == 1
    assert guided.relation_outcomes[(1, "obj")] == scene.pending
    assert not session.progress.completed and tools.motion.active_request_id is None
    scene.parse.assert_not_called()
    assert all(action == "paths" for action, _ in tools.client.commands)


@pytest.mark.parametrize("anchor_visible", [True, False])
def test_relation_recovery_cannot_bypass_common_view_filter(relation_view_scene, anchor_visible):
    scene, options_seen = relation_view_scene, []
    guided = scene.guided
    guided.tools.exploration_map = FakeExplorationMap(positions=[(6., 0., 1.5)])
    guided.tools.exploration_map.approaches = [scene.near, scene.joint]
    if not anchor_visible:
        scene.clouds["anchor"] += [0., 0., 20.]

    def choose(**kwargs):
        context = json.loads(kwargs["messages"][1]["content"][0]["text"])
        options = context["options"]
        options_seen.extend(options)
        choice = next((row for row in options if row["kind"] == "viewpoint"), options[0])
        result = kwargs["response_format"](action_uid=choice["action_uid"], reason="Acquire missing evidence")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))])

    scene.parse.side_effect = choose
    assert guided.recover({"candidate_uid": "obj", "reason": "need_relation_check"}) == "moving"
    views = [row for row in options_seen if row["kind"] == "viewpoint"]
    assert bool(views) is anchor_visible
    published = guided.tools.session.snapshot.viewpoints
    assert {row["action_uid"] for row in views} == {view.uid for view in published}
    assert all(view.pose.position != scene.near for view in published)
    intent = guided.tools.motion.active_intent
    assert intent.target_object_uid == ("obj" if anchor_visible else None)
    assert not guided.tools.session.progress.completed and guided.tools.motion.controller.moves == 0
    scene.parse.assert_called_once()


@pytest.mark.parametrize("change", ["verified", "revision"])
def test_relation_views_do_not_reuse_resolved_or_old_revision_gaps(relation_view_scene, change):
    scene = relation_view_scene
    guided, session = scene.guided, scene.guided.tools.session
    assert all(v.metadata["source"] == "observed_pair_geometry" for v in guided.feedback_views("obj"))
    old_state = guided.view_state
    if change == "verified":
        scene.pending.append({**scene.pending[0], "anchor_uid": "another", "status": "verified"})
    else:
        from instruction_adapter.task_compat import runtime_plan
        session.install_plan(runtime_plan(replace(session.spec, revision=2)))
    views = guided.feedback_views("obj")
    assert views and all(v.metadata["source"] == "shared_path_sampled_geometry" for v in views)
    if change == "revision":
        assert guided.view_state is not old_state
        assert guided.candidate_view_states[(1, "obj")] is old_state
    assert guided.tools.motion.active_request_id is None and not session.progress.completed
    scene.parse.assert_not_called()


@pytest.mark.parametrize("same_predicate", [True, False])
def test_relation_views_allow_incremental_evidence_for_multiple_anchors(relation_view_scene, same_predicate):
    scene = relation_view_scene
    guided, session = scene.guided, scene.guided.tools.session
    other = ObjectNodeSnapshot("other", "table", (4., 2., 21.5), bbox2d_xyxy=(40, 40, 60, 60))
    session.snapshot = replace(session.snapshot, objects=(*session.snapshot.objects, other))
    scene.clouds["other"] = scene.clouds["anchor"] + [0., 0., 20.]
    scene.pending.append({**scene.pending[0], "anchor_uid": "other",
                          "predicate_id": "c0" if same_predicate else "c1"})
    views = guided.feedback_views("obj")
    assert views
    expected = [{"predicate_id": predicate, "anchor_uid": "anchor"}
                for predicate in (("c0",) if same_predicate else ("c0", "c1"))]
    assert all(v.metadata["covisible_pairs"] == expected for v in views)
    assert all(row["status"] == "insufficient_evidence" for row in scene.pending)
    assert not session.progress.completed
    scene.parse.assert_not_called()


@pytest.mark.parametrize("known_status", [None, "rejected_relation"])
def test_new_anchor_enters_joint_views_without_reopening_rejected_pairs(relation_view_scene, known_status):
    scene = relation_view_scene
    guided, session = scene.guided, scene.guided.tools.session
    assert guided.feedback_views("obj")
    guided.defer_candidate("need_relation_check")
    scene.clouds["anchor"] += [0., 0., 20.]
    assert guided.feedback_views("obj") == []
    other = ObjectNodeSnapshot("new-anchor", "table", (4., 2., 1.5), bbox2d_xyxy=(40, 40, 60, 60))
    scene.clouds["new-anchor"] = scene.clouds["anchor"] - [0., 0., 20.]
    if known_status:
        scene.pending.append({**scene.pending[0], "anchor_uid": "new-anchor", "status": known_status})
    session.snapshot = replace(session.snapshot, objects=(*session.snapshot.objects, other))
    guided.observation_updated()
    views = guided.feedback_views("obj")
    if known_status:
        assert views == []
    else:
        assert guided.deferred_candidates == set() and guided.action_epoch("obj") == 1
        assert views and all(v.metadata["covisible_pairs"] == [
            {"predicate_id": "c0", "anchor_uid": "new-anchor"}] for v in views)
    assert len(scene.pending) == (2 if known_status else 1)
    assert not session.progress.completed and guided.tools.motion.active_request_id is None
    scene.parse.assert_not_called()


def test_unreachable_candidate_has_no_view_or_ring_fallback(tmp_path):
    client = PathClient(reachable=False)
    session = _session()
    obj = ObjectNodeSnapshot("obj", "chair", (4., 0., 1.5), bbox2d_xyxy=(1, 1, 3, 3))
    session.snapshot = replace(session.snapshot, objects=(obj,))
    tools = _tools(session, client, tmp_path,
        SimpleNamespace(points=Mock(return_value=np.array([[4., 0., 1.5]]))))
    tools.last_packet = _frame_packet(tmp_path, 1, np.ones((10, 10)))
    assert tools.query_viewpoints("obj") == []
    np.testing.assert_allclose(client.path_queries[0],
        [[4. - .1 * i, 0., 1.5] for i in range(8)])
    tools.observation_map.points.assert_called_once_with("obj")
    assert tools.motion.controller.goals == {}
    with pytest.raises(ValueError, match="Unknown observed instance"):
        tools.query_viewpoints("not-observed")
    assert len(client.path_queries) == 1


@pytest.mark.parametrize("visible", [True, False])
def test_surface_distance_is_depth_filtered_diagnostic_not_stop_reference(monkeypatch, tmp_path, visible):
    from navigation_core.view_geometry import score_points_view

    points, _, _, _ = projection_fixture()
    hidden = np.array([[.1, .1, 2.], [.01, .01, -1.], [30., 0., 1.]])
    all_points = np.vstack((points, hidden))
    packet = _frame_packet(tmp_path, 73, np.full((10, 10), 1. if visible else 0.))
    obj = ObjectNodeSnapshot("obj", "chair", (.01, .01, 1.5))
    session = _session()
    session.snapshot = replace(session.snapshot, objects=(obj,))
    observation_map = SimpleNamespace(points=Mock(return_value=all_points))
    tools = _tools(session, FrameClient([packet]), tmp_path, observation_map)
    score = Mock(wraps=score_points_view)
    monkeypatch.setattr(runtime, "score_points_view", score)
    evidence = tools.capture_evidence("obj")
    geometry = evidence["geometry"]
    assert geometry["distance_source"] == "observed_instance_position_xy"
    assert geometry["stop_reference_position"] == list(obj.position)
    assert geometry["observation_identity"] == evidence["observation_identity"] == {
        "candidate_uid": "obj", "task_revision": 1, "observation_id": "73"}
    assert geometry["visible_surface_distance_source"] == "depth_visible_surface_xy"
    assert geometry["required_stop_distance"] == 2.
    assert geometry["visible_projected_points"] == (20 if visible else 0)
    if visible:
        assert geometry["distance_to_object"] == pytest.approx(np.sqrt(.0002))
        assert geometry["visible_surface_distance"] == pytest.approx(np.sqrt(13.))
        assert geometry["distance_to_object"] < 2. < geometry["visible_surface_distance"]
        score.assert_called_once()
        np.testing.assert_array_equal(score.call_args.kwargs["object_points"], points)
        assert evidence["view_quality_facts"]["projection_failed"] is False
    else:
        assert geometry["distance_to_object"] is None and geometry["bbox_xyxy"] is None
        assert geometry["visible_surface_distance"] is None
        assert evidence["observation_status"] == "need_observation"
        score.assert_not_called()


@pytest.mark.parametrize("visible", [True, False])
@pytest.mark.parametrize("robot_x", [0., 2. - 1e-9, 2., 2. + 1e-9])
def test_large_cloud_cannot_finish_at_near_edge_before_observed_center_radius(tmp_path, visible, robot_x):
    from instruction_adapter.verifier import hard_stop_constraints_from_evidence
    from navigation_core.view_geometry import optical_rotation_towards

    yy, zz = np.meshgrid(np.linspace(-.2, .2, 5), np.linspace(1.3, 1.7, 5))
    near = np.column_stack((np.full(25, 1.), yy.ravel(), zz.ravel()))
    far = near + [6., 0., 0.]
    points = np.vstack((near, far))
    obj = ObjectNodeSnapshot("obj", "chair", (4., 0., 1.5))
    neighbour = ObjectNodeSnapshot("neighbour", "chair", (.1, 0., 1.5))
    session = _session()
    session.snapshot = replace(session.snapshot, objects=(neighbour, obj))
    front_x = 1. if robot_x == 0. else 7.
    packet = _frame_packet(tmp_path, 73, np.full((20, 20), front_x - robot_x if visible else 0.))
    packet["pose"]["position"] = [robot_x, 0., 1.5]
    camera = np.eye(4)
    camera[:3, :3] = optical_rotation_towards(np.zeros(2), np.array([1., 0.]))
    camera[:3, 3] = packet["pose"]["position"]
    packet["frame"]["camera_to_map"] = camera.tolist()
    packet["frame"]["intrinsics"] = [[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]]
    observation_map = SimpleNamespace(points=Mock(side_effect=lambda uid: (
        points if uid == "obj" else near - [5., 0., 0.])))
    tools = _tools(session, FrameClient([packet]), tmp_path, observation_map)
    evidence = tools.capture_evidence("obj")
    geometry = evidence["geometry"]
    assert geometry["stop_reference_position"] == list(obj.position)
    assert geometry["distance_source"] == "observed_instance_position_xy"
    assert geometry["observation_identity"] == evidence["observation_identity"] == {
        "candidate_uid": "obj", "task_revision": 1, "observation_id": "73"}
    assert geometry["required_stop_distance"] == 2.
    assert hard_stop_constraints_from_evidence(evidence)["satisfied"] == (visible and robot_x >= 2.)
    if visible:
        assert geometry["distance_to_object"] == pytest.approx(4. - robot_x)
        assert geometry["visible_surface_distance"] == pytest.approx(front_x - robot_x)
        if robot_x == 0.:
            assert geometry["visible_surface_distance"] < 2. < geometry["distance_to_object"]
    else:
        assert geometry["distance_to_object"] is None and geometry["visible_surface_distance"] is None
        assert evidence["observation_status"] == "need_observation"
    assert geometry["visible_surface_distance_source"] == "depth_visible_surface_xy"
    assert json.loads((tmp_path / "verification" / "000073-evidence.json").read_text()) == evidence
    assert tools.motion.active_request_id is None
    assert tools.motion.controller.moves == 0


def test_better_view_preserves_verified_relation_context_across_fresh_captures(tmp_path):
    packets = [_frame_packet(tmp_path, index, np.ones((10, 10))) for index in (7, 8)]
    observation_map = _surface_map()
    session = _session()
    session.snapshot = replace(session.snapshot, objects=observation_map.objects())
    tools = _tools(session, FrameClient(packets), tmp_path, observation_map)
    agent, _ = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.selected = {"candidate_uid": "candidate", "target_id": "t1", "role": "primary"}
    state = tools.view_states.setdefault((1, "candidate"), ViewControlState())
    context = {"edge": {"subject_uid": "candidate", "object_uid": "anchor", "status": "verified"}}
    session.constraint_evaluator.evaluate_before_final_verifier = Mock(return_value=SimpleNamespace(
        satisfied=True, evidence={"verified_relation_context": context}))
    tools.verifier.final.verify = Mock(return_value=VerificationResult(False, "need_better_view",
        semantic_satisfied=True, task_revision=1, target_uid="candidate",
        view_objective={"goal": "closer unobstructed view"}))
    assert guided.capture_verification((0., 0., 1.5)).decision == "need_better_view"
    assert state.pinned_relation_context == context
    assert tools.last_evidence["verified_relation_context"] == context
    assert state.pinned_visual_evidence["observation_identity"]["observation_id"] == "7"
    assert not session.progress.completed
    guided.capture_verification((0., 0., 1.5))
    evidence = tools.verifier.final.verify.call_args.args[-1]
    assert evidence["view_control"]["pinned_relation_context"] == context
    assert evidence["observation_identity"]["observation_id"] == "8"
    assert len(state.attempts) == 1 and not session.progress.completed


@pytest.mark.parametrize("phase", ["queued", "running", "reached_unpolled", "cancel_unpolled"])
def test_capture_requires_consumed_terminal_feedback(tmp_path, phase):
    client = FrameClient([_frame_packet(tmp_path, 7, np.ones((10, 10)))])
    observation_map = _surface_map()
    session = _session()
    session.snapshot = replace(session.snapshot, objects=observation_map.objects())
    tools = _tools(session, client, tmp_path, observation_map)
    agent, _ = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.selected = {"candidate_uid": "candidate", "target_id": "t1", "role": "primary"}
    tools.view_states[(1, "candidate")] = ViewControlState()
    guided.submit((1., 0., 1.5), candidate_uid="candidate", source="baseline_viewpoint")
    motion = tools.motion
    if phase in {"running", "reached_unpolled"}:
        client.status = "running" if phase == "running" else "reached"
        motion.controller.advance(motion.active_goal_id)
    elif phase == "cancel_unpolled":
        motion.cancel()
    before = list(client.commands)
    with pytest.raises(RuntimeError, match="terminal motion feedback"):
        tools.capture_evidence("candidate")
    assert client.commands == before
    assert tools.exploration_map.added == [] and tools.last_evidence == {}
    assert not session.progress.completed
    assert not (tmp_path / "verification").exists()
    if phase == "reached_unpolled":
        feedback = motion.poll_active()
        assert feedback.status.status == NavigationStatusCode.REACHED
        guided.motion_finished(feedback)
        assert tools.last_evidence == {}  # Bookkeeping still must not verify.
        assert tools.capture_evidence("candidate")["observation_status"] == "current"


def test_only_new_camera_observation_reopens_candidate_action_epoch(tmp_path):
    image_path = tmp_path / "candidate.png"
    Image.new("RGB", (10, 10), (30, 40, 50)).save(image_path)
    obj = ObjectNodeSnapshot("obj", "chair", (2., 0., 1.5), image_ref=str(image_path),
                             bbox2d_xyxy=(2, 2, 8, 8))
    session = _session()
    frame = {"id": "1", "image_ref": str(image_path), "position": [0., 0., 1.5],
             "rotation": np.eye(3).tolist(), "object_boxes": {"obj": [2, 2, 8, 8]},
             "observed_object_uids": ["obj"]}
    session.snapshot = replace(session.snapshot, objects=(obj,), metadata={"observation_images": [frame]})
    client = FakeClient()
    client.status = "blocked"
    tools = _tools(session, client, tmp_path)
    agent, _ = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.selected = {"candidate_uid": "obj", "target_id": "t1", "role": "primary"}
    state = tools.view_states.setdefault((1, "obj"), ViewControlState())
    fingerprint = guided.evidence_fingerprint("obj")
    guided.submit((1., 0., 1.5), candidate_uid="obj", source="baseline_viewpoint")
    assert not guided.view_attempted("obj", (1., 0., 1.5))
    assert tools.motion.active_intent.metadata["candidate_action_epoch"] == 0
    assert tools.motion.active_intent.metadata["candidate_evidence_fingerprint"] == fingerprint
    tools.motion.controller.advance(tools.motion.active_goal_id)
    guided.motion_finished(tools.motion.poll_active())
    guided.defer_candidate("blocked")
    history = guided.candidate_motion_goals
    assert guided.deferred_evidence == {(1, "obj"): {fingerprint}}
    assert history == {(1, "obj", 0): {(1., 0., 1.5): "blocked"}}

    session.snapshot = replace(session.snapshot, timestamp=999., robot_pose=Pose3D((4., 4., 1.5)),
        objects=(obj, ObjectNodeSnapshot("unrelated", "table", (8., 8., 1.5))))
    guided.observation_updated()
    assert guided.deferred_candidates == {"obj"}
    session.snapshot = replace(session.snapshot, objects=(replace(obj, image_ref=None),))
    guided.observation_updated()
    assert guided.deferred_candidates == {"obj"}
    session.snapshot = replace(session.snapshot, objects=(obj,))
    with Image.open(image_path) as original:
        image = original.copy()
    image.putpixel((0, 0), (255, 255, 255))  # Outside the candidate crop.
    image.save(image_path)
    guided.observation_updated()
    assert guided.deferred_candidates == {"obj"} and guided.evidence_fingerprint("obj") == fingerprint
    image.putpixel((4, 4), (255, 0, 0))
    image.save(image_path)
    guided.observation_updated()
    assert guided.evidence_fingerprint("obj") != fingerprint
    assert guided.deferred_candidates == {"obj"} and guided.action_epoch("obj") == 0
    assert guided.view_attempted("obj", (1., 0., 1.5))
    session.snapshot = replace(session.snapshot, metadata={"observation_images": [
        {**frame, "id": "2", "position": [1., 0., 1.5]}]})
    guided.observation_updated()
    assert guided.deferred_candidates == set() and guided.action_epoch("obj") == 1
    assert not guided.view_attempted("obj", (1., 0., 1.5))
    assert guided.candidate_motion_goals is history
    assert history[(1, "obj", 0)] == {(1., 0., 1.5): "blocked"}
    assert guided.candidate_view_states[(1, "obj")] is state


def _candidate_episode(monkeypatch, tmp_path):
    """Replay a terminal other than the first target, using actual verification registration."""
    spec = TaskSpec("Find a chair or a sofa.",
                    (TargetSpec("first", "chair"), TargetSpec("chosen", "sofa")), revision=4)
    replay = _episode_replay(monkeypatch, tmp_path, spec=spec)
    observation_map = _surface_map()
    replay.observation_map.objects = observation_map.objects
    original_points = observation_map.points.side_effect

    def points(uid):
        surface = original_points(uid).copy()
        if uid == "candidate":
            surface[:, :2] *= .25  # Fresh visible support is within the required XY stop radius.
        return surface

    replay.observation_map.points = Mock(side_effect=points)
    replay.exploration.viewpoints = (ViewpointSnapshot("view", Pose3D((1., 0., 1.5)),
                                                        visible_objects=("candidate",)),)
    monkeypatch.setattr(runtime.IsaacNavigationTools, "query_scene", lambda tools: {
        "candidates": [{"candidate_uid": "candidate", "target_id": "chosen", "role": "primary"}],
        "reason": "offline instance matching"})
    # Lifecycle replay supplies observed views. The separate path-query tests
    # above exercise the real Isaac geometry publisher rather than this seam.
    monkeypatch.setattr(runtime.IsaacNavigationTools, "query_viewpoints", NavigationTools.query_viewpoints)

    def accept(verifier, instruction, plan, candidate, evidence):
        return VerificationResult(True, "accept", semantic_satisfied=True, task_revision=4,
                                  target_uid=candidate.uid, observed_category="couch",
                                  dependencies=evidence["dependencies"])

    verified = Mock(side_effect=accept)
    monkeypatch.setattr(runtime.FinalInstructionVerifier, "verify",
                        lambda verifier, *args: verified(verifier, *args))
    replay.verified = verified
    return replay


@pytest.mark.parametrize("receipts", [("reached",), ("running", "reached"), ("blocked",), ("running",)])
def test_episode_registers_current_terminal_prediction_not_stop_or_cancel(monkeypatch, tmp_path, receipts):
    replay = _candidate_episode(monkeypatch, tmp_path)
    replay.exploration.nodes = {}  # Exhausted exploration after a failed candidate motion.
    command = replay.client.command
    sequence = iter(receipts)

    def receipt(action, **arguments):
        if action == "move":
            replay.client.status = next(sequence)
        return command(action, **arguments)

    monkeypatch.setattr(replay.client, "command", receipt)
    successful = receipts[-1] == "reached"
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=5,
                                 max_moves=len(receipts) if not successful else 5, panorama_views=12)
    assert result["task_revision"] == 4
    assert result["predicted_target_id"] == "chosen" and result["requested_target"] == "sofa"
    assert result["predicted_target"] == ("couch" if successful else None)
    assert result["predicted_candidate_uid"] == "candidate"
    assert result["task_complete"] == successful
    assert result["moves"] == len(receipts) and result["model_calls"] == 1
    assert result["reason"] == ("verified_finish" if successful else "movement_budget_exhausted")
    assert replay.verified.call_count == int(successful)
    session = replay.sessions[0]
    tools = replay.agents[0].tools
    record = session.verification_ledger.get(session.spec.raw_instruction, "candidate")
    assert session.progress.target_progress["first"].accepted_candidate_uids == []
    assert session.progress.target_progress["chosen"].accepted_candidate_uids == (["candidate"] if successful else [])
    if successful:
        assert record.status == "accepted" and record.task_revision == 4
        assert record.candidate_uid == "candidate" and record.verdict == "true"
        assert tools.motion.last_terminal_status.status == NavigationStatusCode.REACHED
        assert "STOP" not in [action for action, _ in replay.client.commands]
        evidence = replay.verified.call_args.args[-1]
        assert evidence["observation_identity"]["task_revision"] == 4 and evidence["target_id"] == "chosen"
        assert evidence["observation_status"] == "current"
        assert evidence["geometry"]["distance_to_object"] <= evidence["geometry"]["required_stop_distance"]
    else:
        assert record is None and tools.last_evidence == {}
        expected = NavigationStatusCode.PREEMPTED if receipts == ("running",) else NavigationStatusCode.BLOCKED
        assert tools.motion.last_terminal_status.status == expected
        assert ("STOP" in [action for action, _ in replay.client.commands]) == (receipts == ("running",))
    assert tools.propose_finish()["accepted"] == successful
    assert tools.motion.active_request_id is None
    assert replay.exploration.marked == []
    assert replay.client.commands[-2][0] == "finish"
    assert json.loads((tmp_path / "policy_result.json").read_text()) == result
    decisions = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "motion_terminal" for row in decisions) == int(receipts != ("running",))
    assert sum(row["event"] == "verified" for row in decisions) == int(successful)


@pytest.mark.parametrize("receipt", ["reached", "blocked", "running"])
def test_frontier_marking_follows_terminal_observation_not_selection(monkeypatch, tmp_path, receipt):
    panorama_views = 12
    replay = _episode_replay(monkeypatch, tmp_path)
    replay.client.status = receipt
    result = runtime.run_episode(replay.public, tmp_path, max_model_calls=5,
                                 max_moves=1, panorama_views=panorama_views)
    assert result["reason"] == "movement_budget_exhausted" and not result["task_complete"]
    events = replay.exploration.events
    full_arrival = receipt == "reached"
    if full_arrival:
        mark_index = next(i for i, event in enumerate(events) if event[0] == "mark")
        assert events[mark_index] == ("mark", 9)
        assert [event[1] for event in events[:mark_index] if event[0] == "add"] == list(range(2, 26))
        assert events[mark_index - 1][0] == "refresh"
    assert replay.exploration.marked == ([9] if full_arrival else [])
    assert replay.exploration.nodes[9].state == int(full_arrival)
    assert replay.parse_calls[0].call_count == 0
    after_motion_views = panorama_views if full_arrival else 0
    calls = replay.observation_map.add_panorama.call_args_list
    assert [len(invocation.args[0]) for invocation in calls] == [4] * (6 if full_arrival else 3)
    assert all(invocation.args[1] is replay.sessions[0] for invocation in calls)
    tools = replay.agents[0].tools
    assert tools.last_packet["frame"]["frame_id"] == 1 + panorama_views + max(1, after_motion_views)
    metadata = tools.session.snapshot.metadata
    assert metadata["captured_frames"] == max(1, after_motion_views)
    assert metadata["perception_frames"] == after_motion_views
    assert metadata["panorama_complete"] is full_arrival
    assert metadata["observation_schedule"] == ("adaptive4" if full_arrival else "geometry_only")
    assert metadata["mapping_cycle_completed"] is True
    assert tools.last_evidence == {} and tools.motion.active_request_id is None
    expected = {"reached": NavigationStatusCode.REACHED, "blocked": NavigationStatusCode.BLOCKED,
                "running": NavigationStatusCode.PREEMPTED}[receipt]
    assert tools.motion.last_terminal_status.status == expected


@pytest.mark.parametrize("views,terminal_state,expected_status,frames", [
    (12, "absent", "searched_no_terminal_found", 12),
    (4, "absent", "search_incomplete", 4),
    (12, "available", "terminal_candidate_available", 4),
    (12, "deferred", "search_incomplete", 4),
])
def test_anchor_arrival_scans_before_recording_search_outcome(
        monkeypatch, tmp_path, views, terminal_state, expected_status, frames):
    spec = TaskSpec("Find a chair near the table.", (
        TargetSpec("t1", "chair"), TargetSpec("reference", "table", role="anchor", terminal=False)))
    replay = _episode_replay(monkeypatch, tmp_path, spec=spec)
    anchor = ObjectNodeSnapshot("anchor", "table", (2., 0., 1.5))
    terminal = ObjectNodeSnapshot("terminal", "chair", (2., 1., 1.5))
    observed = [anchor]
    session = _session(spec=spec)
    session.snapshot = replace(session.snapshot, objects=tuple(observed))
    observation_map = replay.observation_map
    observation_map.objects = lambda: tuple(observed)
    observation_map.points = lambda uid: np.empty((0, 3))
    report = observation_map.add_panorama.side_effect

    def perceive(packets, context):
        assert context is session
        assert session.anchor_search_ledger.get(spec.raw_instruction, "reference", "anchor") is None
        assert replay.client.pose["position"] == [1., 0., 1.5]
        if terminal_state != "absent":
            observed[:] = [anchor, terminal]
            observation_map.mapper.objects = [SimpleNamespace(uid="terminal", tag="chair")]
            observation_map.mapper.current_obj_indices = [0]
        return report(packets, context)

    observation_map.add_panorama.side_effect = perceive
    tools = _tools(session, replay.client, tmp_path, observation_map, replay.exploration)
    if terminal_state == "deferred":
        tools.deferred_candidates.add("terminal")
    tools.verify_target = Mock(side_effect=AssertionError("An anchor cannot authorize task completion"))
    agent, parse = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    guided.feedback_views = Mock(return_value=[
        ViewpointSnapshot("anchor-view", Pose3D((1., 0., 1.5)), visible_objects=("anchor",))])
    execution = runtime.IsaacGuidedExecution(guided, max_moves=1, panorama_views=views)
    execution.initialize = Mock()

    assert run_guided_loop(guided, execution) == "movement_budget_exhausted"
    record = session.anchor_search_ledger.get(spec.raw_instruction, "reference", "anchor")
    assert record.status == expected_status
    assert record.evidence["covered_headings"] == list(range(frames))
    assert record.evidence["panorama_complete"] is (frames == 12)
    assert record.evidence["terminal_candidate_uids"] == (["terminal"] if terminal_state == "available" else [])
    assert session.anchor_search_ledger.is_blocked(spec.raw_instruction, "reference", "anchor") is (frames == 12)
    assert session.snapshot.metadata["perception_frames"] == frames
    assert session.snapshot.metadata["observation_schedule"] == "adaptive4"
    assert len(replay.client.packets) == frames
    assert replay.client.commands[0][0] == "move"
    rows = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert next(row for row in rows if row["event"] == "anchor_searched")["status"] == expected_status
    assert not session.progress.completed and tools.motion.active_request_id is None
    tools.verify_target.assert_not_called()
    parse.assert_not_called()


@pytest.mark.parametrize("mapping_completed", [False, True])
def test_partial_three_view_scan_never_marks_frontier(monkeypatch, tmp_path, mapping_completed):
    replay = _episode_replay(monkeypatch, tmp_path)
    session = _session()
    tools = _tools(session, replay.client, tmp_path, replay.observation_map, replay.exploration)
    agent, parse = _agent(tools, tmp_path)
    guided = runtime.IsaacGuidedNavigation(tools, agent, tmp_path)
    execution = runtime.IsaacGuidedExecution(guided, max_moves=1, panorama_views=3)
    # A previous full scan must not carry completion into this partial refresh.
    guided.scan_complete = True
    replay.exploration.finish_observation = Mock(return_value=mapping_completed)
    if mapping_completed:
        execution.observe(3)
        execution.frontier_observed(SimpleNamespace(intent=SimpleNamespace(metadata={"node_index": 9})))
    else:
        with pytest.raises(RuntimeError, match="Original mapper scan produced no navigable geometry"):
            execution.observe(3)
    assert guided.scan_complete is False
    assert replay.exploration.marked == [] and replay.exploration.nodes[9].state == 0
    assert not any(event[0] == "mark" for event in replay.exploration.events)
    replay.observation_map.add_panorama.assert_called_once_with(replay.client.packets[:3], session)
    assert session.snapshot.metadata["captured_frames"] == 3
    assert session.snapshot.metadata["perception_frames"] == 3
    assert session.snapshot.metadata["panorama_complete"] is False
    assert session.snapshot.metadata["mapping_cycle_completed"] is mapping_completed
    assert tools.last_packet is replay.client.packets[-1]
    parse.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["move", "after_motion_observe"])
def test_episode_execution_error_cancels_without_verification(monkeypatch, tmp_path, failure_stage):
    replay = _candidate_episode(monkeypatch, tmp_path)
    original = ConnectionError("offline execution failure")
    command = replay.client.command
    moved = False

    def fail(action, **arguments):
        nonlocal moved
        if action == "move":
            if failure_stage == "move":
                raise original
            moved = True
            replay.client.status = "running"
        if action == "observe" and moved:
            raise original
        return command(action, **arguments)

    monkeypatch.setattr(replay.client, "command", fail)
    with pytest.raises(ConnectionError) as raised:
        runtime.run_episode(replay.public, tmp_path, max_model_calls=5, max_moves=5, panorama_views=12)
    assert raised.value is original
    tools = replay.agents[0].tools
    assert tools.motion.active_request_id is None
    assert tools.motion.last_terminal_status.status == NavigationStatusCode.PREEMPTED
    replay.verified.assert_not_called()
    assert not tools.session.progress.completed
    assert "finish" not in [action for action, _ in replay.client.commands]
    assert replay.client.commands[-1][0] == "STOP"
    result = json.loads((tmp_path / "policy_result.json").read_text())
    assert result["status"] == "error" and result["reason"] == "ConnectionError"
    assert result["error"] == str(original) and not result["task_complete"]
    assert result["moves"] == int(failure_stage == "after_motion_observe")
