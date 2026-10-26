"""Fixed-input regressions for evidence ownership, state and decision boundaries."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from unittest.mock import Mock

import numpy as np
import pytest

from instruction_adapter.contracts import ConceptQuery
from instruction_adapter.concept_matcher import RuntimeConceptMatcher
from instruction_adapter.verifier import candidate_from_object, instruction_hash
from verification.contracts import VerificationResult
from verification.store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]


def methods(path, names, namespace):
    """Execute production methods with simulator imports replaced by test doubles."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
             for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


def test_traversability_preserves_both_input_poses():
    paths = []
    ns = methods("mapper_with_process_obs.py", {"check_traversability"}, {
        "np": np, "habitat_sim": NS(ShortestPath=NS)})
    mapper = NS(floor_height=-.8, initial_position=np.array([1., 2., 3.]),
                env=NS(sim=NS(pathfinder=NS(find_path=lambda path: paths.append(path) or True))))
    start, end = np.array([4., 5., .15]), np.array([7., 8., .4])
    before = start.copy(), end.copy()
    assert ns["check_traversability"](mapper, start, end)
    np.testing.assert_array_equal(start, before[0])
    np.testing.assert_array_equal(end, before[1])
    np.testing.assert_allclose(paths[0].requested_start, [5., 2.2, 7.])


def test_projection_receipt_preserves_metric_depth_and_actual_filter(tmp_path):
    """Replay inputs retain sensor clipping values and the production occlusion mask."""
    projected = np.array([[10, 20, .19], [30, 40, 1.0]], dtype=float)
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    ns = methods("objnav_agent_with_process_obs.py", {"_project_object_bbox_on_current_view"}, {
        "np": np, "os": os, "json": json,
        "project_to_camera": lambda *args: projected.T,
        "torch": NS(tensor=lambda value: NS(unsqueeze=lambda axis: value)),
    })
    agent = NS(save_dir=str(tmp_path), episode_samples=1, episode_steps=4,
               obs={"depth": np.full((480, 640, 1), .5, dtype=np.float32),
                    "rgb": np.zeros((480, 640, 3), dtype=np.uint8)},
               object_final=NS(uid="candidate", pcd=NS(point=NS(positions=NS(
                   cpu=lambda: NS(numpy=lambda: points))))),
               mapper=NS(camera_intrinsic=np.eye(3), current_position=np.zeros(3),
                         current_rotation=np.eye(3)),
               _observation_identity=lambda uid: {"candidate_uid": uid, "observation_id": "0:4", "task_revision": 1})
    bbox, geometry = ns["_project_object_bbox_on_current_view"](agent)
    np.testing.assert_array_equal(bbox, [10, 20, 10, 20])
    with np.load(geometry["projection_evidence_path"], allow_pickle=False) as receipt:
        np.testing.assert_array_equal(receipt["depth_m"], agent.obs["depth"])
        np.testing.assert_array_equal(receipt["points_xyz"], points)
        np.testing.assert_array_equal(receipt["projected_uv_depth"], projected)
        np.testing.assert_array_equal(receipt["passes_occlusion"], [True, False])
        assert json.loads(str(receipt["observation_identity"]))["candidate_uid"] == "candidate"
    # 同一步可能检查多个实例，后一个实例的投影不得覆盖前一个的证据。
    agent.object_final.uid = "other"
    _, other_geometry = ns["_project_object_bbox_on_current_view"](agent)
    assert other_geometry["projection_evidence_path"] != geometry["projection_evidence_path"]
    with np.load(geometry["projection_evidence_path"], allow_pickle=False) as receipt:
        assert json.loads(str(receipt["observation_identity"]))["candidate_uid"] == "candidate"


def test_new_candidate_without_projection_never_uses_previous_image(tmp_path):
    ledger = EvidenceStore(1)
    ns = methods("objnav_agent_with_process_obs.py", {
        "_observation_identity", "final_instruction_check"}, {
        "np": np, "json": json, "os": os, "logger": Mock(),
        "VerificationResult": VerificationResult, "candidate_from_object": candidate_from_object,
        "instruction_hash": instruction_hash, "apply_verification": Mock(return_value=False)})
    agent = NS(task=NS(spec=NS(revision=1), progress=object(),
                       plan=NS(terminal_targets=[NS(id="plant")], diagnostics={}),
                       verification_ledger=ledger, constraint_evaluator=Mock()),
               episode_steps=408, episode_samples=1, save_dir=str(tmp_path), run_id="test",
               selected_candidate=NS(role="primary", target_id="plant"),
               object_final=NS(uid="new-plant", tag="plant", position=[1, 2, 3]),
               mapper=NS(target="plant", current_position=np.zeros(3), current_rotation=np.eye(3)),
               last_check_again_image_path="old-wall-art.jpg", final_instruction_verifier=Mock(),
               view_control_state=NS(active=False))
    agent._observation_identity = MethodType(ns["_observation_identity"], agent)
    agent._raw_instruction_for_verifier = lambda: "find plant"
    agent._project_object_bbox_on_current_view = lambda: (None, {})
    agent._augment_final_stop_geometry = lambda x: x
    agent._record_final_verifier_attempt = lambda *a: 1
    agent._max_final_verifier_attempts_per_candidate = lambda: 5
    agent._final_verifier_attempt_budget_exhausted = lambda *a: False
    agent._pin_visual_reference_from_result = Mock()
    agent._view_control_override_if_needed = lambda c, e, r: r
    agent._annotate_view_budget_if_exhausted = lambda c, e, r, **kw: r
    agent._enforce_planner_hard_stop_contract = lambda e, r: r
    agent._enter_better_view_control = Mock(return_value=False)
    old = {"observation_identity": {"candidate_uid": "old-art"},
           "current_rgb_with_bbox_path": "old-wall-art.jpg"}
    assert ns["final_instruction_check"](agent, evidence_override=old) is False
    agent.final_instruction_verifier.verify.assert_not_called()
    record = ledger.get("find plant", "new-plant")
    assert record.verdict == "unknown" and record.decision == "need_better_view"
    assert record.evidence_paths == []
    assert not ledger.is_hard_rejected("find plant", "new-plant")


def test_concept_cache_tracks_label_and_concept_semantics(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    matcher = RuntimeConceptMatcher()
    concept = ConceptQuery(id="goal", name="chair")
    obj = NS(uid="same", tag="chair", position=[0, 0, 0])
    args = dict(raw_instruction="find furniture", concept=concept, objects=[obj])
    assert matcher.match_many(**args)[0].matches_concept is True
    obj.tag = "sofa"
    assert matcher.match_many(**args)[0].matches_concept is None
    concept.name = "sofa"
    assert matcher.match_many(**args)[0].matches_concept is True


@pytest.mark.parametrize("method", ["match_object", "match_many"])
def test_concept_request_error_propagates_without_false_cache(monkeypatch, method):
    import instruction_adapter.concept_matcher as module
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(module, "HAS_PYDANTIC", True)
    failure = RuntimeError("provider down")
    monkeypatch.setattr(module, "get_client_and_model", Mock(side_effect=failure))
    matcher = RuntimeConceptMatcher()
    obj = NS(uid="x", tag="unknown", position=[0, 0, 0])
    args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
    with pytest.raises(RuntimeError) as raised:
        getattr(matcher, method)(raw_instruction="chair", concept=ConceptQuery("c", "chair"), **args)
    assert raised.value is failure
    assert not matcher.records


def test_search_review_preserves_baseline_option_set_and_interval():
    from backends.habitat.guided_runtime import GuidedNavigation
    guided = GuidedNavigation.__new__(GuidedNavigation)
    nodes = [NS(idx=i) for i in range(3)]
    guided.agent = NS(episode_steps=99, mapper=NS(nodes=nodes,
        prior_map_last_chosen_frontier={"candidates": [
            {"node_idx": 0, "raw_distance_m": 2.},
            {"node_idx": 1, "raw_distance_m": 5.},
            {"node_idx": 2, "raw_distance_m": float("inf")}] }))
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.runtime = NS(model_calls=0, max_model_calls=5, choose_frontier=Mock(return_value="node:1"))
    guided.record = Mock()
    assert guided.review_search_branch(nodes[0], "current_room") == (nodes[0], "current_room")
    guided.runtime.choose_frontier.assert_not_called()
    guided.agent.episode_steps = 100
    assert guided.review_search_branch(nodes[0], "current_room") == (nodes[1], "agent_search_review")
    options, recommended, event = guided.runtime.choose_frontier.call_args.args
    assert [o["frontier_uid"] for o in options] == ["node:0", "node:1"]
    assert recommended == "node:0" and event["reason"] == "prolonged_target_search"
    guided.agent.episode_steps = 101
    guided.review_search_branch(nodes[0], "current_room")
    assert guided.runtime.choose_frontier.call_count == 1


def test_frontier_response_schema_limits_model_to_domain_options(tmp_path):
    from harness.runtime import AgentRuntime
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.max_model_calls, runtime.model_calls = 3, 0
    runtime.tools = NS(observe=lambda: {"task": "find bed"})
    runtime.model, runtime.trace_path = "test", tmp_path / "trace.jsonl"
    def parse(**kw):
        schema = kw["response_format"]
        with pytest.raises(ValueError):
            schema(frontier_uid="invented", reason="invalid")
        return NS(choices=[NS(message=NS(parsed=schema(frontier_uid="node:2", reason="branch")))])
    runtime.client = NS(beta=NS(chat=NS(completions=NS(parse=parse))))
    assert runtime.choose_frontier([{"frontier_uid": "node:1"}, {"frontier_uid": "node:2"}],
                                   "node:1", "stalled") == "node:2"
    assert runtime.model_calls == 1


def test_object_viewpoints_are_shared_and_replaced_on_feedback(monkeypatch):
    import navigation.action_controller as actions
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint", lambda *args: 2.)
    from backends.habitat.agent_runtime import HabitatNavigationTools
    from scene.contracts import SemanticMapSnapshot, Pose3D, ObjectNodeSnapshot
    tools = HabitatNavigationTools.__new__(HabitatNavigationTools)
    tools.agent = NS(episode_steps=10, planner=NS(_goal_radius=.25), confirmed_target_waypoint=None, waypoint=(1, 2, 0), check_again_postion=(3, 4, 0))
    tools.session = NS(snapshot=SemanticMapSnapshot(10, Pose3D((0, 0, 0)),
        objects=(ObjectNodeSnapshot(uid="obj", label="chair"),)))
    views = tools.publish_object_viewpoints("obj")
    assert {v.metadata["purpose"] for v in views} == {"approach", "recheck"}
    assert next(v for v in views if v.metadata["recommended"]).pose.position == (3, 4, 0)
    tools.agent.check_again_postion = (5, 6, 0)
    tools.publish_object_viewpoints("obj")
    offered = tools.query_viewpoints("obj")
    assert len(offered) == 2
    assert next(v for v in offered if v["metadata"]["recommended"])["pose"]["position"] == (5, 6, 0)


def test_visual_history_is_scoped_to_task_revision():
    from instruction_adapter.view_control import ViewControlState
    state = ViewControlState()
    first = {"observation_identity": {"task_revision": 1, "candidate_uid": "obj"},
             "current_rgb_with_bbox_path": "old.jpg"}
    state.start("obj", {}, first)
    state.pin_visual_evidence(first)
    state.start("obj", {}, {"observation_identity": {"task_revision": 2}})
    assert state.task_revision == 2 and state.pinned_visual_evidence == {}


def test_uncertain_exhausted_view_defers_candidate_until_new_observation(monkeypatch):
    import sys
    from backends.habitat.guided_runtime import GuidedNavigation
    from instruction_adapter.view_control import ViewControlState
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller",
                        NS(rotate_toward_object_for_recheck=lambda agent: True))
    state = ViewControlState()
    state.start("obj", {}, {})
    state.exhausted = True
    agent = NS(episode_steps=30, object_final=NS(uid="obj", rgb=np.zeros((4,4,3)), bbox=[0,0,2,2]),
               task=NS(spec=NS(revision=1)), need_check_again=False,
               last_check_again_evidence={}, final_stop_decision="uncertain",
               check_again=Mock(), final_instruction_check=Mock(return_value=False),
               _continue_better_view_control=lambda: False, view_control_state=state)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record = agent, Mock()
    guided.deferred_candidates = set()
    guided.deferred_evidence, guided.candidate_view_states = {}, {}
    guided.tools = NS(refresh_scene=Mock(), objects={"obj": agent.object_final})
    assert guided.verify() == "recovery"
    assert guided.deferred_candidates == {"obj"}
    guided.observation_updated()
    assert guided.deferred_candidates == {"obj"} and state.active
    agent.object_final.rgb[0,0] = 1
    guided.observation_updated()
    assert guided.deferred_candidates == set() and state.active and state.exhausted
    assert guided.candidate_view_states[(1, "obj")] is state
    guided.defer_candidate("new view also exhausted")
    agent.object_final.rgb[0,0] = 0
    guided.observation_updated()
    assert guided.deferred_candidates == {"obj"}  # A → B → A consumes A only once


def test_viewpoint_query_removes_reached_and_unreachable_motion_goals(monkeypatch):
    import navigation.action_controller as actions
    from backends.habitat.agent_runtime import HabitatNavigationTools
    from scene.contracts import SemanticMapSnapshot, Pose3D, ViewpointSnapshot
    from dataclasses import asdict
    views = tuple(ViewpointSnapshot(str(i), Pose3D((i, 0, 0)), visible_objects=("obj",)) for i in range(3))
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint",
                        lambda agent, pose: [0., 2., float("inf")][pose[0]])
    tools = HabitatNavigationTools.__new__(HabitatNavigationTools)
    tools.agent = NS(planner=NS(_goal_radius=.25))
    tools.session = NS(snapshot=SemanticMapSnapshot(30, Pose3D((0, 0, 0)), viewpoints=views))
    assert [v["uid"] for v in tools._moving_viewpoints("obj", [asdict(v) for v in views])] == ["1"]
    assert [v.uid for v in tools.session.snapshot.viewpoints] == ["1"]


def test_evidence_identity_ignores_background_and_tracks_instance_crop():
    from backends.habitat.guided_runtime import GuidedNavigation
    obj = NS(rgb=np.zeros((4,4,3), dtype=np.uint8), bbox=[0,0,2,2])
    original = GuidedNavigation.candidate_evidence(obj)
    obj.rgb[3,3] = 255
    assert GuidedNavigation.candidate_evidence(obj) == original
    obj.rgb[0,0] = 255
    assert GuidedNavigation.candidate_evidence(obj) != original
    obj.rgb = None
    assert GuidedNavigation.candidate_evidence(obj) is None


@pytest.mark.parametrize("target", [False, True])
def test_associated_instance_publishes_corresponding_new_image_and_box(target):
    import torch
    cloud = NS(point=NS(positions=torch.zeros((4,3))))
    def instance(value):
        return NS(uid=str(value), tag="object", alive=True, pcd=cloud, position=np.zeros(3),
                  nodes=[], confidence=.8, num_list={"object": 1}, conf_list={"object": .8},
                  rgb=np.full((4,4,3), value), bbox=[value,0,3,3])
    old, new = instance(0), instance(1)
    namespace = methods("mapper_with_process_obs.py", {"associate_object_entities"}, {
        "np": np, "torch": torch, "pointcloud_distance": lambda *args: torch.zeros(4),
        "gpu_merge_pointcloud": lambda *args: cloud})
    mapper = NS(_is_target_tag=lambda tag: target, target="object")
    merged, _ = namespace["associate_object_entities"](mapper, [old], [new])
    assert merged[0].uid == "0"
    assert merged[0].rgb is new.rgb and merged[0].bbox is new.bbox


def test_physical_infeasibility_reports_limitation_and_preserves_stop_contract():
    namespace = methods("objnav_agent_with_process_obs.py", {"_planner_hard_stop_allows_accept"}, {})
    check = namespace["_planner_hard_stop_allows_accept"]
    for key in ("planner_infeasible", "infeasible_by_geometry"):
        assert not check(NS(), {"hard_stop_constraints": {"satisfied": False, key: True}})
