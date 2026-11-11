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


@pytest.fixture
def grounded_sam():
    """Load actual detector methods without importing or constructing model backends."""
    import torch

    path = ROOT / "cv_utils/sam.py"
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
    nms = Mock(side_effect=lambda boxes, scores, threshold: torch.arange(len(boxes)))
    namespace = {"np": np, "torch": torch, "nms": nms, "logger": Mock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["MMDINO_Grounded_SAM"]
    detector = cls.__new__(cls)
    detector.classes = ["chair", "table", "door", "window", "plant"]
    detector.BOX_THRESHOLD = .2
    detector.sam_model, detector.grounding_model = object(), object()
    detector.sam_predictor = NS(set_image=Mock())
    detector._run_grounding = Mock(return_value={"bboxes": [], "scores": [], "labels": []})
    detector._predict_masks = Mock(side_effect=lambda image, boxes:
        torch.ones((len(boxes), *image.shape[:2]), dtype=torch.bool))
    detector.initialize(target=None)
    return detector, nms


@pytest.mark.parametrize("cleared_terms", [None, []])
def test_live_detector_terms_replace_prompts_and_keep_historical_labels(grounded_sam, cleared_terms):
    detector, _ = grounded_sam
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    base = detector.classes.copy()
    models = detector.sam_model, detector.grounding_model, detector.sam_predictor

    def predict(image, prompt):
        terms = prompt.removesuffix(" .").split(" . ")
        return {"bboxes": [[i, i, i + 2, i + 2] for i in range(len(terms))],
                "scores": [.9] * len(terms), "labels": list(range(len(terms)))}

    detector._run_grounding.side_effect = predict

    def observe(terms, expected):
        detector._run_grounding.reset_mock()
        output = detector.segment(image, target_list=terms)
        prompts = [call.args[1] for call in detector._run_grounding.call_args_list]
        assert prompts == detector.TEXT_PROMPTs
        assert " ".join(prompts) == " ".join(f"{term} ." for term in expected)
        assert output[2].tolist() == expected
        assert all(len(group) == 0 for group in output[4:])
        assert detector.classes == base
        assert (detector.sam_model, detector.grounding_model, detector.sam_predictor) == models
        return output[2]

    terms = ["espresso machine", "chair", "espresso machine", "sofa"]
    historical_labels = observe(terms, base + ["espresso machine", "sofa"])
    cached_prompts, cached_labels = detector.TEXT_PROMPTs, detector.class_labels
    observe(["espresso machine", "sofa"], base + ["espresso machine", "sofa"])
    assert detector.TEXT_PROMPTs is cached_prompts and detector.class_labels is cached_labels
    terms[:] = ["bookshelf"]
    observe(terms, base + ["bookshelf"])
    assert historical_labels.tolist() == base + ["espresso machine", "sofa"]
    assert np.concatenate(cached_labels).tolist() == historical_labels.tolist()
    observe(cleared_terms, base)
    # Omitting terms on a subsequent task-free observation also clears the cache.
    observe(["kettle"], base + ["kettle"])
    detector.segment(image)
    assert np.concatenate(detector.class_labels).tolist() == base


@pytest.mark.parametrize("presence_score,expected_count", [(.49, 0), (.5, 1)])
def test_live_detector_terms_preserve_target_presence_threshold_and_nms(
        grounded_sam, presence_score, expected_count):
    import torch

    detector, nms = grounded_sam
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    target_boxes = [[1., 1., 10., 10.], [2., 2., 11., 11.], [3., 3., 12., 12.]]
    detector._run_grounding.side_effect = lambda image, prompt: (
        {"bboxes": target_boxes, "scores": [presence_score, .3, .24], "labels": [0, 0, 0]}
        if prompt == "chair" else {"bboxes": [], "scores": [], "labels": []})
    nms.side_effect = lambda boxes, scores, threshold: torch.tensor([1])
    result = detector.segment(image, target="chair", target_list=["espresso machine"])
    assert detector._run_grounding.call_args_list[0].args[1] == "chair"
    assert [call.args[1] for call in detector._run_grounding.call_args_list[1:]] == detector.TEXT_PROMPTs
    assert result[0].shape == (0, 4) and result[3].shape == (0, 32, 32)
    assert result[4].shape == (expected_count, 4)
    assert result[6].shape == (expected_count, 32, 32)
    if expected_count:
        boxes, scores, threshold = nms.call_args.args
        assert boxes.tolist() == target_boxes[:2]
        assert scores.tolist() == pytest.approx([presence_score, .3])
        assert threshold == detector.NMS_IOU_THRESHOLD
        assert result[4].tolist() == [target_boxes[1]]
        assert result[5].tolist() == pytest.approx([.3])
    else:
        nms.assert_not_called()
        detector._predict_masks.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
def test_habitat_observation_forwards_live_terms_through_perceiver_to_grounding(grounded_sam, partial):
    import torch

    detector, _ = grounded_sam
    path = ROOT / "cv_utils/image_perceiver.py"
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["MMDINOSAM_Perceiver"]
    perceiver = cls.__new__(cls)
    perceiver.sam = detector
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    class ObservationProcessed(Exception):
        pass

    ns = {
        "detection_step_dir": Mock(return_value="unused"),
        "build_triplet": Mock(return_value=NS(image=image, depth_vis=None)),
        "save_combined_view": Mock(side_effect=ObservationProcessed)}
    pipeline_path = ROOT / "navigation/observation_pipeline.py"
    pipeline = next(node for node in ast.parse(pipeline_path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "process_panoramic_detections")
    exec(compile(ast.Module(body=[pipeline], type_ignores=[]), str(pipeline_path), "exec"), ns)
    ns = methods("objnav_agent_with_process_obs.py", {"rotate_segmentation"}, ns)
    mapper = NS(object_perceiver=perceiver, target="chair", camera_intrinsic=np.eye(3),
                perception_target_list=["stale anchor"], target_list=["stale target"],
                objects=[NS(tag="historical label")])
    agent = NS(task=NS(detector_terms=[]), mapper=mapper, save_dir="unused",
               episode_samples=1, episode_steps=1)
    for terms in (["espresso machine"], ["bookshelf"], [], None):
        agent.task = NS(detector_terms=terms) if terms is not None else None
        detector._run_grounding.reset_mock()
        with pytest.raises(ObservationProcessed):
            ns["rotate_segmentation"](agent, [image], [None], [None], [None], partial=partial)
        prompts = [call.args[1] for call in detector._run_grounding.call_args_list]
        assert prompts[0] == "chair"
        assert " ".join(prompts[1:]) == " ".join(
            f"{term} ." for term in detector.classes + (terms or []))
        assert all(call.args[0] is image for call in detector._run_grounding.call_args_list)
        assert mapper.objects[0].tag == "historical label"
        assert mapper.perception_target_list == ["stale anchor"]
        assert mapper.target_list == ["stale target"]


def test_real_detector_passes_bgr_to_ultralytics_without_double_swap():
    """A red/blue asymmetric camera image must reach the BGR backend unchanged."""
    import time
    namespace = methods(
        "real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/detection_node.py",
        {"inference"}, {"np": np, "time": time})
    image = np.array([[[11, 37, 241], [220, 42, 9]]], dtype=np.uint8)
    before = image.copy()
    backend = Mock()
    backend.track.return_value = [NS(boxes=NS(id=None))]
    node = NS(grounding_model=backend, detector_imgsz=320, device="cpu",
              grounding_score_thresh=.35, CONFIG_DIR=Path("/config"), log_info=Mock())
    result = namespace["inference"](node, image)
    delivered = backend.track.call_args.args[0]
    np.testing.assert_array_equal(delivered, before)
    np.testing.assert_array_equal(image, before)
    assert delivered.flags.c_contiguous
    assert backend.track.call_args.kwargs["conf"] == .35
    assert result["labels"].size == 0


def test_real_mapper_converts_bgr_for_sam2_without_mutating_observation():
    """SAM2 receives RGB while the observation used by mapping remains BGR."""
    import cv2
    import time
    from contextlib import nullcontext

    class CapturedInput(Exception):
        pass

    predictor = Mock()
    predictor.set_image.side_effect = CapturedInput
    namespace = methods(
        "real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/semantic_mapping_node.py",
        {"mapping_processing"}, {
            "profile": lambda method: method, "time": time, "cv2": cv2,
            "torch": NS(inference_mode=nullcontext, bfloat16=None,
                        autocast=lambda *args, **kwargs: nullcontext())})
    node = NS(mapping_processing_lock=nullcontext(), mask_predictor=predictor)
    image = np.array([[[11, 37, 241], [220, 42, 9]]], dtype=np.uint8)
    before = image.copy()
    detections = {"labels": [], "bboxes": [], "confidences": []}
    with pytest.raises(CapturedInput):
        namespace["mapping_processing"](node, image, None, detections, 1., None)
    rgb = predictor.set_image.call_args.args[0]
    np.testing.assert_array_equal(rgb, before[:, :, ::-1])
    np.testing.assert_array_equal(image, before)
    assert rgb.flags.c_contiguous


@pytest.mark.parametrize("backend", ["yoloe", "yolo_world"])
def test_open_vocabulary_initialization_matches_decoder_prompt_order(tmp_path, backend):
    """Both real detector backends must initialize the decoder's exact vocabulary."""
    import time
    import yaml
    from collections import deque

    config = tmp_path / "objects.yaml"
    config.write_text("prompts:\n  chair:\n    prompts: [chair, office chair]\n  desk:\n    prompts: [desk]\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    overrides = {"object_file": str(config), "detector_model_path": str(checkpoint),
                 "detector_model_type": backend, "annotate_image": False}

    class FakeNode:
        def __init__(self, name):
            self.parameters = {}

        def declare_parameter(self, name, default):
            self.parameters[name] = overrides.get(name, default)

        def get_parameter(self, name):
            value = self.parameters[name]
            return NS(get_parameter_value=lambda: NS(
                string_value=value, double_value=value, integer_value=value, bool_value=value))

        def __getattr__(self, name):
            return Mock()

    path = ROOT / "real_robot/ros2_ws/src/semantic_mapping/semantic_mapping/detection_node.py"
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"DetectNode", "resolve_package_path"}]
    factories = {name: Mock() for name in ("YOLO", "YOLOWorld", "YOLOE")}
    namespace = dict(factories, Node=FakeNode, Path=Path, np=np, yaml=yaml,
                     time=time, deque=deque, __file__=str(path), CvBridge=Mock(),
                     Image=object, DetectionResult=object, qos_profile_sensor_data=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    detector = namespace["DetectNode"]()
    detector.grounding_model.set_classes.assert_called_once_with(["chair", "office chair", "desk"])
    assert detector.text_prompt_list.tolist() == ["chair", "office chair", "desk"]


def test_traversability_preserves_both_input_poses(monkeypatch):
    import sys
    from backends.habitat.geometry import HabitatMapperQueries
    monkeypatch.setitem(sys.modules, "habitat_sim", NS(ShortestPath=NS))
    paths = []
    ns = methods("mapper_with_process_obs.py", {"check_traversability"}, {"np": np})
    mapper = NS(floor_height=-.8, initial_position=np.array([1., 2., 3.]),
                env=NS(sim=NS(pathfinder=NS(find_path=lambda path: paths.append(path) or True))))
    mapper.geometry_queries = HabitatMapperQueries(mapper)
    start, end = np.array([4., 5., .15]), np.array([7., 8., .4])
    before = start.copy(), end.copy()
    assert ns["check_traversability"](mapper, start, end)
    np.testing.assert_array_equal(start, before[0])
    np.testing.assert_array_equal(end, before[1])
    np.testing.assert_allclose(paths[0].requested_start, [5., 2.2, 7.])


def test_projection_receipt_preserves_metric_depth_and_actual_filter(tmp_path):
    """Save raw depth, exact surface selection and a candidate-bound frame cache."""
    projected = np.array([[10, 20, .5], [11, 20, .5], [10, 21, .5], [30, 40, 1.0]])
    points = np.array([[1, 2, -3], [1, 2, -4], [1, 3, -4], [0.1, .1, -6], [4, 5, 6.]])
    masks = np.zeros((1, 480, 640), dtype=bool)
    masks[0, 19:23, 9:13] = True
    detector = Mock(return_value=masks)
    ns = methods("objnav_agent_with_process_obs.py", {"_project_object_bbox_on_current_view"}, {
        "np": np, "os": os, "json": json,
        "project_to_camera": lambda *args: projected.T,
        "torch": NS(tensor=lambda value, **kwargs: np.array(value), float32=None),
    })
    agent = NS(save_dir=str(tmp_path), episode_samples=1, episode_steps=4, current_surface_evidence=None,
               env=NS(sim=NS(get_agent_state=lambda: NS(position=np.zeros(3)))),
               obs={"depth": np.full((480, 640, 1), .5, dtype=np.float32),
                    "rgb": np.zeros((480, 640, 3), dtype=np.uint8)},
               object_final=NS(uid="candidate", pcd=NS(point=NS(positions=NS(
                   cpu=lambda: NS(numpy=lambda: points))))),
               mapper=NS(camera_intrinsic=np.eye(3), current_position=np.zeros(3),
                         current_rotation=np.eye(3), initial_position=np.zeros(3),
                         translation_func=lambda x:x, target='chair',
                         object_perceiver=NS(sam=NS(segment_current_target=detector))),
               _observation_identity=lambda uid: {"candidate_uid": uid, "observation_id": "0:4", "task_revision": 1})
    bbox, geometry = ns["_project_object_bbox_on_current_view"](agent)
    np.testing.assert_array_equal(bbox, [[9, 19, 12, 22]])
    assert geometry["distance_source"] == "current_instance_raw_depth"
    with np.load(geometry["projection_evidence_path"], allow_pickle=False) as receipt:
        np.testing.assert_array_equal(receipt["depth_m"], agent.obs["depth"].squeeze())
        np.testing.assert_array_equal(receipt["points_xyz"], points)
        np.testing.assert_array_equal(receipt["projected_uv_depth"], projected)
        np.testing.assert_array_equal(receipt["passes_occlusion"], [True, True, True, False])
        np.testing.assert_array_equal(receipt["visible_point_index"], [0, 1, 2])
        assert str(receipt['depth_source']) == 'sensor_raw'
    ns["_project_object_bbox_on_current_view"](agent)
    assert detector.call_count == 1
    agent.object_final.uid = "other"
    _, other_geometry = ns["_project_object_bbox_on_current_view"](agent)
    assert other_geometry["projection_evidence_path"] != geometry["projection_evidence_path"]
    assert detector.call_count == 2


def test_final_evidence_views_instances_sharing_candidate_surface(tmp_path):
    """同一物理表面拆成多个实例时，取证包附带其它实例自己的代表图与框。"""
    written = {}
    coverage = {"piano": .9, "rug": .1}
    ns = methods("objnav_agent_with_process_obs.py", {"_overlapping_instance_observations"}, {
        "np": np, "os": os, "torch": NS(tensor=lambda value: value),
        "visualize_mask": lambda image, boxes: (image, boxes),
        "cv2": NS(imwrite=lambda path, image: written.setdefault(path, image)),
        # 候选自身或无图实例进入评分会因缺键失败。
        "score_object_candidates": lambda entity, refs: (
            None, [coverage[ref.tag] for ref in refs], None)})
    piano = NS(uid="object:piano", tag="piano", rgb=np.ones((4, 4, 3)), bbox=np.array([0., 1., 3., 4.]))
    candidate = NS(uid="object:sofa", tag="sofa", rgb=np.zeros((4, 4, 3)), bbox=[0, 0, 1, 1])
    agent = NS(episode_steps=34, object_final=candidate, mapper=NS(objects=[
        candidate, piano, NS(uid="object:rug", tag="rug", rgb=np.zeros((4, 4, 3)), bbox=[0, 0, 2, 2]),
        NS(uid="object:chair", tag="chair", rgb=None, bbox=None)]))
    observations = ns["_overlapping_instance_observations"](agent, str(tmp_path))
    path = str(tmp_path / "overlap_34_piano.jpg")
    assert observations == [{"image_path": path, "bbox_xyxy": [0, 1, 3, 4],
                             "candidate_surface_fraction": .9}]
    assert written[path][0] is piano.rgb and written[path][1] == [[0, 1, 3, 4]]


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


@pytest.mark.parametrize("step,calls,level_camera,expected", [
    (99, 0, True, "select"), (100, 0, True, "recovery"),
    (100, 5, True, "select"), (100, 0, False, "episode_limit"),
])
def test_guided_anchor_search_preserves_ledger_without_target_verification(
        monkeypatch, tmp_path, step, calls, level_camera, expected):
    import sys
    from backends.habitat.guided_runtime import GuidedNavigation
    from instruction_adapter.concept_matcher import AnchorSearchLedger
    from harness.runtime import AgentRuntime

    tilt = Mock(return_value=level_camera)
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller", NS(
        rotate_toward_object_for_recheck=lambda agent: True, tilt_camera=tilt))
    ns = methods("objnav_agent_with_process_obs.py", {"final_instruction_check"}, {"logger": Mock()})
    anchor = NS(uid="anchor", position=np.ones(3))
    task = NS(spec=NS(revision=1), plan=NS(diagnostics={}),
              progress=NS(completed=False), anchor_search_ledger=AnchorSearchLedger(),
              verification_ledger=EvidenceStore(1), constraint_evaluator=Mock())
    agent = NS(task=task, episode_steps=step, episode_samples=1, object_final=anchor,
               selected_candidate=NS(role="anchor", target_id="reference", obj=anchor),
               mapper=NS(current_position=np.zeros(3), current_rotation=np.eye(3)[:, ::-1]),
               _observation_identity=lambda uid: {"observation_id": f"0:{step}"},
               _raw_instruction_for_verifier=lambda: "find the book on the table",
               check_again=Mock(), last_check_again_evidence=None, final_instruction_verifier=Mock())
    agent.final_instruction_check = MethodType(ns["final_instruction_check"], agent)
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.history, runtime.trace_path = [], tmp_path / "tools.jsonl"
    runtime.model_calls, runtime.max_model_calls = calls, 5
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.runtime, guided.directory = agent, runtime, tmp_path
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.submit, guided.defer_candidate = Mock(), Mock()
    guided.tools = NS(publish_object_viewpoints=Mock())
    assert guided.verify((0., 0., 0.)) == expected
    record = task.anchor_search_ledger.get("find the book on the table", "reference", "anchor")
    assert record.status == "searched_no_terminal_found" and record.step == step
    assert task.verification_ledger.history == [] and not task.progress.completed
    agent.final_instruction_verifier.verify.assert_not_called()
    guided.submit.assert_not_called()
    guided.defer_candidate.assert_not_called()
    guided.tools.publish_object_viewpoints.assert_not_called()
    tilt.assert_called_once_with(agent, 0.0)
    assert runtime.model_calls == calls and guided.search_review_step == 0
    assert len(runtime.history) == 1 and runtime.history[0]["event"] == "anchor_searched"
    assert runtime.history[0]["anchor_uid"] == "anchor"
    assert runtime.history[0]["concept_id"] == "reference"
    assert runtime.history[0]["task_revision"] == 1
    if expected == "recovery":
        assert guided.recovery_context == {
            "reason": "prolonged_target_search", "anchor_uid": "anchor", "concept_id": "reference",
            "anchor_status": "searched_no_terminal_found", "steps_since_review": 100}


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


@pytest.mark.parametrize("reachable", [1, 2])
def test_search_review_preserves_baseline_option_set_and_interval(reachable):
    from backends.habitat.guided_runtime import GuidedNavigation
    guided = GuidedNavigation.__new__(GuidedNavigation)
    nodes = [NS(idx=i) for i in range(3)]
    guided.agent = NS(episode_steps=99, task=NS(spec=NS(revision=1)), mapper=NS(nodes=nodes,
        prior_map_last_chosen_frontier={"candidates": [
            {"node_idx": 0, "raw_distance_m": 2.},
            {"node_idx": 1, "raw_distance_m": 5. if reachable == 2 else float("inf")},
            {"node_idx": 2, "raw_distance_m": float("inf")}] }))
    guided.search_review_step, guided.search_review_interval = 0, 100
    selected = reachable - 1
    guided.runtime = NS(model_calls=0, max_model_calls=5,
                        choose_frontier=Mock(return_value=f"node:{selected}"))
    guided.record = Mock()
    assert guided.review_search_branch(nodes[0], "current_room", 0) == (nodes[0], "current_room")
    guided.runtime.choose_frontier.assert_not_called()
    guided.agent.episode_steps = 100
    assert guided.review_search_branch(nodes[0], "current_room", 0) == (nodes[selected], "agent_search_review")
    options, recommended, event = guided.runtime.choose_frontier.call_args.args
    assert [o["frontier_uid"] for o in options] == [f"node:{i}" for i in range(reachable)]
    assert recommended == "node:0" and event["reason"] == "prolonged_target_search"
    guided.agent.episode_steps = 101
    guided.review_search_branch(nodes[0], "current_room", 0)
    assert guided.runtime.choose_frontier.call_count == 1
    guided.agent.episode_steps = 200
    guided.runtime.model_calls = guided.runtime.max_model_calls
    assert guided.review_search_branch(nodes[0], "current_room", 0) == (nodes[0], "current_room")
    assert guided.runtime.choose_frontier.call_count == 1


def test_frontier_response_schema_limits_model_to_domain_options(tmp_path):
    from harness.runtime import AgentRuntime
    def parse(**kw):
        schema = kw["response_format"]
        with pytest.raises(ValueError):
            schema(frontier_uid="invented", reason="invalid")
        return NS(choices=[NS(message=NS(parsed=schema(frontier_uid="node:2", reason="branch")))])
    client = NS(beta=NS(chat=NS(completions=NS(parse=parse))))
    from harness.tools import NavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from scene.contracts import SemanticMapSnapshot, Pose3D
    from task.session import TaskSession
    session = TaskSession(SemanticMapSnapshot(1, Pose3D((0, 0, 0))),
                          runtime_plan(TaskSpec("find bed", (TargetSpec("bed", "bed"),))))
    tools = NavigationTools(session, NS(phase="idle", held=False, control_epoch=0,
                                        active_request_id=None), None, None)
    runtime = AgentRuntime(tools, tmp_path / "trace.jsonl", max_model_calls=3,
                           client_model=(client, "test"))
    assert runtime.choose_frontier([{"frontier_uid": "node:1"}, {"frontier_uid": "node:2"}],
                                   "node:1", "stalled") == "node:2"
    assert runtime.model_calls == 1


def test_object_viewpoints_are_shared_and_replaced_on_feedback(monkeypatch):
    import navigation.action_controller as actions
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint", lambda *args: 2.)
    from backends.habitat.agent_runtime import HabitatNavigationTools
    from scene.contracts import SemanticMapSnapshot, Pose3D, ObjectNodeSnapshot
    tools = HabitatNavigationTools.__new__(HabitatNavigationTools)
    tools.agent = NS(episode_steps=10, planner=NS(_goal_radius=.25), approach_viewpoint=(1, 2, 0), check_again_postion=(3, 4, 0))
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
    tools.agent.approach_viewpoint = None
    assert [v.metadata["purpose"] for v in tools.publish_object_viewpoints("obj")] == ["recheck"]


def test_check_again_scores_at_camera_height_and_approach_is_best_feasible_view(monkeypatch):
    import importlib, sys
    import navigation.action_controller as actions
    calls = {}
    proposals = [
        {"position": np.array([.2, 0, .2]), "score": .6, "stop_feasible": True},
        {"position": np.array([.5, 0, .2]), "score": .9, "stop_feasible": True},
        {"position": np.array([2., 0, .2]), "score": .95, "stop_feasible": False},
    ]
    def build(**kwargs):
        calls.update(kwargs)
        metrics = dict.fromkeys(["visible_ratio", "center_score", "border_score", "area_ratio", "distance_to_target"], 0.)
        return [{**metrics, **p} for p in proposals], []
    monkeypatch.setitem(sys.modules, "loguru", NS(logger=NS(info=lambda *a, **k: None)))
    monkeypatch.setitem(sys.modules, "habitat_sim", NS(ShortestPath=lambda: NS(points=[])))
    monkeypatch.setitem(sys.modules, "planning.viewpoint_policy", NS(build_check_again_viewpoints=build))
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller",
                        NS(recheck_pitches=lambda agent: (0., -.5)))
    monkeypatch.delitem(sys.modules, "navigation.view_verification_controller", raising=False)
    controller = importlib.import_module("navigation.view_verification_controller")
    monkeypatch.setattr(actions, "geodesic_distance_to_waypoint", lambda agent, p: float(p[0]) + 1.)
    sim = NS(pathfinder=NS(snap_point=lambda p: p, find_path=lambda request: True),
             get_agent_state=lambda: NS(position=np.zeros(3)))
    agent = NS(env=NS(sim=sim), adaptive_observation=True, episode_steps=3,
               mapper=NS(current_position=np.array([1., 1., .2]), initial_position=np.zeros(3),
                         camera_intrinsic=np.eye(3), target="toilet"),
               object_final=NS(find_closest=lambda p: np.zeros(3), position=np.zeros(3), uid="obj"),
               found_goal_position=np.array([1., 1., .2]), confirmed_target_waypoint=None,
               _augment_final_stop_geometry=lambda d: {"required_stop_distance": 1.},
               stop_criterion=.7, planner=NS(_goal_radius=.15),
               rgb_trajectory=[np.zeros((480, 640, 3))], view_control_state=NS(active=False))
    controller.select_check_again_viewpoint(agent)
    assert calls["target_height"] == pytest.approx(.2)
    # 视点评分的俯仰集合与复核执行一致。
    assert calls["pitches"] == (0., -.5)
    np.testing.assert_allclose(agent.approach_viewpoint, [.5, 0, .2])
    np.testing.assert_allclose(agent.check_again_postion, [.2, 0, .2])


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
                        NS(rotate_toward_object_for_recheck=lambda agent: True,
                           tilt_camera=lambda agent, pitch: True))
    state = ViewControlState()
    state.start("obj", {}, {})
    state.exhausted = True
    agent = NS(episode_steps=30, object_final=NS(uid="obj", rgb=np.zeros((4,4,3)), bbox=[0,0,2,2],
                                                 position=np.ones(3)),
               mapper=NS(current_position=np.zeros(3), current_rotation=np.eye(3)[:, ::-1]),
               _observation_identity=lambda uid: {"observation_id": "0:30"},
               task=NS(spec=NS(revision=1)), need_check_again=False,
               selected_candidate=NS(role="primary"),
               last_check_again_evidence={}, final_stop_decision="uncertain",
               check_again=Mock(), final_instruction_check=Mock(return_value=False),
               _continue_better_view_control=lambda: False, view_control_state=state)
    guided = GuidedNavigation.__new__(GuidedNavigation)
    guided.agent, guided.record = agent, Mock()
    guided.search_review_step, guided.search_review_interval = 0, 100
    guided.deferred_candidates = set()
    guided.deferred_evidence, guided.candidate_view_states = {}, {}
    guided.tools = NS(refresh_scene=Mock(), objects={"obj": agent.object_final})
    assert guided.verify((0, 0, 0)) == "recovery"
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
    # 融合算法已抽到共享模块；测试真实函数，几何计算仍使用固定输入。
    path = ROOT / "mapping/object_fusion.py"
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in {"associate_object_entities",
                 "score_object_candidates", "merge_object_observation", "accumulate_label_vote"}]
    namespace = {"np": np, "torch": torch, "nearest_distances": lambda *args: torch.zeros(4),
                 "clouds_may_overlap": lambda *args: True,
                 "gpu_merge_pointcloud": lambda *args: cloud}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    merged, _ = namespace["associate_object_entities"]([old], [new], target="object",
                                                       is_target_tag=lambda tag: target)
    assert merged[0].uid == "0"
    assert merged[0].rgb is new.rgb and merged[0].bbox is new.bbox


def test_physical_infeasibility_reports_limitation_and_preserves_stop_contract():
    namespace = methods("objnav_agent_with_process_obs.py", {"_planner_hard_stop_allows_accept"}, {})
    check = namespace["_planner_hard_stop_allows_accept"]
    for key in ("planner_infeasible", "infeasible_by_geometry"):
        assert not check(NS(), {"hard_stop_constraints": {"satisfied": False, key: True}})


@pytest.fixture
def relation_scene(tmp_path):
    """Keep real constraint, graph, cache and pair-ledger behavior in the replay."""
    from PIL import Image
    from instruction_adapter.constraints import ConstraintEvaluator
    from instruction_adapter.contracts import Constraint, InstructionPlan, TargetQuery
    from instruction_adapter.semantic_edges import RelationPairLedger

    image = tmp_path / "relation.png"
    Image.new("RGB", (8, 4), "red").save(image)
    subject = NS(uid="cup-1", tag="cup", position=[0, 0, 0], bbox=[0, 0, 3, 3], nodes=[0], metadata={})
    anchor = NS(uid="table-1", tag="table", position=[.5, 0, 0], bbox=[4, 0, 7, 3], nodes=[0], metadata={})
    target = TargetQuery("cup", "cup")
    plan = InstructionPlan("find the cup near the table", targets=[target,
        TargetQuery("reference", "table", role="anchor", terminal=False)], constraints=[
        Constraint(type="relation", subject="cup", relation="near", object="table")])
    mapper = NS(objects=[subject, anchor], target=plan.raw_instruction, relation_pair_ledger=RelationPairLedger())
    evaluator = ConstraintEvaluator()
    callback = Mock(return_value={"verified": True, "confidence": .9, "need_better_view": False})
    evaluator.relation_service.vlm_callback = callback
    frame = {"id": "camera:1", "step": 1, "image_ref": str(image),
             "observed_object_uids": [subject.uid, anchor.uid],
             "object_boxes": {subject.uid: list(subject.bbox), anchor.uid: list(anchor.bbox)}}

    def evaluate(evidence):
        return evaluator.evaluate_before_final_verifier(
            mapper=mapper, plan=plan, target=target, candidate=candidate_from_object(subject),
            candidate_obj=subject, evidence=evidence, step=2)

    return NS(subject=subject, anchor=anchor, image=image, frame=frame, mapper=mapper,
              evaluator=evaluator, callback=callback, evaluate=evaluate,
              ledger=mapper.relation_pair_ledger, instruction_hash=instruction_hash(plan.raw_instruction))


@pytest.mark.parametrize("predicate_id", ["candidate_identity", "attribute:goal:color"])
def test_compiler_rejects_explicit_ids_that_hide_generated_predicates(predicate_id):
    from instruction.compiler import compile_parsed_instruction
    from prompting.schemas import ParsedInstruction

    parsed = ParsedInstruction(targets=[{"id": "goal", "name": "cup", "attributes": {"color": "red"}}],
        constraints=[{"id": predicate_id, "type": "attribute", "subject": "goal",
                      "value": {"state": "clean"}, "hardness": "hard", "verifier": "vlm"}])
    result = compile_parsed_instruction("Find a clean red cup.", parsed, {"attribute"})
    assert result.status == "invalid" and result.task is None
    assert "conflicts with a generated verification predicate" in result.diagnostics[0]


@pytest.mark.parametrize("projection_change", ["reorder", "remove", "replace"])
def test_relation_precheck_uses_authoritative_predicate_ids(relation_scene, projection_change):
    from instruction.contracts import ConstraintSpec, TargetSpec, TaskSpec
    from instruction_adapter.task_compat import runtime_plan

    case = relation_scene
    task = TaskSpec("Find the cup near the table and below the shelf.", (
        TargetSpec("goal", "cup"), TargetSpec("table", "table", terminal=False),
        TargetSpec("shelf", "shelf", terminal=False)), (
        ConstraintSpec("near-table", "relation", "goal", "near", "table"),
        ConstraintSpec("below-shelf", "relation", "goal", "below", "shelf")))
    plan = runtime_plan(task)
    if projection_change == "reorder":
        plan.constraints.reverse()
    elif projection_change == "remove":
        plan.constraints.clear()
    else:
        plan.constraints[0].id = "invented"
        plan.constraints[0].relation = "above"
        plan.constraints[0].object = "shelf"
    before = plan.as_dict()
    case.mapper.objects = [case.subject]
    result = case.evaluator.evaluate_before_final_verifier(
        mapper=case.mapper, plan=plan, target=plan.targets[0],
        candidate=candidate_from_object(case.subject), candidate_obj=case.subject, evidence={})
    assert not result.satisfied and result.decision == "need_relation_check"
    assert [(row["predicate_id"], row["relation"], row["anchor_concept_id"], row["status"])
            for row in result.relation_outcomes] == [
        ("near-table", "near", "table", "missing_anchor"),
        ("below-shelf", "below", "shelf", "missing_anchor")]
    assert plan.as_dict() == before
    case.callback.assert_not_called()


def test_relation_precheck_keeps_explicit_id_without_serialized_task(relation_scene):
    from instruction_adapter.contracts import Constraint, InstructionPlan, TargetQuery

    case = relation_scene
    target = TargetQuery("goal", "cup")
    plan = InstructionPlan(case.mapper.target, targets=[target,
        TargetQuery("anchor", "table", terminal=False)], constraints=[
        Constraint("relation", subject="goal", relation="near", object="anchor", id="placement")])
    case.mapper.objects = [case.subject]
    result = case.evaluator.evaluate_before_final_verifier(
        mapper=case.mapper, plan=plan, target=target,
        candidate=candidate_from_object(case.subject), candidate_obj=case.subject, evidence={})
    assert result.relation_outcomes[0]["predicate_id"] == "placement"
    assert result.relation_outcomes[0]["status"] == "missing_anchor"
    case.callback.assert_not_called()


def test_relation_unknown_anchor_keeps_requested_identity_separate_from_detector(relation_scene):
    from instruction_adapter.contracts import ConceptQuery

    case = relation_scene
    case.anchor.tag = "cup"
    case.mapper.concept_matcher = NS(match_many=Mock(return_value=[
        NS(object_uid=case.anchor.uid, matches_concept=None, candidate_eligible=True)]))
    from instruction_adapter.contracts import Constraint, InstructionPlan, TargetQuery
    target = TargetQuery("cup", "cup")
    constraint = Constraint("relation", subject="cup", relation="near", object="table",
                            object_concept=ConceptQuery("anchor", "table", detector_terms=["cup"]))
    instruction = InstructionPlan(case.mapper.target, targets=[target,
        TargetQuery("anchor", "table", role="anchor", terminal=False, concept=constraint.object_concept)],
        constraints=[constraint])
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    case.callback.return_value = {"verified": False, "need_better_view": True, "confidence": .5}
    result = case.evaluator.evaluate_before_final_verifier(
        mapper=case.mapper, plan=instruction, target=target,
        candidate=candidate_from_object(case.subject), candidate_obj=case.subject,
        evidence={"observation_images": [case.frame]}, step=2)
    assert result.decision == "need_relation_check" and not result.satisfied
    subject, reference, prompt = case.callback.call_args.args[1], case.callback.call_args.args[2], case.callback.call_args.args[4]
    assert subject["requested_concept"] == "cup"
    assert reference["requested_concept"] == "table" and reference["detector_label"] == "cup"
    assert reference["id"] == case.anchor.uid and "bbox_2d" not in reference
    assert "Required reference concept: table" in prompt
    assert "Reference detector label: cup" in prompt
    assert case.ledger.get(case.instruction_hash, case.subject.uid, "near", case.anchor.uid).status == "unknown"


def test_relation_preserves_functional_demand_without_replacing_reference_identity(relation_scene):
    import json
    from instruction_adapter.contracts import Constraint, InstructionPlan, TargetQuery
    case = relation_scene
    demand = {"activity": "write", "manipulated_objects": ["notebook"],
              "required_resource": "stable writing surface"}
    target = TargetQuery("surface", "writing surface", demand=demand)
    plan = InstructionPlan("Find somewhere to write near the table", targets=[target,
        TargetQuery("reference", "table", role="anchor", terminal=False)], constraints=[
        Constraint("relation", subject=target.name, relation="near", object="table")])
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    result = case.evaluator.evaluate_before_final_verifier(
        mapper=case.mapper, plan=plan, target=target, candidate=candidate_from_object(case.subject),
        candidate_obj=case.subject, evidence={"observation_images": [case.frame]}, step=2)
    assert result.satisfied
    _, subject, reference, _, prompt = case.callback.call_args.args
    assert subject["requested_concept"] == "writing surface"
    assert subject["demand"] == demand and subject["detector_label"] == "cup"
    assert reference["requested_concept"] == "table" and reference["demand"] == {}
    assert "Subject demand: " + json.dumps(demand) in prompt
    assert "Required reference concept: table" in prompt


@pytest.mark.parametrize("observations", ["map_only", "separate_frames", "one_bound_instance"])
def test_map_membership_does_not_invent_relation_coview(relation_scene, observations):
    case = relation_scene
    if observations != "map_only":
        case.subject.image_ref = str(case.image)
        case.subject.metadata = {"frame_id": "camera:1", "step": 1}
        if observations == "separate_frames":
            case.anchor.image_ref = str(case.image)
            case.anchor.metadata = {"frame_id": "camera:2", "step": 1}
    case.evaluator.register_mapper_objects(
        mapper=case.mapper, step=1, current_node_idx=0, rgb_path=str(case.image))
    assert case.evaluator.spatial_graph.co_visible_views(case.subject.uid, case.anchor.uid) == []
    result = case.evaluate({"current_rgb_with_bbox_path": str(case.image), "node_idx": 0})
    assert not result.satisfied and result.decision == "need_relation_check"
    assert result.relation_edges[0].verdict == "unknown"
    case.callback.assert_not_called()
    record = case.ledger.get(case.instruction_hash, case.subject.uid, "near", case.anchor.uid)
    assert record.status == "unknown"
    assert not case.ledger.is_rejected(case.instruction_hash, case.subject.uid, "near", case.anchor.uid)


@pytest.mark.parametrize("initial", ["unknown", "false", "true"])
def test_relation_new_frame_revises_pair_ledger_without_sticky_judgment(relation_scene, initial):
    case = relation_scene
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    case.callback.return_value = {"verified": initial == "true", "confidence": .9,
                                  "need_better_view": initial == "unknown"}
    first = case.evaluate({"observation_images": [case.frame]})
    assert first.relation_edges[0].verdict == initial
    key = (case.instruction_hash, case.subject.uid, "near", case.anchor.uid)
    assert case.ledger.is_rejected(*key) == (initial == "false")
    assert case.ledger.is_accepted(*key) == (initial == "true")
    if initial == "unknown":
        assert first.decision == "need_relation_check" and case.ledger.get(*key).status == "unknown"
    case.evaluate({"observation_images": [case.frame]})
    assert case.callback.call_count == 1
    expected = initial != "true"
    case.callback.return_value = {"verified": expected, "confidence": .9, "need_better_view": False}
    new_frame = {**case.frame, "id": "camera:2", "step": 2}
    changed = case.evaluate({"observation_images": [new_frame]})
    assert case.callback.call_count == 2
    assert changed.satisfied is expected
    assert changed.decision == ("pass" if expected else "reject_relation")
    assert case.ledger.is_accepted(*key) is expected
    assert case.ledger.is_rejected(*key) is not expected
    assert "camera:2" in case.ledger.get(*key).evidence_view_ids


@pytest.mark.parametrize("replacement", ["subject", "anchor"])
def test_relation_evidence_and_acceptance_never_transfer_between_uids(relation_scene, replacement):
    case = relation_scene
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    assert case.evaluate({"observation_images": [case.frame]}).satisfied
    old_key = (case.instruction_hash, case.subject.uid, "near", case.anchor.uid)
    getattr(case, replacement).uid += "-new"
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    new_key = (case.instruction_hash, case.subject.uid, "near", case.anchor.uid)
    unbound = case.evaluate({"observation_images": [case.frame]})
    assert unbound.decision == "need_relation_check" and not unbound.satisfied
    assert unbound.relation_edges[0].verdict == "unknown"
    assert case.callback.call_count == 1
    assert case.ledger.is_accepted(*old_key)
    assert case.ledger.get(*new_key).status == "unknown"
    assert case.evaluator.spatial_graph.co_visible_views(case.subject.uid, case.anchor.uid) == []
    new_frame = {**case.frame, "id": "camera:2", "step": 2,
                 "observed_object_uids": [case.subject.uid, case.anchor.uid],
                 "object_boxes": {case.subject.uid: list(case.subject.bbox),
                                  case.anchor.uid: list(case.anchor.bbox)}}
    case.callback.return_value = {"verified": False, "confidence": .9, "need_better_view": False}
    assert case.evaluate({"observation_images": [new_frame]}).decision == "reject_relation"
    assert case.callback.call_count == 2
    assert case.ledger.is_rejected(*new_key) and case.ledger.is_accepted(*old_key)


@pytest.mark.parametrize("missing_box", [None, []])
def test_record_observation_indexes_only_candidates_with_frame_boxes(relation_scene, missing_box):
    case = relation_scene
    graph = case.evaluator.spatial_graph
    case.anchor.bbox = missing_box
    for obj in (case.subject, case.anchor):
        graph.record_observation(candidate=candidate_from_object(obj), view_id="camera:1",
                                 step=1, rgb_path=str(case.image))
    assert set(graph.objects) == {case.subject.uid, case.anchor.uid}
    assert graph.objects[case.subject.uid].observed_view_ids == ["camera:1"]
    assert graph.objects[case.anchor.uid].observed_view_ids == []
    assert graph.views["camera:1"].observed_object_uids == [case.subject.uid]
    assert graph.views["camera:1"].object_boxes == {case.subject.uid: case.subject.bbox}
    assert graph.co_visible_views(case.subject.uid, case.anchor.uid) == []

    # Supplying the missing region later may legitimately complete this frame.
    case.anchor.bbox = [4, 0, 7, 3]
    graph.record_observation(candidate=candidate_from_object(case.anchor), view_id="camera:1",
                             step=1, rgb_path=str(case.image))
    assert [view.id for view in graph.co_visible_views(case.subject.uid, case.anchor.uid)] == ["camera:1"]
    assert graph.objects[case.anchor.uid].observed_view_ids == ["camera:1"]


def test_upsert_view_refreshes_known_step_without_erasing_it(relation_scene):
    case = relation_scene
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    graph = case.evaluator.spatial_graph
    kwargs = dict(view_id="camera:1", rgb_path=str(case.image),
                  observed_object_uids=case.frame["observed_object_uids"],
                  object_boxes=case.frame["object_boxes"])
    graph.upsert_view(**kwargs)
    assert graph.views["camera:1"].step is None
    for incoming, expected in [(0, 0), (30, 30), (None, 30)]:
        graph.upsert_view(**kwargs, step=incoming)
        assert graph.views["camera:1"].step == expected
    for uid in case.frame["observed_object_uids"]:
        assert graph.objects[uid].observed_view_ids == ["camera:1"]
    assert graph.views["camera:1"].observed_object_uids == case.frame["observed_object_uids"]


def test_coviews_sort_known_steps_descending_then_reverse_insertion(relation_scene):
    case = relation_scene
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    graph = case.evaluator.spatial_graph
    for view_id, step in [("unknown-first", None), ("latest", 40), ("old", 2),
                          ("tie-first", 10), ("tie-last", 10), ("zero", 0),
                          ("unknown-last", None)]:
        graph.upsert_view(view_id=view_id, step=step, rgb_path=str(case.image),
                          observed_object_uids=case.frame["observed_object_uids"],
                          object_boxes=case.frame["object_boxes"])
    expected = ["latest", "tie-last", "tie-first", "old", "zero", "unknown-last", "unknown-first"]
    assert [view.id for view in graph.co_visible_views(case.subject.uid, case.anchor.uid, limit=20)] == expected
    assert [view.id for view in graph.co_visible_views(case.subject.uid, case.anchor.uid)] == expected[:4]
    graph.upsert_view(view_id="old", step=50)
    assert graph.co_visible_views(case.subject.uid, case.anchor.uid)[0].id == "old"


@pytest.mark.parametrize("new_step", [0, 2])
def test_new_relation_frame_survives_four_view_limit_over_unknown_history(relation_scene, new_step):
    case = relation_scene
    case.evaluator.register_mapper_objects(mapper=case.mapper)
    historical = [{**case.frame, "id": f"history:{i}", "step": None} for i in range(5)]
    case.callback.return_value = {"verified": False, "confidence": .9, "need_better_view": False}
    assert case.evaluate({"observation_images": historical}).decision == "reject_relation"
    case.callback.assert_called_once()
    assert [view["id"] for view in case.callback.call_args.args[3]] == [
        "history:4", "history:3", "history:2", "history:1"]

    case.callback.return_value = {"verified": True, "confidence": .9, "need_better_view": False}
    current = {**case.frame, "id": "current", "step": new_step}
    result = case.evaluate({"observation_images": [current]})
    assert result.satisfied and result.decision == "pass"
    assert case.callback.call_count == 2
    assert [view["id"] for view in case.callback.call_args.args[3]] == [
        "current", "history:4", "history:3", "history:2"]
    record = case.ledger.get(case.instruction_hash, case.subject.uid, "near", case.anchor.uid)
    assert "current" in record.evidence_view_ids
    assert case.ledger.is_accepted(case.instruction_hash, case.subject.uid, "near", case.anchor.uid)


@pytest.mark.parametrize("missing", ["box", "observed_uid", "image"])
def test_relation_vlm_gate_rejects_unbound_low_level_view_index(relation_scene, missing):
    case = relation_scene
    graph = case.evaluator.spatial_graph
    # The low-level object API permits indexing; the VLM gate must check binding.
    for obj in (case.subject, case.anchor):
        graph.upsert_object(candidate_from_object(obj), view_id="unbound")
    boxes = dict(case.frame["object_boxes"])
    uids = list(case.frame["observed_object_uids"])
    image = str(case.image)
    if missing == "box":
        boxes.pop(case.anchor.uid)
    elif missing == "observed_uid":
        uids.remove(case.anchor.uid)
    else:
        image = str(case.image.with_name("missing.png"))
    graph.upsert_view(view_id="unbound", step=1, rgb_path=image,
                      observed_object_uids=uids, object_boxes=boxes)
    result = case.evaluate({})
    assert not result.satisfied and result.decision == "need_relation_check"
    assert result.relation_edges[0].verdict == "unknown"
    case.callback.assert_not_called()
    assert case.ledger.get(case.instruction_hash, case.subject.uid, "near", case.anchor.uid).status == "unknown"


@pytest.mark.parametrize("anchor_depth", [.5, 2.])
def test_habitat_relation_requires_current_depth_alignment_for_both_uids(relation_scene, monkeypatch, anchor_depth):
    import sys
    from types import ModuleType
    from PIL import Image
    from backends.habitat.scene import current_observation

    case = relation_scene
    geometry = ModuleType("mapping_utils.geometry")
    geometry.project_to_camera = Mock(side_effect=lambda pcd, *_: tuple(pcd.T))
    monkeypatch.setitem(sys.modules, "mapping_utils.geometry", geometry)
    case.subject.pcd = np.array([[0, 0, 1.], [2, 0, 1.], [0, 2, 1.]])
    case.anchor.pcd = np.array([[4, 0, anchor_depth], [6, 0, anchor_depth], [4, 2, anchor_depth]])
    case.mapper.camera_intrinsic = np.eye(3)
    case.mapper.current_position = np.zeros(3)
    case.mapper.current_rotation = np.eye(3)
    agent = NS(mapper=case.mapper, obs={"depth": np.ones((4, 8, 1))},
               episode_samples=2, episode_steps=12, save_dir=str(case.image.parent))
    case.evaluator.register_mapper_objects(mapper=case.mapper)

    def capture():
        frame = current_observation(agent)
        path = Path(frame["image_ref"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 4), "red").save(path)
        return frame

    unbound = capture()
    assert unbound["observed_object_uids"] == [case.subject.uid]
    assert set(unbound["object_boxes"]) == {case.subject.uid}
    result = case.evaluate({"observation_images": [unbound]})
    assert not result.satisfied and result.decision == "need_relation_check"
    assert result.relation_edges[0].verdict == "unknown"
    case.callback.assert_not_called()
    graph = case.evaluator.spatial_graph
    assert graph.objects[case.anchor.uid].observed_view_ids == []
    assert graph.co_visible_views(case.subject.uid, case.anchor.uid) == []

    agent.episode_steps = 13
    agent.obs["depth"][:, 4:, :] = anchor_depth
    bound = capture()
    assert bound["id"] == "1:13"
    assert set(bound["observed_object_uids"]) == {case.subject.uid, case.anchor.uid}
    result = case.evaluate({"observation_images": [bound]})
    assert result.satisfied and result.decision == "pass"
    case.callback.assert_called_once()
    views = case.callback.call_args.args[3]
    assert [view["id"] for view in views] == [bound["id"]]
    assert views[0]["rgb_path"] == bound["image_ref"]
    assert views[0]["object_boxes"] == bound["object_boxes"]


@pytest.fixture
def habitat_interpretation_case():
    """Use the real session and supervisor without constructing a simulator or model."""
    from dataclasses import replace
    from backends.habitat.agent_runtime import HabitatNavigationTools
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from instruction_adapter.view_control import ViewControlState
    from motion.supervisor import MotionSupervisor
    from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    spec = TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),),
                    search_priors={"destination_hypotheses": {"goal": ["chair"]}})
    snapshot = SemanticMapSnapshot(12., Pose3D((1., 2., 0.)),
                                   objects=(ObjectNodeSnapshot("obj", "chair", (3., 2., 0.)),))
    session = TaskSession(snapshot, runtime_plan(spec))
    revised = runtime_plan(replace(spec, revision=2,
        targets=(TargetSpec("goal", "sofa", aliases=("couch",)),),
        search_priors={"destination_hypotheses": {"goal": ["sofa", "bench"]}}))
    state = ViewControlState()
    state.start("obj", {"reason": "old task"}, {"observation_identity": {"task_revision": 1}})
    state.exhausted = True
    controller = NS(send_goal=Mock(return_value="motion:old"), advance=Mock())
    motion = MotionSupervisor(controller)
    node = NS(idx=0, state=1, position=(0., 0., 0.))
    mapper = NS(target="chair", target_list=["chair"], target_aliases=["chair"],
                current_position=np.array([1., 2., 0.]), current_rotation=np.eye(3),
                nodes=[node], objects=[object()], current_pcd=object(), traj=[0], change_state=Mock(),
                search_prior_result=object(), prior_map_prompt_context=object(),
                prior_map_high_level_selection={"selected_uid": "old-room"},
                prior_map_last_chosen_frontier={"selected_uid": "old-frontier"},
                prior_map_current_step=125)
    agent = NS(task=session, mapper=mapper, execution=NS(motion=motion, controller=controller),
               selected_candidate=object(), object_final=object(), found_goal=True, need_check_again=True,
               found_goal_position=(1., 2., 0.), waypoint=(3., 2., 0.), confirmed_target_waypoint=(3., 2., 0.),
               approach_viewpoint=(2., 2., 0.), check_again_postion=(2., 3., 0.), view_control_state=state,
               last_check_again_evidence={"old": True}, current_surface_evidence=("old",),
               last_check_again_image_path="old.png", scan_complete=True,
               semantic_scan_headings={(1, 1., 2., 0.): {0, 1, 2}},
               semantic_label_evidence={"obj": NS(task_revision=1)},
               episode_steps=125, travel_distance=8., current_node_idx=0,
               final_instruction_accepted_this_step=True,
               instruction_success=True, final_stop_success=True,
               instruction_decision="accept", final_stop_decision="accept", final_stop_mode="agent",
               instruction_accept_step=125, final_stop_accept_step=125, accepted_candidate_uid="obj",
               accepted_relation_edge={"subject_uid": "obj", "verdict": "true"},
               accepted_distance_to_target=.5, accepted_distance_source="current_instance_raw_depth",
               final_verifier_attempt_counter={"old-revision": 3})
    tools = HabitatNavigationTools(agent)
    tools.objects = {"obj": agent.object_final}
    tools.deferred_candidates.add("obj")
    return NS(agent=agent, tools=tools, session=session, revised=revised,
              arguments={"task_revision": 1, "search_revision": session.search_revision,
                         "snapshot": snapshot, "record": {"control_epoch": 0, "reason": "reconsider seating"}})


def test_habitat_interpretation_syncs_only_task_local_state(habitat_interpretation_case, monkeypatch):
    from harness.tools import NavigationTools

    case = habitat_interpretation_case
    tools, agent, session = case.tools, case.agent, case.session
    mapper_before = vars(agent.mapper).copy()
    pose_before = agent.mapper.current_position.copy(), agent.mapper.current_rotation.copy()
    old_state, old_progress, old_ledger = agent.view_control_state, session.progress, session.verification_ledger
    old_progress.completed = True
    deferred, objects = tools.deferred_candidates, tools.objects
    headings, labels = agent.semantic_scan_headings, agent.semantic_label_evidence
    base = NavigationTools.install_interpretation
    calls = []

    def install(self, plan, **kwargs):
        result = base(self, plan, **kwargs)
        calls.append((self, plan, kwargs, result))
        return result

    monkeypatch.setattr(NavigationTools, "install_interpretation", install)
    result = tools.install_interpretation(case.revised, **case.arguments)
    assert len(calls) == 1 and calls[0][:3] == (tools, case.revised, case.arguments)
    assert result is calls[0][3]
    assert result["task"]["revision"] == 2 and result["previous_task"]["revision"] == 1
    assert session.spec.revision == 2 and session.progress is not old_progress
    assert session.verification_ledger is not old_ledger
    assert not session.progress.completed and not tools.propose_finish()["accepted"]
    assert agent.mapper.target == session.progress.active_target().name == "sofa"
    assert agent.mapper.target_list == session.detector_terms == ["sofa", "bench"]
    assert agent.mapper.target_aliases == ["sofa", "couch"]
    assert tools.deferred_candidates is deferred and not deferred
    assert tools.objects is objects and "obj" in objects
    for name in ("selected_candidate", "object_final", "found_goal_position", "waypoint",
                 "confirmed_target_waypoint", "approach_viewpoint", "check_again_postion",
                 "last_check_again_evidence", "current_surface_evidence"):
        assert getattr(agent, name) is None
    assert not agent.found_goal and not agent.need_check_again and not agent.scan_complete
    assert agent.last_check_again_image_path == ""
    assert not agent.final_instruction_accepted_this_step
    assert agent.instruction_success is False and agent.final_stop_success is False
    assert agent.instruction_accept_step is None and agent.final_stop_accept_step is None
    assert agent.instruction_decision == agent.final_stop_decision == agent.final_stop_mode == ""
    assert agent.accepted_candidate_uid == agent.accepted_distance_source == ""
    assert agent.accepted_distance_to_target is None and agent.accepted_relation_edge == {}
    assert agent.view_control_state is not old_state and not agent.view_control_state.active
    assert old_state.active and old_state.exhausted and old_state.task_revision == 1
    assert agent.semantic_scan_headings is headings and headings == {(1, 1., 2., 0.): {0, 1, 2}}
    assert agent.semantic_label_evidence is labels and labels["obj"].task_revision == 1
    prior_fields = {"search_prior_result", "prior_map_prompt_context", "prior_map_high_level_selection",
                    "prior_map_last_chosen_frontier", "prior_map_current_step"}
    assert all(getattr(agent.mapper, name) is None for name in prior_fields)
    for name, value in mapper_before.items():
        if name not in {"target", "target_list", "target_aliases"} | prior_fields:
            assert getattr(agent.mapper, name) is value
    np.testing.assert_array_equal(agent.mapper.current_position, pose_before[0])
    np.testing.assert_array_equal(agent.mapper.current_rotation, pose_before[1])
    assert agent.mapper.nodes[0].state == 1 and agent.mapper.traj == [0]
    assert agent.episode_steps == 125 and agent.travel_distance == 8. and agent.current_node_idx == 0
    assert agent.final_verifier_attempt_counter == {"old-revision": 3}
    assert tools.motion.control_epoch == 0 and tools.motion.active_request_id is None
    tools.motion.controller.send_goal.assert_not_called()
    tools.motion.controller.advance.assert_not_called()


@pytest.mark.parametrize("invalid", ["task_revision", "search_revision", "snapshot", "moving", "held"])
def test_habitat_rejected_interpretation_leaves_backend_untouched(habitat_interpretation_case, invalid):
    from dataclasses import replace
    from motion.contracts import MotionGoalMode, NavigationIntent
    from scene.contracts import Pose3D

    case = habitat_interpretation_case
    tools, agent, session = case.tools, case.agent, case.session
    arguments = dict(case.arguments)
    if invalid in {"task_revision", "search_revision"}:
        arguments[invalid] += 1
    elif invalid == "snapshot":
        arguments["snapshot"] = replace(session.snapshot, timestamp=13.)
    elif invalid == "moving":
        intent = NavigationIntent(MotionGoalMode.GO_TO_FRONTIER, goal_pose=Pose3D((4., 0., 0.)))
        tools.motion.submit(intent, intent.to_motion_goal(), request_id="active", control_epoch=0)
    else:
        tools.motion.held = True
    before_agent, before_mapper = vars(agent).copy(), vars(agent.mapper).copy()
    before_spec, before_progress = session.spec, session.progress
    with pytest.raises(RuntimeError if invalid in {"moving", "held"} else ValueError):
        tools.install_interpretation(case.revised, **arguments)
    assert session.spec is before_spec and session.progress is before_progress
    assert session.snapshot is case.arguments["snapshot"]
    assert tools.deferred_candidates == {"obj"} and agent.view_control_state.active
    for name, value in before_agent.items():
        assert getattr(agent, name) is value
    for name, value in before_mapper.items():
        assert getattr(agent.mapper, name) is value
    assert tools.motion.active_request_id == ("active" if invalid == "moving" else None)
    assert tools.motion.controller.send_goal.call_count == int(invalid == "moving")
    tools.motion.controller.advance.assert_not_called()


def test_habitat_adoption_revokes_verified_success_before_finish(habitat_interpretation_case, monkeypatch):
    import sys

    case = habitat_interpretation_case
    agent, tools, session = case.agent, case.tools, case.session
    forbidden = Mock(side_effect=AssertionError("adoption must not capture or move"))
    monkeypatch.setitem(sys.modules, "navigation.goal_approach_controller", NS(tilt_camera=forbidden))
    tools.refresh_scene = forbidden
    tools.evidence_provider = Mock(return_value={})
    tools.verifier = Mock()
    tools.verifier.verify.return_value = VerificationResult(
        True, "accept", semantic_satisfied=True, view_sufficient_for_stop=True,
        task_revision=1, target_uid="obj")
    agent.instruction_success = agent.final_stop_success = False
    assert tools.verify_target("obj", "goal", 1)["decision"] == "accept"
    assert session.progress.completed and agent.instruction_success and agent.final_stop_success
    previous_progress = session.progress.as_dict()
    previous_ledger = session.verification_ledger.as_dict()
    result = tools.install_interpretation(case.revised, **case.arguments)
    assert result["previous_progress"] == previous_progress
    assert result["previous_verification"] == previous_ledger
    assert session.interpretation_history[-1] == result
    assert not session.progress.completed and not tools.propose_finish()["accepted"]
    assert agent.instruction_success is False and agent.final_stop_success is False
    assert agent.instruction_accept_step is None and agent.final_stop_accept_step is None
    tools.verifier.verify.assert_called_once()
    tools.evidence_provider.assert_called_once_with("obj")
    forbidden.assert_not_called()


@pytest.mark.parametrize("persistent_region", [False, True])
def test_habitat_adoption_invalidates_prior_without_erasing_map(
        habitat_interpretation_case, tmp_path, monkeypatch, persistent_region):
    from copy import deepcopy
    import planning.exploration_policy as policy
    from prior_map.alignment import PriorMapAlignment
    from prior_map.contracts import PriorMapData, PriorObject, PriorRoom
    from prior_map.high_level_selector import HighLevelSelectionResult
    from prior_map.simulation import (PriorMapSimulationRuntime, configure_mapper_prior_map,
                                      refresh_mapper_prior_map_query)

    case = habitat_interpretation_case
    agent, mapper, session = case.agent, case.agent.mapper, case.session
    monkeypatch.setattr(policy, "_geodesic_distances_to_nodes", lambda mapper, nodes: np.ones(len(nodes)))
    rooms = tuple(PriorRoom(uid=f"room:{i}", label=f"room {i}", centroid_xy=(x, 0.),
        boundary_xy=((x-1, -1), (x+1, -1), (x+1, 1), (x-1, 1)) if persistent_region else ())
        for i, x in enumerate((0., 4.)))
    base = PriorMapData(scene_id="test", rooms=rooms, objects=tuple(
        PriorObject(uid=f"prior:{label}", label=label, position_xyz=(x, 0., 0.),
                    parent_room_uid=f"room:{i}")
        for i, (label, x) in enumerate((("chair", 0.), ("sofa", 4.)))))
    mapper.nodes = [NS(idx=i, state=0, has_frontier=True, position=np.array([x, 0., 0.]))
                    for i, x in enumerate((0., 4.))]
    mapper.room_nodes = [NS(room_id=i, uid=f"room:{i}", label=f"room {i}", nodes=[node])
                         for i, node in enumerate(mapper.nodes)]
    mapper.initial_position = np.zeros(3)
    selector = Mock()
    prefix = "room:" if persistent_region else "runtime_room:"
    selector.select.side_effect = [HighLevelSelectionResult(
        selected_uid=f"{prefix}{i}", selected_type="room", decision="select") for i in (0, 1)]
    runtime = PriorMapSimulationRuntime(base_map=base, alignment=PriorMapAlignment.identity(),
        artifact_root=tmp_path, high_level_selector=selector,
        room_semantic_classifier=NS(classify=Mock(side_effect=AssertionError("room evidence is unchanged"))))
    configure_mapper_prior_map(mapper, runtime)
    runtime.begin_episode(tmp_path, 0)
    runtime.last_room_semantics = {"0": {"label": "room 0"}, "1": {"label": "room 1"}}
    runtime.last_room_semantic_step = agent.episode_steps
    runtime.query_service.query = Mock(wraps=runtime.query_service.query)
    first = refresh_mapper_prior_map_query(mapper, plan=session.plan, step=agent.episode_steps)
    session.prior_result = first
    session.prior_map_prompt_context = runtime.last_prompt_context
    session.prior_map_high_level_selection = mapper.prior_map_high_level_selection
    session.prior_map_room_semantics = runtime.last_room_semantics
    session.prior_map_policy_adapter = runtime.policy_adapter
    mapper.prior_map_last_chosen_frontier = runtime.last_chosen_frontier = {"selected_uid": "node:0"}
    assert selector.select.call_count == 1 and not runtime._region_decision_due()
    assert first.object_rankings[0].object_uid == "prior:chair"
    runtime_before = vars(runtime).copy()
    mapper_before = vars(mapper).copy()
    memory_before = deepcopy(runtime.memory.state_dict())
    first_before = deepcopy(first.to_dict())

    case.tools.install_interpretation(case.revised, **case.arguments)

    assert runtime.query_service.query.call_count == selector.select.call_count == 1
    runtime.room_semantic_classifier.classify.assert_not_called()
    assert session.prior_result is None and session.prior_map_prompt_context is None
    assert session.prior_map_high_level_selection is None
    assert session.prior_map_room_semantics is runtime_before["last_room_semantics"]
    cleared = {"last_query_result", "last_query_step", "last_prompt_context", "last_multimodal_context",
               "last_chosen_frontier", "last_high_level_selection", "last_high_level_step",
               "last_region_signature"}
    for name, value in runtime_before.items():
        if name in cleared:
            assert getattr(runtime, name) is None
        elif name != "shared_snapshot":
            assert getattr(runtime, name) is value
    assert runtime.shared_snapshot == {**runtime_before["shared_snapshot"], "selection": None}
    assert runtime_before["shared_snapshot"]["selection"]["selected_uid"] == f"{prefix}0"
    assert runtime._region_decision_due()
    prior_fields = {"search_prior_result", "prior_map_prompt_context", "prior_map_high_level_selection",
                    "prior_map_last_chosen_frontier", "prior_map_current_step"}
    for name, value in mapper_before.items():
        if name in prior_fields:
            assert getattr(mapper, name) is None
        elif name not in {"target", "target_list", "target_aliases"}:
            assert getattr(mapper, name) is value
    assert runtime.memory.state_dict() == memory_before
    assert first.to_dict() == first_before

    second = refresh_mapper_prior_map_query(mapper, plan=session.plan, step=agent.episode_steps)
    assert second is not first and second.object_rankings[0].object_uid == "prior:sofa"
    assert mapper.prior_map_high_level_selection["selected_uid"] == f"{prefix}1"
    assert runtime.query_service.query.call_count == selector.select.call_count == 2
    assert selector.select.call_args.kwargs["instruction_plan"] is session.plan
    assert runtime.memory.state_dict() == memory_before
    assert runtime.last_observation is runtime_before["last_observation"]
    assert first.to_dict() == first_before
    assert not runtime._region_decision_due()
    assert refresh_mapper_prior_map_query(mapper, plan=session.plan, step=agent.episode_steps) is second
    assert runtime.query_service.query.call_count == selector.select.call_count == 2
    runtime.room_semantic_classifier.classify.assert_not_called()
