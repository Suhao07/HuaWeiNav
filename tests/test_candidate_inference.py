"""Evidence identity, ordered concurrent commits, and explicit request failures."""
import importlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    # Rendering is irrelevant to these inference contracts; the Habitat image
    # environment supplies supervision for real episode testing.
    monkeypatch.setitem(sys.modules, 'cv_utils.visualizer', NS(visualize_mask=lambda image, *args: image.copy()))
    gpt = importlib.import_module('cv_utils.gpt_utils')
    bbox = importlib.import_module('navigation.bbox_refinement')
    return gpt, bbox


def test_exact_cache_includes_full_context_and_reports_corruption(tmp_path, monkeypatch, modules):
    import torch
    gpt, _ = modules
    monkeypatch.setattr(gpt, '_get_client_and_model', lambda _: (None, 'doubao-test'))
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    box = torch.tensor([[12, 12, 18, 18]])
    a = gpt.prepare_bbox_request(image, box, str(tmp_path), 0, 1, 0, 'ark')
    image[0:5, 0:5] = 255
    b = gpt.prepare_bbox_request(image, box, str(tmp_path), 0, 1, 0, 'ark')
    assert a['key'] != b['key']
    path = tmp_path/'episode-0/detection/object_box_cache.json'
    path.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        gpt.load_bbox_cache(str(tmp_path), 0)


def candidate_setup(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    candidates = [(i, NS(uid=f'o{i}', tag='chair'), NS(image=image, bbox_tensor=None,
                   bbox_xyxy=[0,0,7,7], bbox_real_xyxy=None)) for i in range(2)]
    (tmp_path/'episode-0').mkdir()
    agent = NS(save_dir=str(tmp_path), episode_samples=1, episode_steps=4, vlm='ark',
               task=NS(spec=NS(revision=1)), mapper=NS(target='chair',
               objects=[], object_perceiver=NS(classes=['chair'])), semantic_label_evidence={})
    return agent, candidates


@pytest.mark.parametrize('failure', [False, True])
def test_parallel_inference_commits_in_input_order_or_propagates(tmp_path, monkeypatch, modules, failure):
    gpt, bbox = modules
    agent, candidates = candidate_setup(tmp_path)
    monkeypatch.setenv('STRIVE_EVIDENCE_REUSE', '0')
    monkeypatch.setenv('STRIVE_VLM_WORKERS', '2')
    monkeypatch.setattr(gpt, 'prepare_bbox_request', lambda image, box, save, ep, step, i, vlm:
                        {'model':'doubao-test', 'key':str(i), 'index':i})
    barrier = threading.Barrier(2)
    completion_order, commits = [], []
    def classify(request):
        barrier.wait(timeout=3)
        if request['index'] == 0:
            time.sleep(.03)
            if failure:
                raise ValueError('candidate zero failed')
        completion_order.append(request['index'])
        return 'chair'
    monkeypatch.setattr(gpt, 'classify_bbox_request', classify)
    monkeypatch.setattr(bbox, 'apply_refined_tag', lambda agent, obj, tag, image, **kwargs: commits.append(obj.uid))
    if failure:
        with pytest.raises(ValueError, match='candidate zero failed'):
            bbox.refine_candidates(agent, candidates)
        assert commits == []
    else:
        records = bbox.refine_candidates(agent, candidates)
        assert completion_order == [1,0]
        assert commits == ['o0','o1']
        assert list(records) == commits


def test_concurrent_trace_ids_preserve_all_calls(tmp_path, monkeypatch):
    from llm_utils.lvlm_call_tracker import reset_counts, record_call, counts
    monkeypatch.setenv('STRIVE_LVLM_TRACE_DIR', str(tmp_path))
    reset_counts()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: record_call('probe', raw_response=str(i)), range(64)))
    rows = [json.loads(p.read_text()) for p in tmp_path.glob('[0-9]*.json')]
    assert len(rows) == 64 and len({r['index'] for r in rows}) == 64
    assert {r['raw_response'] for r in rows} == {str(i) for i in range(64)}
    assert counts()['total_calls'] == 64


def test_same_uid_changed_support_expires_signature():
    from navigation.semantic_label_cache import instance_signature, matching_instance
    def obj(points):
        array = np.asarray(points, dtype=float)
        return NS(uid='same', alive=True, tag='chair', pcd=NS(point=NS(
            positions=NS(cpu=lambda: NS(numpy=lambda: array)))))
    a = obj([[0,0,0],[.05,0,0]])
    b = obj([[1,0,0],[1.05,0,0]])
    assert instance_signature(a) != instance_signature(b)
    points = a.pcd.point.positions.cpu().numpy()
    assert matching_instance(points,[a,a]) is None


def test_cached_judgment_preserves_observed_confidence(monkeypatch, modules):
    import torch
    _, bbox = modules
    agent = NS(mapper=NS(target='chair', target_aliases=[]))
    obj = NS(tag='seat', num_list={'seat':1}, conf_list={'seat':.4}, confidence=torch.tensor(.4))
    bbox.apply_refined_tag(agent, obj, 'chair', None, new_judgment=False)
    assert obj.tag == 'chair' and float(obj.confidence) == pytest.approx(.4)


def test_request_signature_tracks_model_schema_and_prompt():
    from llm_utils.structured_output import request_fingerprint
    from prompting.schemas import ParsedVerification, BBoxObjectLabelResponse
    a = request_fingerprint('doubao-test',['prompt'],BBoxObjectLabelResponse,provider='ark')
    assert a != request_fingerprint('doubao-other',['prompt'],BBoxObjectLabelResponse,provider='ark')
    assert a != request_fingerprint('doubao-test',['new prompt'],BBoxObjectLabelResponse,provider='ark')
    assert a != request_fingerprint('doubao-test',['prompt'],ParsedVerification,provider='ark')


@pytest.mark.parametrize("method", ["match_object", "match_many"])
@pytest.mark.parametrize("terminal", [True, False])
def test_name_identity_and_alias_detector_candidates_have_distinct_authority(monkeypatch, method, terminal):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    monkeypatch.setenv("LLM_OFFLINE", "1")
    concept = ConceptQuery("goal", "mug", aliases=["coffee mug"],
                           detector_terms=["container"], terminal=terminal)
    matcher = RuntimeConceptMatcher()
    for label in ("mug", " MUG ", "coffee_mug", "container", "wall"):
        obj = NS(uid=label, tag=label, position=[0, 0, 0], confidence=0.99)
        args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
        result = getattr(matcher, method)(raw_instruction="find mug", concept=concept, **args)
        row = result[0] if method == "match_many" else result
        identity = label.strip().lower() == "mug"
        assert row.source == ("exact" if identity else "offline")
        assert row.matches_concept is (True if identity else None)
        assert row.terminal_eligible is (identity and terminal)
        assert row.candidate_eligible is (label != "wall")
        assert row.as_dict()["verdict"] == ("true" if identity else "unknown")
        assert obj.tag == label


def _scripted_concept_client(monkeypatch, judgments):
    from instruction_adapter import concept_matcher
    calls = []
    def parse(**kwargs):
        judgment = judgments[len(calls)]
        calls.append(kwargs)
        payload = {"object": judgment} if kwargs["trace_label"] == "concept_match_batch" else judgment
        parsed = kwargs["response_format"](**payload)
        return NS(choices=[NS(message=NS(parsed=parsed))])
    client = NS(beta=NS(chat=NS(completions=NS(parse=parse))))
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setattr(concept_matcher, "get_client_and_model", lambda _: (client, "scripted"))
    return calls


@pytest.mark.parametrize("method", ["match_object", "match_many"])
@pytest.mark.parametrize("verdict,candidate", [(True, False), (False, True), (None, True), (None, False)])
def test_broad_alias_requires_semantic_judgment_not_exact_identity(monkeypatch, method, verdict, candidate):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    calls = _scripted_concept_client(monkeypatch, [dict(matches_concept=verdict, confidence=0.9,
        terminal_eligible=True, candidate_eligible=candidate, reason="Judge the original mug identity.")])
    concept = ConceptQuery("goal", "mug", aliases=["container"], terminal=True)
    obj = NS(uid="object", tag="container", position=[0, 0, 0])
    matcher = RuntimeConceptMatcher()
    args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
    result = getattr(matcher, method)(raw_instruction="find a mug", concept=concept, **args)
    row = result[0] if method == "match_many" else result

    assert len(calls) == 1 and matcher.stats_dict()["exact_matches"] == 0
    assert row.source == ("llm" if method == "match_object" else "llm_batch")
    assert row.matches_concept is verdict
    assert row.terminal_eligible is (verdict is True)
    assert row.candidate_eligible is (verdict is True or (verdict is None and candidate))
    payload = json.loads(calls[0]["messages"][1]["content"][0]["text"])
    assert payload["instruction"] == "find a mug"
    assert payload["concept"]["name"] == "mug" and payload["concept"]["aliases"] == ["container"]
    assert obj.tag == "container"


@pytest.mark.parametrize("method", ["match_object", "match_many"])
@pytest.mark.parametrize("verdict", [None, False, True])
@pytest.mark.parametrize("confidence", [0.01, 0.99])
def test_explicit_identity_verdict_not_confidence_controls_candidate_authority(monkeypatch, method, verdict, confidence):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    judgment = dict(matches_concept=verdict, confidence=confidence, terminal_eligible=True,
                    candidate_eligible=True, reason="Scripted semantic evidence.")
    calls = _scripted_concept_client(monkeypatch, [judgment])
    matcher = RuntimeConceptMatcher()
    obj = NS(uid="object", tag="vessel", position=[0, 0, 0])
    args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
    result = getattr(matcher, method)(raw_instruction="find mug", concept=ConceptQuery(
        "goal", "mug", terminal=True), **args)
    row = result[0] if method == "match_many" else result
    assert row.matches_concept is verdict
    assert row.candidate_eligible is (verdict is not False)
    assert row.terminal_eligible is (verdict is True)
    assert row.confidence == confidence and len(calls) == 1


@pytest.mark.parametrize("method", ["match_object", "match_many"])
def test_unknown_without_compatibility_evidence_is_not_a_candidate(monkeypatch, method):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    _scripted_concept_client(monkeypatch, [dict(matches_concept=None, confidence=0.99,
        terminal_eligible=False, candidate_eligible=False, reason="No compatibility evidence.")])
    obj = NS(uid="object", tag="unidentified", position=[0, 0, 0])
    args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
    result = getattr(RuntimeConceptMatcher(), method)(raw_instruction="find mug",
        concept=ConceptQuery("goal", "mug", detector_terms=["container"]), **args)
    row = result[0] if method == "match_many" else result
    assert row.matches_concept is None and not row.candidate_eligible


@pytest.mark.parametrize("method", ["match_object", "match_many"])
def test_unknown_can_be_reconsidered_without_a_cached_false_identity(monkeypatch, method):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    calls = _scripted_concept_client(monkeypatch, [
        dict(matches_concept=None, confidence=0.99, terminal_eligible=False,
             candidate_eligible=True, reason="Compatible but identity unresolved."),
        dict(matches_concept=True, confidence=0.01, terminal_eligible=True,
             candidate_eligible=True, reason="Identity established.")])
    matcher = RuntimeConceptMatcher()
    obj = NS(uid="object", tag="vessel", position=[0, 0, 0])
    args = {"obj": obj} if method == "match_object" else {"objects": [obj]}
    def match():
        result = getattr(matcher, method)(raw_instruction="find mug", concept=ConceptQuery(
            "goal", "mug", terminal=True), **args)
        return result[0] if method == "match_many" else result
    assert match().matches_concept is None
    assert not matcher.label_cache
    assert match().matches_concept is True
    assert match().matches_concept is True
    assert len(calls) == 2
    if method == "match_many":
        obj.uid = "another-instance"
        assert match().matches_concept is True
        assert len(calls) == 2


def test_batch_budget_keeps_bound_unknown_candidates_but_not_unrelated_labels(monkeypatch):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    monkeypatch.setenv("LLM_OFFLINE", "1")
    monkeypatch.setenv("STRIVE_CONCEPT_MATCH_MAX_BATCH", "1")
    concept = ConceptQuery("goal", "mug", detector_terms=["container"], terminal=True)
    objects = [NS(uid=label, tag=label, confidence=confidence, position=[0, 0, 0])
               for label, confidence in (("wall", 0.99), ("container", 0.01))]
    rows = {r.object_uid: r for r in RuntimeConceptMatcher().match_many(
        raw_instruction="find mug", concept=concept, objects=objects)}
    assert rows["container"].source == "prefiltered"
    assert rows["container"].matches_concept is None and rows["container"].candidate_eligible
    assert not rows["container"].terminal_eligible
    assert not rows["wall"].candidate_eligible


def test_detector_binding_changes_expire_unknown_candidate_eligibility(monkeypatch):
    from instruction_adapter.concept_matcher import RuntimeConceptMatcher
    from instruction_adapter.contracts import ConceptQuery
    monkeypatch.setenv("LLM_OFFLINE", "1")
    matcher = RuntimeConceptMatcher()
    concept = ConceptQuery("goal", "mug", detector_terms=["container"], terminal=True)
    obj = NS(uid="object", tag="container", position=[0, 0, 0])
    args = dict(raw_instruction="find mug", concept=concept, objects=[obj])
    assert matcher.match_many(**args)[0].candidate_eligible
    concept.detector_terms = []
    row = matcher.match_many(**args)[0]
    assert row.matches_concept is None and not row.candidate_eligible
