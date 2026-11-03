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
