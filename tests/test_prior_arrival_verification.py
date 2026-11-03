"""Local contract tests; injected judgments are not model evaluation results."""
import hashlib
import json
from types import SimpleNamespace
import pytest
from instruction.contracts import TaskSpec, TargetSpec
from prior_map.bundle import load_bundle, write_bundle
from prior_map.arrival_verification import verify_arrival
from verification.contracts import VerificationResult
from test_prior_compaction import _source


def _inputs(tmp_path):
    root, motion = tmp_path/'current', tmp_path/'motion'
    _source(root)
    bundle = load_bundle(root)
    obs = {**bundle['observations'][0], 'bbox_xyxy': [0, 0, 2, 4]}
    write_bundle(root, map_id='current', revision='1', prior_map=bundle['prior_map'],
                 observations=[obs], viewpoints=[])
    motion.mkdir()
    (motion/'receipt.json').write_text('{"status":"complete","motion_status":"reached"}')
    (root/'build_receipt.json').write_text(json.dumps({'frame_count':1,'max_depth_m':6,
        'capture_manifest':{'source_motion_receipt_sha256':
            hashlib.sha256((motion/'receipt.json').read_bytes()).hexdigest()}}))
    return root, motion, TaskSpec('Find a chair.', (TargetSpec('target', 'chair'),), revision=3)


@pytest.mark.parametrize('accept', [True, False])
def test_arrival_reducer_uses_scoped_verifier_result(tmp_path, accept):
    root, motion, task = _inputs(tmp_path)
    def verify(raw, plan, candidate, evidence):
        assert evidence['observation_identity']['candidate_uid'] == 'instance'
        return VerificationResult(accept, 'accept' if accept else 'need_better_view',
            semantic_satisfied=True, view_sufficient_for_stop=accept,
            task_revision=3, target_uid=candidate.uid)
    report = verify_arrival(root, task, motion, tmp_path/'result', required_stop_distance=1.,
                            candidate_uid='instance', verifier=SimpleNamespace(verify=verify))
    assert report['progress']['completed'] is accept
    assert report['task_stop_sent'] is False


def test_prepare_only_keeps_progress_and_sends_no_request(tmp_path):
    root, motion, task = _inputs(tmp_path)
    def forbidden(*args):
        raise AssertionError('Preparation must not request verification')
    report = verify_arrival(root, task, motion, tmp_path/'prepared', required_stop_distance=1.,
        prepare_only=True, verifier=SimpleNamespace(verify=forbidden))
    assert report['verification_performed'] is False
    assert report['progress']['completed'] is False


def test_arrival_distance_uses_same_sensor_clip_policy_as_mapping(tmp_path):
    import numpy as np
    root, motion, task = _inputs(tmp_path)
    build = json.loads((root/'build_receipt.json').read_text())
    build['capture_manifest']['capture_profile'] = {'depth_min_m': .5, 'depth_max_m': 5.}
    (root/'build_receipt.json').write_text(json.dumps(build))
    depth = np.full((4,4),5.,dtype=np.float32)
    depth[0,1] = .5
    depth[1,1] = 2.
    np.save(root/'evidence/0.npy',depth)
    report=verify_arrival(root,task,motion,tmp_path/'prepared',required_stop_distance=1.,prepare_only=True)
    geometry=report['records'][0]['evidence']['geometry']
    assert geometry['visible_projected_points'] == 1
    assert geometry['distance_to_object'] == pytest.approx(2*np.sqrt(2))


def test_arrival_all_clipped_depth_reports_missing_geometry(tmp_path):
    import numpy as np
    root, motion, task = _inputs(tmp_path)
    build = json.loads((root/'build_receipt.json').read_text())
    build['capture_manifest']['capture_profile'] = {'depth_min_m': .5, 'depth_max_m': 5.}
    (root/'build_receipt.json').write_text(json.dumps(build))
    np.save(root/'evidence/0.npy',np.full((4,4),.5,dtype=np.float32))
    with pytest.raises(ValueError,match='no valid depth support'):
        verify_arrival(root,task,motion,tmp_path/'prepared',required_stop_distance=1.,prepare_only=True)


def test_explicit_current_observation_keeps_mask_bbox_centroid_together(tmp_path):
    root,motion,task=_inputs(tmp_path)
    bundle=load_bundle(root)
    first=bundle['observations'][0]
    other=dict(first,observation_id='other',detector_label='table',bbox_xyxy=[2,0,4,4])
    write_bundle(root,map_id='current',revision='1',prior_map=bundle['prior_map'],
                 observations=[first,other],viewpoints=[])
    report=verify_arrival(root,task,motion,tmp_path/'prepared',required_stop_distance=1.,
        candidate_uid='instance',observation_id=first['observation_id'],prepare_only=True)
    assert report['candidate_count']==1
    row=report['records'][0]
    assert row['candidate']['bbox_2d']==first['bbox_xyxy']
    assert row['candidate']['centroid']==pytest.approx([.5,1.5,2.])
    assert row['evidence']['observation_identity']['observation_id']==first['observation_id']
    assert row['evidence']['observation_identity']['mask_sha256']==hashlib.sha256((root/first['mask_ref']).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='no observations'):
        verify_arrival(root,task,motion,tmp_path/'bad',required_stop_distance=1.,
            candidate_uid='missing',observation_id=first['observation_id'],prepare_only=True)
