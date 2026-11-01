"""Contracts for evidence ownership, deferred resolution and local rebuilding."""
import unittest
from unittest.mock import patch
import numpy as np
import open3d as o3d
from mapping_utils.geometry import gpu_pointcloud_from_array
from mapping_utils.representation import ObjectNode
from mapping.object_fusion import merge_object_observation, score_object_candidates
from prior_map.association import PriorAssociation, AssociationConfig


def entity(oid, x=0., confidence=.7):
    points = np.array([[a+x,b,c] for a in (0.,.03) for b in (0.,.03) for c in (0.,.03)], dtype=np.float32)
    cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 100), o3d.core.Device('CPU:0'))
    obj = ObjectNode(cloud, cloud, 'chair', confidence, points.mean(0), 8, oid, [0,0,2,2])
    obj.uid = oid
    return obj


class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.data = {}
        def load(oid):
            x, mask, confidence = self.data[oid]
            return entity(oid, x, confidence), mask.copy()
        self.load = load
        self.manager = PriorAssociation(load)

    def add(self, oid, x=0., right=False, confidence=.7):
        mask = np.zeros((4,4), bool)
        mask[:, 2:] = right
        mask[:, :2] = not right
        self.data[oid] = x, mask, confidence

    def test_deferred_isolated_then_resolved_by_new_frame(self):
        self.add('a'); self.add('b', right=True, confidence=.9); self.add('c')
        self.manager.add_frame(0, ['a','b'])
        self.assertIsNone(self.manager.assignment['b'])
        self.assertEqual(self.manager.entities['a'].rgb, 'a')
        self.assertEqual(len(self.manager.entities['a'].pcd.point.positions), 8)
        count = len(self.manager.events)
        self.manager.resolve_pending()
        self.assertEqual(len(self.manager.events), count)
        self.manager.add_frame(1, ['c'])
        self.assertEqual(self.manager.assignment['b'], 'a')
        self.assertEqual(set(self.manager.members['a']), {'a','b','c'})
        self.assertEqual(len(self.manager.entities['a'].pcd.point.positions), 24)

    def test_retract_rebuilds_all_derived_fields(self):
        self.add('a'); self.add('b', .01, confidence=.9); self.add('c', .02, confidence=.8)
        for frame, oid in enumerate(['a','b','c']):
            self.manager.add_frame(frame,[oid])
        self.manager.retract('b', reason='reviewed_spatial_conflict')
        actual = self.manager.entities['a']
        expected,_ = self.load('a'); last,_ = self.load('c')
        merge_object_observation(expected,last)
        np.testing.assert_array_equal(actual.pcd.point.positions.numpy(),expected.pcd.point.positions.numpy())
        self.assertEqual(actual.num_list,expected.num_list)
        self.assertEqual(actual.confidence,expected.confidence)
        self.assertEqual(actual.rgb,expected.rgb)
        self.assertIsNone(self.manager.assignment['b'])
        self.assertEqual(self.manager.members['a'], ['a','c'])

    def test_new_candidate_resolves_pending_and_one_pass_is_bounded(self):
        self.add('a'); self.add('b',1.); self.add('p',.5); self.add('new',.5)
        self.manager.add_frame(0,['a','b'])
        # Isolate scheduler from geometry: two equally eligible old candidates.
        original = self.manager._scores
        def scores(obj, refs):
            if obj.uid == 'p' and 'new' not in refs:
                return [dict(uid=u,overlap=.4,new_to_old=.4,old_to_new=.4) for u in ('a','b')]
            return original(obj,refs)
        with patch.object(self.manager,'_scores',side_effect=scores):
            self.manager.add_frame(1,['p'])
            self.assertIsNone(self.manager.assignment['p'])
            self.manager.add_frame(2,['new'])
        self.assertEqual(self.manager.assignment['p'],'new')
        self.assertEqual(self.manager.members['new'],['new','p'])

    def test_unresolved_observations_survive_final_pass(self):
        self.add('a'); self.add('b', right=True)
        self.manager.add_frame(0,['a','b']); self.manager.resolve_pending()
        trace = self.manager.trace()
        self.assertEqual(len(trace['observations']),2)
        self.assertIn('b',trace['pending'])
        self.assertEqual(len(self.manager.entities),1)

    def test_raw_observation_is_preserved_and_scoring_is_read_only(self):
        a,b=entity('a'),entity('b',.01)
        before=a.pcd.point.positions.numpy().copy()
        score_object_candidates(b,[a])
        np.testing.assert_array_equal(before,a.pcd.point.positions.numpy())
        self.add('a'); self.add('b',.01)
        self.manager.add_frame(0,['a','b'])
        raw,_=self.load('a')
        self.assertEqual(len(raw.pcd.point.positions),8)
        self.assertEqual(len(self.manager.entities['a'].pcd.point.positions),16)

    def test_inactive_reference_retains_its_index(self):
        dead,live,obs=entity('dead',2),entity('live'),entity('obs',.01)
        dead.alive=False
        scores,_,_=score_object_candidates(obs,[dead,live])
        self.assertEqual(int(np.argmax(scores)),1)

    def test_duplicate_observation_in_frame_raises(self):
        self.add('a')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            self.manager.add_frame(0,['a','a'])
        self.assertEqual(self.manager.entities,{})

    def test_retraction_waits_for_post_correction_evidence(self):
        for oid in ['a','b','c']:
            self.add(oid)
        self.manager.add_frame(0,['a']); self.manager.add_frame(1,['b'])
        self.manager.retract('b',reason='reviewed_conflict')
        self.manager.resolve_pending()
        self.assertIsNone(self.manager.assignment['b'])
        self.manager.add_frame(2,['c'])
        self.assertEqual(self.manager.assignment['b'],'a')

    def test_retract_last_member_removes_instance(self):
        self.add('a'); self.manager.add_frame(0,['a'])
        self.manager.retract('a',reason='reviewed_conflict')
        self.manager.resolve_pending()
        self.assertEqual(self.manager.entities,{})
        self.assertIsNone(self.manager.assignment['a'])

    def test_capture_builder_exports_identity_from_coordinator(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        from prior_map.observations import write_frame, COORDINATES, SCHEMA_VERSION
        from prior_map.capture_builder import build_capture_map
        from prior_map.bundle import load_bundle
        class Detector:
            def __init__(self, **kwargs): pass
            def initialize(self, **kwargs): pass
            def segment(self, rgb, **kwargs):
                mask=torch.zeros((2,32,32),dtype=torch.bool)
                mask[:,4:20,4:20]=True
                return torch.tensor([[4,4,20,20]]*2),torch.tensor([.7,.8]),['chair']*2,mask,None,None,None
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); capture=root/'capture'; output=root/'map'
            capture.mkdir()
            write_frame(capture,0,np.zeros((32,32,3),np.uint8),np.full((32,32),2.,np.float32),
                        np.array([[30.,0,15.5],[0,30.,15.5],[0,0,1]]),np.eye(4))
            (capture/'manifest.json').write_text(json.dumps(dict(coordinates=COORDINATES,
                schema_version=SCHEMA_VERSION,status='complete',scene_id='fixture',mode='fixed',pose_source='fixture')))
            weights=root/'weights';weights.write_bytes(b'fixture')
            modules={'cv_utils.sam':SimpleNamespace(MMDINO_Grounded_SAM=Detector),
                     'config_utils':SimpleNamespace(SAM_CHECKPOINT=weights,GROUNDING_DINO_CHECKPOINT=weights,GROUNDING_DINO_CONFIG=weights)}
            with patch.dict(sys.modules,modules):
                receipt=build_capture_map(capture,output,classes=['chair'],device='cpu:0',association_config=AssociationConfig())
            bundle=load_bundle(output)
            self.assertEqual(receipt['observation_count'],2)
            self.assertEqual(len(bundle['prior_map'].objects),1)
            ids={o['object_uids'][0] for o in bundle['observations']}
            self.assertEqual(len(ids),1)
            self.assertEqual(len(json.loads((output/'association_trace.json').read_text())['observations']),2)

    def test_balanced_candidate_score_invariant_to_reference_repetition(self):
        from mapping_utils.geometry import gpu_merge_pointcloud
        observed, reference = entity('observed'), entity('reference')
        observed.pcd = gpu_merge_pointcloud(observed.pcd, entity('distant',2.).pcd)
        repeated = entity('repeated')
        for _ in range(7):
            repeated.pcd = gpu_merge_pointcloud(repeated.pcd, entity('copy').pcd)
        manager = PriorAssociation(self.load, AssociationConfig(balanced_coverage=True))
        scores = manager._scores(observed, {'single': reference, 'repeated': repeated})
        self.assertAlmostEqual(scores[0]['overlap'], scores[1]['overlap'], places=6)
        pooled, _, _ = score_object_candidates(observed, [reference, repeated])
        self.assertGreater(float(pooled[1]), float(pooled[0])+.1)

    def test_balanced_ranking_keeps_pooled_admission(self):
        manager = PriorAssociation(self.load, AssociationConfig(balanced_coverage=True))
        with patch('prior_map.association.score_object_candidates',return_value=([.2,.3],[.1,.4],[1.,.4])):
            rows=manager._scores(entity('obs'),{'excluded':entity('a'),'admitted':entity('b')})
        self.assertEqual([r['uid'] for r in rows],['admitted'])

    def test_pending_records_versions_used_for_scoring_before_frame_commit(self):
        for oid in ['a','update','pending']:
            self.add(oid)
        self.manager.add_frame(0,['a'])
        original=self.manager._choose
        def choose(oid, obj, mask, references, members):
            if oid == 'pending' and len(members['a']) == 1:
                return None,'candidate_margin',[dict(uid='a',overlap=.4)]
            return original(oid,obj,mask,references,members)
        with patch.object(self.manager,'_choose',side_effect=choose):
            self.manager.add_frame(1,['update','pending'])
        self.assertEqual(self.manager.assignment['pending'],'a')

    def test_frame_conflict_is_recorded_before_committing_to_old_instance(self):
        self.add('old'); self.add('left'); self.add('right',right=True)
        self.manager.add_frame(0,['old'])
        self.manager.add_frame(1,['left','right'])
        rows=self.manager.trace()['coview_conflicts']
        self.assertTrue(any(r['first']=='left' and r['second']=='right' and r['stage']=='frame_commit' for r in rows))
        self.assertEqual(self.manager.assignment['left'],self.manager.assignment['right'])

    def test_same_mask_duplicates_do_not_emit_conflict(self):
        self.add('old'); self.add('left'); self.add('duplicate')
        self.manager.add_frame(0,['old'])
        self.manager.add_frame(1,['left','duplicate'])
        self.assertEqual(self.manager.trace()['coview_conflicts'],[])
