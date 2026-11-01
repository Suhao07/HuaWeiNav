"""Run in the maintained mapping environment; constraints remain experimental."""
import numpy as np
from scripts.diagnose_prior_associations import coview_bridge_candidates


def test_bridge_keeps_both_disjoint_candidate_identities():
    left = np.array([[True, False]])
    right = ~left
    scores = [{'uid': 'left', 'overlap': .5}, {'uid': 'right', 'overlap': .4}]
    assert coview_bridge_candidates(scores, {'left': [left], 'right': [right]}, .05) == ['left', 'right']
    scores[1]['overlap'] = .2
    assert coview_bridge_candidates(scores, {'left': [left], 'right': [right]}, .05) == []


def test_shared_or_absent_coview_evidence_does_not_establish_conflict():
    mask = np.array([[True, False]])
    scores = [{'uid': 'a', 'overlap': .5}, {'uid': 'b', 'overlap': .4}]
    assert coview_bridge_candidates(scores, {'a': [mask], 'b': [mask]}, .05) == []
    assert coview_bridge_candidates(scores, {'a': [mask]}, .05) == []
