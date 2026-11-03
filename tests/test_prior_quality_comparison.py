"""Quality reports distinguish supported identities from mixed observations."""
import hashlib
import json
import pytest
from prior_map.bundle import load_bundle
from scripts.compare_prior_map_quality import summarize
from test_prior_compaction import _source


def _case(tmp_path):
    root = tmp_path / 'bundle'
    _source(root)
    observations = load_bundle(root)['observations']
    (root / 'build_receipt.json').write_text(json.dumps({
        'frame_count': 4, 'elapsed_s': 1., 'capture_manifest': {'frame_count': 4}}))
    audit = {'alignment_valid': True, 'frame_count': 4, 'mask_count': len(observations),
        'observations': [{'observation_id': o['observation_id']} for o in observations],
        'summary': {'strong_mask_count': 5, 'instance_votes': {'instance': {'1': 4, '2': 1}}},
        'provenance': {'predicted_map': {'sha256': hashlib.sha256(
            (root / 'prior_map.json').read_bytes()).hexdigest()}}}
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(audit))
    return root, path, audit


def test_mixed_identity_increases_supported_gt_but_not_clean_support(tmp_path):
    root, path, _ = _case(tmp_path)
    report, _, _ = summarize(root, path)
    assert report['predicted_instances'] == 1
    assert report['distinct_supported_gt_instances'] == 2
    assert report['mixed_predicted_instances'] == 1
    assert report['gt_instances_in_single_gt_uids'] == 0


@pytest.mark.parametrize('failure', ['alignment', 'different_map', 'partial_observations'])
def test_incomparable_audit_raises(tmp_path, failure):
    root, path, audit = _case(tmp_path)
    if failure == 'alignment':
        audit['alignment_valid'] = False
    elif failure == 'different_map':
        audit['provenance']['predicted_map']['sha256'] = 'different'
    else:
        audit['observations'].pop()
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        summarize(root, path)
