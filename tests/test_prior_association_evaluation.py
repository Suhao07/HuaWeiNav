"""Association metrics account for wrong merges and fragmentation together."""
import pytest
from prior_map.association_evaluation import evaluate_association


def _inputs():
    trace = {'observations': [{'observation_id': str(i), 'assigned_uid': p}
             for i, p in enumerate(['a', 'a', 'a', 'b'])]}
    audit = {'alignment_valid': True, 'observations': [
        {'observation_id': str(i), 'dominant_gt_id': gt, 'dominant_fraction_full_mask': 1.}
        for i, gt in enumerate([1, 1, 2, 1])]}
    return trace, audit


def test_merge_and_split_counts_are_independent_of_order():
    trace, audit = _inputs()
    report = evaluate_association(trace, audit)
    assert (report['pair_tp'], report['pair_fp'], report['pair_fn']) == (1, 2, 2)
    assert report['mixed_predicted_instances'] == report['split_gt_instances'] == 1
    assert report['pair_f1'] == pytest.approx(1/3)
    trace['observations'].reverse()
    assert evaluate_association(trace, audit) == report


def test_unsupported_masks_remain_in_report_denominator():
    trace, audit = _inputs()
    for row in audit['observations']:
        row['dominant_fraction_full_mask'] = .5
    report = evaluate_association(trace, audit)
    assert report['unsupported_observation_count'] == 4
    assert report['supported_observation_count'] == 0
    assert report['pair_f1'] is None


def test_missing_or_misaligned_ground_truth_raises():
    trace, audit = _inputs()
    audit['alignment_valid'] = False
    with pytest.raises(ValueError, match='aligned'):
        evaluate_association(trace, audit)
    audit['alignment_valid'] = True
    audit['observations'].pop()
    with pytest.raises(KeyError):
        evaluate_association(trace, audit)


def test_deferred_observations_keep_missed_pairs_in_denominator():
    trace, audit = _inputs()
    for row in trace['observations']:
        row['assigned_uid'] = None
    report = evaluate_association(trace, audit)
    assert report['deferred_observation_count'] == 4
    assert report['supported_predicted_instances'] == 0
    assert (report['pair_tp'], report['pair_fp'], report['pair_fn']) == (0, 0, 3)
    assert report['pair_recall'] == report['pair_f1'] == 0.


def test_resolution_reports_wrong_merges_and_retained_gt():
    trace, audit = _inputs()
    trace['events'] = [{'observation_id': '2', 'assigned_uid': None},
                       {'observation_id': '2', 'assigned_uid': 'a'}]
    report = evaluate_association(trace, audit)
    assert report['resolved_count'] == report['resolution_incorrect'] == 1
    assert report['resolution_correct'] == 0
    assert report['assigned_supported_gt_ids'] == [1, 2]
    trace['observations'][2]['assigned_uid'] = None
    report = evaluate_association(trace, audit)
    assert report['assigned_supported_gt_ids'] == [1]
    assert report['resolved_count'] == 0


def test_mixed_resolution_reference_is_unknown():
    trace, audit = _inputs()
    trace['events'] = [{'observation_id': '0', 'assigned_uid': None}]
    result = evaluate_association(trace, audit)
    assert result['resolution_unknown'] == 1
    assert result['resolution_incorrect'] == 0
