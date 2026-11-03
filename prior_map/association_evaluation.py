"""Offline association metrics from frozen trace and independent semantic audit."""
from collections import Counter, defaultdict
from math import comb


def evaluate_association(trace, audit):
    """Evaluate pairs of majority-supported observations, never runtime choices.

    TP counts pairs sharing both predicted UID and GT instance; FP shares only
    predicted UID; FN shares only GT. Precision=TP/(TP+FP), recall=TP/(TP+FN).
    Masks with <=50% support are reported separately. These metrics measure
    association of observations, not mask purity, class accuracy or navigation.
    Resolution quality uses the final peer cluster: a mixed peer cluster is
    unknown, a pure different-GT peer cluster is incorrect. It does not establish
    the causal correctness at the historical resolution instant.
    Deferred observations (assigned_uid=None) each form a separate evaluation
    singleton: they retain same-GT missed pairs in FN and cannot share an ID.
    """
    if not audit['alignment_valid']:
        raise ValueError('Association evaluation requires aligned semantic evidence')
    evidence = {row['observation_id']: row for row in audit['observations']}
    predicted, truth, joint = Counter(), Counter(), Counter()
    clusters = defaultdict(set)
    gt_clusters = defaultdict(set)
    ignored = 0
    ids = [row['observation_id'] for row in trace['observations']]
    if len(ids) != len(set(ids)):
        raise ValueError('Trace observation IDs must be unique')
    for row in trace['observations']:
        observation = evidence[row['observation_id']]
        if observation['dominant_fraction_full_mask'] <= .5:
            ignored += 1
            continue
        uid = (('deferred', row['observation_id']) if row['assigned_uid'] is None
               else ('assigned', row['assigned_uid']))
        gt = observation['dominant_gt_id']
        predicted[uid] += 1
        truth[gt] += 1
        joint[(uid, gt)] += 1
        clusters[uid].add(gt)
        gt_clusters[gt].add(uid)
    tp = sum(comb(n, 2) for n in joint.values())
    fp = sum(comb(n, 2) for n in predicted.values()) - tp
    fn = sum(comb(n, 2) for n in truth.values()) - tp
    ever_deferred = {e['observation_id'] for e in trace.get('events', []) if e['assigned_uid'] is None}
    assignments = {r['observation_id']: r['assigned_uid'] for r in trace['observations']}
    resolved = {oid for oid in ever_deferred if assignments[oid] is not None}
    resolution_counts = Counter()
    for oid in resolved:
        item = evidence[oid]
        peers = [evidence[other]['dominant_gt_id'] for other, uid in assignments.items()
                 if other != oid and uid == assignments[oid]
                 and evidence[other]['dominant_fraction_full_mask'] > .5]
        if item['dominant_fraction_full_mask'] <= .5 or not peers or len(set(peers)) > 1:
            resolution_counts['unknown'] += 1
        elif all(gt == item['dominant_gt_id'] for gt in peers):
            resolution_counts['correct'] += 1
        else:
            resolution_counts['incorrect'] += 1
    assigned_gt = sorted({evidence[oid]['dominant_gt_id'] for oid, uid in assignments.items()
                          if uid is not None and evidence[oid]['dominant_fraction_full_mask'] > .5})
    return {'ever_deferred_count': len(ever_deferred), 'resolved_count': len(resolved),
        'resolution_rate': len(resolved) / len(ever_deferred) if ever_deferred else None,
        'resolution_correct': resolution_counts['correct'],
        'resolution_incorrect': resolution_counts['incorrect'],
        'resolution_unknown': resolution_counts['unknown'],
        'assigned_supported_gt_ids': assigned_gt,
        'observation_count': len(ids), 'supported_observation_count': sum(predicted.values()),
        'unsupported_observation_count': ignored,
        'deferred_observation_count': sum(row['assigned_uid'] is None for row in trace['observations']),
        'supported_predicted_instances': sum(uid[0] == 'assigned' for uid in predicted),
        'supported_gt_instances': len(truth),
        'mixed_predicted_instances': sum(len(v) > 1 for v in clusters.values()),
        'split_gt_instances': sum(len(v) > 1 for v in gt_clusters.values()),
        'pair_tp': tp, 'pair_fp': fp, 'pair_fn': fn,
        'pair_precision': tp / (tp + fp) if tp + fp else None,
        'pair_recall': tp / (tp + fn) if tp + fn else None,
        'pair_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
