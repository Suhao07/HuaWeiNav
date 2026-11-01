#!/usr/bin/env python3
"""Score appearance on frozen geometric candidates; use GT only for evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import roc_auc_score
from prior_map.observations import write_json


def main():
    """Replay feature aggregation under original assignments without re-fusion.

    f_i <- normalize((n_i*f_i + f_obs)/(n_i+1)), as in DualMap's local update.
    Candidate identities and geometry stay fixed. Candidate GT sets classify
    evaluation rows after scoring and never affect feature aggregation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'trace', 'audit', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.features / 'receipt.json').read_text())
    feature_path = args.features / 'features.npz'
    if hashlib.sha256(feature_path.read_bytes()).hexdigest() != receipt['features_sha256']:
        raise ValueError('Feature bytes differ from extraction receipt')
    features = np.load(feature_path)
    index = {o['observation_id']: i for i, o in enumerate(receipt['observations'])}
    trace = json.loads(args.trace.read_text())
    audit = json.loads(args.audit.read_text())
    if not audit['alignment_valid']:
        raise ValueError('Semantic audit must pass alignment')
    evidence = {o['observation_id']: o for o in audit['observations']}
    states, counts, votes = {}, defaultdict(int), defaultdict(set)
    rows = []
    for row in trace['observations']:
        oid = row['observation_id']
        vectors = {k: features[k][index[oid]] for k in ('image', 'weighted')}
        gt = evidence[oid]
        supported = gt['dominant_fraction_full_mask'] > .5
        for candidate in row['candidates']:
            uid = candidate['uid']
            scores = {k: float(v @ states[uid][k]) for k, v in vectors.items()}
            truth = (gt['dominant_gt_id'] == next(iter(votes[uid]))
                     if supported and len(votes[uid]) == 1 else None)
            rows.append({'observation_id': oid, 'candidate_uid': uid,
                'geometric_score': candidate['overlap'], 'appearance_scores': scores,
                'same_gt_instance': truth})
        uid = row['assigned_uid']
        if uid is None:
            raise ValueError('Use a completed original-assignment trace for feature diagnosis')
        if counts[uid] == 0:
            states[uid] = vectors
        else:
            for k, v in vectors.items():
                updated = counts[uid] * states[uid][k] + v
                states[uid][k] = updated / np.linalg.norm(updated)
        counts[uid] += 1
        if supported:
            votes[uid].add(gt['dominant_gt_id'])
    usable = [r for r in rows if r['same_gt_instance'] is not None]
    summary = {'candidate_pairs': len(rows), 'supported_single_gt_pairs': len(usable),
               'same_gt_pairs': sum(r['same_gt_instance'] for r in usable), 'scores': {}}
    truth = [r['same_gt_instance'] for r in usable]
    for key in ('image', 'weighted'):
        positive = [r['appearance_scores'][key] for r in usable if r['same_gt_instance']]
        negative = [r['appearance_scores'][key] for r in usable if not r['same_gt_instance']]
        summary['scores'][key] = {
            'auc': float(roc_auc_score(truth, [r['appearance_scores'][key] for r in usable]))
                   if positive and negative else None,
            'same_gt_quantiles_10_50_90': np.quantile(positive, [.1, .5, .9]).tolist() if positive else [],
            'different_gt_quantiles_10_50_90': np.quantile(negative, [.1, .5, .9]).tolist() if negative else []}
    groups = defaultdict(list)
    for row in usable:
        groups[row['observation_id']].append(row)
    positives = [group for group in groups.values() if any(r['same_gt_instance'] for r in group)]
    ranking = {}
    for name, score in {
        'geometry': lambda r: r['geometric_score'],
        'geometry_plus_weighted': lambda r: r['geometric_score'] + r['appearance_scores']['weighted'],
    }.items():
        ranking[name] = {
            'candidate_auc': float(roc_auc_score(truth, [score(r) for r in usable]))
                             if len(set(truth)) == 2 else None,
            'top1_same_gt': sum(max(group, key=score)['same_gt_instance'] for group in positives),
            'observations_with_known_positive': len(positives)}
    write_json(args.output, {'scope': 'conditional_candidate_appearance_diagnostic_not_matching_accuracy',
        'notes': ['Original trace exposes at most three geometric candidates per observation.',
                  'Mixed or unsupported prior GT sets are excluded from AUC, reported in total count.',
                  'Features aggregate under original assignments; no mapping decisions change.'],
        'summary': summary, 'frozen_candidate_ranking_diagnostic': ranking, 'pairs': rows,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.features / 'receipt.json', args.trace, args.audit)}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
