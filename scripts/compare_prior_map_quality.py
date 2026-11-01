#!/usr/bin/env python3
"""Compare audited maps built from one frozen capture, with explicit denominators."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.bundle import load_bundle
from prior_map.observations import write_json


def summarize(root, audit_path):
    """Summarize majority-supported GT identities and retain unassessed counts.

    A mixed UID has observations voting for several annotation instances.
    A single-GT UID passes this vote consistency check, not pixel-purity or
    category correctness. Observation denominators can differ across builders.
    """
    bundle = load_bundle(root)
    receipt = json.loads((root / 'build_receipt.json').read_text())
    audit = json.loads(audit_path.read_text())
    if not audit['alignment_valid']:
        raise ValueError('Semantic audit failed depth alignment')
    core_hash = hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest()
    if core_hash != audit['provenance']['predicted_map']['sha256']:
        raise ValueError('Audit belongs to a different predicted map')
    if {o['observation_id'] for o in bundle['observations']} != {
            o['observation_id'] for o in audit['observations']}:
        raise ValueError('Audit must cover the complete bundle observation set')
    votes = audit['summary']['instance_votes']
    support = Counter(gt for instance in votes.values() for gt in instance)
    clean = {next(iter(v)) for v in votes.values() if len(v) == 1}
    rows = {'bundle': str(root), 'capture_frames': receipt['frame_count'],
        'evidence_frames': audit['frame_count'], 'predicted_instances': len(bundle['prior_map'].objects),
        'observation_masks': audit['mask_count'], 'strong_masks': audit['summary']['strong_mask_count'],
        'supported_predicted_instances': len(votes),
        'unsupported_predicted_instances': len(bundle['prior_map'].objects) - len(votes),
        'mixed_predicted_instances': sum(len(v) > 1 for v in votes.values()),
        'distinct_supported_gt_instances': len(support),
        'gt_instances_in_single_gt_uids': len(clean),
        'gt_instances_with_multiple_uids': sum(n > 1 for n in support.values()),
        'build_seconds': receipt['elapsed_s'],
        'bundle_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'prior_map_sha256': core_hash, 'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    return rows, receipt['capture_manifest'], audit['provenance']


def main():
    """Write comparison only after checking common input and annotation assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('first_bundle', 'first_audit', 'second_bundle', 'second_audit', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    a, ac, ap = summarize(args.first_bundle, args.first_audit)
    b, bc, bp = summarize(args.second_bundle, args.second_audit)
    if (ac != bc or a['capture_frames'] != b['capture_frames']
            or a['capture_frames'] != ac['frame_count']):
        raise ValueError('Full input capture and processed frame count must match')
    for field in ('semantic_mesh', 'semantic_labels'):
        if ap[field]['sha256'] != bp[field]['sha256']:
            raise ValueError(f'Annotation asset differs: {field}')
    write_json(args.output, {'scope': 'same_capture_system_comparison_not_fusion_only_ablation',
        'limitations': ['Detector and stability filters differ; accepted-mask denominators differ.',
                        'Supported GT count is not recall over all scene objects.',
                        'Single-GT vote consistency does not establish mask purity.'],
        'maps': [a, b]})
    print(json.dumps([a, b], indent=2))


if __name__ == '__main__':
    main()
