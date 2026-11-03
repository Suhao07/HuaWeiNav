#!/usr/bin/env python3
"""Measure sampling-only repeatability using saved masks and optical depth.

This diagnostic repeats one observation, not a collected trajectory. It omits
DBSCAN, bounding-box gates and appearance: results isolate random sampling and
cannot be reported as the full DualMap association success rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def main():
    """Save per-mask overlap under independent deterministic random subsets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('detections', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = {}
    for line in (args.capture / 'frames.jsonl').read_text().splitlines():
        frame = json.loads(line)
        mask_path = args.detections / str(frame['frame_id']) / 'masks.npz'
        # An absent mask artifact represents a frame with no saved detection.
        if not mask_path.exists():
            continue
        sources[str(mask_path)] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        depth = np.load(args.capture / frame['depth_path'])
        k = np.asarray(frame['intrinsics'])
        masks = np.load(mask_path)['arr_0']
        for index, mask in enumerate(masks):
            v, u = np.where(mask.astype(bool) & (depth > 0))
            z = depth[v, u]
            points = np.column_stack(((u-k[0, 2])*z/k[0, 0],
                                      (v-k[1, 2])*z/k[1, 1], z))
            if len(points) < 25:
                continue
            for ratio in (0.04, 0.2, 1.0):
                count = int(len(points) * ratio)
                rng = np.random.default_rng([17, frame['frame_id'], index])
                a, b = [points[rng.choice(len(points), count, replace=False)] for _ in range(2)]
                # s = |{p in A : min_q ||p-q|| < 0.02}| / |A|.
                # 相同视觉特征的理想条件下，s + 1 > 1.2 才能通过官方阈值。
                overlap = float(np.mean(cKDTree(b).query(a)[0] < 0.02))
                rows.append(dict(frame_id=frame['frame_id'], mask_index=index,
                    valid_points=len(points), median_depth_m=float(np.median(z)),
                    ratio=ratio, sampled_points=count, spatial_overlap=overlap))
    summary = {}
    for ratio in (0.04, 0.2, 1.0):
        values = np.asarray([row['spatial_overlap'] for row in rows if row['ratio'] == ratio])
        summary[str(ratio)] = dict(mask_count=len(values), mean=float(values.mean()),
            median=float(np.median(values)), fraction_above_point_two=float(np.mean(values > 0.2)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='same_observation_sampling_only_before_clustering',
        radius_m=0.02, seed=17, mask_sha256=sources, summary=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
