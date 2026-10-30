"""Measure instance-evidence consistency without treating predictions as truth."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from .bundle import load_bundle


def audit_bundle(directory):
    """Report label disagreement and co-visible disjoint masks per fused instance.

    Co-visible mask IoU = |A intersection B| / |A union B|. A value below 0.05
    identifies a review candidate, not an association error: separate visible
    parts can belong to one object. Label disagreement likewise requires visual
    review because the detector may describe a whole object and its parts.
    """
    root = Path(directory)
    bundle = load_bundle(root)
    by_object = defaultdict(list)
    for observation in bundle["observations"]:
        for uid in observation["object_uids"]:
            by_object[uid].append(observation)
    records = []
    for obj in bundle["prior_map"].objects:
        observations = by_object[obj.uid]
        labels = Counter(o["detector_label"] for o in observations)
        by_frame = defaultdict(list)
        for obs in observations:
            by_frame[obs["frame_id"]].append(obs)
        disjoint = []
        for frame_id, group in by_frame.items():
            if len(group) < 2:
                continue
            masks = {o["observation_id"]: np.asarray(Image.open(root / o["mask_ref"])) > 0 for o in group}
            for first, second in combinations(group, 2):
                a, b = masks[first["observation_id"]], masks[second["observation_id"]]
                iou = float((a & b).sum() / (a | b).sum())
                if iou < .05:
                    disjoint.append({"frame_id": frame_id, "iou": iou,
                        "observation_ids": [first["observation_id"], second["observation_id"]]})
        records.append({"object_uid": obj.uid, "label": obj.label, "evidence_count": len(observations),
            "frame_count": len(by_frame), "observed_labels": dict(labels), "disjoint_same_frame_pairs": disjoint})
    return {"scope": "internal_evidence_consistency_not_semantic_accuracy",
        "object_count": len(records), "evidence_count": len(bundle["observations"]),
        "objects_with_multiple_frames": sum(r["frame_count"] > 1 for r in records),
        "objects_with_multiple_labels": sum(len(r["observed_labels"]) > 1 for r in records),
        "objects_with_disjoint_same_frame_masks": sum(bool(r["disjoint_same_frame_pairs"]) for r in records),
        "objects": records}
