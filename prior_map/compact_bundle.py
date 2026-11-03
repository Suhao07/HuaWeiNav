"""Select portable runtime evidence while retaining explicit uncertainty witnesses."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .bundle import load_bundle, write_bundle
from .evidence_audit import audit_bundle
from .observations import write_json


def compact_bundle(source, output, *, views_per_object=3):
    """Keep label/disjoint-mask witnesses, then add diverse camera positions.

    Args:
        source: Complete portable predicted-map bundle, kept unchanged.
        output: New self-contained bundle directory.
        views_per_object: Desired evidence count; mandatory uncertainty witnesses
            take priority and may exceed this budget, as reported in the receipt.

    Returns:
        Selection receipt with IDs, source hashes and actual evidence counts.

    Camera diversity uses max_c min_s (1 - dot(d_c, d_s)), with d the unit
    direction from object centre to camera. Scores break ties, not semantics.
    Selecting evidence changes neither object identity nor its predicted label.
    """
    if views_per_object < 1:
        raise ValueError("Use a positive per-object evidence budget")
    source, output = Path(source), Path(output)
    bundle = load_bundle(source)
    audit = audit_bundle(source)
    audits = {item["object_uid"]: item for item in audit["objects"]}
    by_object = defaultdict(list)
    by_id = {obs["observation_id"]: obs for obs in bundle["observations"]}
    for obs in bundle["observations"]:
        for uid in obs["object_uids"]:
            by_object[uid].append(obs)
    selected, decisions = set(), []
    for obj in bundle["prior_map"].objects:
        observations = by_object[obj.uid]
        if not observations:
            decisions.append({"object_uid": obj.uid, "source_count": 0, "selected_ids": []})
            continue
        by_label = defaultdict(list)
        for obs in observations:
            by_label[obs["detector_label"]].append(obs)
        # 每个曾出现的标签保留一个样本，避免把少数反例静默丢弃。
        chosen = {max(items, key=lambda row: (row["detector_score"], row["observation_id"]))["observation_id"]
                  for items in by_label.values()}
        pairs = audits[obj.uid]["disjoint_same_frame_pairs"]
        if pairs:
            witness = min(pairs, key=lambda pair: (pair["iou"], pair["observation_ids"]))
            chosen.update(witness["observation_ids"])
        mandatory_count = len(chosen)
        directions = {}
        for obs in observations:
            vector = np.asarray(obs["camera_to_map"])[:3, 3] - np.asarray(obj.position_xyz)
            directions[obs["observation_id"]] = vector / np.linalg.norm(vector)
        remaining = {obs["observation_id"] for obs in observations} - chosen
        while remaining and len(chosen) < views_per_object:
            def rank(uid):
                novelty = min(1. - float(directions[uid] @ directions[old]) for old in chosen)
                return novelty, by_id[uid]["detector_score"], uid
            uid = max(remaining, key=rank)
            chosen.add(uid)
            remaining.remove(uid)
        selected.update(chosen)
        decisions.append({"object_uid": obj.uid, "source_count": len(observations),
            "source_label_counts": audits[obj.uid]["observed_labels"],
            "source_disjoint_pair_count": len(pairs), "mandatory_witness_count": mandatory_count,
            "selected_ids": sorted(chosen), "budget_exceeded_for_witnesses": mandatory_count > views_per_object})
    observations = [obs for obs in bundle["observations"] if obs["observation_id"] in selected]
    viewpoints = []
    for view in bundle["viewpoints"]:
        refs = [uid for uid in view["observation_ids"] if uid in selected]
        if refs:
            viewpoints.append({**view, "observation_ids": refs})
    output.mkdir(parents=True, exist_ok=False)
    evidence_paths = {obs[key] for obs in observations
                      for key in ("image_ref", "depth_ref", "mask_ref", "confidence_ref") if obs.get(key) is not None}
    copied_hashes = {}
    for relative in sorted(evidence_paths):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        copied_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    # 几何保持原值；证据集合变更形成新 revision。既有 navigation_viewpoint
    # 的 checked_map_revision 保留原值，由执行侧重新检查可达性。
    selection_hash = hashlib.sha256(json.dumps(sorted(selected)).encode()).hexdigest()
    revision = f"{bundle['manifest']['map_revision']}-evidence-{selection_hash[:12]}"
    write_bundle(output, map_id=bundle["manifest"]["map_id"], revision=revision,
        prior_map=bundle["prior_map"], observations=observations, viewpoints=viewpoints)
    receipt = {"scope": "evidence_selection_not_map_quality_repair", "status": "complete",
        "views_per_object": views_per_object, "source_map_revision": bundle["manifest"]["map_revision"],
        "map_revision": revision, "source_observation_count": len(bundle["observations"]),
        "observation_count": len(observations), "object_count": len(bundle["prior_map"].objects),
        "source_evidence_bytes": sum((source / path).stat().st_size for path in {
            obs[key] for obs in bundle["observations"]
            for key in ("image_ref", "depth_ref", "mask_ref", "confidence_ref") if obs.get(key) is not None}),
        "selected_evidence_bytes": sum((output / path).stat().st_size for path in evidence_paths),
        "source_index_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "prior_map.json", "observations.jsonl", "viewpoints.json")},
        "selected_file_sha256": copied_hashes, "objects": decisions}
    build_path = source / "build_receipt.json"
    if build_path.is_file():
        shutil.copyfile(build_path, output / "source_build_receipt.json")
        original = json.loads(build_path.read_text())
        write_json(output / "build_receipt.json", {**original,
            "source_build_receipt_sha256": hashlib.sha256(build_path.read_bytes()).hexdigest(),
            "observation_count": len(observations), "evidence_count": len(observations),
            "compaction_receipt": "compaction_receipt.json"})
    write_json(output / "compaction_receipt.json", receipt)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["evidence_selection"] = "compaction_receipt.json"
    write_json(output / "manifest.json", manifest)
    load_bundle(output)
    return receipt
