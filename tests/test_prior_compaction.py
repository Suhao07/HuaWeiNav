"""Evidence selection keeps identity, uncertainty witnesses and portable files."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prior_map.bundle import load_bundle, write_bundle
from prior_map.compact_bundle import compact_bundle
from prior_map.contracts import PriorMapData, PriorObject
from prior_map.evidence_audit import audit_bundle


def _source(root):
    root.mkdir()
    (root / "evidence").mkdir()
    observations, views = [], []
    poses = [(0, 0, 1), (2, 0, 1), (-2, 0, 1), (0, 2, 1)]
    for frame, position in enumerate(poses):
        pose = np.eye(4)
        pose[:3, 3] = position
        Image.fromarray(np.full((4, 4, 3), frame * 20, dtype=np.uint8)).save(root / f"evidence/{frame}.png")
        np.save(root / f"evidence/{frame}.npy", np.ones((4, 4), dtype=np.float32))
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, :2] = 255
        Image.fromarray(mask).save(root / f"evidence/m{frame}.png")
        observations.append({"observation_id": f"o{frame}", "frame_id": frame,
            "object_uids": ["instance"], "detector_label": "chair", "detector_score": .9 - frame * .1,
            "image_ref": f"evidence/{frame}.png", "depth_ref": f"evidence/{frame}.npy",
            "mask_ref": f"evidence/m{frame}.png", "camera_to_map": pose.tolist(),
            "intrinsics": np.eye(3).tolist()})
        views.append({"uid": f"v{frame}", "kind": "observation_pose",
                      "pose_in_map": pose.tolist(), "observation_ids": [f"o{frame}"]})
    # 同帧另一标签、分离 mask：压缩时必须保留这种已有不确定性。
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, 2:] = 255
    Image.fromarray(mask).save(root / "evidence/conflict.png")
    observations.append({**observations[0], "observation_id": "conflict",
                         "detector_label": "table", "detector_score": .3,
                         "mask_ref": "evidence/conflict.png"})
    views[0]["observation_ids"].append("conflict")
    prior = PriorMapData(scene_id="test", frame_id="map_z_up", objects=(
        PriorObject(uid="instance", label="chair", position_xyz=(0, 0, 0)),))
    write_bundle(root, map_id="source-map", revision="1", prior_map=prior,
                 observations=observations, viewpoints=views)


def test_compaction_preserves_witnesses_even_above_budget(tmp_path):
    source, output = tmp_path / "source", tmp_path / "selected"
    _source(source)
    original = load_bundle(source)
    before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    receipt = compact_bundle(source, output, views_per_object=1)
    selected = load_bundle(output)
    assert selected["prior_map"].to_dict() == original["prior_map"].to_dict()
    assert selected["manifest"]["map_id"] == "source-map"
    assert selected["manifest"]["map_revision"] != "1"
    assert {obs["observation_id"] for obs in selected["observations"]} == {"o0", "conflict"}
    assert receipt["objects"][0]["budget_exceeded_for_witnesses"]
    assert audit_bundle(output)["objects_with_disjoint_same_frame_masks"] == 1
    assert audit_bundle(output)["objects_with_multiple_labels"] == 1
    assert before == {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob("*") if p.is_file()}
    for ref, digest in receipt["selected_file_sha256"].items():
        assert hashlib.sha256((source / ref).read_bytes()).hexdigest() == digest
    moved = tmp_path / "relocated"
    output.rename(moved)
    assert len(load_bundle(moved)["observations"]) == 2
    assert receipt["selected_evidence_bytes"] < receipt["source_evidence_bytes"]


def test_compaction_adds_different_camera_position(tmp_path):
    source, output = tmp_path / "source", tmp_path / "selected"
    _source(source)
    compact_bundle(source, output, views_per_object=3)
    selected = load_bundle(output)
    assert len(selected["observations"]) == 3
    assert len({obs["frame_id"] for obs in selected["observations"]}) == 2
