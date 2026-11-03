"""Offline controls and sampled surface coverage for paired capture runs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .observations import read_records, backproject, metric_depth_validity


def compare_captures(first, second, *, stride=16, voxel_m=.1):
    """Compare shared controls before interpreting observed surface differences.

    Surface metric: S = {floor(p / voxel_m): p sampled from valid measured depth}.
    Counts measure sampled surface cells, not floor area, object recall or task
    success. Depth at the configured clipping maximum is excluded from S.
    """
    roots = [Path(first), Path(second)]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("Compare completed captures")
    profiles = [m["capture_profile"] for m in manifests]
    if profiles[0] != profiles[1]:
        raise ValueError("Capture profiles differ; fix controls before comparing coverage")
    if stride < 1 or voxel_m <= 0:
        raise ValueError("Use positive spatial sampling parameters")
    rows = [read_records(root / "frames.jsonl") for root in roots]
    initial = [frames[0] for frames in rows]
    images = [np.asarray(Image.open(root / row["rgb_path"])) for root, row in zip(roots, initial)]
    depths = [np.load(root / row["depth_path"], allow_pickle=False) for root, row in zip(roots, initial)]
    checks = {
        "initial_pose_equal": bool(np.allclose(initial[0]["camera_to_map"], initial[1]["camera_to_map"], atol=1e-6)),
        "intrinsics_equal": bool(np.allclose(initial[0]["intrinsics"], initial[1]["intrinsics"], atol=1e-6)),
        "initial_rgb_equal": bool(np.array_equal(*images)),
        "initial_depth_max_error_m": float(np.max(np.abs(depths[0] - depths[1]))),
    }
    checks["initial_observation_matched"] = (checks["initial_pose_equal"] and checks["intrinsics_equal"]
        and checks["initial_rgb_equal"] and checks["initial_depth_max_error_m"] <= 1e-4)
    result = {"scope": "matched_capture_sampled_surface_comparison", "controls": checks,
              "stride": stride, "voxel_m": voxel_m, "runs": []}
    surfaces = []
    for root, manifest, frames in zip(roots, manifests, rows):
        cells = set()
        for row in frames:
            depth = np.load(root / row["depth_path"], allow_pickle=False)
            depth = np.where(metric_depth_validity(depth, profiles[0]), depth, 0)
            points = backproject(depth, np.asarray(row["intrinsics"]), np.asarray(row["camera_to_map"]), stride=stride)
            cells.update(map(tuple, np.floor(points / voxel_m).astype(np.int64)))
        surfaces.append(cells)
        result["runs"].append({"capture": str(root), "frame_count": len(frames),
            "action_count": len(frames) - 1, "sampled_surface_cells": len(cells),
            "path_length_m": manifest["actual_path_length_m"], "termination": manifest["termination"]})
    result["common_surface_cells"] = len(surfaces[0] & surfaces[1])
    result["union_surface_cells"] = len(surfaces[0] | surfaces[1])
    result["coverage_comparison_admissible"] = checks["initial_observation_matched"]
    return result
