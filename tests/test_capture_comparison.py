"""Matched controls must precede interpretation of capture coverage."""
import json
import numpy as np
import pytest

from prior_map.capture_comparison import compare_captures
from prior_map.observations import write_frame, write_json


def _capture(root, *, changed_rgb=False):
    root.mkdir()
    profile = {"depth_min_m": .5, "depth_max_m": 5., "max_actions": 0}
    write_json(root / "manifest.json", {"status": "complete", "capture_profile": profile,
        "actual_path_length_m": 0., "termination": "step_budget"})
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    if changed_rgb:
        image[0, 0] = 255
    # 深度截断的近点和远点都没有真实表面距离证据。
    write_frame(root, 0, image, np.array([[.5, 1.], [2., 5.]], dtype=np.float32),
                np.eye(3), np.eye(4))


def test_same_capture_controls_and_clipped_depth_exclusion(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _capture(a)
    _capture(b)
    report = compare_captures(a, b, stride=1)
    assert report["coverage_comparison_admissible"]
    assert report["common_surface_cells"] == 2
    assert report["union_surface_cells"] == 2


def test_changed_initial_image_prevents_matched_claim(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _capture(a)
    _capture(b, changed_rgb=True)
    assert not compare_captures(a, b)["coverage_comparison_admissible"]


def test_profile_mismatch_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _capture(a)
    _capture(b)
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["capture_profile"]["depth_max_m"] = 6.
    write_json(b / "manifest.json", manifest)
    with pytest.raises(ValueError, match="profiles differ"):
        compare_captures(a, b)
