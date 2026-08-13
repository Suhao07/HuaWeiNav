import json
import math
from pathlib import Path

import pytest

from prior_map.alignment import PriorMapAlignment, PriorMapAlignmentError


def test_alignment_stays_platform_neutral() -> None:
    source = Path("prior_map/alignment.py").read_text(encoding="utf-8")

    forbidden = (
        "import rclpy",
        "import rospy",
        "import habitat",
        "import cv2",
        "import numpy",
        "from rclpy",
        "from rospy",
        "from habitat",
        "from cv2",
        "from numpy",
        "openai",
        "anthropic",
    )
    for text in forbidden:
        assert text not in source


def test_identity_alignment_inverse_and_json_roundtrip(tmp_path) -> None:
    alignment = PriorMapAlignment.identity(prior_frame_id="prior_map", runtime_frame_id="map")

    assert alignment.prior_to_runtime((1.0, 2.0)) == (1.0, 2.0, 0.0)
    assert alignment.runtime_to_prior((1.0, 2.0, 0.0)) == (1.0, 2.0)
    assert alignment.confidence() == pytest.approx(1.0)
    assert alignment.can_rank_geometry() is True

    path = tmp_path / "alignment.json"
    alignment.save(path)
    restored = PriorMapAlignment.load(path)

    saved_payload = json.loads(path.read_text(encoding="utf-8"))
    assert restored.prior_to_runtime((1.0, 2.0)) == (1.0, 2.0, 0.0)
    assert restored.runtime_to_prior((1.0, 2.0, 0.0)) == (1.0, 2.0)
    assert restored.to_dict() == saved_payload
    assert saved_payload["diagnostics"]["enabled_for_ranking"] is True


def test_affine_alignment_forward_inverse() -> None:
    alignment = PriorMapAlignment.affine_2d(
        scale=2.0,
        rotation_rad=math.pi / 2.0,
        translation_xyz=(10.0, -1.0, 0.5),
        confidence=0.8,
    )

    runtime = alignment.prior_to_runtime((1.0, 2.0), z=0.2)
    prior = alignment.runtime_to_prior(runtime)

    assert runtime[0] == pytest.approx(6.0)
    assert runtime[1] == pytest.approx(1.0)
    assert runtime[2] == pytest.approx(0.7)
    assert prior[0] == pytest.approx(1.0)
    assert prior[1] == pytest.approx(2.0)
    assert alignment.confidence() == pytest.approx(0.8)


def test_estimate_affine_from_point_pairs_records_diagnostics() -> None:
    source_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    target_points = ((10.0, -1.0), (10.0, 1.0), (8.0, -1.0), (8.0, 1.0))

    alignment = PriorMapAlignment.from_point_pairs(
        source_points_prior=source_points,
        target_points_runtime=target_points,
        prior_frame_id="floorplan",
        runtime_frame_id="map",
    )

    assert alignment.alignment_type == "affine_2d"
    assert alignment.scale == pytest.approx(2.0)
    assert alignment.rotation_rad == pytest.approx(math.pi / 2.0)
    assert alignment.translation_xyz == pytest.approx((10.0, -1.0, 0.0))
    assert alignment.prior_to_runtime((1.0, 1.0)) == pytest.approx((8.0, 1.0, 0.0))

    diagnostics = alignment.diagnostics_payload()
    assert diagnostics["prior_frame_id"] == "floorplan"
    assert diagnostics["runtime_frame_id"] == "map"
    assert diagnostics["source_points_prior"] == [list(point) for point in source_points]
    assert diagnostics["target_points_runtime"] == [list(point) for point in target_points]
    assert diagnostics["rmse"] == pytest.approx(0.0)
    assert diagnostics["max_error"] == pytest.approx(0.0)
    assert diagnostics["enabled_for_ranking"] is True


def test_unavailable_alignment_disables_geometry_ranking_and_transform() -> None:
    alignment = PriorMapAlignment.unavailable(
        reason="not enough corresponding points",
        prior_frame_id="floorplan",
        runtime_frame_id="map",
    )

    assert alignment.confidence() == 0.0
    assert alignment.can_rank_geometry() is False
    assert alignment.diagnostics_payload()["fallback"] == "prompt_context_only"

    with pytest.raises(PriorMapAlignmentError):
        alignment.prior_to_runtime((1.0, 2.0))

    with pytest.raises(PriorMapAlignmentError):
        alignment.runtime_to_prior((1.0, 2.0, 0.0))


def test_point_pair_estimation_rejects_degenerate_inputs() -> None:
    with pytest.raises(PriorMapAlignmentError):
        PriorMapAlignment.from_point_pairs(((0.0, 0.0),), ((1.0, 1.0),))

    with pytest.raises(PriorMapAlignmentError):
        PriorMapAlignment.from_point_pairs(((0.0, 0.0), (0.0, 0.0)), ((1.0, 1.0), (2.0, 2.0)))
