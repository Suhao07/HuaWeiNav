"""Tests for calibrated real-robot LiDAR-to-camera projection profiles."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SEMANTIC_MAPPING_PARENT = (
    Path(__file__).resolve().parents[1]
    / "real_robot"
    / "ros2_ws"
    / "src"
    / "semantic_mapping"
)
if str(SEMANTIC_MAPPING_PARENT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_MAPPING_PARENT))

from semantic_mapping.projection_config import CalibrationError, CameraProjectionConfig, load_projection_config


def _pinhole_mapping(*, status: str = "calibrated") -> dict:
    return {
        "calibration_status": status,
        "model": "pinhole",
        "image": {"width": 640, "height": 480},
        "intrinsics": {"fx": 200.0, "fy": 200.0, "cx": 320.0, "cy": 240.0},
        "lidar_to_camera": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_rpy_rad": [0.0, 0.0, 0.0],
        },
    }


def test_pinhole_projection_uses_calibrated_intrinsics() -> None:
    """A forward point projects to the calibrated optical centre."""

    config = CameraProjectionConfig.from_mapping(_pinhole_mapping())
    pixels = config.project(np.asarray([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0], [0.0, 0.0, -1.0]]))

    np.testing.assert_allclose(pixels[0], [320.0, 240.0, 2.0])
    np.testing.assert_allclose(pixels[1], [420.0, 190.0, 2.0])
    np.testing.assert_allclose(pixels[2], [-1.0, -1.0, -1.0])


def test_uncalibrated_profile_is_rejected_when_required(tmp_path: Path) -> None:
    """Semantic fusion cannot start from an unapproved deployment asset."""

    projection_path = tmp_path / "projection.yaml"
    projection_path.write_text(
        "camera_projection:\n"
        "  calibration_status: uncalibrated\n"
        "  model: pinhole\n"
        "  image: {width: 640, height: 480}\n"
        "  intrinsics: {fx: 200.0, fy: 200.0, cx: 320.0, cy: 240.0}\n"
        "  lidar_to_camera:\n"
        "    translation_m: [0.0, 0.0, 0.0]\n"
        "    rotation_rpy_rad: [0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="not calibrated"):
        load_projection_config(projection_path, require_calibrated=True)


def test_pinhole_requires_positive_focal_length() -> None:
    """Invalid intrinsics are rejected before they can enter the fusion loop."""

    data = _pinhole_mapping()
    data["intrinsics"]["fx"] = 0.0
    with pytest.raises(CalibrationError, match="focal lengths"):
        CameraProjectionConfig.from_mapping(data)
