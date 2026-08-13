"""Validated LiDAR-to-camera projection profiles for real robots.

The legacy SysNav projection code contains camera dimensions and extrinsics
that belong to one specific mecanum robot.  This module keeps those values out
of the runtime and makes a calibrated projection an explicit deployment asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


class CalibrationError(ValueError):
    """Raised when a real-robot projection profile is missing or unsafe."""


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the active ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` rotation matrix."""

    sin_r, cos_r = np.sin(roll), np.cos(roll)
    sin_p, cos_p = np.sin(pitch), np.cos(pitch)
    sin_y, cos_y = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CameraProjectionConfig:
    """A calibrated transform and camera model used to project LiDAR points.

    Args:
        model: ``pinhole`` or ``equirectangular`` camera model.
        image_width: RGB image width in pixels.
        image_height: RGB image height in pixels.
        translation_m: LiDAR-origin translation in the camera frame, in metres.
        rotation_rpy_rad: LiDAR-to-camera roll, pitch, yaw in radians.
        min_depth_m: Points at or behind this camera-frame depth are rejected.
        calibration_status: Must be ``calibrated`` for safety-critical fusion.
        intrinsics: ``fx``, ``fy``, ``cx`` and ``cy`` for the pinhole model.
        horizontal_fov_deg: Horizontal field of view for equirectangular images.
        vertical_fov_deg: Vertical field of view for equirectangular images.
    """

    model: str
    image_width: int
    image_height: int
    translation_m: tuple[float, float, float]
    rotation_rpy_rad: tuple[float, float, float]
    min_depth_m: float
    calibration_status: str
    intrinsics: Mapping[str, float]
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraProjectionConfig":
        """Build and validate a profile from a YAML mapping.

        Args:
            data: The ``camera_projection`` mapping from a robot profile.

        Returns:
            A projection profile suitable for cloud-to-image fusion.

        Raises:
            CalibrationError: If required fields are missing or invalid.
        """

        model = str(data.get("model", "")).strip().lower()
        image = data.get("image", {})
        extrinsics = data.get("lidar_to_camera", {})
        translation = tuple(float(value) for value in extrinsics.get("translation_m", ()))
        rotation = tuple(float(value) for value in extrinsics.get("rotation_rpy_rad", ()))
        if model not in {"pinhole", "equirectangular"}:
            raise CalibrationError("camera_projection.model must be pinhole or equirectangular")
        if len(translation) != 3 or len(rotation) != 3:
            raise CalibrationError("lidar_to_camera translation_m and rotation_rpy_rad must each contain three values")
        try:
            image_width = int(image["width"])
            image_height = int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("camera_projection.image.width and height are required") from exc
        if image_width <= 0 or image_height <= 0:
            raise CalibrationError("camera_projection image dimensions must be positive")

        intrinsics = {key: float(value) for key, value in dict(data.get("intrinsics", {})).items()}
        if model == "pinhole":
            missing = {"fx", "fy", "cx", "cy"}.difference(intrinsics)
            if missing:
                raise CalibrationError(f"pinhole profile is missing intrinsics: {sorted(missing)}")
            if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
                raise CalibrationError("pinhole focal lengths must be positive")

        return cls(
            model=model,
            image_width=image_width,
            image_height=image_height,
            translation_m=translation,
            rotation_rpy_rad=rotation,
            min_depth_m=float(data.get("min_depth_m", 0.05)),
            calibration_status=str(data.get("calibration_status", "uncalibrated")).strip().lower(),
            intrinsics=intrinsics,
            horizontal_fov_deg=(float(data["horizontal_fov_deg"]) if "horizontal_fov_deg" in data else None),
            vertical_fov_deg=(float(data["vertical_fov_deg"]) if "vertical_fov_deg" in data else None),
        )

    def require_calibrated(self) -> None:
        """Reject use of a profile that has not passed physical calibration."""

        if self.calibration_status != "calibrated":
            raise CalibrationError(
                "camera projection is not calibrated; set calibration_status=calibrated only after "
                "intrinsics, LiDAR-to-camera extrinsics, and projection error have been verified"
            )

    def project(self, cloud: np.ndarray) -> np.ndarray:
        """Project a LiDAR cloud to ``[u_px, v_px, depth_m]`` image coordinates.

        Invalid or out-of-view points carry ``u=v=-1`` so existing fusion code
        filters them before indexing masks.  The transform convention is
        ``p_camera = R_lidar_to_camera @ p_lidar + t_lidar_to_camera``.

        Args:
            cloud: Array with at least three columns ``[x, y, z]`` in LiDAR frame.

        Returns:
            A floating-point ``N x 3`` array containing pixel coordinates and depth.

        Raises:
            CalibrationError: If the cloud has an invalid shape or model FOV is absent.
        """

        points = np.asarray(cloud)
        if points.ndim != 2 or points.shape[1] < 3:
            raise CalibrationError("cloud must be an N x 3 (or wider) array")
        rotation = _rotation_matrix_from_rpy(*self.rotation_rpy_rad)
        camera_points = points[:, :3].astype(np.float64, copy=False) @ rotation.T
        camera_points += np.asarray(self.translation_m, dtype=np.float64)
        output = np.full((camera_points.shape[0], 3), -1.0, dtype=np.float64)

        if self.model == "pinhole":
            depth = camera_points[:, 2]
            valid = depth > self.min_depth_m
            output[valid, 0] = self.intrinsics["fx"] * camera_points[valid, 0] / depth[valid] + self.intrinsics["cx"]
            output[valid, 1] = self.intrinsics["fy"] * camera_points[valid, 1] / depth[valid] + self.intrinsics["cy"]
            output[valid, 2] = depth[valid]
            return output

        if not self.horizontal_fov_deg or not self.vertical_fov_deg:
            raise CalibrationError("equirectangular profiles require horizontal_fov_deg and vertical_fov_deg")
        horizontal = np.hypot(camera_points[:, 0], camera_points[:, 1])
        depth = np.linalg.norm(camera_points, axis=1)
        valid = depth > self.min_depth_m
        horizontal_angle = np.arctan2(camera_points[:, 1], camera_points[:, 0])
        vertical_angle = np.arctan2(camera_points[:, 2], horizontal)
        hfov_rad = np.deg2rad(self.horizontal_fov_deg)
        vfov_rad = np.deg2rad(self.vertical_fov_deg)
        valid &= np.abs(horizontal_angle) <= hfov_rad / 2.0
        valid &= np.abs(vertical_angle) <= vfov_rad / 2.0
        output[valid, 0] = (horizontal_angle[valid] / hfov_rad + 0.5) * self.image_width
        output[valid, 1] = (0.5 - vertical_angle[valid] / vfov_rad) * self.image_height
        output[valid, 2] = depth[valid]
        return output


def load_projection_config(path: str | Path, *, require_calibrated: bool) -> CameraProjectionConfig:
    """Load a YAML projection asset and optionally require approved calibration.

    Args:
        path: YAML file holding either a projection mapping or ``camera_projection``.
        require_calibrated: Whether fusion must reject an uncalibrated profile.

    Returns:
        The parsed projection configuration.

    Raises:
        CalibrationError: If the asset is missing, malformed, or unapproved.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise CalibrationError(f"camera projection config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    projection = raw.get("camera_projection", raw)
    if not isinstance(projection, Mapping):
        raise CalibrationError("camera projection YAML must contain a mapping")
    result = CameraProjectionConfig.from_mapping(projection)
    if require_calibrated:
        result.require_calibrated()
    return result
