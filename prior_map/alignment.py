"""Coordinate alignment for VLN prior-map mode.

This module owns only coordinate transforms between a prior-map frame and a
runtime frame. It does not inspect mapper state, rank frontiers, or publish
navigation goals.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .contracts import Vector2, Vector3


class PriorMapAlignmentError(ValueError):
    """Raised when a prior-map alignment cannot safely transform geometry."""


@dataclass(frozen=True)
class PriorMapAlignment:
    """Transform between prior-map coordinates and runtime coordinates.

    The affine 2-D model is a similarity transform:

    \[
    p_{runtime,xy} = s R_\theta p_{prior,xy} + t_{xy}
    \]

    Args:
        alignment_type: ``"identity"``, ``"affine_2d"``, or ``"unavailable"``.
        prior_frame_id: Source frame for prior-map geometry.
        runtime_frame_id: Target runtime frame, such as ``map``.
        scale: Uniform 2-D scale factor.
        rotation_rad: Counter-clockwise rotation in radians.
        translation_xyz: Runtime translation, with z used as a constant offset.
        base_confidence: Alignment confidence in ``[0, 1]``.
        enabled_for_ranking: Whether geometry from this alignment may bias
            frontier/object ranking. Disabled alignments can still provide
            prompt context but must not rank geometry.
        diagnostics: JSON-friendly diagnostic fields.
    """

    alignment_type: str = "identity"
    prior_frame_id: str = "prior_map"
    runtime_frame_id: str = "map"
    scale: float = 1.0
    rotation_rad: float = 0.0
    translation_xyz: Vector3 = (0.0, 0.0, 0.0)
    base_confidence: float = 1.0
    enabled_for_ranking: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate alignment fields.

        Raises:
            PriorMapAlignmentError: If the alignment type, scale, translation,
                or confidence is invalid.
        """

        if self.alignment_type not in {"identity", "affine_2d", "unavailable"}:
            raise PriorMapAlignmentError(f"Unsupported alignment_type: {self.alignment_type}")
        if not self.prior_frame_id:
            raise PriorMapAlignmentError("prior_frame_id must be non-empty")
        if not self.runtime_frame_id:
            raise PriorMapAlignmentError("runtime_frame_id must be non-empty")
        if not isinstance(self.scale, (int, float)) or self.scale <= 0.0:
            raise PriorMapAlignmentError("scale must be positive")
        if not isinstance(self.rotation_rad, (int, float)):
            raise PriorMapAlignmentError("rotation_rad must be numeric")
        if len(self.translation_xyz) != 3:
            raise PriorMapAlignmentError("translation_xyz must have length 3")
        for value in self.translation_xyz:
            if not isinstance(value, (int, float)):
                raise PriorMapAlignmentError("translation_xyz must be numeric")
        if not isinstance(self.base_confidence, (int, float)) or not 0.0 <= self.base_confidence <= 1.0:
            raise PriorMapAlignmentError("base_confidence must be in [0, 1]")

    @classmethod
    def identity(
        cls,
        prior_frame_id: str = "prior_map",
        runtime_frame_id: str = "map",
        confidence: float = 1.0,
    ) -> "PriorMapAlignment":
        """Create an identity alignment.

        Args:
            prior_frame_id: Source prior-map frame.
            runtime_frame_id: Target runtime frame.
            confidence: Alignment confidence in ``[0, 1]``.

        Returns:
            Identity ``PriorMapAlignment``.
        """

        return cls(
            alignment_type="identity",
            prior_frame_id=prior_frame_id,
            runtime_frame_id=runtime_frame_id,
            scale=1.0,
            rotation_rad=0.0,
            translation_xyz=(0.0, 0.0, 0.0),
            base_confidence=confidence,
            enabled_for_ranking=confidence > 0.0,
            diagnostics={
                "method": "identity",
                "prior_frame_id": prior_frame_id,
                "runtime_frame_id": runtime_frame_id,
            },
        )

    @classmethod
    def affine_2d(
        cls,
        scale: float,
        rotation_rad: float,
        translation_xyz: Vector3 = (0.0, 0.0, 0.0),
        prior_frame_id: str = "prior_map",
        runtime_frame_id: str = "map",
        confidence: float = 1.0,
        source_points_prior: Sequence[Vector2] = (),
        target_points_runtime: Sequence[Vector2] = (),
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "PriorMapAlignment":
        """Create an affine 2-D similarity alignment.

        Args:
            scale: Uniform 2-D scale factor.
            rotation_rad: Counter-clockwise rotation in radians.
            translation_xyz: Runtime translation, with z as a constant offset.
            prior_frame_id: Source prior-map frame.
            runtime_frame_id: Target runtime frame.
            confidence: Alignment confidence in ``[0, 1]``.
            source_points_prior: Optional prior-frame calibration points.
            target_points_runtime: Optional runtime-frame calibration points.
            diagnostics: Extra JSON-friendly diagnostics.

        Returns:
            Affine ``PriorMapAlignment``.
        """

        diag = dict(diagnostics or {})
        diag.update(
            _point_pair_diagnostics(
                scale=scale,
                rotation_rad=rotation_rad,
                translation_xyz=translation_xyz,
                source_points_prior=source_points_prior,
                target_points_runtime=target_points_runtime,
            )
        )
        diag.update(
            {
                "method": "affine_2d",
                "prior_frame_id": prior_frame_id,
                "runtime_frame_id": runtime_frame_id,
            }
        )
        return cls(
            alignment_type="affine_2d",
            prior_frame_id=prior_frame_id,
            runtime_frame_id=runtime_frame_id,
            scale=float(scale),
            rotation_rad=float(rotation_rad),
            translation_xyz=_vector3(translation_xyz),
            base_confidence=confidence,
            enabled_for_ranking=confidence > 0.0,
            diagnostics=diag,
        )

    @classmethod
    def from_point_pairs(
        cls,
        source_points_prior: Sequence[Vector2],
        target_points_runtime: Sequence[Vector2],
        prior_frame_id: str = "prior_map",
        runtime_frame_id: str = "map",
        confidence: Optional[float] = None,
    ) -> "PriorMapAlignment":
        """Estimate a uniform-scale affine 2-D alignment from point pairs.

        The least-squares estimate solves for \(a, b, t_x, t_y\) where:

        \[
        x' = a x - b y + t_x,\quad y' = b x + a y + t_y
        \]

        Args:
            source_points_prior: Prior-frame calibration points.
            target_points_runtime: Corresponding runtime-frame points.
            prior_frame_id: Source prior-map frame.
            runtime_frame_id: Target runtime frame.
            confidence: Optional confidence override. If omitted, confidence is
                derived from residual RMSE.

        Returns:
            Estimated affine ``PriorMapAlignment``.

        Raises:
            PriorMapAlignmentError: If there are fewer than two point pairs or
                the source points are degenerate.
        """

        source = tuple(_vector2(point) for point in source_points_prior)
        target = tuple(_vector2(point) for point in target_points_runtime)
        if len(source) != len(target):
            raise PriorMapAlignmentError("source_points_prior and target_points_runtime must have equal length")
        if len(source) < 2:
            raise PriorMapAlignmentError("at least two point pairs are required")

        source_centroid = _centroid(source)
        target_centroid = _centroid(target)
        ss = 0.0
        dot = 0.0
        cross = 0.0
        for src, dst in zip(source, target):
            sx = src[0] - source_centroid[0]
            sy = src[1] - source_centroid[1]
            tx = dst[0] - target_centroid[0]
            ty = dst[1] - target_centroid[1]
            ss += sx * sx + sy * sy
            dot += tx * sx + ty * sy
            cross += ty * sx - tx * sy
        if ss <= 0.0:
            raise PriorMapAlignmentError("source point pairs are degenerate")

        a = dot / ss
        b = cross / ss
        scale = math.hypot(a, b)
        if scale <= 0.0:
            raise PriorMapAlignmentError("estimated scale is degenerate")
        rotation_rad = math.atan2(b, a)

        rotated_centroid = _apply_similarity_xy(source_centroid, scale, rotation_rad, (0.0, 0.0, 0.0))
        translation_xyz = (
            target_centroid[0] - rotated_centroid[0],
            target_centroid[1] - rotated_centroid[1],
            0.0,
        )
        diag = _point_pair_diagnostics(
            scale=scale,
            rotation_rad=rotation_rad,
            translation_xyz=translation_xyz,
            source_points_prior=source,
            target_points_runtime=target,
        )
        computed_confidence = confidence
        if computed_confidence is None:
            rmse = float(diag.get("rmse", 0.0))
            computed_confidence = 1.0 / (1.0 + rmse)

        return cls.affine_2d(
            scale=scale,
            rotation_rad=rotation_rad,
            translation_xyz=translation_xyz,
            prior_frame_id=prior_frame_id,
            runtime_frame_id=runtime_frame_id,
            confidence=max(0.0, min(1.0, float(computed_confidence))),
            source_points_prior=source,
            target_points_runtime=target,
            diagnostics={"estimated_from_point_pairs": True},
        )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        prior_frame_id: str = "prior_map",
        runtime_frame_id: str = "map",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "PriorMapAlignment":
        """Create an unavailable alignment for prompt-context-only fallback.

        Args:
            reason: Concrete reason geometry alignment is unavailable.
            prior_frame_id: Source prior-map frame.
            runtime_frame_id: Target runtime frame.
            diagnostics: Extra JSON-friendly diagnostics.

        Returns:
            Alignment object with zero confidence and geometry ranking disabled.
        """

        diag = dict(diagnostics or {})
        diag.update(
            {
                "method": "unavailable",
                "reason": reason,
                "prior_frame_id": prior_frame_id,
                "runtime_frame_id": runtime_frame_id,
                "fallback": "prompt_context_only",
            }
        )
        return cls(
            alignment_type="unavailable",
            prior_frame_id=prior_frame_id,
            runtime_frame_id=runtime_frame_id,
            scale=1.0,
            rotation_rad=0.0,
            translation_xyz=(0.0, 0.0, 0.0),
            base_confidence=0.0,
            enabled_for_ranking=False,
            diagnostics=diag,
        )

    def prior_to_runtime(self, point_xy: Sequence[float], z: float = 0.0) -> Vector3:
        """Transform a prior-map point into the runtime frame.

        Args:
            point_xy: Prior-frame ``(x, y)`` point.
            z: Optional prior z coordinate. The affine 2-D model only applies a
                constant z translation.

        Returns:
            Runtime-frame ``(x, y, z)`` point.

        Raises:
            PriorMapAlignmentError: If geometry alignment is unavailable.
        """

        self._ensure_can_transform()
        x, y = _apply_similarity_xy(_vector2(point_xy), self.scale, self.rotation_rad, self.translation_xyz)
        return (x, y, float(z) + float(self.translation_xyz[2]))

    def runtime_to_prior(self, point_xyz: Sequence[float]) -> Vector2:
        """Transform a runtime-frame point into prior-map coordinates.

        Args:
            point_xyz: Runtime-frame point with at least ``(x, y)``.

        Returns:
            Prior-frame ``(x, y)`` point.

        Raises:
            PriorMapAlignmentError: If geometry alignment is unavailable.
        """

        self._ensure_can_transform()
        point = _vector3(point_xyz)
        dx = (point[0] - float(self.translation_xyz[0])) / self.scale
        dy = (point[1] - float(self.translation_xyz[1])) / self.scale
        cos_theta = math.cos(self.rotation_rad)
        sin_theta = math.sin(self.rotation_rad)
        return (
            cos_theta * dx + sin_theta * dy,
            -sin_theta * dx + cos_theta * dy,
        )

    def confidence(self) -> float:
        """Return alignment confidence for geometry ranking.

        Returns:
            Confidence in ``[0, 1]``. Unavailable or ranking-disabled
            alignments return ``0``.
        """

        if not self.enabled_for_ranking or self.alignment_type == "unavailable":
            return 0.0
        return float(self.base_confidence)

    def can_rank_geometry(self) -> bool:
        """Return whether this alignment may bias geometric ranking.

        Returns:
            ``True`` only when alignment is available and confidence is
            positive.
        """

        return self.confidence() > 0.0

    def diagnostics_payload(self) -> Dict[str, Any]:
        """Return diagnostics with mandatory frame and fallback fields.

        Returns:
            JSON-friendly diagnostics dictionary.
        """

        payload = dict(self.diagnostics)
        payload.update(
            {
                "alignment_type": self.alignment_type,
                "prior_frame_id": self.prior_frame_id,
                "runtime_frame_id": self.runtime_frame_id,
                "scale": self.scale,
                "rotation_rad": self.rotation_rad,
                "translation_xyz": list(self.translation_xyz),
                "confidence": self.confidence(),
                "enabled_for_ranking": self.can_rank_geometry(),
            }
        )
        if not self.can_rank_geometry():
            payload.setdefault("fallback", "prompt_context_only")
        return _json_ready(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly alignment representation.

        Returns:
            Dictionary suitable for ``json.dumps``.
        """

        return {
            "alignment_type": self.alignment_type,
            "prior_frame_id": self.prior_frame_id,
            "runtime_frame_id": self.runtime_frame_id,
            "scale": self.scale,
            "rotation_rad": self.rotation_rad,
            "translation_xyz": list(self.translation_xyz),
            "base_confidence": self.base_confidence,
            "enabled_for_ranking": self.enabled_for_ranking,
            "diagnostics": self.diagnostics_payload(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorMapAlignment":
        """Create an alignment from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from ``alignment.json``.

        Returns:
            Reconstructed ``PriorMapAlignment``.
        """

        data = dict(payload)
        data["translation_xyz"] = _vector3(data.get("translation_xyz", (0.0, 0.0, 0.0)))
        data.setdefault("diagnostics", {})
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save alignment to ``alignment.json``.

        Args:
            path: Destination JSON path.
        """

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PriorMapAlignment":
        """Load alignment from ``alignment.json``.

        Args:
            path: Alignment JSON path.

        Returns:
            Loaded ``PriorMapAlignment``.
        """

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PriorMapAlignmentError("alignment JSON root must be an object")
        return cls.from_dict(payload)

    def _ensure_can_transform(self) -> None:
        """Raise if geometry should not be transformed.

        Raises:
            PriorMapAlignmentError: If this alignment is unavailable.
        """

        if self.alignment_type == "unavailable":
            reason = self.diagnostics.get("reason", "alignment unavailable")
            raise PriorMapAlignmentError(f"Prior map geometry alignment unavailable: {reason}")


def _point_pair_diagnostics(
    scale: float,
    rotation_rad: float,
    translation_xyz: Vector3,
    source_points_prior: Sequence[Vector2],
    target_points_runtime: Sequence[Vector2],
) -> Dict[str, Any]:
    """Compute calibration point residual diagnostics.

    Args:
        scale: Uniform 2-D scale factor.
        rotation_rad: Counter-clockwise rotation in radians.
        translation_xyz: Runtime translation.
        source_points_prior: Prior-frame calibration points.
        target_points_runtime: Runtime-frame calibration points.

    Returns:
        JSON-friendly diagnostics. Empty when no points are provided.
    """

    if not source_points_prior and not target_points_runtime:
        return {}
    source = tuple(_vector2(point) for point in source_points_prior)
    target = tuple(_vector2(point) for point in target_points_runtime)
    if len(source) != len(target):
        return {
            "source_points_prior": [list(point) for point in source],
            "target_points_runtime": [list(point) for point in target],
            "point_pair_error": "source and target point counts differ",
        }

    residuals = []
    errors = []
    for src, dst in zip(source, target):
        projected = _apply_similarity_xy(src, scale, rotation_rad, translation_xyz)
        residual = (projected[0] - dst[0], projected[1] - dst[1])
        error = math.hypot(residual[0], residual[1])
        residuals.append(residual)
        errors.append(error)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0
    return {
        "source_points_prior": [list(point) for point in source],
        "target_points_runtime": [list(point) for point in target],
        "residuals_xy": [list(item) for item in residuals],
        "rmse": rmse,
        "max_error": max(errors) if errors else 0.0,
        "num_point_pairs": len(source),
    }


def _apply_similarity_xy(
    point_xy: Vector2,
    scale: float,
    rotation_rad: float,
    translation_xyz: Vector3,
) -> Vector2:
    """Apply a 2-D similarity transform.

    Args:
        point_xy: Prior-frame point.
        scale: Uniform scale factor.
        rotation_rad: Rotation angle in radians.
        translation_xyz: Runtime translation.

    Returns:
        Runtime-frame 2-D point.
    """

    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    x = scale * (cos_theta * point_xy[0] - sin_theta * point_xy[1]) + translation_xyz[0]
    y = scale * (sin_theta * point_xy[0] + cos_theta * point_xy[1]) + translation_xyz[1]
    return (x, y)


def _centroid(points: Sequence[Vector2]) -> Vector2:
    """Compute an arithmetic centroid.

    Args:
        points: Non-empty 2-D points.

    Returns:
        Centroid.
    """

    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _vector2(value: Sequence[float]) -> Vector2:
    """Validate and convert a 2-D point.

    Args:
        value: Sequence with at least two numeric values.

    Returns:
        ``(x, y)`` tuple.

    Raises:
        PriorMapAlignmentError: If the value is invalid.
    """

    if len(value) < 2:
        raise PriorMapAlignmentError("point must have at least two coordinates")
    return (_number(value[0]), _number(value[1]))


def _vector3(value: Sequence[float]) -> Vector3:
    """Validate and convert a 3-D point.

    Args:
        value: Sequence with at least two numeric values.

    Returns:
        ``(x, y, z)`` tuple. Missing z defaults to zero.

    Raises:
        PriorMapAlignmentError: If the value is invalid.
    """

    if len(value) < 2:
        raise PriorMapAlignmentError("point must have at least two coordinates")
    z = value[2] if len(value) >= 3 else 0.0
    return (_number(value[0]), _number(value[1]), _number(z))


def _number(value: float) -> float:
    """Validate and convert a number.

    Args:
        value: Candidate numeric value.

    Returns:
        Float value.

    Raises:
        PriorMapAlignmentError: If the value is not numeric.
    """

    if not isinstance(value, (int, float)):
        raise PriorMapAlignmentError("coordinate values must be numeric")
    return float(value)


def _json_ready(value: Any) -> Any:
    """Convert tuples and dictionaries to JSON-native values.

    Args:
        value: Arbitrary diagnostic value.

    Returns:
        JSON-friendly value.
    """

    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
