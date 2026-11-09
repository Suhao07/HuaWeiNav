from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionQuality:
    score: float
    visible_ratio: float
    area_ratio: float
    center_score: float
    border_score: float
    distance_score: float
    distance_to_target: float
    predicted_quality: dict[str, float]


def interpolate_polyline(points: np.ndarray, spacing: float = 0.25) -> list[np.ndarray]:
    """Interpolate a polyline with approximately fixed spacing.

    This is deliberately pure geometry: it does not know object classes or
    instruction semantics, so it can be reused by both benchmark and
    instruction-aware view-control.
    """

    if points is None or len(points) < 2:
        return []
    out: list[np.ndarray] = []
    for idx in range(len(points) - 1):
        point1 = np.asarray(points[idx], dtype=float)
        point2 = np.asarray(points[idx + 1], dtype=float)
        distance = float(np.linalg.norm(point1 - point2))
        num_points = max(2, int(np.ceil(distance / spacing)) + 1)
        out.extend([(1.0 - t) * point1 + t * point2 for t in np.linspace(0.0, 1.0, num_points)])
    return out


def truncate_at_stop_radius(
    path: list[np.ndarray],
    object_positions: np.ndarray,
    success_distance: float,
    stop_criterion: float,
) -> list[np.ndarray]:
    """Keep the portion of a path up to the last point inside stop radius."""

    if not path:
        return []
    if object_positions is None or len(object_positions) == 0:
        return path
    stop_idx = None
    positions_xy = np.asarray(object_positions, dtype=float)[:, :2]
    threshold = float(success_distance) * float(stop_criterion)
    for point_idx, point in enumerate(path):
        to_target_distance = float(np.min(np.linalg.norm(positions_xy - np.asarray(point)[:2], axis=1)))
        if to_target_distance <= threshold:
            stop_idx = point_idx
    if stop_idx is None:
        stop_idx = len(path) - 1
    return path[: stop_idx + 1]


def rotation_towards(stop_pos_xy: np.ndarray, target_pos_xy: np.ndarray, *,
                     pitch: float = 0.0) -> np.ndarray | None:
    """Camera-to-mapper rotation; positive pitch raises the optical forward axis.

    R = R_yaw @ R_x(pitch). The optical forward direction is camera -Z;
    pitch is in radians. Mapper axes (x, z, y) include the existing reflection;
    this basis transform has determinant -1. Zero pitch preserves baseline views.
    """
    orient = np.asarray(target_pos_xy, dtype=float) - np.asarray(stop_pos_xy, dtype=float)
    orient_norm = float(np.linalg.norm(orient))
    if orient_norm < 1e-6:
        return None
    orient = orient / orient_norm
    yaw_rotation = np.array(
        [
            [-orient[1], 0.0, -orient[0]],
            [orient[0], 0.0, -orient[1]],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return yaw_rotation @ pitch_rotation(pitch)


def optical_rotation_towards(
    stop_pos_xy: np.ndarray, target_pos_xy: np.ndarray, *, pitch: float = 0.0,
) -> np.ndarray | None:
    """Return optical right/down/forward axes in a right-handed Z-up map.

    Args:
        stop_pos_xy: Camera horizontal position in map coordinates.
        target_pos_xy: Target horizontal position in the same map.
        pitch: Executable elevation in radians; positive looks up.
    Returns:
        A 3x3 camera-to-map rotation, or None for coincident XY positions.
        For unit heading (x, y), right = (y, -x, 0), down = (0, 0, -1)
        and forward = (x, y, 0) at zero pitch.
    """
    orient = np.asarray(target_pos_xy, dtype=float) - np.asarray(stop_pos_xy, dtype=float)
    orient_norm = float(np.linalg.norm(orient))
    if orient_norm < 1e-6:
        return None
    x, y = orient / orient_norm
    return np.array([[y, 0., x], [-x, 0., y], [0., -1., 0.]]) @ pitch_rotation(pitch)


def pitch_rotation(pitch: float) -> np.ndarray:
    r"""Camera-local rotation about the right axis; positive pitch looks up.

    .. math::
        R_x(\phi) = \begin{bmatrix}1&0&0\\0&\cos\phi&-\sin\phi\\0&\sin\phi&\cos\phi\end{bmatrix}

    右乘作用于相机局部系，与 Habitat LOOK_UP/LOOK_DOWN 的传感器旋转一致。

    Args:
        pitch: Rotation angle in radians.
    Returns:
        The 3x3 rotation matrix.
    """
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def camera_pitch(rotation: np.ndarray) -> float:
    r"""Return the optical-axis elevation of a camera-to-mapper rotation.

    光轴 f = -R_{:,2}，建图系第三轴竖直向上：\phi = \arcsin(f_z) = \arcsin(-R_{2,2})。

    Args:
        rotation: Camera-to-mapper rotation.
    Returns:
        Pitch in radians; positive above the horizon.
    """
    return float(np.arcsin(np.clip(-rotation[2, 2], -1.0, 1.0)))


def score_points_view(
    *,
    object_points: np.ndarray,
    original_bbox: Any,
    camera_intrinsic: np.ndarray,
    camera_to_map: np.ndarray,
    success_distance: float,
    arrival_tolerance: float = 0.0,
    image_width: int = 640,
    image_height: int = 480,
) -> tuple[ProjectionQuality, bool] | None:
    r"""Score map points from an optical camera without runtime dependencies.

    The map is right-handed Z-up; ``camera_to_map`` maps optical camera
    coordinates (+X right, +Y down, +Z forward) into that same map. Projection
    uses u = fx*x/z + cx and v = fy*y/z + cy, truncated to int32 as in the
    baseline. Only positive-depth, in-frame support contributes to distance:

    .. math::
        d = \min_{p \in V} \|p_{xy} - v_{xy}\|_2,\qquad d + \rho \le d_{stop}.

    The visible ratio denominator counts all positive-depth instance points.
    This is predicted support, not depth/occlusion or semantic verification.

    Args:
        object_points: Nx3 instance points in the map, not camera coordinates.
        original_bbox: Observed (xmin, ymin, xmax, ymax) pixel bounds.
        camera_intrinsic: 3x3 pinhole intrinsics (zero skew).
        camera_to_map: Homogeneous 4x4 optical-camera-to-map transform.
        success_distance: Required XY stop distance in map units.
        arrival_tolerance: Nonnegative planner arrival radius rho.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    Returns:
        (ProjectionQuality, stop_feasible), or None without visible support.
    Raises:
        ValueError: Nonempty points or transform have invalid shapes.
        numpy.linalg.LinAlgError: The camera transform is singular.
    """
    positions = np.asarray(object_points, dtype=float)
    if positions.size == 0:
        return None
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("object_points must have shape (N, 3)")
    camera_to_map = np.asarray(camera_to_map, dtype=float)
    if camera_to_map.shape != (4, 4):
        raise ValueError("camera_to_map must have shape (4, 4)")
    intrinsic = np.asarray(camera_intrinsic, dtype=float)
    homogeneous = np.concatenate((positions, np.ones((len(positions), 1))), axis=1)
    camera_points = (np.linalg.inv(camera_to_map) @ homogeneous.T).T[:, :3]
    front = camera_points[:, 2] > 0
    camera_points = camera_points[front]
    if not len(camera_points):
        return None
    depth = camera_points[:, 2]
    pixels = np.column_stack((
        camera_points[:, 0] * intrinsic[0, 0] / depth + intrinsic[0, 2],
        camera_points[:, 1] * intrinsic[1, 1] / depth + intrinsic[1, 2],
    )).astype(np.int32)
    in_frame = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)
    )
    bbox = np.asarray(original_bbox)
    quality = score_projection(
        camera_points=pixels[in_frame],
        all_point_count=len(camera_points),
        object_positions=positions[front][in_frame],
        viewpoint_position=camera_to_map[:3, 3],
        original_bbox_area=float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        original_bbox_length=float(bbox[2] - bbox[0]),
        original_bbox_width=float(bbox[3] - bbox[1]),
        success_distance=success_distance,
        image_width=image_width,
        image_height=image_height,
    )
    if quality is None:
        return None
    return quality, quality.distance_to_target + arrival_tolerance <= success_distance


def _proposal_rank(candidate: dict[str, Any]) -> tuple[float, float, float]:
    """Preserve baseline terminal-cost versus intermediate-quality ordering."""
    if candidate.get("stop_feasible", False):
        return (1, -candidate["travel_distance"], candidate["score"])
    return (0, candidate["score"], -candidate.get("travel_distance", 0.0))


def rank_view_proposals(
    proposals: list[dict[str, Any]], *, approach: bool = False,
) -> list[dict[str, Any]]:
    """Return reachable proposals best first without modifying input records.

    Args:
        proposals: Geometry proposals with score, stop_feasible and
            travel_distance. Backends should replace the builder's XY distance
            with actual navigation cost before ranking. Active rechecks must
            additionally filter travel_distance > arrival_tolerance.
        approach: If True, keep only stop-feasible proposals and rank by score.
            Otherwise rank feasible views by shortest travel then quality;
            rank intermediate views by quality then shortest travel.
    Returns:
        A new, stably sorted list of the original dictionaries, excluding
        nonfinite travel costs. Equal keys preserve proposal generation order.
    """
    reachable = [p for p in proposals if np.isfinite(p.get("travel_distance", 0.0))]
    if approach:
        return sorted((p for p in reachable if p.get("stop_feasible", False)),
                      key=lambda p: p["score"], reverse=True)
    return sorted(reachable, key=_proposal_rank, reverse=True)


def score_projection(
    *,
    camera_points: np.ndarray,
    all_point_count: int,
    object_positions: np.ndarray,
    viewpoint_position: np.ndarray,
    original_bbox_area: float,
    original_bbox_length: float,
    original_bbox_width: float,
    success_distance: float,
    image_width: int = 640,
    image_height: int = 480,
) -> ProjectionQuality | None:
    """Score one projected object view using class-agnostic visual geometry.

    这里不写任何 book/shelf/chair 等语义规则，只衡量候选视角
    对当前候选对象是否更可见、更居中、更适合最终复核。
    """

    if camera_points is None or len(camera_points) == 0 or all_point_count <= 0:
        return None

    camera_points = np.asarray(camera_points, dtype=float)
    bbox = np.array([np.min(camera_points, axis=0), np.max(camera_points, axis=0)], dtype=float)
    length = max(1.0, float(bbox[1][0] - bbox[0][0]))
    width = max(1.0, float(bbox[1][1] - bbox[0][1]))
    bbox_area = length * width

    cx = float((bbox[0][0] + bbox[1][0]) / 2.0 / image_width)
    cy = float((bbox[0][1] + bbox[1][1]) / 2.0 / image_height)
    center_offset = float(np.linalg.norm(np.array([cx - 0.5, cy - 0.5], dtype=float)))
    center_score = max(0.0, 1.0 - center_offset / np.sqrt(0.5))
    border_margin = float(min(cx, 1.0 - cx, cy, 1.0 - cy))
    border_score = max(0.0, min(1.0, border_margin / 0.25))

    visible_ratio = float(len(camera_points) / max(1, all_point_count))
    area_ratio = float(bbox_area / max(1.0, original_bbox_area))
    area_score = max(0.0, min(1.0, np.sqrt(area_ratio) / 2.0))
    to_target_distance = float(
        np.min(np.linalg.norm(np.asarray(object_positions, dtype=float)[:, :2] - np.asarray(viewpoint_position)[:2], axis=1))
    )
    preferred_distance = max(0.25, float(success_distance))
    distance_score = max(0.0, 1.0 - abs(to_target_distance - preferred_distance) / max(preferred_distance, 1.0))
    aspect_stability = min(
        1.0,
        float(length / max(1.0, original_bbox_length)),
        float(width / max(1.0, original_bbox_width)),
    )
    score = (
        0.35 * visible_ratio
        + 0.25 * center_score
        + 0.15 * border_score
        + 0.15 * area_score
        + 0.10 * distance_score
    ) * (0.75 + 0.25 * aspect_stability)

    return ProjectionQuality(
        score=float(score),
        visible_ratio=visible_ratio,
        area_ratio=area_ratio,
        center_score=float(center_score),
        border_score=float(border_score),
        distance_score=float(distance_score),
        distance_to_target=to_target_distance,
        predicted_quality={
            "score": float(score),
            "area_score": float(area_score),
            "center_score": float(center_score),
            "border_score": float(border_score),
            "visible_score": float(visible_ratio),
            "distance_score": float(distance_score),
            "bbox_area_ratio": float(bbox_area / (float(image_width) * float(image_height))),
            "center_offset_norm": float(center_offset),
            "border_margin_norm": float(border_margin),
        },
    )


def habitat_path_to_strive_points(
    points: Any,
    *,
    mapper_initial_position: np.ndarray,
    target_height: float,
) -> np.ndarray:
    """Convert Habitat xyz path points into VLN's local x/z/y convention."""

    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 3), dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    converted = np.array([points[:, 0], points[:, 2], points[:, 1]], dtype=float).T
    converted = converted - np.asarray(mapper_initial_position, dtype=float)
    converted[:, 2] = float(target_height)
    return converted

