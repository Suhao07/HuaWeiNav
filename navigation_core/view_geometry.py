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
        num_points = max(1, int(distance / spacing))
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


def rotation_towards(stop_pos_xy: np.ndarray, target_pos_xy: np.ndarray) -> np.ndarray | None:
    orient = np.asarray(target_pos_xy, dtype=float) - np.asarray(stop_pos_xy, dtype=float)
    orient_norm = float(np.linalg.norm(orient))
    if orient_norm < 1e-6:
        return None
    orient = orient / orient_norm
    return np.array(
        [
            [-orient[1], 0.0, -orient[0]],
            [orient[0], 0.0, -orient[1]],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


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
    """Convert Habitat xyz path points into STRIVE's local x/z/y convention."""

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

