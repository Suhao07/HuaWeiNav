"""Frontier extraction helpers for STRIVE mapper.

The full ``get_nodes()`` pipeline still lives in the mapper because it touches
many runtime caches and debug artifacts. This module extracts pure frontier
sub-steps first, so the largest mapper function can be reduced safely over
multiple small changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def concat_frontier_points(frontier_clusters: Iterable[np.ndarray]) -> np.ndarray:
    """Return all frontier points as one array for debug point-cloud saving."""

    clusters = list(frontier_clusters or [])
    if clusters:
        return np.concatenate(clusters, axis=0)
    return np.array([]).reshape(0, 3)


def adaptive_intersection_distance(
    frontier_clusters: Iterable[np.ndarray],
    standing_position: np.ndarray,
    *,
    default_distance: float = 2.5,
    scale: float = 1.2,
) -> tuple[float, float | None]:
    """Choose the local visibility radius from the nearest frontier distance.

    近 frontier 表示当前视野已经接近未知区域，局部截断半径应收紧；
    没有 frontier 时保守使用默认半径，避免空地图阶段直接退出。
    """

    clusters = list(frontier_clusters or [])
    if not clusters:
        return default_distance, None
    frontiers_all = np.concatenate(clusters, axis=0)
    distance_to_frontiers = np.linalg.norm(frontiers_all - standing_position, axis=1)
    min_distance = float(np.min(distance_to_frontiers))
    return min(default_distance, min_distance * scale), min_distance


def prune_redundant_visible_centers(
    centers: list[np.ndarray],
    map_indices: list[np.ndarray],
    *,
    standing_position: np.ndarray,
    cluster_points: np.ndarray,
    obstacle_points: np.ndarray,
    is_visible: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool],
    angle_threshold_rad: float = 10 * np.pi / 180,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove duplicate frontier centers that share direction and visibility.

    多个 frontier center 如果方向几乎一致且互相可见，通常来自同一个门口或
    通道。保留离机器人更近的一个，减少重复 viewpoint node。
    """

    center_idx_to_remove: set[int] = set()
    for index1 in range(len(centers)):
        for index2 in range(index1 + 1, len(centers)):
            if index1 in center_idx_to_remove or index2 in center_idx_to_remove:
                continue
            center1 = centers[index1]
            center2 = centers[index2]
            angle1 = np.arctan2(center1[1] - standing_position[1], center1[0] - standing_position[0])
            angle2 = np.arctan2(center2[1] - standing_position[1], center2[0] - standing_position[0])
            angle_diff = np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi)
            if angle_diff >= angle_threshold_rad:
                continue
            if not is_visible(center1, center2, cluster_points, obstacle_points):
                continue
            distance1 = np.linalg.norm(center1[:2] - standing_position[:2])
            distance2 = np.linalg.norm(center2[:2] - standing_position[:2])
            center_idx_to_remove.add(index2 if distance1 < distance2 else index1)

    centers_new = [center for idx, center in enumerate(centers) if idx not in center_idx_to_remove]
    map_idxs_new = [map_idx for idx, map_idx in enumerate(map_indices) if idx not in center_idx_to_remove]
    return centers_new, map_idxs_new


def append_traversable_candidate(
    mapper: Any,
    valid_centers: list[tuple[np.ndarray, bool, np.ndarray]],
    *,
    center: np.ndarray,
    has_frontier: bool,
    frontier_idxs: np.ndarray,
    current_position: np.ndarray,
) -> None:
    """Project a candidate center onto traversable space and append it if reachable."""

    # candidate center 必须先吸附到可通行点云，再通过 Habitat pathfinder
    # 验证可达；否则 topology graph 会包含几何上存在但机器人走不到的 node。
    center = mapper.find_closest_point_in_pc(center, mapper.traversable_pcd)
    if center is None:
        return
    if not mapper.check_traversability(current_position, center):
        return
    center[2] = mapper.current_position[2]
    valid_centers.append((center, has_frontier, frontier_idxs))
