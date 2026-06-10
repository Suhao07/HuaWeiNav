"""Build topology nodes from traversable frontier candidates.

This module is the boundary between frontier candidate geometry and topology
mutation. Candidate selection happens before this module; graph consistency is
still delegated to the mapper's existing ``add_node`` and ``add_edge`` wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d


def add_nodes_from_candidates(
    mapper: Any,
    *,
    valid_centers: list[tuple[np.ndarray, bool, np.ndarray]],
    current_node_pcd: Any,
    angles_current_node: np.ndarray,
) -> None:
    """Create graph nodes for traversable frontier candidates.

    Each new node stores the local point-cloud slice that faces the candidate
    direction. This preserves STRIVE's legacy debug and view scoring behavior
    while keeping angular slicing out of ``mapper.get_nodes``.
    """

    if len(valid_centers) == 0:
        return

    block_current = (len(valid_centers) == 1)
    for center, has_frontier, frontier_idxs in valid_centers:
        pcd_to_node = _slice_pcd_toward_center(
            mapper,
            center=center,
            current_node_pcd=current_node_pcd,
            angles_current_node=angles_current_node,
        )
        _, add_node_flag = mapper.add_node(center, pcd_to_node, has_frontier, frontier_idxs, block_current)
        if add_node_flag:
            mapper.add_edge(mapper.current_node_idx, mapper.nodes[-1].idx)


def _slice_pcd_toward_center(
    mapper: Any,
    *,
    center: np.ndarray,
    current_node_pcd: Any,
    angles_current_node: np.ndarray,
) -> Any:
    """Return the local 80-degree point-cloud sector facing a candidate center."""

    angle_center = np.arctan2(center[1] - mapper.current_position[1],
                              center[0] - mapper.current_position[0])
    angle_center = np.where(angle_center < 0, angle_center + 2 * np.pi, angle_center)
    angle_center = np.where(angle_center > 2 * np.pi, angle_center - 2 * np.pi, angle_center)

    start_angle = mapper.normalize_angle(angle_center - 40 * np.pi / 180)
    end_angle = mapper.normalize_angle(angle_center + 40 * np.pi / 180)
    if start_angle < end_angle:
        selected_indices = np.where((angles_current_node > start_angle) & (angles_current_node < end_angle))[0]
    else:
        # 跨越 0/2pi 的扇区需要用 OR，否则正前方附近的候选会丢失局部点云。
        selected_indices = np.where((angles_current_node > start_angle) | (angles_current_node < end_angle))[0]
    mask_idx_tensor = o3d.core.Tensor(selected_indices, o3d.core.Dtype.Int64, device=mapper.pcd_device)
    return current_node_pcd.select_by_index(mask_idx_tensor)
