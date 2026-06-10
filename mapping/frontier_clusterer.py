"""Frontier-cluster analysis for STRIVE mapper.

This module converts DBSCAN clusters around the current visible navigable
region into candidate viewpoint centers. It uses mapper-provided geometry
helpers for visibility merging and grid conversion, but it does not create graph
nodes or update room state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

from artifact_utils.path_builder import episode_subdir
from artifact_utils.pointcloud_writer import write_line_set
from mapping.frontier_extractor import prune_redundant_visible_centers
from mapping_utils.projection import translate_point_to_grid


@dataclass
class FrontierClusterResult:
    """Candidate centers and debug clusters produced from frontier DBSCAN labels."""

    clusters: list[Any]
    centers_from_frontiers: list[np.ndarray]
    centers_from_frontiers_map_idxs: list[np.ndarray]
    centers_from_clusters: list[np.ndarray]


def analyze_frontier_clusters(
    mapper: Any,
    *,
    pcd_removed: Any,
    labels: Any,
    current_navigable_pcd: Any,
    frontier_clusters: list[np.ndarray],
    frontier_centers: list[np.ndarray],
    global_frontier_map_idxs: list[np.ndarray],
    obstacle_points: np.ndarray,
    current_position: np.ndarray,
    standing_position: np.ndarray,
    all_points_num: int,
    episode_idx: int,
    step: int,
) -> FrontierClusterResult:
    """Analyze navigable DBSCAN clusters and produce candidate viewpoint centers.

    Parameters are the current local navigable cluster state and global frontier
    metadata from the mapper. The returned centers are not guaranteed reachable;
    traversability filtering remains in ``frontier_extractor``.
    """

    clusters: list[Any] = []
    centers_from_frontiers: list[np.ndarray] = []
    centers_from_frontiers_map_idxs: list[np.ndarray] = []
    centers_from_clusters: list[np.ndarray] = []

    max_label = labels.max().cpu().numpy()
    for cluster_idx in range(max_label + 1):
        mask_idx_tensor = o3d.core.Tensor(
            (labels == cluster_idx).nonzero()[0],
            o3d.core.Dtype.Int64,
            device=current_navigable_pcd.device,
        )
        cluster = pcd_removed.select_by_index(mask_idx_tensor)
        cluster_points = cluster.point.positions.cpu().numpy()
        cluster.paint_uniform_color(np.random.rand(3))

        current_frontier_clusters = []
        current_frontier_centers = []
        current_frontier_map_idxs = []
        for frontier_index, frontier_cluster in enumerate(frontier_clusters):
            distances = np.linalg.norm(
                frontier_cluster[:, np.newaxis, :2] - cluster_points[np.newaxis, :, :2],
                axis=2,
            )
            if np.min(distances) < 0.2:
                current_frontier_clusters.append(frontier_cluster)
                current_frontier_centers.append(frontier_centers[frontier_index])
                current_frontier_map_idxs.append(global_frontier_map_idxs[frontier_index])

        if len(current_frontier_clusters) != 0 and len(cluster_points) > 50:
            clusters.append(cluster)
            _append_visible_frontier_centers(
                mapper,
                cluster_idx=cluster_idx,
                cluster_points=cluster_points,
                obstacle_points=obstacle_points,
                current_frontier_clusters=current_frontier_clusters,
                current_frontier_centers=current_frontier_centers,
                current_frontier_map_idxs=current_frontier_map_idxs,
                current_position=current_position,
                standing_position=standing_position,
                episode_idx=episode_idx,
                step=step,
                centers_from_frontiers=centers_from_frontiers,
                centers_from_frontiers_map_idxs=centers_from_frontiers_map_idxs,
            )

        if _is_uncovered_large_cluster(
            mapper,
            cluster,
            current_frontier_clusters=current_frontier_clusters,
            all_points_num=all_points_num,
        ):
            # 没有 frontier 的大块可通行区域可能是当前视角漏掉的通路。
            # 它只能产生 fallback exploration node，不携带 frontier map index。
            clusters.append(cluster)
            centers_from_clusters.append(np.mean(cluster.point.positions.cpu().numpy(), axis=0))

    return FrontierClusterResult(
        clusters=clusters,
        centers_from_frontiers=centers_from_frontiers,
        centers_from_frontiers_map_idxs=centers_from_frontiers_map_idxs,
        centers_from_clusters=centers_from_clusters,
    )


def _append_visible_frontier_centers(
    mapper: Any,
    *,
    cluster_idx: int,
    cluster_points: np.ndarray,
    obstacle_points: np.ndarray,
    current_frontier_clusters: list[np.ndarray],
    current_frontier_centers: list[np.ndarray],
    current_frontier_map_idxs: list[np.ndarray],
    current_position: np.ndarray,
    standing_position: np.ndarray,
    episode_idx: int,
    step: int,
    centers_from_frontiers: list[np.ndarray],
    centers_from_frontiers_map_idxs: list[np.ndarray],
) -> None:
    """Merge visible frontier fragments and append non-redundant centers."""

    merged_clusters, merged_centers, merged_map_idxs, line_set = mapper.merge_frontier_with_visibility_1(
        cluster_points,
        obstacle_points,
        current_frontier_clusters,
        current_frontier_centers,
        current_frontier_map_idxs,
        current_position,
    )
    _ = merged_clusters
    write_line_set(
        line_set,
        episode_subdir(mapper.save_dir, episode_idx, "frontier_line"),
        f"line_set_{step}_cluster_{cluster_idx}.ply",
    )
    centers_new, map_idxs_new = prune_redundant_visible_centers(
        list(merged_centers),
        list(merged_map_idxs),
        standing_position=standing_position,
        cluster_points=cluster_points,
        obstacle_points=obstacle_points,
        is_visible=mapper.is_visible,
    )
    centers_from_frontiers.extend(centers_new)
    centers_from_frontiers_map_idxs.extend(map_idxs_new)


def _is_uncovered_large_cluster(
    mapper: Any,
    cluster: Any,
    *,
    current_frontier_clusters: list[np.ndarray],
    all_points_num: int,
) -> bool:
    """Return whether a cluster without frontier should still become a candidate."""

    if len(current_frontier_clusters) != 0:
        return False
    cluster_points = cluster.point.positions.cpu().numpy()
    if not ((all_points_num / 20 < len(cluster_points)) or (50 < len(cluster_points))):
        return False

    nav_map = np.zeros((mapper.voxel_dimension[0], mapper.voxel_dimension[1]))
    nav_grid_idxs = translate_point_to_grid(cluster_points, mapper.grid_resolution, mapper.voxel_dimension)
    nav_map[nav_grid_idxs[:, 0], nav_grid_idxs[:, 1]] = 1
    node_positions = np.array([node.position for node in mapper.nodes])
    node_grid_idxs = translate_point_to_grid(node_positions, mapper.grid_resolution, mapper.voxel_dimension)
    for node_grid_idx in node_grid_idxs:
        if nav_map[node_grid_idx[0], node_grid_idx[1]] == 1:
            return False
    return True
