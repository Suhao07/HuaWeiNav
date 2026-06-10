"""Frontier exploration policies for STRIVE navigation.

These helpers choose the next viewpoint node from the mapper graph. They do not
perform perception, room segmentation, target verification, or Habitat actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d
from loguru import logger


def find_closest_nodes(mapper: Any, nodes: list[Any]) -> Any | None:
    """Pick the geodesically closest node from a candidate node list."""

    if not nodes:
        return None
    nodes_positions = np.array([node.position for node in nodes], dtype=float)
    nodes_positions = nodes_positions + mapper.initial_position
    nodes_positions[:, 2] = mapper.initial_position[2] - 0.88
    nodes_positions = nodes_positions[:, [0, 2, 1]]

    current_position = mapper.current_position + mapper.initial_position
    current_position = np.array([
        current_position[0],
        mapper.initial_position[2] - 0.88,
        current_position[1],
    ])
    distance = np.array([
        mapper.env.sim.geodesic_distance(current_position, node_position)
        for node_position in nodes_positions
    ])
    return nodes[int(np.argmin(distance))]


def find_closest_viewpoint_in_room(mapper: Any, room_node: Any) -> Any | None:
    """Return the closest unexplored frontier viewpoint inside a room."""

    nodes = [
        node
        for node in room_node.nodes
        if node.state == 0 and node.has_frontier is True
    ]
    return find_closest_nodes(mapper, nodes)


def explore_in_room(mapper: Any, room_node: Any) -> Any | None:
    """Choose the next frontier node inside the current room."""

    _reset_process_pointclouds(mapper)
    nodes = [node for node in room_node.nodes if node.has_frontier is True]
    return find_closest_nodes(mapper, nodes)


def explore_in_room_relocate(mapper: Any, room_node: Any) -> Any | None:
    """Choose a frontier during relocation and close exhausted rooms."""

    _reset_process_pointclouds(mapper)
    nodes_true_frontier = [
        node
        for node in room_node.nodes
        if node.state == 0 and node.has_true_frontier is True
    ]
    if nodes_true_frontier:
        return find_closest_nodes(mapper, nodes_true_frontier)

    logger.info("No true frontier in this room")
    inner_nodes = [
        node
        for node in room_node.nodes
        if (
            node.has_frontier is True
            and node.state == 0
            and node.has_true_frontier is False
            and node.frontier_idxs.shape[0] > mapper.frontier_thres
        )
    ]
    if inner_nodes:
        return find_closest_nodes(mapper, inner_nodes)

    # 房间内既没有跨房间 true frontier，也没有足够大的内部 frontier；
    # 将该 room 标记为已探索，避免 relocation 反复选回来。
    logger.info("No inner nodes in this room")
    room_node.state = 1
    for node in room_node.nodes:
        node.has_frontier = False
        node.has_true_frontier = False
    return None


def explore_after_check(mapper: Any) -> Any | None:
    """Continue exploration after a rejected check-again candidate."""

    nodes_true_frontier = [
        node
        for node in mapper.nodes
        if node.state == 0 and node.has_frontier is True
    ]
    return find_closest_nodes(mapper, nodes_true_frontier)


def explore_after_fully_explored(mapper: Any) -> Any | None:
    """Fallback to the nearest unexplored node when room-level policy is exhausted."""

    nodes = [node for node in mapper.nodes if node.state == 0]
    return find_closest_nodes(mapper, nodes)


def _reset_process_pointclouds(mapper: Any) -> None:
    # exploration policy 只需要清理临时点云缓存；具体点云更新仍由 mapper 完成。
    mapper.process_obs_pcd = o3d.t.geometry.PointCloud(mapper.pcd_device)
    mapper.process_nav_pcd = o3d.t.geometry.PointCloud(mapper.pcd_device)
