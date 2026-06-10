"""Topology graph operations for STRIVE mapper runtime.

The mapper still owns graph state for compatibility. This module provides the
operations that mutate or query that state, keeping node/object graph mechanics
out of the mapper orchestration layer.
"""

from __future__ import annotations

from typing import Any

import heapq

import numpy as np
from loguru import logger

from mapping_utils.representation import NodeState, our_Node


def add_node(
    mapper: Any,
    position: np.ndarray,
    pcd: Any = None,
    has_frontier: bool = False,
    frontier_idxs: np.ndarray = np.array([]),
    block_current: bool = False,
) -> tuple[int, bool]:
    """Add a viewpoint node unless an existing nearby node should absorb it."""

    logger.info("---------- Checking ---------- {}", position)
    if len(mapper.nodes) > 1:
        block_node = _blocking_node(mapper, position, block_current=block_current)
        if block_node is not None:
            _merge_frontier_into_blocking_node(
                mapper,
                block_node=block_node,
                position=position,
                pcd=pcd,
                has_frontier=has_frontier,
                frontier_idxs=frontier_idxs,
            )
            return mapper.node_cnt - 1, False

    logger.info("---------- adding node ---------- {}", position)
    position = position.copy()
    node = our_Node(None, None, pcd, position, None, mapper.node_cnt, has_frontier, frontier_idxs.astype(int))
    mapper.nodes.append(node)
    mapper.node_cnt += 1
    mapper.neighbors.append([])
    mapper.nodes_pos_to_idx[tuple(position)] = mapper.node_cnt - 1

    if has_frontier:
        frontier_idxs = frontier_idxs.astype(int)
        mapper.frontiers_considered[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1
    return mapper.node_cnt - 1, True


def visit_node(mapper: Any, node_idx: int) -> None:
    """Mark a viewpoint node as explored and clear its frontier flags."""

    node_pos = mapper.nodes[node_idx].position
    logger.info(f"---------- visiting node ---------- {node_idx} at {node_pos}")
    for key, value in mapper.nodes_pos_to_idx.items():
        logger.info("{} {}", key, value)
    mapper.nodes[node_idx].state = NodeState.EXPLORED
    mapper.nodes[node_idx].has_frontier = False
    mapper.nodes[node_idx].has_true_frontier = False


def update_node_frontier(mapper: Any) -> None:
    """Refresh node frontier flags from the latest room/frontier map labels."""

    for node in mapper.nodes:
        if node.has_frontier:
            frontier_idxs = np.array(node.frontier_idxs).astype(int)
            frontier_idxs = [idx for idx in frontier_idxs if mapper.grid_map[idx[0], idx[1]] != 0]
            if len(frontier_idxs) == 0:
                node.has_frontier = False
                node.has_true_frontier = False
                node.frontier_idxs = np.array(frontier_idxs).reshape((-1, 2))
            else:
                node.has_frontier = True
                node.frontier_idxs = np.array(frontier_idxs).reshape((-1, 2))
        else:
            node.has_frontier = False
            node.has_true_frontier = False
            node.frontier_idxs = np.array([]).reshape((-1, 2))


def update_node_true_frontier(mapper: Any) -> None:
    """Mark frontier nodes that lead outside the current room region."""

    for node in mapper.nodes:
        if node.has_frontier:
            frontier_idxs = np.array(node.frontier_idxs).astype(int)
            frontier_idxs = [idx for idx in frontier_idxs if mapper.grid_map[idx[0], idx[1]] == 2]
            node.has_true_frontier = len(frontier_idxs) > 0


def update_room_state(mapper: Any) -> None:
    """Refresh each room state from its node frontier flags."""

    for room_node in mapper.room_nodes:
        room_node.update_state()


def add_edge(mapper: Any, node1_idx: int, node2_idx: int) -> None:
    mapper.neighbors[node1_idx].append(node2_idx)
    mapper.neighbors[node2_idx].append(node1_idx)


def remove_edge(mapper: Any, node1_idx: int, node2_idx: int) -> None:
    mapper.neighbors[node1_idx].remove(node2_idx)
    mapper.neighbors[node2_idx].remove(node1_idx)


def get_edges(mapper: Any) -> list[tuple[int, int]]:
    edges = []
    for i in range(mapper.node_cnt):
        for j in mapper.neighbors[i]:
            if i < j:
                edges.append((i, j))
    return edges


def update_obj(mapper: Any, current_node_idx: int, obj_indices: list[int]) -> None:
    """Update object-node edges around the current viewpoint node."""

    dis_thres = 2.5
    node = mapper.nodes[current_node_idx]
    for ind in obj_indices:
        if ind in node.objects:
            continue

        nwdist = np.linalg.norm(mapper.objects[ind].position[:2] - node.position[:2])
        if nwdist < dis_thres:
            need_del = []
            for other_obj in mapper.objects[ind].nodes:
                dis = np.linalg.norm(mapper.nodes[other_obj].position[:2] - mapper.objects[ind].position[:2])
                if dis >= dis_thres:
                    need_del.append(other_obj)

            for other_obj in need_del:
                mapper.nodes[other_obj].objects.remove(ind)
                mapper.objects[ind].nodes.remove(other_obj)

            mapper.objects[ind].nodes.append(node.idx)
            node.objects.append(ind)
        elif len(mapper.objects[ind].nodes) == 0:
            mapper.objects[ind].nodes.append(node.idx)
            node.objects.append(ind)


def get_nodes_positions(mapper: Any) -> np.ndarray:
    return np.array([node.position for node in mapper.nodes])


def get_nodes_states(mapper: Any) -> np.ndarray:
    return np.array([node.state for node in mapper.nodes])


def find_closest_node(mapper: Any, position: np.ndarray) -> Any:
    dist = np.array([np.linalg.norm(node.position[:2] - position[:2]) for node in mapper.nodes])
    return mapper.nodes[np.argmin(dist)]


def find_closest_unexplored_node(mapper: Any, node_pos: Any = None) -> Any | None:
    """Find the closest unexplored node by graph distance from current node."""

    node_idx = mapper.nodes[mapper.current_node_idx].idx
    dist = np.full(mapper.node_cnt, np.inf)
    dist[node_idx] = 0
    pq = [(0, node_idx)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if mapper.nodes[u].state == NodeState.UNEXPLORED:
            return mapper.nodes[u]
        for v in mapper.neighbors[u]:
            alt = current_dist + np.linalg.norm(mapper.nodes[u].position[:2] - mapper.nodes[v].position[:2])
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return None


def find_the_closest_path(mapper: Any, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Run Dijkstra over viewpoint graph and return node positions and indices."""

    start_node = find_closest_node(mapper, start)
    end_node = find_closest_node(mapper, end)
    start_idx = start_node.idx
    end_idx = end_node.idx

    dist = np.full(mapper.node_cnt, np.inf)
    prev = np.full(mapper.node_cnt, -1, dtype=int)
    dist[start_idx] = 0
    pq = [(0, start_idx)]
    while pq:
        current_dist, u_idx = heapq.heappop(pq)
        if u_idx == end_idx:
            break
        if current_dist > dist[u_idx]:
            continue
        for v_idx in mapper.neighbors[u_idx]:
            alt = dist[u_idx] + np.linalg.norm(mapper.nodes[u_idx].position[:2] - mapper.nodes[v_idx].position[:2])
            if alt < dist[v_idx]:
                dist[v_idx] = alt
                prev[v_idx] = u_idx
                heapq.heappush(pq, (alt, v_idx))

    path_node_position = []
    path_node_idx = []
    u_idx = end_idx
    while u_idx != -1:
        path_node_position.append(mapper.nodes[u_idx].position)
        path_node_idx.append(u_idx)
        u_idx = prev[u_idx]
    path_node_position.reverse()
    path_node_idx.reverse()
    if not path_node_position or tuple(path_node_position[0]) != tuple(start_node.position):
        return np.array([]), []
    return np.array(path_node_position), path_node_idx


def check_connected(mapper: Any, start: int, end: int) -> bool:
    """Return whether two graph node indices are connected."""

    start_idx = mapper.nodes[start].idx
    end_idx = mapper.nodes[end].idx
    dist = np.full(mapper.node_cnt, np.inf)
    dist[start_idx] = 0
    pq = [(0, start_idx)]
    while pq:
        current_dist, u_idx = heapq.heappop(pq)
        if u_idx == end_idx:
            return True
        if current_dist > dist[u_idx]:
            continue
        for v_idx in mapper.neighbors[u_idx]:
            alt = dist[u_idx] + np.linalg.norm(mapper.nodes[u_idx].position[:2] - mapper.nodes[v_idx].position[:2])
            if alt < dist[v_idx]:
                dist[v_idx] = alt
                heapq.heappush(pq, (alt, v_idx))
    return False


def _blocking_node(mapper: Any, position: np.ndarray, *, block_current: bool) -> Any | None:
    current_node_idx = mapper.nodes[mapper.current_node_idx].idx
    if block_current:
        nodes = [node for node in mapper.nodes if node.idx != current_node_idx]
    else:
        nodes = list(mapper.nodes)
    if not nodes:
        return None
    nodes_positions = np.array([node.position for node in nodes], dtype=float)
    nodes_positions = nodes_positions + mapper.initial_position
    nodes_positions[:, 2] = mapper.initial_position[2] - 0.88
    nodes_positions = nodes_positions[:, [0, 2, 1]]
    current_node_position = position + mapper.initial_position
    current_node_position = np.array([
        current_node_position[0],
        mapper.initial_position[2] - 0.88,
        current_node_position[1],
    ])
    distances = [
        mapper.env.sim.geodesic_distance(current_node_position, node_position)
        for node_position in nodes_positions
    ]
    min_idx = int(np.argmin(distances))
    if distances[min_idx] < 1.35 * 0.7:
        return nodes[min_idx]
    return None


def _merge_frontier_into_blocking_node(
    mapper: Any,
    *,
    block_node: Any,
    position: np.ndarray,
    pcd: Any,
    has_frontier: bool,
    frontier_idxs: np.ndarray,
) -> None:
    logger.info(f"Node at {block_node.position} is blocking adding new node at {position}")
    current_node_idx = mapper.nodes[mapper.current_node_idx].idx
    if not (has_frontier and block_node.idx != current_node_idx and block_node.state == NodeState.UNEXPLORED):
        return

    logger.info(f"Updating frontier of node at {position} to node at {block_node.position}")
    had_frontier = bool(block_node.has_frontier)
    frontier_idxs = frontier_idxs.astype(int)
    block_node.frontier_idxs = np.concatenate((block_node.frontier_idxs, frontier_idxs), axis=0)
    block_node.has_frontier = True
    block_node.state = NodeState.UNEXPLORED
    if not had_frontier:
        # 保留旧逻辑语义：没有 frontier 的 blocking node 会继承新 node 的位置和点云。
        logger.info(f"Node at {block_node.position} has no frontier, move it to the new node at {position}")
        block_node.position = position
        block_node.pcd = pcd
    mapper.frontiers_considered[frontier_idxs[:, 0], frontier_idxs[:, 1]] = 1
