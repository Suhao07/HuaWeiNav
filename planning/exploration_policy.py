"""Frontier exploration policies for VLN navigation.

These helpers choose the next viewpoint node from the mapper graph. They do not
perform perception, room segmentation, target verification, or Habitat actions.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
try:
    from loguru import logger
except Exception:  # pragma: no cover - only used in minimal unit-test envs.
    class _FallbackLogger:
        """Minimal logger used when loguru is unavailable."""

        def info(self, *args: Any, **kwargs: Any) -> None:
            """Ignore info logs."""

        def warning(self, *args: Any, **kwargs: Any) -> None:
            """Ignore warning logs."""

    logger = _FallbackLogger()


def find_closest_nodes(mapper: Any, nodes: list[Any]) -> Any | None:
    """Pick the next node with optional prior-map frontier bias.

    Args:
        mapper: Mapper-like runtime with Habitat simulator and optional
            prior-map query result.
        nodes: Candidate frontier/viewpoint nodes.

    Returns:
        Selected node, or ``None`` when no candidates are available.
    """

    if not nodes:
        return None
    distance = _geodesic_distances_to_nodes(mapper, nodes)
    baseline_idx = int(np.argmin(distance))
    prior_scores = np.array([_frontier_prior_score(mapper, node) for node in nodes], dtype=float)
    bias_m = _prior_frontier_distance_bias_m(mapper)
    prior_enabled = _prior_frontier_bias_enabled(mapper) and bool(np.any(prior_scores > 0.0))
    adjusted_distance = distance - bias_m * prior_scores if prior_enabled else distance.copy()
    selected_idx = int(np.argmin(adjusted_distance))
    if getattr(mapper, "coverage_guided_exploration", False):
        selected_idx = select_coverage_frontier(nodes, adjusted_distance, mapper.frontier_distance_tie_m)
    _record_frontier_choice(
        mapper=mapper,
        nodes=nodes,
        raw_distances=distance,
        prior_scores=prior_scores,
        adjusted_distances=adjusted_distance,
        baseline_idx=baseline_idx,
        selected_idx=selected_idx,
        prior_enabled=prior_enabled,
        distance_bias_m=bias_m,
    )
    return nodes[selected_idx]


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
    o3d = _open3d()
    mapper.process_obs_pcd = o3d.t.geometry.PointCloud(mapper.pcd_device)
    mapper.process_nav_pcd = o3d.t.geometry.PointCloud(mapper.pcd_device)


def _geodesic_distances_to_nodes(mapper: Any, nodes: list[Any]) -> np.ndarray:
    """Return geodesic distances from the current pose to candidate nodes.

    Args:
        mapper: Mapper-like runtime.
        nodes: Candidate nodes.

    Returns:
        Distance array aligned with ``nodes``.
    """

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
    return np.array([
        float(mapper.env.sim.geodesic_distance(current_position, node_position))
        for node_position in nodes_positions
    ])


def _prior_frontier_bias_enabled(mapper: Any) -> bool:
    """Return whether frontier bias may affect active planning."""

    adapter = getattr(mapper, "prior_map_policy_adapter", None)
    if adapter is not None and not bool(getattr(adapter, "enabled", True)):
        return False
    prior_result = getattr(mapper, "search_prior_result", None)
    return bool(getattr(prior_result, "frontier_biases", ()) or ())


def _frontier_prior_score(mapper: Any, node: Any) -> float:
    """Return the soft prior score for a candidate node.

    Args:
        mapper: Mapper-like runtime.
        node: Candidate frontier node.

    Returns:
        Maximum matching ``FrontierPrior.score_delta``.
    """

    prior_result = getattr(mapper, "search_prior_result", None)
    priors = tuple(getattr(prior_result, "frontier_biases", ()) or ())
    if not priors:
        return 0.0
    node_uid = _node_uid(node)
    room_terms = _node_room_terms(mapper, node)
    best = 0.0
    for prior in priors:
        prior_uid = _norm(getattr(prior, "frontier_uid", ""))
        direct_match = prior_uid and prior_uid == _norm(node_uid)
        prior_rooms = {
            _norm(getattr(prior, "prior_room_uid", "")),
            _norm(getattr(prior, "target_region_uid", "")),
        }
        room_match = bool(room_terms & {term for term in prior_rooms if term})
        if direct_match or room_match:
            best = max(best, float(getattr(prior, "score_delta", 0.0) or 0.0))
    return best


def _node_room_terms(mapper: Any, node: Any) -> set[str]:
    """Return normalized room identifiers attached to a node."""

    terms = {
        _norm(getattr(node, "room_uid", "")),
        _norm(getattr(node, "room_id", "")),
        _norm(getattr(node, "prior_room_uid", "")),
        _norm(getattr(node, "target_region_uid", "")),
        _norm(getattr(node, "room_idx", "")),
    }
    try:
        room = mapper.room_nodes[int(getattr(node, "room_idx"))]
    except Exception:
        room = None
    if room is not None:
        terms.update(
            {
                _norm(getattr(room, "uid", "")),
                _norm(getattr(room, "room_uid", "")),
                _norm(getattr(room, "label", "")),
                _norm(getattr(room, "name", "")),
                _norm(getattr(room, "idx", "")),
                _norm(getattr(room, "room_id", "")),
            }
        )
    return {term for term in terms if term}


def _prior_frontier_distance_bias_m(mapper: Any) -> float:
    """Return meters of distance discount per frontier prior score."""

    value = getattr(mapper, "prior_map_frontier_distance_bias_m", None)
    if value is None:
        value = os.getenv("STRIVE_PRIOR_MAP_FRONTIER_DISTANCE_BIAS_M", "1.0")
    try:
        return max(0.0, float(value))
    except Exception:
        return 1.0


def _record_frontier_choice(
    *,
    mapper: Any,
    nodes: list[Any],
    raw_distances: np.ndarray,
    prior_scores: np.ndarray,
    adjusted_distances: np.ndarray,
    baseline_idx: int,
    selected_idx: int,
    prior_enabled: bool,
    distance_bias_m: float,
) -> None:
    """Record before/after frontier ordering for prior-map diagnostics."""

    candidates = []
    for index, node in enumerate(nodes):
        candidates.append(
            {
                "rank_input": int(index),
                "frontier_uid": _node_uid(node),
                "node_idx": _safe_int(getattr(node, "idx", index)),
                "room_idx": _safe_int(getattr(node, "room_idx", -1)),
                "position": _position_payload(getattr(node, "position", None)),
                "raw_distance_m": float(raw_distances[index]),
                "prior_score": float(prior_scores[index]),
                "adjusted_distance_m": float(adjusted_distances[index]),
            }
        )
    payload = {
        "step": _safe_int(getattr(mapper, "prior_map_current_step", -1)),
        "authority": "ranking_only",
        "policy": "find_closest_nodes",
        "prior_enabled": bool(prior_enabled),
        "distance_bias_m": float(distance_bias_m),
        "baseline_selected_uid": _node_uid(nodes[baseline_idx]),
        "selected_uid": _node_uid(nodes[selected_idx]),
        "baseline_selected_index": int(baseline_idx),
        "selected_index": int(selected_idx),
        "prior_changed_selection": bool(selected_idx != baseline_idx),
        "candidates": candidates,
    }
    setattr(mapper, "prior_map_last_chosen_frontier", payload)
    runtime = getattr(mapper, "prior_map_runtime", None)
    if runtime is not None and hasattr(runtime, "record_chosen_frontier"):
        runtime.record_chosen_frontier(payload, step=payload["step"])


def _node_uid(node: Any) -> str:
    """Return the stable frontier uid used by query/adapters."""

    for name in ("frontier_uid", "uid", "id", "idx"):
        value = getattr(node, name, None)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown_frontier"


def _position_payload(value: Any) -> list[float]:
    """Convert a numpy-like position to a JSON-friendly list."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return []


def _safe_int(value: Any) -> int:
    """Return an integer fallback for debug payloads."""

    try:
        return int(value)
    except Exception:
        return -1


def _norm(value: Any) -> str:
    """Normalize identifiers for prior/frontier matching."""

    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _open3d() -> Any:
    """Import Open3D lazily for point-cloud reset paths."""

    import open3d as o3d  # type: ignore

    return o3d


def select_exploration_node(agent, *, episode_idx, step, relocate=None, use_gpt_relocate=None):
    """Select the original room-first exploration goal for either runtime.

    With priors disabled, n* = argmin d_nav(x, n) within the current room.
    Room relocation runs only after its local policy returns no candidate.
    This cycle operation retains the legacy buffer reset; callers invoke it
    once per planning cycle, never as a read-only model tool.
    """
    mapper = agent.mapper
    relocate = agent.relocate if relocate is None else relocate
    use_gpt_relocate = agent.gpt_relocate if use_gpt_relocate is None else use_gpt_relocate
    room = mapper.room_nodes[mapper.nodes[agent.current_node_idx].room_idx]
    node = (mapper.explore_in_room_relocate(room) if relocate else mapper.explore_in_room(room))
    if node is not None:
        return node, "current_room"
    if all(room.state != 0 for room in mapper.room_nodes):
        return mapper.explore_after_fully_explored(), "remaining_nodes"
    if relocate or use_gpt_relocate:
        room = mapper.get_candidate_room_fully_explored(agent.instruct_goal, idx=episode_idx, step=step)
        source = "room_semantic_policy"
    else:
        room = mapper.get_candidate_room_fully_explored_by_distance(
            agent.instruct_goal, idx=episode_idx, step=step, plan=agent.task.plan)
        source = "room_distance_policy"
    return (mapper.find_closet_viewpoint_in_room(room) if room is not None else None), source


def select_coverage_frontier(nodes, distances, tie_distance):
    """Deduplicate identical frontier cell sets and rank near-cost options by coverage.

    C = {n: d(n) <= min(d) + delta}; n* = argmax_C |F(n)|.
    delta 使用一个仿真前进步长；代价相近时才以覆盖收益改变原排序。
    同一 frontier cell 集合保留最近节点，稳定 ID 打破完全并列。
    """
    representatives = {}
    for i, node in enumerate(nodes):
        if not np.isfinite(distances[i]):
            continue
        cells = frozenset(tuple(int(v) for v in cell) for cell in node.frontier_idxs)
        key = cells if cells else ("node", node.idx)
        old = representatives.get(key)
        if old is None or (distances[i], node.idx) < (distances[old], nodes[old].idx):
            representatives[key] = i
    if not representatives:
        return int(np.argmin(distances))
    minimum = min(distances[i] for i in representatives.values())
    choices = [i for i in representatives.values() if distances[i] <= minimum + tie_distance]
    return min(choices, key=lambda i: (-len(set(map(tuple, nodes[i].frontier_idxs))), distances[i], nodes[i].idx))
