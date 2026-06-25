from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RoomSelection:
    room: Any | None
    reason: str
    closest_node_idx: int | None = None
    distances: list[float] | None = None


def select_nearest_frontier_room(mapper: Any) -> RoomSelection:
    """Choose the unexplored room whose frontier node is geodesically closest.

    This is the deterministic fallback for relocation when LLM room selection is
    disabled. It is a planning policy, not a semantic rule: it does not inspect
    target categories or instruction text.
    """

    frontier_nodes = [node for node in mapper.nodes if node.has_frontier == 1]
    if not frontier_nodes:
        return RoomSelection(room=None, reason="No frontier nodes are available.")

    nodes_positions = np.array([node.position for node in frontier_nodes], dtype=float)
    nodes_positions = nodes_positions + mapper.initial_position
    nodes_positions[:, 2] = mapper.initial_position[2] - 0.88
    nodes_positions = nodes_positions[:, [0, 2, 1]]

    current_node_position = mapper.current_position + mapper.initial_position
    current_node_position = np.array(
        [current_node_position[0], mapper.initial_position[2] - 0.88, current_node_position[1]],
        dtype=float,
    )
    distances = [
        float(mapper.env.sim.geodesic_distance(current_node_position, node_position))
        for node_position in nodes_positions
    ]
    prior_scores = [_prior_room_score_for_node(mapper, node) for node in frontier_nodes]
    bias_m = _prior_room_distance_bias_m(mapper)
    adjusted_distances = [
        distance - bias_m * prior_score
        for distance, prior_score in zip(distances, prior_scores)
    ]
    best_idx = int(np.argmin(adjusted_distances))
    closest_node = frontier_nodes[best_idx]
    room = mapper.room_nodes[closest_node.room_idx]
    prior_suffix = ""
    if prior_scores[best_idx] > 0.0:
        prior_suffix = (
            f" Prior map room score={prior_scores[best_idx]:.3f}, "
            f"distance_bias_m={bias_m:.3f}, adjusted_distance={adjusted_distances[best_idx]:.3f}."
        )
    return RoomSelection(
        room=room,
        closest_node_idx=closest_node.idx,
        distances=distances,
        reason=(
            f"Node {closest_node.idx} in Room {closest_node.room_idx} is the closest frontier "
            f"from current position.{prior_suffix}"
        ),
    )


def _prior_room_score_for_node(mapper: Any, node: Any) -> float:
    prior_result = getattr(mapper, "search_prior_result", None)
    if prior_result is None:
        return 0.0
    try:
        room = mapper.room_nodes[node.room_idx]
    except Exception:
        return 0.0
    room_terms = {
        _norm(getattr(room, "uid", "")),
        _norm(getattr(room, "room_uid", "")),
        _norm(getattr(room, "label", "")),
        _norm(getattr(room, "tag", "")),
        _norm(getattr(room, "name", "")),
        _norm(getattr(room, "idx", "")),
        _norm(getattr(node, "room_idx", "")),
    }
    best = 0.0
    for prior in getattr(prior_result, "room_rankings", ()) or ():
        prior_terms = {_norm(getattr(prior, "room_uid", "")), _norm(getattr(prior, "label", ""))}
        if room_terms & prior_terms:
            best = max(best, float(getattr(prior, "score", 0.0) or 0.0))
    return best


def _prior_room_distance_bias_m(mapper: Any) -> float:
    value = getattr(mapper, "prior_map_room_distance_bias_m", None)
    if value is None:
        value = os.getenv("STRIVE_PRIOR_MAP_ROOM_DISTANCE_BIAS_M", "1.0")
    try:
        return max(0.0, float(value))
    except Exception:
        return 1.0


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")
