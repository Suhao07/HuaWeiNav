from __future__ import annotations

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
    best_idx = int(np.argmin(distances))
    closest_node = frontier_nodes[best_idx]
    room = mapper.room_nodes[closest_node.room_idx]
    return RoomSelection(
        room=room,
        closest_node_idx=closest_node.idx,
        distances=distances,
        reason=(
            f"Node {closest_node.idx} in Room {closest_node.room_idx} is the closest frontier "
            f"from current position."
        ),
    )
