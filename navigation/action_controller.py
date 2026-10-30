"""Habitat action-control helpers for VLN agent."""

from __future__ import annotations

from typing import Any

import numpy as np


def habitat_waypoint(agent: Any, waypoint: np.ndarray | None = None, *, frame_id="mapper_local") -> np.ndarray:
    """Convert the declared waypoint frame to Habitat coordinates.

    Historical map positions use (x, -z, y) relative to Habitat world, so
    p_habitat = (p_map.x, p_map.z, -p_map.y). Their height is a base position,
    whereas legacy mapper waypoints intentionally use the current floor height.
    """

    local_waypoint = agent.waypoint if waypoint is None else waypoint
    if frame_id == "map_z_up":
        x, y, z = local_waypoint
        return np.array([x, z, -y], dtype=float)
    if frame_id != "mapper_local":
        raise ValueError(f"Unsupported Habitat waypoint frame: {frame_id}")
    pid_waypoint = local_waypoint + agent.mapper.initial_position
    return np.array([
        pid_waypoint[0],
        agent.env.sim.get_agent_state().position[1],
        pid_waypoint[1],
    ])


def current_habitat_position(agent: Any) -> np.ndarray:
    """Return current agent position in Habitat simulator coordinates."""

    current_position = agent.mapper.current_position + agent.mapper.initial_position
    return np.array([
        current_position[0],
        agent.mapper.initial_position[2] - 0.88,
        current_position[1],
    ])


def geodesic_distance_to_waypoint(agent: Any, waypoint: np.ndarray | None = None, *, frame_id="mapper_local") -> float:
    """Compute Habitat geodesic distance from current pose to a waypoint."""

    return agent.env.sim.geodesic_distance(
        (agent.env.sim.get_agent_state().position if frame_id == "map_z_up"
         else current_habitat_position(agent)),
        habitat_waypoint(agent, waypoint, frame_id=frame_id),
    )


def next_action_to_waypoint(agent: Any, waypoint: np.ndarray | None = None, *, frame_id="mapper_local") -> int:
    """Ask the low-level planner for the next Habitat action toward a waypoint."""

    # planner 是底层运动控制器；这里不做语义判断，只做坐标转换。
    return agent.planner.get_next_action(habitat_waypoint(agent, waypoint, frame_id=frame_id))
