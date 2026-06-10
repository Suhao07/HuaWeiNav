"""Habitat action-control helpers for STRIVE agent."""

from __future__ import annotations

from typing import Any

import numpy as np


def habitat_waypoint(agent: Any, waypoint: np.ndarray | None = None) -> np.ndarray:
    """Convert a mapper-local waypoint into Habitat simulator coordinates."""

    local_waypoint = agent.waypoint if waypoint is None else waypoint
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


def geodesic_distance_to_waypoint(agent: Any, waypoint: np.ndarray | None = None) -> float:
    """Compute Habitat geodesic distance from current pose to a waypoint."""

    return agent.env.sim.geodesic_distance(
        current_habitat_position(agent),
        habitat_waypoint(agent, waypoint),
    )


def next_action_to_waypoint(agent: Any, waypoint: np.ndarray | None = None) -> int:
    """Ask the low-level planner for the next Habitat action toward a waypoint."""

    # planner 是底层运动控制器；这里不做语义判断，只做坐标转换。
    return agent.planner.get_next_action(habitat_waypoint(agent, waypoint))
