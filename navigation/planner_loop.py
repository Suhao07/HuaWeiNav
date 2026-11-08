"""Planner-loop helpers for VLN agent.

This module owns the repeated "sense -> update map" prefix of one planning
cycle. The high-level exploration/relocation state machine stays in the agent
for now because it changes many runtime fields and is still the riskiest part
of the navigation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanningCycleResult:
    """Result produced after the observation/map-update prefix of a cycle."""

    step: int
    episode_over: bool = False
    geometry_available: bool = True


def run_observation_mapping_cycle(agent: Any, *, node: Any, episode_idx: int | None) -> PlanningCycleResult:
    """Run the selected observation mode, then update map nodes and objects."""

    agent.mapper.current_node_idx = agent.current_node_idx
    single_view = agent.observation_mode == "single" and not agent.patrol
    if single_view:
        from navigation.observation_pipeline import collect_single_observation
        collect_single_observation(agent, node=node)
    elif getattr(agent, "adaptive_observation", False):
        from navigation.observation_pipeline import collect_adaptive_observations
        collect_adaptive_observations(agent)
        if not agent.temporary_pcd:
            return PlanningCycleResult(step=agent.episode_steps, episode_over=agent.env.episode_over)
    else:
        agent.rotate_panoramic()
    if agent.env.episode_over:
        return PlanningCycleResult(step=agent.episode_steps, episode_over=True)

    if single_view and (agent.mapper.traversable_pcd_all.is_empty()
                        or agent.mapper.process_nav_pcd.is_empty()):
        return PlanningCycleResult(step=agent.episode_steps, geometry_available=False)

    agent.current_pcd = agent._merge_temporary_pointclouds()
    step = agent.episode_steps
    agent._save_obs_pointcloud(agent.current_pcd, idx=episode_idx, step=step)
    agent._log_mapper_state_before_after_get_nodes(step, node, episode_idx)
    agent.mapper.update_obj(agent.current_node_idx, agent.mapper.current_obj_indices)
    return PlanningCycleResult(step=step, episode_over=False)
