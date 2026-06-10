"""Planner-loop helpers for STRIVE agent.

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


def run_observation_mapping_cycle(agent: Any, *, node: Any, episode_idx: int | None) -> PlanningCycleResult:
    """Run panoramic perception, point-cloud merge, node extraction, and object update.

    输入是 agent 当前状态；输出只暴露 step 和 episode_over。目标选择、
    room relocation、stop 判断都留在调用方，避免把策略状态机过早搬进工具层。
    """

    agent.rotate_panoramic()
    if agent.env.episode_over:
        return PlanningCycleResult(step=agent.episode_steps, episode_over=True)

    agent.current_pcd = agent._merge_temporary_pointclouds()
    step = agent.episode_steps
    agent._save_obs_pointcloud(agent.current_pcd, idx=episode_idx, step=step)
    agent._log_mapper_state_before_after_get_nodes(step, node, episode_idx)
    agent.mapper.update_obj(agent.current_node_idx, agent.mapper.current_obj_indices)
    return PlanningCycleResult(step=step, episode_over=False)
