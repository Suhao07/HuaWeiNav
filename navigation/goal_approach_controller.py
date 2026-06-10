"""Goal-approach helpers for STRIVE target verification.

This module handles geometric movement around an already selected target
candidate. It does not run the final instruction verifier; the agent remains
the owner of semantic accept/reject decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from mapping_utils.transform import habitat_rotation
from navigation.action_controller import habitat_waypoint, next_action_to_waypoint


def distance_to_object(agent: Any) -> float:
    """Compute the closest 2D distance from the robot to the current object."""

    positions = agent.object_final.pcd.point.positions.cpu().numpy()
    return float(np.min(np.linalg.norm(positions[:, :2] - agent.mapper.current_position[:2], axis=1)))


def rotate_toward_object_for_recheck(agent: Any) -> bool:
    """Rotate the robot toward the selected object before bbox re-verification."""

    final_pos = agent.object_final.position[:2]
    nw_pos = agent.mapper.current_position[:2]
    nw_ori = habitat_rotation(agent.rotation)
    nw_ori = np.array([-nw_ori[0, 2], -nw_ori[1, 2]])
    final_pos = final_pos - nw_pos

    nw_ori = nw_ori / np.linalg.norm(nw_ori)
    final_pos = final_pos / np.linalg.norm(final_pos)
    dot_product = np.clip(np.dot(nw_ori, final_pos), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_product))
    direction = 3 if np.cross(nw_ori, final_pos) > 0 else 2
    num_turns = int(np.round(angle_deg / 30))

    logger.info("Now step: {}", agent.episode_steps)
    logger.info("Direction: {}", direction)
    logger.info("Num turns: {}", num_turns)

    # 复核前只做最小必要转向，避免为了确认目标额外触发规划状态变化。
    for _ in range(num_turns):
        agent.obs = agent.env.step(direction)
        agent.update_trajectory(agent.on_node_flag)
        if agent.env.episode_over:
            return False
    return True


def action_after_instruction_reject(agent: Any) -> int:
    """Recover from a failed final-instruction check and return the next action."""

    _ = habitat_waypoint(agent)
    if agent.need_check_again:
        act = next_action_to_waypoint(agent)
        if act == 0:
            agent.need_check_again = False
            agent.after_check_again()
            _ = habitat_waypoint(agent)
    else:
        agent.after_check_again()
        _ = habitat_waypoint(agent)
    return next_action_to_waypoint(agent)
