"""Path-following helpers for STRIVE step loop.

The functions here advance the current waypoint/path and trigger replanning
when a path is exhausted. They do not run visual verification or decide final
instruction success.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from navigation.action_controller import habitat_waypoint, next_action_to_waypoint


def action_after_replan(agent: Any, *, episode_idx: int) -> tuple[int, bool]:
    """Return the next action while following the current path.

    The boolean flag indicates whether the caller should continue stepping the
    environment. A false value preserves the legacy early-return behavior when
    no further planning target exists.
    """

    act = next_action_to_waypoint(agent)
    while act == 0 and not agent.found_goal:
        agent.path_index += 1
        if agent.path_index >= len(agent.path):
            # 当前 path 已经走完，交回高层 planner 选择新的 exploration/relocation node。
            if agent.relocate:
                flag, agent.found_goal = agent.make_plan_mod_relocate(
                    rotate=True,
                    idx=episode_idx,
                    node=agent.waypoint_final,
                )
            else:
                flag, agent.found_goal = agent.make_plan_mod_no_relocate(
                    rotate=True,
                    idx=episode_idx,
                    node=agent.waypoint_final,
                    use_gpt_relocate=agent.gpt_relocate,
                )
            if agent.env.episode_over:
                return act, False

            if not flag and not agent.env.episode_over:
                agent.obs = agent.env.step(act)
                agent.update_trajectory(agent.on_node_flag)
                logger.info(agent.env.episode_over)
                return act, False
        else:
            logger.info('!!!!!!!!!!!!!!!!Bug: enter make_plan_mod_process')
            agent.obs = agent.env.step(0)
            agent.update_trajectory(agent.on_node_flag)
            logger.info(agent.env.episode_over)
            return 0, False

        agent.waypoint = agent.path[agent.path_index]
        agent.waypoint[2] = agent.mapper.current_position[2]
        logger.info(f'Waypoint: {agent.waypoint}')
        _ = habitat_waypoint(agent)
        act = next_action_to_waypoint(agent)

    if act == 1:
        agent.on_node_flag = False
    return act, True
