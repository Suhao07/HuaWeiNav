"""View verification and check-again viewpoint control for STRIVE agent."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from instruction_adapter.verifier import candidate_from_object
from planning.viewpoint_policy import build_check_again_viewpoints


def select_check_again_viewpoint(agent: Any) -> None:
    """Select the next check-again or view-control viewpoint for the agent."""

    logger.info("Need to check again")
    agent.need_check_again = True

    pathfinder = agent.env.sim.pathfinder
    import habitat_sim

    path_request = habitat_sim.ShortestPath()
    current_position = agent.found_goal_position + agent.mapper.initial_position
    current_position = np.array([
        current_position[0],
        agent.env.sim.get_agent_state().position[1],
        current_position[1],
    ])
    path_request.requested_start = current_position
    pid_waypoint = agent.waypoint + agent.mapper.initial_position
    pid_waypoint = np.array([
        pid_waypoint[0],
        agent.env.sim.get_agent_state().position[1],
        pid_waypoint[1],
    ])
    path_request.requested_end = pid_waypoint

    pathfinder.find_path(path_request)
    points = path_request.points
    logger.info(f"Path: {points}")
    proposals, interpolated_path = build_check_again_viewpoints(
        object_node=agent.object_final,
        camera_intrinsic=agent.mapper.camera_intrinsic,
        mapper_initial_position=agent.mapper.initial_position,
        habitat_path_points=points,
        target_height=agent.found_goal_position[2] + 0.88,
        success_distance=agent.success_distance,
        stop_criterion=agent.stop_criterion,
    )
    logger.info(f"Interpolated_path: {interpolated_path}")

    best_candidate = _log_and_pick_best_proposal(proposals)
    current_candidate = candidate_from_object(
        agent.object_final,
        canonical_label=getattr(agent.mapper, "target", ""),
        step=agent.episode_steps,
    )
    if _try_view_control_proposal(agent, proposals, current_candidate):
        return
    if best_candidate is not None:
        logger.info(
            "Selected check-again viewpoint: {}, score {:.3f}, visible {:.3f}, center {:.3f}, border {:.3f}, area {:.3f}, dist {:.3f}",
            best_candidate["position"],
            best_candidate["score"],
            best_candidate["visible_ratio"],
            best_candidate["center_score"],
            best_candidate["border_score"],
            best_candidate["area_ratio"],
            best_candidate["distance_to_target"],
        )
        agent.check_again_postion = best_candidate["position"]
    elif len(interpolated_path) > 0:
        logger.info("Can't score a good point to check, use the last point")
        agent.check_again_postion = interpolated_path[-1]
    else:
        logger.info("Can't find a good point to check, use the current position")
        agent.check_again_postion = agent.mapper.current_position


def _log_and_pick_best_proposal(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_candidate = None
    for candidate in proposals:
        logger.info(
            "Check-again candidate {} score {:.3f}, visible {:.3f}, center {:.3f}, border {:.3f}, area {:.3f}, dist {:.3f}",
            candidate["position"],
            candidate["score"],
            candidate["visible_ratio"],
            candidate["center_score"],
            candidate["border_score"],
            candidate["area_ratio"],
            candidate["distance_to_target"],
        )
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate
    return best_candidate


def _try_view_control_proposal(agent: Any, proposals: list[dict[str, Any]], current_candidate: Any) -> bool:
    state = getattr(agent.mapper, "instruction_execution_state", None)
    pending_pair_active = (
        getattr(state, "mode", "") == "better_view_for_verified_pair"
        and bool(getattr(state, "pending_verified_pair", {}) or {})
    )
    if not (
        getattr(agent.view_control_state, "active", False)
        and (
            agent.view_control_state.candidate_uid == current_candidate.uid
            or pending_pair_active
        )
    ):
        return False

    agent.view_control_state.set_proposals(proposals)
    proposal = agent.view_control_state.next_proposal()
    if proposal is not None:
        logger.info(
            "Selected view-control proposal: {}, score {:.3f}, remaining {}",
            proposal.pose,
            proposal.score,
            agent.view_control_state.remaining_count(),
        )
        agent.check_again_postion = np.array(proposal.pose, dtype=float)
        agent.need_check_again = True
        return True
    logger.info("View-control proposals exhausted for {}", current_candidate.uid)
    agent.need_check_again = False
    agent.check_again_postion = agent.mapper.current_position
    return True
