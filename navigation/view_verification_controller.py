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
    current_position = agent.mapper.current_position + agent.mapper.initial_position
    current_position = np.array([
        current_position[0],
        agent.env.sim.get_agent_state().position[1],
        current_position[1],
    ])
    path_request.requested_start = current_position
    target_waypoint = _target_waypoint_for_view_objective(agent)
    pid_waypoint = _habitat_point(agent, target_waypoint)
    path_request.requested_end = pid_waypoint

    found_path = pathfinder.find_path(path_request)
    if not found_path and getattr(agent, "confirmed_target_waypoint", None) is not None:
        # 如果目标表面点不可达，退回已确认导航 waypoint。这里不改变语义目标，
        # 只是为 view-control 选择一个可执行路径端点。
        target_waypoint = np.asarray(agent.confirmed_target_waypoint, dtype=float)
        path_request.requested_end = _habitat_point(agent, target_waypoint)
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


def _habitat_point(agent: Any, strive_point: np.ndarray) -> np.ndarray:
    point = np.asarray(strive_point, dtype=float) + agent.mapper.initial_position
    return np.array([
        point[0],
        agent.env.sim.get_agent_state().position[1],
        point[1],
    ])


def _target_waypoint_for_view_objective(agent: Any) -> np.ndarray:
    """Choose a geometry endpoint for better-view path generation.

    If the verifier asks to move closer, the path should be generated toward the
    currently pinned object instance.  Reusing a stale discovery waypoint can keep
    all proposals far away and make `budget_exhausted` meaningless.
    """

    if getattr(agent.view_control_state, "active", False):
        objective = dict(getattr(agent.view_control_state, "objective", {}) or {})
        try:
            current_distance = float(objective.get("current_distance_to_object", 0.0))
            required_distance = float(objective.get("required_stop_distance", agent.success_distance))
        except Exception:
            current_distance = 0.0
            required_distance = 0.0
        improve_goals = " ".join(str(x).lower() for x in objective.get("improve_goals", []) or [])
        wants_closer = "closer" in improve_goals or (
            required_distance > 0 and current_distance > required_distance
        )
        if wants_closer:
            try:
                waypoint = agent.object_final.find_closest(agent.mapper.current_position)
                return np.asarray(waypoint, dtype=float)
            except Exception:
                return np.asarray(agent.object_final.position, dtype=float)

    target_waypoint = getattr(agent, "confirmed_target_waypoint", None)
    if target_waypoint is None:
        target_waypoint = agent.waypoint
    return np.asarray(target_waypoint, dtype=float)
