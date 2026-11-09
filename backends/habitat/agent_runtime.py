"""Habitat episode binding for the tool-driven navigation Agent."""
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from harness.runtime import AgentRuntime
from harness.tools import NavigationTools, ToolProposalError
from instruction_adapter.verifier import candidate_from_object
from instruction_adapter.view_control import ViewControlState
from prior_map.simulation import invalidate_mapper_task_prior
from scene.contracts import FrontierSnapshot, Pose3D, ViewpointSnapshot
from verification.contracts import VerificationResult


class HabitatEvidenceVerifier:
    """Reuse STRIVE relation/geometry prechecks and the existing final verifier."""
    def __init__(self, agent):
        self.agent = agent

    def verify(self, *, raw_instruction, instruction_plan, candidate, evidence):
        """Return a scoped judgment; task updates remain in the tool's reducer."""
        agent = self.agent
        self.evidence = evidence
        if (evidence.get("observation_identity") != agent._observation_identity(candidate.uid)
                or evidence.get("observation_status") == "need_observation"):
            return VerificationResult(
                False, "need_better_view", task_revision=agent.task.spec.revision,
                target_uid=candidate.uid, view_sufficient_for_stop=False,
                reason="Capture a current observation of this candidate.",
                diagnostics={"observation_status": "need_observation"},
            )
        target = next(t for t in instruction_plan.terminal_targets if t.id == evidence["target_id"])
        checked = agent.task.constraint_evaluator.evaluate_before_final_verifier(
            mapper=agent.task, plan=instruction_plan, target=target,
            candidate=candidate, candidate_obj=agent.object_final,
            evidence=evidence, step=agent.episode_steps,
        )
        precheck = checked.as_dict()
        precheck.pop("evidence")
        evidence.update(checked.evidence)
        evidence["constraint_precheck"] = precheck
        return agent.final_instruction_verifier.verify(
            raw_instruction=raw_instruction, instruction_plan=instruction_plan,
            candidate=candidate, evidence=evidence,
        )


class HabitatNavigationTools(NavigationTools):
    """Expose enhanced navigation capabilities without choosing the Agent's actions."""
    names = (*NavigationTools.names, "scan_scene")

    def __init__(self, agent):
        self.agent = agent
        self.objects = {}
        self.deferred_candidates = set()
        self.deferred_evidence = {}
        self.candidate_view_states = {}
        self.candidate_motion_goals = {}
        self.motion_evidence = {}
        self.current_evidence = {}
        super().__init__(agent.task, agent.execution.motion, HabitatEvidenceVerifier(agent), self.capture_evidence)
        agent.execution.controller.agent = agent

    def install_interpretation(self, plan, *, task_revision, search_revision, snapshot, record):
        """Adopt the shared revision before invalidating Habitat's task-local choices."""
        result = super().install_interpretation(
            plan, task_revision=task_revision, search_revision=search_revision,
            snapshot=snapshot, record=record)
        agent = self.agent
        active = self.session.progress.active_target()
        agent.mapper.target = active.name
        agent.mapper.target_list = list(self.session.detector_terms)
        agent.mapper.target_aliases = list(dict.fromkeys([active.name, *active.aliases]))
        invalidate_mapper_task_prior(agent.mapper)
        self.deferred_candidates.clear()
        self.current_evidence = {}
        agent.selected_candidate = agent.object_final = None
        agent.found_goal = agent.need_check_again = False
        agent.final_instruction_accepted_this_step = False
        agent.instruction_success = agent.final_stop_success = False
        agent.instruction_decision = agent.final_stop_decision = agent.final_stop_mode = ""
        agent.instruction_accept_step = agent.final_stop_accept_step = None
        agent.accepted_candidate_uid = ""
        agent.accepted_relation_edge = {}
        agent.accepted_distance_to_target = None
        agent.accepted_distance_source = ""
        agent.found_goal_position = agent.waypoint = agent.confirmed_target_waypoint = None
        agent.approach_viewpoint = agent.check_again_postion = None
        # Replace, rather than mutate, the state retained in revision-keyed history.
        agent.view_control_state = ViewControlState()
        agent.last_check_again_evidence = agent.current_surface_evidence = None
        agent.last_check_again_image_path = ""
        # Heading coverage is already revision-keyed; only its cached summary is stale.
        agent.scan_complete = False
        return result

    def refresh_scene(self):
        """Publish current observations and reopen candidates only for new evidence."""
        from backends.habitat.query import refresh_scene
        self.objects = refresh_scene(self.agent)
        for uid in tuple(self.deferred_candidates):
            obj = self.objects.get(uid)
            if obj is None:
                continue
            fingerprint = self.candidate_evidence(obj)
            key = (self.session.spec.revision, uid)
            if fingerprint is not None and fingerprint not in self.deferred_evidence.get(key, set()):
                self.deferred_candidates.remove(uid)
                self.candidate_view_states[key] = ViewControlState()

    @staticmethod
    def candidate_evidence(obj):
        """Identify actual crop content rather than unrelated map-clock changes."""
        if obj.rgb is None or obj.bbox is None:
            return None
        image = np.asarray(obj.rgb)
        x1, y1, x2, y2 = np.asarray(obj.bbox).reshape(-1)[:4].astype(int)
        height, width = image.shape[:2]
        crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if not crop.size:
            return None
        digest = sha256(str((crop.shape, crop.dtype)).encode())
        digest.update(crop.tobytes())
        return digest.hexdigest()

    def _candidate_state(self, candidate_uid):
        agent = self.agent
        agent.object_final = self.objects[candidate_uid]
        key = (self.session.spec.revision, candidate_uid)
        agent.view_control_state = self.candidate_view_states.setdefault(key, ViewControlState())
        return agent.view_control_state

    def _defer_candidate(self, candidate_uid):
        self.deferred_candidates.add(candidate_uid)
        key = (self.session.spec.revision, candidate_uid)
        self.deferred_evidence.setdefault(key, set()).add(self.candidate_evidence(self.objects[candidate_uid]))

    def query_viewpoints(self, candidate_uid):
        """Regenerate untried enhanced viewpoints; querying consumes no proposal."""
        from navigation.view_verification_controller import build_viewpoint_options
        state = self._candidate_state(candidate_uid)
        agent = self.agent
        agent.found_goal_position = agent.mapper.current_position.copy()
        agent.find_final_waypoint()
        proposals, _ = build_viewpoint_options(agent)
        state.set_proposals(proposals)
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:view:{agent.episode_steps}:{index}",
            pose=Pose3D(tuple(proposal.pose), frame_id="mapper_local"),
            visible_objects=(candidate_uid,), metadata={
                "source": "strive_viewpoint_geometry", "purpose": "recheck",
                "recommended": index == 0, "score": proposal.score,
                "stop_feasible": proposal.stop_feasible, "travel_distance": proposal.travel_distance,
            }) for index, proposal in enumerate(state.proposals)
            if not proposal.attempted and not state.budget_exhausted())
        retained = tuple(v for v in self.session.snapshot.viewpoints if candidate_uid not in v.visible_objects)
        self.session.snapshot = replace(self.session.snapshot, viewpoints=retained + views)
        available = self._moving_viewpoints(candidate_uid, super().query_viewpoints(candidate_uid))
        if not available:
            self._defer_candidate(candidate_uid)
        return available

    def _moving_viewpoints(self, candidate_uid, views):
        """Offer executable displacement after the current view has been checked."""
        from navigation.action_controller import geodesic_distance_to_waypoint
        available = []
        key = (self.session.spec.revision, candidate_uid, self.candidate_evidence(self.objects[candidate_uid]))
        attempted = self.candidate_motion_goals.get(key, {})
        radius = self.agent.planner._goal_radius
        for view in views:
            position = view["pose"]["position"]
            distance = geodesic_distance_to_waypoint(self.agent, position)
            repeated = any(np.linalg.norm(np.asarray(position)[:2] - np.asarray(old)[:2]) <= radius
                           for old in attempted)
            if math.isfinite(distance) and distance > radius and not repeated:
                available.append(view)
        allowed = {view["uid"] for view in available}
        self.session.snapshot = replace(self.session.snapshot, viewpoints=tuple(
            view for view in self.session.snapshot.viewpoints
            if candidate_uid not in view.visible_objects or view.uid in allowed))
        return available

    def observe(self):
        """Expose scan coverage alongside the shared cached observation."""
        result = super().observe()
        result["scan_complete"] = getattr(self.agent, "scan_complete", False)
        return result

    def query_scene(self):
        """Return all eligible alternatives and enhanced exploration options."""
        result = super().query_scene()
        result["candidates"] = [item for item in result["candidates"]
                                if item["candidate_uid"] not in self.deferred_candidates]
        result["deferred_candidate_uids"] = sorted(self.deferred_candidates)
        result["exploration"] = self.exploration_options()
        return result

    def exploration_options(self):
        """Publish room-first, path-cost and coverage recommendations without motion."""
        from planning.exploration_policy import find_closest_nodes
        mapper = self.agent.mapper
        nodes = [node for node in mapper.nodes if node.has_frontier and node.state == 0]
        if not nodes:
            nodes = [node for node in mapper.nodes if node.state == 0]
        current_room = mapper.nodes[self.agent.current_node_idx].room_idx
        local = [node for node in nodes if node.room_idx == current_room]
        recommended = find_closest_nodes(mapper, local or nodes)
        recommended_uid = f"node:{recommended.idx}" if recommended is not None else None
        if nodes:
            find_closest_nodes(mapper, nodes)
            rows = [row for row in mapper.prior_map_last_chosen_frontier["candidates"]
                    if math.isfinite(row["raw_distance_m"])]
        else:
            rows = []
        options = [{**row, "frontier_uid": f"node:{row['node_idx']}",
                    "current_room": row["room_idx"] == current_room,
                    "coverage_cells": len(set(map(tuple, mapper.nodes[row["node_idx"]].frontier_idxs))),
                    "room_object_labels": sorted({mapper.objects[index].tag
                        for node in mapper.room_nodes[row["room_idx"]].nodes
                        for index in node.objects})} for row in rows]
        allowed = {row["frontier_uid"] for row in options}
        if recommended_uid not in allowed:
            recommended_uid = None
        self.session.snapshot = replace(self.session.snapshot, frontiers=tuple(FrontierSnapshot(
            uid=row["frontier_uid"], position=tuple(row["position"]), room_id=str(row["room_idx"]),
            metadata={"frame_id": "mapper_local", "recommended": row["frontier_uid"] == recommended_uid,
                      "path_cost_m": row["raw_distance_m"], "coverage_cells": row["coverage_cells"],
                      "room_object_labels": row["room_object_labels"], "current_room": row["current_room"]},
        ) for row in options))
        return {"recommended_uid": recommended_uid, "options": options}

    def scan_scene(self):
        """Execute remaining adaptive headings at this pose, then fuse the observations."""
        from navigation.planner_loop import run_observation_mapping_cycle
        if self.motion.active_request_id is not None or self.motion.held:
            raise ToolProposalError("Scanning requires idle, unheld motion")
        if self.movement_budget_remaining is not None and self.movement_budget_remaining <= 0:
            return {"accepted": False, "reason": "movement_budget_exhausted"}
        self._finish_verification = None
        agent = self.agent
        node = agent.mapper.nodes[agent.current_node_idx]
        if np.linalg.norm(np.asarray(node.position)[:2] - agent.mapper.current_position[:2]) > agent.planner._goal_radius:
            node = None
        cycle = run_observation_mapping_cycle(agent, node=node, episode_idx=agent.episode_samples - 1)
        if node is not None and agent.scan_complete and not cycle.episode_over:
            agent.mapper.change_state(node)
        self.refresh_scene()
        self.exploration_options()
        return {"accepted": True, "scan_complete": agent.scan_complete,
                "geometry_available": cycle.geometry_available, "observation": self.observe()}

    def submit_motion(self, task_revision, control_epoch, request_id,
                      candidate_uid=None, viewpoint_uid=None, frontier_uid=None):
        """Submit the Agent-selected view and retain its evidence version for feedback."""
        result = super().submit_motion(task_revision, control_epoch, request_id,
            candidate_uid=candidate_uid, viewpoint_uid=viewpoint_uid, frontier_uid=frontier_uid)
        if candidate_uid is not None and "goal_id" in result:
            self.motion_evidence.setdefault(result["goal_id"], (
                task_revision, candidate_uid, self.candidate_evidence(self.objects[candidate_uid])))
        return result

    def motion_finished(self, feedback):
        """Consume a viewpoint only after a terminal execution receipt."""
        key = self.motion_evidence.pop(feedback.status.goal_id, None)
        if key is None:
            return
        position = tuple(feedback.intent.goal_pose.position)
        self.candidate_motion_goals.setdefault(key, {})[position] = feedback.status.status.value
        state = self._candidate_state(key[1])
        for index, proposal in enumerate(state.proposals):
            if np.linalg.norm(np.asarray(position)[:2] - np.asarray(proposal.pose)[:2]) <= self.agent.planner._goal_radius:
                proposal.attempted = True
                state.last_selected_index = index

    def publish_object_viewpoints(self, candidate_uid):
        """Publish projection-scored approach/recheck options for one object.

        approach 是停止半径内投影质量最高的视点，满足
        d_object(v) + rho <= d_stop 且目标点云落入画面；物体表面点不作为视点。
        """
        agent = self.agent
        options = (("approach", agent.approach_viewpoint), ("recheck", agent.check_again_postion))
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:{name}:{agent.episode_steps}",
            pose=Pose3D(tuple(float(x) for x in position), frame_id="mapper_local"),
            visible_objects=(candidate_uid,), metadata={"source": "strive_viewpoint_geometry",
                                                        "purpose": name, "recommended": name == "recheck"},
        ) for name, position in options if position is not None)
        retained = tuple(view for view in self.session.snapshot.viewpoints if candidate_uid not in view.visible_objects)
        self.session.snapshot = replace(self.session.snapshot, viewpoints=retained + views)
        return views

    def capture_evidence(self, candidate_uid):
        """Face the object and capture the current RGB/depth geometry packet."""
        from navigation.goal_approach_controller import rotate_toward_object_for_recheck
        agent = self.agent
        if self.motion.active_request_id is not None:
            raise RuntimeError("Evidence capture requires terminal motion feedback")
        self._candidate_state(candidate_uid)
        captured = rotate_toward_object_for_recheck(agent)
        self.refresh_scene()
        candidate = candidate_from_object(agent.object_final, step=agent.episode_steps)
        bbox, geometry = agent._project_object_bbox_on_current_view() if captured else (None, {})
        if bbox is None:
            evidence = {
                "observation_identity": agent._observation_identity(candidate_uid),
                "observation_status": "need_observation",
                "geometry": {"visible_projected_points": 0, "projection_failed_in_final_view": True},
                "view_quality_facts": {"visible_projected_points": 0},
            }
        else:
            evidence = agent._build_final_verifier_evidence(candidate, bbox, geometry)
        self.current_evidence = agent._attach_view_control_context(evidence, candidate)
        return self.current_evidence

    def verify_target(self, candidate_uid, target_id, task_revision):
        """Apply shared verification and export benchmark-facing diagnostics.

        判定绑定取证时的观察；判定后若任务未完成，恢复水平相机供后续移动与建图。
        """
        from navigation.goal_approach_controller import tilt_camera
        result = super().verify_target(candidate_uid, target_id, task_revision)
        agent = self.agent
        agent.instruction_decision = agent.final_stop_decision = result["decision"]
        agent.final_stop_mode = "agent"
        state = agent.view_control_state
        self.current_evidence = self.verifier.evidence
        if result["decision"] in {"need_better_view", "uncertain"}:
            objective = dict(result.get("view_objective") or {})
            if result.get("preferred_view_goal"):
                objective["preferred_view_goal"] = result["preferred_view_goal"]
            if result.get("view_feedback"):
                objective["view_feedback"] = result["view_feedback"]
            state.start(candidate_uid, objective, self.current_evidence)
            if result["semantic_satisfied"]:
                state.pin_visual_evidence(self.current_evidence, step=agent.episode_steps,
                                          decision=result["decision"], reason=result["reason"])
            state.pin_relation_context(self.current_evidence.get("verified_relation_context"))
        state.record_attempt(agent.episode_steps, self.current_evidence, result["decision"],
                             semantic_satisfied=result["semantic_satisfied"], reason=result["reason"])
        if result["decision"] in {"reject_candidate", "reject_relation"}:
            state.reset()
        result["view_control"] = state.as_context()
        if self.propose_finish()["accepted"]:
            agent.instruction_success = agent.final_stop_success = True
            agent.instruction_accept_step = agent.final_stop_accept_step = agent.episode_steps
            agent.accepted_candidate_uid = candidate_uid
        elif not agent.env.episode_over:
            tilt_camera(agent, 0.0)
            self.refresh_scene()
        return result


def run_agent_episode(agent, *, episode_idx, max_steps, max_model_calls=100,
                      stop_on_success=True, output_dir=None):
    """Run Agent-selected enhanced skills, leaving sequential STOP to its parent driver."""
    from backends.habitat.motion import EpisodeStepLimit
    from navigation.planner_loop import PlanningCycleResult, run_observation_mapping_cycle

    directory = Path(output_dir) if output_dir is not None else Path(agent.save_dir) / f"episode-{episode_idx}" / "harness"
    directory.mkdir(parents=True, exist_ok=True)
    episode = agent.env.current_episode
    (directory / "episode_identity.json").write_text(json.dumps({
        "episode_id": episode.episode_id, "scene_id": episode.scene_id,
        "start_position": list(episode.start_position), "start_rotation": list(episode.start_rotation),
        "model_provider": agent.vlm, "decision_owner": "agent",
    }, indent=2))
    agent.adaptive_observation = True
    agent.semantic_scan_headings = {}
    agent.semantic_label_evidence = {}
    agent.scan_complete = False
    agent.mapper.coverage_guided_exploration = True
    agent.mapper.frontier_distance_tie_m = agent.env.sim.sim_config.agents[0].action_space[1].actuation.amount
    tools = HabitatNavigationTools(agent)
    agent.execution.max_steps = max_steps
    runtime = AgentRuntime(tools, directory / "tools.jsonl", max_model_calls=max_model_calls, vlm=agent.vlm)
    try:
        cycle = run_observation_mapping_cycle(agent, node=None, episode_idx=episode_idx)
    except EpisodeStepLimit:
        cycle = PlanningCycleResult(agent.episode_steps, episode_over=True)
    if cycle.geometry_available and not cycle.episode_over:
        tools.refresh_scene()
        tools.exploration_options()
    termination = "episode_limit" if cycle.geometry_available else "observation_geometry_unavailable"
    while cycle.geometry_available and not agent.env.episode_over and agent.execution.controller.action_count < max_steps:
        if tools.motion.active_request_id is not None:
            agent.execution.controller.advance()
            feedback = tools.motion.poll_active()
            if not feedback.tracking_active:
                tools.motion_finished(feedback)
                runtime.record_event({"event": "motion_terminal", "step": agent.episode_steps,
                    "status": feedback.status.status.value, "candidate_uid": feedback.intent.target_object_uid,
                    "goal_pose": feedback.intent.goal_pose.as_dict()})
                frontier_uid = feedback.intent.metadata.get("frontier_uid")
                if feedback.status.succeeded() and frontier_uid:
                    node = agent.mapper.nodes[int(frontier_uid.split(":")[1])]
                    agent.current_node_idx = node.idx
                    agent.on_node_flag = True
                    try:
                        cycle = run_observation_mapping_cycle(agent, node=node, episode_idx=episode_idx)
                    except EpisodeStepLimit:
                        break
                    if not cycle.geometry_available:
                        termination = "observation_geometry_unavailable"
                        break
                    if agent.scan_complete and not cycle.episode_over:
                        agent.mapper.change_state(node)
                if not agent.env.episode_over:
                    tools.refresh_scene()
                    tools.exploration_options()
            continue
        if runtime.model_calls >= max_model_calls:
            termination = "decision_budget_exhausted"
            break
        tools.movement_budget_remaining = max_steps - agent.execution.controller.action_count if math.isfinite(max_steps) else None
        search_revision = agent.task.search_revision
        try:
            outcome = runtime.step()
        except EpisodeStepLimit:
            break
        if agent.task.search_revision != search_revision:
            agent.semantic_scan_headings.clear()
            agent.scan_complete = False
        if outcome["name"] == "propose_finish" and outcome["result"]["accepted"]:
            agent.found_goal = True
            if stop_on_success:
                agent.obs = agent.execution.step(0)
                agent.update_trajectory()
            termination = "verified_finish"
            break
    if tools.motion.active_request_id is not None:
        tools.motion.cancel()
        feedback = tools.motion.poll_active()
        tools.motion_finished(feedback)
    agent.instruction_success = agent.final_stop_success = termination == "verified_finish"
    result = {
        "reason": termination, "model_calls": runtime.model_calls,
        "decision_owner": "agent", "observation_mode": "multi",
        "steps": agent.episode_steps, "task_complete": termination == "verified_finish",
        "structured_task_complete": agent.task.progress.completed,
    }
    (directory / "termination.json").write_text(json.dumps(result, indent=2))
    return result
