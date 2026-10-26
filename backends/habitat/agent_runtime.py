"""Habitat episode binding for the tool-driven navigation Agent."""
from dataclasses import replace
import json
import math
from pathlib import Path

from harness.runtime import AgentRuntime
from harness.tools import NavigationTools
from instruction_adapter.verifier import candidate_from_object
from scene.contracts import Pose3D, ViewpointSnapshot
from verification.contracts import VerificationResult


class HabitatEvidenceVerifier:
    """Reuse STRIVE relation/geometry prechecks and the existing final verifier."""
    def __init__(self, agent):
        self.agent = agent

    def verify(self, *, raw_instruction, instruction_plan, candidate, evidence):
        """Return a scoped judgment; task updates remain in the tool's reducer."""
        agent = self.agent
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
        if not checked.satisfied:
            return VerificationResult(
                False, "need_relation_check" if checked.decision == "need_relation_check" else "uncertain",
                confidence=checked.confidence, reason=checked.reason,
                task_revision=agent.task.spec.revision, target_uid=candidate.uid,
                failed_constraints=checked.failed_constraints,
                diagnostics={"constraint_eval": checked.as_dict()},
            )
        # 预筛生成的关系证据与最终验证、reducer 使用同一个 evidence 包。
        evidence.update(checked.evidence)
        return agent.final_instruction_verifier.verify(
            raw_instruction=raw_instruction, instruction_plan=instruction_plan,
            candidate=candidate, evidence=evidence,
        )


class HabitatNavigationTools(NavigationTools):
    """Bind observed map IDs to STRIVE viewpoint geometry and image capture."""
    def __init__(self, agent):
        self.agent = agent
        self.objects = {}
        self.deferred_candidates = set()
        super().__init__(agent.task, agent.execution.motion, HabitatEvidenceVerifier(agent), self.capture_evidence)
        agent.execution.controller.agent = agent

    def refresh_scene(self):
        """Publish observed geometry after a perception cycle or physical motion."""
        from backends.habitat.query import refresh_scene
        self.objects = refresh_scene(self.agent)

    def query_viewpoints(self, candidate_uid):
        """Generate STRIVE's approach/recheck viewpoints for one observed object."""
        existing = super().query_viewpoints(candidate_uid)
        if existing:
            return self._moving_viewpoints(candidate_uid, existing)
        agent = self.agent
        agent.object_final = self.objects[candidate_uid]
        agent.found_goal_position = agent.mapper.current_position.copy()
        agent.find_final_waypoint()
        agent.whether_to_check_again()
        self.publish_object_viewpoints(candidate_uid)
        return self._moving_viewpoints(candidate_uid, super().query_viewpoints(candidate_uid))

    def _moving_viewpoints(self, candidate_uid, views):
        """Offer executable displacement after the current view has been checked."""
        from navigation.action_controller import geodesic_distance_to_waypoint
        available = []
        for view in views:
            distance = geodesic_distance_to_waypoint(self.agent, view["pose"]["position"])
            if math.isfinite(distance) and distance > self.agent.planner._goal_radius:
                available.append(view)
        allowed = {view["uid"] for view in available}
        self.session.snapshot = replace(self.session.snapshot, viewpoints=tuple(
            view for view in self.session.snapshot.viewpoints
            if candidate_uid not in view.visible_objects or view.uid in allowed))
        return available

    def query_scene(self):
        """Return candidates with an available observation/execution opportunity."""
        result = super().query_scene()
        result["candidates"] = [item for item in result["candidates"]
                                if item["candidate_uid"] not in self.deferred_candidates]
        result["deferred_candidate_uids"] = sorted(self.deferred_candidates)
        return result

    def publish_object_viewpoints(self, candidate_uid):
        """Publish the same approach/recheck options used by the default policy."""
        agent = self.agent
        approach = agent.confirmed_target_waypoint if agent.confirmed_target_waypoint is not None else agent.waypoint
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:{name}:{agent.episode_steps}",
            pose=Pose3D(tuple(float(x) for x in position), frame_id="mapper_local"),
            visible_objects=(candidate_uid,), metadata={"source": "strive_viewpoint_geometry",
                                                        "purpose": name, "recommended": name == "recheck"},
        ) for name, position in (("approach", approach), ("recheck", agent.check_again_postion)))
        retained = tuple(view for view in self.session.snapshot.viewpoints if candidate_uid not in view.visible_objects)
        self.session.snapshot = replace(self.session.snapshot, viewpoints=retained + views)
        return views

    def capture_evidence(self, candidate_uid):
        """Face the object and capture the current RGB/depth geometry packet."""
        from navigation.goal_approach_controller import rotate_toward_object_for_recheck
        agent = self.agent
        if self.motion.active_request_id is not None:
            raise RuntimeError("Evidence capture requires terminal motion feedback")
        agent.object_final = self.objects[candidate_uid]
        if not rotate_toward_object_for_recheck(agent):
            raise RuntimeError("Episode ended during evidence capture")
        candidate = candidate_from_object(agent.object_final, step=agent.episode_steps)
        bbox, geometry = agent._project_object_bbox_on_current_view()
        if bbox is None:
            return {
                "observation_identity": agent._observation_identity(candidate_uid),
                "observation_status": "need_observation",
                "geometry": {"visible_projected_points": 0, "projection_failed_in_final_view": True},
                "view_quality_facts": {"visible_projected_points": 0},
            }
        return agent._build_final_verifier_evidence(candidate, bbox, geometry)

    def verify_target(self, candidate_uid, target_id, task_revision):
        """Apply shared verification and export benchmark-facing diagnostics."""
        result = super().verify_target(candidate_uid, target_id, task_revision)
        agent = self.agent
        agent.instruction_decision = agent.final_stop_decision = result["decision"]
        agent.final_stop_mode = "agent"
        if self.session.progress.completed:
            agent.instruction_success = agent.final_stop_success = True
            agent.instruction_accept_step = agent.final_stop_accept_step = agent.episode_steps
            agent.accepted_candidate_uid = candidate_uid
        return result


def run_agent_episode(agent, *, episode_idx, max_steps, max_model_calls=100):
    """Run one episode with Agent decisions and the original navigation backend.

    The legacy planning/step loop is never entered in this mode. The driver
    advances admitted waypoint motion; the model only chooses tools/goals.
    """
    from navigation.planner_loop import run_observation_mapping_cycle

    directory = Path(agent.save_dir) / f"episode-{episode_idx}" / "harness"
    directory.mkdir(parents=True, exist_ok=True)
    episode = agent.env.current_episode
    (directory / "episode_identity.json").write_text(json.dumps({
        "episode_id": episode.episode_id, "scene_id": episode.scene_id,
        "start_position": list(episode.start_position), "start_rotation": list(episode.start_rotation),
        "model_provider": agent.vlm, "decision_owner": "agent",
    }, indent=2))
    tools = HabitatNavigationTools(agent)
    runtime = AgentRuntime(tools, directory / "tools.jsonl", max_model_calls=max_model_calls, vlm=agent.vlm)
    run_observation_mapping_cycle(agent, node=None, episode_idx=episode_idx)
    tools.refresh_scene()
    termination = "episode_limit"
    while not agent.env.episode_over and agent.episode_steps < max_steps:
        if tools.motion.active_request_id is not None:
            agent.execution.controller.advance()
            feedback = tools.motion.poll_active()
            if not feedback.tracking_active:
                if feedback.status.succeeded():
                    frontier_uid = feedback.intent.metadata.get("frontier_uid")
                    node = None
                    if frontier_uid:
                        node = agent.mapper.nodes[int(frontier_uid.split(":")[1])]
                        agent.current_node_idx = node.idx
                        agent.mapper.change_state(node)
                    run_observation_mapping_cycle(agent, node=node, episode_idx=episode_idx)
                tools.refresh_scene()
            continue
        if runtime.model_calls >= max_model_calls:
            termination = "decision_budget_exhausted"
            break
        outcome = runtime.step()
        if outcome["name"] == "propose_finish" and outcome["result"]["accepted"]:
            agent.found_goal = True
            agent.obs = agent.execution.step(0)
            agent.update_trajectory()
            termination = "verified_finish"
            break
    # 截止时终止当前导航尝试，保留未完成状态供独立 evaluator 评分。
    if tools.motion.active_request_id is not None:
        tools.motion.cancel()
        tools.motion.poll_active()
    (directory / "termination.json").write_text(json.dumps({
        "reason": termination, "model_calls": runtime.model_calls,
        "steps": agent.episode_steps, "task_complete": agent.task.progress.completed,
    }, indent=2))
