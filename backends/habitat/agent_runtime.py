"""Habitat episode binding for the tool-driven navigation Agent."""
from dataclasses import replace
import json
from pathlib import Path

from backends.habitat.scene import snapshot_from_mapper
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
        super().__init__(agent.task, agent.execution.motion, HabitatEvidenceVerifier(agent), self.capture_evidence)
        agent.execution.controller.agent = agent

    def refresh_scene(self):
        """Publish observed geometry after a perception cycle or physical motion."""
        agent = self.agent
        previous = self.session.snapshot
        snapshot, self.objects = snapshot_from_mapper(
            agent.mapper, step=agent.episode_steps, nodes=agent.mapper.nodes,
        )
        # 导航视点是场景中的位置，机器人移动后仍可使用；所属实例离开快照时清理。
        # 保留同一对象的 recheck 视点，使到达后的补拍请求引用同一份已发布协议。
        object_views = tuple(view for view in previous.viewpoints
                             if view.visible_objects and
                             all(uid in self.objects for uid in view.visible_objects))
        self.session.snapshot = replace(snapshot, viewpoints=snapshot.viewpoints + object_views)
        self.session.constraint_evaluator.register_mapper_objects(
            mapper=self.session, step=agent.episode_steps, current_node_idx=agent.current_node_idx,
        )

    def query_viewpoints(self, candidate_uid):
        """Generate STRIVE's approach/recheck viewpoints for one observed object."""
        existing = super().query_viewpoints(candidate_uid)
        if existing:
            return existing
        agent = self.agent
        agent.object_final = self.objects[candidate_uid]
        agent.found_goal_position = agent.mapper.current_position.copy()
        agent.find_final_waypoint()
        agent.whether_to_check_again()
        views = tuple(ViewpointSnapshot(
            uid=f"{candidate_uid}:{name}:{agent.episode_steps}",
            pose=Pose3D(tuple(float(x) for x in position), frame_id="mapper_local"),
            visible_objects=(candidate_uid,), metadata={"source": "strive_viewpoint_geometry"},
        ) for name, position in (("approach", agent.waypoint), ("recheck", agent.check_again_postion)))
        self.session.snapshot = replace(self.session.snapshot, viewpoints=self.session.snapshot.viewpoints + views)
        return super().query_viewpoints(candidate_uid)

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
                "current_rgb_with_bbox_path": str(Path(agent.save_dir) / f"episode-{agent.episode_samples - 1}"
                                                   / "rgb" / f"monitor-rgb_{agent.episode_steps}.jpg"),
                "geometry": {"visible_projected_points": 0},
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
