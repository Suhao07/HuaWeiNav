"""Baseline navigation skills with event-scoped Agent recovery in Habitat."""
from pathlib import Path
import json
import math
import os
from hashlib import sha256
import numpy as np

from instruction_adapter.view_control import ViewControlState

from backends.habitat.agent_runtime import HabitatNavigationTools
from backends.habitat.query import select_observed_candidate
from harness.runtime import AgentRuntime
from motion.contracts import MotionGoalMode, NavigationIntent
from navigation.planner_loop import run_observation_mapping_cycle
from navigation_core.view_geometry import camera_pitch
from planning.exploration_policy import (select_baseline_exploration_node, select_exploration_node,
                                         select_prior_region_node)
from scene.contracts import Pose3D


class GuidedNavigation:
    """Bind existing navigation skills to one motion owner and decision journal.

    Ordinary exploration uses the room-first baseline policy. Semantic room
    selection retains its existing model call; AgentRuntime handles explicit
    execution recovery. Task completion remains owned by the shared reducer.
    """
    def __init__(self, agent, directory, max_model_calls):
        self.agent = agent
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tools = HabitatNavigationTools(agent)
        self.runtime = AgentRuntime(self.tools, self.directory / "tools.jsonl",
                                    max_model_calls=max_model_calls, vlm=agent.vlm)
        self.sequence = 0
        self.recovery_context = {}
        self.deferred_candidates = self.tools.deferred_candidates
        self.deferred_evidence = {}
        self.candidate_view_states = {}
        self.candidate_motion_goals = {}
        self.search_review_step = agent.episode_steps
        self.search_review_interval = int(os.getenv("STRIVE_SEARCH_REVIEW_STEPS", "100"))
        agent.adaptive_observation = True
        agent.semantic_scan_headings = {}
        agent.semantic_label_evidence = {}
        agent.scan_complete = False
        agent.mapper.coverage_guided_exploration = True
        agent.mapper.frontier_distance_tie_m = agent.env.sim.sim_config.agents[0].action_space[1].actuation.amount

    def record(self, event, **fields):
        """Append the actual state transition without hiding execution errors."""
        record = {"step": self.agent.episode_steps, "event": event,
                  "task_revision": self.agent.task.spec.revision, **fields}
        with (self.directory / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if event in {"submitted", "motion_terminal", "verified", "candidate_deferred",
                     "candidate_evidence_updated", "anchor_searched"}:
            self.runtime.record_event(record)

    def submit(self, position, *, candidate_uid=None, node=None, source):
        """Submit a domain-selected waypoint through the existing supervisor."""
        self.sequence += 1
        position = tuple(float(x) for x in position)
        metadata = {"task_revision": self.agent.task.spec.revision,
                    "decision_source": source}
        if node is not None:
            metadata["node_index"] = node.idx
            mode = MotionGoalMode.GO_TO_FRONTIER
        else:
            mode = MotionGoalMode.IMPROVE_VIEW
            metadata["candidate_evidence_fingerprint"] = self.candidate_evidence(self.tools.objects[candidate_uid])
            # 所有对象视点使用局部规划器到达容差，与视点可行条件一致。
        intent = NavigationIntent(mode=mode, target_object_uid=candidate_uid,
                                  goal_pose=Pose3D(position, frame_id="mapper_local"), metadata=metadata)
        self.tools.motion.submit(intent, intent.to_motion_goal(),
                                 request_id=f"guided-{self.sequence}",
                                 control_epoch=self.tools.motion.control_epoch)
        self.record("submitted", decision_source=source, node_index=metadata.get("node_index"),
                    candidate_uid=candidate_uid, position=position)

    def attempted_positions(self, candidate_uid):
        """Return completed execution attempts for this instance evidence version."""
        key = (self.agent.task.spec.revision, candidate_uid,
               self.candidate_evidence(self.tools.objects[candidate_uid]))
        return self.candidate_motion_goals.get(key, {})

    def view_attempted(self, candidate_uid, position):
        """Use the controller's arrival tolerance to identify the same view."""
        radius = self.agent.planner._goal_radius
        # Habitat 将 mapper-local XY 转为地面导航坐标，Z 使用当前机器人高度。
        return any(np.linalg.norm(np.asarray(position)[:2] - np.asarray(old)[:2]) <= radius
                   for old in self.attempted_positions(candidate_uid))

    def motion_finished(self, feedback):
        """Retain terminal facts; a submitted goal alone consumes no observation.

        执行机会键为 (任务版本, 实例, 证据内容, 视点)。新证据可以在旧位置
        再取证；旧版本的成功、阻塞与取消事实仍保留，供恢复决策追溯。
        """
        uid = feedback.intent.target_object_uid
        if uid is None:
            return
        metadata = feedback.intent.metadata
        key = (metadata["task_revision"], uid, metadata["candidate_evidence_fingerprint"])
        pose = tuple(float(x) for x in feedback.intent.goal_pose.position)
        self.candidate_motion_goals.setdefault(key, {})[pose] = feedback.status.status.value

    def select(self, episode_idx):
        """Query candidates once per decision cycle, then apply baseline order."""
        agent = self.agent
        agent.on_node_flag = True
        agent.found_goal, agent.object_final = select_observed_candidate(
            agent, episode_idx=episode_idx, step=agent.episode_steps,
            deferred_uids=self.deferred_candidates)
        self.tools.refresh_scene()
        if agent.found_goal:
            key = (agent.task.spec.revision, agent.object_final.uid)
            agent.view_control_state = self.candidate_view_states.setdefault(key, ViewControlState())
            agent.found_goal_position = agent.mapper.current_position.copy()
            agent.confirmed_target_waypoint = None
            agent.find_final_waypoint()
            agent.whether_to_check_again()
            views = self.tools.publish_object_viewpoints(agent.object_final.uid)
            available = [view for view in views
                         if not self.view_attempted(agent.object_final.uid, view.pose.position)]
            if available:
                recommended = next((view for view in available if view.metadata["recommended"]), available[0])
                self.submit(recommended.pose.position, candidate_uid=agent.object_final.uid,
                            source="baseline_viewpoint")
                return True
            self.defer_candidate("available_views_already_attempted")
        if agent.observation_mode == "multi" and not agent.scan_complete:
            run_observation_mapping_cycle(agent, node=agent.mapper.nodes[agent.current_node_idx], episode_idx=episode_idx)
            self.observation_updated()
            return True
        node, source = select_exploration_node(agent, episode_idx=episode_idx,
                                              step=agent.episode_steps)
        if node is None:
            self.record("search_exhausted", decision_source=source)
            return False
        node, source = self.review_search_branch(node, source, episode_idx)
        if source in {"interpretation_changed", "decision_reselect"}:
            return True
        # selected 由 active goal 表达；观察完成后才更新地图的 explored 状态。
        if source == "agent_search_review":
            with self.runtime.choice_guard():
                self.submit(node.position, node=node, source=source)
        else:
            self.submit(node.position, node=node, source=source)
        return True

    @staticmethod
    def candidate_evidence(obj):
        """Fingerprint the observed instance crop, independent of map clocks.

        恢复依赖 E(uid) 的内容变化；地图版本、机器人位姿和其他对象更新
        不构成该实例的新证据。空图像保持 unknown，不能解除暂缓。
        """
        if obj.rgb is None or obj.bbox is None:
            return None
        image = np.asarray(obj.rgb)
        x1, y1, x2, y2 = np.asarray(obj.bbox).reshape(-1)[:4].astype(int)
        height, width = image.shape[:2]
        crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if crop.size == 0:
            return None
        digest = sha256(str((crop.shape, crop.dtype)).encode())
        digest.update(crop.tobytes())
        return digest.hexdigest()

    def defer_candidate(self, reason):
        """Suspend this evidence version while retaining its execution history."""
        agent = self.agent
        uid = agent.object_final.uid
        self.deferred_candidates.add(uid)
        key = (agent.task.spec.revision, uid)
        fingerprint = self.candidate_evidence(agent.object_final)
        self.deferred_evidence.setdefault(key, set()).add(fingerprint)
        self.candidate_view_states[key] = agent.view_control_state
        self.record("candidate_deferred", candidate_uid=uid, reason=reason,
                    evidence_fingerprint=fingerprint)

    def observation_updated(self):
        """Resume only instances that acquired different visual evidence."""
        self.tools.refresh_scene()
        revision = self.agent.task.spec.revision
        for uid in tuple(self.deferred_candidates):
            obj = self.tools.objects.get(uid)
            if obj is None:
                continue
            fingerprint = self.candidate_evidence(obj)
            key = (revision, uid)
            if fingerprint is not None and fingerprint not in self.deferred_evidence.get(key, set()):
                self.deferred_candidates.remove(uid)
                self.record("candidate_evidence_updated", candidate_uid=uid,
                            evidence_fingerprint=fingerprint)

    def exploration_options(self, episode_idx):
        r"""Return the default branch and the reachable exploration option set.

        .. math::
            U(s) = N_{baseline}(s) \cup \{n_{prior}\}

        先验区域只追加其推荐分支并标记，不裁剪 baseline 候选；默认分支与
        select_exploration_node 一致。baseline 先计算，使最后记录的排序为默认选择。
        """
        agent = self.agent
        node, source = select_baseline_exploration_node(agent, episode_idx=episode_idx,
                                                        step=agent.episode_steps)
        rows = list(agent.mapper.prior_map_last_chosen_frontier["candidates"]) if node is not None else []
        prior = select_prior_region_node(agent)
        if prior is not None:
            node, source = prior, "prior_region_policy"
            prior_row = next(row for row in agent.mapper.prior_map_last_chosen_frontier["candidates"]
                             if row["node_idx"] == prior.idx)
            rows = [row for row in rows if row["node_idx"] != prior.idx]
            rows.append({**prior_row, "prior_region_recommended": True})
        return node, source, [row for row in rows if math.isfinite(row["raw_distance_m"])]

    def review_search_branch(self, node, source, episode_idx):
        """Finding an unverified candidate must not postpone periodic search review."""
        agent = self.agent
        if (agent.episode_steps - self.search_review_step < self.search_review_interval
                or self.runtime.model_calls >= self.runtime.max_model_calls):
            return node, source
        if source == "prior_region_policy":
            node, source, rows = self.exploration_options(episode_idx)
        else:
            rows = [row for row in agent.mapper.prior_map_last_chosen_frontier["candidates"]
                    if math.isfinite(row["raw_distance_m"])]
        options = [{**row, "frontier_uid": f"node:{row['node_idx']}"} for row in rows]
        if not options:
            return node, source
        recommended = f"node:{node.idx}"
        revision = agent.task.spec.revision
        selected = self.runtime.choose_frontier(options, recommended, {
            "reason": "prolonged_target_search",
            "steps_since_review": agent.episode_steps - self.search_review_step,
        })
        self.search_review_step = agent.episode_steps
        if selected is None:
            event = "interpretation_changed" if agent.task.spec.revision != revision else "decision_reselect"
            self.record(event, decision_source="agent_search_review",
                        task_revision=agent.task.spec.revision)
            return None, event
        selected_index = next(row["node_idx"] for row in options if row["frontier_uid"] == selected)
        self.record("search_review", decision_source="agent", baseline_uid=recommended,
                    selected_uid=selected, options=options)
        return agent.mapper.nodes[selected_index], "agent_search_review"

    def verify(self, scored_position):
        r"""Reuse baseline evidence capture, final verifier and view-control.

        Args:
            scored_position: Viewpoint ``v`` scored by the projection policy.

        评分假设相机位于 v 且光轴正对目标中心；执行在到达容差内停下并按
        30° 离散转向。日志记录两者差异，与本次取证的观察绑定：

        .. math::
            \Delta p = \lVert x_{xy} - v_{xy} \rVert_2,\qquad
            \Delta\psi = \arccos\frac{f_{xy} \cdot (o_{xy} - x_{xy})}
                {\lVert f_{xy} \rVert \lVert o_{xy} - x_{xy} \rVert}

        其中 x 为执行位置，f = -R_{:,2} 为相机光轴，o 为目标中心。
        """
        from navigation.goal_approach_controller import rotate_toward_object_for_recheck, tilt_camera
        agent = self.agent
        agent.need_check_again = False
        if not rotate_toward_object_for_recheck(agent):
            return "episode_limit"
        agent.check_again(agent.episode_steps)
        position = np.asarray(agent.mapper.current_position, dtype=float)
        forward = -np.asarray(agent.mapper.current_rotation)[:2, 2]
        bearing = np.asarray(agent.object_final.position[:2], dtype=float) - position[:2]
        heading_offset = np.degrees(np.arccos(np.clip(
            forward @ bearing / (np.linalg.norm(forward) * np.linalg.norm(bearing)), -1.0, 1.0)))
        pose_gap = {"observation_id": agent._observation_identity(agent.object_final.uid)["observation_id"],
                    "scored_position": [float(x) for x in scored_position],
                    "execution_position": position.tolist(),
                    "position_offset_m": float(np.linalg.norm(position[:2] - np.asarray(scored_position)[:2])),
                    "heading_offset_deg": float(heading_offset),
                    "pitch_deg": float(np.degrees(camera_pitch(np.asarray(agent.mapper.current_rotation))))}
        complete = agent.final_instruction_check(evidence_override=agent.last_check_again_evidence)
        if agent.selected_candidate.role == "anchor":
            self.record("anchor_searched", anchor_uid=agent.object_final.uid,
                        concept_id=agent.selected_candidate.target_id,
                        status="searched_no_terminal_found", pose_gap=pose_gap)
            if not tilt_camera(agent, 0.0):
                return "episode_limit"
            if (agent.episode_steps - self.search_review_step >= self.search_review_interval
                    and self.runtime.model_calls < self.runtime.max_model_calls):
                self.recovery_context = {
                    "reason": "prolonged_target_search", "anchor_uid": agent.object_final.uid,
                    "concept_id": agent.selected_candidate.target_id,
                    "anchor_status": "searched_no_terminal_found",
                    "steps_since_review": agent.episode_steps - self.search_review_step}
                return "recovery"
            return "select"
        self.record("verified", candidate_uid=agent.object_final.uid,
                    decision=agent.final_stop_decision, task_complete=complete, pose_gap=pose_gap)
        if complete:
            return "verified_finish"
        # 复核结束后恢复水平相机，后续移动与建图沿用水平观测。
        if not tilt_camera(agent, 0.0):
            return "episode_limit"
        if (agent.episode_steps - self.search_review_step >= self.search_review_interval
                and self.runtime.model_calls < self.runtime.max_model_calls):
            self.recovery_context = {
                "reason": "prolonged_unverified_candidate", "candidate_uid": agent.object_final.uid,
                "decision": agent.final_stop_decision,
                "steps_since_review": agent.episode_steps - self.search_review_step}
            return "recovery"
        if agent.need_check_again or agent._continue_better_view_control():
            views = self.tools.publish_object_viewpoints(agent.object_final.uid)
            key = (agent.task.spec.revision, agent.object_final.uid)
            available = [view for view in views
                         if not self.view_attempted(agent.object_final.uid, view.pose.position)]
            if available:
                recommended = next((view for view in available if view.metadata["recommended"]), available[0])
                self.submit(recommended.pose.position, candidate_uid=agent.object_final.uid,
                            source="baseline_view_feedback")
                return "moving"
            agent.view_control_state.exhausted = True
        if agent.view_control_state.active and agent.view_control_state.budget_exhausted():
            self.defer_candidate("view_options_exhausted")
            self.recovery_context = {"reason": "view_options_exhausted",
                                     "candidate_uid": agent.object_final.uid,
                                     "view_objective": agent.view_control_state.objective}
            return "recovery"
        self.defer_candidate(agent.final_stop_decision)
        return "select"

    def recover(self, reason, episode_idx):
        """Choose one executable view or exploration action, then return to the driver.

        Agent 的输出空间是领域候选集合 U(s)，不是任意工具序列。
        单一合法动作仍可复议检索假设；执行与 STOP 仍由领域控制器负责。
        """
        agent = self.agent
        self.tools.refresh_scene()
        _, _, rows = self.exploration_options(episode_idx)
        failed_index = reason.get("goal", {}).get("node_index")
        options = [{**row, "action_uid": f"frontier:{row['node_idx']}", "kind": "frontier"}
                   for row in rows if row["node_idx"] != failed_index]
        uid = reason.get("candidate_uid")
        if uid is not None:
            for view in self.tools.query_viewpoints(uid):
                if not self.view_attempted(uid, view["pose"]["position"]):
                    options.append({**view, "action_uid": view["uid"], "kind": "viewpoint",
                                    "candidate_uid": uid})
        if not options:
            return "select" if reason.get("anchor_uid") is not None else "search_exhausted"
        if self.runtime.model_calls >= self.runtime.max_model_calls:
            return "decision_budget_exhausted"
        revision = agent.task.spec.revision
        selected = self.runtime.choose_recovery(options, reason)
        self.search_review_step = agent.episode_steps
        source = "agent_recovery"
        if selected is None:
            event = "interpretation_changed" if agent.task.spec.revision != revision else "decision_reselect"
            self.record(event, decision_source=source, task_revision=agent.task.spec.revision)
            return "select"
        self.record("agent_intervention", reason=reason, decision_source=source,
                    selected_uid=selected, options=options)
        choice = next(row for row in options if row["action_uid"] == selected)
        with self.runtime.choice_guard():
            if choice["kind"] == "viewpoint":
                self.submit(choice["pose"]["position"], candidate_uid=uid, source=source)
            else:
                node = agent.mapper.nodes[choice["node_idx"]]
                self.submit(node.position, node=node, source=source)
        return "moving"


def run_guided_episode(agent, *, episode_idx, max_steps, max_model_calls=100,
                       stop_on_success=True, output_dir=None):
    """Execute baseline skills, optionally leaving STOP ownership to the sequence driver."""
    directory = (Path(output_dir) if output_dir is not None
                 else Path(agent.save_dir) / f"episode-{episode_idx}" / "harness")
    guided = GuidedNavigation(agent, directory, max_model_calls)
    agent.on_node_flag = True
    cycle = run_observation_mapping_cycle(agent, node=None, episode_idx=episode_idx)
    state = "select"
    termination = "episode_limit" if cycle.geometry_available else "observation_geometry_unavailable"
    while cycle.geometry_available and not agent.env.episode_over and agent.episode_steps < max_steps:
        if guided.tools.motion.active_request_id is not None:
            agent.execution.controller.advance()
            feedback = guided.tools.motion.poll_active()
            if feedback.tracking_active:
                continue
            guided.motion_finished(feedback)
            guided.record("motion_terminal", status=feedback.status.status.value,
                          candidate_uid=feedback.intent.target_object_uid,
                          position=feedback.intent.goal_pose.position,
                          task_revision=feedback.intent.metadata["task_revision"],
                          metadata=feedback.intent.metadata)
            if not feedback.status.succeeded():
                guided.recovery_context = {"reason": "motion_terminal_failure",
                                           "status": feedback.status.status.value,
                                           "candidate_uid": feedback.intent.target_object_uid,
                                           "goal": feedback.intent.metadata}
                state = "recovery"
                continue
            node_index = feedback.intent.metadata.get("node_index")
            frontier_uid = feedback.intent.metadata.get("frontier_uid")
            if frontier_uid is not None:
                node_index = int(frontier_uid.split(":")[1])
            if node_index is not None:
                node = agent.mapper.nodes[node_index]
                agent.current_node_idx = node.idx
                agent.mapper.traj.append(node.idx)
                cycle = run_observation_mapping_cycle(agent, node=node, episode_idx=episode_idx)
                if not cycle.geometry_available:
                    termination = "observation_geometry_unavailable"
                    break
                guided.observation_updated()
                if not agent.env.episode_over and (agent.observation_mode == "single" or agent.scan_complete):
                    agent.mapper.change_state(node)
                state = "select"
            else:
                agent.object_final = guided.tools.objects[feedback.intent.target_object_uid]
                state = guided.verify(feedback.intent.goal_pose.position)
                if agent.observation_mode == "single" and state in {"select", "recovery"} and not agent.env.episode_over:
                    cycle = run_observation_mapping_cycle(
                        agent, node=agent.mapper.nodes[agent.current_node_idx], episode_idx=episode_idx)
                    if not cycle.geometry_available:
                        termination = "observation_geometry_unavailable"
                        break
                    guided.observation_updated()
        elif state == "select":
            if not guided.select(episode_idx):
                termination = "search_exhausted"
                break
        elif state == "recovery":
            state = guided.recover(guided.recovery_context, episode_idx)
            if state in {"search_exhausted", "decision_budget_exhausted"}:
                termination = state
                break
        if state == "verified_finish":
            agent.found_goal = True
            if stop_on_success:
                agent.obs = agent.execution.step(0)
                agent.update_trajectory()
            termination = "verified_finish"
            break
    if guided.tools.motion.active_request_id is not None:
        guided.tools.motion.cancel()
        guided.tools.motion.poll_active()
    result = {
        "reason": termination, "decision_owner": "baseline_guided",
        "observation_mode": agent.observation_mode,
        "model_calls": guided.runtime.model_calls, "steps": agent.episode_steps,
        "task_complete": agent.task.progress.completed,
    }
    (guided.directory / "termination.json").write_text(json.dumps(result, indent=2))
    return result
