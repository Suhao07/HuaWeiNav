"""Baseline navigation skills with event-scoped Agent recovery in Habitat."""
from pathlib import Path
import json
import math
import os
import numpy as np

from instruction_adapter.view_control import ViewControlState

from backends.habitat.agent_runtime import HabitatNavigationTools
from backends.habitat.query import select_observed_candidate
from harness.runtime import AgentRuntime
from motion.contracts import MotionGoalMode, NavigationIntent
from navigation.guided import GuidedPolicy, GuidedVerification, run_guided_loop
from navigation.planner_loop import run_observation_mapping_cycle
from navigation_core.view_geometry import camera_pitch
from planning.exploration_policy import (select_baseline_exploration_node, select_exploration_node,
                                         select_prior_region_node)
from scene.contracts import Pose3D


class GuidedNavigation(GuidedPolicy):
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
        self.initialize_state(deferred_candidates=self.tools.deferred_candidates,
                              review_interval=int(os.getenv("STRIVE_SEARCH_REVIEW_STEPS", "100")))
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
            metadata["candidate_action_epoch"] = self.action_epoch(candidate_uid)
            # 所有对象视点使用局部规划器到达容差，与视点可行条件一致。
        intent = NavigationIntent(mode=mode, target_object_uid=candidate_uid,
                                  goal_pose=Pose3D(position, frame_id="mapper_local"), metadata=metadata)
        self.tools.motion.submit(intent, intent.to_motion_goal(),
                                 request_id=f"guided-{self.sequence}",
                                 control_epoch=self.tools.motion.control_epoch)
        self.record("submitted", decision_source=source, node_index=metadata.get("node_index"),
                    candidate_uid=candidate_uid, position=position)

    @property
    def step(self):
        """Return the executed Habitat step count."""
        return self.agent.episode_steps

    @property
    def revision(self):
        """Return the installed task revision."""
        return self.agent.task.spec.revision

    @property
    def arrival_tolerance(self):
        """Use the local planner's XY arrival radius for view history."""
        return self.agent.planner._goal_radius

    @property
    def candidate_uid(self):
        """Return the currently selected instance."""
        return self.agent.object_final.uid

    @property
    def view_state(self):
        """Expose baseline feedback state to the shared policy."""
        return self.agent.view_control_state

    @property
    def scan_complete(self):
        """Return whether adaptive observation covered this node."""
        return self.agent.scan_complete

    def evidence_fingerprint(self, uid):
        """Return visual content identity, or unknown for a missing instance."""
        obj = self.tools.objects.get(uid)
        return None if obj is None else self.candidate_evidence(obj)

    def refresh_scene(self):
        """Publish the latest mapped instances to navigation tools."""
        self.tools.refresh_scene()

    def select_candidate(self, episode_idx):
        """Select an observed candidate that is not deferred."""
        agent = self.agent
        agent.on_node_flag = True
        agent.found_goal, agent.object_final = select_observed_candidate(
            agent, episode_idx=episode_idx, step=self.step,
            deferred_uids=self.deferred_candidates)
        self.refresh_scene()
        return agent.object_final.uid if agent.found_goal else None

    def initial_views(self, uid):
        """Prepare the baseline approach and restore revision-local view state."""
        agent = self.agent
        agent.view_control_state = self.candidate_view_states.setdefault((self.revision, uid), ViewControlState())
        agent.found_goal_position = agent.mapper.current_position.copy()
        agent.confirmed_target_waypoint = None
        agent.find_final_waypoint()
        agent.whether_to_check_again()
        return self.tools.publish_object_viewpoints(uid)

    def feedback_views(self, uid):
        """Publish the existing feedback controller's view proposals."""
        return self.tools.publish_object_viewpoints(uid)

    def observe_pending(self, episode_idx):
        """Continue the adaptive scan without consuming an exploration branch."""
        run_observation_mapping_cycle(self.agent, node=self.node_for_index(self.agent.current_node_idx),
                                      episode_idx=episode_idx)

    def select_exploration(self, episode_idx):
        """Delegate ordinary exploration to the existing room-first policy."""
        return select_exploration_node(self.agent, episode_idx=episode_idx, step=self.step)

    def node_for_index(self, index):
        """Resolve a published node index to its mapper node."""
        return self.agent.mapper.nodes[index]

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

    def review_options(self, node, source, episode_idx):
        """Return (default node, source, rows), retaining a recomputed prior choice."""
        if source == "prior_region_policy":
            return self.exploration_options(episode_idx)
        rows = [row for row in self.agent.mapper.prior_map_last_chosen_frontier["candidates"]
                if math.isfinite(row["raw_distance_m"])]
        return node, source, rows

    def capture_verification(self, scored_position):
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
        from navigation.goal_approach_controller import rotate_toward_object_for_recheck
        agent = self.agent
        agent.need_check_again = False
        if not rotate_toward_object_for_recheck(agent):
            return None
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
        agent.final_verification_result = None
        complete = agent.final_instruction_check(evidence_override=agent.last_check_again_evidence)
        result = agent.final_verification_result
        return GuidedVerification(
            candidate_uid=agent.object_final.uid, role=agent.selected_candidate.role,
            target_id=agent.selected_candidate.target_id,
            decision="anchor_searched" if agent.selected_candidate.role == "anchor" else agent.final_stop_decision,
            complete=complete, pose_gap=pose_gap,
            relation_outcomes=result.view_objective.get("relation_outcomes", []) if result is not None else [])

    def restore_camera(self):
        """Restore horizontal observations before further mapping or motion."""
        from navigation.goal_approach_controller import tilt_camera
        return tilt_camera(self.agent, 0.0)

    def continue_view_control(self):
        """Consult the baseline directed-view feedback controller."""
        return self.agent.need_check_again or self.agent._continue_better_view_control()


class HabitatGuidedExecution:
    """Adapt simulator effects to the shared guided execution lifecycle."""

    def __init__(self, guided, agent, *, episode_idx, max_steps, stop_on_success):
        self.guided = guided
        self.agent = agent
        self.episode_idx = episode_idx
        self.max_steps = max_steps
        self.stop_on_success = stop_on_success

    def initialize(self):
        """Run the original initial observation before selecting a destination."""
        self.agent.on_node_flag = True
        run_observation_mapping_cycle(self.agent, node=None, episode_idx=self.episode_idx)

    def limit_reason(self):
        """Enforce physical episode limits, not the budget of an accepted choice."""
        if self.agent.env.episode_over or self.agent.episode_steps >= self.max_steps:
            return "episode_limit"
        return None

    def advance(self):
        """Execute one accepted waypoint primitive."""
        self.agent.execution.controller.advance()

    def after_motion(self, feedback):
        """Habitat already stores the primitive observation during advance."""

    def _frontier_node(self, feedback):
        metadata = feedback.intent.metadata
        index = metadata.get("node_index")
        if metadata.get("frontier_uid") is not None:
            index = int(metadata["frontier_uid"].split(":")[1])
        return self.agent.mapper.nodes[index]

    def frontier_arrived(self, feedback):
        """Observe the reached node before publishing evidence or marking coverage."""
        node = self._frontier_node(feedback)
        self.agent.current_node_idx = node.idx
        self.agent.mapper.traj.append(node.idx)
        run_observation_mapping_cycle(self.agent, node=node, episode_idx=self.episode_idx)

    def frontier_observed(self, feedback):
        """Mark explored only after evidence refresh and a completed scan."""
        if not self.agent.env.episode_over and self.agent.scan_complete:
            self.agent.mapper.change_state(self._frontier_node(feedback))

    def object_arrived(self, feedback):
        """Resolve the executed instance for baseline verification."""
        self.agent.object_final = self.guided.tools.objects[feedback.intent.target_object_uid]

    def finish_success(self):
        """Issue successful STOP only when this episode owns it."""
        self.agent.found_goal = True
        if self.stop_on_success:
            self.agent.obs = self.agent.execution.step(0)
            self.agent.update_trajectory()


def run_guided_episode(agent, *, episode_idx, max_steps, max_model_calls=100,
                       stop_on_success=True, output_dir=None):
    """Execute baseline skills, optionally leaving STOP ownership to the sequence driver."""
    directory = (Path(output_dir) if output_dir is not None
                 else Path(agent.save_dir) / f"episode-{episode_idx}" / "harness")
    guided = GuidedNavigation(agent, directory, max_model_calls)
    termination = run_guided_loop(guided, HabitatGuidedExecution(
        guided, agent, episode_idx=episode_idx, max_steps=max_steps, stop_on_success=stop_on_success))
    result = {
        "reason": termination, "decision_owner": "baseline_guided",
        "model_calls": guided.runtime.model_calls, "steps": agent.episode_steps,
        "task_complete": agent.task.progress.completed,
    }
    (guided.directory / "termination.json").write_text(json.dumps(result, indent=2))
    return result
