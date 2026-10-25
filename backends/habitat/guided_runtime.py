"""Baseline navigation skills with event-scoped Agent recovery in Habitat."""
from pathlib import Path
import json

from backends.habitat.agent_runtime import HabitatNavigationTools
from backends.habitat.query import select_observed_candidate
from harness.runtime import AgentRuntime
from motion.contracts import MotionGoalMode, NavigationIntent
from navigation.planner_loop import run_observation_mapping_cycle
from planning.exploration_policy import select_exploration_node
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
        agent.adaptive_observation = True
        agent.semantic_scan_headings = {}
        agent.semantic_label_evidence = {}
        agent.scan_complete = False
        agent.mapper.coverage_guided_exploration = True
        agent.mapper.frontier_distance_tie_m = agent.env.sim.sim_config.agents[0].action_space[1].actuation.amount

    def record(self, event, **fields):
        """Append the actual state transition without hiding execution errors."""
        with (self.directory / "decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": self.agent.episode_steps, "event": event,
                                     **fields}, ensure_ascii=False) + "\n")

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
            # 所有对象视点使用局部规划器到达容差，与视点可行条件一致。
        intent = NavigationIntent(mode=mode, target_object_uid=candidate_uid,
                                  goal_pose=Pose3D(position, frame_id="mapper_local"), metadata=metadata)
        self.tools.motion.submit(intent, intent.to_motion_goal(),
                                 request_id=f"guided-{self.sequence}",
                                 control_epoch=self.tools.motion.control_epoch)
        self.record("submitted", decision_source=source, node_index=metadata.get("node_index"),
                    candidate_uid=candidate_uid, position=position)

    def select(self, episode_idx):
        """Query candidates once per decision cycle, then apply baseline order."""
        agent = self.agent
        agent.on_node_flag = True
        agent.found_goal, agent.object_final = select_observed_candidate(
            agent, episode_idx=episode_idx, step=agent.episode_steps)
        self.tools.refresh_scene()
        if agent.found_goal:
            agent.found_goal_position = agent.mapper.current_position.copy()
            agent.confirmed_target_waypoint = None
            agent.find_final_waypoint()
            agent.whether_to_check_again()
            self.submit(agent.check_again_postion, candidate_uid=agent.object_final.uid,
                        source="baseline_viewpoint")
            return True
        if not agent.scan_complete:
            run_observation_mapping_cycle(agent, node=agent.mapper.nodes[agent.current_node_idx], episode_idx=episode_idx)
            return True
        node, source = select_exploration_node(agent, episode_idx=episode_idx,
                                              step=agent.episode_steps)
        if node is None:
            self.record("search_exhausted", decision_source=source)
            return False
        # selected 由 active goal 表达；观察完成后才更新地图的 explored 状态。
        self.submit(node.position, node=node, source=source)
        return True

    def verify(self):
        """Reuse baseline evidence capture, final verifier and view-control."""
        from navigation.goal_approach_controller import rotate_toward_object_for_recheck
        agent = self.agent
        agent.need_check_again = False
        if not rotate_toward_object_for_recheck(agent):
            return "episode_limit"
        agent.check_again(agent.episode_steps)
        complete = agent.final_instruction_check(evidence_override=agent.last_check_again_evidence)
        self.record("verified", candidate_uid=agent.object_final.uid,
                    decision=agent.final_stop_decision, task_complete=complete)
        if complete:
            return "verified_finish"
        if agent.need_check_again or agent._continue_better_view_control():
            self.submit(agent.check_again_postion, candidate_uid=agent.object_final.uid,
                        source="baseline_view_feedback")
            return "moving"
        if agent._pending_better_view_active():
            self.recovery_context = {"reason": "view_options_exhausted",
                                     "candidate_uid": agent.object_final.uid,
                                     "view_objective": agent.view_control_state.objective}
            return "recovery"
        return "select"

    def recover(self, reason):
        """Request one Agent proposal only for an explicit recovery event."""
        self.record("agent_intervention", reason=reason, decision_source="agent")
        self.tools.refresh_scene()
        self.runtime.history.append({"event": "recovery_required", "reason": reason})
        return self.runtime.step()


def run_guided_episode(agent, *, episode_idx, max_steps, max_model_calls=100):
    """Execute baseline skills; retain model calls for semantic/recovery events.

    Exploration arrivals update the scene. Object arrivals capture directed
    evidence, preserving the original verification/view-control feedback loop.
    """
    guided = GuidedNavigation(agent, Path(agent.save_dir) / f"episode-{episode_idx}" / "harness",
                              max_model_calls)
    agent.on_node_flag = True
    run_observation_mapping_cycle(agent, node=None, episode_idx=episode_idx)
    state, termination = "select", "episode_limit"
    while not agent.env.episode_over and agent.episode_steps < max_steps:
        if guided.tools.motion.active_request_id is not None:
            agent.execution.controller.advance()
            feedback = guided.tools.motion.poll_active()
            if feedback.tracking_active:
                continue
            guided.record("motion_terminal", status=feedback.status.status.value,
                          metadata=feedback.intent.metadata)
            if not feedback.status.succeeded():
                guided.recovery_context = {"reason": "motion_terminal_failure",
                                           "status": feedback.status.status.value,
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
                run_observation_mapping_cycle(agent, node=node, episode_idx=episode_idx)
                if not agent.env.episode_over and agent.scan_complete:
                    agent.mapper.change_state(node)
                state = "select"
            else:
                agent.object_final = guided.tools.objects[feedback.intent.target_object_uid]
                state = guided.verify()
        elif state == "select":
            if not guided.select(episode_idx):
                termination = "search_exhausted"
                break
        elif state == "recovery":
            if guided.runtime.model_calls >= max_model_calls:
                termination = "decision_budget_exhausted"
                break
            outcome = guided.recover(guided.recovery_context)
            if outcome["name"] == "propose_finish" and outcome["result"]["accepted"]:
                state = "verified_finish"
        if state == "verified_finish":
            agent.found_goal = True
            agent.obs = agent.execution.step(0)
            agent.update_trajectory()
            termination = "verified_finish"
            break
    if guided.tools.motion.active_request_id is not None:
        guided.tools.motion.cancel()
        guided.tools.motion.poll_active()
    (guided.directory / "termination.json").write_text(json.dumps({
        "reason": termination, "decision_owner": "baseline_guided",
        "model_calls": guided.runtime.model_calls, "steps": agent.episode_steps,
        "task_complete": agent.task.progress.completed,
    }, indent=2))
