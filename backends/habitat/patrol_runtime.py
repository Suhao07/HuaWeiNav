"""Task-free patrol using STRIVE's room-first exploration and motion owner."""
import json
from pathlib import Path

from motion.contracts import MotionGoalMode, NavigationIntent
from navigation.planner_loop import run_observation_mapping_cycle
from planning.exploration_policy import select_exploration_node
from scene.contracts import Pose3D


def run_patrol(agent, *, episode_idx, max_steps):
    """Collect complete panoramas between baseline frontier moves.

    Args:
        agent: Existing agent initialized with ``reset(..., patrol=True)``.
        episode_idx: Artifact directory index, independent of task identity.
        max_steps: Observation-step limit in the existing agent convention;
            reset records step one, and each executed action adds one step.

    Returns:
        Receipt distinguishing coverage exhaustion, budget and motion failure.

    同一房间内沿用 n*=argmin d_nav(x,n)。完整观察后提交节点覆盖状态；
    预算不足以容纳 12 次全景转向时结束采集，保留尚未观察的节点。
    """
    if not agent.patrol or agent.task is not None:
        raise ValueError("Patrol requires task-free reset")
    directory = Path(agent.save_dir) / f"episode-{episode_idx}" / "patrol"
    directory.mkdir(parents=True, exist_ok=False)
    execution = agent.execution
    execution.controller.agent = agent
    sequence, scans = 0, 0
    termination = "step_budget"

    def record(event, **fields):
        with (directory / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": agent.episode_steps, "event": event,
                                     **fields}) + "\n")

    # 巡视使用完整全景，不启用依赖任务目标的自适应提前停止。
    agent.adaptive_observation = False
    node = None
    try:
        while not agent.env.episode_over:
            if agent.episode_steps + 12 > max_steps:
                break
            agent.on_node_flag = True
            cycle = run_observation_mapping_cycle(agent, node=node, episode_idx=episode_idx)
            if cycle.episode_over:
                termination = "environment_limit"
                break
            scans += 1
            if node is not None:
                agent.mapper.change_state(node)
            record("observed", node_index=agent.current_node_idx)
            node, source = select_exploration_node(agent, episode_idx=episode_idx,
                step=agent.episode_steps, relocate=False, use_gpt_relocate=False)
            if node is None:
                termination = "frontiers_exhausted"
                break
            intent = NavigationIntent(MotionGoalMode.GO_TO_FRONTIER,
                goal_pose=Pose3D(tuple(float(x) for x in node.position), frame_id="mapper_local"),
                metadata={"node_index": node.idx, "decision_source": source})
            sequence += 1
            execution.motion.submit(intent, intent.to_motion_goal(), request_id=f"patrol-{sequence}")
            record("submitted", node_index=node.idx, decision_source=source)
            feedback = None
            while execution.motion.active_request_id is not None and agent.episode_steps < max_steps:
                execution.controller.advance()
                feedback = execution.motion.poll_active()
            if execution.motion.active_request_id is not None:
                break
            record("motion_terminal", status=feedback.status.status.value, node_index=node.idx)
            if not feedback.status.succeeded():
                termination = "motion_" + feedback.status.status.value
                break
            agent.current_node_idx = node.idx
            agent.mapper.traj.append(node.idx)
        else:
            termination = "environment_limit"
    finally:
        if execution.motion.active_request_id is not None:
            execution.motion.cancel()
            feedback = execution.motion.poll_active()
            record("motion_terminal", status=feedback.status.status.value,
                   node_index=feedback.intent.metadata["node_index"], reason="patrol_exit")
    receipt = {"termination": termination, "steps": agent.episode_steps,
               "complete_panoramas": scans, "submitted_goals": sequence,
               "path_length_m": float(agent.travel_distance),
               "scope": "task_free_baseline_patrol", "task_complete": None}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
