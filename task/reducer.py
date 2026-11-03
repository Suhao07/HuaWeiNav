"""根据验证事实推进任务；验证服务只负责生成判断。"""
from instruction.contracts import TargetSpec
from task.state import TaskProgress
from verification.contracts import CandidateInstance, VerificationResult
from verification.images import image_dependencies_valid


def apply_verification(
    state: TaskProgress,
    target: TargetSpec,
    candidate: CandidateInstance,
    result: VerificationResult,
    evidence: dict,
) -> bool:
    """Apply a final verification result and return task completion.

    Args:
        state: Task-owned execution progress.
        target: Explicit semantic target selected by scene query.
        candidate: Observed instance being verified.
        result: Semantic and stop-view judgment.
        evidence: Observations and relation context supporting the result.

    Returns:
        Whether the task is complete after this update.

    核心规则：数量为已接受实例 UID 集合的基数，N_i = |A_i|。
    uncertain 保持进度；关系失败属于对象对，只有实例反证记录为候选拒绝。
    """
    # 异步验证只影响其发起时的任务版本；迟到结果保留在日志中。
    if result.task_revision != state.task.revision:
        return state.completed
    if result.target_uid != candidate.uid:
        return state.completed
    if not image_dependencies_valid(result.dependencies):
        return state.completed
    # 顺序约束作用于整个事件，补拍和拒绝也只能更新当前目标。
    if state.task.ordered and target.id != state.active_target().id:
        return state.completed
    progress = state.target_progress[target.id]
    if result.accepted:
        if not candidate.uid:
            raise ValueError("Verified candidate requires an instance UID")
        if candidate.uid not in progress.accepted_candidate_uids:
            progress.accepted_candidate_uids.append(candidate.uid)
        _clear_pending_pair(state)
        if state.task.ordered:
            if progress.complete and state.active_target_index < len(state.target_progress) - 1:
                state.active_target_index += 1
        all_required = (
            state.task.ordered or state.task.exhaustive
            or state.task.completion_mode in ("all_targets_success", "all", "exhaustive")
        )
        completed = [p.complete for p in state.target_progress.values()]
        state.completed = all(completed) if all_required else any(completed)
        return state.completed
    if result.decision == "need_better_view" and result.semantic_satisfied:
        relation_context = evidence.get("verified_relation_context", {})
        state.mode = "better_view_for_verified_pair"
        state.pending_verified_pair = {
            "target_id": target.id,
            "candidate_uid": candidate.uid,
            "candidate_record": candidate.as_dict(),
            "relation_context": dict(relation_context),
            "edge": dict(relation_context.get("edge") or {}),
            "decision": result.decision,
            "confidence": result.confidence,
            "reason": result.reason,
            "view_objective": dict(result.view_objective),
            "evidence": {
                "current_rgb_with_bbox_path": evidence.get("current_rgb_with_bbox_path", ""),
                "centered_view_path": evidence.get("centered_view_path", ""),
                "view_quality_facts": dict(evidence.get("view_quality_facts") or {}),
            },
        }
    elif result.decision in ("reject_candidate", "reject_relation"):
        if state.pending_pair_candidate_uid() == candidate.uid:
            _clear_pending_pair(state)
        if result.decision == "reject_candidate":
            if candidate.uid not in progress.rejected_candidate_uids:
                progress.rejected_candidate_uids.append(candidate.uid)
    return state.completed


def _clear_pending_pair(state: TaskProgress) -> None:
    state.mode = "search"
    state.pending_verified_pair = {}
