"""验证结论的作用域：实例、对象对与证据不足分别处理。"""
import pytest

from instruction_adapter.contracts import InstructionPlan, TargetQuery
from task.state import TaskProgress
from instruction.compiler import task_from_parsed_plan
from instruction_adapter.verifier import CandidateInstance, VerificationResult
from task.reducer import apply_verification


@pytest.mark.parametrize("decision", ["uncertain", "need_relation_check", "reject_relation"])
def test_non_instance_failure_keeps_candidate_available(decision):
    target = TargetQuery(id="cup", name="cup")
    plan = InstructionPlan(raw_instruction="find a cup", targets=[target])
    state = TaskProgress(task_from_parsed_plan(plan))
    result = VerificationResult(satisfied=False, decision=decision, task_revision=1, target_uid="cup-1")
    apply_verification(state, target, CandidateInstance("cup-1", "cup"), result, {})
    progress = state.target_progress["cup"]
    assert progress.accepted_candidate_uids == []
    assert progress.rejected_candidate_uids == []
    assert not state.completed


def test_verified_instance_is_counted_once_and_explicit_rejection_is_local():
    target = TargetQuery(id="cup", name="cup", min_count=2)
    plan = InstructionPlan(raw_instruction="find two cups", targets=[target])
    state = TaskProgress(task_from_parsed_plan(plan))
    accepted = VerificationResult(True, "accept", semantic_satisfied=True,
                                  view_sufficient_for_stop=True, task_revision=1, target_uid="cup-1")
    for _ in range(2):
        assert not apply_verification(state, target, CandidateInstance("cup-1", "cup"), accepted, {})
    rejected = VerificationResult(False, "reject_candidate", task_revision=1, target_uid="cup-2")
    apply_verification(state, target, CandidateInstance("cup-2", "cup"), rejected, {})
    assert state.target_progress["cup"].count == 1
    assert state.target_progress["cup"].rejected_candidate_uids == ["cup-2"]
    accepted.target_uid = "cup-3"
    assert apply_verification(state, target, CandidateInstance("cup-3", "cup"), accepted, {})


@pytest.mark.parametrize("result_revision", [None, 1])
def test_result_from_another_revision_preserves_current_progress(result_revision):
    target = TargetQuery(id="cup", name="cup")
    plan = InstructionPlan(raw_instruction="find a cup", targets=[target])
    state = TaskProgress(task_from_parsed_plan(plan, revision=2))
    result = VerificationResult(True, "accept", semantic_satisfied=True,
                                view_sufficient_for_stop=True, task_revision=result_revision, target_uid="cup-1")
    before = state.as_dict()
    assert not apply_verification(state, target, CandidateInstance("cup-1", "cup"), result, {})
    assert state.as_dict() == before


def test_progress_uses_frozen_count_when_runtime_plan_changes():
    target = TargetQuery(id="cup", name="cup", min_count=2)
    plan = InstructionPlan(raw_instruction="find two cups", targets=[target])
    state = TaskProgress(task_from_parsed_plan(plan))
    target.min_count = 1
    for uid, completed in [("cup-1", False), ("cup-2", True)]:
        result = VerificationResult(True, "accept", semantic_satisfied=True,
                                    task_revision=1, target_uid=uid)
        assert apply_verification(state, target, CandidateInstance(uid, "cup"), result, {}) == completed


def test_session_installs_new_revision_and_preserves_same_revision_progress():
    from dataclasses import replace
    from instruction_adapter.task_compat import runtime_plan
    from scene.contracts import Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    plan = InstructionPlan(raw_instruction="find a cup", targets=[TargetQuery(id="cup", name="cup")])
    spec = task_from_parsed_plan(plan)
    snapshot = SemanticMapSnapshot(0.0, Pose3D((0.0, 0.0, 0.0)))
    session = TaskSession(snapshot, runtime_plan(spec))
    result = VerificationResult(True, "accept", semantic_satisfied=True,
                                task_revision=1, target_uid="cup-1")
    apply_verification(session.progress, spec.targets[0], CandidateInstance("cup-1", "cup"), result, {})
    session.install_plan(runtime_plan(spec))
    assert session.progress.completed
    changed = replace(spec, raw_instruction="find another cup")
    with pytest.raises(ValueError, match="newer revision"):
        session.install_plan(runtime_plan(changed))
    session.install_plan(runtime_plan(replace(changed, revision=2)))
    assert session.spec.revision == 2
    assert not session.progress.completed
    assert session.progress.target_progress["cup"].count == 0


@pytest.mark.parametrize("decision", ["accept", "need_better_view", "reject_candidate", "reject_relation"])
def test_future_target_result_preserves_current_ordered_task(decision):
    from copy import deepcopy
    from instruction_adapter.contracts import ExecutionPolicy

    plan = InstructionPlan(
        raw_instruction="find a cup then a chair",
        targets=[TargetQuery(id="cup", name="cup"), TargetQuery(id="chair", name="chair")],
        execution=ExecutionPolicy(ordered=True),
    )
    state = TaskProgress(task_from_parsed_plan(plan))
    cup = CandidateInstance("cup-1", "cup")
    apply_verification(state, state.task.targets[0], cup, VerificationResult(
        False, "need_better_view", semantic_satisfied=True, task_revision=1, target_uid=cup.uid,
    ), {})
    before = deepcopy(state.as_dict())
    result = VerificationResult(
        decision == "accept", decision, semantic_satisfied=True,
        task_revision=1, target_uid="chair-1",
    )
    assert not apply_verification(state, state.task.targets[1], CandidateInstance("chair-1", "chair"), result, {})
    assert state.as_dict() == before
