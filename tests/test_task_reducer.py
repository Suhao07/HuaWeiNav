"""验证结论的作用域：实例、对象对与证据不足分别处理。"""
from copy import deepcopy
from dataclasses import replace

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


@pytest.fixture
def locked_session(monkeypatch):
    from instruction_adapter.task_compat import runtime_plan
    from scene.contracts import Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = InstructionPlan(raw_instruction="find two red cups", targets=[
        TargetQuery(id="cup", name="cup", min_count=2, attributes={"color": "red"}),
    ])
    spec = task_from_parsed_plan(plan, revision=2)
    session = TaskSession(SemanticMapSnapshot(1.0, Pose3D((0.0, 0.0, 0.0))),
                          runtime_plan(spec), interpretation_locked=True)
    for uid, decision in [("cup-1", "accept"), ("cup-2", "reject_candidate")]:
        result = VerificationResult(decision == "accept", decision, semantic_satisfied=decision == "accept",
                                    task_revision=2, target_uid=uid)
        session.verification_ledger.put(spec.raw_instruction, uid, result)
        apply_verification(session.progress, spec.targets[0], CandidateInstance(uid, "cup"), result, {})
    session.revise_search([{"concept_id": "cup", "detector_terms": ["mug"],
                           "basis_ids": ["instruction"], "reason": "Alternate retrieval"}],
                          task_revision=2, search_revision=0, snapshot=session.snapshot)
    session.interpretation_history.append({"task_revision": 1, "reason": "Audited earlier interpretation"})
    session.destination_hypotheses.append({"candidate_uid": "cup-1", "task_revision": 2})
    session.prior_result = {"ranking": ["cup-1"]}
    session.prior_map_prompt_context = {"room": "kitchen"}
    session.prior_map_high_level_selection = {"frontier_uid": "node:1"}
    session.prior_map_diagnostics = {"aligned": True}
    return session


def _session_state(session):
    return deepcopy({
        "spec": session.spec.as_dict(), "plan": session.plan.as_dict(),
        "progress": session.progress.as_dict(), "evidence": session.verification_ledger.as_dict(),
        "anchors": vars(session.anchor_search_ledger), "pairs": vars(session.relation_pair_ledger),
        "identities": session.constraint_evaluator.identity_records,
        "target": session.target, "destinations": session.destination_hypotheses,
        "search_revision": session.search_revision, "search": session.search_hypotheses,
        "history": session.interpretation_history, "prior": session.prior_result,
        "prior_context": session.prior_map_prompt_context,
        "prior_selection": session.prior_map_high_level_selection,
        "prior_diagnostics": session.prior_map_diagnostics,
    })


@pytest.mark.parametrize("entry,change", [
    ("install", "revision"), ("install", "instruction"), ("install", "target"),
    ("install", "same_revision"), ("install", "older_revision"), ("adopt", "target"),
])
def test_locked_session_rejects_changed_contract_without_mutation(locked_session, entry, change):
    from instruction_adapter.task_compat import runtime_plan

    session = locked_session
    spec = replace(session.spec, revision=3)
    if change == "instruction":
        spec = replace(spec, raw_instruction="find any cup")
    elif change in {"target", "same_revision", "older_revision"}:
        spec = replace(spec, targets=(replace(spec.targets[0], min_count=1),),
                       revision={"target": 3, "same_revision": 2, "older_revision": 1}[change])
    plan = runtime_plan(spec)
    references, before = dict(vars(session)), _session_state(session)
    assert session.progress.target_progress["cup"].count == 1
    assert session.progress.target_progress["cup"].rejected_candidate_uids == ["cup-2"]
    with pytest.raises(ValueError, match="interpretation is locked"):
        if entry == "install":
            session.install_plan(plan)
        else:
            session.adopt_interpretation(plan, task_revision=2, search_revision=1,
                                         snapshot=session.snapshot, record={"reason": "Injected revision"})
    assert _session_state(session) == before
    assert all(getattr(session, name) is value for name, value in references.items())


def test_locked_session_allows_identical_contract_refresh(locked_session):
    from instruction_adapter.task_compat import runtime_plan

    session = locked_session
    references, before = dict(vars(session)), _session_state(session)
    session.install_plan(session.plan)
    assert _session_state(session) == before
    replacement = runtime_plan(session.spec)
    session.install_plan(replacement)
    assert session.plan is replacement and session.search_revision == 2
    assert session.detector_terms == ["mug"]
    assert all(concept.detector_terms == ["mug"] for concept in session.search_concepts)
    after = _session_state(session)
    assert {key: value for key, value in after.items() if key != "search_revision"} == {
        key: value for key, value in before.items() if key != "search_revision"}
    assert all(getattr(session, name) is value for name, value in references.items()
               if name not in {"plan", "search_revision"})


def test_locked_session_search_revision_preserves_semantics_and_evidence(locked_session):
    session = locked_session
    references, before = dict(vars(session)), _session_state(session)
    updates = session.revise_search([{"concept_id": "cup", "detector_terms": ["teacup"],
                                     "basis_ids": ["instruction"], "reason": "Try a narrower retrieval"}],
                                    task_revision=2, search_revision=1, snapshot=session.snapshot)
    assert session.search_revision == 2 and session.detector_terms == ["teacup"]
    assert updates == [session.search_hypotheses["cup"]]
    assert all(concept.detector_terms == ["teacup"] for concept in session.search_concepts)
    after = _session_state(session)
    changed_fields = {"plan", "search_revision", "search"}
    assert {key: value for key, value in after.items() if key not in changed_fields} == {
        key: value for key, value in before.items() if key not in changed_fields}
    assert all(getattr(session, name) is value for name, value in references.items()
               if name != "search_revision")


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
