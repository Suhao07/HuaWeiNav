"""Regression cases for separating task semantics from search and progress."""

from copy import deepcopy

from instruction_adapter.contracts import (
    ConceptQuery, Constraint, ExecutionPolicy, InstructionPlan, TargetQuery,
)
import pytest
from task.state import TaskProgress
from instruction.compiler import task_from_parsed_plan
from instruction_adapter.grounding import ground_plan
from planning.mode_policy import is_anchor_first_relation_search


def test_task_spec_is_deeply_immutable_and_export_is_detached():
    from dataclasses import FrozenInstanceError
    import pytest
    from instruction.contracts import TaskSpec, TargetSpec

    attributes = {"colors": ["red"]}
    task = TaskSpec("red cup", (TargetSpec("cup", "cup", attributes=attributes),))
    attributes["colors"].append("blue")
    assert task.targets[0].attributes["colors"] == ("red",)
    with pytest.raises(FrozenInstanceError):
        task.ordered = True
    with pytest.raises(TypeError):
        task.targets[0].attributes["colors"] = ()
    exported = task.as_dict()
    exported["targets"][0]["attributes"]["colors"].append("green")
    assert task.targets[0].attributes["colors"] == ("red",)


def test_strict_compiler_does_not_replace_unavailable_parse(monkeypatch):
    from instruction.compiler import compile_instruction

    monkeypatch.setenv("LLM_OFFLINE", "1")
    result = compile_instruction("Find two cups then a chair; avoid the bedroom", set())
    assert result.status == "parse_unavailable"
    assert result.task is None


def test_objectnav_input_needs_no_model(monkeypatch):
    from backends.habitat.task_input import adapt_objectnav_task

    monkeypatch.setenv("LLM_OFFLINE", "1")
    task = adapt_objectnav_task("chair")
    assert task.targets[0].name == "chair"
    assert task.source == "objectnav_category"


def test_compat_projection_cannot_change_task_contract():
    from backends.habitat.task_input import adapt_objectnav_task
    from instruction_adapter.task_compat import runtime_plan

    task = adapt_objectnav_task("chair")
    plan = runtime_plan(task)
    plan.targets[0].name = "table"
    plan.execution.ordered = True
    assert task.targets[0].name == "chair"
    assert not task.ordered


def test_compiler_rejects_unsupported_hard_condition(monkeypatch):
    from instruction import compiler

    monkeypatch.setattr(compiler, "parse_instruction_with_llm", lambda *a, **k: InstructionPlan(
        raw_instruction="find a cup without entering the bedroom",
        targets=[TargetQuery(id="cup", name="cup")],
        constraints=[Constraint(type="avoid_room", value="bedroom", hardness="hard")],
    ))
    result = compiler.compile_instruction("find a cup without entering the bedroom", {"relation"})
    assert result.status == "unsupported"
    assert result.task is None


def test_grounding_preserves_order_and_count_with_anchor(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = InstructionPlan(
        raw_instruction="Find two cups on the table, then a chair",
        targets=[TargetQuery(id="cups", name="cup", min_count=2),
                 TargetQuery(id="chair", name="chair")],
        constraints=[Constraint(type="relation", subject="cup", relation="on",
                                object="table", hardness="hard")],
        execution=ExecutionPolicy(mode="all_targets_success", ordered=True, exhaustive=True),
    )
    before = deepcopy(plan.execution.as_dict())
    ground_plan(plan, available_classes=["cup", "chair", "table"])
    assert plan.execution.as_dict() == before
    assert is_anchor_first_relation_search(plan)
    assert plan.targets[0].min_count == 2


def test_progress_does_not_mutate_plan_or_accept_future_target():
    plan = InstructionPlan(
        raw_instruction="Find two cups, then a chair",
        targets=[TargetQuery(id="cups", name="cup", min_count=2),
                 TargetQuery(id="chair", name="chair")],
        execution=ExecutionPolicy(ordered=True),
    )
    before = deepcopy(plan.as_dict())
    progress = TaskProgress(task_from_parsed_plan(plan))
    from task.reducer import apply_verification
    from verification.contracts import CandidateInstance, VerificationResult

    def accept(target, uid):
        return apply_verification(
            progress, target, CandidateInstance(uid, target.name),
            VerificationResult(True, "accept", semantic_satisfied=True,
                               task_revision=1, target_uid=uid), {},
        )

    assert not accept(plan.targets[1], "chair-1")
    assert not accept(plan.targets[0], "cup-1")
    assert not accept(plan.targets[0], "cup-1")
    assert progress.target_progress["cups"].count == 1
    assert not accept(plan.targets[0], "cup-2")
    assert progress.active_target().id == "chair"
    with pytest.raises(ValueError, match="instance UID"):
        accept(plan.targets[1], "")
    assert accept(plan.targets[1], "chair-1")
    assert plan.as_dict() == before
