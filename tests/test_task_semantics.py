"""Regression cases for separating task semantics from search and progress."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("category", ["chair", "bed", "dining table"])
def test_objectnav_input_needs_no_model(monkeypatch, category):
    from backends.habitat.task_input import adapt_objectnav_task
    from instruction_adapter.task_compat import runtime_plan

    monkeypatch.setenv("LLM_OFFLINE", "1")
    task = adapt_objectnav_task(category, revision=3)
    assert task.targets[0].name == category
    assert task.raw_instruction == f"Find {category}"
    assert task.source == "objectnav_category" and task.revision == 3
    assert not task.targets[0].demand
    assert not task.constraints and not task.search_priors
    assert runtime_plan(task).target_detector_prompts == [category]
    assert runtime_plan(task).target_match_terms == [category]


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


def _structured_parser_provider(monkeypatch, payload):
    """Replace only the provider; retain real schema, parser and compiler code."""
    from instruction import parser
    from prompting.schemas import ParsedInstruction

    requests = []

    def parse(**kwargs):
        assert kwargs["response_format"] is ParsedInstruction
        requests.append(kwargs)
        parsed = kwargs["response_format"].model_validate(payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(parser, "get_client_and_model", lambda _: (client, "test-only"))
    # The parser's offline guard precedes the provider boundary. The boundary
    # above is replaced before enabling this path; no live client is created.
    monkeypatch.setenv("LLM_OFFLINE", "0")
    return requests


@pytest.fixture(params=[
    pytest.param(
        ("I need somewhere soft and unoccupied to lie down, preferably in the bedroom.",
         "resting surface", "lie down", ["body"], "soft body-supporting surface",
         {"softness": "soft", "state": "unoccupied"}, ["bed", "sofa"], "bedroom"),
        id="soft-unoccupied-resting-surface"),
    pytest.param(
        ("I want to lay out plates and food on a clean surface, preferably in the kitchen.",
         "dining surface", "lay out plates and food", ["plates", "food"], "flat support surface",
         {"state": "clean"}, ["dining table", "kitchen counter"], "kitchen"),
        id="clean-surface-for-plates-and-food"),
])
def structured_demand(request, monkeypatch):
    """Supply explicit semantics without asking any model to infer them."""
    raw, name, activity, objects, resource, attributes, hypotheses, room = request.param
    payload = {
        "targets": [{
            "name": name,
            "attributes": dict(attributes),
            "demand": {"activity": activity, "manipulated_objects": list(objects),
                       "required_resource": resource, "destination_hypotheses": list(hypotheses)},
        }],
        "constraints": [
            {"type": "state" if key == "state" else "attribute", "subject": name,
             "value": {key: value}, "hardness": "hard", "verifier": "vlm"}
            for key, value in attributes.items()
        ] + [{"type": "room", "subject": name, "relation": "in", "value": room,
              "hardness": "soft", "verifier": "planner"}],
        "room_hints": [room],
    }
    return raw, payload, _structured_parser_provider(monkeypatch, payload)


def test_structured_demand_survives_real_parse_compile_and_is_deeply_immutable(structured_demand):
    from dataclasses import FrozenInstanceError
    from instruction.compiler import compile_instruction
    from instruction.contracts import TaskSpec
    from instruction.parser import parse_instruction_with_llm

    raw, payload, requests = structured_demand
    expected = {k: deepcopy(v) for k, v in payload["targets"][0]["demand"].items()
                if k != "destination_hypotheses"}
    initial = payload["targets"][0]["demand"]["destination_hypotheses"]
    plan = parse_instruction_with_llm(raw, strict=True)
    assert len(plan.targets) == 1  # Manipulated objects and guessed facilities are not goals.
    parsed_target = plan.targets[0]
    assert parsed_target.demand == expected
    assert plan.search_priors.destination_hypotheses == {parsed_target.id: initial}

    result = compile_instruction(raw, capabilities={"affordance", "attribute", "state"}, revision=7)
    assert result.status == "success", result.diagnostics
    assert len(requests) == 2
    assert all(raw in call["messages"][1]["content"] for call in requests)
    from pathlib import Path
    skill = (Path(__file__).resolve().parents[1] / "skills" / "understand_instruction.md").read_text(encoding="utf-8")
    assert all(skill in call["messages"][0]["content"] for call in requests)
    task = result.task
    assert len(task.targets) == 1
    target = task.targets[0]
    assert target.terminal and target.min_count == 1
    assert task.raw_instruction == raw and task.revision == 7
    assert target.id == parsed_target.id and target.name == parsed_target.name
    assert task.as_dict()["targets"][0]["demand"] == expected
    assert target.attributes == payload["targets"][0]["attributes"]
    assert target.demand["manipulated_objects"] == tuple(expected["manipulated_objects"])
    assert task.search_priors["destination_hypotheses"][target.id] == tuple(initial)
    assert not set(initial).intersection(target.aliases)

    # Explicit softness/state predicates and the soft room preference must not
    # be replaced by the generated hard resource predicate.
    explicit = task.as_dict()["constraints"][:-1]
    assert explicit == [
        {"id": f"c{i}", "type": c.type, "subject": c.subject, "relation": c.relation,
         "object": c.object, "value": c.value, "hardness": c.hardness,
         "verifier": c.verifier, "source": "llm"}
        for i, c in enumerate(plan.constraints)
    ]
    assert [c["value"] for c in explicit] == [c["value"] for c in payload["constraints"]]
    assert [c["hardness"] for c in explicit] == [c["hardness"] for c in payload["constraints"]]
    assert task.as_dict()["constraints"][-1] == {
        "id": f"demand:{target.id}", "type": "affordance", "subject": target.id,
        "relation": "", "object": "", "value": expected,
        "hardness": "hard", "verifier": "vlm", "source": "demand",
    }

    before = task.as_dict()
    with pytest.raises(FrozenInstanceError):
        task.raw_instruction = "Find a bed"
    for field in ("activity", "required_resource", "manipulated_objects"):
        with pytest.raises(TypeError):
            target.demand[field] = "replacement"
    with pytest.raises(TypeError):
        target.demand["manipulated_objects"][0] = "replacement"
    with pytest.raises(TypeError):
        task.constraints[-1].value["activity"] = "replacement"
    with pytest.raises(TypeError):
        target.attributes["state"] = "occupied"

    # Freezing also detaches from a mutable parsed plan, not just the provider.
    frozen_plan = task_from_parsed_plan(plan, revision=7)
    assert frozen_plan == task
    parsed_target.demand["activity"] = "different activity"
    parsed_target.demand["manipulated_objects"].append("another object")
    parsed_target.attributes["state"] = "occupied"
    plan.search_priors.destination_hypotheses[target.id].append("another facility")
    plan.raw_instruction = "Find a bed"
    exported = task.as_dict()
    exported["targets"][0]["demand"]["manipulated_objects"].append("changed export")
    exported["constraints"][-1]["value"]["required_resource"] = "changed export"
    assert frozen_plan.as_dict() == task.as_dict() == before
    assert TaskSpec.from_dict(before) == task


def test_demand_compilation_requires_affordance_capability(structured_demand):
    from instruction.compiler import compile_instruction

    raw, _, _ = structured_demand
    result = compile_instruction(raw, capabilities={"attribute", "state"})
    assert result.status == "unsupported"
    assert result.task is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].startswith("demand:")
    assert result.diagnostics[0].endswith(":affordance")


def test_initial_facility_guesses_are_retrieval_only_not_semantic_identity(structured_demand):
    from instruction.compiler import compile_instruction
    from instruction_adapter.task_compat import runtime_plan

    raw, payload, _ = structured_demand
    result = compile_instruction(raw, capabilities={"affordance", "attribute", "state"})
    assert result.status == "success", result.diagnostics
    task = result.task
    before = task.as_dict()
    plan = runtime_plan(task)
    target = plan.targets[0]
    initial = payload["targets"][0]["demand"]["destination_hypotheses"]
    assert target.detector_terms == initial
    assert target.aliases == list(task.targets[0].aliases)
    assert target.concept_query().demand == before["targets"][0]["demand"]
    assert "destination_hypotheses" not in target.demand
    assert "detector_terms" not in before["targets"][0]
    for terms in (target.match_terms, target.concept_query().match_terms,
                  plan.target_match_terms, plan.to_legacy_spec().target_aliases):
        assert not set(initial).intersection(terms)

    payload["targets"][0]["demand"]["destination_hypotheses"] = ["alternative facility"]
    alternative = compile_instruction(raw, capabilities={"affordance", "attribute", "state"})
    assert alternative.status == "success", alternative.diagnostics
    assert alternative.task.targets == task.targets
    assert alternative.task.constraints == task.constraints
    assert alternative.task.raw_instruction == task.raw_instruction
    assert alternative.task.search_priors != task.search_priors

    target.detector_terms.append("newly observed facility")
    target.aliases.append("changed runtime alias")
    target.demand["manipulated_objects"].clear()
    plan.search_priors.destination_hypotheses[target.id].clear()
    assert task.as_dict() == before
    assert runtime_plan(task).targets[0].detector_terms == initial


@pytest.mark.parametrize("backend", ["legacy", "rules"])
def test_legacy_entry_freezes_demand_before_runtime_verification(monkeypatch, structured_demand, backend):
    import instruction_adapter.compiler as adapter
    from instruction_adapter.verifier import FinalInstructionVerifier

    raw, _, requests = structured_demand
    monkeypatch.setattr(adapter, "ground_plan", lambda plan, **kwargs: plan)
    plan = adapter.compile_instruction_plan(raw, backend=backend)
    target = plan.targets[0]
    task = plan.diagnostics["task_spec"]
    assert len(requests) == 1 and task["raw_instruction"] == raw
    demand = next(c for c in task["constraints"] if c["source"] == "demand")
    assert demand["id"] == f"demand:{target.id}" and demand["hardness"] == "hard"
    assert demand["value"] == target.demand
    visual = FinalInstructionVerifier._visual_task(plan.as_dict(), {"target_id": target.id})
    assert demand["id"] in {c["id"] for c in visual["visual_predicates"]}
    assert visual["active_target"]["demand"] == target.demand
    assert task["revision"] == 1


@pytest.mark.parametrize("backend", ["legacy", "rules"])
def test_legacy_explicit_affordance_receives_stable_predicate_id(monkeypatch, backend):
    import instruction_adapter.compiler as adapter
    from instruction_adapter.verifier import FinalInstructionVerifier

    raw = "Find a surface able to support a large bowl."
    _structured_parser_provider(monkeypatch, {
        "targets": [{"name": "surface"}],
        "constraints": [{"type": "affordance", "subject": "surface", "value": "support a large bowl",
                         "hardness": "hard", "verifier": "vlm"}],
    })
    monkeypatch.setattr(adapter, "ground_plan", lambda plan, **kwargs: plan)
    plan = adapter.compile_instruction_plan(raw, backend=backend)
    visual = FinalInstructionVerifier._visual_task(plan.as_dict(), {"target_id": plan.targets[0].id})
    assert [c["id"] for c in visual["visual_predicates"]] == ["c0"]
    assert visual["visual_predicates"][0]["value"] == "support a large bowl"


def test_explicit_object_instruction_does_not_acquire_a_demand(monkeypatch):
    from instruction.compiler import compile_instruction
    from instruction_adapter.task_compat import runtime_plan

    raw = "Find the red chair."
    requests = _structured_parser_provider(monkeypatch, {
        "targets": [{"name": "chair", "attributes": {"color": "red"}}],
        "constraints": [{"type": "attribute", "subject": "chair", "value": {"color": "red"},
                         "hardness": "hard", "verifier": "vlm"}],
    })
    result = compile_instruction(raw, capabilities={"attribute"})
    assert result.status == "success", result.diagnostics
    assert len(requests) == 1
    task = result.task
    assert task.raw_instruction == raw and task.targets[0].name == "chair"
    assert task.targets[0].attributes == {"color": "red"}
    assert not task.targets[0].demand
    assert [c.type for c in task.constraints] == ["attribute"]
    assert not task.search_priors["destination_hypotheses"]
    plan = runtime_plan(task)
    assert plan.target_detector_prompts == ["chair"]
    assert plan.target_match_terms == ["chair"]


def test_parsed_projection_and_revision_preserve_immutable_semantics(monkeypatch):
    from dataclasses import FrozenInstanceError
    from instruction import compiler, parser
    from prompting.schemas import ParsedInstruction

    raw = "  Find two red cups, then a clean writing surface near the work desk.  "
    payload = {
        "task_type": " MULTI_GOAL ", "eval_mode": " ALL_TARGETS_SUCCESS ",
        "targets": [
            {"name": " Cup ", "aliases": ["mug", "MUG"], "min_count": 2,
             "attributes": {"colors": ["red"]}},
            {"name": "writing_surface", "role": " SECONDARY ", "min_count": 1,
             "attributes": {"state": "clean"},
             "demand": {"activity": "write notes", "manipulated_objects": ["notebook"],
                        "required_resource": "usable writing surface",
                        "destination_hypotheses": ["table", "TABLE", "counter"]}},
            {"name": "work desk", "role": "anchor", "terminal": False},
        ],
        "constraints": [
            {"type": " SEQUENCE ", "subject": " Cup ", "object": "writing_surface"},
            {"type": " RELATION ", "subject": "writing_surface", "relation": " NEAR ",
             "object": "work desk", "hardness": " HARD ", "verifier": " VLM "},
        ],
        "room_hints": ["study", "STUDY"], "support_objects": ["lamp"],
        "affordances": ["writing"], "exhaustive": True, "requires_runtime_relation": True,
    }
    parsed = ParsedInstruction.model_validate(payload)
    before = parsed.model_dump()
    requests = _structured_parser_provider(monkeypatch, payload)
    plan = parser.plan_from_parsed_instruction(raw, parsed)
    assert not requests
    assert plan.as_dict() == parser.parse_instruction_with_llm(raw, strict=True).as_dict()
    assert len(requests) == 1
    assert parsed.model_dump() == before
    assert plan.raw_instruction == raw.strip() and plan.dataset_target == ""
    assert plan.task_type == "multi_goal" and plan.eval_mode == "all_targets_success"
    assert plan.valid and plan.execution.ordered and plan.execution.exhaustive
    assert plan.diagnostics["requires_runtime_relation"] is True
    assert plan.search_priors.room_hints == ["study"]
    assert plan.search_priors.support_objects == ["lamp"]
    assert plan.search_priors.affordances == ["writing"]
    assert plan.constraints[1].as_dict() == Constraint(
        type="relation", subject="writing surface", relation="near", object="work desk",
        hardness="hard", verifier="vlm", source="llm",
    ).as_dict()

    forbidden = Mock(side_effect=AssertionError("parsed compilation must not request a model"))
    monkeypatch.setattr(parser, "get_client_and_model", forbidden)
    monkeypatch.setattr(compiler, "parse_instruction_with_llm", forbidden)
    result = compiler.compile_parsed_instruction(raw, parsed, {"relation", "affordance"}, revision=7)
    assert result.status == "success", result.diagnostics
    task = result.task
    assert task == task_from_parsed_plan(plan, revision=7)
    assert task.raw_instruction == raw.strip() and task.revision == 7
    assert [(t.id, t.name, t.role, t.min_count, t.terminal) for t in task.targets] == [
        ("t0_cup", "cup", "primary", 2, True),
        ("t1_writing_surface", "writing surface", "secondary", 1, True),
        ("t2_work_desk", "work desk", "anchor", 1, False),
    ]
    assert task.targets[0].aliases == ("mug", "cup")
    assert task.targets[0].attributes["colors"] == ("red",)
    assert task.targets[1].attributes == {"state": "clean"}
    assert task.targets[1].demand == {
        "activity": "write notes", "manipulated_objects": ("notebook",),
        "required_resource": "usable writing surface",
    }
    assert task.constraints[-1].value == task.targets[1].demand
    assert task.constraints[-1].id == "demand:t1_writing_surface"
    assert task.search_priors["destination_hypotheses"][task.targets[1].id] == ("table", "counter")
    frozen = task.as_dict()
    with pytest.raises(FrozenInstanceError):
        task.raw_instruction = "replacement"
    with pytest.raises(TypeError):
        task.targets[1].demand["activity"] = "replacement"
    with pytest.raises(TypeError):
        task.targets[0].attributes["colors"][0] = "blue"
    parsed.targets[0].min_count = 3
    parsed.targets[0].attributes["colors"].append("blue")
    parsed.targets[1].demand.manipulated_objects.append("paper")
    revised = compiler.compile_parsed_instruction(raw, parsed, {"relation", "affordance"}, revision=8)
    assert revised.status == "success", revised.diagnostics
    assert revised.task.revision == 8 and revised.task.raw_instruction == task.raw_instruction
    assert revised.task.targets[0].min_count == 3
    assert revised.task.targets[1].demand["manipulated_objects"] == ("notebook", "paper")
    assert task.as_dict() == frozen
    forbidden.assert_not_called()


@pytest.mark.parametrize("strict", [True, False])
def test_parsed_projection_preserves_strict_dataset_fallback_policy(monkeypatch, strict):
    from instruction import parser
    from prompting.schemas import ParsedInstruction

    payload = {"targets": [{"name": "", "terminal": True},
                           {"name": " Desk ", "role": "anchor", "terminal": False}]}
    requests = _structured_parser_provider(monkeypatch, payload)
    plan = parser.plan_from_parsed_instruction(
        "look near desk", ParsedInstruction.model_validate(payload),
        dataset_target=" Coffee_Mug ", strict=strict,
    )
    assert not requests
    assert plan.as_dict() == parser.parse_instruction_with_llm(
        "look near desk", dataset_target=" Coffee_Mug ", strict=strict,
    ).as_dict()
    assert len(requests) == 1
    assert plan.dataset_target == "coffee mug" and plan.valid is (not strict)
    assert plan.targets[-1].id == "t1_desk"
    assert [t.name for t in plan.terminal_targets] == ([] if strict else ["coffee mug"])
    if not strict:
        assert plan.targets[0].source == "dataset_fallback"


@pytest.mark.parametrize("text,payload,capabilities,revision,status", [
    pytest.param(" \n", {}, set(), 1, "invalid", id="empty"),
    pytest.param("find it", {"unresolved": ["which object?"]}, set(), 1,
                 "ambiguous", id="unresolved-before-invalid"),
    pytest.param("find it", {}, set(), 1, "invalid", id="no-target"),
    pytest.param("near desk", {"targets": [{"name": "desk", "terminal": False}]},
                 set(), 1, "invalid", id="anchor-only"),
    pytest.param("find cup", {"targets": [{"name": "cup"}]}, set(), 0,
                 "invalid", id="invalid-revision"),
    pytest.param("find surface", {"targets": [{"name": "surface", "demand": {
        "activity": "", "required_resource": "surface"}}]}, {"affordance"}, 1,
        "invalid", id="invalid-demand"),
    pytest.param("avoid bedroom", {"targets": [{"name": "cup"}], "constraints": [{
        "type": "avoid_room", "value": "bedroom", "hardness": "hard"}]}, {"relation"}, 1,
        "unsupported", id="unsupported-hard-constraint"),
    pytest.param("find surface", {"targets": [{"name": "surface", "demand": {
        "activity": "write", "required_resource": "surface"}}]}, set(), 1,
        "unsupported", id="unsupported-demand"),
    pytest.param("find cup", {"targets": [{"name": "cup"}], "constraints": [{
        "type": "room", "value": "study", "hardness": "soft"}]}, set(), 1,
        "success", id="soft-constraint"),
])
def test_parsed_compiler_uses_initial_gates_without_reparsing(
    monkeypatch, text, payload, capabilities, revision, status,
):
    from instruction import compiler, parser
    from prompting.schemas import ParsedInstruction

    requests = _structured_parser_provider(monkeypatch, payload)
    initial = compiler.compile_instruction(text, capabilities, revision=revision)
    assert len(requests) == int(bool(text.strip()))
    forbidden = Mock(side_effect=AssertionError("unexpected parse request"))
    monkeypatch.setattr(parser, "get_client_and_model", forbidden)
    monkeypatch.setattr(compiler, "parse_instruction_with_llm", forbidden)
    result = compiler.compile_parsed_instruction(
        text, ParsedInstruction.model_validate(payload), capabilities, revision=revision,
    )
    assert result == initial
    assert result.status == status
    assert (result.task is not None) is (status == "success")
    forbidden.assert_not_called()


@pytest.mark.parametrize("adopt", [False, True])
def test_task_revision_invalidates_only_task_prior_state(adopt):
    from dataclasses import replace
    from instruction.contracts import TaskSpec, TargetSpec
    from instruction_adapter.task_compat import runtime_plan
    from scene.contracts import Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    spec = TaskSpec("Find somewhere to sit.", (TargetSpec("goal", "chair"),))
    snapshot = SemanticMapSnapshot(12., Pose3D((1., 2., 0.)))
    session = TaskSession(snapshot, runtime_plan(spec))
    task_fields = ("prior_result", "prior_map_prompt_context",
                   "prior_map_high_level_selection", "prior_map_diagnostics")
    for name in task_fields:
        setattr(session, name, object())
    session.prior_map_room_semantics = {"room": {"label": "living room"}}
    session.prior_map_policy_adapter = object()
    before = vars(session).copy()

    session.install_plan(session.plan)
    session.install_plan(runtime_plan(spec))
    with pytest.raises(ValueError, match="newer revision"):
        session.install_plan(runtime_plan(replace(spec, targets=(TargetSpec("goal", "sofa"),))))
    assert all(getattr(session, name) is before[name] for name in task_fields)

    plan = runtime_plan(replace(spec, revision=2, targets=(TargetSpec("goal", "sofa"),)))
    if adopt:
        session.adopt_interpretation(plan, task_revision=1, search_revision=session.search_revision,
                                     snapshot=snapshot, record={"reason": "new interpretation"})
    else:
        session.install_plan(plan)
    assert all(getattr(session, name) is None for name in task_fields)
    assert session.spec.revision == 2 and not session.progress.completed
    for name in ("snapshot", "prior_map_room_semantics", "prior_map_policy_adapter",
                 "concept_matcher", "constraint_evaluator"):
        assert getattr(session, name) is before[name]
    assert len(session.interpretation_history) == int(adopt)


@pytest.mark.parametrize("model_calls", [2, 3])
def test_initial_compiler_admission_flows_through_parser(monkeypatch, model_calls):
    from instruction.contracts import CompileResult, TargetSpec, TaskSpec
    from instruction_adapter import compiler, StriveInstructionParser

    raw = "Find a chair."
    task = TaskSpec(raw, (TargetSpec("goal", "chair"),))
    initial = SimpleNamespace(model_calls=model_calls, compile=Mock(
        return_value=CompileResult("success", task)))
    legacy = Mock(side_effect=AssertionError("initial admission must not use legacy parsing"))
    monkeypatch.setattr(compiler, "compile_instruction", legacy)
    plan = StriveInstructionParser().parse_plan(
        raw, dataset_target="bed", available_classes=["chair"], initial_compiler=initial)
    initial.compile.assert_called_once_with(raw, compiler.RUNTIME_CONSTRAINT_TYPES)
    legacy.assert_not_called()
    assert plan.diagnostics["initial_model_calls"] == model_calls
    assert plan.diagnostics["task_spec"] == task.as_dict()
    assert plan.target_detector_prompts == ["chair"]


@pytest.mark.parametrize("valid", [True, False])
def test_metadata_does_not_invoke_initial_recovery_or_inherit_its_calls(monkeypatch, valid):
    from instruction_adapter import compiler

    forbidden = Mock(side_effect=AssertionError("metadata must not invoke a language compiler"))
    initial = SimpleNamespace(model_calls=17, compile=forbidden)
    monkeypatch.setattr(compiler, "compile_instruction", forbidden)
    monkeypatch.setattr(compiler, "parse_instruction_with_llm", forbidden)
    metadata = {"instruction_targets": ["chair"] if valid else []}
    if valid:
        plan = compiler.compile_instruction_plan("Find a chair.", episode_info=metadata,
            available_classes=["chair"], initial_compiler=initial)
        assert plan.diagnostics["task_spec"]["source"] == "benchmark_metadata"
        assert plan.terminal_targets[0].name == "chair"
        assert "initial_model_calls" not in plan.diagnostics
    else:
        with pytest.raises(ValueError, match="terminal target"):
            compiler.compile_instruction_plan("Find it.", episode_info=metadata,
                available_classes=["chair"], initial_compiler=initial)
    forbidden.assert_not_called()
    assert initial.model_calls == 17


@pytest.mark.parametrize("backend", ["llm", "legacy", "rules"])
def test_adapter_preserves_legacy_and_unsupplied_compiler_paths(monkeypatch, backend):
    from instruction.contracts import CompileResult
    from instruction_adapter import compiler

    parsed = InstructionPlan("Find a chair.", targets=[TargetQuery("goal", "chair")])
    initial = SimpleNamespace(model_calls=11, compile=Mock(
        side_effect=AssertionError("legacy backend must ignore initial compiler")))
    parse = Mock(return_value=parsed)
    compile_once = Mock(return_value=CompileResult("success", task_from_parsed_plan(parsed)))
    monkeypatch.setenv("LLM_OFFLINE", "1")
    monkeypatch.setattr(compiler, "parse_instruction_with_llm", parse)
    monkeypatch.setattr(compiler, "compile_instruction", compile_once)
    plan = compiler.compile_instruction_plan(parsed.raw_instruction, available_classes=["chair"],
        backend=backend, initial_compiler=None if backend == "llm" else initial)
    assert "initial_model_calls" not in plan.diagnostics
    assert plan.terminal_targets[0].name == "chair"
    assert plan.diagnostics["task_spec"]["source"] == ("language" if backend == "llm" else backend)
    assert compile_once.call_count == int(backend == "llm")
    assert parse.call_count == int(backend != "llm")
    initial.compile.assert_not_called()


@pytest.mark.parametrize("source", ["benchmark_object_goal", "metadata", "language", "invalid_language"])
@pytest.mark.parametrize("budget", [None, 5])
def test_benchmark_installer_limits_initial_compilation_to_language(monkeypatch, tmp_path, source, budget):
    """Run the actual installer without importing Habitat or constructing sensors."""
    import ast
    import json
    import os
    from pathlib import Path
    from backends.habitat import scene
    from instruction.contracts import CompileResult, TargetSpec, TaskSpec
    from instruction_adapter import compiler, StriveInstructionParser, render_instruction_context
    from scene.contracts import Pose3D, SemanticMapSnapshot
    from task import session

    path = Path(__file__).resolve().parents[1] / "objnav_benchmark_with_process_obs.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_install_instruction_plan")
    raw = "Find a chair."
    invalid = source == "invalid_language"
    result = (CompileResult("invalid", diagnostics=("no admitted task",)) if invalid else
              CompileResult("success", TaskSpec(raw, (TargetSpec("goal", "chair"),))))
    initial = SimpleNamespace(model_calls=3, compile=Mock(return_value=result))
    factory = Mock(return_value=initial)
    namespace = {"InitialInstructionCompiler": factory, "Path": Path, "json": json, "os": os,
                 "render_instruction_context": render_instruction_context}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    compile_once = Mock(return_value=result)
    monkeypatch.setattr(compiler, "compile_instruction", compile_once)
    snapshot = Mock(return_value=(SemanticMapSnapshot(0., Pose3D((0., 0., 0.))), {}))
    monkeypatch.setattr(scene, "snapshot_from_mapper", snapshot)
    task_factory = Mock(wraps=session.TaskSession)
    monkeypatch.setattr(session, "TaskSession", task_factory)
    mapper = SimpleNamespace(object_perceiver=SimpleNamespace(classes=["chair"]))
    agent = SimpleNamespace(save_dir=str(tmp_path))
    parser = StriveInstructionParser(vlm="cognav")
    kwargs = dict(habitat_mapper=mapper, habitat_agent=agent, instruction_parser=parser,
        raw_instruction=raw, dataset_target="chair", episode_dir=tmp_path,
        episode_info={"instruction_targets": ["chair"]} if source == "metadata" else {},
        source=source, initial_model_budget=budget)
    if invalid:
        with pytest.raises(ValueError, match="invalid: no admitted task"):
            namespace["_install_instruction_plan"](**kwargs)
        snapshot.assert_not_called()
        task_factory.assert_not_called()
        assert "task" not in vars(agent)
        assert "target" not in vars(mapper)
    else:
        plan, _ = namespace["_install_instruction_plan"](**kwargs)
        assert agent.task.plan is plan
        assert mapper.target == "chair"
        assert ("initial_model_calls" in plan.diagnostics) == (source == "language" and budget is not None)
        if source in {"benchmark_object_goal", "metadata"}:
            assert plan.diagnostics["task_spec"]["source"] == (
                "objectnav_category" if source == "benchmark_object_goal" else "benchmark_metadata")
    if budget is not None and source != "benchmark_object_goal":
        factory.assert_called_once_with(max_model_calls=budget,
            trace_path=tmp_path / "initial_interpretation.jsonl", vlm=parser.vlm)
    else:
        factory.assert_not_called()
    is_language = source in {"language", "invalid_language"}
    assert initial.compile.call_count == int(is_language and budget is not None)
    assert compile_once.call_count == int(is_language and budget is None)


def _initial_compiler_replay(tmp_path, outputs, *, budget=2):
    """按顺序回放响应，只替换模型边界。"""
    import json
    from instruction.compiler import InitialInstructionCompiler

    requests = []
    pending = iter(outputs)

    def parse(**kwargs):
        requests.append(kwargs)
        assert kwargs["parse_retries"] == 0
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        parsed = kwargs["response_format"].model_validate(value) if value is not None else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            parsed=parsed, content=json.dumps(value) if value is not None else None))])

    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse))))
    compiler = InitialInstructionCompiler(max_model_calls=budget,
        trace_path=tmp_path / "initial.jsonl", client_model=(client, "offline-replay"))
    return compiler, requests


def test_demand_schema_orders_resource_inference_before_naming_without_repairing_output(tmp_path):
    from prompting.schemas import ParsedTarget
    from llm_utils.structured_output import StructuredOutputError

    schema = ParsedTarget.model_json_schema()
    assert list(schema["properties"]).index("demand") < list(schema["properties"]).index("name")
    corrected = {"targets": [{"name": "ingredients", "terminal": False, "demand": {
        "activity": "clean", "manipulated_objects": ["ingredients"],
        "required_resource": "cleaning_equipment", "destination_hypotheses": []}}]}
    initial = StructuredOutputError("malformed", raw_content="{")
    compiler, requests = _initial_compiler_replay(tmp_path, [initial, corrected])
    result = compiler.compile("Gonna clean these ingredients before I start cooking.", {"affordance"})
    assert result.status == "invalid" and result.task is None
    assert result.diagnostics == ("Demand requires a terminal resource target and an activity",)
    assert [request["trace_label"] for request in requests] == ["initial_parse", "initial_correction"]


@pytest.mark.parametrize("raw,payload", [
    ("Find a chair.", {"targets": [{"name": "chair"}]}),
    ("Find a place to wash socks.", {"targets": [{"name": "washing resource", "demand": {
        "activity": "wash socks", "manipulated_objects": ["socks"],
        "required_resource": "washing resource", "destination_hypotheses": ["basin"]}}]}),
])
@pytest.mark.parametrize("budget", [1, 20])
def test_initial_valid_interpretation_compiles_in_one_call(tmp_path, raw, payload, budget):
    import json
    from prompting.schemas import ParsedInstruction

    compiler, requests = _initial_compiler_replay(tmp_path, [payload], budget=budget)
    result = compiler.compile(raw, {"affordance"}, revision=4)
    assert result.status == "success" and result.task.revision == 4
    assert result.task.raw_instruction == raw
    assert result.task.targets[0].name == payload["targets"][0]["name"]
    assert result.task.targets[0].terminal and result.task.targets[0].min_count == 1
    assert not result.task.ordered and not result.task.exhaustive
    assert result.task.completion_mode == "any_target_success"
    assert compiler.model_calls == len(requests) == 1 and compiler.max_model_calls == budget
    assert [request["trace_label"] for request in requests] == ["initial_parse"]
    assert requests[0]["response_format"] is ParsedInstruction
    assert json.loads(requests[0]["messages"][1]["content"]) == {"original_instruction": raw}
    events = [json.loads(line) for line in compiler.trace_path.read_text().splitlines()]
    assert not any(event.get("task") for event in events[:-1])
    assert events[-1]["status"] == "success" and events[-1]["model_calls"] == 1
    assert events[-1]["task"] == result.task.as_dict()


def test_initial_relation_schema_precedes_unary_properties():
    from prompting.schemas import ParsedInstruction

    properties = list(ParsedInstruction.model_json_schema()["properties"])
    assert properties.index("constraints") < properties.index("targets")


@pytest.mark.parametrize("initial", [
    {}, None, {"targets": [{"name": "socks", "terminal": False}]},
])
@pytest.mark.parametrize("budget", [2, 20])
def test_initial_invalid_interpretation_recovers_once(tmp_path, initial, budget):
    import json
    from prompting.schemas import ParsedInstruction
    from prompting.templates import INSTRUCTION_UNDERSTANDING_SKILL

    corrected = {"targets": [{"name": "washing resource", "demand": {
        "activity": "wash socks", "manipulated_objects": ["socks"],
        "required_resource": "washing resource", "destination_hypotheses": ["basin", "sink"]}}]}
    compiler, requests = _initial_compiler_replay(tmp_path, [initial, corrected], budget=budget)
    raw = "Find a place to wash socks."
    result = compiler.compile(raw, {"affordance"})
    assert result.status == "success" and result.task.raw_instruction == raw
    target = result.task.targets[0]
    assert len(result.task.targets) == 1 and target.terminal and target.name == "washing resource"
    assert target.demand == {"activity": "wash socks", "manipulated_objects": ("socks",),
                             "required_resource": "washing resource"}
    assert result.task.search_priors["destination_hypotheses"][target.id] == ("basin", "sink")
    assert result.task.constraints[0].type == "affordance"
    assert result.task.constraints[0].source == "demand"
    assert compiler.model_calls == len(requests) == 2 and compiler.max_model_calls == budget
    assert [request["trace_label"] for request in requests] == ["initial_parse", "initial_correction"]
    assert all(request["response_format"] is ParsedInstruction for request in requests)
    correction = json.loads(requests[1]["messages"][1]["content"])
    assert correction["status"] == "invalid" and correction["diagnostics"]
    assert correction["original_instruction"] == raw
    assert correction["previous_interpretation"] == (
        ParsedInstruction.model_validate(initial).model_dump() if initial is not None else None)
    assert all(INSTRUCTION_UNDERSTANDING_SKILL in request["messages"][0]["content"] for request in requests)
    events = [json.loads(line) for line in compiler.trace_path.read_text().splitlines()]
    assert events[-1]["status"] == "success" and events[-1]["model_calls"] == 2
    assert not any(event.get("task") for event in events[:-1])


@pytest.mark.parametrize("initial,status,diagnostics", [
    ({"targets": [{"name": "chair"}], "unresolved": ["which chair?", "which room?"]},
     "ambiguous", ("which chair?", "which room?")),
    ({"targets": [{"name": "chair"}], "constraints": [{
        "type": "avoid_room", "value": "bedroom", "hardness": "hard"}]},
     "unsupported", ("c0:avoid room",)),
])
@pytest.mark.parametrize("budget", [1, 20])
def test_initial_ambiguous_or_unsupported_returns_original_diagnostics(tmp_path, initial, status, diagnostics, budget):
    compiler, requests = _initial_compiler_replay(tmp_path, [initial], budget=budget)
    result = compiler.compile("Find a chair without entering the bedroom.", set())
    assert result.status == status and result.task is None
    assert result.diagnostics == diagnostics
    assert compiler.model_calls == len(requests) == 1 and compiler.max_model_calls == budget
    assert [request["trace_label"] for request in requests] == ["initial_parse"]


@pytest.mark.parametrize("budget,expected_calls,stage", [
    (0, 0, "initial_parse"), (1, 1, "initial_correction"),
])
def test_initial_recovery_shares_a_strict_budget(tmp_path, budget, expected_calls, stage):
    compiler, requests = _initial_compiler_replay(tmp_path,
        [{}, {"targets": [{"name": "chair"}]}], budget=budget)
    result = compiler.compile("Find a chair.", set())
    assert result.status == "budget_exhausted" and result.task is None
    assert result.diagnostics == (stage,)
    assert compiler.model_calls == len(requests) == expected_calls


@pytest.mark.parametrize("corrected,status", [
    pytest.param({}, "invalid", id="empty-targets"),
    pytest.param(None, "invalid", id="empty-response"),
    pytest.param({"targets": [{"name": "chair", "terminal": False}]}, "invalid", id="no-destination"),
    pytest.param({"unresolved": ["which one?"]}, "ambiguous", id="unresolved"),
    pytest.param({"targets": [{"name": "chair"}], "constraints": [{
        "type": "avoid_room", "value": "bedroom", "hardness": "hard"}]}, "unsupported", id="unsupported"),
    pytest.param("malformed", "invalid", id="malformed-response"),
])
@pytest.mark.parametrize("budget", [2, 20])
def test_initial_recovery_never_retries_an_invalid_correction(tmp_path, corrected, status, budget):
    from llm_utils.structured_output import StructuredOutputError

    if corrected == "malformed":
        corrected = StructuredOutputError("malformed correction", raw_content="{")
    compiler, requests = _initial_compiler_replay(tmp_path, [{}, corrected], budget=budget)
    result = compiler.compile("Find a chair without entering the bedroom.", set())
    assert result.status == status and result.task is None and result.diagnostics
    assert compiler.model_calls == len(requests) == 2 and compiler.max_model_calls == budget
    assert [request["trace_label"] for request in requests] == ["initial_parse", "initial_correction"]
    assert "Preserve the original instruction's requirements" in requests[1]["messages"][0]["content"]


def test_initial_recovery_preserves_complete_explicit_requirements(tmp_path):
    corrected = {"targets": [{"name": "cup", "aliases": ["mug"], "min_count": 2, "attributes": {"color": "red"}},
                             {"name": "table", "terminal": False}, {"name": "chair"}],
                 "constraints": [{"type": "spatial", "subject": "cup", "relation": "on",
                                  "object": "table", "hardness": "hard"}],
                 "ordered": True, "exhaustive": True, "eval_mode": "all_targets_success"}
    raw = "Find all two red cups on the table, then the chair."
    compiler, requests = _initial_compiler_replay(tmp_path, [{}, corrected])
    result = compiler.compile(raw, {"spatial"}, revision=7)
    assert result.status == "success" and result.task is not None
    task = result.task
    assert task.raw_instruction == raw and task.revision == 7
    assert task.ordered and task.exhaustive
    assert task.completion_mode == "all_targets_success"
    assert [target.name for target in task.targets] == ["cup", "table", "chair"]
    assert [target.name for target in task.targets if target.terminal] == ["cup", "chair"]
    assert task.targets[0].aliases == ("mug", "cup")
    assert task.targets[0].min_count == 2 and task.targets[0].attributes == {"color": "red"}
    assert [(c.type, c.subject, c.relation, c.object, c.hardness) for c in task.constraints] == [
        ("spatial", "cup", "on", "table", "hard")]
    assert compiler.model_calls == len(requests) == 2
    assert [request["trace_label"] for request in requests] == ["initial_parse", "initial_correction"]


@pytest.mark.parametrize("failure_stage", [0, 1])
@pytest.mark.parametrize("error", [
    ConnectionError("transport"), ValueError("transport validation"), OSError("IO"), RuntimeError("provider refusal"),
])
def test_initial_recovery_does_not_treat_transport_failures_as_semantic_errors(tmp_path, failure_stage, error):
    import json

    compiler, requests = _initial_compiler_replay(tmp_path, [{}] * failure_stage + [error], budget=20)
    with pytest.raises(type(error)) as caught:
        compiler.compile("Find a chair.", set())
    assert caught.value is error
    assert compiler.model_calls == len(requests) == failure_stage + 1
    assert [request["trace_label"] for request in requests] == [
        "initial_parse", "initial_correction"][:failure_stage + 1]
    events = [json.loads(line) for line in compiler.trace_path.read_text().splitlines()]
    assert events[-1]["event"] == "request_failed"
    assert events[-1]["stage"] == requests[-1]["trace_label"]
    assert events[-1]["error_type"] == type(error).__name__
    assert not any(event.get("task") or event["event"] == "admission" for event in events)


def test_parsed_all_targets_mode_cannot_complete_after_one_target():
    from instruction.compiler import compile_parsed_instruction
    from prompting.schemas import ParsedInstruction
    from task.reducer import apply_verification
    from verification.contracts import CandidateInstance, VerificationResult

    result = compile_parsed_instruction("Find both a cup and a chair.", ParsedInstruction(
        eval_mode="all_targets_success", targets=[{"name": "cup"}, {"name": "chair"}]), set())
    task = result.task
    assert not task.ordered and not task.exhaustive
    state = TaskProgress(task)
    for index, target in enumerate(task.targets):
        uid = f"instance-{index}"
        complete = apply_verification(state, target, CandidateInstance(uid, target.name),
            VerificationResult(True, "accept", semantic_satisfied=True,
                               task_revision=1, target_uid=uid), {})
        assert complete is (index == 1)


@pytest.mark.parametrize("invalid", [
    {"targets": [{"name": "chair"}, {"name": ""}]},
    {"targets": [{"name": "chair"}], "constraints": [{"type": "", "value": "avoid bedroom", "hardness": "hard"}]},
    {"targets": [{"name": "chair", "min_count": 0}]},
])
def test_initial_lossy_structures_require_recovery(tmp_path, invalid):
    compiler, requests = _initial_compiler_replay(tmp_path, [invalid, invalid])
    result = compiler.compile("Find a chair and avoid the bedroom.", set())
    assert result.status == "invalid" and result.task is None
    assert len(requests) == 2 and result.diagnostics


@pytest.mark.parametrize("strict", [False, True])
def test_initial_grounding_cannot_make_unbudgeted_calls(monkeypatch, tmp_path, strict):
    from instruction_adapter.compiler import compile_instruction_plan
    from perception import detector_adapter

    provider = Mock(side_effect=AssertionError("unbudgeted grounding call"))
    monkeypatch.setattr(detector_adapter, "get_client_and_model", provider)
    monkeypatch.setenv("LLM_OFFLINE", "0")
    payload = {"targets": [{"name": "armchair"}], "constraints": [{
        "type": "spatial", "subject": "armchair", "relation": "near", "object": "work desk", "hardness": "hard"}]}
    compiler, requests = _initial_compiler_replay(tmp_path, [payload], budget=1)
    plan = compile_instruction_plan("Find the armchair near the work desk.", initial_compiler=compiler,
                                    available_classes=["chair", "table"], strict_available_classes=strict)
    assert plan.diagnostics["initial_model_calls"] == compiler.model_calls == len(requests) == 1
    assert plan.targets[0].detector_terms == ([] if strict else ["armchair"])
    assert plan.constraints[0].object_concept.name == "work desk"
    provider.assert_not_called()


def test_initial_recovery_uses_malformed_response_diagnostics(tmp_path):
    import json
    from llm_utils.structured_output import StructuredOutputError

    error = StructuredOutputError("truncated JSON", raw_content='{"targets": [')
    compiler, requests = _initial_compiler_replay(tmp_path, [error, {"targets": [{"name": "chair"}]}])
    result = compiler.compile("Find a chair.", set())
    assert result.status == "success" and compiler.model_calls == len(requests) == 2
    assert [request["trace_label"] for request in requests] == ["initial_parse", "initial_correction"]
    correction = json.loads(requests[1]["messages"][1]["content"])
    assert correction["status"] == "invalid"
    assert correction["previous_interpretation"] is None
    assert correction["previous_response"] == error.raw_content
    assert correction["diagnostics"] == ["truncated JSON"]


def test_initial_single_parse_installs_session_without_accepting_candidates(monkeypatch, tmp_path):
    import ast
    import json
    import os
    from pathlib import Path
    from backends.habitat import scene
    from instruction_adapter import StriveInstructionParser, render_instruction_context
    from scene.contracts import Pose3D, SemanticMapSnapshot
    from task.session import TaskSession

    monkeypatch.setenv("LLM_OFFLINE", "1")
    compiler, requests = _initial_compiler_replay(tmp_path, [{"targets": [{"name": "chair"}]}], budget=1)
    path = Path(__file__).resolve().parents[1] / "objnav_benchmark_with_process_obs.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_install_instruction_plan")
    factory = Mock(return_value=compiler)
    namespace = {"InitialInstructionCompiler": factory, "Path": Path, "json": json, "os": os,
                 "render_instruction_context": render_instruction_context}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    snapshot = SemanticMapSnapshot(0., Pose3D((0., 0., 0.)))
    monkeypatch.setattr(scene, "snapshot_from_mapper", Mock(return_value=(snapshot, {})))
    agent = SimpleNamespace(save_dir=str(tmp_path))
    mapper = SimpleNamespace(object_perceiver=SimpleNamespace(classes=["chair"]))
    parser = StriveInstructionParser()
    plan, _ = namespace["_install_instruction_plan"](habitat_mapper=mapper, habitat_agent=agent,
        instruction_parser=parser, raw_instruction="Find a chair.", dataset_target="bed",
        episode_info={}, episode_dir=tmp_path, source="language", initial_model_budget=1)
    factory.assert_called_once_with(max_model_calls=1,
        trace_path=tmp_path / "initial_interpretation.jsonl", vlm=parser.vlm)
    assert plan.diagnostics["initial_model_calls"] == compiler.model_calls == len(requests) == 1
    assert [request["trace_label"] for request in requests] == ["initial_parse"]
    assert isinstance(agent.task, TaskSession) and agent.task.plan is plan
    assert agent.task.snapshot is snapshot and mapper.target == "chair"
    assert agent.task.spec.raw_instruction == "Find a chair."
    assert [target.name for target in agent.task.spec.targets] == ["chair"]
    assert not agent.task.verification_ledger.records and not agent.task.verification_ledger.history
    # 编译成功只建立任务，完成状态仍由观测验证推进。
    assert not agent.task.progress.completed and agent.task.progress.mode == "search"
    assert all(not progress.accepted_candidate_uids and not progress.complete
               for progress in agent.task.progress.target_progress.values())
    persisted = tmp_path / "instruction_adapter" / "task_spec.json"
    assert json.loads(persisted.read_text()) == agent.task.spec.as_dict()


@pytest.mark.parametrize("raw,status", [(" ", "invalid"), ("Find a chair.", "parse_unavailable")])
def test_initial_offline_or_empty_input_has_no_model_calls(monkeypatch, tmp_path, raw, status):
    from instruction.compiler import InitialInstructionCompiler
    from llm_utils import cognav_llm_adapter

    monkeypatch.setenv("LLM_OFFLINE", "1")
    provider = Mock(side_effect=AssertionError("must not load provider"))
    monkeypatch.setattr(cognav_llm_adapter, "get_client_and_model", provider)
    compiler = InitialInstructionCompiler(max_model_calls=3, trace_path=tmp_path / "initial.jsonl")
    result = compiler.compile(raw, set())
    assert result.status == status and result.task is None and compiler.model_calls == 0
    provider.assert_not_called()


@pytest.mark.parametrize("failure_stage", [0, 1])
def test_initial_recovery_propagates_provider_refusal(monkeypatch, tmp_path, failure_stage):
    import json

    compiler, requests = _initial_compiler_replay(tmp_path, [{}] * failure_stage + [None], budget=20)
    completions = compiler.client_model[0].beta.chat.completions
    replay_parse = completions.parse

    def parse(**kwargs):
        completion = replay_parse(**kwargs)
        if len(requests) == failure_stage + 1:
            completion.choices[0].message.refusal = "Scripted provider refusal."
        return completion

    monkeypatch.setattr(completions, "parse", parse)
    with pytest.raises(RuntimeError, match="refused"):
        compiler.compile("Find a chair.", set())
    assert compiler.model_calls == len(requests) == failure_stage + 1
    assert [request["trace_label"] for request in requests] == [
        "initial_parse", "initial_correction"][:failure_stage + 1]
    events = [json.loads(line) for line in compiler.trace_path.read_text().splitlines()]
    assert events[-1]["event"] == "request_refused"
    assert events[-1]["stage"] == requests[-1]["trace_label"]
    assert not any(event.get("task") or event["event"] == "admission" for event in events)
