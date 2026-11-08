import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instruction_adapter.contracts import ConceptQuery, Constraint, InstructionPlan, TargetQuery
from perception import detector_adapter
from prompting.schemas import GroundingResult
from real_robot.detector_vocabulary import DetectorVocabularyAdapter, merge_label_provenance


def test_sysnav_objects_yaml_loads_label_and_prompt_space(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text(
        """
prompts:
  tv_monitor:
    prompts:
      - tv_monitor
    is_instance: true
  garbage_bin:
    prompts:
      - trash can
    is_instance: true
  wall:
    prompts:
      - wall
    is_instance: false
""",
        encoding="utf-8",
    )

    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config))

    assert vocabulary.label_space == ("tv_monitor", "garbage_bin", "wall")
    assert "trash can" in vocabulary.prompt_space
    assert vocabulary.find_entry("tv monitor").canonical_label == "tv_monitor"
    assert vocabulary.find_entry("trash can").canonical_label == "garbage_bin"


def test_detector_provenance_keeps_raw_label_and_records_config_match(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text(
        """
prompts:
  garbage_bin:
    prompts:
      - trash can
    is_instance: true
""",
        encoding="utf-8",
    )
    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config), detector_name="sysnav_yolo_world")

    provenance = vocabulary.provenance_for("trash can")

    assert provenance["raw_detector_label"] == "trash can"
    assert provenance["known_in_detector_vocabulary"] is True
    assert provenance["canonical_label"] == "garbage_bin"
    assert provenance["matched_by"] == "prompt"
    assert provenance["detector_name"] == "sysnav_yolo_world"


def test_unknown_label_provenance_is_explicit(tmp_path) -> None:
    config = tmp_path / "objects.yaml"
    config.write_text("prompts: {}\n", encoding="utf-8")
    vocabulary = DetectorVocabularyAdapter.from_sysnav_objects_yaml(str(config))

    provenance = vocabulary.provenance_for("bookcase")

    assert provenance["raw_detector_label"] == "bookcase"
    assert provenance["known_in_detector_vocabulary"] is False
    assert provenance["canonical_label"] is None
    assert provenance["prompt_labels"] == []


def test_merge_label_provenance_preserves_existing_metadata() -> None:
    merged = merge_label_provenance(
        {"source": "sysnav"},
        {"raw_detector_label": "cup", "known_in_detector_vocabulary": True},
    )

    assert merged["source"] == "sysnav"
    assert merged["label_provenance"]["raw_detector_label"] == "cup"


@pytest.mark.parametrize("terminal", [True, False])
def test_seed_detector_terms_never_become_identity_aliases(monkeypatch, terminal):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    concept = ConceptQuery("mug", "mug", detector_terms=["container"],
                           aliases=["coffee mug"], terminal=terminal)
    target = TargetQuery("mug", "mug", terminal=terminal, concept=concept)
    grounded = detector_adapter.ground_target(target, available_classes=["container"])

    assert grounded.detector_terms == ["container"]
    assert grounded.aliases == ["coffee mug"]
    assert grounded.match_terms == ["mug", "coffee mug"]
    assert grounded.concept.match_terms == grounded.match_terms
    again = detector_adapter.ground_target(grounded, available_classes=["container"])
    assert again == grounded


@pytest.mark.parametrize("terminal", [True, False])
def test_grounding_preserves_existing_target_and_anchor_semantics(monkeypatch, terminal):
    proposal = Mock(return_value=GroundingResult(detector_terms=["container"]))
    monkeypatch.setattr(detector_adapter, "_llm_ground_terms", proposal)
    role = "primary" if terminal else "anchor"
    concept = ConceptQuery("mug", "mug", role=role, aliases=["coffee mug"],
                           description="A ceramic drinking mug with a handle.",
                           negative_terms=["bowl"], terminal=terminal)
    target = TargetQuery("mug", "mug", role=role, aliases=["drinking mug"],
                         terminal=terminal, concept=concept)
    original = concept.as_dict()
    grounded = detector_adapter.ground_target(target, available_classes=["container"])

    proposal.assert_called_once()
    assert grounded.name == grounded.concept.name == "mug"
    assert grounded.detector_terms == grounded.concept.detector_terms == ["container"]
    assert grounded.aliases == grounded.concept.aliases == ["drinking mug", "coffee mug"]
    assert grounded.concept.description == original["description"]
    assert grounded.concept.negative_terms == original["negative_terms"]
    assert grounded.concept.role == role and grounded.concept.terminal is terminal
    assert "container" not in grounded.match_terms
    assert concept.as_dict() == original


@pytest.mark.parametrize("demand", [{}, {"activity": "write notes", "manipulated_objects": ["notebook"],
                                      "required_resource": "a usable writing surface"}])
def test_grounding_request_preserves_demand_without_promoting_detector_identity(monkeypatch, demand):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    parsed = GroundingResult(detector_terms=["table"])
    parse = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(parsed=parsed))]))
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr(detector_adapter, "get_client_and_model", Mock(return_value=(client, "offline-replay")))
    target = TargetQuery("goal", "writing surface", demand=demand)
    result = detector_adapter.ground_target(target, ["table"], vlm="self_hosted")
    request = parse.call_args.kwargs
    assert request["response_format"] is GroundingResult
    assert f"Demand: {json.dumps(demand, ensure_ascii=False)}\n" in request["messages"][1]["content"]
    assert result.demand == result.concept.demand == demand
    assert result.detector_terms == ["table"] and "table" not in result.match_terms
    parse.assert_called_once()


def test_dataset_fallback_is_retrieval_only_in_legacy_view(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = InstructionPlan("find a mug", dataset_target="container",
                           targets=[TargetQuery("mug", "mug")])
    detector_adapter.ground_plan(plan, ["container"])
    assert plan.targets[0].detector_terms == ["container"]
    assert plan.targets[0].aliases == []
    assert plan.target_match_terms == ["mug"]
    legacy = plan.to_legacy_spec()
    assert legacy.target_detector_prompts == ["container"]
    assert legacy.target_aliases == legacy.target_match_terms == ["mug"]


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("available", [(), ("mug", "table")])
def test_constraint_anchor_grounding_preserves_original_semantics(monkeypatch, preexisting, available):
    proposal = Mock(return_value=GroundingResult(detector_terms=["table"]))
    monkeypatch.setattr(detector_adapter, "_llm_ground_terms", proposal)
    original = ConceptQuery("desk", "work desk", role="anchor", aliases=["writing desk"],
                            description="The desk used for writing.", negative_terms=["dining table"])
    plan = InstructionPlan("mug on work desk", targets=[TargetQuery("mug", "mug")],
                           constraints=[Constraint("spatial", subject="mug", object="work desk",
                                                   object_concept=original if preexisting else None)])
    detector_adapter.ground_plan(plan, available, use_legacy_similarity_fallback=False)
    if available:
        proposal.assert_called_once()
    else:
        proposal.assert_not_called()
    anchor = plan.constraints[0].object_concept
    assert anchor.role == "anchor" and not anchor.terminal
    assert anchor.name == "work desk"
    assert anchor.detector_terms == (["table"] if available else ["work desk"])
    assert anchor.aliases == (original.aliases if preexisting else ["work desk"])
    assert anchor.description == (original.description if preexisting else "")
    assert anchor.negative_terms == (original.negative_terms if preexisting else [])
    assert anchor.match_terms == (["work desk", "writing desk"] if preexisting else ["work desk"])
    assert original.detector_terms == []
    assert anchor in plan.concept_queries


def test_constraint_reference_does_not_bind_target_by_broad_detector_label(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "1")
    plan = InstructionPlan("workbench near table", targets=[
        TargetQuery("goal", "workbench", detector_terms=["table"])], constraints=[
        Constraint("spatial", subject="workbench", object="table")])
    detector_adapter.ground_plan(plan, ["table"])
    assert plan.constraints[0].object_concept.id != "goal"
    assert plan.constraints[0].object_concept.name == "table"
    assert plan.constraints[0].subject_concept is plan.targets[0].concept


def test_similarity_extension_updates_all_concept_views_without_alias_pollution(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    monkeypatch.setitem(sys.modules, "cv_utils.gpt_utils", SimpleNamespace(
        ask_gpt_similar_objects=lambda *args: ["container"]))
    monkeypatch.setattr(detector_adapter, "_llm_ground_terms", lambda *args:
                        GroundingResult(detector_terms=["table"]))
    plan = InstructionPlan("mug on table", targets=[TargetQuery("mug", "mug")],
                           constraints=[Constraint("spatial", subject="mug", object="table")])
    detector_adapter.ground_plan(plan, ["mug", "container", "table"])
    target = plan.targets[0]
    assert target.detector_terms == target.concept.detector_terms == ["mug", "container"]
    assert target.aliases == target.concept.aliases == []
    assert target.match_terms == ["mug"]
    assert plan.concept_queries[0] is target.concept
    assert plan.constraints[0].subject_concept is target.concept


def test_similarity_extension_does_not_swallow_failure(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE", "0")
    def fail(*args):
        raise RuntimeError("similarity unavailable")
    monkeypatch.setitem(sys.modules, "cv_utils.gpt_utils", SimpleNamespace(ask_gpt_similar_objects=fail))
    plan = InstructionPlan("mug", targets=[TargetQuery("mug", "mug")])
    with pytest.raises(RuntimeError, match="similarity unavailable"):
        detector_adapter.ground_plan(plan, ["mug"])


@pytest.mark.parametrize("anchor_field", ["object", "value"])
def test_empty_vocabulary_runtime_grounding_never_requests_model(monkeypatch, anchor_field):
    from instruction.compiler import compile_parsed_instruction
    from instruction_adapter.task_compat import runtime_plan
    from prompting.schemas import ParsedInstruction

    forbidden = Mock(side_effect=AssertionError("empty vocabulary must not request a model"))
    monkeypatch.setattr(detector_adapter, "_llm_ground_terms", forbidden)
    monkeypatch.setattr(detector_adapter, "get_client_and_model", forbidden)
    monkeypatch.setitem(sys.modules, "cv_utils.gpt_utils", SimpleNamespace(
        ask_gpt_similar_objects=forbidden))
    # 关闭离线保护，确保空词表本身阻止模型调用。
    monkeypatch.setenv("LLM_OFFLINE", "0")
    result = compile_parsed_instruction(
        "Find two clean writing surfaces near the work desk and lamp.",
        ParsedInstruction.model_validate({
            "targets": [
                {"name": "writing surface", "aliases": ["work surface"], "min_count": 2,
                 "attributes": {"state": "clean"}, "demand": {
                     "activity": "write", "manipulated_objects": ["notebook"],
                     "required_resource": "usable surface", "destination_hypotheses": ["table"]}},
                {"name": "lamp", "aliases": ["desk light"], "role": "anchor", "terminal": False},
            ],
            "constraints": [
                {"type": "relation", "subject": "writing surface", "relation": "near",
                 anchor_field: "work desk", "hardness": "hard"},
                {"type": "relation", "subject": "writing surface", "relation": "near",
                 "object": "desk light", "hardness": "hard"},
            ],
            "ordered": True, "exhaustive": True,
        }),
        {"relation", "affordance"}, revision=4,
    )
    assert result.status == "success", result.diagnostics
    task = result.task
    before = task.as_dict()
    plan = detector_adapter.ground_plan(
        runtime_plan(task), available_classes=(), use_legacy_similarity_fallback=False,
    )
    forbidden.assert_not_called()
    assert plan.valid and plan.raw_instruction == task.raw_instruction
    assert plan.execution.ordered and plan.execution.exhaustive
    target, explicit_anchor = plan.targets
    assert target.min_count == 2 and target.attributes == {"state": "clean"}
    assert target.demand == before["targets"][0]["demand"]
    assert target.detector_terms == ["writing surface", "work surface", "table"]
    assert target.match_terms == target.concept.match_terms == ["writing surface", "work surface"]
    assert target.concept.demand == target.demand
    assert explicit_anchor.detector_terms == ["lamp", "desk light"]
    assert explicit_anchor.role == "anchor" and not explicit_anchor.terminal
    implicit_anchor = plan.constraints[0].object_concept
    assert implicit_anchor.name == "work desk"
    assert implicit_anchor.role == "anchor" and not implicit_anchor.terminal
    assert implicit_anchor.detector_terms == implicit_anchor.aliases == ["work desk"]
    assert plan.constraints[1].object_concept is explicit_anchor.concept
    assert plan.constraints[0].subject_concept is target.concept
    assert implicit_anchor in plan.concept_queries
    assert task.as_dict() == before
