"""Compatibility entry point: strict semantic compilation then detector mapping.

The domain compiler never sees detector vocabulary or benchmark hidden state.
This adapter is retired when all consumers use TaskSpec and SceneQuery directly.
"""
from __future__ import annotations

from typing import Any, Iterable

from instruction.compiler import InitialInstructionCompiler, compile_instruction, task_from_parsed_plan
from instruction.parser import parse_instruction_with_llm
from backends.habitat.task_input import plan_from_episode_info
from perception.detector_adapter import ground_plan
from .contracts import InstructionPlan
from .ontology import normalize_term
from .task_compat import runtime_plan

# Predicate families handled by the current task/verification consumers.
# Path exclusions are deliberately absent until an enforcing backend exists.
RUNTIME_CONSTRAINT_TYPES = frozenset({
    "spatial", "relation", "object relation", "co occurrence", "room",
    "attribute", "color", "material", "state", "affordance", "sequence", "count",
})


def compile_instruction_plan(
    raw_instruction: str,
    dataset_target: str = "",
    episode_info: dict[str, Any] | None = None,
    available_classes: Iterable[str] | None = None,
    backend: str = "llm",
    vlm: str = "cognav",
    strict_available_classes: bool = False,
    *,
    initial_compiler: InitialInstructionCompiler | None = None,
) -> InstructionPlan:
    """Adapt explicit task inputs, rejecting unavailable language compilation.

    Only the explicitly selected legacy backend permits dataset fallback.
    Metadata is used only when supplied by the benchmark input boundary.
    The optional initial compiler owns language admission and its call budget.
    """
    used_initial_compiler = False
    mode = normalize_term(backend)
    if mode in {"legacy", "rules"}:
        parsed = parse_instruction_with_llm(raw_instruction, dataset_target, vlm=vlm)
        plan = runtime_plan(task_from_parsed_plan(parsed, source=mode))
    else:
        metadata = plan_from_episode_info(raw_instruction, dataset_target, episode_info)
        if metadata is not None:
            task = task_from_parsed_plan(metadata, source="benchmark_metadata")
            unsupported = [c.type for c in task.constraints
                           if c.hardness == "hard" and c.type not in RUNTIME_CONSTRAINT_TYPES]
            if unsupported:
                raise ValueError(f"unsupported constraints: {unsupported}")
        else:
            if initial_compiler is not None:
                used_initial_compiler = True
                result = initial_compiler.compile(raw_instruction, RUNTIME_CONSTRAINT_TYPES)
            else:
                result = compile_instruction(raw_instruction, RUNTIME_CONSTRAINT_TYPES, vlm=vlm)
            if result.task is None:
                raise ValueError(f"{result.status}: {'; '.join(result.diagnostics)}")
            task = result.task
        plan = runtime_plan(task)
        if used_initial_compiler:
            plan.diagnostics["initial_model_calls"] = initial_compiler.model_calls
    return ground_plan(plan, available_classes=available_classes, vlm=vlm,
                       strict_available_classes=strict_available_classes,
                       allow_model_calls=not used_initial_compiler,
                       use_legacy_similarity_fallback=mode in {"legacy", "rules"})
