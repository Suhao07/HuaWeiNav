from __future__ import annotations

from typing import Any, Iterable

from .contracts import InstructionPlan, TargetQuery
from .grounding import ground_plan
from .ontology import normalize_term
from .parser_llm import parse_instruction_with_llm
from .parser_metadata import plan_from_episode_info


def _fallback_plan(raw_instruction: str, dataset_target: str) -> InstructionPlan:
    dataset = normalize_term(dataset_target)
    target_name = dataset or normalize_term(raw_instruction)
    targets = []
    if target_name:
        targets.append(
            TargetQuery(
                id=f"t0_{target_name.replace(' ', '_')}",
                name=target_name,
                detector_terms=[target_name],
                aliases=[target_name],
                terminal=True,
                source="dataset_fallback",
            )
        )
    return InstructionPlan(
        raw_instruction=str(raw_instruction or ""),
        dataset_target=dataset,
        targets=targets,
        valid=bool(targets),
        diagnostics={"source": ["dataset_fallback"]},
    )


def compile_instruction_plan(
    raw_instruction: str,
    dataset_target: str = "",
    episode_info: dict[str, Any] | None = None,
    available_classes: Iterable[str] | None = None,
    backend: str = "llm",
    vlm: str = "cognav",
    strict_available_classes: bool = False,
) -> InstructionPlan:
    """Compile instruction text/metadata into a grounded InstructionPlan.

    Source priority:
    1. CogNav episode.info metadata, when available.
    2. Prompt-first LLM structured parsing.
    3. Dataset target fallback.
    """

    info_plan = plan_from_episode_info(raw_instruction, dataset_target, episode_info)
    if info_plan is not None:
        plan = info_plan
    elif normalize_term(backend) in ("llm", "cognav", "prompt", "prompt_first"):
        plan = parse_instruction_with_llm(raw_instruction, dataset_target, vlm=vlm)
    else:
        # "rules" is kept as a compatibility backend name, but no target
        # ontology rules are used anymore.
        plan = _fallback_plan(raw_instruction, dataset_target)

    if not plan.targets:
        plan = _fallback_plan(raw_instruction, dataset_target)

    plan = ground_plan(
        plan,
        available_classes=available_classes,
        vlm=vlm,
        strict_available_classes=strict_available_classes,
    )
    plan.valid = bool(plan.terminal_targets and plan.target_detector_prompts)
    return plan
