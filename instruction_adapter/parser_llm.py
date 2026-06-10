from __future__ import annotations

import os

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import INSTRUCTION_PARSE
from prompting.schemas import HAS_PYDANTIC, ParsedConstraint, ParsedInstruction, ParsedTarget
from prompting.templates import INSTRUCTION_PARSE_PROMPT

from .contracts import Constraint, ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery
from .ontology import dedupe_terms, normalize_term


def _target_id(name: str, index: int) -> str:
    base = normalize_term(name).replace(" ", "_") or "target"
    return f"t{index}_{base}"


def _fallback_plan(raw_instruction: str, dataset_target: str, reason: str) -> InstructionPlan:
    target = normalize_term(dataset_target) or normalize_term(raw_instruction)
    targets = []
    if target:
        targets.append(
            TargetQuery(
                id=_target_id(target, 0),
                name=target,
                detector_terms=[target],
                aliases=[target],
                terminal=True,
                source="fallback",
            )
        )
    return InstructionPlan(
        raw_instruction=str(raw_instruction or ""),
        dataset_target=normalize_term(dataset_target),
        targets=targets,
        valid=bool(targets),
        diagnostics={"source": ["fallback"], "fallback_reason": reason},
    )


def parse_instruction_with_llm(
    raw_instruction: str,
    dataset_target: str = "",
    vlm: str = "cognav",
) -> InstructionPlan:
    """Parse free-form instructions with the configured CogNav LLM client."""

    raw = str(raw_instruction or "").strip()
    dataset = normalize_term(dataset_target)
    if not HAS_PYDANTIC:
        return _fallback_plan(raw, dataset, "pydantic_unavailable")
    if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on"):
        return _fallback_plan(raw, dataset, "llm_offline")

    try:
        client, model = get_client_and_model(vlm)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTION_PARSE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Instruction: {raw}\n"
                        f"Dataset target fallback: {dataset or 'None'}"
                    ),
                },
            ],
            response_format=ParsedInstruction,
            trace_label=INSTRUCTION_PARSE.trace_label,
        )
        parsed = completion.choices[0].message.parsed
    except Exception as exc:
        return _fallback_plan(raw, dataset, f"llm_parse_failed: {exc}")

    if parsed is None:
        return _fallback_plan(raw, dataset, "empty_llm_parse")

    targets = []
    for idx, item in enumerate(parsed.targets):
        name = normalize_term(item.name)
        if not name:
            continue
        targets.append(
            TargetQuery(
                id=_target_id(name, idx),
                name=name,
                role=normalize_term(item.role) or "primary",
                detector_terms=[name],
                aliases=dedupe_terms(item.aliases + [name]),
                attributes=dict(item.attributes or {}),
                min_count=max(1, int(item.min_count or 1)),
                terminal=bool(item.terminal),
                source="llm",
            )
        )

    if not any(target.terminal for target in targets) and dataset:
        targets.insert(
            0,
            TargetQuery(
                id=_target_id(dataset, 0),
                name=dataset,
                detector_terms=[dataset],
                aliases=[dataset],
                terminal=True,
                source="dataset_fallback",
            ),
        )

    constraints = [
        Constraint(
            type=normalize_term(item.type),
            subject=normalize_term(item.subject),
            relation=normalize_term(item.relation),
            object=normalize_term(item.object),
            value=item.value,
            hardness=normalize_term(item.hardness) or "soft",
            verifier=normalize_term(item.verifier) or "planner",
            source="llm",
        )
        for item in parsed.constraints
        if normalize_term(item.type)
    ]
    return InstructionPlan(
        raw_instruction=raw,
        dataset_target=dataset,
        task_type=normalize_term(parsed.task_type) or "object_goal",
        eval_mode=normalize_term(parsed.eval_mode) or "any_target_success",
        targets=targets,
        constraints=constraints,
        search_priors=SearchPriors(
            room_hints=dedupe_terms(parsed.room_hints),
            support_objects=dedupe_terms(parsed.support_objects),
            affordances=dedupe_terms(parsed.affordances),
        ),
        execution=ExecutionPolicy(
            mode=normalize_term(parsed.eval_mode) or "any_target_success",
            ordered=any(c.type == "sequence" for c in constraints),
            exhaustive=any(c.type == "count" and str(c.relation) in (">=", "at least") for c in constraints),
        ),
        valid=bool(targets and any(target.terminal for target in targets)),
        diagnostics={
            "source": ["llm"],
            "requires_runtime_relation": bool(parsed.requires_runtime_relation),
        },
    )
