from __future__ import annotations

import os
from typing import Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ModuleNotFoundError:
    HAS_PYDANTIC = False

    class BaseModel:
        def __init__(self, **kwargs):
            for key in getattr(self, "__annotations__", {}):
                default = getattr(type(self), key, None)
                if isinstance(default, (list, dict, set)):
                    default = default.copy()
                setattr(self, key, kwargs.get(key, default))

    def Field(default=None, default_factory=None, **_kwargs):
        return default_factory() if default_factory is not None else default

from llm_utils.cognav_llm_adapter import get_client_and_model

from .contracts import Constraint, ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery
from .ontology import dedupe_terms, normalize_term


class ParsedTarget(BaseModel):
    name: str = ""
    role: str = Field(default="primary", description="primary, secondary, anchor, or support")
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    min_count: int = 1
    terminal: bool = True


class ParsedConstraint(BaseModel):
    type: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    value: Any = None
    hardness: str = "soft"
    verifier: str = "planner"


class ParsedInstruction(BaseModel):
    task_type: str = "object_goal"
    eval_mode: str = "any_target_success"
    targets: list[ParsedTarget] = Field(default_factory=list)
    constraints: list[ParsedConstraint] = Field(default_factory=list)
    room_hints: list[str] = Field(default_factory=list)
    support_objects: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)
    requires_runtime_relation: bool = False


INSTRUCTION_PARSE_PROMPT = """
You are a semantic compiler for indoor object navigation instructions.

Return only structured JSON matching the schema. Do not solve navigation and do
not invent scene-specific facts. Extract what the user asks for:
- targets: object concepts mentioned or implied by the instruction.
- terminal=true only for objects that may satisfy the final goal.
- anchors/support objects must be terminal=false.
- constraints: room, spatial, sequence, count, area, attribute, co_occurrence.
- Use hard constraints for explicit requirements in the instruction.
- Use soft constraints only for hints explicitly present in the instruction.
- Do not encode common-sense priors such as "TV is in living room" unless the
  room or context is explicitly stated by the instruction.

Examples:
Instruction: "Find a cup on the table in the kitchen."
targets: cup terminal true; table role anchor terminal false.
constraints: room cup in kitchen hard; spatial cup on table hard verifier vlm.

Instruction: "I need somewhere to sit."
targets: object or affordance concept "seat" terminal true, with affordance sitting.
Do not choose chair/sofa by hard-coded prior; detector grounding will map it.

Instruction: "First find the bed, then locate the towel."
targets: bed terminal true; towel terminal true.
constraints: sequence bed before towel hard.
"""


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
            trace_label="instruction_parser",
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
