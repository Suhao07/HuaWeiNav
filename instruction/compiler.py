"""Single strict natural-language compiler. Dataset inputs use backend adapters."""
from __future__ import annotations

from instruction.contracts import CompileResult, ConstraintSpec, TargetSpec, TaskSpec
from instruction.parser import InstructionParseUnavailable, parse_instruction_with_llm


def task_from_parsed_plan(plan, *, revision=1, source="language") -> TaskSpec:
    """Freeze parsed semantics before any detector mapping or strategy choice."""
    return TaskSpec(
        raw_instruction=plan.raw_instruction,
        targets=tuple(TargetSpec(
            id=t.id, name=t.name, role=t.role, terminal=t.terminal,
            min_count=t.min_count, aliases=tuple(t.aliases), attributes=t.attributes,
        ) for t in plan.targets),
        constraints=tuple(ConstraintSpec(
            id=f"c{i}", type=c.type, subject=c.subject, relation=c.relation,
            object=c.object, value=c.value, hardness=c.hardness,
            verifier=c.verifier, source=c.source,
        ) for i, c in enumerate(plan.constraints)),
        revision=revision, ordered=plan.execution.ordered,
        exhaustive=plan.execution.exhaustive,
        completion_mode=plan.eval_mode,
        search_priors=plan.search_priors.as_dict(), source=source,
    )


def compile_instruction(text, capabilities, *, vlm="cognav", revision=1) -> CompileResult:
    """Parse once and reject unsupported hard predicates without executing them.

    Args:
        text: User-visible instruction, without hidden benchmark metadata.
        capabilities: Supported constraint types advertised by the backend.
        vlm: Existing configured model provider.
        revision: Task revision allocated by the task owner.
    """
    if not str(text or "").strip():
        return CompileResult("invalid", diagnostics=("empty_instruction",))
    try:
        plan = parse_instruction_with_llm(text, vlm=vlm, strict=True)
    except InstructionParseUnavailable as exc:
        return CompileResult("parse_unavailable", diagnostics=(str(exc),))
    if plan.diagnostics.get("unresolved"):
        return CompileResult("ambiguous", diagnostics=tuple(plan.diagnostics["unresolved"]))
    try:
        task = task_from_parsed_plan(plan, revision=revision)
    except (ValueError, TypeError) as exc:
        return CompileResult("invalid", diagnostics=(str(exc),))
    unsupported = tuple(c.id + ":" + c.type for c in task.constraints
                        if c.hardness == "hard" and c.type not in capabilities)
    if unsupported:
        return CompileResult("unsupported", diagnostics=unsupported)
    return CompileResult("success", task)
