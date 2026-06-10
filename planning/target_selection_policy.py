"""Target candidate selection for benchmark and instruction modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from planning.mode_policy import is_ordered_execution


@dataclass
class TargetSelectionResult:
    found: bool
    obj: Any | None = None
    answer: str = ""
    anchor_record: dict[str, Any] = field(default_factory=dict)
    skipped_objs: list[dict[str, Any]] = field(default_factory=list)
    concept_debug: list[dict[str, Any]] = field(default_factory=list)


def select_target_candidate(mapper: Any, *, plan: Any | None, step: int | None = None) -> TargetSelectionResult:
    """Select the next candidate object for target verification.

    Instruction mode delegates to ``InstructionObjectSearchPolicy``. Benchmark
    mode stays class-label based and never calls LLM/VLM.
    """

    if plan is not None:
        _prepare_instruction_target_state(mapper, plan, step)
        result = mapper.instruction_object_search_policy.select(mapper, plan=plan, step=step)
        return TargetSelectionResult(
            found=result.found,
            obj=result.obj,
            answer=result.answer,
            anchor_record=dict(result.anchor_record or {}),
            skipped_objs=list(result.skipped_objs or []),
            concept_debug=list(result.concept_debug or []),
        )

    target_objs = [obj for obj in mapper.objects if mapper._is_target_tag(obj.tag)]
    if not target_objs:
        return TargetSelectionResult(
            found=False,
            answer=f"No target object '{mapper.target}' found; accepted aliases: {sorted(mapper._target_match_terms())}",
        )

    target_objs = sorted(target_objs, key=lambda x: x.confidence.numpy().item(), reverse=True)
    target_obj = target_objs[0]
    if target_obj.tag != mapper.target:
        target_obj.tag = mapper.target
        target_obj.conf_list[mapper.target] = target_obj.confidence
    return TargetSelectionResult(
        found=True,
        obj=target_obj,
        answer=(
            f"Choose Obj '{target_obj.tag}' at position {target_obj.position} "
            f"with confidence {target_obj.confidence.numpy().item()}."
        ),
    )


def _prepare_instruction_target_state(mapper: Any, plan: Any, step: int | None) -> None:
    # 指令模式先把当前 mapper object 注册到语义图，再按 plan 选择
    # terminal candidate 或 anchor reference。这里不做最终成功判断。
    mapper.instruction_constraint_evaluator.register_mapper_objects(
        mapper=mapper,
        step=step,
        current_node_idx=getattr(mapper, "current_node_idx", None),
    )
    state = mapper.instruction_constraint_evaluator.ensure_state(mapper, plan)
    active = state.active_target(plan)
    if active is not None and is_ordered_execution(plan):
        # sequence 模式只允许当前子目标触发 stop。
        # 后续目标即使被 detector 看见，也不能提前终止。
        mapper.target_list = list(active.detector_terms or [active.name])
        mapper.target_aliases = list(active.match_terms)
        mapper.target = active.name
