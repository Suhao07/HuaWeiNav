from __future__ import annotations

from typing import Any

from .contracts import Constraint, ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery
from .ontology import dedupe_terms, normalize_term


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _norm_list(value) -> list[str]:
    return dedupe_terms(str(item) for item in _as_list(value))


def _task_execution(task_type: str, eval_mode: str) -> ExecutionPolicy:
    task = normalize_term(task_type)
    mode = normalize_term(eval_mode) or "any_target_success"
    return ExecutionPolicy(
        mode=mode,
        ordered="sequential" in task or "sequence" in task or "ordered" in mode,
        exhaustive="exhaustive" in task or "multi_instance" in task or "all" in mode,
    )


def _target_id(name: str, index: int) -> str:
    base = normalize_term(name).replace(" ", "_") or "target"
    return f"t{index}_{base}"


def plan_from_episode_info(
    raw_instruction: str,
    dataset_target: str = "",
    episode_info: dict[str, Any] | None = None,
) -> InstructionPlan | None:
    """Compile CogNav/Habitat episode metadata into InstructionPlan.

    中文说明：CogNav 的 instruction benchmark 已经在 episode.info 中保存了
    结构化语义。这里优先复用这些字段，避免用手写规则重新猜指令含义。
    """

    info = episode_info or {}
    if not isinstance(info, dict):
        return None

    has_instruction_metadata = any(
        key in info for key in (
            "instruction_targets",
            "candidate_targets",
            "candidate_concepts",
            "expected_concepts",
            "target_sequence",
            "room_constraints",
            "min_instance_counts",
            "complex_constraints",
            "task_type",
            "eval_mode",
            "contract",
            "scene_groundings",
        )
    )
    if not has_instruction_metadata:
        return None

    raw = str(raw_instruction or info.get("instruction") or "").strip()
    dataset = normalize_term(dataset_target or info.get("ovon_object_category") or info.get("object_category") or "")
    contract = info.get("contract") or {}
    if not isinstance(contract, dict):
        contract = {}

    instruction_targets = _norm_list(info.get("instruction_targets") or contract.get("canonical_target_concepts"))
    candidate_targets = _norm_list(
        info.get("candidate_targets")
        or info.get("candidate_concepts")
        or contract.get("candidate_target_concepts")
        or contract.get("detector_prompts")
    )
    target_sequence = _norm_list(info.get("target_sequence"))
    primary_targets = _norm_list(
        info.get("primary_targets")
        or info.get("expected_concepts")
        or contract.get("primary_target_concepts")
    )
    task_type = normalize_term(info.get("task_type", "") or contract.get("task_type", ""))
    if not task_type:
        task_type = "implicit_object_goal" if candidate_targets and raw else "object_goal"
    eval_mode = normalize_term(info.get("eval_mode", ""))
    if not eval_mode:
        eval_mode = "any_target_success"
    execution = _task_execution(task_type, eval_mode)

    if execution.ordered and target_sequence:
        target_names = target_sequence
    else:
        target_names = primary_targets or instruction_targets or candidate_targets or [dataset]
    target_names = dedupe_terms(target_names)
    secondary_targets = dedupe_terms(
        _norm_list(info.get("secondary_targets"))
        + _norm_list(contract.get("secondary_target_concepts"))
    )

    min_counts = info.get("min_instance_counts") or {}
    if not isinstance(min_counts, dict):
        min_counts = {}

    targets = []
    for idx, name in enumerate(target_names):
        min_count = int(min_counts.get(name, 1) or 1)
        targets.append(
            TargetQuery(
                id=_target_id(name, idx),
                name=name,
                role="primary",
                detector_terms=[name],
                aliases=[name],
                min_count=min_count,
                terminal=True,
                source="episode_info",
            )
        )

    for name in secondary_targets:
        if name in target_names:
            continue
        targets.append(
            TargetQuery(
                id=_target_id(name, len(targets)),
                name=name,
                role="secondary",
                detector_terms=[name],
                aliases=[name],
                min_count=1,
                terminal=False,
                source="episode_info",
            )
        )

    constraints: list[Constraint] = []
    room_hints = dedupe_terms(
        _norm_list(info.get("room_constraints"))
        + _norm_list(info.get("room_hints"))
        + _norm_list(contract.get("room_hints"))
    )
    for room in room_hints:
        constraints.append(
            Constraint(
                type="room",
                relation="in",
                value=room,
                hardness="soft",
                verifier="planner",
                source="episode_info",
            )
        )
    for target in targets:
        if target.min_count > 1:
            constraints.append(
                Constraint(
                    type="count",
                    subject=target.id,
                    relation=">=",
                    value=target.min_count,
                    hardness="hard",
                    verifier="planner",
                    source="episode_info",
                )
            )

    terminal_targets = [target for target in targets if target.terminal]
    if len(terminal_targets) > 1 and execution.ordered:
        for prev, nxt in zip(terminal_targets, terminal_targets[1:]):
            constraints.append(
                Constraint(
                    type="sequence",
                    subject=prev.id,
                    relation="before",
                    object=nxt.id,
                    hardness="hard",
                    verifier="planner",
                    source="episode_info",
                )
            )

    support_objects = _norm_list(contract.get("support_context_concepts"))
    if normalize_term(info.get("support_policy", "")) not in ("none", ""):
        # support_policy is a policy name, not an object class. Keep it as an
        # affordance hint instead of inventing support objects.
        affordances = [normalize_term(info.get("support_policy", ""))]
    else:
        affordances = _norm_list(contract.get("affordances"))

    plan = InstructionPlan(
        raw_instruction=raw,
        dataset_target=dataset,
        task_type=task_type,
        eval_mode=eval_mode,
        targets=targets,
        constraints=constraints,
        search_priors=SearchPriors(
            room_hints=room_hints,
            support_objects=dedupe_terms(support_objects),
            affordances=dedupe_terms(affordances),
        ),
        execution=execution,
        valid=bool(targets),
        diagnostics={
            "source": ["episode_info"],
            "instruction_id": info.get("instruction_id", ""),
            "target_role": info.get("target_role", ""),
            "complex_constraints": info.get("complex_constraints", {}),
        },
    )
    return plan
