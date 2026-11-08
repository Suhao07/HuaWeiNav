from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Iterable

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import CONCEPT_GROUNDING
from prompting.schemas import HAS_PYDANTIC, GroundingResult
from prompting.templates import CONCEPT_GROUNDING_PROMPT

from instruction_adapter.contracts import ConceptQuery, Constraint, InstructionPlan, TargetQuery
from instruction_adapter.ontology import dedupe_terms, filter_terms_to_available, normalize_term


def _llm_offline() -> bool:
    return os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on")


def _exact_or_compact_matches(candidates: Iterable[str], classes: Iterable[str]) -> list[str]:
    class_map = {normalize_term(cls): str(cls) for cls in classes or []}
    compact_map = {normalize_term(cls).replace(" ", "_"): str(cls) for cls in classes or []}
    out = []
    for candidate in candidates:
        norm = normalize_term(candidate)
        if norm in class_map:
            out.append(class_map[norm])
        compact = norm.replace(" ", "_")
        if compact in compact_map:
            out.append(compact_map[compact])
    return dedupe_terms(out)


def _llm_ground_terms(concept: ConceptQuery, available_classes: Iterable[str], vlm: str) -> GroundingResult:
    """Map a concept into a bounded vocabulary proposal; model failures propagate.

    空词表表示模型明确给出的映射结果。传输或解析异常交给调用方处理，
    保留失败事实，避免将服务错误伪装成有效的空映射。
    """
    if _llm_offline() or not HAS_PYDANTIC:
        return GroundingResult()
    client, model = get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CONCEPT_GROUNDING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Target concept: {concept.name}\n"
                    f"Concept role: {concept.role}\n"
                    f"Concept terminal: {concept.terminal}\n"
                    f"Target aliases: {concept.aliases}\n"
                    f"Demand: {json.dumps(concept.demand, ensure_ascii=False)}\n"
                    f"Available detector classes: {list(available_classes or [])}"
                ),
            },
        ],
        response_format=GroundingResult,
        trace_label=CONCEPT_GROUNDING.trace_label,
    )
    return completion.choices[0].message.parsed


def ground_target(
    target: TargetQuery,
    available_classes: Iterable[str] | None = None,
    dataset_target: str = "",
    vlm: str = "cognav",
    strict_available_classes: bool = False,
    *,
    allow_model_calls: bool = True,
) -> TargetQuery:
    """Ground one semantic target into detector terms.

    The function is intentionally target-local. It never expands to support
    objects; those remain in SearchPriors.
    """

    available = list(available_classes or [])
    base_concept = target.concept_query()
    aliases = dedupe_terms(list(target.aliases) + list(base_concept.aliases))
    seeds = dedupe_terms([base_concept.name] + aliases + list(base_concept.detector_terms))
    dataset = normalize_term(dataset_target)

    exact = _exact_or_compact_matches(seeds, available)
    detector_terms = exact or filter_terms_to_available(seeds, available)
    description = base_concept.description
    negative_terms = list(base_concept.negative_terms)

    if not detector_terms and available and allow_model_calls:
        grounded = _llm_ground_terms(base_concept, available, vlm)
        detector_terms = filter_terms_to_available(grounded.detector_terms, available)

    if not detector_terms and dataset and target.terminal:
        dataset_matches = _exact_or_compact_matches([dataset], available)
        detector_terms = dataset_matches or [dataset]
    if not detector_terms and not strict_available_classes:
        detector_terms = seeds or [target.name]

    concept = ConceptQuery(
        id=target.id,
        name=target.name,
        role=target.role,
        detector_terms=dedupe_terms(detector_terms),
        aliases=aliases,
        description=description,
        negative_terms=dedupe_terms(negative_terms),
        terminal=target.terminal,
        demand=dict(target.demand),
        source=target.source or "grounding",
    )
    return replace(
        target,
        detector_terms=dedupe_terms(detector_terms),
        aliases=aliases,
        concept=concept,
    )


def _target_by_name_or_id(plan: InstructionPlan, value: str) -> TargetQuery | None:
    key = normalize_term(value)
    if not key:
        return None
    for target in plan.targets:
        if key == normalize_term(target.id) or key in {normalize_term(x) for x in target.match_terms}:
            return target
    return None


def _constraint_concept_id(name: str, index: int) -> str:
    base = normalize_term(name).replace(" ", "_") or "anchor"
    return f"c{index}_{base}"


def _concept_from_constraint(plan: InstructionPlan, constraint: Constraint, index: int, available_classes,
                             vlm: str, allow_model_calls: bool = True) -> ConceptQuery | None:
    object_name = normalize_term(constraint.object)
    if not object_name and isinstance(constraint.value, str):
        object_name = normalize_term(constraint.value)
    if not object_name:
        return None
    existing = _target_by_name_or_id(plan, object_name)
    if existing is not None:
        return existing.concept_query()
    concept = constraint.object_concept or ConceptQuery(
        id=_constraint_concept_id(object_name, index),
        name=object_name,
        role="anchor",
        detector_terms=[object_name],
        aliases=[object_name],
        terminal=False,
        source=constraint.source or "constraint",
    )
    grounded = (_llm_ground_terms(concept, available_classes, vlm)
                if available_classes and allow_model_calls else GroundingResult())
    detector_terms = filter_terms_to_available(grounded.detector_terms, available_classes or [])
    if not detector_terms:
        detector_terms = _exact_or_compact_matches([object_name], available_classes or []) or [object_name]
    return replace(concept, detector_terms=dedupe_terms(detector_terms))


def ground_plan(
    plan: InstructionPlan,
    available_classes: Iterable[str] | None = None,
    vlm: str = "cognav",
    strict_available_classes: bool = False,
    use_legacy_similarity_fallback: bool = True,
    *,
    allow_model_calls: bool = True,
) -> InstructionPlan:
    grounded_targets = [
        ground_target(
            target,
            available_classes=available_classes,
            dataset_target=plan.dataset_target,
            vlm=vlm,
            strict_available_classes=strict_available_classes,
            allow_model_calls=allow_model_calls,
        )
        for target in plan.targets
    ]
    if allow_model_calls and use_legacy_similarity_fallback and len(grounded_targets) == 1:
        target = grounded_targets[0]
        if target.terminal and len(target.detector_terms) <= 1 and not _llm_offline():
            from cv_utils.gpt_utils import ask_gpt_similar_objects

            similar = ask_gpt_similar_objects(list(available_classes or []), target.name, vlm)
            detector_terms = dedupe_terms(target.detector_terms + similar)
            # 检索扩展不提供身份证据；先同步概念，再绑定约束引用。
            grounded_targets[0] = replace(
                target,
                detector_terms=detector_terms,
                concept=replace(target.concept_query(), detector_terms=detector_terms),
            )
    plan.targets = grounded_targets

    concept_queries: list[ConceptQuery] = []
    for target in grounded_targets:
        concept_queries.append(target.concept_query())

    grounded_constraints = []
    for idx, constraint in enumerate(plan.constraints):
        obj_concept = _concept_from_constraint(plan, constraint, idx, available_classes or [], vlm,
                                               allow_model_calls=allow_model_calls)
        subject_target = _target_by_name_or_id(plan, constraint.subject)
        grounded_constraints.append(
            replace(
                constraint,
                object_concept=obj_concept or constraint.object_concept,
                subject_concept=subject_target.concept_query() if subject_target is not None else constraint.subject_concept,
            )
        )
        if obj_concept is not None:
            concept_queries.append(obj_concept)
    plan.constraints = grounded_constraints
    plan.concept_queries = dedupe_concepts(concept_queries)

    plan.valid = bool(plan.terminal_targets and plan.target_detector_prompts)
    plan.diagnostics.setdefault("grounding", {})
    plan.diagnostics["grounding"].update(
        {
            "available_class_count": len(list(available_classes or [])),
            "strict_available_classes": strict_available_classes,
        }
    )
    return plan


def dedupe_concepts(values: Iterable[ConceptQuery]) -> list[ConceptQuery]:
    seen = set()
    out = []
    for concept in values or []:
        key = concept.id or normalize_term(concept.name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(concept)
    return out
