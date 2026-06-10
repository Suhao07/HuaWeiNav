from __future__ import annotations

import os
from dataclasses import replace
from typing import Iterable

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

from .contracts import ConceptQuery, Constraint, ExecutionPolicy, InstructionPlan, TargetQuery
from .ontology import dedupe_terms, filter_terms_to_available, normalize_term


class GroundingResult(BaseModel):
    detector_terms: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    negative_terms: list[str] = Field(default_factory=list)
    reason: str = ""


class ExecutionStrategyResult(BaseModel):
    mode: str = "any_target_success"
    use_anchor_first: bool = False
    anchor_concept_ids: list[str] = Field(default_factory=list)
    reason: str = ""


GROUNDING_PROMPT = """
You map an instruction target concept to detector vocabulary for open-vocabulary
object navigation.

Return JSON only. Choose detector_terms from the provided available classes when
possible. Include aliases only if they are plausible names for the same target
concept, not nearby/support objects. Do not include room names or support
objects as detector terms.
Also provide a concise semantic description and negative_terms that distinguish
this concept from related but different concepts.
"""


EXECUTION_STRATEGY_PROMPT = """
You choose an execution strategy for an indoor navigation instruction.

Return JSON only. Use anchor_first only when the instruction contains a
non-terminal reference/anchor object that can guide search for a harder terminal
target. The anchor itself must not satisfy the final goal.
Do not use object-name rules; reason from the structured plan.
"""


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
    if _llm_offline() or not HAS_PYDANTIC:
        return GroundingResult()
    try:
        client, model = get_client_and_model(vlm)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": GROUNDING_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Target concept: {concept.name}\n"
                        f"Concept role: {concept.role}\n"
                        f"Concept terminal: {concept.terminal}\n"
                        f"Target aliases: {concept.aliases}\n"
                        f"Available detector classes: {list(available_classes or [])}"
                    ),
                },
            ],
            response_format=GroundingResult,
            trace_label="concept_grounding",
        )
        return completion.choices[0].message.parsed or GroundingResult()
    except Exception:
        return GroundingResult()


def ground_target(
    target: TargetQuery,
    available_classes: Iterable[str] | None = None,
    dataset_target: str = "",
    vlm: str = "cognav",
    strict_available_classes: bool = False,
) -> TargetQuery:
    """Ground one semantic target into detector terms.

    The function is intentionally target-local. It never expands to support
    objects; those remain in SearchPriors.
    """

    available = list(available_classes or [])
    base_concept = target.concept_query()
    seeds = dedupe_terms([base_concept.name] + list(base_concept.aliases) + list(base_concept.detector_terms))
    dataset = normalize_term(dataset_target)

    exact = _exact_or_compact_matches(seeds, available)
    detector_terms = exact or filter_terms_to_available(seeds, available)
    description = base_concept.description
    negative_terms = list(base_concept.negative_terms)

    if not detector_terms and available:
        grounded = _llm_ground_terms(base_concept, available, vlm)
        detector_terms = filter_terms_to_available(grounded.detector_terms, available)
        aliases = dedupe_terms(list(target.aliases) + grounded.aliases + seeds)
        description = grounded.description or description
        negative_terms = dedupe_terms(negative_terms + grounded.negative_terms)
    else:
        aliases = dedupe_terms(list(target.aliases) + seeds)

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
        aliases=dedupe_terms(aliases + detector_terms),
        description=description,
        negative_terms=dedupe_terms(negative_terms),
        terminal=target.terminal,
        source=target.source or "grounding",
    )
    return replace(
        target,
        detector_terms=dedupe_terms(detector_terms),
        aliases=dedupe_terms(aliases + detector_terms),
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


def _concept_from_constraint(plan: InstructionPlan, constraint: Constraint, index: int, available_classes, vlm: str) -> ConceptQuery | None:
    object_name = normalize_term(constraint.object)
    if not object_name and isinstance(constraint.value, str):
        object_name = normalize_term(constraint.value)
    if not object_name:
        return None
    existing = _target_by_name_or_id(plan, object_name)
    if existing is not None:
        return existing.concept_query()
    concept = ConceptQuery(
        id=_constraint_concept_id(object_name, index),
        name=object_name,
        role="anchor",
        detector_terms=[object_name],
        aliases=[object_name],
        terminal=False,
        source=constraint.source or "constraint",
    )
    # Reuse the same prompt-first grounding path for constraint anchors.  This
    # is the key difference from string alias rules: constraint.object is
    # grounded as a concept just like a TargetQuery.
    grounded = _llm_ground_terms(concept, available_classes or [], vlm)
    detector_terms = filter_terms_to_available(grounded.detector_terms, available_classes or [])
    if not detector_terms:
        detector_terms = _exact_or_compact_matches([object_name], available_classes or []) or [object_name]
    return replace(
        concept,
        detector_terms=dedupe_terms(detector_terms),
        aliases=dedupe_terms([object_name] + grounded.aliases + detector_terms),
        description=grounded.description,
        negative_terms=dedupe_terms(grounded.negative_terms),
    )


def _choose_execution_strategy(plan: InstructionPlan, vlm: str) -> ExecutionPolicy:
    has_relation_anchor = any(
        normalize_term(getattr(constraint, "type", "")) in ("spatial", "relation", "object_relation", "co_occurrence")
        and getattr(constraint, "object_concept", None) is not None
        and not getattr(constraint.object_concept, "terminal", False)
        for constraint in getattr(plan, "constraints", []) or []
    )
    if _llm_offline() or not HAS_PYDANTIC:
        if has_relation_anchor:
            return replace(plan.execution, mode="anchor_first_relation_search", ordered=False, exhaustive=False)
        return plan.execution
    if not plan.anchor_targets and not any(c.object_concept for c in plan.constraints):
        return plan.execution
    try:
        client, model = get_client_and_model(vlm)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": EXECUTION_STRATEGY_PROMPT},
                {"role": "user", "content": str(plan.as_dict())},
            ],
            response_format=ExecutionStrategyResult,
            trace_label="execution_strategy",
        )
        parsed = completion.choices[0].message.parsed or ExecutionStrategyResult()
    except Exception:
        return plan.execution
    if not parsed.use_anchor_first and not has_relation_anchor:
        return plan.execution
    return replace(
        plan.execution,
        mode="anchor_first_relation_search",
        ordered=False,
        exhaustive=False,
    )


def ground_plan(
    plan: InstructionPlan,
    available_classes: Iterable[str] | None = None,
    vlm: str = "cognav",
    strict_available_classes: bool = False,
    use_legacy_similarity_fallback: bool = True,
) -> InstructionPlan:
    grounded_targets = [
        ground_target(
            target,
            available_classes=available_classes,
            dataset_target=plan.dataset_target,
            vlm=vlm,
            strict_available_classes=strict_available_classes,
        )
        for target in plan.targets
    ]
    plan.targets = grounded_targets

    concept_queries: list[ConceptQuery] = []
    for target in grounded_targets:
        concept_queries.append(target.concept_query())

    grounded_constraints = []
    for idx, constraint in enumerate(plan.constraints):
        obj_concept = _concept_from_constraint(plan, constraint, idx, available_classes or [], vlm)
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
    plan.execution = _choose_execution_strategy(plan, vlm)

    if use_legacy_similarity_fallback and len(grounded_targets) == 1:
        target = grounded_targets[0]
        if target.terminal and len(target.detector_terms) <= 1 and not _llm_offline():
            try:
                from cv_utils.gpt_utils import ask_gpt_similar_objects

                similar = ask_gpt_similar_objects(list(available_classes or []), target.name, vlm)
                target = replace(
                    target,
                    detector_terms=dedupe_terms(target.detector_terms + similar),
                    aliases=dedupe_terms(target.aliases + similar),
                )
                grounded_targets[0] = target
            except Exception:
                pass

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
