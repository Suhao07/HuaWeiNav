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

from .contracts import InstructionPlan, TargetQuery
from .ontology import dedupe_terms, filter_terms_to_available, normalize_term


class GroundingResult(BaseModel):
    detector_terms: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    reason: str = ""


GROUNDING_PROMPT = """
You map an instruction target concept to detector vocabulary for open-vocabulary
object navigation.

Return JSON only. Choose detector_terms from the provided available classes when
possible. Include aliases only if they are plausible names for the same target
concept, not nearby/support objects. Do not include room names or support
objects as detector terms.
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


def _llm_ground_terms(target: TargetQuery, available_classes: Iterable[str], vlm: str) -> GroundingResult:
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
                        f"Target concept: {target.name}\n"
                        f"Target aliases: {target.aliases}\n"
                        f"Target attributes: {target.attributes}\n"
                        f"Available detector classes: {list(available_classes or [])}"
                    ),
                },
            ],
            response_format=GroundingResult,
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
    seeds = dedupe_terms([target.name] + list(target.aliases) + list(target.detector_terms))
    dataset = normalize_term(dataset_target)

    exact = _exact_or_compact_matches(seeds, available)
    detector_terms = exact or filter_terms_to_available(seeds, available)

    if not detector_terms and available:
        grounded = _llm_ground_terms(target, available, vlm)
        detector_terms = filter_terms_to_available(grounded.detector_terms, available)
        aliases = dedupe_terms(list(target.aliases) + grounded.aliases + seeds)
    else:
        aliases = dedupe_terms(list(target.aliases) + seeds)

    if not detector_terms and dataset:
        dataset_matches = _exact_or_compact_matches([dataset], available)
        detector_terms = dataset_matches or [dataset]
    if not detector_terms and not strict_available_classes:
        detector_terms = seeds or [target.name]

    return replace(
        target,
        detector_terms=dedupe_terms(detector_terms),
        aliases=dedupe_terms(aliases + detector_terms),
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

    plan.targets = grounded_targets
    plan.valid = bool(plan.terminal_targets and plan.target_detector_prompts)
    plan.diagnostics.setdefault("grounding", {})
    plan.diagnostics["grounding"].update(
        {
            "available_class_count": len(list(available_classes or [])),
            "strict_available_classes": strict_available_classes,
        }
    )
    return plan
