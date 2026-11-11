"""One-way TaskSpec projection for runtime consumers still using InstructionPlan.

Remove after query and verification consumers accept TaskSpec directly. This
projection is disposable; its mutations never modify the authoritative task.
"""
from dataclasses import replace

from instruction.contracts import TARGET_OBJECT_CONSTRAINT_TYPES
from instruction_adapter.contracts import (
    Constraint, ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery,
)
from instruction_adapter.ontology import normalize_term


def runtime_plan(task, *, plan=None):
    """Project canonical semantics while retaining grounded retrieval state."""
    data = task.as_dict()
    if plan is None:
        plan = InstructionPlan(raw_instruction=task.raw_instruction, valid=True,
                               diagnostics={"source": [task.source]})
    previous = {target.id: target for target in plan.targets}
    seeds = {concept.id: concept for concept in plan.concept_queries}
    for constraint in plan.constraints:
        for concept in (constraint.subject_concept, constraint.object_concept):
            if concept is not None:
                seeds[concept.id] = concept
    seeds.update({target.id: target.concept for target in plan.targets if target.concept is not None})
    targets = []
    for item in data["targets"]:
        old = previous.get(item["id"])
        target = (replace(old, **item) if old is not None else TargetQuery(
            **item, source=task.source,
            detector_terms=list(data["search_priors"].get("destination_hypotheses", {}).get(item["id"], [])),
        ))
        seed = seeds.get(target.id)
        target.concept = replace(
            seed or target.concept_query(), id=target.id, name=target.name,
            role=target.role, aliases=list(target.aliases), terminal=target.terminal,
            demand=dict(target.demand), attributes=dict(target.attributes),
            constraints=[item for item in data["constraints"] if item["subject"] == target.id
                         and item["hardness"] == "hard"
                         and normalize_term(item["type"]) in
                         {"attribute", "color", "material", "state", "affordance", "room"}],
            detector_terms=list(dict.fromkeys([*target.detector_terms, *(seed.detector_terms if seed else [])])),
        )
        targets.append(target)
    concepts = {target.id: target.concept_query() for target in targets}
    plan.raw_instruction = task.raw_instruction
    plan.targets = targets
    plan.constraints = [Constraint(
        **item,
        subject_concept=concepts.get(item["subject"]),
        object_concept=(concepts.get(item["object"])
                        if normalize_term(item["type"]) in TARGET_OBJECT_CONSTRAINT_TYPES else None),
    ) for item in data["constraints"]]
    plan.concept_queries = [*concepts.values(), *(concept for concept in plan.concept_queries
                                                if concept.id not in concepts)]
    plan.execution = replace(plan.execution, mode=task.completion_mode,
                             ordered=task.ordered, exhaustive=task.exhaustive)
    plan.eval_mode = task.completion_mode
    plan.search_priors = SearchPriors(**data["search_priors"])
    plan.diagnostics["task_spec"] = data
    return plan
