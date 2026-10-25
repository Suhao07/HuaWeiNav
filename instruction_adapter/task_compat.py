"""One-way TaskSpec projection for runtime consumers still using InstructionPlan.

Remove after query and verification consumers accept TaskSpec directly. This
projection is disposable; its mutations never modify the authoritative task.
"""
from instruction_adapter.contracts import (
    Constraint, ExecutionPolicy, InstructionPlan, SearchPriors, TargetQuery,
)


def runtime_plan(task):
    """Build a detached legacy runtime view without adding detector semantics."""
    data = task.as_dict()
    return InstructionPlan(
        raw_instruction=task.raw_instruction,
        targets=[TargetQuery(**t, source=task.source) for t in data["targets"]],
        constraints=[Constraint(**{k: v for k, v in c.items() if k != "id"})
                     for c in data["constraints"]],
        execution=ExecutionPolicy(mode=task.completion_mode, ordered=task.ordered,
                                  exhaustive=task.exhaustive),
        eval_mode=task.completion_mode,
        search_priors=SearchPriors(**data["search_priors"]),
        valid=True,
        diagnostics={"source": [task.source], "task_spec": data},
    )
