from __future__ import annotations

from typing import Any

from instruction_adapter.ontology import compact_key


ANCHOR_FIRST_RELATION_SEARCH = "anchor_first_relation_search"


def execution_mode(plan: Any) -> str:
    return compact_key(getattr(getattr(plan, "execution", None), "mode", ""))


def is_anchor_first_relation_search(plan: Any) -> bool:
    # Search strategy is inferred without rewriting ordered/count semantics.
    if execution_mode(plan) == ANCHOR_FIRST_RELATION_SEARCH:
        return True  # Explicit legacy strategy remains supported during migration.
    return any(
        compact_key(getattr(c, "type", "")) in
        {"spatial", "relation", "object_relation", "co_occurrence"}
        and getattr(c, "object_concept", None) is not None
        and not c.object_concept.terminal
        for c in getattr(plan, "constraints", ())
    )


def is_ordered_execution(plan: Any) -> bool:
    return bool(getattr(getattr(plan, "execution", None), "ordered", False))
