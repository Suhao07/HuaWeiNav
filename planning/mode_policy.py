from __future__ import annotations

from typing import Any

from instruction_adapter.ontology import compact_key


ANCHOR_FIRST_RELATION_SEARCH = "anchor_first_relation_search"


def execution_mode(plan: Any) -> str:
    return compact_key(getattr(getattr(plan, "execution", None), "mode", ""))


def is_anchor_first_relation_search(plan: Any) -> bool:
    return execution_mode(plan) == ANCHOR_FIRST_RELATION_SEARCH


def is_ordered_execution(plan: Any) -> bool:
    return bool(getattr(getattr(plan, "execution", None), "ordered", False))
