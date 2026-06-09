from __future__ import annotations

from typing import Iterable


def normalize_term(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def compact_key(value: str) -> str:
    return normalize_term(value).replace(" ", "_")


def dedupe_terms(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values or []:
        raw = str(value or "").strip()
        key = normalize_term(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def filter_terms_to_available(terms: Iterable[str], available_classes: Iterable[str]) -> list[str]:
    available = {normalize_term(x) for x in available_classes or []}
    if not available:
        return dedupe_terms(terms)
    return [term for term in dedupe_terms(terms) if normalize_term(term) in available]


def first_non_empty(*values: str) -> str:
    for value in values:
        text = normalize_term(value)
        if text:
            return text
    return ""
