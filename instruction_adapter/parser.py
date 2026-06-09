from __future__ import annotations

import re
from typing import Any, Iterable

from .compiler import compile_instruction_plan
from .contracts import InstructionPlan, StriveInstructionSpec
from .ontology import normalize_term


def extract_dataset_target(instruct_goal: str) -> str:
    """Extract Habitat ObjectNav target from strings like ``Find <tv_monitor>``."""

    text = str(instruct_goal or "")
    match = re.search(r"<([^>]+)>", text)
    if match:
        return normalize_term(match.group(1))
    return normalize_term(text)


class StriveInstructionParser:
    """

    parse_plan() returns the canonical InstructionPlan. parse() keeps the old
    StriveInstructionSpec return type for existing call sites.
    """

    def __init__(
        self,
        backend: str = "llm",
        strict_available_classes: bool = False,
        vlm: str = "cognav",
    ):
        self.backend = str(backend or "llm")
        self.strict_available_classes = bool(strict_available_classes)
        self.vlm = str(vlm or "cognav")

    def parse_plan(
        self,
        raw_instruction: str,
        dataset_target: str = "",
        available_classes: Iterable[str] | None = None,
        episode_info: dict[str, Any] | None = None,
    ) -> InstructionPlan:
        return compile_instruction_plan(
            raw_instruction=raw_instruction,
            dataset_target=dataset_target,
            episode_info=episode_info,
            available_classes=available_classes,
            backend=self.backend,
            vlm=self.vlm,
            strict_available_classes=self.strict_available_classes,
        )

    def parse(
        self,
        raw_instruction: str,
        dataset_target: str = "",
        available_classes: Iterable[str] | None = None,
        episode_info: dict[str, Any] | None = None,
    ) -> StriveInstructionSpec:
        return self.parse_plan(
            raw_instruction=raw_instruction,
            dataset_target=dataset_target,
            available_classes=available_classes,
            episode_info=episode_info,
        ).to_legacy_spec()
