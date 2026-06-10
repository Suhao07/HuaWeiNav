from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _dedupe_text(values) -> list[str]:
    seen = set()
    out = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower().replace("_", " ")
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


@dataclass
class ConceptQuery:
    """Unified instruction concept for terminal targets and anchors.

    ConceptQuery 是“指令概念”，不是 detector label。LLM/VLM
    grounding 产生 detector_terms、aliases、description 和 negative_terms
    并写入 plan.json 
    """

    id: str
    name: str
    role: str = "primary"
    detector_terms: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    negative_terms: list[str] = field(default_factory=list)
    terminal: bool = False
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def match_terms(self) -> list[str]:
        return _dedupe_text([self.name] + list(self.detector_terms) + list(self.aliases))


@dataclass
class TargetQuery:
    """One object concept extracted from an instruction.

    role controls how STRIVE uses the object. Only terminal=True targets may
    satisfy stop conditions; anchors/support objects are search or relation
    context.
    """

    id: str
    name: str
    role: str = "primary"
    detector_terms: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    min_count: int = 1
    terminal: bool = True
    source: str = ""
    concept: ConceptQuery | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def match_terms(self) -> list[str]:
        terms = [self.name] + list(self.detector_terms) + list(self.aliases)
        if self.concept is not None:
            terms.extend(self.concept.match_terms)
        return _dedupe_text(terms)

    def concept_query(self) -> ConceptQuery:
        if self.concept is not None:
            return self.concept
        return ConceptQuery(
            id=self.id,
            name=self.name,
            role=self.role,
            detector_terms=list(self.detector_terms),
            aliases=list(self.aliases),
            terminal=self.terminal,
            source=self.source,
        )


@dataclass
class Constraint:
    """Declarative task constraint.

    The parser only declares constraints. Runtime components decide whether a
    constraint can be checked by geometry, cached semantic edges, VLM, or
    dataset metadata.
    """

    type: str
    subject: str = ""
    relation: str = ""
    object: str = ""
    value: Any = None
    hardness: str = "soft"
    verifier: str = "planner"
    source: str = ""
    object_concept: ConceptQuery | None = None
    subject_concept: ConceptQuery | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchPriors:
    """Non-terminal hints used for exploration ranking, never for stopping."""

    room_hints: list[str] = field(default_factory=list)
    support_objects: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPolicy:
    """How subgoals should be evaluated by the navigator."""

    mode: str = "any_target_success"
    ordered: bool = False
    exhaustive: bool = False
    active_target_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InstructionPlan:
    """Prompt-first, planner-agnostic instruction contract.

    This is the canonical contract. StriveInstructionSpec below is kept as a
    compatibility view for the existing STRIVE benchmark/mapper integration.
    """

    raw_instruction: str
    dataset_target: str = ""
    task_type: str = "object_goal"
    eval_mode: str = "any_target_success"
    targets: list[TargetQuery] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    search_priors: SearchPriors = field(default_factory=SearchPriors)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    concept_queries: list[ConceptQuery] = field(default_factory=list)
    valid: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def terminal_targets(self) -> list[TargetQuery]:
        return [target for target in self.targets if target.terminal]

    @property
    def anchor_targets(self) -> list[TargetQuery]:
        return [
            target for target in self.targets
            if not target.terminal and target.role in ("anchor", "support", "secondary")
        ]

    @property
    def active_terminal_target(self) -> TargetQuery | None:
        terminals = self.terminal_targets
        if not terminals:
            return None
        idx = max(0, min(self.execution.active_target_index, len(terminals) - 1))
        return terminals[idx]

    @property
    def target_match_terms(self) -> list[str]:
        terms = []
        for target in self.terminal_targets:
            terms.extend(target.match_terms)
        return _dedupe_text(terms)

    @property
    def target_detector_prompts(self) -> list[str]:
        terms = []
        for target in self.terminal_targets:
            terms.extend(target.detector_terms or [target.name])
        return _dedupe_text(terms)

    def to_legacy_spec(self) -> "StriveInstructionSpec":
        active = self.active_terminal_target
        canonical = active.name if active is not None else self.dataset_target
        legacy_targets = [active] if self.execution.ordered and active is not None else self.terminal_targets
        detector_prompts = []
        match_terms = []
        for target in legacy_targets:
            detector_prompts.extend(target.detector_terms or [target.name])
            match_terms.extend(target.match_terms)
        aliases = []
        attrs: dict[str, Any] = {}
        if active is not None:
            aliases.extend(active.aliases)
            attrs.update(active.attributes)
        return StriveInstructionSpec(
            raw_instruction=self.raw_instruction,
            dataset_target=self.dataset_target,
            canonical_target=canonical,
            target_detector_prompts=_dedupe_text(detector_prompts),
            target_aliases=_dedupe_text(aliases + match_terms),
            room_hints=list(self.search_priors.room_hints),
            support_objects=list(self.search_priors.support_objects),
            affordances=list(self.search_priors.affordances),
            attributes=attrs,
            task_mode=self.task_type,
            valid=self.valid,
            diagnostics={**self.diagnostics, "plan": self.as_dict()},
        )


@dataclass
class StriveInstructionSpec:
    """Normalized instruction contract consumed by STRIVE.

    The adapter is intentionally planner-agnostic: it only separates executable
    target terms from room/search context.  STRIVE remains responsible for
    Room/Viewpoint/Object navigation after this contract is produced.
    """

    raw_instruction: str
    dataset_target: str = ""
    canonical_target: str = ""
    target_detector_prompts: list[str] = field(default_factory=list)
    target_aliases: list[str] = field(default_factory=list)
    room_hints: list[str] = field(default_factory=list)
    support_objects: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    task_mode: str = "object_goal"
    valid: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def target_match_terms(self) -> list[str]:
        """Terms that are allowed to satisfy the goal.

        Support objects are deliberately excluded.  For example, couch may help
        STRIVE choose a living room for "watch a movie", but it must never be a
        terminal target for that task.
        """

        terms = [self.canonical_target]
        terms.extend(self.target_detector_prompts)
        terms.extend(self.target_aliases)
        return _dedupe_text(terms)
