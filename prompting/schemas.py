from __future__ import annotations

from typing import Any

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


class ReasoningStep(BaseModel):
    explanation: str = ""
    output: str = ""


class ParsedTarget(BaseModel):
    name: str = ""
    role: str = Field(default="primary", description="primary, secondary, anchor, or support")
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    min_count: int = 1
    terminal: bool = True


class ParsedConstraint(BaseModel):
    type: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    value: Any = None
    hardness: str = "soft"
    verifier: str = "planner"


class ParsedInstruction(BaseModel):
    task_type: str = "object_goal"
    eval_mode: str = "any_target_success"
    targets: list[ParsedTarget] = Field(default_factory=list)
    constraints: list[ParsedConstraint] = Field(default_factory=list)
    room_hints: list[str] = Field(default_factory=list)
    support_objects: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)
    requires_runtime_relation: bool = False


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


class ParsedConceptMatch(BaseModel):
    matches_concept: bool = False
    confidence: float = 0.0
    terminal_eligible: bool = False
    reason: str = ""


class ParsedBatchConceptItem(BaseModel):
    uid: str = ""
    matches_concept: bool = False
    confidence: float = 0.0
    terminal_eligible: bool = False
    reason: str = ""


class ParsedBatchConceptMatch(BaseModel):
    matches: list[ParsedBatchConceptItem] = Field(default_factory=list)


class ParsedRelationResult(BaseModel):
    verified: bool = False
    confidence: float = 0.0
    need_better_view: bool = False
    reason: str = ""


class ParsedVerification(BaseModel):
    satisfied: bool = False
    semantic_satisfied: bool = False
    view_sufficient_for_stop: bool = True
    decision: str = "uncertain"
    confidence: float = 0.0
    satisfied_constraints: list[str] = Field(default_factory=list)
    failed_constraints: list[str] = Field(default_factory=list)
    view_feedback: str = ""
    preferred_view_goal: str = ""
    view_objective: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class BBoxObjectLabelResponse(BaseModel):
    steps: list[ReasoningStep] = Field(default_factory=list)
    res: str = "unknown"


class TagRefineResponse(BaseModel):
    res: list[str] = Field(default_factory=list)


class TagRefineWithObjectListResponse(BaseModel):
    steps: list[ReasoningStep] = Field(default_factory=list)
    output: str = "unknown"


class SimilarObjectsResponse(BaseModel):
    steps: list[ReasoningStep] = Field(default_factory=list)
    object_list: list[str] = Field(default_factory=list)


class CheckAgainBBoxResponse(BaseModel):
    steps: list[ReasoningStep] = Field(default_factory=list)
    flag: bool = False

