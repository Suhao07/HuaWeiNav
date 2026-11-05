"""Structured response schemas shared by prompt callers.

The fallback classes keep offline smoke tests usable when pydantic is missing
inside a minimal container.
"""

from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, Field, constr
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

    def constr(**_kwargs):
        return str


class ParsedInstanceCorrespondence(BaseModel):
    """Cross-map identity judgment, separate from category/task acceptance."""

    outcome: Literal["unique", "ambiguous", "fragments", "no_match", "insufficient"] = "insufficient"
    prior_uid: str = ""
    supporting_observation_ids: list[str] = Field(default_factory=list)
    conflicting_observation_ids: list[str] = Field(default_factory=list)
    evidence_summary: str = ""


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
    unresolved: list[str] = Field(default_factory=list)
    ordered: bool = False
    exhaustive: bool = False


class GroundingResult(BaseModel):
    """A small detector vocabulary proposal, with bounded phrases and explanations."""
    detector_terms: list[constr(max_length=64)] = Field(default_factory=list, max_items=4)
    aliases: list[constr(max_length=64)] = Field(default_factory=list, max_items=4)
    description: str = Field(default="", max_length=240)
    negative_terms: list[constr(max_length=64)] = Field(default_factory=list, max_items=4)
    reason: str = Field(default="", max_length=160)


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


class CandidateConceptJudgment(BaseModel):
    """Required label-compatibility judgment; absence is distinct from a negative match."""
    matches_concept: bool
    confidence: float = Field(ge=0, le=1)
    terminal_eligible: bool
    reason: str = Field(max_length=160)

    class Config:
        extra = 'forbid'


def concept_batch_response_model(object_uids):
    """Encode dom(response)=candidate UIDs as required JSON object properties.

    每个 UID 对应一个判断，负例也必须返回。将完整性写进生成与解析协议，
    避免可变长列表允许空结果、漏项或重复身份。
    """
    from pydantic import create_model
    return create_model('CandidateConceptJudgments',
                        **{uid: (CandidateConceptJudgment, ...) for uid in object_uids})


class ParsedRelationResult(BaseModel):
    verified: bool = False
    confidence: float = 0.0
    need_better_view: bool = False
    reason: str = ""


class ParsedViewObjective(BaseModel):
    """Observation request for unresolved predicates, with an explicit schema."""
    keep_visible_roles: list[str] = Field(default_factory=list)
    improve_goals: list[str] = Field(default_factory=list)
    minimum_expected_improvement: str = ""
    accept_if_no_better_view: bool = False
    reason: str = ""


class ParsedVerification(BaseModel):
    """Visual judgment with closed, typed nested objects for strict generation."""
    satisfied: bool = False
    semantic_satisfied: bool = False
    view_sufficient_for_stop: bool = True
    decision: str = "uncertain"
    confidence: float = 0.0
    satisfied_constraints: list[str] = Field(default_factory=list)
    failed_constraints: list[str] = Field(default_factory=list)
    view_feedback: str = ""
    preferred_view_goal: str = ""
    view_objective: ParsedViewObjective = Field(default_factory=ParsedViewObjective)
    reason: str = ""


class ParsedRoomSemantic(BaseModel):
    """Open-ended room semantic annotation returned from room evidence."""

    label: str = "unknown"
    description: str = ""
    confidence: float = 0.0
    alternatives: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    uncertainty: str = ""


class ParsedHighLevelSelection(BaseModel):
    """Room/frontier selection returned from a dynamic prior-map BEV."""

    selected_uid: str = ""
    selected_type: Literal["", "room", "frontier"] = ""
    # 与规划器消费的动作域一致，未知动词在模型协议边界显式报错。
    decision: Literal["select", "uncertain"] = "uncertain"
    confidence: float = 0.0
    reason: str = ""
    alternatives: list[str] = Field(default_factory=list)
    rejected_candidates: list[str] = Field(default_factory=list)
    uncertainty: str = ""


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
