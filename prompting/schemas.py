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


class ParsedDemand(BaseModel):
    activity: str = Field(description="Verbatim original sentences giving this destination's need and situation, including motivation, timing, posture, participants, intended placement and purpose. Copy the whole instruction for a single destination.")
    manipulated_objects: list[str] = Field(default_factory=list,
        description="Inputs the user acts on with the resource. Requested outputs, people, destinations and references are not inputs; use [] when none are stated.")
    required_resource: str = Field(
        description="The named destination's required capability for this use. An explicitly requested object remains the resource; describe what that object enables, not a support on which to put it. A source provides the requested output, as general or specific as the original wording.")
    destination_hypotheses: list[str] = Field(default_factory=list,
        description="Possible realizations of the destination consistent with its stated identity and intended use; search hypotheses, not additional requirements or supporting objects.")


class ParsedTarget(BaseModel):
    id: str = Field(default="", description="Opaque entity ID. Preserve it when revising the same entity; use it in constraint references.")
    demand: ParsedDemand | None = Field(
        description="Required for an explicit or implicit destination's stated need or use. Preserve the full use clause in activity and the needed capability in required_resource. Null for reference anchors or destinations with no stated use.")
    name: str = Field(default="",
        description="The requested object identity, or the functional resource described by demand.required_resource; not the manipulated material.")
    role: str = Field(default="primary", description="primary, secondary, anchor, or support")
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict,
        description="Stated unary properties of this entity, including depicted content. The object's identity is its name, not its material. Properties of a requested output belong in demand.required_resource; relations to other physical entities belong in constraints.")
    min_count: int = 1
    terminal: bool = Field(default=True,
        description="True for a navigation destination, including a demanded resource; false only for reference anchors or support objects.")


class ParsedConstraint(BaseModel):
    id: str = Field(default="", description="Opaque predicate ID. Preserve it when revising the same requirement.")
    type: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    value: Any = None
    hardness: str = "soft"
    verifier: str = "planner"


class ParsedInstruction(BaseModel):
    # 自回归生成先明确原文角色与用途，再选择实体和属性，避免先选类别后补理由。
    instruction_reading: str = Field(default="", max_length=1200,
        description="Brief semantic reading of the complete original request, before selecting targets. State the destination resource, what it enables, which entities are references, inputs, outputs or depicted content, and whether each location describes an existing relation or a future use. Attribute each modifier to its own entity and preserve the original level of specificity.")
    task_type: str = "object_goal"
    eval_mode: str = "any_target_success"
    # 流式生成关系前需要已声明的实体 ID。
    targets: list[ParsedTarget] = Field(default_factory=list)
    constraints: list[ParsedConstraint] = Field(default_factory=list,
        description="Bind every stated entity-to-entity relation to the declared target IDs, including relative placements embedded in noun phrases. Unary target properties do not replace relations.")
    room_hints: list[str] = Field(default_factory=list)
    support_objects: list[str] = Field(default_factory=list)
    affordances: list[str] = Field(default_factory=list)
    requires_runtime_relation: bool = False
    unresolved: list[str] = Field(default_factory=list,
        description="Only alternative meanings of the instruction that cannot be resolved from its wording. An unknown scene location, unseen object or missing observation is a search task, so leaves this list empty.")
    ordered: bool = False
    exhaustive: bool = False


class GroundingResult(BaseModel):
    """A small detector vocabulary proposal, with bounded phrases and explanations."""
    detector_terms: list[constr(max_length=64)] = Field(default_factory=list, max_items=4)
    reason: str = Field(default="", max_length=160)

    class Config:
        extra = "forbid"


class ExecutionStrategyResult(BaseModel):
    mode: str = "any_target_success"
    use_anchor_first: bool = False
    anchor_concept_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class ParsedConceptMatch(BaseModel):
    """身份判断显式允许 null（未知），取证候选资格独立表达。"""
    matches_concept: bool | None = Field(...)
    confidence: float = 0.0
    terminal_eligible: bool = False
    reason: str = ""
    candidate_eligible: bool = False


class CandidateConceptJudgment(BaseModel):
    """身份判断必填；仅语义相容只能允许取证，不能判定成功。"""
    matches_concept: bool | None = Field(...)
    confidence: float = Field(ge=0, le=1)
    terminal_eligible: bool
    reason: str = Field(max_length=160)
    candidate_eligible: bool = False

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


class RelationJudgment(BaseModel):
    verdict: Literal["true", "false", "unknown"]
    reason: str = Field(min_length=1)

    class Config:
        extra = "forbid"


class ParsedRelationResult(BaseModel):
    subject_identity: RelationJudgment
    reference_identity: RelationJudgment
    relation: RelationJudgment
    confidence: float = Field(ge=0, le=1)

    class Config:
        extra = "forbid"


class ParsedViewObjective(BaseModel):
    """Observation request for unresolved predicates, with an explicit schema."""
    keep_visible_roles: list[str] = Field(default_factory=list)
    improve_goals: list[str] = Field(default_factory=list)
    minimum_expected_improvement: str = ""
    accept_if_no_better_view: bool = False
    reason: str = ""


class ObservedInstanceFacts(BaseModel):
    observed_category: str = Field(max_length=80)
    visible_parts: list[str] = Field(max_length=16)
    support_and_attachment: str
    visible_surface_content: str
    visible_context: list[str] = Field(max_length=16)
    uncertainties: list[str] = Field(max_length=16)

    class Config:
        extra = "forbid"


class ReferenceImageComparison(BaseModel):
    verdict: Literal["same", "different", "unknown"]
    reason: str = Field(min_length=1, max_length=240)
    supporting_observation_ids: list[str]

    class Config:
        extra = "forbid"


def instance_observation_response_model(observation_ids, *, compare_reference=False):
    from pydantic import create_model
    observations = create_model(
        "BoundInstanceFacts", __config__={"extra": "forbid"},
        **{uid: (ObservedInstanceFacts, ...) for uid in sorted(observation_ids)})
    fields = {"observations": (observations, ...)}
    if compare_reference:
        fields["reference_comparison"] = (ReferenceImageComparison, ...)
    return create_model("InstanceObservation", __config__={"extra": "forbid"}, **fields)


class ParsedVerification(BaseModel):
    """Observed instance description; the controller derives decisions from evidence."""
    observed_category: str = Field(default="", max_length=80)
    observed_structure: str = Field(default="", max_length=240)
    confidence: float = Field(default=0.0, ge=0, le=1)
    view_feedback: str = ""
    preferred_view_goal: str = ""
    view_objective: ParsedViewObjective = Field(default_factory=ParsedViewObjective)
    reason: str = ""

    class Config:
        extra = "forbid"


class VisualPredicateJudgment(BaseModel):
    """Candidate-bound evidence for an identity, property or functional requirement."""
    reason: str = Field(min_length=1, max_length=240)
    supporting_observation_ids: list[str]
    verdict: Literal["true", "false", "unknown"]

    class Config:
        extra = "forbid"


class InstructionRequirementJudgment(BaseModel):
    instruction_quote: str = Field(min_length=1)
    requirement: str = Field(min_length=1, max_length=240)
    reason: str = Field(min_length=1, max_length=240)
    supporting_observation_ids: list[str]
    verdict: Literal["true", "false", "unknown"]

    class Config:
        extra = "forbid"


def visual_verification_response_model(predicate_ids, *, original_instruction=False):
    """Require evidence for fixed predicates and, when authorized, original wording."""
    if not predicate_ids or any(not isinstance(uid, str) or not uid for uid in predicate_ids) or len(set(predicate_ids)) != len(predicate_ids):
        raise ValueError("Visual predicates require unique nonempty IDs")
    from pydantic import create_model
    judgments = create_model(
        "VisualPredicateJudgments", __config__={"extra": "forbid"},
        **{uid: (VisualPredicateJudgment, ...) for uid in predicate_ids})
    conflict = create_model(
        "InterpretationConflict", __config__={"extra": "forbid"},
        predicate_id=(Literal[tuple(predicate_ids)], ...),
        instruction_quote=(str, Field(min_length=1)),
        reason=(str, Field(min_length=1, max_length=240)))
    fields = {}
    if original_instruction:
        fields["instruction_reading"] = (str, ParsedInstruction.model_fields["instruction_reading"])
    for name in ("observed_category", "observed_structure"):
        field = ParsedVerification.model_fields[name]
        fields[name] = (field.annotation, field)
    if original_instruction:
        fields["instruction_judgments"] = (list[InstructionRequirementJudgment], Field(min_length=1))
    fields["interpretation_conflicts"] = (list[conflict], Field(max_length=len(predicate_ids)))
    fields["predicate_judgments"] = (judgments, ...)
    # 生成顺序是推理依赖：原文含义、可见证据、逐项判断先于汇总和下一视角。
    fields.update({name: (field.annotation, field)
                   for name, field in ParsedVerification.model_fields.items() if name not in fields})
    return create_model("VisualVerification", __config__={"extra": "forbid"}, **fields)


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
