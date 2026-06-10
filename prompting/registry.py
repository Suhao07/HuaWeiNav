from __future__ import annotations

from dataclasses import dataclass


PROMPT_TEMPLATE_VERSION = "2026-06-10"


@dataclass(frozen=True)
class PromptSpec:
    """Stable prompt identity for logs and future prompt A/B tests."""

    prompt_id: str
    trace_label: str
    schema_name: str
    version: str = PROMPT_TEMPLATE_VERSION


INSTRUCTION_PARSE = PromptSpec("instruction.parse.v1", "instruction_parser", "ParsedInstruction")
CONCEPT_GROUNDING = PromptSpec("concept.grounding.v1", "concept_grounding", "GroundingResult")
EXECUTION_STRATEGY = PromptSpec("execution.strategy.v1", "execution_strategy", "ExecutionStrategyResult")
CONCEPT_MATCH_SINGLE = PromptSpec("concept.match.single.v1", "concept_match_single", "ParsedConceptMatch")
CONCEPT_MATCH_BATCH = PromptSpec("concept.match.batch.v1", "concept_match_batch", "ParsedBatchConceptMatch")
RELATION_VERIFY = PromptSpec("relation.verify.v1", "relation_verifier", "ParsedRelationResult")
FINAL_VERIFY = PromptSpec("final.verify.v2", "final_instruction_verifier", "ParsedVerification")
BBOX_OBJECT_LABEL = PromptSpec("bbox.object_label.v1", "bbox_object_in_box", "BBoxObjectLabelResponse")
TAG_REFINE = PromptSpec("tag.refine.v1", "refine_tag_with_target", "TagRefineResponse")
TAG_REFINE_OBJECT_LIST = PromptSpec(
    "tag.refine_object_list.v1",
    "refine_tag_with_target_obj_list",
    "TagRefineWithObjectListResponse",
)
SIMILAR_OBJECTS = PromptSpec("similar_objects.v1", "similar_objects", "SimilarObjectsResponse")
CHECK_AGAIN_BBOX = PromptSpec("check_again.bbox.v1", "check_again_object_in_bbox", "CheckAgainBBoxResponse")

