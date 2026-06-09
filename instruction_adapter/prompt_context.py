from __future__ import annotations

import json

from .contracts import InstructionPlan, StriveInstructionSpec


def render_instruction_context(spec: StriveInstructionSpec | InstructionPlan) -> str:
    """Render structured task facts for STRIVE's existing VLM prompts.

    The text is intentionally compact and factual.  It does not tell the VLM
    which room/viewpoint to choose; it only exposes target and context fields
    that STRIVE's Room/Viewpoint/Object representation can reason over.
    """

    if isinstance(spec, InstructionPlan):
        payload = {
            "raw_instruction": spec.raw_instruction,
            "task_type": spec.task_type,
            "eval_mode": spec.eval_mode,
            "terminal_targets": [target.as_dict() for target in spec.terminal_targets],
            "target_detector_prompts": list(spec.target_detector_prompts),
            "target_match_terms": list(spec.target_match_terms),
            "constraints": [constraint.as_dict() for constraint in spec.constraints],
            "search_priors": spec.search_priors.as_dict(),
            "execution": spec.execution.as_dict(),
        }
    else:
        payload = {
            "raw_instruction": spec.raw_instruction,
            "task_type": spec.task_mode,
            "eval_mode": spec.diagnostics.get("plan", {}).get("eval_mode", "any_target_success"),
            "terminal_targets": [{
                "name": spec.canonical_target,
                "detector_terms": list(spec.target_detector_prompts),
                "aliases": list(spec.target_aliases),
                "attributes": dict(spec.attributes),
                "terminal": True,
            }],
            "target_detector_prompts": list(spec.target_detector_prompts),
            "target_match_terms": list(spec.target_match_terms),
            "constraints": spec.diagnostics.get("plan", {}).get("constraints", []),
            "search_priors": {
                "room_hints": list(spec.room_hints),
                "support_objects": list(spec.support_objects),
                "affordances": list(spec.affordances),
            },
            "execution": spec.diagnostics.get("plan", {}).get("execution", {}),
        }
    return (
        "Structured instruction plan for STRIVE navigation.\n"
        "Only terminal_targets and target_match_terms may satisfy the goal. "
        "search_priors are exploration hints only. Constraints declare what "
        "must be checked by planner, geometry, metadata, or VLM at runtime.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
