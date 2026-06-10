"""Central prompt templates for STRIVE LLM/VLM calls.

This module owns prompt text only. Runtime state, image packaging, caching, and
model invocation stay in the caller modules so prompt changes remain isolated.
"""

from __future__ import annotations

from llm_utils.nav_prompt_room import OBJECT_PROMPT, RELOCATE_PROMPT, ROOM_PROMPT


#指令解析
INSTRUCTION_PARSE_PROMPT = """
You are a semantic compiler for indoor object navigation instructions.

Return only structured JSON matching the schema. Do not solve navigation and do
not invent scene-specific facts. Extract what the user asks for:
- targets: object concepts mentioned or implied by the instruction.
- terminal=true only for objects that may satisfy the final goal.
- anchors/support objects must be terminal=false.
- constraints: room, spatial, sequence, count, area, attribute, co_occurrence.
- Use hard constraints for explicit requirements in the instruction.
- Use soft constraints only for hints explicitly present in the instruction.
- Do not encode common-sense priors such as "TV is in living room" unless the
  room or context is explicitly stated by the instruction.

Examples:
Instruction: "Find a cup on the table in the kitchen."
targets: cup terminal true; table role anchor terminal false.
constraints: room cup in kitchen hard; spatial cup on table hard verifier vlm.

Instruction: "I need somewhere to sit."
targets: object or affordance concept "seat" terminal true, with affordance sitting.
Do not choose chair/sofa by hard-coded prior; detector grounding will map it.

Instruction: "First find the bed, then locate the towel."
targets: bed terminal true; towel terminal true.
constraints: sequence bed before towel hard.
"""

#概念映射
CONCEPT_GROUNDING_PROMPT = """
You map an instruction target concept to detector vocabulary for open-vocabulary
object navigation.

Return JSON only. Choose detector_terms from the provided available classes when
possible. Include aliases only if they are plausible names for the same target
concept, not nearby/support objects. Do not include room names or support
objects as detector terms.
Also provide a concise semantic description and negative_terms that distinguish
this concept from related but different concepts.
"""

#执行策略
EXECUTION_STRATEGY_PROMPT = """
You choose an execution strategy for an indoor navigation instruction.

Return JSON only. Use anchor_first only when the instruction contains a
non-terminal reference/anchor object that can guide search for a harder terminal
target. The anchor itself must not satisfy the final goal.
Do not use object-name rules; reason from the structured plan.
"""

#概念匹配
CONCEPT_MATCH_PROMPT = """
You decide whether a mapped object instance satisfies an instruction concept.

Use the concept role carefully:
- terminal concepts may satisfy the final goal.
- anchor/support concepts are only reference objects for search or relation
  verification and must never be treated as final goal success.

Do not rely on hard-coded synonym tables. Judge whether the observed object can
play the requested role for this specific instruction. Return strict JSON.
"""

#关系验证（几何和语义关系）
RELATION_VERIFIER_PROMPT = """
You verify one spatial/semantic relation for an indoor navigation robot.

Use only the provided object records, geometry hints, and images. If the
relation is not visible or cannot be inferred from geometry, return verified
false or need_better_view true. Do not use generic common-sense priors.

Return strict JSON:
- verified: true only when the requested relation clearly holds.
- confidence: number in [0, 1].
- need_better_view: true if the pair is plausible but visual evidence is weak.
- reason: concise visual/geometric explanation.
"""

# LVLM最终决策（是否接受当前候选对象作为停靠目标）
FINAL_VERIFIER_PROMPT = """
You are the final instruction-satisfaction verifier for an indoor navigation robot.

Decide whether stopping at the current candidate object satisfies the original
natural-language instruction. Use the visual evidence and factual geometry only.
Do not invent room/object facts that are not visible or provided. If a required
condition cannot be determined from the evidence, do not accept.

Return strict JSON:
- satisfied: true only if the robot may stop for the original instruction from
  the current evidence.
- semantic_satisfied: true if the candidate and explicit instruction constraints
  are satisfied, even if the current camera view is not yet sufficient to stop.
- view_sufficient_for_stop: true if the current view is good enough for final
  stopping evidence. Consider whether the target/relation are clearly visible,
  the target is not clipped or too close to image borders, and the evidence is
  not too weak. Use the provided geometry facts as factual cues, not as rigid
  thresholds. A view can be semantically correct but still insufficient for
  stopping if a clearer, more centered, less clipped, or closer view is needed.
- decision:
  - accept: all explicit requirements are satisfied.
  - reject_candidate: the candidate is clearly the wrong instance or violates a hard requirement.
  - need_better_view: the candidate may be correct but the evidence is too weak.
  - need_relation_check: a spatial/semantic relation must be checked with additional evidence.
  - uncertain: evidence is insufficient and no specific next check is obvious.
- confidence: number in [0, 1].
- satisfied_constraints: short strings.
- failed_constraints: short strings.
- view_feedback: concise feedback for the navigation controller if a better
  view is needed.
- preferred_view_goal: a short natural-language view goal, e.g. keep both the
  candidate and relation anchor visible while making the candidate more central.
- view_objective: an optional JSON object for the geometry controller when
  decision=need_better_view. It may include keep_visible_roles, improve_goals,
  minimum_expected_improvement, accept_if_no_better_view, and reason.
- reason: concise explanation grounded in the evidence.

Important:
- accept requires satisfied=true, semantic_satisfied=true, and
  view_sufficient_for_stop=true.
- If the candidate appears to satisfy the instruction but the target is near
  the image border, very small, clipped, mostly low/outside the useful camera
  area, or relation evidence is not jointly visible enough, return
  decision=need_better_view with semantic_satisfied=true.
- Do not reject a candidate only because the view is poor; use
  need_better_view unless the candidate clearly violates the instruction.
- If view_control history is provided, use it. If there are remaining feasible
  proposals and previous attempts did not substantially improve the evidence,
  prefer need_better_view over accept. If no better view remains, decide whether
  limited-view acceptance is justified by the instruction and evidence.
"""

# 目标识别（基于bbox的单目标识别）
BBOX_OBJECT_LABEL_PROMPT = """
I will provide you an image with one bounding box drawn on it and the cropped
image inside the bounding box. For this bounding box, reason step-by-step and
consider surrounding context to determine what object is inside this bounding
box.

Details:
- The image is input as a base64 string. The bounding box is visually drawn on
  the image.

Your goal:
- Choose the most appropriate label from the following predefined object list
  for the object inside the bounding box.
- If you are unsure, respond with "unknown".
- Output a JSON object without markdown.

Pre-defined object list: {detect_objects}
"""

# 标签细化（基于候选对象列表的标签规范化）
TAG_REFINE_PROMPT = (
    "Here is a list of words and a target word. For each word in the list, if "
    "it has the same meaning as the target, please replace it with the target. "
    "Otherwise, keep it unchanged."
)


TAG_REFINE_WITH_OBJECT_LIST_PROMPT = """
Here is a predefined object list: {object_list}.
You will be given one tag. You need to find the object in the list that has the
closest meaning to this tag. If you find the object, please output the object.
Otherwise, output "unknown". If you are not sure, please output "unknown".
"""


SIMILAR_OBJECTS_PROMPT = """
Here is a predefined object list: {object_list}.

You are given a target object. Your task is to identify all objects in the
object list that have the same meaning as the target object.

Your response should include:
- `object_list`: a list of objects that have the same meaning with target
  object. Follow the python list format, e.g., ['object1', 'object2',
  'object3'].
"""

#bbox复核（给定bbox和标签，判断是否满足停靠要求）
CHECK_AGAIN_BBOX_PROMPT = """
I will give you an image with a bbox drawn on it and an object class label. Your
task is to determine whether the object within the bbox is the given object
class.

The image is input as a base64 string. Please notice that the bbox may only
cover part of the object. You should use common-sense reasoning to determine
whether the main object in the bbox is the given class.

Instructions:
1. Carefully examine the RGB image and the region specified by the bbox.
2. Use visual cues and common-sense reasoning to assess whether the object
   matches the given class.
3. Consider the surrounding context of the image and the object class label.
4. Make your decision through step-by-step observation and reasoning.

Your response should include:
- 'steps': the process of chain of thought reasoning.
- `flag`: a boolean value. If the object in the bbox is the given class, output
  True. If the object is not the given class, output False.
"""
