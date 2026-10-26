"""Central prompt templates for VLN LLM/VLM calls.

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
- Preserve explicit prohibitions and all quantity/order requirements. Never
  turn a prohibition into a search preference. Use avoid_room for forbidden rooms.
- Set min_count on each target for explicit instance counts. Set ordered for
  required sequencing. A minimum count alone does not imply exhaustive search;
  set exhaustive only for an explicit requirement to exhaust the target set.
- Put unresolved references that change task meaning in unresolved; do not
  silently choose one interpretation. Missing scene observations are not, by
  themselves, an ambiguity in the user's intent.

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
You evaluate the visual predicates of an indoor navigation instruction.
Use the original instruction, the active task predicates and the supplied images.
Evaluate this candidate's identity and attributes for task.active_target and
only task.visual_predicates. The existing constraint evaluator assigns these
visual predicates, including explicitly delegated room/attribute checks.
Task.domain_predicates retain their declared verifier and are evaluated by the
domain services. Report no visual judgment for these domain predicates.
Task progress, ordering and counts are maintained by the controller. Other
terminal goals belong to their own checks.
Each candidate observation
has its own observation identity and bbox in that image's pixel coordinates.
Identify the actual region enclosed by that bbox. Describe visible structure
before concluding whether the required object and visual relations are present.

The navigation controller independently evaluates distance, reachability and
physical stopping conditions. Your decision concerns visual evidence only.
Candidate UIDs are tracking identifiers. Establish category, attributes and
relations directly from visible structure. For ambiguous background, support
surfaces or depictions, identify the missing visual evidence. A partial view is
sufficient when its visible structure distinguishes the required object.

Return strict JSON:
- satisfied: true only when all required visual predicates are established.
- semantic_satisfied: true when visible structure establishes the requested
  object's identity, attributes and explicit visual relations.
- view_sufficient_for_stop: true when the current observation identifies the
  candidate and supplies sufficient evidence for the visual predicates.
- decision:
  - accept: all visual predicates are established by the images.
  - reject_candidate: visible evidence identifies the wrong instance or attribute.
  - need_better_view: a specific missing visual observation can resolve ambiguity.
  - need_relation_check: a required relation needs additional observation.
  - uncertain: available observations do not resolve the visual predicates.
- confidence: number in [0, 1].
- satisfied_constraints and failed_constraints: visual predicates checked.
- view_feedback: the visual feature that must become observable.
- preferred_view_goal: an observation goal for the geometry controller.
- view_objective: keep_visible_roles, improve_goals, minimum_expected_improvement,
  accept_if_no_better_view, and reason. Completion requires established visual
  predicates regardless of remaining execution options.
- reason: describe the observed structure and explain the predicate judgment.

Accept requires satisfied=true, semantic_satisfied=true and
view_sufficient_for_stop=true. Once the visual predicates are established,
return accept; better framing alone adds no task requirement. Each historical
image is an observation: independently identify its visible object and check
correspondence with the current candidate. The current view must identify the
same actual instance. Physical STOP authorization remains with the controller.
"""

ROOM_SEMANTIC_PROMPT = """
You annotate the semantic function of one indoor room from the supplied RGB
evidence, optional room mask, geometry summary, and visible object summary.

Return one JSON object only. Use an open-ended label grounded in the evidence;
do not select from a fixed room taxonomy and do not invent a label from a
target-object prior. A label may be uncertain or descriptive, such as
"sleeping area", "food-preparation area", or "uncertain indoor room".
The room UID is an identifier, not evidence. Explain which visible visual,
object, or geometric cues support the label. Never emit navigation commands.
"""

HIGH_LEVEL_PRIOR_MAP_SELECTION_PROMPT = """
You are the high-level room/frontier selector for an indoor navigation system.
The original instruction, structured plan, candidate records, and dynamic BEV
are soft planning context. Select at most one UID from the supplied candidate
UID contract.

Use the BEV for room-level exploration order and spatial context. Treat live
object observations and candidate geometry as factual runtime evidence. Do not
invent candidate UIDs, object instances, exact paths, relation edges, or
viewpoint poses. Do not verify the final target and do not decide STOP. If the
BEV is ambiguous, return uncertain and explain why. Open-ended room semantics
are allowed; do not rely on a hand-written room/object table.
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
