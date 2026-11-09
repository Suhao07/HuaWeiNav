"""Central prompt templates for VLN LLM/VLM calls.

This module owns prompt text only. Runtime state, image packaging, caching, and
model invocation stay in the caller modules so prompt changes remain isolated.
"""

from __future__ import annotations

from pathlib import Path

from llm_utils.nav_prompt_room import OBJECT_PROMPT, RELOCATE_PROMPT, ROOM_PROMPT


INSTRUCTION_UNDERSTANDING_SKILL = (Path(__file__).resolve().parents[1] / "skills" / "understand_instruction.md").read_text(encoding="utf-8")

#指令解析
INSTRUCTION_PARSE_PROMPT = """
Interpret the indoor navigation instruction using the instruction-understanding
skill below. Return only structured JSON matching the supplied schema. There are
no scene observations at this stage; do not invent observed objects or states.

Represent navigation destinations and actual reference anchors in targets.
For an explicit object, use its name and equivalent object-name aliases, with
demand=null. A descriptive clause is not an alias; leave aliases empty rather
than inserting the object's attributes or relationships as another name.
For an implied resource, create a terminal=true destination whose name describes
the resource that enables the activity. Attach demand to that resource target:
activity, manipulated_objects, required_resource and destination_hypotheses.
The material being acted on belongs in manipulated_objects, not as the resource
or an anchor target. Add a material as a navigation target only if the instruction
separately asks to navigate to it. Facility alternatives are hypotheses, not
identity aliases. Extract entity-to-entity predicates into constraints before
writing unary attributes. A noun phrase can describe several entities: a second
entity's relative placement is a directed relation, not a property of the first
entity. Resolve the subject and reference of each placement from the wording.
Attributes contain only unary properties; use {} when none are stated. Represent
each independent object relation as a separate constraint; do not concatenate
relation phrases or hide them inside attributes.

Set terminal=false for anchors/support objects. Bind constraint subjects and
objects consistently to the target names; references to the same object must
not become independent identities. Preserve explicit constraints as hard, and
explicit hints as soft. Use room, spatial, sequence, count, area, attribute,
co_occurrence or affordance as appropriate; use avoid_room for forbidden rooms.
Set min_count and ordered from explicit counts/sequences; set exhaustive only
when the user requests exhausting the target set. Separate interpretation from
observation: unresolved contains only genuine ambiguity in the user's meaning
that requires clarification. Unknown location, existence, availability, visibility
or satisfaction of an object's requested properties are observation_requirements,
not reasons to reject an otherwise interpretable navigation request. A target
need not already be found or known to exist before navigation can begin. Preserve
its requested properties and relations as obligations to verify, not facts already
established. Do not invent a missing referent or discard contradictory instructions.

Representation examples, not facts about the current instruction:
- "Find the metal trolley with wheels and a carton underneath": trolley is the
  destination with material=metal and component=wheels; carton is a nonterminal
  anchor; a separate hard spatial constraint is (carton, below, trolley).
  The carton's placement cannot be stored as a trolley attribute or alias.
- "I need to dry this wet scarf": a terminal drying resource has demand activity
  drying, manipulated_objects=[scarf], required_resource=a place suitable for
  drying fabric. A clothesline is a possible search hypothesis, not a required
  identity; scarf is not the destination. Infer a capability, not a fixed facility.
Resolve each directed relation in the wording without unnecessary inversion.

""" + INSTRUCTION_UNDERSTANDING_SKILL

#概念映射
CONCEPT_GROUNDING_PROMPT = """
You map an instruction target concept to detector vocabulary for open-vocabulary
object navigation.

Return JSON only. Choose detector_terms from the provided available classes when
possible. These are retrieval hints: a broader class may help find candidates but
does not establish the requested identity. Return only detector_terms and reason;
do not change the supplied concept, aliases, description or exclusions. Apply
this retrieval-only boundary to terminal targets and non-terminal anchors.
Do not include room names or support objects as detector terms.
For a demand concept, ground its required_resource and activity to possible
facility classes, not to its manipulated_objects. Alternative realizations
remain detector hints, never aliases for the functional requirement.
Use up to four distinct phrases, each at most 64 characters, and one short
reason (up to 160 characters). An empty list is valid when no relevant term exists.
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

Separate semantic identity from retrieval compatibility for both targets and
anchors. Check the concept name against the original instruction. Aliases are
proposed equivalences, not proof: reject a proposed alias if it changes or
broadens the requested identity. Detector_terms are retrieval hints and may be
broader than that identity. A matching detector term or related category is not
proof of identity. For a demand concept, its name
is a functional resource, not a fixed facility category. Use demand.activity,
demand.manipulated_objects and demand.required_resource to consider observed
facilities, including alternatives absent from the initial detector terms.
A plausible facility is eligible for inspection; its specific functional
suitability, requested attributes and current state remain unverified until
instance-level visual checks. Do not require the manipulated objects to be
present unless the original instruction asks to find them.

Return matches_concept=true only when the supplied evidence establishes the
concept identity, false for an incompatible object, and null when identity is
unknown. With only a broad label, use null rather than guessing true or false.
Set candidate_eligible=true for a confirmed match or an unresolved but
semantically compatible candidate worth inspecting; set it false for unrelated
labels or when there is no evidence of compatibility. Do not turn every unknown
label into a candidate. Confidence describes the judgment, not an identity gate.
Set terminal_eligible=true only for an established identity of a terminal
concept. It does not authorize STOP: instance attributes, relations and final
view requirements must still be verified. Missing relation anchors alone must
not negate the candidate's concept identity or retrieval compatibility.
Do not rely on hard-coded synonym tables. Return strict JSON with
matches_concept, candidate_eligible, confidence, terminal_eligible and reason.
"""

#关系验证（几何和语义关系）
RELATION_VERIFIER_PROMPT = """
You verify one spatial/semantic relation for an indoor navigation robot.

Use only the provided object records, geometry hints, and images. Each frame's
object_boxes bind the two instance IDs to pixel regions in that frame only.
Role-labeled crops show exactly those regions and carry their source frame, box
and instance ID. Use the crops to assess identity and the full source frames to
assess spatial relationships; crop placement does not define relative position.
requested_concept is a task requirement, not proof that the bound instance has
that identity. detector_label is a fallible observation, not a replacement for
the requirement. A record with demand describes a functional resource: assess
visible support for its activity on its manipulated_objects using required_resource,
not whether the detector names a facility identical to the functional label.
Do not require the manipulated materials to be visible. A plausible facility label
alone does not establish functional suitability; missing structural evidence is
unknown. Records without demand retain their requested object identity.
Never substitute an unbound object elsewhere in the image.
Do not use generic common-sense priors.

Return strict JSON with three separate judgments and confidence in [0, 1]:
- subject_identity: does the subject region satisfy its requested concept and demand, if present?
- reference_identity: does the reference region satisfy its requested concept and demand, if present?
- relation: does the exact requested directed relation hold between those regions?
Each judgment must contain verdict (true, false, or unknown) and a concise reason
citing visible evidence for that role. Use false only for a clear contradiction;
missing evidence is unknown. If either identity is not established, relation must
be unknown. Proximity to a different visible object cannot establish this relation.
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

Task.active_target.demand is the immutable requested activity, manipulated_objects
and required_resource. For every hard affordance in task.visual_predicates, judge
whether this actual candidate supports that demand, not whether its category is
a typical destination. Search hypotheses and category typicality are never
instance proof. Visible functional structure is sufficient for a design
affordance; do not require performing the activity. Hidden state such as being
operational or softness cannot be asserted solely from category or appearance
when the required property is not visually established. Report unknown rather
than infer an unobserved property. Visible incompatible structure supports false;
a missing or ambiguous view supports unknown, not false.

When the response schema includes affordance_judgments, return exactly every
required predicate ID as a key, each with:
- verdict: "true", "false", or "unknown".
- reason: a concise visual reason (1-240 characters), identifying observed
  functional structure or the specific missing evidence.
- supporting_observation_ids: observation_id strings from actually supplied,
  candidate-bound images supporting this judgment; use [] for no such evidence.
True and false require supporting observations of this candidate. Goal reference
images, unrelated relation views, labels and unsent observations are not support.
Historical candidate views can support a predicate only with established instance
correspondence; a current candidate-bound image is always required. If any hard
affordance is false, reject_candidate; if none is false but any is unknown,
request need_better_view or return uncertain. Only all true permits semantic
acceptance. When this field is absent from the schema, do not add it.

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
same actual instance. An overlapping_instance_observation shows another mapped
instance carrying the stated fraction of the candidate's 3D surface; it may be
the same physical object seen from another viewpoint. Identify its object
independently and weigh it as evidence of the candidate's identity.
Physical STOP authorization remains with the controller.
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

Return decision="select" with the chosen UID and its candidate type (room or
frontier). When evidence is insufficient, return decision="uncertain" with
empty selected_uid and selected_type. These are the two supported decisions.

Region observed_from_region summaries describe predicted labels seen by cameras
in that region, possibly across a doorway. Counts are distinct frames, not
independent objects or calibrated confidence. Preserve conflicting labels as
uncertain evidence. Use these observations to reason about room function; an
unknown region label alone does not imply equal target relevance.

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
