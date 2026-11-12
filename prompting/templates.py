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

Begin with instruction_reading: a concise account of the complete situation,
its destination resource and each entity's role, using only the original wording.
Then express that meaning as targets and constraints. Resolve the use before
choosing a destination name; keep the semantic reading distinct from search guesses.
An explicitly requested object is the destination and retains its identity in
name. Its demand describes what that object enables, rather than a support on
which to place it. An implicit destination is named by the required capability.
Possible facilities are search hypotheses, not additional identities.
For obtaining something from a stated source, the source is the destination.
Its demand is to provide the requested output, preserving the output's original
level of specificity. Output properties belong to this provision requirement;
attributes describe the destination itself. Depicted content is an attribute of
its physical display. Category names are identities, not materials.

Declare all physical entities that locate the destination, including references
embedded in its noun phrase. "I need an object by/at/under another object" locates
the requested object even when its intended use will happen later. Represent
that reference as a nonterminal target and bind a hard directed relation between
the two entity IDs. Copying the phrase into demand does not replace this edge.
A desired relocation, such as wanting something to put elsewhere, is intended
use context rather than evidence of the object's existing position.
The speaker's future posture and viewpoint belong in demand, not in a physical
person target or a relation from an object to itself. A described place does not
locate the speaker. An object's own posture belongs in its attributes.

For each destination with a stated need or use, populate demand even when the
object is explicitly named. In activity, copy the complete original sentences
relevant to that destination, including motivation, timing, posture, participants,
intended placement and purpose. For a single destination, copy the whole request.
required_resource states what this named destination must enable or provide in
that situation. Materials acted on go in manipulated_objects; outputs obtained
from a source stay in required_resource. Use demand=null for references and for
destinations with no stated use.
Use equivalent object names in aliases, stated unary properties in attributes,
and separate constraints for independent relationships between declared entities.
Use {} for absent attributes.

Preserve the complete activity, intended users and required capability, not just
an easy-to-detect resource noun. Retain every stated unary property, depicted
content, mounting/state and directed relationship, including clauses explaining
why the resource is needed. Emotion alone is not a physical requirement.
Set terminal=false for anchors/support objects. Every entity used in a relation
must have a target record, even if it is only a reference. Assign distinct opaque
IDs to targets and constraints; bind constraint subjects and objects to target
IDs. Preserve IDs when revising the same entities and requirements, even if names
or list order change. Unbound or ambiguous references are compilation errors,
not optional hints.
References to the same object must not become independent identities.
Preserve explicit constraints as hard, and
explicit hints as soft. Use room, spatial, sequence, count, area, attribute,
co_occurrence or affordance as appropriate; use avoid_room for forbidden rooms.
Set min_count and ordered from explicit counts/sequences; set exhaustive only
when the user requests exhausting the target set. Record genuinely unresolved
meaning in unresolved, not missing scene evidence.

Representation examples, not facts about the current instruction:
- "Find the metal trolley with wheels and a carton underneath": trolley is the
  destination with material=metal and component=wheels; carton is a nonterminal
  anchor; a separate hard spatial constraint is (carton, below, trolley).
  The carton's placement cannot be stored as a trolley attribute or alias.
- "I need to dry this wet scarf": a terminal drying resource has demand activity
  "I need to dry this wet scarf", manipulated_objects=[scarf], required_resource=
  a place suitable for drying fabric. A clothesline is a possible search hypothesis, not a required
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
Each record's attributes and constraints are required properties, including
contents, shape and mounting. A correct category without those properties does
not satisfy identity.
Explicit contents requirements still require visible contents; do not confuse them
with manipulated materials in a functional demand.
Never substitute an unbound object elsewhere in the image.
Do not use generic common-sense priors.

Return strict JSON with three separate judgments and confidence in [0, 1]:
- subject_identity: does the subject region satisfy its requested concept, attributes and demand?
- reference_identity: does the reference region satisfy its requested concept, attributes and demand?
- relation: does the exact requested directed relation hold between those regions?
Each judgment must contain verdict (true, false, or unknown) and a concise reason
citing visible evidence for that role. Use false only for a clear contradiction;
missing evidence is unknown. If either identity is not established, relation must
be unknown. Proximity to a different visible object cannot establish this relation.
"""

INSTANCE_OBSERVER_PROMPT = """
Describe the selected physical instance from the supplied images, separately
for each observation ID. Its source frame and crop show the same bound region.
For each ID, use short factual phrases and avoid repetitive enumeration.
Report visible parts, support or attachment and visible surface content. In
visible_context, separately describe visible surrounding fixtures and boundaries,
and whether the selected instance shares their space or is across a doorway.
Do not transfer surroundings to the instance's parts. A crop without surrounding
evidence has empty visible_context; unclear spatial membership is an uncertainty.
Do not infer a desired task, purpose, hidden parts, operating state or room label.
A caption, depiction, reflection or neighbouring object is not a physical part of
the selected instance. Overlap alone does not prove another region is this object.
Use uncertainties for facts the image cannot establish, not invented negatives.
Describe appearance without inferring manufacturing method or unseen materials.
For a goal_reference image, compare physical instance identity with the candidate,
not merely category similarity. Report same, different or unknown with visual
support; same requires distinguishing evidence from a current candidate observation.
The reference is not a candidate observation and cannot support candidate facts.
Return only the required JSON. Do not judge task suitability or propose actions.
"""


# LVLM最终决策（是否接受当前候选对象作为停靠目标）
FINAL_VERIFIER_PROMPT = """
Evaluate the actual observed instance against the original navigation instruction.
Inspect the supplied current candidate images against the complete intended use.
For the current observation, bound_observations supplies image provenance only.
Establish visible instance boundaries, physical structure and surroundings from
the images. depth_support, when present, measures max-minus-min sample coordinates
in metres along map X/Y/Z (Z is up). These are partial, axis-aligned surface samples,
not full object dimensions, usable area or proof that the mask contains one object.
Use measurements for metric scale; report unknown for unmeasured dimensions.
Historical descriptions are fallible model reports; current images establish the
present facts. Identify any unresolved premise. A category
or nearby furniture alone does not establish capacity for the intended users.
False requires visible evidence incompatible with the exact requirement;
lack of confirmation is unknown. Distinguish the resource, its output and its
contents: each property belongs to the entity described in the original wording.
The original instruction defines the requirements. The compiled task helps locate
them but cannot delete, weaken or replace an activity, intended user, attribute,
depicted content or relationship. Check the original wording even if the compiler
omitted a requirement.
Judge only requirements of the active destination; other terminal goals, ordering
and counts are handled by the controller.

Report observed_category and observed_structure from the current candidate images,
including its distinguishing visible parts, support or mounting and boundaries. This is not the requested resource name or a guess at a
benchmark label. An image of an object is not the physical object; a shared material
or nearby support is not evidence of the demanded function.

Each bound observation carries a candidate UID, observation ID, image hashes and
bboxes. Use the bound image region and its observation ID; paths and identifiers
are provenance, not semantic clues. Distinguish the candidate's identity, parts
and affordance from neighbouring objects. When instance boundaries remain
ambiguous in the image, report unknown.
visible_context describes surroundings, not candidate parts. A room requirement
needs visible environmental cues and evidence the candidate occupies that space;
a single object category or a room visible across a doorway is insufficient.
Historical facts must describe the same instance and cannot replace current
candidate-bound evidence.

When verification_basis is original_instruction, begin with instruction_reading:
state the original destination and use, the roles of source, output, contents and
references, and which entity each property describes. Resolve this meaning before
comparing the candidate. Then derive instruction_judgments from that wording.
For each entry, state the quoted requirement and relevant visual evidence before
its verdict. Summarize and request a view only after the individual judgments.
List every
visual requirement of the active destination, including intended use, stated users,
context and properties of the resource, its output or contents. Each entry quotes
its instruction_quote verbatim and states the requirement with its proper entity.
Cover the complete request, not only phrases confirmed by the image. A stated use
is a requirement even when evidence is unavailable; judge it unknown in that case.
For multiple destinations, active_target.compiled_scope_hint locates the relevant
part of the original wording, including distinctions between same-name targets.
It supplies fallible scoping clues, not additional requirements; resolve each
requirement from the original wording. An unresolved scope is an interpretation
conflict for candidate_identity, not permission to choose a different destination.
Domain predicates retain their separate services and exact bound references;
exclude their relation, count, ordering and geometry judgments from this list.
When verification_basis is fixed_task, all supplied compiled predicates remain
binding. In both modes candidate_identity checks the complete original destination.

Return exactly one predicate_judgments entry for every verification_predicates ID:
- candidate_identity: does this bound instance satisfy the original destination
  identity and any original requirements absent from the compiled checks? Compare
  the full wording, not just task.active_target.name. A compiled omission is not
  permission to ignore a requirement. Domain predicates have separate services;
  do not re-adjudicate verified relations or geometry. Their bound observations
  identify the exact reference instance, not a substitute elsewhere in the image.
- attribute predicates: check the specified property of this instance, not another
  object, a reflection or a depiction.
- affordance predicates: check visible functional structure supporting the entire
  activity on the manipulated materials, including stated users and context.
  Material/category similarity alone is insufficient. Materials being acted on
  need not be present unless the instruction asks to find them. Design affordances
  need not be physically performed, but hidden operational state is unknown.
- other visual predicates: check the exact stated requirement and bound references.

Each entry contains verdict ("true", "false", "unknown"), a concise visual reason
(1-240 characters), and supporting_observation_ids from the supplied
bound_observations. True and false require such support; use [] when none is
available. candidate_identity must cite the current observation, not history alone.
A clear incompatible structure/property is false even if a closer view is possible.
An occluded or absent observation is unknown, not false. Never weaken a requirement
to turn a previously contradictory observation into a match.

Report interpretation_conflicts separately when a compiled predicate assigns a
requirement to the wrong entity or contradicts the original instruction. Each
conflict names its predicate_id, quotes the relevant original wording verbatim in
instruction_quote, and explains the discrepancy in reason. Give that predicate an
unknown visual verdict: a faulty requirement is a language conflict, not evidence
against the candidate. Return [] when the interpretation is consistent. Hidden
state or ambiguous visual evidence remains an ordinary unknown, not an
interpretation conflict.

A conflict suspends candidate judgment until the controller resolves the task
interpretation. Otherwise the controller derives rejection from any false and
semantic acceptance only from all true. You do not output acceptance booleans or a
STOP decision. Distance and reachability are independent geometry checks.
For unknown evidence, give view_feedback and view_objective describing the specific
newly observable feature that could resolve it; do not request another look at an
already disproven property or suggest acceptance because no better view is possible.
Return the supplied schema only, with confidence and a short reason. Confidence is
not proof, and better framing adds no requirement once the evidence is established.
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
