"""A bounded Agent tool loop; backends retain sensor and motion execution."""
from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field, create_model

from harness.tools import NavigationTools
from scene.dependencies import object_dependencies
from llm_utils.cognav_llm_adapter import get_client_and_model
from verification.images import image_input
from prompting.schemas import ParsedInstruction


class ToolArguments(BaseModel):
    """Tool boundaries accept exactly the declared fields."""
    class Config:
        extra = "forbid"


class CandidateArguments(ToolArguments):
    """Observed instance for viewpoint lookup."""
    candidate_uid: str


class VerificationArguments(CandidateArguments):
    """Task-scoped final verification request."""
    target_id: str
    task_revision: int


class MotionArguments(ToolArguments):
    """Ownership and idempotency fields shared by motion requests."""
    task_revision: int
    control_epoch: int
    request_id: str


class ObjectMotionArguments(MotionArguments):
    """Navigate to an observed object's associated viewpoint."""
    candidate_uid: str
    viewpoint_uid: str


class FrontierMotionArguments(MotionArguments):
    """Navigate to a frontier published by the observation backend."""
    frontier_uid: str


# 每个 name 与唯一参数协议绑定；schema 从结构上排除跨工具参数。
_TOOL_ARGUMENTS = {
    "observe": ToolArguments,
    "query_scene": ToolArguments,
    "query_viewpoints": CandidateArguments,
    "submit_motion": Union[ObjectMotionArguments, FrontierMotionArguments],
    "motion_status": ToolArguments,
    "cancel_motion": ToolArguments,
    "verify_target": VerificationArguments,
    "propose_finish": ToolArguments,
}
_TOOL_CALLS = tuple(create_model(
    "Call_" + name, __base__=ToolArguments,
    name=(Literal[name], ...), arguments=(arguments, ...), reason=(str, ...),
) for name, arguments in _TOOL_ARGUMENTS.items())


class SearchUpdate(ToolArguments):
    concept_id: str
    detector_terms: list[str] = Field(min_length=1, max_length=4)
    basis_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class InterpretationUpdate(ToolArguments):
    interpretation: ParsedInstruction
    basis_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class ToolCall(ToolArguments):
    """One typed action with optional interpretation or retrieval revision."""
    action: Union[_TOOL_CALLS]
    search_updates: list[SearchUpdate] = Field(default_factory=list)
    interpretation_update: InterpretationUpdate | None = None


class AgentRuntime:
    """Execute one model/tool turn at a time with a run-owned budget and trace.

    Args:
        tools: Backend-bound domain tools, sharing the existing motion owner.
        trace_path: JSONL journal for tool proposals and outcomes.
        max_model_calls: Total decision-model calls allowed for this episode.
        vlm: Existing model provider name.
        client_model: Optional client/model pair for deterministic replay.

    在已有推理 worker 中调用 step；控制 timer 可独立调用 supervisor.hold/cancel。
    模型历史只保留最近四个选择或工具结果，完整输入由模型适配器逐次记录。
    """
    def __init__(self, tools: NavigationTools, trace_path, *, max_model_calls=100,
                 vlm="ark", client_model=None):
        self.tools = tools
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_model_calls = max_model_calls
        self.model_calls = tools.session.plan.diagnostics.get("initial_model_calls", 0)
        self.history = []
        self._choice_context = None
        self.client, self.model = client_model or get_client_and_model(vlm)
        skill_root = Path(__file__).resolve().parent.parent / "skills"
        self.skill_content = "\n\n".join((skill_root / name).read_text(encoding="utf-8") for name in (
            "understand_instruction.md", "search.md", "active_observation.md", "recovery.md", "completion.md",
        ))
        self.decision_context = (
            "Use the original instruction, task contract, authoritative progress, stored verification "
            "records and recent events. Verification records are past judgments scoped to their "
            "task revision, predicate, object pair and observation dependencies, not assertions "
            "that predicates are currently satisfied. Missing evidence remains unknown. "
            "Only observation_images identifies current camera frames; mapped objects may be historical. "
            "An empty list or a missing image supplies no visual evidence; exploration may still "
            "continue using the supplied legal options. A selected action is not an execution "
            "receipt or proof of task completion. Destination hypotheses are revisable observed "
            "facility candidates, not replacements for the original demand. An instance rejection "
            "does not reject every realization of that resource; keep all original functional "
            "and attribute requirements when choosing another candidate.\n\n"
            "Revision protocol: search_updates replaces only detector_terms for a supplied concept_id; "
            "copy basis_ids from search_context.basis and cite only supplied evidence. "
            "interpretation_update proposes a complete corrected instruction structure, with a reason "
            "and basis_ids including instruction. Preserve every explicit requirement; do not alter "
            "the request to fit scene convenience. Submit only one revision type, or [] and null. "
            "In tool mode pair a revision only with observe or query_scene. A changed interpretation "
            "is compiled and installed in this decision turn; read its result and reselect before executing. "
            "An invalid interpretation retains the current task and is not an instance rejection.\n\n"
            + self.skill_content
        )
        contracts = "\n".join(
            f"{name}{inspect.signature(getattr(tools, name))}: {getattr(tools, name).__doc__}"
            for name in tools.names
        )
        self.system_prompt = (
            "You control navigation exclusively through the listed tools. Return one ToolCall. "
            "Copy observed IDs verbatim, including prefixes; use the current task revision/control epoch. "
            "A reached motion is distinct from verified task completion. "
            "Pair search_updates or interpretation_update only with observe or query_scene, "
            "never an execution action; never submit both revision types.\n"
            + contracts + "\n\n" + self.decision_context
        )

    def choose_frontier(self, options, recommended_uid, reason):
        """Choose once from the baseline's reachable options at a search event.

        The domain retains coordinates and execution ownership. The response
        schema restricts the model to the exact option IDs supplied this turn.
        """
        self._choice_context = None
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("Agent decision-model budget exhausted")
        source_snapshot = self.tools.session.snapshot
        messages = self._messages(
            "Select one reachable exploration branch for the current task. "
            "Use observed semantics and path costs. The baseline recommendation is a valid option. "
            "The navigation driver owns motion, verification and STOP. Return a FrontierChoice "
            "with an exact frontier_uid from options, your reason, and search_updates for any "
            "justified retrieval revision (otherwise []).\n\n" + self.decision_context,
            options=options, baseline_recommendation=recommended_uid, event=reason,
        )
        choice_type = self._response_type("FrontierChoice", messages,
            frontier_uid=(Literal[tuple(option["frontier_uid"] for option in options)], ...),
            reason=(str, ...))
        self.model_calls += 1
        completion = self.client.beta.chat.completions.parse(
            model=self.model, temperature=0.0, parse_retries=0, response_format=choice_type,
            trace_label="navigation_agent", messages=messages,
        )
        choice = completion.choices[0].message.parsed
        revision, updates = self._apply_decision_updates(choice, source_snapshot, messages)
        selected = None if revision else choice.frontier_uid
        self.record_event({"event": "frontier_choice", "call_index": self.model_calls,
                        "options": options, "baseline_recommendation": recommended_uid,
                        "trigger": reason, "search_updates": updates,
                        "selected_uid": selected, "reason": choice.reason,
                        "interpretation_changed": bool(revision and revision["interpretation_changed"]),
                        "interpretation_result": revision})
        return selected

    def choose_recovery(self, options, reason):
        """Select one domain-owned action; the driver executes its lifecycle."""
        self._choice_context = None
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("Agent decision-model budget exhausted")
        source_snapshot = self.tools.session.snapshot
        messages = self._messages(
            "Choose one executable recovery action for the task. Options contain untried object "
            "views and reachable exploration branches. Use unresolved evidence and travel cost. "
            "The navigation driver owns motion, verification and STOP. Return a RecoveryChoice "
            "with an exact action_uid from options, your reason, and search_updates for any "
            "justified retrieval revision (otherwise []).\n\n" + self.decision_context,
            options=options, event=reason,
        )
        choice_type = self._response_type("RecoveryChoice", messages,
            action_uid=(Literal[tuple(option["action_uid"] for option in options)], ...),
            reason=(str, ...))
        self.model_calls += 1
        completion = self.client.beta.chat.completions.parse(
            model=self.model, temperature=0.0, parse_retries=0, response_format=choice_type,
            trace_label="navigation_agent", messages=messages,
        )
        choice = completion.choices[0].message.parsed
        revision, updates = self._apply_decision_updates(choice, source_snapshot, messages)
        selected = None if revision else choice.action_uid
        self.record_event({"event": "recovery_choice", "call_index": self.model_calls,
                        "options": options, "trigger": reason, "search_updates": updates,
                        "selected_uid": selected, "reason": choice.reason,
                        "interpretation_changed": bool(revision and revision["interpretation_changed"]),
                        "interpretation_result": revision})
        return selected

    def step(self):
        """Execute a single proposal and return its recorded tool outcome.

        Raises:
            RuntimeError: The configured episode model budget is exhausted.
            Exception: Original model, tool, or trace IO failure.
        """
        self._choice_context = None
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("Agent decision-model budget exhausted")
        source_snapshot = self.tools.session.snapshot
        messages = self._messages(self.system_prompt)
        response_type = self._response_type("ToolCall", messages, base=ToolCall)
        self.model_calls += 1
        completion = self.client.beta.chat.completions.parse(
            model=self.model, messages=messages, response_format=response_type,
            temperature=0.0, parse_retries=0, trace_label="navigation_agent",
        )
        response = completion.choices[0].message.parsed
        call = response.action
        arguments = call.arguments.dict(exclude_none=True)
        proposal = {"call_index": self.model_calls, "name": call.name,
                    "arguments": arguments, "reason": call.reason,
                    "search_updates": [update.dict() for update in response.search_updates]}
        if response.interpretation_update is not None:
            proposal["interpretation_update"] = response.interpretation_update.dict()
        self._record({**proposal, "event": "proposed"})
        try:
            if (response.search_updates or response.interpretation_update is not None
                    ) and call.name not in {"observe", "query_scene"}:
                raise ValueError("Revise search or interpretation with observe or query_scene, not an execution action")
            revision, _ = self._apply_decision_updates(response, source_snapshot, messages)
            if revision:
                outcome = {**proposal, "event": "returned", "result": revision}
                self.record_event(outcome)
                return outcome
            if call.name in {"submit_motion", "verify_target"}:
                uid = arguments.get("candidate_uid", "")
                current_snapshot = self.tools.session.snapshot
                if object_dependencies(source_snapshot, uid) != object_dependencies(current_snapshot, uid):
                    raise ValueError("Proposal observation dependencies changed during inference")
                if call.name == "submit_motion":
                    frontier = arguments.get("frontier_uid")
                    view_uid = frontier or arguments["viewpoint_uid"]
                    before_items = source_snapshot.frontiers if frontier else source_snapshot.viewpoints
                    after_items = current_snapshot.frontiers if frontier else current_snapshot.viewpoints
                    before = next((v for v in before_items if v.uid == view_uid), None)
                    after = next((v for v in after_items if v.uid == view_uid), None)
                    if before != after:
                        raise ValueError("Proposal viewpoint changed during inference")
            result = self.tools.dispatch(call.name, arguments)
        except Exception as error:
            self.record_event({**proposal, "event": "failed", "error_type": type(error).__name__,
                            "error": str(error)})
            raise
        outcome = {**proposal, "event": "returned", "result": result}
        self.record_event(outcome)
        return outcome

    def _response_type(self, name, messages, *, base=ToolArguments, **fields):
        observation = json.loads(messages[1]["content"][0]["text"])["observation"]
        context = observation["search_context"]
        update_type = create_model("ScopedSearchUpdate", __base__=SearchUpdate,
            concept_id=(Literal[tuple(item["id"] for item in context["concepts"])], ...),
            basis_ids=(list[Literal[tuple(context["basis"])]], Field(min_length=1)))
        interpretation_type = create_model("ScopedInterpretationUpdate", __base__=InterpretationUpdate,
            basis_ids=(list[Literal[tuple(context["basis"])]], Field(min_length=1)))
        return create_model(name, __base__=base, **fields,
            interpretation_update=(Union[interpretation_type, None], None),
            search_updates=(list[update_type], Field(default_factory=list,
                                                    max_length=len(context["concepts"]))))

    def _validate_context(self, source_snapshot, observation):
        session, motion = self.tools.session, self.tools.motion
        if observation["task"] != session.spec.as_dict():
            raise ValueError("Decision task changed during inference")
        if observation["search_context"]["revision"] != session.search_revision:
            raise ValueError("Decision belongs to a stale search revision")
        if source_snapshot is not session.snapshot:
            raise ValueError("Decision observation dependencies changed during inference")
        if observation["progress"] != session.progress.as_dict():
            raise ValueError("Decision task progress changed during inference")
        if observation["motion"] != {"phase": motion.phase, "held": motion.held,
                                    "control_epoch": motion.control_epoch,
                                    "active_request_id": motion.active_request_id}:
            raise RuntimeError("Decision motion state or control epoch changed during inference")

    @contextmanager
    def choice_guard(self):
        if self._choice_context is None:
            raise ValueError("No current guided choice to execute")
        snapshot, observation = self._choice_context
        self._choice_context = None
        with self.tools.motion.idle_guard(control_epoch=observation["motion"]["control_epoch"]):
            self._validate_context(snapshot, observation)
            yield

    def _apply_decision_updates(self, response, source_snapshot, messages):
        observation = json.loads(messages[1]["content"][0]["text"])["observation"]
        self._validate_context(source_snapshot, observation)
        update = response.interpretation_update
        if update is not None and response.search_updates:
            raise ValueError("Interpretation and search revisions are mutually exclusive")
        if update is not None:
            outcome = self._apply_interpretation(update, source_snapshot, messages)
            if outcome:
                return outcome, []
        updates = self._apply_search_updates(response.search_updates, source_snapshot, messages)
        observation["search_context"]["revision"] = self.tools.session.search_revision
        self._choice_context = (source_snapshot, observation)
        return None, updates

    def _apply_interpretation(self, update, source_snapshot, messages):
        from instruction.compiler import compile_parsed_instruction
        from instruction.contracts import interpretation_changes
        from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES
        from instruction_adapter.task_compat import runtime_plan
        from perception.detector_adapter import ground_plan

        observation = json.loads(messages[1]["content"][0]["text"])["observation"]
        context, session = observation["search_context"], self.tools.session
        if "instruction" not in update.basis_ids or not set(update.basis_ids) <= set(context["basis"]):
            raise ValueError("Interpretation must cite the original instruction and supplied evidence")
        if observation["task"] != session.spec.as_dict() or context["revision"] != session.search_revision:
            raise ValueError("Interpretation proposal belongs to a stale revision")
        if source_snapshot is not session.snapshot:
            raise ValueError("Interpretation observation changed during inference")
        if session.spec.source != "language":
            raise ValueError("Only language interpretations may be revised")
        epoch = observation["motion"]["control_epoch"]
        with self.tools.motion.idle_guard(control_epoch=epoch):
            current = session.spec
        compiled = compile_parsed_instruction(current.raw_instruction, update.interpretation,
            RUNTIME_CONSTRAINT_TYPES, revision=current.revision + 1)
        if compiled.task is None:
            return {"interpretation_changed": False, "task_revision": current.revision,
                    "interpretation_status": compiled.status,
                    "reason": "; ".join(compiled.diagnostics)}
        proposed = compiled.task
        changes = interpretation_changes(current, proposed)
        if not changes:
            return None
        # 完整解释已由决策模型结合观测提出；编译后在原有守卫内直接安装。
        with self.tools.motion.idle_guard(control_epoch=epoch):
            self._validate_context(source_snapshot, observation)
            plan = ground_plan(runtime_plan(proposed), available_classes=(),
                               use_legacy_similarity_fallback=False)
            record = self.tools.install_interpretation(plan, task_revision=current.revision,
                search_revision=context["revision"], snapshot=source_snapshot,
                record={"reason": update.reason, "basis_ids": update.basis_ids,
                        "basis": {uid: context["basis"][uid] for uid in update.basis_ids},
                        "control_epoch": epoch, "changes": changes})
        self._record({"event": "interpretation_adopted", "call_index": self.model_calls, **record})
        self.history.clear()
        return {"interpretation_changed": True, "task_revision": session.spec.revision}

    def _apply_search_updates(self, updates, source_snapshot, messages):
        if not updates:
            return []
        observation = json.loads(messages[1]["content"][0]["text"])["observation"]
        context = observation["search_context"]
        allowed = set(context["basis"])
        if any(not set(update.basis_ids) <= allowed for update in updates):
            raise ValueError("Search proposal cites evidence absent from the request")
        session = self.tools.session
        if observation["task"] != session.spec.as_dict():
            raise ValueError("Search proposal task changed during inference")
        return session.revise_search([
            {**update.dict(), "basis": {key: context["basis"][key] for key in update.basis_ids}}
            for update in updates], task_revision=observation["task"]["revision"],
            search_revision=context["revision"], snapshot=source_snapshot)

    def _messages(self, system_prompt, **context):
        """Bind current backend frames to pixels without capturing new evidence.

        Reuse the verifier's local, content-addressed image encoder. Missing
        sources remain explicit facts; other source/archive IO errors propagate.
        """
        observation = self.tools.observe()
        image_blocks = []
        for image in observation["observation_images"]:
            ref = image["image_ref"]
            if not isinstance(ref, str) or not Path(ref).is_absolute():
                image.update(image_status="missing", missing_reason="no_local_absolute_image_ref")
                continue
            try:
                # Check absence before MIME validation, including unknown suffixes.
                Path(ref).stat()
                block, archived, digest = image_input(ref, self.trace_path.parent / "observation_images")
            except FileNotFoundError as error:
                if error.filename != ref:
                    raise  # An archive failure is not a missing camera observation.
                image.update(image_status="missing", missing_reason="file_not_found")
                continue
            image.update(image_status="available", request_image_ref=archived, image_sha256=digest)
            image_blocks.extend([
                {"type": "text", "text": "Current observation frame and visible object UIDs "
                 "(object_boxes are xyxy): " + json.dumps(image, ensure_ascii=False)},
                block,
            ])
        basis = {"instruction": {"type": "original_instruction"}}
        basis.update({f"frame:{image['id']}": {"type": "current_frame", "id": image["id"],
                                             "image_sha256": image["image_sha256"]}
                      for image in observation["observation_images"]
                      if image["image_status"] == "available"})
        basis.update({f"verification:{index}": {"type": "stored_judgment", "record": record}
                      for index, record in enumerate(observation["verification_records"])})
        observation["search_context"]["basis"] = basis
        context = {"observation": observation, "recent_tools": self.history[-4:], **context}
        content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}, *image_blocks]
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

    def record_event(self, event):
        """Journal a decision or execution fact and retain only four recent events."""
        self._record(event)
        self.history.append(event)
        self.history = self.history[-4:]

    def _record(self, event):
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
