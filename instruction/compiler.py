"""Single strict natural-language compiler. Dataset inputs use backend adapters."""
from __future__ import annotations

import json
import os
from pathlib import Path

from instruction.contracts import CompileResult, ConstraintSpec, TargetSpec, TaskSpec
from instruction.parser import (
    InstructionParseUnavailable, parse_instruction_with_llm, plan_from_parsed_instruction,
)


def task_from_parsed_plan(plan, *, revision=1, source="language") -> TaskSpec:
    """Freeze parsed semantics before any detector mapping or strategy choice."""
    demand_constraints = tuple(ConstraintSpec(
        id=f"demand:{t.id}", type="affordance", subject=t.id, value=t.demand,
        hardness="hard", verifier="vlm", source="demand",
    ) for t in plan.targets if t.demand and not any(
        c.source == "demand" and c.subject == t.id for c in plan.constraints))
    return TaskSpec(
        raw_instruction=plan.raw_instruction,
        targets=tuple(TargetSpec(
            id=t.id, name=t.name, role=t.role, terminal=t.terminal,
            min_count=t.min_count, aliases=tuple(t.aliases), attributes=t.attributes, demand=t.demand,
        ) for t in plan.targets),
        constraints=tuple(ConstraintSpec(
            id=f"c{i}", type=c.type, subject=c.subject, relation=c.relation,
            object=c.object, value=c.value, hardness=c.hardness,
            verifier=c.verifier, source=c.source,
        ) for i, c in enumerate(plan.constraints)) + demand_constraints,
        revision=revision, ordered=plan.execution.ordered,
        exhaustive=plan.execution.exhaustive,
        completion_mode=plan.eval_mode,
        search_priors=plan.search_priors.as_dict(), source=source,
    )


def compile_instruction(text, capabilities, *, vlm="cognav", revision=1) -> CompileResult:
    """Parse once and reject unsupported hard predicates without executing them.

    Args:
        text: User-visible instruction, without hidden benchmark metadata.
        capabilities: Supported constraint types advertised by the backend.
        vlm: Existing configured model provider.
        revision: Task revision allocated by the task owner.
    """
    if not str(text or "").strip():
        return CompileResult("invalid", diagnostics=("empty_instruction",))
    try:
        plan = parse_instruction_with_llm(text, vlm=vlm, strict=True)
    except InstructionParseUnavailable as exc:
        return CompileResult("parse_unavailable", diagnostics=(str(exc),))
    return _compile_plan(plan, capabilities, revision=revision)


def compile_parsed_instruction(text, parsed, capabilities, *, revision=1) -> CompileResult:
    """将结构化解释编译为统一任务契约。"""
    if not str(text or "").strip():
        return CompileResult("invalid", diagnostics=("empty_instruction",))
    plan = plan_from_parsed_instruction(text, parsed)
    return _compile_plan(plan, capabilities, revision=revision)


class InitialInstructionCompiler:
    """在调用预算内解析指令，无效结构允许一次修正。"""

    def __init__(self, *, max_model_calls, trace_path, vlm="cognav", client_model=None):
        self.max_model_calls = max_model_calls
        self.model_calls = 0
        self.trace_path = Path(trace_path)
        self.vlm = vlm
        self.client_model = client_model

    def compile(self, text, capabilities, *, revision=1) -> CompileResult:
        from prompting.schemas import ParsedInstruction
        from prompting.templates import INSTRUCTION_PARSE_PROMPT

        if not str(text or "").strip():
            return self._finish(CompileResult("invalid", diagnostics=("empty_instruction",)))
        if self.client_model is None and os.getenv("LLM_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}:
            return self._finish(CompileResult("parse_unavailable", diagnostics=("llm_offline",)))
        request = {"original_instruction": text}
        for stage in ("initial_parse", "initial_correction"):
            if self.model_calls >= self.max_model_calls:
                return self._finish(CompileResult("budget_exhausted", diagnostics=(stage,)))
            prompt = INSTRUCTION_PARSE_PROMPT
            if stage == "initial_correction":
                prompt += (
                    "\nReturn one complete interpretation that resolves the reported structural errors. "
                    "Preserve the original instruction's requirements and express genuine ambiguity "
                    "in unresolved. Use the previous response and diagnostics to locate the error."
                )
            parsed, raw, errors = self._request(stage, request, prompt, ParsedInstruction)
            result = (compile_parsed_instruction(text, parsed, capabilities, revision=revision)
                      if parsed is not None else CompileResult("invalid", diagnostics=errors))
            current = parsed.model_dump() if parsed is not None else None
            self._record({"event": "compilation", "stage": stage, "interpretation": current,
                          "status": result.status, "diagnostics": result.diagnostics})
            if result.status != "invalid" or stage == "initial_correction":
                return self._finish(result)
            request.update(previous_interpretation=current, previous_response=raw,
                           diagnostics=list(result.diagnostics), status=result.status)

    def _request(self, stage, request, prompt, response_format):
        from llm_utils.cognav_llm_adapter import get_client_and_model
        from llm_utils.structured_output import StructuredOutputError

        if self.client_model is None:
            self.client_model = get_client_and_model(self.vlm)
        client, model = self.client_model
        self.model_calls += 1
        self._record({"event": "request", "stage": stage, "input": request})
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=[{"role": "system", "content": prompt},
                                       {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
                response_format=response_format, trace_label=stage,
                temperature=0.0, max_tokens=4096, parse_retries=0)
        except StructuredOutputError as error:
            self._record({"event": "invalid_response", "stage": stage,
                          "raw_response": error.raw_content, "diagnostics": [str(error)]})
            return None, error.raw_content, (str(error),)
        except Exception as error:
            self._record({"event": "request_failed", "stage": stage, "error_type": type(error).__name__})
            raise
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            self._record({"event": "request_refused", "stage": stage})
            raise RuntimeError("Initial interpretation request was refused")
        parsed = message.parsed
        raw = message.content or ""
        self._record({"event": "response", "stage": stage, "raw_response": raw,
                      "parsed": parsed.model_dump() if parsed is not None else None})
        return parsed, raw, ("empty_llm_parse",) if parsed is None else ()

    def _record(self, event):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**event, "model_calls": self.model_calls}, ensure_ascii=False) + "\n")

    def _finish(self, result):
        self._record({"event": "admission", "status": result.status,
                      "diagnostics": result.diagnostics,
                      "task": result.task.as_dict() if result.task is not None else None})
        return result


def _compile_plan(plan, capabilities, *, revision) -> CompileResult:
    if plan.diagnostics.get("unresolved"):
        return CompileResult("ambiguous", diagnostics=tuple(plan.diagnostics["unresolved"]))
    if plan.diagnostics.get("invalid"):
        return CompileResult("invalid", diagnostics=tuple(plan.diagnostics["invalid"]))
    try:
        task = task_from_parsed_plan(plan, revision=revision)
    except (ValueError, TypeError) as exc:
        return CompileResult("invalid", diagnostics=(str(exc),))
    unsupported = tuple(c.id + ":" + c.type for c in task.constraints
                        if c.hardness == "hard" and c.type not in capabilities)
    if unsupported:
        return CompileResult("unsupported", diagnostics=unsupported)
    return CompileResult("success", task)
