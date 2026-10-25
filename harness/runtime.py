"""A bounded Agent tool loop; backends retain sensor and motion execution."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, create_model

from harness.tools import NavigationTools
from scene.dependencies import object_dependencies
from llm_utils.cognav_llm_adapter import get_client_and_model


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


class ToolCall(BaseModel):
    """One typed action; the tool name determines its legal argument fields."""
    action: Union[_TOOL_CALLS]


class AgentRuntime:
    """Execute one model/tool turn at a time with a run-owned budget and trace.

    Args:
        tools: Backend-bound domain tools, sharing the existing motion owner.
        trace_path: JSONL journal for tool proposals and outcomes.
        max_model_calls: Total decision-model calls allowed for this episode.
        vlm: Existing model provider name.
        client_model: Optional client/model pair for deterministic replay.

    在已有推理 worker 中调用 step；控制 timer 可独立调用 supervisor.hold/cancel。
    模型历史只保留最近四个工具结果，完整输入由模型适配器逐次记录。
    """
    def __init__(self, tools: NavigationTools, trace_path, *, max_model_calls=100,
                 vlm="ark", client_model=None):
        self.tools = tools
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_model_calls = max_model_calls
        self.model_calls = 0
        self.history = []
        self.client, self.model = client_model or get_client_and_model(vlm)
        skill_root = Path(__file__).resolve().parent.parent / "skills"
        skills = "\n\n".join((skill_root / name).read_text() for name in (
            "search.md", "active_observation.md", "recovery.md", "completion.md",
        ))
        contracts = "\n".join(
            f"{name}{inspect.signature(getattr(tools, name))}: {getattr(tools, name).__doc__}"
            for name in tools.names
        )
        self.system_prompt = (
            "You control navigation exclusively through the listed tools. Return one ToolCall. "
            "Copy observed IDs verbatim, including prefixes; use the current task revision/control epoch. "
            "A reached motion is distinct from verified task completion.\n"
            + contracts + "\n\n" + skills
        )

    def step(self):
        """Execute a single proposal and return its recorded tool outcome.

        Raises:
            RuntimeError: The configured episode model budget is exhausted.
            Exception: Original model, tool, or trace IO failure.
        """
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("Agent decision-model budget exhausted")
        source_snapshot = self.tools.session.snapshot
        context = {"observation": self.tools.observe(), "recent_tools": self.history[-4:]}
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        self.model_calls += 1
        completion = self.client.beta.chat.completions.parse(
            model=self.model, messages=messages, response_format=ToolCall,
            temperature=0.0, trace_label="navigation_agent",
        )
        call = completion.choices[0].message.parsed.action
        arguments = call.arguments.dict(exclude_none=True)
        proposal = {"call_index": self.model_calls, "name": call.name,
                    "arguments": arguments, "reason": call.reason}
        self._record({**proposal, "event": "proposed"})
        try:
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
            self._record({**proposal, "event": "failed", "error_type": type(error).__name__,
                          "error": str(error)})
            raise
        outcome = {**proposal, "event": "returned", "result": result}
        self._record(outcome)
        self.history.append(outcome)
        self.history = self.history[-4:]
        return outcome

    def _record(self, event):
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
