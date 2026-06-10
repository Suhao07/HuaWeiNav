from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import RELATION_VERIFY
from prompting.schemas import HAS_PYDANTIC, ParsedRelationResult
from prompting.templates import RELATION_VERIFIER_PROMPT

from .semantic_edges import DynamicSemanticEdgeVerifier, SemanticEdge


@dataclass
class RelationVerificationRequest:
    relation: str
    subject: dict[str, Any]
    object_: dict[str, Any]
    evidence_views: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["object"] = data.pop("object_")
        return data


def _image_block(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


class VLMRelationVerifier:
    """Prompt-first relation verifier used after geometric prefiltering."""

    def __init__(self, vlm: str = "cognav"):
        self.vlm = vlm

    def __call__(
        self,
        relation: str,
        subject: dict[str, Any],
        object_: dict[str, Any],
        evidence_views: list[dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any]:
        if os.getenv("STRIVE_RELATION_VERIFIER", "auto").lower() in ("0", "false", "no", "off"):
            return {"verified": False, "confidence": 0.0, "reason": "relation verifier disabled"}
        if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on"):
            return {"verified": False, "confidence": 0.0, "reason": "llm offline"}
        if not HAS_PYDANTIC:
            return {"verified": False, "confidence": 0.0, "reason": "pydantic unavailable"}

        request = RelationVerificationRequest(
            relation=relation,
            subject=subject,
            object_=object_,
            evidence_views=evidence_views,
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": json.dumps(request.as_dict(), ensure_ascii=False, indent=2)},
        ]
        for view in evidence_views:
            block = _image_block(str(view.get("rgb_path") or view.get("path") or ""))
            if block is not None:
                content.append({"type": "text", "text": f"evidence_view: {view.get('id', '')}"})
                content.append(block)

        try:
            client, model = get_client_and_model(self.vlm)
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": RELATION_VERIFIER_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=ParsedRelationResult,
                trace_label=RELATION_VERIFY.trace_label,
            )
            parsed = completion.choices[0].message.parsed
            return {
                "verified": bool(getattr(parsed, "verified", False)),
                "confidence": max(0.0, min(1.0, float(getattr(parsed, "confidence", 0.0) or 0.0))),
                "need_better_view": bool(getattr(parsed, "need_better_view", False)),
                "reason": str(getattr(parsed, "reason", "") or ""),
            }
        except Exception as exc:
            return {
                "verified": False,
                "confidence": 0.0,
                "need_better_view": False,
                "reason": f"relation vlm failed: {exc}",
            }


class DynamicRelationService:
    """Geometry-first, VLM-on-demand relation service."""

    def __init__(self, vlm: str = "cognav"):
        self.edge_verifier = DynamicSemanticEdgeVerifier()
        self.vlm_callback = VLMRelationVerifier(vlm=vlm)

    def reset(self):
        self.edge_verifier = DynamicSemanticEdgeVerifier()

    def verify(
        self,
        *,
        subject: dict[str, Any],
        relation: str,
        object_: dict[str, Any],
        evidence_views: list[dict[str, Any]],
        use_vlm: bool = True,
    ) -> SemanticEdge:
        callback = self.vlm_callback if use_vlm else None
        return self.edge_verifier.verify(
            subject=subject,
            relation=relation,
            object_=object_,
            evidence_views=evidence_views,
            vlm_callback=callback,
        )

    def as_dict(self) -> dict[str, Any]:
        return self.edge_verifier.cache.as_dict()
