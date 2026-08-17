#!/usr/bin/env python3
"""Validate VLN production schemas against a remote LVLM service."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from llm_utils.self_hosted_llm_adapter import SelfHostedOpenAICompatibleClient
from prompting.registry import (
    CONCEPT_GROUNDING,
    CONCEPT_MATCH_BATCH,
    FINAL_VERIFY,
    INSTRUCTION_PARSE,
    RELATION_VERIFY,
)
from prompting.schemas import (
    GroundingResult,
    ParsedBatchConceptMatch,
    ParsedInstruction,
    ParsedRelationResult,
    ParsedVerification,
)
from prompting.templates import (
    CONCEPT_GROUNDING_PROMPT,
    CONCEPT_MATCH_PROMPT,
    FINAL_VERIFIER_PROMPT,
    INSTRUCTION_PARSE_PROMPT,
    RELATION_VERIFIER_PROMPT,
)

from deployment.lvlm_server.smoke_client import image_data_url, request_json, request_status


CheckFunction = Callable[[Any], list[str]]


@dataclass(frozen=True)
class SchemaSmokeCase:
    """One production prompt/schema acceptance case.

    Args:
        name: Stable case name used in the deployment receipt.
        trace_label: Production LVLM trace label.
        system_prompt: Production system prompt.
        user_content: OpenAI-compatible text or multimodal user content.
        response_format: Pydantic response model class.
        check: Semantic invariant checker returning failure messages.
    """

    name: str
    trace_label: str
    system_prompt: str
    user_content: Any
    response_format: Any
    check: CheckFunction


def build_schema_cases(image_url: str) -> list[SchemaSmokeCase]:
    """Build the five production schema calls required for server acceptance.

    Args:
        image_url: Base64 data URL for one representative navigation frame.

    Returns:
        Ordered smoke cases for parser, grounding, matching, relation, and final
        verification.
    """

    instruction_payload = (
        "Instruction: Find a red book on a shelf.\n"
        "Dataset target fallback: book"
    )
    grounding_payload = {
        "instruction": "Find the television.",
        "concept": {
            "id": "t0_television",
            "name": "television",
            "role": "primary",
            "terminal": True,
        },
        "available_detector_classes": ["tv_monitor", "chair", "bookshelf", "book"],
    }
    match_payload = {
        "instruction": "Find a book on a shelf.",
        "concept": {
            "id": "c0_shelf",
            "name": "shelf",
            "role": "anchor",
            "detector_terms": ["bookshelf"],
            "terminal": False,
        },
        "observed_objects": [
            {"uid": "anchor_1", "label": "bookshelf", "confidence": 0.91},
            {"uid": "distractor_1", "label": "chair", "confidence": 0.88},
        ],
        "task": "Return one concept-match item per observed object uid.",
    }
    relation_payload = {
        "relation": "on",
        "subject": {"uid": "book_1", "label": "book"},
        "object": {"uid": "shelf_1", "label": "bookshelf"},
        "evidence_views": [{"id": "schema_smoke_view", "source": "operator_frame"}],
    }
    final_payload = {
        "raw_instruction": "Find a book on a shelf.",
        "instruction_plan": {
            "targets": [{"name": "book", "terminal": True}],
            "constraints": [{"type": "spatial", "relation": "on", "object": "shelf"}],
        },
        "candidate": {"uid": "book_1", "detector_label": "book"},
        "evidence": {
            "hard_stop_constraints": {
                "satisfied": False,
                "planner_infeasible": False,
                "reason": "schema smoke intentionally keeps the physical stop contract unsatisfied",
            },
            "view_control": {"budget_exhausted": False, "remaining_proposals": 1},
        },
    }

    return [
        SchemaSmokeCase(
            name="instruction_parser",
            trace_label=INSTRUCTION_PARSE.trace_label,
            system_prompt=INSTRUCTION_PARSE_PROMPT,
            user_content=instruction_payload,
            response_format=ParsedInstruction,
            check=_check_instruction,
        ),
        SchemaSmokeCase(
            name="concept_grounding",
            trace_label=CONCEPT_GROUNDING.trace_label,
            system_prompt=CONCEPT_GROUNDING_PROMPT,
            user_content=json.dumps(grounding_payload, ensure_ascii=False, indent=2),
            response_format=GroundingResult,
            check=_check_grounding,
        ),
        SchemaSmokeCase(
            name="concept_match_batch",
            trace_label=CONCEPT_MATCH_BATCH.trace_label,
            system_prompt=CONCEPT_MATCH_PROMPT,
            user_content=[
                {"type": "text", "text": json.dumps(match_payload, ensure_ascii=False, indent=2)}
            ],
            response_format=ParsedBatchConceptMatch,
            check=_check_batch_match,
        ),
        SchemaSmokeCase(
            name="relation_verifier",
            trace_label=RELATION_VERIFY.trace_label,
            system_prompt=RELATION_VERIFIER_PROMPT,
            user_content=[
                {"type": "text", "text": json.dumps(relation_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            response_format=ParsedRelationResult,
            check=_check_relation,
        ),
        SchemaSmokeCase(
            name="final_instruction_verifier",
            trace_label=FINAL_VERIFY.trace_label,
            system_prompt=FINAL_VERIFIER_PROMPT,
            user_content=[
                {"type": "text", "text": json.dumps(final_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            response_format=ParsedVerification,
            check=_check_final_guard,
        ),
    ]


def run_schema_cases(
    client: SelfHostedOpenAICompatibleClient,
    *,
    model: str,
    cases: Sequence[SchemaSmokeCase],
) -> list[dict[str, Any]]:
    """Execute schema cases and return deployment receipt records.

    Args:
        client: Configured self-hosted structured client.
        model: Served model name.
        cases: Ordered production schema cases.

    Returns:
        JSON-friendly records containing raw, parsed, latency, and invariant
        results. Individual call failures are captured instead of aborting the
        remaining cases.
    """

    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": case.system_prompt},
                    {"role": "user", "content": case.user_content},
                ],
                response_format=case.response_format,
                trace_label=case.trace_label,
                temperature=0,
                max_tokens=1024,
            )
            message = completion.choices[0].message
            parsed = message.parsed
            failures = list(case.check(parsed))
            records.append(
                {
                    "name": case.name,
                    "trace_label": case.trace_label,
                    "schema": case.response_format.__name__,
                    "success": not failures,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "parsed": _model_payload(parsed),
                    "raw_response": str(getattr(message, "content", "") or ""),
                    "failures": failures,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "name": case.name,
                    "trace_label": case.trace_label,
                    "schema": case.response_format.__name__,
                    "success": False,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "parsed": None,
                    "raw_response": "",
                    "failures": [f"{type(exc).__name__}: {exc}"],
                }
            )
    return records


def _check_instruction(parsed: ParsedInstruction) -> list[str]:
    """Check that instruction parsing produces an executable terminal target."""

    targets = list(parsed.targets or [])
    failures = []
    if not targets:
        failures.append("instruction parser returned no targets")
    if not any(bool(target.terminal) for target in targets):
        failures.append("instruction parser returned no terminal target")
    return failures


def _check_grounding(parsed: GroundingResult) -> list[str]:
    """Check that grounding selects at least one available detector class."""

    available = {"tv monitor", "chair", "bookshelf", "book"}
    terms = {_normalize(term) for term in parsed.detector_terms or []}
    if not terms:
        return ["concept grounding returned no detector terms"]
    if not terms & available:
        return [f"grounded terms do not use available detector classes: {sorted(terms)}"]
    return []


def _check_batch_match(parsed: ParsedBatchConceptMatch) -> list[str]:
    """Check that batched grounding returns one record per requested uid."""

    returned = {str(item.uid) for item in parsed.matches or []}
    expected = {"anchor_1", "distractor_1"}
    missing = sorted(expected - returned)
    return [f"batch matcher omitted uids: {missing}"] if missing else []


def _check_relation(parsed: ParsedRelationResult) -> list[str]:
    """Check relation response bounds without assuming the operator image label."""

    confidence = float(parsed.confidence)
    if not 0.0 <= confidence <= 1.0:
        return [f"relation confidence outside [0,1]: {confidence}"]
    return []


def _check_final_guard(parsed: ParsedVerification) -> list[str]:
    """Check that an unsatisfied physical stop contract cannot authorize STOP."""

    decision = _normalize(parsed.decision).replace(" ", "_")
    failures = []
    if bool(parsed.satisfied):
        failures.append("final verifier satisfied=true while hard stop constraints are false")
    if decision == "accept":
        failures.append("final verifier decision=accept while hard stop constraints are false")
    return failures


def _model_payload(model: Any) -> dict[str, Any]:
    """Serialize a Pydantic v1/v2 response model."""

    dumper = getattr(model, "model_dump", None)
    if dumper is not None:
        return dict(dumper())
    return dict(model.dict())


def _normalize(value: Any) -> str:
    """Normalize one semantic term for acceptance checks."""

    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def build_parser() -> argparse.ArgumentParser:
    """Build the deployment schema-smoke CLI parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="vln-qwen2.5-vl-7b")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("lvlm_schema_smoke_receipt.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run health, discovery, and five production schema checks.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Zero only when every endpoint and schema invariant passes.
    """

    args = build_parser().parse_args(argv)
    base_url = str(args.base_url).rstrip("/")
    server_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    health_status = request_status(server_root + "/health", api_key=args.api_key, timeout_s=args.timeout)
    models = request_json(base_url + "/models", api_key=args.api_key, timeout_s=args.timeout)
    model_ids = [str(item.get("id", "")) for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"served model {args.model!r} not found in {model_ids}")

    client = SelfHostedOpenAICompatibleClient(
        base_url=base_url,
        api_key=args.api_key or "EMPTY",
        timeout_s=args.timeout,
        parse_retries=max(0, int(args.parse_retries)),
    )
    previous_fallback = os.environ.get("STRIVE_LLM_PARSE_FALLBACK")
    os.environ["STRIVE_LLM_PARSE_FALLBACK"] = "0"
    try:
        records = run_schema_cases(
            client,
            model=args.model,
            cases=build_schema_cases(image_data_url(args.image)),
        )
    finally:
        if previous_fallback is None:
            os.environ.pop("STRIVE_LLM_PARSE_FALLBACK", None)
        else:
            os.environ["STRIVE_LLM_PARSE_FALLBACK"] = previous_fallback

    receipt = {
        "schema": "strive.lvlm_schema_smoke_receipt/v1",
        "base_url": base_url,
        "model": args.model,
        "health_status": health_status,
        "advertised_models": model_ids,
        "image": str(args.image.resolve()),
        "success": all(bool(record["success"]) for record in records),
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
