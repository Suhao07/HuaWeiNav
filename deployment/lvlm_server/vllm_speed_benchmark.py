#!/usr/bin/env python3
"""Measure sequential speed of an OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


def _image_data_url(path: Path) -> str:
    """Encode one local image as a data URL."""

    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _post(
    endpoint: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Send one non-streaming chat completion and attach client timing."""

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
            body["x_http_status"] = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc
    body["x_client_wall_seconds"] = time.perf_counter() - started
    return body


def _run_case(
    endpoint: str,
    *,
    name: str,
    model: str,
    messages: list[dict[str, Any]],
    warmups: int,
    measured: int,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Run warmups followed by measured sequential requests."""

    for _ in range(warmups):
        _post(
            endpoint,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    records = [
        _post(
            endpoint,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        for _ in range(measured)
    ]
    wall = [float(item["x_client_wall_seconds"]) for item in records]
    tokens = [int((item.get("usage") or {}).get("completion_tokens", 0)) for item in records]
    end_to_end_tps = [token / elapsed if elapsed > 0 else 0.0 for token, elapsed in zip(tokens, wall)]
    return {
        "name": name,
        "warmups": warmups,
        "measured": measured,
        "max_tokens": max_tokens,
        "completion_tokens": tokens,
        "client_wall_seconds": wall,
        "end_to_end_output_tokens_per_second": end_to_end_tps,
        "summary": {
            "client_wall_mean_seconds": statistics.mean(wall),
            "client_wall_p50_seconds": statistics.median(wall),
            "completion_tokens_mean": statistics.mean(tokens),
            "end_to_end_output_tokens_per_second_mean": statistics.mean(end_to_end_tps),
            "end_to_end_output_tokens_per_second_p50": statistics.median(end_to_end_tps),
        },
        "responses": records,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--visible-gpus", default="4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run text and single-image speed measurements."""

    args = build_parser().parse_args(argv)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    image_url = _image_data_url(args.image)
    text_messages = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "Return one short sentence confirming the service is ready."},
    ]
    image_messages = [
        {"role": "system", "content": "Answer concisely."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one short sentence."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    result = {
        "schema": "qwen25_vl_vllm_speed_benchmark/v1",
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "image": str(args.image.resolve()),
        "settings": {
            "warmups": args.warmups,
            "measured": args.measured,
            "max_tokens": args.max_tokens,
            "backend": args.backend,
            "visible_gpus": args.visible_gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        "cases": [
            _run_case(
                endpoint,
                name="text",
                model=args.model,
                messages=text_messages,
                warmups=args.warmups,
                measured=args.measured,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
            ),
            _run_case(
                endpoint,
                name="single_image",
                model=args.model,
                messages=image_messages,
                warmups=args.warmups,
                measured=args.measured,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
