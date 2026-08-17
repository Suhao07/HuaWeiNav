#!/usr/bin/env python3
"""Exercise health, model discovery, text, and optional image inference."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


def request_status(
    url: str,
    *,
    api_key: str = "",
    timeout_s: float = 60.0,
) -> int:
    """Send one status-only HTTP request.

    Args:
        url: Absolute endpoint URL.
        api_key: Optional bearer token.
        timeout_s: Socket timeout.

    Returns:
        HTTP response status code.

    Raises:
        RuntimeError: If the server cannot be reached or returns a non-2xx
            response.
    """

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str = "",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Send one JSON API request.

    Args:
        url: Absolute endpoint URL.
        method: HTTP method.
        payload: Optional JSON request body.
        api_key: Optional bearer token.
        timeout_s: Socket timeout.

    Returns:
        Decoded JSON object.

    Raises:
        RuntimeError: If the server returns a non-2xx response or invalid JSON.
    """

    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {raw[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"expected JSON object from {url}, got {type(parsed).__name__}")
    return parsed


def image_data_url(path: Path) -> str:
    """Encode one local image as an OpenAI-compatible data URL.

    Args:
        path: Local image file.

    Returns:
        Base64 data URL.

    Raises:
        FileNotFoundError: If the image does not exist.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-client CLI parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="strive-qwen2.5-vl-7b")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run model discovery and one deterministic chat request.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    server_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    health_status = request_status(
        f"{server_root}/health",
        api_key=args.api_key,
        timeout_s=args.timeout,
    )
    models = request_json(
        f"{base_url}/models",
        api_key=args.api_key,
        timeout_s=args.timeout,
    )
    model_ids = [str(item.get("id", "")) for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"served model {args.model!r} not found in {model_ids}")

    content: list[dict[str, Any]] = []
    if args.image is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(args.image)}})
    content.append(
        {
            "type": "text",
            "text": (
                "Return only JSON with keys ok, image_received, and summary. "
                "Set ok to true."
            ),
        }
    )
    completion = request_json(
        f"{base_url}/chat/completions",
        method="POST",
        api_key=args.api_key,
        timeout_s=args.timeout,
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 256,
        },
    )
    choices = completion.get("choices") or []
    if not choices:
        raise RuntimeError(f"completion contains no choices: {completion}")
    content_text = str((choices[0].get("message") or {}).get("content") or "")
    if not content_text.strip():
        raise RuntimeError("completion content is empty")
    print(
        json.dumps(
            {
                "health_status": health_status,
                "model": args.model,
                "models": model_ids,
                "response": content_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
