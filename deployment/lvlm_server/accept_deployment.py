#!/usr/bin/env python3
"""Run the formal acceptance gate for one deployed VLN LVLM server."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from llm_utils.self_hosted_llm_adapter import SelfHostedOpenAICompatibleClient

from deployment.lvlm_server.preflight import validate_model_directory, validate_runtime
from deployment.lvlm_server.schema_smoke import build_schema_cases, run_schema_cases
from deployment.lvlm_server.smoke_client import image_data_url, request_json, request_status


DEFAULT_MS_SWIFT_REVISION = ""


@dataclass(frozen=True)
class AcceptanceStage:
    """Result of one independently auditable deployment stage.

    Args:
        name: Stable stage identifier.
        success: Whether the stage satisfied its acceptance contract.
        data: JSON-friendly evidence produced by the stage.
        failures: Concrete failure messages.
        latency_ms: Wall-clock stage duration in milliseconds.
    """

    name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly stage record.

        Returns:
            Serialized stage payload.
        """

        payload = asdict(self)
        payload["failures"] = list(self.failures)
        return payload


def validate_ms_swift_checkout(root: Path, expected_revision: str) -> dict[str, Any]:
    """Validate the ms-swift source checkout used by the serving process.

    Args:
        root: Target-server ms-swift checkout.
        expected_revision: Required Git commit. An empty value disables exact
            revision comparison but still records the current revision.

    Returns:
        Checkout path and Git revision.

    Raises:
        FileNotFoundError: If the checkout does not expose the Swift package.
        RuntimeError: If Git inspection fails or the revision is unexpected.
    """

    root = root.expanduser().resolve()
    if not (root / "swift").is_dir():
        raise FileNotFoundError(f"ms-swift package directory is missing: {root / 'swift'}")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot inspect ms-swift revision: {completed.stderr.strip()}")
    revision = completed.stdout.strip()
    if expected_revision and revision != expected_revision:
        raise RuntimeError(
            f"ms-swift revision mismatch: expected={expected_revision}, actual={revision}"
        )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"cannot inspect ms-swift worktree: {status.stderr.strip()}")
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty_entries:
        raise RuntimeError(
            "ms-swift checkout contains uncommitted changes; deploy a clean pinned revision"
        )
    return {
        "root": str(root),
        "revision": revision,
        "expected_revision": expected_revision,
        "worktree_clean": True,
    }


def capture_stage(name: str, operation: Callable[[], dict[str, Any]]) -> AcceptanceStage:
    """Execute one acceptance operation without losing the final receipt.

    Args:
        name: Stable stage identifier.
        operation: Zero-argument operation returning JSON-friendly evidence.

    Returns:
        Successful or failed stage record. Exceptions are converted to concrete
        failure evidence so later independent checks can still run.
    """

    started = time.perf_counter()
    try:
        data = operation()
        return AcceptanceStage(
            name=name,
            success=True,
            data=data,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return AcceptanceStage(
            name=name,
            success=False,
            failures=(f"{type(exc).__name__}: {exc}",),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def schema_stage(records: Sequence[dict[str, Any]], latency_ms: float) -> AcceptanceStage:
    """Build the production-schema acceptance stage from per-call records.

    Args:
        records: Records returned by ``run_schema_cases``.
        latency_ms: Total wall-clock duration for all schema calls.

    Returns:
        Stage that succeeds only when every production schema invariant passes.
    """

    failed = [record for record in records if not bool(record.get("success"))]
    failures = tuple(
        f"{record.get('name', 'unknown')}: {', '.join(record.get('failures') or ['failed'])}"
        for record in failed
    )
    return AcceptanceStage(
        name="production_schemas",
        success=not failed,
        data={"case_count": len(records), "cases": list(records)},
        failures=failures,
        latency_ms=latency_ms,
    )


def build_receipt(
    *,
    base_url: str,
    served_model: str,
    image_path: Path,
    stages: Sequence[AcceptanceStage],
    started_at: str,
) -> dict[str, Any]:
    """Assemble the versioned deployment acceptance receipt.

    Args:
        base_url: OpenAI-compatible API root.
        served_model: Model id expected from ``/v1/models``.
        image_path: Representative image used by multimodal schema checks.
        stages: Ordered acceptance stages.
        started_at: UTC ISO timestamp captured before validation.

    Returns:
        Versioned receipt whose success requires every stage to pass.
    """

    stage_payloads = [stage.to_dict() for stage in stages]
    return {
        "schema": "strive.lvlm_deployment_acceptance/v1",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "base_url": base_url,
        "served_model": served_model,
        "image": str(image_path.expanduser().resolve()),
        "success": bool(stage_payloads) and all(item["success"] for item in stage_payloads),
        "stages": stage_payloads,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the target-server acceptance CLI.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--ms-swift-root", type=Path, default=Path(os.getenv("MS_SWIFT_ROOT", "/opt/vln/ms-swift")))
    parser.add_argument(
        "--expected-ms-swift-revision",
        default=os.getenv("MS_SWIFT_REVISION", DEFAULT_MS_SWIFT_REVISION),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default="vln-qwen2.5-vl-7b")
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLN_LVLM_API_KEY", os.getenv("STRIVE_LVLM_API_KEY", "")),
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lvlm_deployment_acceptance.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run local-server and remote-API acceptance as one auditable gate.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero only when checkout, model, runtime, API, and all production schemas
        pass their contracts.
    """

    args = build_parser().parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    base_url = str(args.base_url).rstrip("/")
    server_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    stages: list[AcceptanceStage] = []

    # ms-swift 可以直接从源码 checkout 加载。验收进程必须与服务入口使用同一源码根，
    # 防止检查了一个版本，实际服务却从另一个 site-packages 或工作区导入。
    ms_swift_import_root = str(args.ms_swift_root.expanduser().resolve())
    if ms_swift_import_root not in sys.path:
        sys.path.insert(0, ms_swift_import_root)

    stages.append(
        capture_stage(
            "ms_swift_checkout",
            lambda: validate_ms_swift_checkout(args.ms_swift_root, args.expected_ms_swift_revision),
        )
    )
    stages.append(
        capture_stage(
            "model_snapshot",
            lambda: validate_model_directory(args.model_path, compute_sha256=bool(args.sha256)),
        )
    )
    stages.append(capture_stage("serving_runtime", lambda: validate_runtime(require_gpu=True)))
    stages.append(
        capture_stage(
            "health",
            lambda: {
                "status": request_status(
                    server_root + "/health", api_key=args.api_key, timeout_s=args.timeout
                )
            },
        )
    )

    def discover_model() -> dict[str, Any]:
        models = request_json(base_url + "/models", api_key=args.api_key, timeout_s=args.timeout)
        model_ids = [str(item.get("id", "")) for item in models.get("data", [])]
        if args.served_model not in model_ids:
            raise RuntimeError(f"served model {args.served_model!r} not found in {model_ids}")
        return {"advertised_models": model_ids}

    discovery = capture_stage("model_discovery", discover_model)
    stages.append(discovery)

    # 只有服务发现成功才执行昂贵的五类请求；前置失败仍会写入完整失败回执。
    health_ok = stages[-2].success
    if health_ok and discovery.success:
        previous_fallback = os.environ.get("STRIVE_LLM_PARSE_FALLBACK")
        os.environ["STRIVE_LLM_PARSE_FALLBACK"] = "0"
        schema_started = time.perf_counter()
        try:
            client = SelfHostedOpenAICompatibleClient(
                base_url=base_url,
                api_key=args.api_key or "EMPTY",
                timeout_s=args.timeout,
                parse_retries=max(0, int(args.parse_retries)),
            )
            records = run_schema_cases(
                client,
                model=args.served_model,
                cases=build_schema_cases(image_data_url(args.image)),
            )
            stages.append(
                schema_stage(records, (time.perf_counter() - schema_started) * 1000.0)
            )
        except Exception as exc:
            stages.append(
                AcceptanceStage(
                    name="production_schemas",
                    success=False,
                    failures=(f"{type(exc).__name__}: {exc}",),
                    latency_ms=(time.perf_counter() - schema_started) * 1000.0,
                )
            )
        finally:
            if previous_fallback is None:
                os.environ.pop("STRIVE_LLM_PARSE_FALLBACK", None)
            else:
                os.environ["STRIVE_LLM_PARSE_FALLBACK"] = previous_fallback
    else:
        stages.append(
            AcceptanceStage(
                name="production_schemas",
                success=False,
                failures=("skipped because health or model discovery failed",),
            )
        )

    receipt = build_receipt(
        base_url=base_url,
        served_model=args.served_model,
        image_path=args.image,
        stages=stages,
        started_at=started_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
