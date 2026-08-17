"""Tests for the formal self-hosted LVLM deployment receipt."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from deployment.lvlm_server.accept_deployment import (
    AcceptanceStage,
    build_receipt,
    capture_stage,
    schema_stage,
    validate_ms_swift_checkout,
)


def test_capture_stage_preserves_failure_evidence() -> None:
    """Verify an exception becomes a failed stage instead of losing the receipt."""

    def fail() -> dict:
        raise RuntimeError("CUDA unavailable")

    stage = capture_stage("serving_runtime", fail)

    assert stage.success is False
    assert stage.data == {}
    assert "CUDA unavailable" in stage.failures[0]


def test_schema_stage_requires_every_case() -> None:
    """Verify one failed production schema rejects deployment acceptance."""

    stage = schema_stage(
        [
            {"name": "instruction_parser", "success": True, "failures": []},
            {
                "name": "final_instruction_verifier",
                "success": False,
                "failures": ["unsafe accept"],
            },
        ],
        latency_ms=12.5,
    )

    assert stage.success is False
    assert stage.data["case_count"] == 2
    assert "unsafe accept" in stage.failures[0]


def test_receipt_requires_every_stage() -> None:
    """Verify receipt success is the conjunction of all acceptance stages."""

    receipt = build_receipt(
        base_url="http://server:8000/v1",
        served_model="strive-qwen2.5-vl-7b",
        image_path=Path("frame.jpg"),
        stages=[
            AcceptanceStage(name="model_snapshot", success=True),
            AcceptanceStage(name="production_schemas", success=False, failures=("failed",)),
        ],
        started_at="2026-08-17T00:00:00+00:00",
    )

    assert receipt["schema"] == "strive.lvlm_deployment_acceptance/v1"
    assert receipt["success"] is False
    assert [stage["name"] for stage in receipt["stages"]] == [
        "model_snapshot",
        "production_schemas",
    ]


def test_ms_swift_checkout_must_be_clean_and_pinned(tmp_path: Path) -> None:
    """Verify formal acceptance rejects uncommitted serving-source changes."""

    root = tmp_path / "ms-swift"
    (root / "swift").mkdir(parents=True)
    (root / "swift" / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "swift/__init__.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    report = validate_ms_swift_checkout(root, revision)

    assert report["worktree_clean"] is True
    assert report["revision"] == revision

    (root / "swift" / "__init__.py").write_text("__version__ = 'dirty'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        validate_ms_swift_checkout(root, revision)
