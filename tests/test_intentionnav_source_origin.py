import hashlib
import json
from unittest.mock import Mock

import pytest

from scripts import launch_intentionnav_episode as launcher
from scripts.serve_intentionnav import source_manifest


@pytest.fixture
def frozen_origin(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "runtime.py").write_text("value = 1\n")
    origin = {
        "schema": "strive_frozen_worktree_v1", "source_root": "/original/main",
        "git_head": "a" * 40, "git_dirty": True, "git_status_porcelain": " M runtime.py\n",
        "git_head_after_copy": "a" * 40, "git_status_after_copy": " M runtime.py\n",
        "snapshot_semantics": "verified_worktree_copy_not_git_checkout",
        "source_manifest": source_manifest(root),
    }
    receipt = tmp_path / "origin.json"
    receipt.write_text(json.dumps(origin))
    return root, receipt, origin


def test_frozen_origin_binds_bytes_without_local_git(frozen_origin, monkeypatch):
    root, receipt, origin = frozen_origin
    run = Mock(side_effect=AssertionError("Frozen source must not invoke Git"))
    monkeypatch.setattr(launcher.subprocess, "check_output", run)
    actual = launcher.load_source_origin(root, receipt)
    assert actual["git_head"] == actual["git_head_after_copy"] == origin["git_head"]
    assert actual["git_dirty"] is True
    assert actual["git_provenance_source"] == "frozen_origin_receipt_not_local_git"
    assert actual["snapshot_semantics"] == "verified_worktree_copy_not_git_checkout"
    assert actual["source_origin_root"] == "/original/main"
    assert actual["origin_receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    run.assert_not_called()


@pytest.mark.parametrize("change", ["edit", "add", "delete"])
def test_frozen_origin_rejects_source_changes(frozen_origin, change):
    root, receipt, _ = frozen_origin
    if change == "edit":
        (root / "runtime.py").write_text("value = 2\n")
    elif change == "add":
        (root / "extra.py").write_text("value = 1\n")
    else:
        (root / "runtime.py").unlink()
    with pytest.raises(ValueError, match="source bytes"):
        launcher.load_source_origin(root, receipt)


@pytest.mark.parametrize("key,value", [
    ("schema", "unknown"), ("git_head", "not-a-commit"), ("git_head", "g" * 40),
    ("git_dirty", 1), ("git_dirty", False), ("git_status_porcelain", None),
    ("git_head_after_copy", "b" * 40), ("git_status_after_copy", ""),
    ("snapshot_semantics", "clean_git_checkout"), ("source_root", None),
])
def test_frozen_origin_rejects_invalid_claims(frozen_origin, key, value):
    root, receipt, origin = frozen_origin
    origin[key] = value
    receipt.write_text(json.dumps(origin))
    with pytest.raises(ValueError, match="Invalid frozen"):
        launcher.load_source_origin(root, receipt)


def test_origin_receipt_must_stay_outside_source(frozen_origin):
    root, receipt, _ = frozen_origin
    nested = root / "origin.json"
    nested.write_bytes(receipt.read_bytes())
    with pytest.raises(ValueError, match="outside"):
        launcher.load_source_origin(root, nested)


def test_live_source_still_reads_git_provenance(tmp_path, monkeypatch):
    run = Mock(side_effect=["b" * 40 + "\n", " M runtime.py\n"])
    monkeypatch.setattr(launcher.subprocess, "check_output", run)
    assert launcher.load_source_origin(tmp_path) == {
        "git_head": "b" * 40, "git_dirty": True, "git_status_porcelain": " M runtime.py\n"}
    assert [call.args[0][3:] for call in run.call_args_list] == [["rev-parse", "HEAD"], ["status", "--porcelain"]]
