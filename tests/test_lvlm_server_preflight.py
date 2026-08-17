"""Tests for portable Qwen-VL model snapshot validation."""

import hashlib
import json

import pytest

from deployment.lvlm_server.preflight import validate_model_directory


def _write_model_snapshot(root, *, missing_second_shard: bool = False):
    """Create a minimal indexed model snapshot for one test."""

    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first")
    if not missing_second_shard:
        (root / "model-00002-of-00002.safetensors").write_bytes(b"second")


def test_preflight_accepts_complete_indexed_snapshot(tmp_path) -> None:
    model_path = tmp_path / "model"
    _write_model_snapshot(model_path)

    report = validate_model_directory(model_path)

    assert report["model_type"] == "qwen3_vl"
    assert report["weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert report["weight_bytes"] == 11
    assert report["auxiliary_files"] == [
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]


def test_preflight_rejects_missing_indexed_shard(tmp_path) -> None:
    model_path = tmp_path / "model"
    _write_model_snapshot(model_path, missing_second_shard=True)

    with pytest.raises(FileNotFoundError, match="incomplete model weights"):
        validate_model_directory(model_path)


def test_preflight_builds_optional_migration_hashes(tmp_path) -> None:
    model_path = tmp_path / "model"
    _write_model_snapshot(model_path)

    report = validate_model_directory(model_path, compute_sha256=True)

    assert report["weight_sha256"] == {
        "model-00001-of-00002.safetensors": hashlib.sha256(b"first").hexdigest(),
        "model-00002-of-00002.safetensors": hashlib.sha256(b"second").hexdigest(),
    }


def test_preflight_rejects_missing_visual_processor(tmp_path) -> None:
    model_path = tmp_path / "model"
    _write_model_snapshot(model_path)
    (model_path / "preprocessor_config.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer/processor"):
        validate_model_directory(model_path)
