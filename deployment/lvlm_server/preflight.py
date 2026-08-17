#!/usr/bin/env python3
"""Validate local Qwen-VL weights and the ms-swift serving environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Sequence


def validate_model_directory(
    model_path: Path,
    *,
    compute_sha256: bool = False,
) -> dict[str, Any]:
    """Validate a local Hugging Face/ModelScope model snapshot.

    Args:
        model_path: Directory containing model config and weight files.
        compute_sha256: Compute one SHA256 digest per weight shard. Enable this
            for migration manifests, not routine service restarts.

    Returns:
        JSON-friendly model inventory.

    Raises:
        FileNotFoundError: If required files or indexed shards are missing.
        ValueError: If the weight index is malformed or no weights exist.
    """

    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_auxiliary = ("tokenizer_config.json", "preprocessor_config.json")
    missing_auxiliary = [
        name for name in required_auxiliary
        if not (model_path / name).is_file()
    ]
    tokenizer_payloads = (
        "tokenizer.json",
        "vocab.json",
        "tokenizer.model",
    )
    if not any((model_path / name).is_file() for name in tokenizer_payloads):
        missing_auxiliary.append("tokenizer.json|vocab.json|tokenizer.model")
    if missing_auxiliary:
        raise FileNotFoundError(
            f"incomplete tokenizer/processor files: missing={missing_auxiliary}"
        )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(dict(index.get("weight_map") or {}).values()))
        if not shard_names:
            raise ValueError(f"weight index contains no shards: {index_path}")
    else:
        shard_names = sorted(path.name for path in model_path.glob("*.safetensors"))
        if not shard_names:
            raise ValueError(f"no safetensors weights found under {model_path}")

    missing = [name for name in shard_names if not (model_path / name).is_file()]
    empty = [name for name in shard_names if (model_path / name).is_file() and (model_path / name).stat().st_size <= 0]
    if missing or empty:
        raise FileNotFoundError(f"incomplete model weights: missing={missing}, empty={empty}")

    report = {
        "model_path": str(model_path.resolve()),
        "model_type": config.get("model_type", ""),
        "architectures": list(config.get("architectures") or []),
        "auxiliary_files": sorted(
            path.name
            for path in model_path.iterdir()
            if path.is_file()
            and path.name in {
                "tokenizer_config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "generation_config.json",
                "chat_template.json",
            }
        ),
        "weight_shards": shard_names,
        "weight_bytes": sum((model_path / name).stat().st_size for name in shard_names),
    }
    if compute_sha256:
        report["weight_sha256"] = {
            name: _sha256_file(model_path / name)
            for name in shard_names
        }
    return report


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute a streaming SHA256 digest for one model shard.

    Args:
        path: File to hash.
        chunk_bytes: Read chunk size used to bound memory consumption.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime(require_gpu: bool = True) -> dict[str, Any]:
    """Validate serving dependencies and optional CUDA availability.

    Args:
        require_gpu: Require at least one CUDA device when true.

    Returns:
        Package versions and CUDA inventory.

    Raises:
        RuntimeError: If a dependency is unavailable or CUDA is required but
            cannot be used.
    """

    versions: dict[str, str] = {}
    for name in ("swift", "transformers", "vllm", "torch"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f"required serving dependency is unavailable: {name}: {exc}") from exc
        versions[name] = str(getattr(module, "__version__", "unknown"))

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError("CUDA is unavailable; vLLM Qwen-VL serving requires a visible GPU")
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {"versions": versions, "cuda_available": cuda_available, "cuda_devices": devices}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Qwen-VL model directory")
    parser.add_argument("--skip-gpu", action="store_true", help="Only validate files and Python dependencies")
    parser.add_argument("--model-only", action="store_true", help="Validate model files without importing serving dependencies")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Include per-shard SHA256 digests for a migration manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight validation and print one JSON report.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    report = {
        "model": validate_model_directory(
            args.model,
            compute_sha256=bool(args.sha256),
        )
    }
    if not args.model_only:
        report["runtime"] = validate_runtime(require_gpu=not args.skip_gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
