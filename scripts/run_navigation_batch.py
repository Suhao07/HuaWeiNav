#!/usr/bin/env python3
"""Run frozen navigation conditions once and preserve completed or interrupted evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("conditions"), list):
        raise ValueError("manifest requires a conditions list")
    rows = {}
    for condition in manifest["conditions"]:
        if not isinstance(condition, dict) or any(not isinstance(condition.get(k), dict)
                                                  for k in ("identity", "protocol", "versions")):
            raise ValueError("condition requires identity/protocol/versions objects")
        if any(not isinstance(condition["identity"].get(k), str) or not condition["identity"][k]
               for k in ("benchmark", "split", "scene_id")):
            raise ValueError("identity requires benchmark/split/scene_id strings")
        key = _digest({k: condition[k] for k in ("identity", "protocol", "versions")})
        if key in rows:
            raise ValueError(f"duplicate condition_key: {key}")
        command = condition.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(a, str) for a in command):
            raise ValueError("command must be a nonempty argv list")
        if not isinstance(condition.get("assets", {}), dict):
            raise ValueError("assets must be an absolute-path to SHA256/null object")
        for name in (condition.get("result_file", "run_result.json"), condition.get("score_file")):
            if name is not None and (not isinstance(name, str) or not name or Path(name).is_absolute()
                                     or ".." in Path(name).parts or name == "."):
                raise ValueError("artifact files must be relative paths within run/")
        if condition.get("result_file", "run_result.json") is None or not isinstance(
                condition.get("completed_status", "completed"), str):
            raise ValueError("result_file and completed_status must be strings")
        rows[key] = condition
    _canonical(manifest)
    return rows, hashlib.sha256(raw).hexdigest()


def _preflight(assets):
    """Check existence/checksums only; episode runtime must record actual scene loading."""
    for name, expected in assets.items():
        path = Path(name)
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"asset missing or not absolute: {name}")
        if expected is not None:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if not isinstance(expected, str) or digest.hexdigest() != expected.lower():
                raise ValueError(f"asset SHA256 mismatch: {name}")


def _state(folder, key):
    """Treat damaged or orphaned evidence as blocked, never as pending."""
    if not folder.exists() and not folder.is_symlink():
        return {"status": "pending"}
    if not (folder / "receipt.json").exists():
        return {"status": "orphaned_output", "error": "output exists without receipt; inspect manually"}
    try:
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        if (not isinstance(receipt, dict) or receipt.get("condition_key") != key
                or not isinstance(receipt.get("status"), str) or receipt["status"] == "pending"):
            raise ValueError("invalid receipt identity/status")
        _canonical(receipt)
        return receipt
    except (OSError, ValueError) as error:
        return {"status": "receipt_error", "error": str(error)}


def _execute(folder, key, condition, manifest_hash):
    folder.mkdir()
    output = folder / "run"
    command = [arg.replace("{output}", str(output)) for arg in condition["command"]]
    receipt = dict(condition, condition_key=key, manifest_sha256=manifest_hash, status="running",
                   executed_command=command, output=str(output), exit_code=None, result=None, score=None,
                   started_at=time.time(), asset_preflight="existence/sha256 only; not scene-load validation")
    _atomic_json(folder / "receipt.json", receipt)
    failure = "asset_error"
    try:
        _preflight(condition.get("assets", {}))
        failure = "execution_error"
        # The episode runner owns creation of run/ (some require exist_ok=False).
        with (folder / "process.log").open("x", encoding="utf-8") as log:
            receipt["exit_code"] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        # Preserve available artifacts even when the subprocess exits nonzero.
        for field, name in (("result", condition.get("result_file", "run_result.json")),
                            ("score", condition.get("score_file"))):
            if name is not None:
                try:
                    artifact = json.loads((output / name).read_text(encoding="utf-8"))
                    _canonical(artifact)  # Reject NaN/Infinity before committing the receipt.
                    receipt[field] = artifact
                except (OSError, ValueError) as error:
                    receipt.setdefault("artifact_errors", {})[field] = str(error)
        if receipt["exit_code"] != 0:
            raise ValueError(f"process exited {receipt['exit_code']}")
        result = receipt["result"]
        if not isinstance(result, dict) or result.get("status") != condition.get("completed_status", "completed"):
            raise ValueError("result missing/invalid or runtime did not report completed")
        failure = "scoring_error"
        if condition.get("score_file") is not None:
            score = receipt["score"]
            if not isinstance(score, dict):
                raise ValueError("required score missing/invalid")
            if score.get("unavailable_metrics") or score.get("aggregate_ready") is False:
                raise ValueError("score unavailable_metrics nonempty or aggregate_ready=false")
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status=failure, error=f"{type(error).__name__}: {error}")
    receipt["finished_at"] = time.time()
    _atomic_json(folder / "receipt.json", receipt)


def _summary(rows, output, manifest_hash):
    """Partition every unique key; do not recompute or pool navigation metrics."""
    report = dict(manifest_sha256=manifest_hash, expected=0, completed=0, pending=0, errors=0,
                  unresolved=0, conditions=[], groups=[])
    groups = {}
    for key, condition in rows.items():
        receipt = _state(output / key, key)
        status = receipt["status"]
        bucket = status if status in ("completed", "pending") else (
            "errors" if status.endswith("error") else "unresolved")
        group = dict(benchmark=condition["identity"]["benchmark"], split=condition["identity"]["split"],
                     protocol=condition["protocol"], versions=condition["versions"])
        counts = groups.setdefault(_digest(group), dict(group, expected=0, completed=0, pending=0, errors=0, unresolved=0))
        for target in (report, counts):
            target["expected"] += 1
            target[bucket] += 1
        report["conditions"].append(dict(condition_key=key, identity=condition["identity"],
            protocol=condition["protocol"], versions=condition["versions"], status=status, bucket=bucket,
            error=receipt.get("error"), receipt_file=str(output / key / "receipt.json")))
    report["groups"] = list(groups.values())
    report["coverage_complete"] = report["expected"] == report["completed"]
    return report


def main(argv=None):
    """Run pending conditions or summarize evidence; return 1 for errors/unresolved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="maximum new conditions attempted this invocation")
    args = parser.parse_args(argv)
    if args.limit is not None and (args.limit < 0 or args.action != "run"):
        parser.error("--limit must be nonnegative and used with run")
    try:
        rows, manifest_hash = _load_manifest(args.manifest)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".batch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("batch is locked by another runner") from None
            attempted = 0
            if args.action == "run":
                for key, condition in rows.items():
                    if args.limit is not None and attempted >= args.limit:
                        break
                    if (output / key).exists() or (output / key).is_symlink():
                        continue  # Protect completed, running, errors and orphaned output alike.
                    _execute(output / key, key, condition, manifest_hash)
                    attempted += 1
            report = _summary(rows, output, manifest_hash)
            _atomic_json(output / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["errors"] or report["unresolved"]))
    except (OSError, ValueError) as error:
        parser.exit(2, f"batch error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
