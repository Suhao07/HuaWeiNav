"""Own one isolated STRIVE/Isaac run, then score its actual trajectory offline."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import select
import signal
import subprocess
import sys
import time
import urllib.request
import uuid


POLICY_IDENTITY = {"decision_owner": "baseline_guided", "policy": "strive_baseline_guided_isaac"}
MODE_EQUIVALENCE = {
    "shared_policy": "baseline_guided shared pure policy with backend-specific execution",
    "backend": "Isaac RGBD and freemap-checked kinematic camera; not a dynamic robot",
    "official_isaac45_equivalent": False,
    "dynamic_robot_equivalent": False,
}


def load_source_origin(root, receipt_path=None):
    """Bind a frozen source tree to its origin receipt, without claiming a local Git checkout."""
    if receipt_path is None:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        return {"git_head": head, "git_dirty": bool(dirty.strip()), "git_status_porcelain": dirty}
    from scripts.serve_intentionnav import source_manifest

    receipt_path = Path(receipt_path).resolve(strict=True)
    if Path(root).resolve() in receipt_path.parents:
        raise ValueError("Source origin receipt must be outside the frozen source tree")
    payload = receipt_path.read_bytes()
    origin = json.loads(payload)
    fields = ("git_head", "git_dirty", "git_status_porcelain", "git_head_after_copy", "git_status_after_copy")
    if (not isinstance(origin, dict) or origin.get("schema") != "strive_frozen_worktree_v1"
            or origin.get("snapshot_semantics") != "verified_worktree_copy_not_git_checkout"
            or not isinstance(origin.get("source_root"), str)
            or not isinstance(origin.get("git_head"), str)
            or len(origin["git_head"]) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in origin["git_head"])
            or not isinstance(origin.get("git_status_porcelain"), str)
            or type(origin.get("git_dirty")) is not bool
            or origin["git_dirty"] != bool(origin["git_status_porcelain"].strip())
            or origin.get("git_head_after_copy") != origin["git_head"]
            or origin.get("git_status_after_copy") != origin["git_status_porcelain"]):
        raise ValueError("Invalid frozen source origin receipt")
    if origin.get("source_manifest") != source_manifest(root):
        raise ValueError("Frozen source bytes differ from their origin receipt")
    return {**{key: origin[key] for key in fields},
            "source_origin_root": origin["source_root"],
            "origin_receipt_sha256": hashlib.sha256(payload).hexdigest(),
            "git_provenance_source": "frozen_origin_receipt_not_local_git",
            "snapshot_semantics": origin["snapshot_semantics"]}


def validate_listener(pid, port):
    """Require the reported live server PID to own the bound IPv4 loopback socket."""
    sockets = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:  # Unrelated descriptor closed while enumerating.
            continue
        if target.startswith("socket:["):
            sockets.add(target[8:-1])
    rows = Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]
    if not any(parts[1] == f"0100007F:{port:04X}" and parts[3] == "0A" and parts[9] in sockets
               for parts in (row.split() for row in rows)):
        raise RuntimeError("Readiness PID does not own the loopback listening port")


def validate_ready(ready, server, port, expected_gpu, output, *, run_token, owner, inventory,
                   provenance, public_input):
    """Reject stale lifetimes, substituted files and unobserved device claims before policy."""
    from scripts.serve_intentionnav import process_identity, renderer_identity, source_manifest, validate_rgbd

    output = Path(output).resolve(strict=True)
    if (server.poll() is not None or process_identity(server.pid) != owner
            or ready["run_token"] != run_token or ready["port"] != port
            or ready["process"] != process_identity(ready["pid"])):
        raise RuntimeError("Readiness process lifetime/port/token mismatch")
    # Isaac python.sh may retain a shell parent. Accept only its live descendant
    # in the new owned session, never an unrelated PID or a same-number stale PID.
    process = ready["process"]
    if process["session_id"] != server.pid or process["start_ticks"] < owner["start_ticks"]:
        raise RuntimeError("Readiness does not belong to the owned simulator session")
    for _ in range(32):
        if process["pid"] == server.pid:
            break
        process = process_identity(process["parent_pid"])
    else:
        raise RuntimeError("Readiness process is not an owned simulator descendant")
    validate_listener(ready["pid"], port)
    if (ready["output_root"] != str(output) or ready["capture_root"] != str(output / "capture")
            or (output / "capture").resolve(strict=True) != output / "capture"):
        raise RuntimeError("Readiness output/capture root mismatch")
    if ready["source_provenance"] != provenance:
        raise RuntimeError("Readiness source provenance mismatch")
    manifest = output / "source_manifest.json"
    if (hashlib.sha256(manifest.read_bytes()).hexdigest() != provenance["source_manifest_sha256"]
            or source_manifest(output / "source") != json.loads(manifest.read_bytes())
            or hashlib.sha256((output / "public_episode.json").read_bytes()).hexdigest()
            != provenance["public_episode_sha256"]):
        raise RuntimeError("Copied source/public input changed before readiness")
    gpu = ready["gpu"]
    validate_cuda_receipt(gpu["cuda"], expected_gpu)
    renderer = gpu["renderer"]
    log = output / "kit-device.log"
    if (gpu["requested"] != expected_gpu or renderer["receipt_path"] != str(log)
            or log.resolve(strict=True) != log or type(renderer["receipt_bytes"]) is not int
            or renderer["receipt_bytes"] <= 0
            or renderer_identity(log, inventory, byte_length=renderer["receipt_bytes"]) != renderer
            or any(renderer[key] != expected_gpu[key] for key in expected_gpu)):
        raise RuntimeError("Readiness renderer UUID/PCI/log receipt mismatch")
    physics = gpu["physics"]
    if (physics["source"] != "SimulationView.device_ordinal + PhysX.get_overwrite_gpu_setting + World.get_physics_context"
            or type(physics["gpu_dynamics"]) is not bool or type(physics["cuda_ordinal"]) is not int
            or type(physics["gpu_override"]) is not int or physics["gpu_override"] not in {-1, 0, 1}
            or physics["broadphase"] not in {"GPU", "MBP", "SAP"}):
        raise RuntimeError("Readiness lacks observed physics execution evidence")
    if physics["gpu_override"] == 1 or (physics["gpu_override"] == -1 and (
            physics["gpu_dynamics"] or physics["broadphase"] == "GPU")):
        if (physics["device"] != "cuda:0" or physics.get("uuid") != expected_gpu["uuid"]
                or physics.get("logical_index") != 0 or physics["cuda_ordinal"] != 0):
            raise RuntimeError("Readiness physics UUID mismatch")
    elif physics["device"] != "cpu" or physics["cuda_ordinal"] != -1:
        raise RuntimeError("Readiness physics CPU evidence mismatch")
    observation = ready["initial_observation"]
    packet = {"frame": observation["frame"], "capture_root": ready["capture_root"], "pose": ready["start_pose"]}
    if (validate_rgbd(packet, output / "capture") != observation
            or observation["frame_id"] != ready["initial_observation_frame_id"]):
        raise RuntimeError("Readiness persisted RGBD receipt mismatch")
    pose = ready["start_pose"]
    w, x, y, z = public_input["start_rotation_quat_wxyz"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if (pose["position"] != public_input["start_position"]
            or abs(math.atan2(math.sin(pose["yaw"] - yaw), math.cos(pose["yaw"] - yaw))) > 1e-5):
        raise RuntimeError("Readiness pose differs from the public episode start")
    if server.poll() is not None or process_identity(ready["pid"]) != ready["process"]:
        raise RuntimeError("Simulator exited or changed lifetime during readiness validation")


def validate_cuda_receipt(receipt, expected):
    """Require the isolated CUDA driver's UUID, not a requested ordinal or setting."""
    if (receipt["source"] != "CUDA driver cuDeviceGetUuid" or receipt["uuid"] != expected["uuid"]
            or type(receipt["device_count"]) is not int or receipt["device_count"] != 1
            or type(receipt["logical_index"]) is not int or receipt["logical_index"] != 0):
        raise RuntimeError("Actual CUDA identity differs from requested GPU")


def run_policy_child():
    """Wrap the real policy in-process; attest its CUDA allocator use, never create probe work."""
    import runpy
    from scripts.serve_intentionnav import cuda_identity

    sys.argv = sys.argv[1:]
    output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve(strict=True)
    expected = {"uuid": os.environ["STRIVE_POLICY_GPU_UUID"]}
    before = cuda_identity()
    validate_cuda_receipt(before, expected)
    runpy.run_path(sys.argv[0], run_name="__main__")
    after = cuda_identity()
    validate_cuda_receipt(after, expected)
    # This is evidence of CUDA allocation by this policy process, not proof that
    # every operation ran on GPU or that the separately hosted local LLM did.
    torch = sys.modules.get("torch")
    if (torch is None or not torch.cuda.is_initialized() or torch.cuda.current_device() != 0
            or torch.cuda.device_count() != 1):
        raise RuntimeError("Policy has no initialized isolated CUDA execution context")
    peak = torch.cuda.max_memory_allocated(0)
    if type(peak) is not int or peak <= 0:
        raise RuntimeError("Policy did not record actual CUDA allocator use")
    receipt = {"cuda": after, "peak_allocated_bytes": peak, "pid": os.getpid(),
               "run_token": os.environ["STRIVE_RUN_TOKEN"], "output_root": str(output),
               "source": "policy process CUDA driver + PyTorch peak allocator",
               "policy_result_sha256": hashlib.sha256((output / "policy_result.json").read_bytes()).hexdigest()}
    with (output / "policy_gpu.json").open("x") as stream:
        json.dump(receipt, stream, allow_nan=False)


def validate_policy_result(policy, output, expected_gpu, run_token, *, task_spec_sha256=None):
    """Check policy identity, fixed-task integrity and CUDA evidence before scoring."""
    if policy["status"] != "completed":
        raise RuntimeError("Policy did not complete a bounded episode")
    if any(policy.get(key) != value for key, value in POLICY_IDENTITY.items()):
        raise RuntimeError("Policy result identity does not match baseline_guided contract")
    tier = policy["policy"]
    if (policy.get("implementation_tier") != "shared_baseline_guided"
            or ("tier" in policy and policy["tier"] != tier)):
        raise RuntimeError("Policy result tier disagrees with its shared policy identity")
    manifest = json.loads((output / "source_manifest.json").read_bytes())
    hashes = policy.get("source_sha256", {})
    if (not {"navigation/guided.py", "backends/isaac/runtime.py"}.issubset(hashes)
            or any(manifest.get(name) != value for name, value in hashes.items())):
        raise RuntimeError("Policy source provenance differs from the copied source manifest")
    public = json.loads((output / "public_episode.json").read_bytes())
    if any(policy.get(name) != value for name, value in public.items()):
        raise RuntimeError("Policy result belongs to a different public episode")
    if policy.get("task_spec_sha256") != task_spec_sha256:
        raise RuntimeError("Policy fixed task hash differs from the expected task_spec_sha256")
    if task_spec_sha256 is not None:
        from instruction.compiler import load_fixed_task
        from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES

        if (policy.get("interpretation_locked") is not True
                or type(policy.get("initial_model_calls")) is not int
                or policy["initial_model_calls"] != 0):
            raise RuntimeError("Policy fixed task requires locked interpretation and zero initial model calls")
        task = load_fixed_task(output / "task_spec.json", task_spec_sha256,
                               public["instruction"], RUNTIME_CONSTRAINT_TYPES)
        plan = json.loads((output / "instruction_plan.json").read_bytes())
        if plan.get("diagnostics", {}).get("task_spec") != task.as_dict():
            raise RuntimeError("Policy saved interpretation differs from the fixed task")
        if policy.get("task_revision") != task.revision:
            raise RuntimeError("Policy final task_revision differs from the fixed task")
    receipt = json.loads((output / "policy_gpu.json").read_bytes())
    validate_cuda_receipt(receipt["cuda"], expected_gpu)
    if (receipt["source"] != "policy process CUDA driver + PyTorch peak allocator"
            or receipt["run_token"] != run_token or receipt["output_root"] != str(output)
            or type(receipt["pid"]) is not int or receipt["pid"] <= 0
            or type(receipt["peak_allocated_bytes"]) is not int or receipt["peak_allocated_bytes"] <= 0
            or receipt["policy_result_sha256"] != hashlib.sha256((output / "policy_result.json").read_bytes()).hexdigest()):
        raise RuntimeError("Policy GPU execution receipt is missing, stale or mismatched")
    return tier, receipt


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(port, action, run_token):
    """Bind a bounded shutdown request to the validated server's launch attempt."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/command",
        data=json.dumps({"action": action, "run_token": run_token}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _matches_process(identity):
    from scripts.serve_intentionnav import process_identity

    try:
        current = process_identity(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        return False
    # 父进程退出后的重新托管不改变子进程的生命周期。
    return all(current[key] == identity[key] for key in ("pid", "start_ticks", "boot_id", "session_id"))


def collect_owned_processes(owner, ready_process=None):
    from scripts.serve_intentionnav import process_identity

    anchors = {p["pid"]: p for p in (owner, ready_process) if p is not None}
    if owner["session_id"] != owner["pid"] or not any(_matches_process(p) for p in anchors.values()):
        raise RuntimeError("Owned session has no live captured anchor; descendant cleanup is unverified")
    owned = dict(anchors)
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            identity = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (identity["session_id"] == owner["pid"] and identity["boot_id"] == owner["boot_id"]
                and identity["start_ticks"] >= owner["start_ticks"]):
            owned.setdefault(identity["pid"], identity)
    if not any(_matches_process(p) for p in anchors.values()):
        raise RuntimeError("Owned session anchor exited during descendant discovery")
    return owned


def terminate_owned_processes(owned):
    errors, signaled = [], []
    with ExitStack() as stack:
        pending = {}
        for pid, identity in owned.items():
            try:
                fd = os.pidfd_open(pid)
                stack.callback(os.close, fd)
                if _matches_process(identity):
                    pending[fd] = identity
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(str(error))
        for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            for fd, identity in pending.items():
                try:
                    if _matches_process(identity):
                        signal.pidfd_send_signal(fd, sig)
                        if identity["pid"] not in signaled:
                            signaled.append(identity["pid"])
                except ProcessLookupError:
                    pass
                except OSError as error:
                    errors.append(str(error))
            deadline = time.monotonic() + timeout
            while pending:
                exited, _, _ = select.select(list(pending), [], [], max(0, deadline - time.monotonic()))
                for fd in exited:
                    pending.pop(fd)
                if not exited or time.monotonic() >= deadline:
                    break
        if pending:
            errors.append("Owned processes did not exit within TERM/KILL deadlines")
    if errors:
        raise RuntimeError("; ".join(errors))
    return signaled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--selection-id", default=None)
    source.add_argument("--public-episode", type=Path)
    parser.add_argument("--style")
    parser.add_argument("--port", type=int, default=9466)
    parser.add_argument("--sim-gpu", type=int, default=7)
    parser.add_argument("--renderer-gpu", type=int, help="optional Kit/Vulkan ordinal, verified by UUID")
    parser.add_argument("--policy-gpu", type=int, default=5)
    parser.add_argument("--model-port", type=int, default=18182)
    parser.add_argument("--image", default="strive-hm3d:local")
    parser.add_argument("--max-moves", type=int, default=30)
    parser.add_argument("--max-model-calls", type=int, default=60)
    parser.add_argument("--decision-owner", choices=("baseline_guided",), default="baseline_guided")
    parser.add_argument("--search-review-moves", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-origin", type=Path, help="external origin receipt for a frozen source tree")
    parser.add_argument("--task-spec", type=Path, help="audited fixed TaskSpec JSON file")
    parser.add_argument("--task-spec-sha256", help="expected file-byte SHA256; not proof of semantic correctness")
    args = parser.parse_args()
    if (args.task_spec is None) != (args.task_spec_sha256 is None):
        parser.error("--task-spec and --task-spec-sha256 must be supplied together")
    if args.selection_id is None and args.public_episode is None:
        args.selection_id = "SEL_001"
    if args.renderer_gpu is not None and args.renderer_gpu < 0:
        parser.error("renderer-gpu must be nonnegative")
    if min(args.max_moves, args.max_model_calls, args.search_review_moves) < 1:
        parser.error("episode budgets must be positive")
    if min(args.sim_gpu, args.policy_gpu) < 0:
        parser.error("physical GPU indices must be nonnegative")
    if not all(1 <= port <= 65535 for port in (args.port, args.model_port)) or args.port == args.model_port:
        parser.error("simulator/model ports must be distinct and between 1 and 65535")
    if args.public_episode is not None and args.style is not None:
        parser.error("--style cannot override --public-episode")
    root = Path(os.environ["STRIVE_ROOT"]).resolve()
    data_root = Path(os.environ["INTENTIONNAV_DATA_ROOT"]).resolve()
    official_root = Path(os.environ["INTENTIONNAV_ROOT"]).resolve()
    isaac_root = Path(os.environ["ISAACSIM_ROOT"]).resolve()
    sys.path.insert(0, str(root))
    from benchmark.intentionnav import load_episode, score_trajectory
    from scripts.serve_intentionnav import (gpu_identity, gpu_inventory, load_public_episode,
                                           process_identity, source_manifest)

    episode = None
    binding = {"status": "matched", "source": "official_dataset"}
    if args.public_episode is None:
        episode = load_episode(data_root, args.selection_id, "formal" if args.style is None else args.style)
        public_input = episode.policy_input()
        run_label = episode.selection_id
    else:
        public_input = load_public_episode(args.public_episode)
        run_label = "public-" + hashlib.sha256(
            json.dumps(public_input, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        try:
            candidate = load_episode(data_root, public_input["selection_id"], public_input["style"])
        except (OSError, ValueError) as error:
            binding = {"status": "unavailable", "reason": "official_episode_unavailable: " + str(error)}
        else:
            original = json.loads(json.dumps(candidate.policy_input()))
            mismatched = [field for field in original if original[field] != public_input[field]]
            if mismatched:
                binding = {"status": "unavailable", "reason": "public_episode_mismatch",
                           "mismatched_fields": mismatched}
            else:
                episode = candidate
    if args.task_spec is not None:
        from instruction.compiler import load_fixed_task
        from instruction_adapter.compiler import RUNTIME_CONSTRAINT_TYPES

        load_fixed_task(args.task_spec, args.task_spec_sha256,
                        public_input["instruction"], RUNTIME_CONSTRAINT_TYPES)
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    inventory = gpu_inventory()
    simulator_gpu = gpu_identity(args.sim_gpu, inventory)
    policy_gpu = gpu_identity(args.policy_gpu, inventory)
    origin = load_source_origin(root, args.source_origin)
    provenance = dict(origin)
    run_id = "strive-" + run_label + "-guided-isaac6-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else data_root / "results" / run_id
    output.mkdir(exist_ok=False)
    server = owner = server_ready = None
    owned_server = policy_started = policy_finished = False
    result = {"run_id": run_id, "status": "starting", **POLICY_IDENTITY, "source_provenance": provenance,
              "requested_devices": {"simulator": simulator_gpu, "policy": policy_gpu},
              "evaluation_binding": binding, "mode_equivalence": MODE_EQUIVALENCE,
              "task_spec_sha256": args.task_spec_sha256,
              "motion_semantics": "freemap_checked_kinematic_camera_not_dynamic_robot"}
    try:
        print(f"RUN_ROOT={output}", flush=True)
        source = output / "source"
        source.mkdir()
        packages = ("artifact_utils", "backends", "benchmark", "cv_utils", "harness", "instruction",
                    "instruction_adapter", "llm_utils", "mapping", "mapping_utils", "motion", "navigation",
                    "navigation_core", "perception", "planning", "prior_map", "prompting", "scene", "scripts",
                    "task", "verification", "skills", "configs")
        for name in packages:
            shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in root.glob("*.py"):
            shutil.copy2(path, source / path.name)
        public = output / "public_episode.json"
        public.write_text(json.dumps(public_input, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                          encoding="utf-8")
        load_public_episode(public)
        public_hash = hashlib.sha256(public.read_bytes()).hexdigest()
        result["public_episode_sha256"] = public_hash
        if args.task_spec is not None:
            fixed_task = output / "task_spec.json"
            fixed_task.write_bytes(args.task_spec.read_bytes())
            load_fixed_task(fixed_task, args.task_spec_sha256,
                            json.loads(public.read_bytes())["instruction"], RUNTIME_CONSTRAINT_TYPES)
        manifest = output / "source_manifest.json"
        write_json(manifest, source_manifest(source))
        provenance["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if args.source_origin is None:
            provenance["snapshot_semantics"] = "copied_worktree_bytes_not_atomic_git_checkout"
            provenance["git_head_after_copy"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            provenance["git_status_after_copy"] = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True)
        elif (load_source_origin(root, args.source_origin) != origin
                or json.loads(manifest.read_bytes()) != source_manifest(root)):
            raise RuntimeError("Frozen source changed during launch copy")
        for key, filename in (("simulator_source", "iss_env.py"), ("walkable_map_source", "walkable_map.py")):
            path = (official_root / "eval/simulator" / filename).resolve(strict=True)
            provenance[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        expected_provenance = {"source_root": str(source),
            "source_manifest_sha256": provenance["source_manifest_sha256"], "public_episode_sha256": public_hash,
            "simulator_source": provenance["simulator_source"], "walkable_map_source": provenance["walkable_map_source"]}
        run_token = uuid.uuid4().hex
        server_env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                          CUDA_VISIBLE_DEVICES=simulator_gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            server_env.pop(name, None)
        server_cmd = [str(isaac_root / "python.sh"), str(source / "scripts/serve_intentionnav.py"),
                      "--public-episode", str(public), "--data-root", str(data_root), "--output", str(output),
                      "--port", str(args.port), "--gpu", str(args.sim_gpu), "--gpu-uuid", simulator_gpu["uuid"],
                      "--run-token", run_token]
        if args.renderer_gpu is not None:
            server_cmd += ["--renderer-gpu", str(args.renderer_gpu)]
        ready = output / "server_ready.json"
        with (output / "simulator.log").open("w") as simulator_log:
            server = subprocess.Popen(server_cmd, cwd=source, env=server_env, stdout=simulator_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            owner = process_identity(server.pid)
            deadline = time.monotonic() + 300
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"Isaac exited before readiness: {server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Isaac readiness exceeded 300 seconds")
                time.sleep(0.5)
            server_ready = json.loads(ready.read_text())
            validate_ready(server_ready, server, args.port, simulator_gpu, output, run_token=run_token,
                           owner=owner, inventory=inventory, provenance=expected_provenance,
                           public_input=json.loads(public.read_bytes()))
            if server.poll() is not None:
                raise RuntimeError("Isaac exited immediately after readiness")
            owned_server = True
            result["server"] = server_ready
            print(json.dumps({"event": "simulator_ready", **server_ready}), flush=True)
            env = dict(os.environ)
            settings = {
                "STRIVE_LLM_CLIENT": "self_hosted", "LLM_PROVIDER": "self_hosted",
                "VLN_LVLM_BASE_URL": f"http://127.0.0.1:{args.model_port}/v1", "VLN_LVLM_MODEL": "huaweivln-qwen3.5-4b",
                "VLN_LVLM_API_KEY": os.environ["VLN_LVLM_API_KEY"], "VLN_LVLM_GUIDED_JSON": "1",
                "VLN_LVLM_TIMEOUT_S": "180", "VLN_LVLM_PARSE_RETRIES": "1", "STRIVE_VLM_WORKERS": "1",
                "VLN_LVLM_TRANSPORT_RETRIES": "0",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                "CUDA_VISIBLE_DEVICES": policy_gpu["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "STRIVE_POLICY_GPU_UUID": policy_gpu["uuid"], "STRIVE_RUN_TOKEN": run_token,
                "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_HOME": "/root/.cache/huggingface",
                "SAM_CHECKPOINT": "/weights/sam_vit_h_4b8939.pth", "GROUNDING_DINO_PATH": "/opt/mmdetection",
                "GROUNDING_DINO_CONFIG": "/opt/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py",
                "GROUNDING_DINO_CHECKPOINT": "/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
                "LLM_OFFLINE": "0", "STRIVE_LLM_FALLBACK": "0", "STRIVE_EVIDENCE_REUSE": "0",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
            env.update(settings)
            docker = ["docker", "run", "--rm", "--name", run_id, "--gpus", f"device={policy_gpu['uuid']}",
                      "--network", "host", "--shm-size=12g", "--entrypoint", "/bin/bash"]
            for name in settings:
                docker += ["-e", name]
            docker += ["-v", f"{output}:{output}", "-v", f"{source}:{source}:ro",
                       "-v", f"{os.environ['HF_HOME_HOST']}:/root/.cache/huggingface:ro",
                       "-v", f"{os.environ['SAM_CHECKPOINT']}:{settings['SAM_CHECKPOINT']}:ro",
                       "-v", f"{os.environ['GROUNDING_DINO_CHECKPOINT']}:{settings['GROUNDING_DINO_CHECKPOINT']}:ro",
                       "-w", str(source), image_id, "-lc", 'exec "$@"', "bash",
                       "python", "-c", "from scripts.launch_intentionnav_episode import run_policy_child; run_policy_child()",
                       "scripts/run_intentionnav.py", "--device", "cuda:0", "--public-episode", str(public),
                       "--output", str(output), "--port", str(args.port),
                       "--max-model-calls", str(args.max_model_calls), "--max-moves", str(args.max_moves),
                       "--panorama-views", "12", "--decision-owner", args.decision_owner,
                       "--search-review-moves", str(args.search_review_moves)]
            if args.task_spec is not None:
                docker += ["--task-spec", str(fixed_task), "--task-spec-sha256", args.task_spec_sha256]
            write_json(output / "execution.json", {"run_id": run_id, "image_id": image_id,
                       "task_spec_sha256": args.task_spec_sha256,
                       **POLICY_IDENTITY, "mode_equivalence": MODE_EQUIVALENCE, "source_provenance": provenance,
                       "requested_devices": result["requested_devices"],
                       "public_episode_sha256": public_hash, "evaluation_binding": binding,
                       "server_command": server_cmd, "policy_command": docker,
                       "policy_env_names": sorted(settings),
                       "policy_mounts_exclude_dataset_and_scene": True})
            policy_started = True
            with (output / "policy.log").open("w") as policy_log:
                process = subprocess.run(docker, env=env, stdout=policy_log, stderr=subprocess.STDOUT, timeout=7200)
            policy_finished = True
            result["policy_exit_code"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f"Policy process exited {process.returncode}; see policy.log")
            policy = json.loads((output / "policy_result.json").read_text())
            tier, policy_device = validate_policy_result(policy, output, policy_gpu, run_token,
                                                         task_spec_sha256=args.task_spec_sha256)
            if server.poll() is not None or process_identity(server_ready["pid"]) != server_ready["process"]:
                raise RuntimeError("Simulator exited or changed lifetime before policy completion")
            if source_manifest(source) != json.loads(manifest.read_bytes()):
                raise RuntimeError("Copied source changed during the episode")
            result["policy_gpu"] = policy_device
            result["tier"] = tier
            result["stop_receipt"] = policy["stop_receipt"]
            trajectory_bytes = (output / "trajectory.jsonl").read_bytes()
            trajectory_hash = hashlib.sha256(trajectory_bytes).hexdigest()
            if trajectory_hash != policy["stop_receipt"]["trajectory_sha256"]:
                raise RuntimeError("Scoring trajectory differs from the policy's final STOP receipt")
            trajectory = [json.loads(line) for line in trajectory_bytes.splitlines()]
            final_pose = policy["final_frame"]["pose"]
            if (trajectory[-1]["action"] != "STOP" or trajectory[-1]["position"] != final_pose["position"]
                    or trajectory[-1]["yaw"] != final_pose["yaw"]):
                raise RuntimeError("Final observation and STOP trajectory disagree")
            if episode is not None:
                scored = score_trajectory(episode, trajectory, official_root,
                    prediction_target=policy["predicted_target"], tier=tier,
                    model=settings["VLN_LVLM_MODEL"], radius_m=2.0, dataset_root=data_root / "dataset",
                    scene_summary_root=data_root / "metadata/kujiale_scene_summary")
                scored.update(episode_meta=episode._evaluation_meta, dataset_sha256=episode.source_sha256)
            else:
                names = ("IM", "SR_hit", "OSR_hit", "SPL", "G", "GSR_hit")
                scored = {"metrics": dict.fromkeys(names), "aggregate_ready": False,
                          "unavailable_metrics": dict.fromkeys(names, binding["reason"]),
                          "protocol": {"name": "unscored_public_episode"}}
            result["trajectory_sha256"] = trajectory_hash
            scene = data_root / "scenes" / public_input["scene_id"] / "start_result_navigation.usd"
            freemap = data_root / "metadata/metadata_train" / public_input["scene_id"] / "freemap.npy"
            write_json(output / "evaluation.json", {**scored, **POLICY_IDENTITY, "tier": tier,
                "task_spec_sha256": args.task_spec_sha256,
                "mode_equivalence": MODE_EQUIVALENCE, "source_provenance": provenance,
                "public_episode_sha256": public_hash, "evaluation_binding": binding,
                "trajectory_sha256": trajectory_hash, "trajectory_steps": len(trajectory),
                "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                "freemap_sha256": hashlib.sha256(freemap.read_bytes()).hexdigest(),
                "policy_termination": policy["reason"], "task_complete": policy["task_complete"]})
            result.update(status="completed", metrics=scored["metrics"],
                          aggregate_ready=scored["aggregate_ready"], termination=policy["reason"],
                          task_complete=policy["task_complete"], model_calls=policy["model_calls"], moves=policy["moves"])
    except BaseException as error:
        result.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        execution_error = sys.exc_info()[1]
        cleanup_errors = []
        owned = {p["pid"]: p for p in (owner, server_ready["process"] if owned_server else None) if p is not None}
        try:
            if owner is not None:
                owned = collect_owned_processes(owner, server_ready["process"] if owned_server else None)
        except Exception as error:
            cleanup_errors.append({"stage": "owned_process_discovery", "error": str(error)})
        try:
            if policy_started and not policy_finished:
                stopped = subprocess.run(["docker", "stop", "--time", "10", run_id], timeout=30, check=False)
                if stopped.returncode != 0:
                    raise RuntimeError(f"docker stop exited {stopped.returncode}")
        except Exception as error:
            cleanup_errors.append({"stage": "docker_stop", "error": str(error)})
        try:
            if owned_server and server.poll() is None:
                if (process_identity(server.pid) != owner
                        or process_identity(server_ready["pid"]) != server_ready["process"]):
                    raise RuntimeError("Shutdown process lifetime changed")
                validate_listener(server_ready["pid"], args.port)
                if process_identity(server_ready["pid"]) != server_ready["process"]:
                    raise RuntimeError("Shutdown process lifetime changed during listener validation")
                result["shutdown_receipt"] = command(args.port, "shutdown", run_token)
                exit_code = server.wait(timeout=60)
                if exit_code != 0:
                    raise RuntimeError(f"Simulator exited {exit_code} after shutdown")
        except Exception as error:
            cleanup_errors.append({"stage": "simulator_shutdown", "error": str(error)})
        try:
            if owner is not None:
                terminated = terminate_owned_processes(owned)
                result["server_required_termination"] = bool(terminated)
                result["terminated_pids"] = terminated
            if server is not None:
                result["server_exit_code"] = server.poll()
        except Exception as error:
            cleanup_errors.append({"stage": "owned_process_termination", "error": str(error)})
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            if execution_error is None:
                result.update(status="error", error_type="CleanupError", error=str(cleanup_errors))
        try:
            write_json(output / "run_result.json", result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as error:
            if execution_error is None:
                raise
            print(f"Failed to persist run result: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if cleanup_errors and execution_error is None:
            raise RuntimeError(f"Run cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    main()
