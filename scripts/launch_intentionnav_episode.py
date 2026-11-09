"""Own one isolated STRIVE/Isaac run, then score its actual trajectory offline."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(port, action):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/command",
        data=json.dumps({"action": action}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--selection-id", default="SEL_001")
    source.add_argument("--public-episode", type=Path)
    parser.add_argument("--style")
    parser.add_argument("--port", type=int, default=9466)
    parser.add_argument("--sim-gpu", type=int, default=7)
    parser.add_argument("--policy-gpu", type=int, default=5)
    parser.add_argument("--model-port", type=int, default=18182)
    parser.add_argument("--image", default="strive-hm3d:local")
    parser.add_argument("--max-moves", type=int, default=30)
    parser.add_argument("--max-model-calls", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.public_episode is not None and args.style is not None:
        parser.error("--style cannot override --public-episode")
    root = Path(os.environ["STRIVE_ROOT"])
    data_root = Path(os.environ["INTENTIONNAV_DATA_ROOT"])
    official_root = Path(os.environ["INTENTIONNAV_ROOT"])
    isaac_root = Path(os.environ["ISAACSIM_ROOT"])
    sys.path.insert(0, str(root))
    from benchmark.intentionnav import load_episode, score_trajectory
    from scripts.serve_intentionnav import load_public_episode

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
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    run_id = "strive-" + run_label + "-guided-isaac6-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else data_root / "results" / run_id
    output.mkdir(exist_ok=False)
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
    write_json(output / "source_manifest.json", {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source.rglob("*") if p.is_file()})
    server_env = dict(os.environ, PYTHONNOUSERSITE="1")
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES"):
        server_env.pop(name, None)
    server_cmd = [str(isaac_root / "python.sh"), str(source / "scripts/serve_intentionnav.py"),
                  "--public-episode", str(public), "--data-root", str(data_root), "--output", str(output),
                  "--port", str(args.port), "--gpu", str(args.sim_gpu)]
    ready = output / "server_ready.json"
    owned_server = False
    policy_started = False
    policy_finished = False
    result = {"run_id": run_id, "status": "starting", "policy": "strive_isaac_guided",
              "public_episode_sha256": public_hash, "evaluation_binding": binding,
              "mode_equivalence": "not_default_baseline_guided_or_official_isaac45",
              "motion_semantics": "freemap_checked_kinematic_camera_not_dynamic_robot"}
    with (output / "simulator.log").open("w") as simulator_log:
        server = subprocess.Popen(server_cmd, cwd=source, env=server_env, stdout=simulator_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 300
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError(f"Isaac exited before readiness: {server.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Isaac readiness exceeded 300 seconds")
                time.sleep(0.5)
            server_ready = json.loads(ready.read_text())
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
                "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_HOME": "/root/.cache/huggingface",
                "SAM_CHECKPOINT": "/weights/sam_vit_h_4b8939.pth", "GROUNDING_DINO_PATH": "/opt/mmdetection",
                "GROUNDING_DINO_CONFIG": "/opt/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py",
                "GROUNDING_DINO_CHECKPOINT": "/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
                "LLM_OFFLINE": "0", "STRIVE_LLM_FALLBACK": "0", "STRIVE_EVIDENCE_REUSE": "0",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
            env.update(settings)
            docker = ["docker", "run", "--rm", "--name", run_id, "--gpus", f"device={args.policy_gpu}",
                      "--network", "host", "--shm-size=12g", "--entrypoint", "/bin/bash"]
            for name in settings:
                docker += ["-e", name]
            docker += ["-v", f"{output}:{output}", "-v", f"{source}:{source}:ro",
                       "-v", f"{os.environ['HF_HOME_HOST']}:/root/.cache/huggingface:ro",
                       "-v", f"{os.environ['SAM_CHECKPOINT']}:{settings['SAM_CHECKPOINT']}:ro",
                       "-v", f"{os.environ['GROUNDING_DINO_CHECKPOINT']}:{settings['GROUNDING_DINO_CHECKPOINT']}:ro",
                       "-w", str(source), image_id, "-lc", 'exec "$@"', "bash",
                       "python", "scripts/run_intentionnav.py", "--public-episode", str(public),
                       "--output", str(output), "--port", str(args.port),
                       "--max-model-calls", str(args.max_model_calls), "--max-moves", str(args.max_moves),
                       "--panorama-views", "4"]
            write_json(output / "execution.json", {"run_id": run_id, "image_id": image_id,
                       "public_episode_sha256": public_hash, "evaluation_binding": binding,
                       "server_command": server_cmd, "policy_command": docker,
                       "policy_env": {k: v for k, v in settings.items() if "KEY" not in k},
                       "policy_mounts_exclude_dataset_and_scene": True})
            policy_started = True
            with (output / "policy.log").open("w") as policy_log:
                process = subprocess.run(docker, env=env, stdout=policy_log, stderr=subprocess.STDOUT, timeout=7200)
            policy_finished = True
            result["policy_exit_code"] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f"Policy process exited {process.returncode}; see policy.log")
            policy = json.loads((output / "policy_result.json").read_text())
            if policy["status"] != "completed":
                raise RuntimeError("Policy did not complete a bounded episode")
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
                    prediction_target=policy["predicted_target"], tier="strive_isaac_guided",
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
            write_json(output / "evaluation.json", {**scored,
                "public_episode_sha256": public_hash, "evaluation_binding": binding,
                "trajectory_sha256": trajectory_hash, "trajectory_steps": len(trajectory),
                "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
                "freemap_sha256": hashlib.sha256(freemap.read_bytes()).hexdigest(),
                "policy_termination": policy["reason"], "task_complete": policy["task_complete"]})
            result.update(status="completed", metrics=scored["metrics"],
                          aggregate_ready=scored["aggregate_ready"], termination=policy["reason"],
                          task_complete=policy["task_complete"], model_calls=policy["model_calls"], moves=policy["moves"])
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if policy_started and not policy_finished:
                subprocess.run(["docker", "stop", "--time", "10", run_id], timeout=30, check=False)
            if owned_server and server.poll() is None:
                try:
                    result["shutdown_receipt"] = command(args.port, "shutdown")
                    server.wait(timeout=60)
                except Exception as error:
                    result["shutdown_error"] = str(error)
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=30)
                result["server_required_termination"] = True
            result["server_exit_code"] = server.poll()
            write_json(output / "run_result.json", result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
