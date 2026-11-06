"""Launch one isolated GOAT parent episode with the local Qwen service."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--model-port", type=int, default=18182)
    parser.add_argument("--image", default="strive-hm3d:local")
    parser.add_argument("policy_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    root = Path(os.environ["STRIVE_ROOT"]).resolve(strict=True)
    data = args.data_root.absolute()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    policy_args = args.policy_args[1:] if args.policy_args[:1] == ["--"] else args.policy_args
    settings = {
        "STRIVE_LLM_CLIENT": "self_hosted", "LLM_PROVIDER": "self_hosted",
        "VLN_LVLM_BASE_URL": f"http://127.0.0.1:{args.model_port}/v1",
        "VLN_LVLM_MODEL": "huaweivln-qwen3.5-4b",
        "VLN_LVLM_API_KEY": os.environ["VLN_LVLM_API_KEY"], "VLN_LVLM_GUIDED_JSON": "1",
        "VLN_LVLM_TIMEOUT_S": "180", "VLN_LVLM_PARSE_RETRIES": "1", "STRIVE_VLM_WORKERS": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "HF_HOME": "/root/.cache/huggingface", "SAM_CHECKPOINT": "/weights/sam_vit_h_4b8939.pth",
        "GROUNDING_DINO_PATH": "/opt/mmdetection",
        "GROUNDING_DINO_CONFIG": "/opt/mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py",
        "GROUNDING_DINO_CHECKPOINT": "/weights/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth",
        "LLM_OFFLINE": "0", "STRIVE_LLM_FALLBACK": "0", "STRIVE_EVIDENCE_REUSE": "0",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet",
    }
    run_id = "strive-goat-sequence-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    command = ["docker", "run", "--rm", "--name", run_id, "--gpus", f"device={args.gpu}",
               "--network", "host", "--shm-size=12g", "--entrypoint", "/bin/bash"]
    for name in settings:
        command += ["-e", name]
    for mount in (
        f"{root}:/workspace/STRIVE:ro", f"{data}:{data}:ro",
        f"{output}:{output}", f"{os.environ['HF_HOME_HOST']}:/root/.cache/huggingface:ro",
        f"{os.environ['SAM_CHECKPOINT']}:{settings['SAM_CHECKPOINT']}:ro",
        f"{os.environ['GROUNDING_DINO_CHECKPOINT']}:{settings['GROUNDING_DINO_CHECKPOINT']}:ro",
    ):
        command += ["-v", mount]
    command += ["-w", "/workspace/STRIVE", image_id, "-lc", 'exec "$@"', "bash",
                "python", "scripts/run_goat.py", *policy_args, "--output", str(output / "episode")]
    receipt = {
        "run_id": run_id, "command": command,
        "environment": {k: v for k, v in settings.items() if "KEY" not in k},
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in root.rglob("*.py") if "__pycache__" not in p.parts},
        "image_id": image_id,
    }
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "policy.log").open("x") as stream:
        process = subprocess.run(command, env=dict(os.environ, **settings), stdout=stream,
                                 stderr=subprocess.STDOUT)
    (output / "process_result.json").write_text(json.dumps({
        "run_id": run_id, "exit_code": process.returncode,
        "status": "execution_completed" if process.returncode == 0 else "execution_error",
    }, indent=2) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
