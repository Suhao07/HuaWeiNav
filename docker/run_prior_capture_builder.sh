#!/usr/bin/env bash
set -euo pipefail
STRIVE_ROOT="${STRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 使用现有镜像和权重配置，独立于 episode 数据与在线模型凭证。
python3 - "$STRIVE_ROOT" "$@" <<'PY'
import os
from pathlib import Path
import sys
import yaml
root = Path(sys.argv[1])
config = yaml.safe_load(Path(os.environ.get("STRIVE_WEIGHTS_CONFIG", root / "configs/strive_weights.yaml")).read_text())
sam = Path(os.environ.get("SAM_CHECKPOINT", config["sam_checkpoint"]))
dino = Path(os.environ.get("GROUNDING_DINO_CHECKPOINT", config["grounding_dino_checkpoint"]))
for path in (sam, dino):
    if not path.is_file():
        raise FileNotFoundError(path)
args = ["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
    "--shm-size=4g", "-v", f"{root}:/workspace/STRIVE",
    "-v", f"{sam}:/weights/sam.pth:ro", "-v", f"{dino}:/weights/dino.pth:ro",
    "-v", f"{Path.home()}/.cache/huggingface:/hf:ro"]
for key, value in {"PYTHONNOUSERSITE": "1", "HF_HOME": "/hf", "TRANSFORMERS_OFFLINE": "1",
    "SAM_CHECKPOINT": "/weights/sam.pth", "GROUNDING_DINO_CHECKPOINT": "/weights/dino.pth",
    "GROUNDING_DINO_PATH": config["grounding_dino_path_container"],
    "GROUNDING_DINO_CONFIG": config["grounding_dino_config_container"]}.items():
    args += ["-e", f"{key}={value}"]
args += [os.environ.get("IMAGE_TAG", "strive-hm3d:local"), "bash", "-lc",
    'cd /workspace/STRIVE && python scripts/build_prior_capture.py "$@"', "bash", *sys.argv[2:]]
os.execvp(args[0], args)
PY
