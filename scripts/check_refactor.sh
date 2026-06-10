#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_strive_refactor}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

python -m py_compile \
  config_utils.py \
  constants.py \
  mapper_with_process_obs.py \
  objnav_agent_with_process_obs.py \
  objnav_benchmark_with_process_obs.py \
  cv_utils/gpt_utils.py \
  cv_utils/image_perceiver.py \
  cv_utils/sam.py \
  cv_utils/stitch.py \
  cv_utils/visualizer.py \
  artifact_utils/*.py \
  instruction_adapter/*.py \
  llm_utils/*.py \
  mapping/*.py \
  mapping_utils/*.py \
  navigation/*.py \
  navigation_core/*.py \
  planning/*.py

PYTHONPATH=. pytest -q tests

echo "refactor compile checks passed"
