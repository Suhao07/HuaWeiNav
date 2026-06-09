#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_hm3d_baseline.sh" bash -lc 'cd /workspace/STRIVE && python docker/preflight.py'
