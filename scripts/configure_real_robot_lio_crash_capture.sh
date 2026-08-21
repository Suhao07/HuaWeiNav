#!/usr/bin/env bash
set -euo pipefail

# Configure crash evidence for only the supplied Point-LIO process.  This
# deliberately does not change kernel.core_pattern or any other workspace.
PID="${1:-${POINT_LIO_PID:-}}"
OUT_DIR="${LIO_CRASH_CAPTURE_DIR:-${PWD}/logs/diagnostics/lio-crash}"
mkdir -p "${OUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE="${OUT_DIR}/configure-${STAMP}.txt"

{
  echo "timestamp_utc=${STAMP}"
  echo "pid=${PID:-<not supplied>}"
  echo "host=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "shell_core_limit=$(ulimit -c)"
  echo "core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo unavailable)"
  if command -v coredumpctl >/dev/null 2>&1; then
    echo "coredumpctl=present"
  else
    echo "coredumpctl=absent"
  fi
  if [[ -n "${PID}" && -d "/proc/${PID}" ]]; then
    echo "process=$(tr '\0' ' ' < "/proc/${PID}/cmdline")"
    echo "proc_limits_core=$(awk '/Max core file size/ {print}' "/proc/${PID}/limits")"
    if command -v prlimit >/dev/null 2>&1; then
      prlimit --pid "${PID}" --core
    fi
    # This is a per-process setting; the system-wide limit and apport routing
    # remain unchanged.
    if command -v prlimit >/dev/null 2>&1; then
      prlimit --pid "${PID}" --core=unlimited
      echo "per_process_core_limit=unlimited"
      prlimit --pid "${PID}" --core
    fi
  fi
  echo "crash_directories="
  find /var/crash /var/lib/apport/coredump -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null || true
} | tee "${EVIDENCE}"

cat > "${OUT_DIR}/gdb-postmortem-command.txt" <<'EOF'
# Fill CORE and EXE after a crash; this is intentionally not run automatically.
gdb -q -batch \
  -ex 'set pagination off' \
  -ex 'thread apply all bt full' \
  -ex 'info registers' \
  -ex 'quit' \
  "$EXE" "$CORE"
EOF

echo "evidence=${EVIDENCE}"
echo "postmortem_command=${OUT_DIR}/gdb-postmortem-command.txt"
