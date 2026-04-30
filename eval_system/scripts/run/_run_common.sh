#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "error: $*" >&2
  exit 1
}

run_python() {
  local env_name="$1"
  shift
  if [[ -n "${env_name}" ]]; then
    command -v conda >/dev/null 2>&1 || die "conda is required when an env name is provided"
    conda run --no-capture-output -n "${env_name}" env PYTHONPATH="${API_ROOT}:${PYTHONPATH:-}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" "$@"
  else
    PYTHONPATH="${API_ROOT}:${PYTHONPATH:-}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" "$@"
  fi
}
