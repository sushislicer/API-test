#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"
API_CONDA_ENVS_DIR="${API_CONDA_ENVS_DIR:-${API_ROOT}/.conda-envs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

die() {
  echo "error: $*" >&2
  exit 1
}

run_python() {
  local env_spec="$1"
  shift
  if [[ -n "${env_spec}" ]]; then
    command -v conda >/dev/null 2>&1 || die "conda is required when an env is provided"
    if conda_env_is_prefix "${env_spec}"; then
      conda run --no-capture-output -p "$(conda_env_prefix "${env_spec}")" env PYTHONPATH="${API_ROOT}:${PYTHONPATH:-}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" "$@"
    else
      conda run --no-capture-output -n "${env_spec}" env PYTHONPATH="${API_ROOT}:${PYTHONPATH:-}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" "$@"
    fi
  else
    PYTHONPATH="${API_ROOT}:${PYTHONPATH:-}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" "$@"
  fi
}

conda_env_is_prefix() {
  local env_spec="$1"
  [[ "${env_spec}" == /* || "${env_spec}" == ./* || "${env_spec}" == ../* || "${env_spec}" == */* ]]
}

conda_env_prefix() {
  local env_spec="$1"
  if [[ "${env_spec}" == /* ]]; then
    printf '%s\n' "${env_spec}"
  else
    printf '%s\n' "${API_ROOT}/${env_spec}"
  fi
}
