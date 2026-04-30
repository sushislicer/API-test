#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "[env-setup] $*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_conda() {
  require_command conda
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

create_conda_env() {
  local env_name="$1"
  local python_version="$2"
  require_conda
  if conda_env_exists "${env_name}"; then
    info "conda env '${env_name}' already exists"
  else
    info "creating conda env '${env_name}' with python=${python_version}"
    conda create -y -n "${env_name}" "python=${python_version}" pip
  fi
}

run_in_env() {
  local env_name="$1"
  shift
  conda run -n "${env_name}" "$@"
}

pip_in_env() {
  local env_name="$1"
  shift
  run_in_env "${env_name}" python -m pip "$@"
}

upgrade_pip() {
  local env_name="$1"
  info "upgrading pip in '${env_name}'"
  pip_in_env "${env_name}" install --upgrade pip setuptools wheel
}

link_api_package() {
  local env_name="$1"
  info "linking API checkout into '${env_name}': ${API_ROOT}"
  conda run -n "${env_name}" env API_ROOT="${API_ROOT}" python -c 'import os, site; root=os.environ["API_ROOT"]; paths=site.getsitepackages(); target=os.path.join(paths[0], "eval_system_api.pth"); open(target, "w", encoding="utf-8").write(root + "\n"); print(target)'
}

default_repo_path() {
  local candidate="$1"
  if [[ -d "${candidate}" ]]; then
    cd "${candidate}" && pwd
  fi
  return 0
}

assert_repo_dir() {
  local name="$1"
  local path="$2"
  [[ -n "${path}" ]] || die "${name} repo path is required"
  [[ -d "${path}" ]] || die "${name} repo path does not exist: ${path}"
}
