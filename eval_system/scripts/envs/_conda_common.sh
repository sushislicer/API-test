#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"
API_CONDA_ENVS_DIR="${API_CONDA_ENVS_DIR:-${API_ROOT}/.conda-envs}"
API_CONDA_PKGS_DIR="${API_CONDA_PKGS_DIR:-${API_ROOT}/.conda-pkgs}"
API_PIP_CACHE_DIR="${API_PIP_CACHE_DIR:-${API_ROOT}/.pip-cache}"
CONDA_CREATE_LOCK="${CONDA_CREATE_LOCK:-${API_ROOT}/.conda-create.lock}"

if [[ -z "${CONDA_PKGS_DIRS:-}" ]]; then
  mkdir -p "${API_CONDA_PKGS_DIR}"
  export CONDA_PKGS_DIRS="${API_CONDA_PKGS_DIR}"
fi

if [[ -z "${PIP_CACHE_DIR:-}" ]]; then
  mkdir -p "${API_PIP_CACHE_DIR}"
  export PIP_CACHE_DIR="${API_PIP_CACHE_DIR}"
fi

export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

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
  local env_spec="$1"
  if conda_env_is_prefix "${env_spec}"; then
    [[ -x "$(conda_env_prefix "${env_spec}")/bin/python" ]]
  else
    conda env list | awk '{print $1}' | grep -qx "${env_spec}"
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

conda_env_label() {
  local env_spec="$1"
  if conda_env_is_prefix "${env_spec}"; then
    printf 'prefix %s\n' "$(conda_env_prefix "${env_spec}")"
  else
    printf 'name %s\n' "${env_spec}"
  fi
}

conda_activate_arg() {
  local env_spec="$1"
  if conda_env_is_prefix "${env_spec}"; then
    conda_env_prefix "${env_spec}"
  else
    printf '%s\n' "${env_spec}"
  fi
}

with_conda_create_lock() {
  local lock_path="${CONDA_CREATE_LOCK}"
  if [[ "${lock_path}" != /* ]]; then
    lock_path="${API_ROOT}/${lock_path}"
  fi
  mkdir -p "$(dirname "${lock_path}")"
  if command -v flock >/dev/null 2>&1; then
    flock "${lock_path}" "$@"
  else
    "$@"
  fi
}

create_conda_env() {
  local env_spec="$1"
  local python_version="$2"
  require_conda
  if conda_env_exists "${env_spec}"; then
    info "conda env $(conda_env_label "${env_spec}") already exists"
  elif conda_env_is_prefix "${env_spec}"; then
    local env_prefix
    env_prefix="$(conda_env_prefix "${env_spec}")"
    if [[ -e "${env_prefix}" && ! -x "${env_prefix}/bin/python" ]]; then
      die "conda prefix exists but does not contain bin/python: ${env_prefix}; remove it before retrying"
    fi
    mkdir -p "$(dirname "${env_prefix}")"
    info "creating conda env prefix '${env_prefix}' with python=${python_version}"
    with_conda_create_lock conda create -y -p "${env_prefix}" "python=${python_version}" pip
  else
    info "creating conda env name '${env_spec}' with python=${python_version}"
    with_conda_create_lock conda create -y -n "${env_spec}" "python=${python_version}" pip
  fi
}

run_in_env() {
  local env_spec="$1"
  shift
  if conda_env_is_prefix "${env_spec}"; then
    conda run -p "$(conda_env_prefix "${env_spec}")" "$@"
  else
    conda run -n "${env_spec}" "$@"
  fi
}

pip_in_env() {
  local env_spec="$1"
  shift
  run_in_env "${env_spec}" python -m pip "$@"
}

upgrade_pip() {
  local env_spec="$1"
  info "upgrading pip in $(conda_env_label "${env_spec}")"
  pip_in_env "${env_spec}" install --upgrade pip setuptools wheel
}

link_api_package() {
  local env_spec="$1"
  info "linking API checkout into $(conda_env_label "${env_spec}"): ${API_ROOT}"
  run_in_env "${env_spec}" env API_ROOT="${API_ROOT}" python -c 'import os, site; root=os.environ["API_ROOT"]; paths=site.getsitepackages(); target=os.path.join(paths[0], "eval_system_api.pth"); open(target, "w", encoding="utf-8").write(root + "\n"); print(target)'
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
