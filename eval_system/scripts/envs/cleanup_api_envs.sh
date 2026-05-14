#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_ROOT="${API_CONDA_ENVS_DIR}"
DO_REMOVE=0
CLEAN_NAMED=1
CLEAN_PREFIXES=1
CLEAN_PACKAGE_CACHE=0

NAMED_ENVS=(api-lda-1b api-lingbot-va api-robotwin)
PREFIX_ENVS=(api-lda-1b api-lingbot-va api-robotwin api-libero api-simplerenv)

usage() {
  cat <<EOF
Usage: $0 [options]

Remove API-owned conda environments created by these setup scripts.
By default this is a dry run. Pass --yes to delete.

Targets:
  named envs: ${NAMED_ENVS[*]}
  prefix envs under: ${ENV_ROOT}

Options:
  --yes              Actually delete. Without this, only print planned actions.
  --env-root PATH    Prefix env root. Default: ${ENV_ROOT}
  --named-only       Remove only old named conda envs.
  --prefix-only      Remove only checkout-local prefix envs.
  --keep-named       Do not remove named envs.
  --keep-prefixes    Do not remove prefix envs.
  --package-cache    Also remove ${API_ROOT}/.conda-pkgs and ${API_ROOT}/.pip-cache if present.
  -h, --help         Show this help.

This script refuses to remove arbitrary paths. Prefix deletion is limited to
known API env names directly under --env-root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      DO_REMOVE=1
      shift
      ;;
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --named-only)
      CLEAN_NAMED=1
      CLEAN_PREFIXES=0
      shift
      ;;
    --prefix-only)
      CLEAN_NAMED=0
      CLEAN_PREFIXES=1
      shift
      ;;
    --keep-named)
      CLEAN_NAMED=0
      shift
      ;;
    --keep-prefixes)
      CLEAN_PREFIXES=0
      shift
      ;;
    --package-cache)
      CLEAN_PACKAGE_CACHE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ "${ENV_ROOT}" != /* ]]; then
  ENV_ROOT="${API_ROOT}/${ENV_ROOT}"
fi

run_or_print() {
  if [[ "${DO_REMOVE}" -eq 1 ]]; then
    "$@"
  else
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  fi
}

ensure_safe_prefix_target() {
  local path="$1"
  local base
  base="$(basename "${path}")"
  [[ "$(dirname "${path}")" == "${ENV_ROOT}" ]] || die "refusing to remove path outside env root: ${path}"
  for known in "${PREFIX_ENVS[@]}"; do
    if [[ "${base}" == "${known}" ]]; then
      return 0
    fi
  done
  die "refusing to remove unknown prefix env path: ${path}"
}

if [[ "${DO_REMOVE}" -eq 0 ]]; then
  info "dry run only. Re-run with --yes to delete."
fi

if [[ "${CLEAN_NAMED}" -eq 1 ]]; then
  if [[ "${DO_REMOVE}" -eq 0 ]] && ! command -v conda >/dev/null 2>&1; then
    for env_name in "${NAMED_ENVS[@]}"; do
      run_or_print conda env remove -y -n "${env_name}"
    done
  else
    require_conda
    for env_name in "${NAMED_ENVS[@]}"; do
      if conda_env_exists "${env_name}"; then
        info "removing named conda env: ${env_name}"
        run_or_print conda env remove -y -n "${env_name}"
      else
        info "named conda env not found: ${env_name}"
      fi
    done
  fi
fi

if [[ "${CLEAN_PREFIXES}" -eq 1 ]]; then
  if [[ "${DO_REMOVE}" -eq 1 ]]; then
    require_conda
  fi
  for env_name in "${PREFIX_ENVS[@]}"; do
    env_prefix="${ENV_ROOT}/${env_name}"
    ensure_safe_prefix_target "${env_prefix}"
    if [[ ! -e "${env_prefix}" ]]; then
      info "prefix env not found: ${env_prefix}"
      continue
    fi
    if [[ -d "${env_prefix}/conda-meta" ]]; then
      if [[ "${DO_REMOVE}" -eq 1 ]]; then
        info "removing conda prefix env: ${env_prefix}"
        run_or_print conda env remove -y -p "${env_prefix}"
      else
        run_or_print conda env remove -y -p "${env_prefix}"
      fi
    fi
    if [[ -e "${env_prefix}" ]]; then
      info "removing leftover prefix directory: ${env_prefix}"
      run_or_print rm -rf -- "${env_prefix}"
    fi
  done
fi

if [[ "${CLEAN_PACKAGE_CACHE}" -eq 1 ]]; then
  for cache_path in "${API_ROOT}/.conda-pkgs" "${API_ROOT}/.pip-cache"; do
    if [[ -e "${cache_path}" ]]; then
      info "removing API-local package cache: ${cache_path}"
      run_or_print rm -rf -- "${cache_path}"
    else
      info "API-local package cache not found: ${cache_path}"
    fi
  done
fi
