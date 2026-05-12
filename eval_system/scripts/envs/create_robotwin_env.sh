#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${ROBOTWIN_ENV_NAME:-${API_CONDA_ENVS_DIR}/api-robotwin}"
PYTHON_VERSION="${ROBOTWIN_PYTHON_VERSION:-3.10}"
ROBOTWIN_REPO="${ROBOTWIN_REPO:-$(default_repo_path "${API_ROOT}/simulators/RoboTwin")}"
INSTALL_MODE="official"
DOWNLOAD_ASSETS=0
UPDATE_EMBODIMENT_PATHS=1

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for RoboTwin 2.0.

Options:
  --env ENV               Conda env name or prefix path. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing RoboTwin checkout. Default: ${ROBOTWIN_REPO:-<none>}
  --manual               Use pip requirements fallback instead of script/_install.sh.
  --download-assets      Also run script/_download_assets.sh. This can be large.
  --no-update-paths      Skip script/update_embodiment_config_path.py.
  -h, --help             Show this help.

Default mode follows the official RoboTwin installer:
  cd RoboTwin && bash script/_install.sh

The manual mode uses requirements.txt if present, otherwise script/requirements.txt.
It does not install CuRobo/pytorch3d unless those are covered by the repo files.
Default artifact cache: ${DOWNLOAD_ROOT}/assets/robotwin
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --repo)
      ROBOTWIN_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --manual)
      INSTALL_MODE="manual"
      shift
      ;;
    --download-assets)
      DOWNLOAD_ASSETS=1
      shift
      ;;
    --no-update-paths)
      UPDATE_EMBODIMENT_PATHS=0
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

assert_repo_dir "RoboTwin" "${ROBOTWIN_REPO}"
[[ -d "${ROBOTWIN_REPO}/script" ]] || die "RoboTwin repo is missing script/: ${ROBOTWIN_REPO}"

create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
upgrade_pip "${ENV_NAME}"

if [[ "${INSTALL_MODE}" == "official" ]]; then
  [[ -f "${ROBOTWIN_REPO}/script/_install.sh" ]] || die "missing ${ROBOTWIN_REPO}/script/_install.sh; try --manual"
  info "running official RoboTwin installer"
  run_in_env "${ENV_NAME}" bash -lc "cd '${ROBOTWIN_REPO}' && bash script/_install.sh"
else
  if [[ -f "${ROBOTWIN_REPO}/requirements.txt" ]]; then
    REQ_FILE="${ROBOTWIN_REPO}/requirements.txt"
  elif [[ -f "${ROBOTWIN_REPO}/script/requirements.txt" ]]; then
    REQ_FILE="${ROBOTWIN_REPO}/script/requirements.txt"
  else
    die "missing RoboTwin requirements.txt or script/requirements.txt"
  fi
  info "running manual RoboTwin requirements install: ${REQ_FILE}"
  pip_in_env "${ENV_NAME}" install -r "${REQ_FILE}"
fi

if [[ "${UPDATE_EMBODIMENT_PATHS}" -eq 1 && -f "${ROBOTWIN_REPO}/script/update_embodiment_config_path.py" ]]; then
  info "updating RoboTwin embodiment config paths"
  run_in_env "${ENV_NAME}" bash -lc "cd '${ROBOTWIN_REPO}' && python script/update_embodiment_config_path.py"
fi

if [[ "${DOWNLOAD_ASSETS}" -eq 1 ]]; then
  [[ -f "${ROBOTWIN_REPO}/script/_download_assets.sh" ]] || die "missing ${ROBOTWIN_REPO}/script/_download_assets.sh"
  info "downloading RoboTwin assets"
  run_in_env "${ENV_NAME}" bash -lc "cd '${ROBOTWIN_REPO}' && bash script/_download_assets.sh"
else
  info "skipping RoboTwin asset download"
fi

link_api_package "${ENV_NAME}"

info "done. Run with: conda activate $(conda_activate_arg "${ENV_NAME}")"
info "for this API adapter, set ROBOTWIN_ROOT=${ROBOTWIN_REPO} or pass task.metadata.repo_path"
info "use ${DOWNLOAD_ROOT}/assets/robotwin for manually cached RoboTwin assets, or set DOWNLOAD_ROOT to another disk"
