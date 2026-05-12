#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${LIBERO_ENV_NAME:-${API_CONDA_ENVS_DIR}/api-libero}"
PYTHON_VERSION="${LIBERO_PYTHON_VERSION:-3.8.13}"
LIBERO_REPO="${LIBERO_REPO:-$(default_repo_path "${API_ROOT}/simulators/LIBERO")}"
INSTALL_TORCH=1
TORCH_SPEC=(torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113)

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for the official LIBERO simulator.

Options:
  --env ENV           Conda env name or prefix path. Default: ${ENV_NAME}
  --python VERSION   Python version. Default: ${PYTHON_VERSION}
  --repo PATH        Existing LIBERO checkout. Default: ${LIBERO_REPO:-<none>}
  --skip-torch       Do not install the official torch/cu113 wheel set.
  -h, --help         Show this help.

This script does not download LIBERO datasets. Run the official
benchmark_scripts/download_libero_datasets.py command yourself if needed.
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
      LIBERO_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=0
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

assert_repo_dir "LIBERO" "${LIBERO_REPO}"
[[ -f "${LIBERO_REPO}/requirements.txt" ]] || die "missing ${LIBERO_REPO}/requirements.txt"

create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
upgrade_pip "${ENV_NAME}"

info "installing LIBERO requirements"
pip_in_env "${ENV_NAME}" install -r "${LIBERO_REPO}/requirements.txt"

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  info "installing official LIBERO torch wheel set"
  pip_in_env "${ENV_NAME}" install "${TORCH_SPEC[@]}"
else
  info "skipping torch install"
fi

info "installing LIBERO editable package"
pip_in_env "${ENV_NAME}" install -e "${LIBERO_REPO}"

link_api_package "${ENV_NAME}"

info "done. Run with: conda activate $(conda_activate_arg "${ENV_NAME}")"
