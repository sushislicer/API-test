#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${LDA_1B_ENV_NAME:-env-lda-1b}"
PYTHON_VERSION="${LDA_1B_PYTHON_VERSION:-3.10}"
LDA_REPO="${LDA_1B_REPO:-$(default_repo_path "${API_ROOT}/models/LDA-1B")}"
INSTALL_REQUIREMENTS=1
INSTALL_FLASH_ATTN=1
INSTALL_EDITABLE=1

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for LDA-1B inference.

Options:
  --env NAME              Conda environment name. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing LDA-1B checkout. Default: ${LDA_REPO:-<none>}
  --skip-requirements    Do not install requirements.txt.
  --skip-flash-attn      Do not install flash-attn separately.
  --no-editable          Skip pip install -e on the LDA-1B checkout.
  -h, --help             Show this help.

Default mode follows the LDA-1B README installation:
  pip install -r requirements.txt
  pip install flash-attn --no-build-isolation
  pip install --no-deps -e .

This script does not download LDA checkpoints, Qwen checkpoints, or DINO checkpoints.
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
      LDA_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --skip-requirements)
      INSTALL_REQUIREMENTS=0
      shift
      ;;
    --skip-flash-attn)
      INSTALL_FLASH_ATTN=0
      shift
      ;;
    --no-editable)
      INSTALL_EDITABLE=0
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

assert_repo_dir "LDA-1B" "${LDA_REPO}"

create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
upgrade_pip "${ENV_NAME}"

if [[ "${INSTALL_REQUIREMENTS}" -eq 1 ]]; then
  [[ -f "${LDA_REPO}/requirements.txt" ]] || die "missing ${LDA_REPO}/requirements.txt"
  info "installing LDA-1B requirements"
  pip_in_env "${ENV_NAME}" install -r "${LDA_REPO}/requirements.txt"
else
  info "skipping requirements.txt install"
fi

if [[ "${INSTALL_FLASH_ATTN}" -eq 1 ]]; then
  info "installing flash-attn with --no-build-isolation"
  pip_in_env "${ENV_NAME}" install flash-attn --no-build-isolation
else
  info "skipping flash-attn install"
fi

if [[ "${INSTALL_EDITABLE}" -eq 1 ]]; then
  info "installing LDA-1B editable package without changing installed pins"
  pip_in_env "${ENV_NAME}" install --no-deps -e "${LDA_REPO}"
else
  info "skipping editable LDA-1B install"
fi

link_api_package "${ENV_NAME}"

info "done. Run with: conda activate ${ENV_NAME}"
info "for this API adapter, set LDA_1B_ROOT=${LDA_REPO} or pass task.metadata.repo_path"
