#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${LDA_1B_ENV_NAME:-${API_CONDA_ENVS_DIR}/api-lda-1b}"
PYTHON_VERSION="${LDA_1B_PYTHON_VERSION:-3.10}"
LDA_REPO="${LDA_1B_REPO:-$(default_repo_path "${API_ROOT}/models/LDA-1B")}"
INSTALL_REQUIREMENTS=1
INSTALL_TORCH=1
INSTALL_FLASH_ATTN=1
INSTALL_EDITABLE=1
RUN_VALIDATION=1
LINK_API=1
REPAIR_REQUIREMENTS=0
REPAIR_FLASH_ATTN=0
BUILD_FLASH_ATTN=0
REQUIRE_CUDA=0
FLASH_ATTN_MAX_JOBS="${LDA_1B_FLASH_ATTN_MAX_JOBS:-${FLASH_ATTN_MAX_JOBS:-4}}"
FLASH_ATTN_NVCC_THREADS="${LDA_1B_FLASH_ATTN_NVCC_THREADS:-${FLASH_ATTN_NVCC_THREADS:-1}}"
TORCH_INDEX_URL="${LDA_1B_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGES=(torch==2.6.0 torchvision==0.21.0)

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for LDA-1B inference.

Options:
  --env ENV               Conda env name or prefix path. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing LDA-1B checkout. Default: ${LDA_REPO:-<none>}
  --torch-index URL      PyTorch wheel index. Default: ${TORCH_INDEX_URL}
  --skip-torch           Do not preinstall the pinned LDA PyTorch CUDA wheel set.
  --skip-requirements    Do not install requirements.txt.
  --skip-flash-attn      Do not install flash-attn separately.
  --no-editable          Skip pip install -e on the LDA-1B checkout.
  --skip-validation      Do not run import/native-server help checks after install.
  --validate-only        Only run validation against an existing env.
  --repair-requirements  Repair a partial env by skipping torch and running the remaining LDA install.
  --repair-flash-attn    Reinstall flash-attn without changing torch, then validate.
  --build-flash-attn     With --repair-flash-attn, force a local source build.
  --flash-attn-jobs N    Max parallel jobs for flash-attn builds. Default: ${FLASH_ATTN_MAX_JOBS}
  --flash-attn-nvcc-threads N
                          NVCC threads per flash-attn compile job. Default: ${FLASH_ATTN_NVCC_THREADS}
  --require-cuda         Fail validation unless torch can see CUDA.
  -h, --help             Show this help.

Default mode follows the LDA-1B README installation:
  pip install torch==2.6.0 torchvision==0.21.0 --index-url ${TORCH_INDEX_URL}
  pip install -r requirements.txt
  pip install flash-attn --no-build-isolation
  pip install --no-deps -e .

This script does not download LDA checkpoints, Qwen checkpoints, or DINO checkpoints.
Default artifact cache: ${DOWNLOAD_ROOT}/checkpoints/lda-1b
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
    --torch-index)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      shift
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
    --skip-validation)
      RUN_VALIDATION=0
      shift
      ;;
    --validate-only)
      INSTALL_TORCH=0
      INSTALL_REQUIREMENTS=0
      INSTALL_FLASH_ATTN=0
      INSTALL_EDITABLE=0
      LINK_API=0
      RUN_VALIDATION=1
      shift
      ;;
    --repair-requirements)
      INSTALL_TORCH=0
      INSTALL_REQUIREMENTS=1
      INSTALL_FLASH_ATTN=1
      INSTALL_EDITABLE=1
      LINK_API=1
      REPAIR_REQUIREMENTS=1
      RUN_VALIDATION=1
      shift
      ;;
    --repair-flash-attn)
      INSTALL_TORCH=0
      INSTALL_REQUIREMENTS=0
      INSTALL_FLASH_ATTN=1
      INSTALL_EDITABLE=0
      LINK_API=0
      REPAIR_FLASH_ATTN=1
      RUN_VALIDATION=1
      shift
      ;;
    --build-flash-attn)
      BUILD_FLASH_ATTN=1
      shift
      ;;
    --flash-attn-jobs)
      FLASH_ATTN_MAX_JOBS="$2"
      shift 2
      ;;
    --flash-attn-nvcc-threads)
      FLASH_ATTN_NVCC_THREADS="$2"
      shift 2
      ;;
    --require-cuda)
      REQUIRE_CUDA=1
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
[[ -f "${LDA_REPO}/deployment/model_server/server_policy.py" ]] || die "LDA-1B repo is missing deployment/model_server/server_policy.py: ${LDA_REPO}"

validate_lda_env() {
  info "validating LDA-1B environment imports"
  local cuda_arg=()
  if [[ "${REQUIRE_CUDA}" -eq 1 ]]; then
    cuda_arg=(--require-cuda)
  fi
  run_in_env "${ENV_NAME}" python "${SCRIPT_DIR}/validate_lda_1b_env.py" \
    --api-root "${API_ROOT}" \
    --repo "${LDA_REPO}" \
    --env "${ENV_NAME}" \
    "${cuda_arg[@]}"
  info "validating LDA-1B native server CLI import"
  run_in_env "${ENV_NAME}" bash -lc "cd '${LDA_REPO}' && python -m deployment.model_server.server_policy --help >/dev/null"
}

if ! conda_env_exists "${ENV_NAME}" && [[ "${INSTALL_TORCH}" -eq 0 && "${INSTALL_REQUIREMENTS}" -eq 0 && "${INSTALL_FLASH_ATTN}" -eq 0 && "${INSTALL_EDITABLE}" -eq 0 ]]; then
  die "conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if ! conda_env_exists "${ENV_NAME}" && [[ "${REPAIR_REQUIREMENTS}" -eq 1 ]]; then
  die "cannot repair requirements because conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if ! conda_env_exists "${ENV_NAME}" && [[ "${REPAIR_FLASH_ATTN}" -eq 1 ]]; then
  die "cannot repair flash-attn because conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if [[ "${INSTALL_TORCH}" -eq 1 || "${INSTALL_REQUIREMENTS}" -eq 1 || "${INSTALL_FLASH_ATTN}" -eq 1 || "${INSTALL_EDITABLE}" -eq 1 ]]; then
  create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
  upgrade_pip "${ENV_NAME}"
else
  info "skipping install steps"
fi

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  info "installing LDA-1B pinned torch/cu124 wheel set"
  pip_in_env "${ENV_NAME}" install "${TORCH_PACKAGES[@]}" --index-url "${TORCH_INDEX_URL}"
else
  info "skipping torch install"
fi

if [[ "${INSTALL_REQUIREMENTS}" -eq 1 ]]; then
  [[ -f "${LDA_REPO}/requirements.txt" ]] || die "missing ${LDA_REPO}/requirements.txt"
  info "installing LDA-1B requirements"
  pip_in_env "${ENV_NAME}" install -r "${LDA_REPO}/requirements.txt"
else
  info "skipping requirements.txt install"
fi

if [[ "${INSTALL_FLASH_ATTN}" -eq 1 ]]; then
  if [[ "${REPAIR_FLASH_ATTN}" -eq 1 ]]; then
    info "removing existing flash-attn before repair"
    pip_in_env "${ENV_NAME}" uninstall -y flash-attn || true
  fi
  info "installing flash-attn with --no-build-isolation and without changing torch"
  flash_attn_env=(
    env
    "MAX_JOBS=${FLASH_ATTN_MAX_JOBS}"
    "NVCC_THREADS=${FLASH_ATTN_NVCC_THREADS}"
    "CMAKE_BUILD_PARALLEL_LEVEL=${FLASH_ATTN_MAX_JOBS}"
    "MAKEFLAGS=-j${FLASH_ATTN_MAX_JOBS}"
    PIP_PROGRESS_BAR=off
    PIP_DISABLE_PIP_VERSION_CHECK=1
    PIP_ROOT_USER_ACTION=ignore
    NO_COLOR=1
    CLICOLOR=0
    TERM=dumb
  )
  flash_attn_args=(install flash-attn --no-build-isolation --no-deps --progress-bar off)
  if [[ "${REPAIR_FLASH_ATTN}" -eq 1 ]]; then
    flash_attn_args+=(--force-reinstall --no-cache-dir)
  fi
  if [[ "${BUILD_FLASH_ATTN}" -eq 1 ]]; then
    flash_attn_env+=(FLASH_ATTENTION_FORCE_BUILD=TRUE)
    flash_attn_args+=(--no-binary flash-attn)
  fi
  run_in_env "${ENV_NAME}" "${flash_attn_env[@]}" python -m pip "${flash_attn_args[@]}"
else
  info "skipping flash-attn install"
fi

if [[ "${INSTALL_EDITABLE}" -eq 1 ]]; then
  info "installing LDA-1B editable package without changing installed pins"
  pip_in_env "${ENV_NAME}" install --no-deps -e "${LDA_REPO}"
else
  info "skipping editable LDA-1B install"
fi

if [[ "${LINK_API}" -eq 1 ]]; then
  link_api_package "${ENV_NAME}"
fi

if [[ "${RUN_VALIDATION}" -eq 1 ]]; then
  validate_lda_env
else
  info "skipping validation"
fi

info "done. Run with: conda activate $(conda_activate_arg "${ENV_NAME}")"
info "for this API adapter, set LDA_1B_ROOT=${LDA_REPO} or pass task.metadata.repo_path"
info "put LDA checkpoints under ${DOWNLOAD_ROOT}/checkpoints/lda-1b or set DOWNLOAD_ROOT to another disk"
