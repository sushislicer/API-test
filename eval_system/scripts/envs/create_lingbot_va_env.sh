#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${LINGBOT_VA_ENV_NAME:-${API_CONDA_ENVS_DIR}/api-lingbot-va}"
PYTHON_VERSION="${LINGBOT_VA_PYTHON_VERSION:-3.10.16}"
LINGBOT_REPO="${LINGBOT_VA_REPO:-$(default_repo_path "${API_ROOT}/models/lingbot-va")}"
INSTALL_TORCH=1
INSTALL_FLASH_ATTN=1
INSTALL_DEPS=1
INSTALL_REQUIREMENTS=0
INSTALL_EDITABLE=1
INSTALL_POSTTRAIN=0
RUN_VALIDATION=1
LINK_API=1
REPAIR_REQUIREMENTS=0
REPAIR_FLASH_ATTN=0
BUILD_FLASH_ATTN=0
REQUIRE_CUDA=0
FLASH_ATTN_MAX_JOBS="${LINGBOT_VA_FLASH_ATTN_MAX_JOBS:-${FLASH_ATTN_MAX_JOBS:-4}}"
FLASH_ATTN_NVCC_THREADS="${LINGBOT_VA_FLASH_ATTN_NVCC_THREADS:-${FLASH_ATTN_NVCC_THREADS:-1}}"
TORCH_INDEX_URL="${LINGBOT_VA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_PACKAGES=(torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0)
BASE_PACKAGES=(
  websockets
  einops
  diffusers==0.36.0
  transformers==4.55.2
  accelerate
  msgpack
  opencv-python
  matplotlib
  ftfy
  easydict
  numpy==1.26.4
  tqdm
  "imageio[ffmpeg]"
  safetensors
  Pillow
)

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for LingBot-VA inference.

Options:
  --env ENV               Conda env name or prefix path. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing LingBot-VA checkout. Default: ${LINGBOT_REPO:-<none>}
  --torch-index URL      PyTorch wheel index. Default: ${TORCH_INDEX_URL}
  --skip-torch           Do not install the README torch/cu126 wheel set.
  --requirements         Install requirements.txt exactly instead of the README package list.
  --skip-requirements    Do not install the README package set or requirements.txt.
  --skip-flash-attn      Do not install flash-attn separately.
  --post-training        Also install LingBot-VA post-training extras.
  --no-editable          Skip linking the LingBot-VA checkout into site-packages.
  --skip-validation      Do not run import/native-server help checks after install.
  --validate-only        Only run validation against an existing env.
  --repair-requirements  Repair a partial env by skipping torch and running the remaining LingBot install.
  --repair-flash-attn    Reinstall flash-attn without changing torch, then validate.
  --build-flash-attn     With --repair-flash-attn, force a local source build.
  --flash-attn-jobs N    Max parallel jobs for flash-attn builds. Default: ${FLASH_ATTN_MAX_JOBS}
  --flash-attn-nvcc-threads N
                          NVCC threads per flash-attn compile job. Default: ${FLASH_ATTN_NVCC_THREADS}
  --require-cuda         Fail validation unless torch can see CUDA.
  -h, --help             Show this help.

Default mode follows the LingBot-VA README installation:
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url ${TORCH_INDEX_URL}
  pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
  pip install flash-attn --no-build-isolation

This script does not download LingBot-VA checkpoints or datasets.
Default artifact cache: ${DOWNLOAD_ROOT}/checkpoints/lingbot-va
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
      LINGBOT_REPO="$(cd "$2" && pwd)"
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
    --requirements)
      INSTALL_DEPS=1
      INSTALL_REQUIREMENTS=1
      shift
      ;;
    --skip-requirements)
      INSTALL_DEPS=0
      shift
      ;;
    --skip-flash-attn)
      INSTALL_FLASH_ATTN=0
      shift
      ;;
    --post-training)
      INSTALL_POSTTRAIN=1
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
      INSTALL_FLASH_ATTN=0
      INSTALL_DEPS=0
      INSTALL_REQUIREMENTS=0
      INSTALL_EDITABLE=0
      INSTALL_POSTTRAIN=0
      LINK_API=0
      RUN_VALIDATION=1
      shift
      ;;
    --repair-requirements)
      INSTALL_TORCH=0
      INSTALL_FLASH_ATTN=1
      INSTALL_DEPS=1
      INSTALL_EDITABLE=1
      LINK_API=1
      REPAIR_REQUIREMENTS=1
      RUN_VALIDATION=1
      shift
      ;;
    --repair-flash-attn)
      INSTALL_TORCH=0
      INSTALL_FLASH_ATTN=1
      INSTALL_DEPS=0
      INSTALL_REQUIREMENTS=0
      INSTALL_EDITABLE=0
      INSTALL_POSTTRAIN=0
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

assert_repo_dir "LingBot-VA" "${LINGBOT_REPO}"
[[ -f "${LINGBOT_REPO}/wan_va/wan_va_server.py" ]] || die "LingBot-VA repo is missing wan_va/wan_va_server.py: ${LINGBOT_REPO}"

validate_lingbot_env() {
  info "validating LingBot-VA environment imports"
  local cuda_arg=()
  if [[ "${REQUIRE_CUDA}" -eq 1 ]]; then
    cuda_arg=(--require-cuda)
  fi
  run_in_env "${ENV_NAME}" python "${SCRIPT_DIR}/validate_lingbot_va_env.py" \
    --api-root "${API_ROOT}" \
    --repo "${LINGBOT_REPO}" \
    --env "${ENV_NAME}" \
    "${cuda_arg[@]}"
  info "validating LingBot-VA native server CLI import"
  run_in_env "${ENV_NAME}" bash -lc "cd '${LINGBOT_REPO}' && python wan_va/wan_va_server.py --help >/dev/null"
}

link_lingbot_repo() {
  info "linking LingBot-VA checkout into $(conda_env_label "${ENV_NAME}"): ${LINGBOT_REPO}"
  run_in_env "${ENV_NAME}" env LINGBOT_REPO="${LINGBOT_REPO}" python -c 'import os, site; root=os.environ["LINGBOT_REPO"]; paths=site.getsitepackages(); target=os.path.join(paths[0], "lingbot_va_repo.pth"); open(target, "w", encoding="utf-8").write(root + "\n"); print(target)'
}

if ! conda_env_exists "${ENV_NAME}" && [[ "${INSTALL_TORCH}" -eq 0 && "${INSTALL_DEPS}" -eq 0 && "${INSTALL_FLASH_ATTN}" -eq 0 && "${INSTALL_EDITABLE}" -eq 0 && "${INSTALL_POSTTRAIN}" -eq 0 ]]; then
  die "conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if ! conda_env_exists "${ENV_NAME}" && [[ "${REPAIR_REQUIREMENTS}" -eq 1 ]]; then
  die "cannot repair requirements because conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if ! conda_env_exists "${ENV_NAME}" && [[ "${REPAIR_FLASH_ATTN}" -eq 1 ]]; then
  die "cannot repair flash-attn because conda env $(conda_env_label "${ENV_NAME}") does not exist"
fi

if [[ "${INSTALL_TORCH}" -eq 1 || "${INSTALL_DEPS}" -eq 1 || "${INSTALL_FLASH_ATTN}" -eq 1 || "${INSTALL_EDITABLE}" -eq 1 || "${INSTALL_POSTTRAIN}" -eq 1 ]]; then
  create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
  upgrade_pip "${ENV_NAME}"
else
  info "skipping install steps"
fi

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  info "installing LingBot-VA torch/cu126 wheel set"
  pip_in_env "${ENV_NAME}" install "${TORCH_PACKAGES[@]}" --index-url "${TORCH_INDEX_URL}"
else
  info "skipping torch install"
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  if [[ "${INSTALL_REQUIREMENTS}" -eq 1 ]]; then
    [[ -f "${LINGBOT_REPO}/requirements.txt" ]] || die "missing ${LINGBOT_REPO}/requirements.txt"
    info "installing LingBot-VA requirements.txt"
    pip_in_env "${ENV_NAME}" install -r "${LINGBOT_REPO}/requirements.txt"
  else
    info "installing LingBot-VA README package set"
    pip_in_env "${ENV_NAME}" install "${BASE_PACKAGES[@]}"
  fi
else
  info "skipping LingBot-VA package set"
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

if [[ "${INSTALL_POSTTRAIN}" -eq 1 ]]; then
  info "installing LingBot-VA post-training extras"
  pip_in_env "${ENV_NAME}" install lerobot==0.3.3 scipy wandb --no-deps
fi

if [[ "${INSTALL_EDITABLE}" -eq 1 ]]; then
  link_lingbot_repo
else
  info "skipping LingBot-VA repo link"
fi

if [[ "${LINK_API}" -eq 1 ]]; then
  link_api_package "${ENV_NAME}"
fi

if [[ "${RUN_VALIDATION}" -eq 1 ]]; then
  validate_lingbot_env
else
  info "skipping validation"
fi

info "done. Run with: conda activate $(conda_activate_arg "${ENV_NAME}")"
info "for this API adapter, set LINGBOT_VA_ROOT=${LINGBOT_REPO} or pass task.metadata.repo_path"
info "put LingBot-VA checkpoints under ${DOWNLOAD_ROOT}/checkpoints/lingbot-va or set DOWNLOAD_ROOT to another disk"
