#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${LINGBOT_VA_ENV_NAME:-env-lingbot-va}"
PYTHON_VERSION="${LINGBOT_VA_PYTHON_VERSION:-3.10.16}"
LINGBOT_REPO="${LINGBOT_VA_REPO:-$(default_repo_path "${API_ROOT}/models/lingbot-va")}"
INSTALL_TORCH=1
INSTALL_FLASH_ATTN=1
INSTALL_REQUIREMENTS=0
INSTALL_EDITABLE=1
INSTALL_POSTTRAIN=0
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
  --env NAME              Conda environment name. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing LingBot-VA checkout. Default: ${LINGBOT_REPO:-<none>}
  --torch-index URL      PyTorch wheel index. Default: ${TORCH_INDEX_URL}
  --skip-torch           Do not install the README torch/cu126 wheel set.
  --requirements         Install requirements.txt exactly instead of the README package list.
  --skip-flash-attn      Do not install flash-attn separately.
  --post-training        Also install LingBot-VA post-training extras.
  --no-editable          Skip pip install -e on the LingBot-VA checkout.
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
      INSTALL_REQUIREMENTS=1
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

create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
upgrade_pip "${ENV_NAME}"

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  info "installing LingBot-VA torch/cu126 wheel set"
  pip_in_env "${ENV_NAME}" install "${TORCH_PACKAGES[@]}" --index-url "${TORCH_INDEX_URL}"
else
  info "skipping torch install"
fi

if [[ "${INSTALL_REQUIREMENTS}" -eq 1 ]]; then
  [[ -f "${LINGBOT_REPO}/requirements.txt" ]] || die "missing ${LINGBOT_REPO}/requirements.txt"
  info "installing LingBot-VA requirements.txt"
  pip_in_env "${ENV_NAME}" install -r "${LINGBOT_REPO}/requirements.txt"
else
  info "installing LingBot-VA README package set"
  pip_in_env "${ENV_NAME}" install "${BASE_PACKAGES[@]}"
fi

if [[ "${INSTALL_FLASH_ATTN}" -eq 1 ]]; then
  info "installing flash-attn with --no-build-isolation"
  pip_in_env "${ENV_NAME}" install flash-attn --no-build-isolation
else
  info "skipping flash-attn install"
fi

if [[ "${INSTALL_POSTTRAIN}" -eq 1 ]]; then
  info "installing LingBot-VA post-training extras"
  pip_in_env "${ENV_NAME}" install lerobot==0.3.3 scipy wandb --no-deps
fi

if [[ "${INSTALL_EDITABLE}" -eq 1 ]]; then
  info "installing LingBot-VA editable package without changing installed pins"
  pip_in_env "${ENV_NAME}" install --no-deps -e "${LINGBOT_REPO}"
else
  info "skipping editable LingBot-VA install"
fi

link_api_package "${ENV_NAME}"

info "done. Run with: conda activate ${ENV_NAME}"
info "for this API adapter, set LINGBOT_VA_ROOT=${LINGBOT_REPO} or pass task.metadata.repo_path"
info "put LingBot-VA checkpoints under ${DOWNLOAD_ROOT}/checkpoints/lingbot-va or set DOWNLOAD_ROOT to another disk"
