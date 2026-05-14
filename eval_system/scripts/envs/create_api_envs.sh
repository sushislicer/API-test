#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_ROOT="${API_CONDA_ENVS_DIR}"
CREATE_LDA=1
CREATE_LINGBOT=1
CREATE_ROBOTWIN=1
CLEAN_STALE_NAMED=0

LDA_REPO="${LDA_1B_REPO:-$(default_repo_path "${API_ROOT}/models/LDA-1B")}"
LINGBOT_REPO="${LINGBOT_VA_REPO:-$(default_repo_path "${API_ROOT}/models/lingbot-va")}"
ROBOTWIN_REPO="${ROBOTWIN_REPO:-$(default_repo_path "${API_ROOT}/simulators/RoboTwin")}"

LDA_ARGS=()
LINGBOT_ARGS=()
ROBOTWIN_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options]

Create the default API-owned conda environments under one checkout-local root.
The default envs are:
  ${ENV_ROOT}/api-lda-1b
  ${ENV_ROOT}/api-lingbot-va
  ${ENV_ROOT}/api-robotwin

Options:
  --env-root PATH              Env root. Default: ${ENV_ROOT}
  --lda-repo PATH              Existing LDA-1B checkout. Default: ${LDA_REPO:-<none>}
  --lingbot-repo PATH          Existing LingBot-VA checkout. Default: ${LINGBOT_REPO:-<none>}
  --robotwin-repo PATH         Existing RoboTwin checkout. Default: ${ROBOTWIN_REPO:-<none>}
  --skip-lda                   Do not create the LDA-1B env.
  --skip-lingbot               Do not create the LingBot-VA env.
  --skip-robotwin              Do not create the RoboTwin env.
  --lda-torch-index URL        PyTorch wheel index for LDA-1B. Default: create_lda_1b_env.sh default.
  --skip-lda-torch             Skip LDA-1B torch wheel preinstall.
  --skip-flash-attn            Skip flash-attn in LDA-1B and LingBot-VA envs.
  --flash-attn-jobs N          Max parallel jobs for flash-attn builds in LDA-1B and LingBot-VA.
  --flash-attn-nvcc-threads N  NVCC threads per flash-attn compile job in LDA-1B and LingBot-VA.
  --skip-lingbot-torch         Skip LingBot-VA torch wheel installation.
  --lingbot-require-cuda       Fail LingBot-VA validation if torch cannot see CUDA.
  --lda-require-cuda           Fail LDA-1B validation if torch cannot see CUDA.
  --lingbot-requirements       Install LingBot-VA requirements.txt instead of README package set.
  --robotwin-manual            Use RoboTwin requirements fallback instead of script/_install.sh.
  --download-robotwin-assets   Also run RoboTwin script/_download_assets.sh.
  --clean-stale-named          Remove old named api-* envs before creating prefix envs.
  -h, --help                   Show this help.

This script runs env creation sequentially to avoid conda package-cache races.
It does not download model checkpoints.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --lda-repo)
      LDA_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --lingbot-repo)
      LINGBOT_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --robotwin-repo)
      ROBOTWIN_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --skip-lda)
      CREATE_LDA=0
      shift
      ;;
    --skip-lingbot)
      CREATE_LINGBOT=0
      shift
      ;;
    --skip-robotwin)
      CREATE_ROBOTWIN=0
      shift
      ;;
    --lda-torch-index)
      LDA_ARGS+=(--torch-index "$2")
      shift 2
      ;;
    --skip-lda-torch)
      LDA_ARGS+=(--skip-torch)
      shift
      ;;
    --skip-flash-attn)
      LDA_ARGS+=(--skip-flash-attn)
      LINGBOT_ARGS+=(--skip-flash-attn)
      shift
      ;;
    --flash-attn-jobs)
      LDA_ARGS+=(--flash-attn-jobs "$2")
      LINGBOT_ARGS+=(--flash-attn-jobs "$2")
      shift 2
      ;;
    --flash-attn-nvcc-threads)
      LDA_ARGS+=(--flash-attn-nvcc-threads "$2")
      LINGBOT_ARGS+=(--flash-attn-nvcc-threads "$2")
      shift 2
      ;;
    --lda-require-cuda)
      LDA_ARGS+=(--require-cuda)
      shift
      ;;
    --skip-lingbot-torch)
      LINGBOT_ARGS+=(--skip-torch)
      shift
      ;;
    --lingbot-require-cuda)
      LINGBOT_ARGS+=(--require-cuda)
      shift
      ;;
    --lingbot-requirements)
      LINGBOT_ARGS+=(--requirements)
      shift
      ;;
    --robotwin-manual)
      ROBOTWIN_ARGS+=(--manual)
      shift
      ;;
    --download-robotwin-assets)
      ROBOTWIN_ARGS+=(--download-assets)
      shift
      ;;
    --clean-stale-named)
      CLEAN_STALE_NAMED=1
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
mkdir -p "${ENV_ROOT}"

if [[ "${CLEAN_STALE_NAMED}" -eq 1 ]]; then
  bash "${SCRIPT_DIR}/cleanup_api_envs.sh" --yes --named-only --env-root "${ENV_ROOT}"
fi

if [[ "${CREATE_LDA}" -eq 1 ]]; then
  info "creating API LDA-1B env at ${ENV_ROOT}/api-lda-1b"
  API_CONDA_ENVS_DIR="${ENV_ROOT}" \
  LDA_1B_ENV_NAME="${ENV_ROOT}/api-lda-1b" \
  bash "${SCRIPT_DIR}/create_lda_1b_env.sh" --repo "${LDA_REPO}" "${LDA_ARGS[@]}"
fi

if [[ "${CREATE_LINGBOT}" -eq 1 ]]; then
  info "creating API LingBot-VA env at ${ENV_ROOT}/api-lingbot-va"
  API_CONDA_ENVS_DIR="${ENV_ROOT}" \
  LINGBOT_VA_ENV_NAME="${ENV_ROOT}/api-lingbot-va" \
  bash "${SCRIPT_DIR}/create_lingbot_va_env.sh" --repo "${LINGBOT_REPO}" "${LINGBOT_ARGS[@]}"
fi

if [[ "${CREATE_ROBOTWIN}" -eq 1 ]]; then
  info "creating API RoboTwin env at ${ENV_ROOT}/api-robotwin"
  API_CONDA_ENVS_DIR="${ENV_ROOT}" \
  ROBOTWIN_ENV_NAME="${ENV_ROOT}/api-robotwin" \
  bash "${SCRIPT_DIR}/create_robotwin_env.sh" --repo "${ROBOTWIN_REPO}" "${ROBOTWIN_ARGS[@]}"
fi

info "done. API env root: ${ENV_ROOT}"
info "run scripts can target these with --env ${ENV_ROOT}/api-lda-1b, --env ${ENV_ROOT}/api-lingbot-va, or --env ${ENV_ROOT}/api-robotwin"
