#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_conda_common.sh"

ENV_NAME="${SIMPLERENV_ENV_NAME:-simpler_env}"
PYTHON_VERSION="${SIMPLERENV_PYTHON_VERSION:-3.10}"
SIMPLERENV_REPO="${SIMPLERENV_REPO:-$(default_repo_path "${API_ROOT}/../SimplerEnv")}"
INSTALL_FULL_REQUIREMENTS=0

usage() {
  cat <<EOF
Usage: $0 [options]

Create/update a conda environment for SimplerEnv.

Options:
  --env NAME              Conda environment name. Default: ${ENV_NAME}
  --python VERSION       Python version. Default: ${PYTHON_VERSION}
  --repo PATH            Existing SimplerEnv checkout. Default: ${SIMPLERENV_REPO:-<none>}
  --full-requirements    Install requirements_full_install.txt if present.
  -h, --help             Show this help.

This expects a checkout cloned with submodules, because official SimplerEnv
installs ManiSkill2_real2sim from:
  SimplerEnv/ManiSkill2_real2sim
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
      SIMPLERENV_REPO="$(cd "$2" && pwd)"
      shift 2
      ;;
    --full-requirements)
      INSTALL_FULL_REQUIREMENTS=1
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

assert_repo_dir "SimplerEnv" "${SIMPLERENV_REPO}"
[[ -d "${SIMPLERENV_REPO}/ManiSkill2_real2sim" ]] || die "missing ${SIMPLERENV_REPO}/ManiSkill2_real2sim; clone SimplerEnv with --recurse-submodules"

create_conda_env "${ENV_NAME}" "${PYTHON_VERSION}"
upgrade_pip "${ENV_NAME}"

info "installing SimplerEnv numpy pin"
pip_in_env "${ENV_NAME}" install numpy==1.24.4

if [[ "${INSTALL_FULL_REQUIREMENTS}" -eq 1 && -f "${SIMPLERENV_REPO}/requirements_full_install.txt" ]]; then
  info "installing SimplerEnv full requirements"
  pip_in_env "${ENV_NAME}" install -r "${SIMPLERENV_REPO}/requirements_full_install.txt"
fi

info "installing ManiSkill2_real2sim editable package"
pip_in_env "${ENV_NAME}" install -e "${SIMPLERENV_REPO}/ManiSkill2_real2sim"

info "installing SimplerEnv editable package"
pip_in_env "${ENV_NAME}" install -e "${SIMPLERENV_REPO}"

link_api_package "${ENV_NAME}"

info "done. Run with: conda activate ${ENV_NAME}"
