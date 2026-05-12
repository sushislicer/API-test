#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_common.sh"

ENV_NAME="${SIM_ENV:-}"
ADAPTER="${SIM_ADAPTER:-dummy-simulator}"
HOST="${HOST:-127.0.0.1}"
PORT="${SIM_PORT:-50052}"

usage() {
  cat <<EOF
Usage: $0 [options]

Start a simulator API server without Docker.

Options:
  --env ENV         Conda env name or prefix path. Default: SIM_ENV, adapter API env, or current Python.
  --adapter NAME    Simulator adapter. Default: ${ADAPTER}
  --host HOST       Bind host. Default: ${HOST}
  --port PORT       Bind port. Default: ${PORT}
  -h, --help        Show this help.

Environment variables:
  SIM_ENV, SIM_ADAPTER, HOST, SIM_PORT, LIBERO_ROOT, ROBOTWIN_ROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --adapter)
      ADAPTER="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
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

if [[ -z "${ENV_NAME}" ]]; then
  case "${ADAPTER}" in
    robotwin)
      ENV_NAME="${API_CONDA_ENVS_DIR}/api-robotwin"
      ;;
    libero)
      ENV_NAME="${API_CONDA_ENVS_DIR}/api-libero"
      ;;
    simplerenv)
      ENV_NAME="${API_CONDA_ENVS_DIR}/api-simplerenv"
      ;;
  esac
fi

run_python "${ENV_NAME}" "${PYTHON_BIN}" -m eval_system.sim_server.server \
  --adapter "${ADAPTER}" \
  --host "${HOST}" \
  --port "${PORT}"
