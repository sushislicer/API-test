#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_common.sh"

ENV_NAME="${MODEL_ENV:-}"
ADAPTER="${MODEL_ADAPTER:-dummy-model}"
HOST="${HOST:-127.0.0.1}"
PORT="${MODEL_PORT:-50051}"

usage() {
  cat <<EOF
Usage: $0 [options]

Start a model API server without Docker.

Options:
  --env ENV         Conda env name or prefix path. Default: MODEL_ENV, adapter API env, or current Python.
  --adapter NAME    Model adapter. Default: ${ADAPTER}
  --host HOST       Bind host. Default: ${HOST}
  --port PORT       Bind port. Default: ${PORT}
  -h, --help        Show this help.

Environment variables:
  MODEL_ENV, MODEL_ADAPTER, HOST, MODEL_PORT,
  LINGBOT_VA_ROOT, LINGBOT_VA_HOST, LINGBOT_VA_PORT, LINGBOT_VA_API_KEY,
  LDA_1B_ROOT, LDA_1B_HOST, LDA_1B_PORT, LDA_1B_API_KEY
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
    lda-1b)
      ENV_NAME="${API_CONDA_ENVS_DIR}/api-lda-1b"
      ;;
    lingbot-va)
      ENV_NAME="${API_CONDA_ENVS_DIR}/api-lingbot-va"
      ;;
  esac
fi

run_python "${ENV_NAME}" "${PYTHON_BIN}" -m eval_system.model_server.server \
  --adapter "${ADAPTER}" \
  --host "${HOST}" \
  --port "${PORT}"
