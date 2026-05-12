#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_common.sh"

ENV_NAME="${EVAL_ENV:-}"
BENCHMARK="${BENCHMARK:-robotwin}"
TASK="${TASK:-robotwin_place_empty_cup}"
MODEL_ADAPTER="${MODEL_ADAPTER:-lda-1b}"
SIM_ADAPTER="${SIM_ADAPTER:-${BENCHMARK}}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-}"
SIM_ENDPOINT="${SIM_ENDPOINT:-}"
METADATA_JSON="${METADATA_JSON:-}"
CHECK_ENDPOINTS="${CHECK_ENDPOINTS:-0}"
OUTPUT="${OUTPUT:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Run static API preflight checks. This does not launch model/simulator code.

Options:
  --env ENV               Conda env name or prefix path. Default: EVAL_ENV or current Python.
  --benchmark NAME        Default: ${BENCHMARK}
  --task NAME             Default: ${TASK}
  --model-adapter NAME    Default: ${MODEL_ADAPTER}
  --sim-adapter NAME      Default: ${SIM_ADAPTER}
  --model-endpoint URL    Optional endpoint string to normalize or health-check.
  --sim-endpoint URL      Optional endpoint string to normalize or health-check.
  --metadata-json JSON    Merge JSON object into TaskSpec.metadata.
  --check-endpoints       POST /health to the configured endpoints.
  --output PATH           JSON report path. Default: results/preflight/<stamp>.json.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --model-adapter|--model)
      MODEL_ADAPTER="$2"
      shift 2
      ;;
    --sim-adapter|--simulator)
      SIM_ADAPTER="$2"
      shift 2
      ;;
    --model-endpoint)
      MODEL_ENDPOINT="$2"
      shift 2
      ;;
    --sim-endpoint)
      SIM_ENDPOINT="$2"
      shift 2
      ;;
    --metadata-json)
      METADATA_JSON="$2"
      shift 2
      ;;
    --check-endpoints)
      CHECK_ENDPOINTS=1
      shift
      ;;
    --output)
      OUTPUT="$2"
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

ARGS=(
  "${PYTHON_BIN}" -m eval_system.diagnostics.preflight
  --benchmark "${BENCHMARK}"
  --task "${TASK}"
  --model-adapter "${MODEL_ADAPTER}"
  --sim-adapter "${SIM_ADAPTER}"
)

[[ -n "${MODEL_ENDPOINT}" ]] && ARGS+=(--model-endpoint "${MODEL_ENDPOINT}")
[[ -n "${SIM_ENDPOINT}" ]] && ARGS+=(--sim-endpoint "${SIM_ENDPOINT}")
[[ -n "${METADATA_JSON}" ]] && ARGS+=(--metadata-json "${METADATA_JSON}")
[[ -n "${OUTPUT}" ]] && ARGS+=(--output "${OUTPUT}")
if [[ "${CHECK_ENDPOINTS}" != "0" ]]; then
  ARGS+=(--check-endpoints)
fi

run_python "${ENV_NAME}" "${ARGS[@]}"
