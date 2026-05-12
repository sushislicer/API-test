#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_common.sh"

ENV_NAME="${EVAL_ENV:-}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-grpc://127.0.0.1:50051}"
SIM_ENDPOINT="${SIM_ENDPOINT:-grpc://127.0.0.1:50052}"
BENCHMARK="${BENCHMARK:-libero}"
TASK="${TASK:-libero_spatial}"
EPISODES="${EPISODES:-1}"
HORIZON="${HORIZON:-100}"
SEED="${SEED:-0}"
MODE="${MODE:-policy}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eval}"
PREDICT_NEXT="${PREDICT_NEXT:-0}"
METADATA_JSON="${METADATA_JSON:-}"

usage() {
  cat <<EOF
Usage: $0 [options]

Run the evaluator without Docker.

Options:
  --env ENV               Conda env name or prefix path. Default: EVAL_ENV or current Python.
  --model-endpoint URL    Default: ${MODEL_ENDPOINT}
  --sim-endpoint URL      Default: ${SIM_ENDPOINT}
  --benchmark NAME        Default: ${BENCHMARK}
  --task NAME             Default: ${TASK}
  --episodes N            Default: ${EPISODES}
  --horizon N             Default: ${HORIZON}
  --seed N                Default: ${SEED}
  --mode NAME             Default: ${MODE}
  --output-dir PATH       Default: ${OUTPUT_DIR}
  --metadata-json JSON    Merge JSON object into TaskSpec.metadata.
  --predict-next          Enable model next-observation prediction.
  --no-predict-next       Disable model next-observation prediction. Default.
  -h, --help              Show this help.

Environment variables mirror the long option names in uppercase.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
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
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --horizon)
      HORIZON="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --metadata-json)
      METADATA_JSON="$2"
      shift 2
      ;;
    --predict-next)
      PREDICT_NEXT=1
      shift
      ;;
    --no-predict-next)
      PREDICT_NEXT=0
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

ARGS=(
  "${PYTHON_BIN}" -m eval_system.evaluator.run_eval
  --model_endpoint "${MODEL_ENDPOINT}"
  --sim_endpoint "${SIM_ENDPOINT}"
  --benchmark "${BENCHMARK}"
  --task "${TASK}"
  --episodes "${EPISODES}"
  --horizon "${HORIZON}"
  --seed "${SEED}"
  --mode "${MODE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${PREDICT_NEXT}" != "0" ]]; then
  ARGS+=(--predict_next)
fi
if [[ -n "${METADATA_JSON}" ]]; then
  ARGS+=(--metadata_json "${METADATA_JSON}")
fi

run_python "${ENV_NAME}" "${ARGS[@]}"
