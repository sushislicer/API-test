#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_common.sh"

ENV_NAME="${EVAL_ENV:-}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-grpc://127.0.0.1:50051}"
MODEL="${MODEL_ADAPTER:-lingbot-va}"
BENCHMARK="${BENCHMARK:-libero}"
TASK="${TASK:-libero_spatial}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/train}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CONFIG_NAME="${CONFIG_NAME:-}"
SEED="${SEED:-0}"
LAUNCH="${LAUNCH:-0}"
PARAMS=()

usage() {
  cat <<EOF
Usage: $0 [options]

Prepare or launch model training through the model API. Default is dry-run.

Options:
  --env NAME              Conda env for this lightweight client. Default: EVAL_ENV or current Python.
  --model-endpoint URL    Default: ${MODEL_ENDPOINT}
  --model NAME            Model adapter name. Default: ${MODEL}
  --benchmark NAME        Simulator/benchmark name. Default: ${BENCHMARK}
  --task NAME             Task config name. Default: ${TASK}
  --dataset-path PATH     Training dataset path.
  --output-dir PATH       Training output root.
  --checkpoint-path PATH  Resume/checkpoint path.
  --config-name NAME      Optional model config/run name.
  --seed N                Default: ${SEED}
  --param KEY=VALUE       Override a training parameter. Repeat as needed.
  --launch                Actually launch training from the model server. Default dry-run.
  -h, --help              Show this help.
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
    --model)
      MODEL="$2"
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
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --param)
      PARAMS+=("--param" "$2")
      shift 2
      ;;
    --launch)
      LAUNCH=1
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
  "${PYTHON_BIN}" -m eval_system.training.run_train
  --model_endpoint "${MODEL_ENDPOINT}"
  --model "${MODEL}"
  --benchmark "${BENCHMARK}"
  --task "${TASK}"
  --seed "${SEED}"
)

[[ -n "${DATASET_PATH}" ]] && ARGS+=(--dataset_path "${DATASET_PATH}")
[[ -n "${OUTPUT_DIR}" ]] && ARGS+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${CHECKPOINT_PATH}" ]] && ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
[[ -n "${CONFIG_NAME}" ]] && ARGS+=(--config_name "${CONFIG_NAME}")
if [[ "${LAUNCH}" != "0" ]]; then
  ARGS+=(--launch)
fi
ARGS+=("${PARAMS[@]}")

run_python "${ENV_NAME}" "${ARGS[@]}"
