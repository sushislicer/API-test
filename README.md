# Lightweight Evaluation API Prototype

This directory contains a minimal prototype for decoupled world-model and simulator evaluation.

Design goals:

- keep model and simulator dependencies isolated
- use a shared canonical schema
- avoid importing simulator code into model code
- avoid importing model code into simulator code
- keep the evaluator as the only component that knows both endpoints
- stay lightweight: standard-library HTTP/JSON transport, no repo downloads

The transport is intentionally simple for local prototyping. The interfaces are shaped so the same adapters and evaluator logic can later be moved behind gRPC.

## Layout

```text
api/
  downloads/
    checkpoints/
    assets/
    datasets/
  eval_system/
    common/
    model_server/
    sim_server/
    evaluator/
    configs/
    docker/
    scripts/
  results/
    eval/
    train/
    preflight/
```

`downloads/` is the default local cache for heavyweight checkpoints, simulator
assets, and datasets. It is intentionally ignored by git except for placeholder
files and `downloads/README.md`. On a remote machine with a larger disk, set
`DOWNLOAD_ROOT=/path/to/cache` before starting services.

## Quick start

Optional environment setup scripts live under `eval_system/scripts/envs/`.
They create per-stack conda environments for LingBot-VA, LDA-1B, LIBERO,
RoboTwin, and SimplerEnv from existing upstream checkouts.

Docker is not required. On a remote machine, run the model server, simulator server, and evaluator as separate conda processes. Helper scripts live under `eval_system/scripts/run/`.

Start a dummy model service:

```bash
./eval_system/scripts/run/start_model_server.sh --adapter dummy-model --host 127.0.0.1 --port 50051
```

Start a simulator service:

```bash
./eval_system/scripts/run/start_sim_server.sh --adapter dummy-simulator --host 127.0.0.1 --port 50052
```

Run evaluation:

```bash
./eval_system/scripts/run/run_eval.sh \
  --model-endpoint grpc://127.0.0.1:50051 \
  --sim-endpoint grpc://127.0.0.1:50052 \
  --benchmark libero \
  --task libero_spatial \
  --episodes 3 \
  --output-dir results/eval
```

Before a real run, use preflight to catch adapter, endpoint, action-dimension,
normalization-stat, prompt, image-view, and state-shape issues without starting
training or simulator code:

```bash
./eval_system/scripts/run/preflight.sh \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --model-adapter lda-1b \
  --sim-adapter robotwin
```

For a fuller remote checklist, see
`eval_system/docs/first_remote_run.md`.

For a real simulator environment, pass the conda env and adapter:

```bash
./eval_system/scripts/run/start_sim_server.sh --env env-robotwin --adapter robotwin --port 50052
./eval_system/scripts/run/start_sim_server.sh --env env-libero --adapter libero --port 50052
./eval_system/scripts/run/start_sim_server.sh --env env-simpler --adapter simplerenv --port 50052
```

For LingBot-VA on LIBERO, start the LIBERO simulator server in `env-libero`,
start LingBot's native WebSocket inference server from `models/lingbot-va`, then
run this API model adapter as a client:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export LIBERO_ROOT="${API_ROOT}/simulators/LIBERO"
./eval_system/scripts/run/start_sim_server.sh --env env-libero --adapter libero --port 50052

export LINGBOT_VA_ROOT="${API_ROOT}/models/lingbot-va"
export LINGBOT_VA_HOST=127.0.0.1
export LINGBOT_VA_PORT=29056
./eval_system/scripts/run/start_model_server.sh --env env-lingbot-va --adapter lingbot-va --port 50051
```

For LDA-1B on LIBERO, keep the same LIBERO simulator server running, start
LDA's native WebSocket policy server from `models/LDA-1B`, then run this API
model adapter as a client:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export LDA_1B_ROOT="${API_ROOT}/models/LDA-1B"
export LDA_1B_HOST=127.0.0.1
export LDA_1B_PORT=10093
./eval_system/scripts/run/start_model_server.sh --env env-lda-1b --adapter lda-1b --port 50051
```

Note:

- `grpc://` endpoints are accepted for CLI compatibility, but this prototype currently uses HTTP/JSON over the same host and port.
- `lingbot-va` and `lda-1b` are implemented as clients for their native WebSocket inference servers. Model adapter files for `motus`, `pi0.7`, and `fastwam` are still lightweight placeholders built on shared dummy logic.
- `lingbot-va` and `lda-1b` also expose `/train` through the model API. Training requests are dry-run by default and return the native command; set `TrainingSpec.dry_run=False` only when you want the API server to launch training. Placeholder adapters such as `motus` and `fastwam` can use `/train` when you provide `metadata.training_command`.
- Task configs can scope settings with `metadata.simulators.libero`, `metadata.models.lingbot-va`, and `metadata.models.lda-1b`; the evaluator sends the same `TaskSpec` to both isolated conda processes.
- Simulator adapters for `LIBERO`, `RoboTwin`, and `SimplerEnv` are lazy native wrappers: they import official simulator packages only when the selected adapter is reset. Configure native paths and task IDs through `TaskSpec.metadata`.
- Evaluation reports include action diagnostics such as action dimensions, repeat rate, finite-value checks, and action norm statistics. These are intended to catch collapse modes like repeated task-independent actions early.
