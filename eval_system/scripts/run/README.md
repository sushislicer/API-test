# No-Docker Run Scripts

Use these scripts when the remote machine does not support Docker. Run each long-lived server in a separate terminal or `tmux` pane.
Set `PYTHON_BIN=python` if a conda environment only exposes `python`; the
default command is `python3`.
Examples assume you are at the repo root. `API_ROOT="$(pwd)"` makes the same
commands work after cloning to any remote path.
Set `DOWNLOAD_ROOT` to the checkpoint/asset cache. The default is
`${API_ROOT}/downloads`; on the A800 machine this can point to a larger mounted
disk.
Examples use API-owned conda prefix envs under `.conda-envs/`. Create them with
`bash eval_system/scripts/envs/create_api_envs.sh`, or pass any other env name
or prefix path to `--env`. For known heavy adapters, `start_model_server.sh`
and `start_sim_server.sh` default to the matching `.conda-envs/api-*` env when
`--env` is omitted.

## 1. Start Simulator Server

LIBERO:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export LIBERO_ROOT="${API_ROOT}/simulators/LIBERO"
./eval_system/scripts/run/start_sim_server.sh \
  --env .conda-envs/api-libero \
  --adapter libero \
  --host 127.0.0.1 \
  --port 50052
```

RoboTwin:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export ROBOTWIN_ROOT="${API_ROOT}/simulators/RoboTwin"
./eval_system/scripts/run/start_sim_server.sh \
  --env .conda-envs/api-robotwin \
  --adapter robotwin \
  --host 127.0.0.1 \
  --port 50052
```

SimplerEnv:

```bash
./eval_system/scripts/run/start_sim_server.sh \
  --env .conda-envs/api-simplerenv \
  --adapter simplerenv \
  --host 127.0.0.1 \
  --port 50052
```

## 2. Start Model Server

The `lingbot-va` adapter is a client for LingBot-VA's native WebSocket server.
Start the native LingBot-VA server first, then start this API model server:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export LINGBOT_VA_ROOT="${API_ROOT}/models/lingbot-va"
export LINGBOT_VA_HOST=127.0.0.1
export LINGBOT_VA_PORT=29056
./eval_system/scripts/run/start_model_server.sh \
  --env .conda-envs/api-lingbot-va \
  --adapter lingbot-va \
  --host 127.0.0.1 \
  --port 50051
```

The `lda-1b` adapter is a client for LDA's native WebSocket policy server:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export LDA_1B_ROOT="${API_ROOT}/models/LDA-1B"
export LDA_1B_HOST=127.0.0.1
export LDA_1B_PORT=10093
./eval_system/scripts/run/start_model_server.sh \
  --env .conda-envs/api-lda-1b \
  --adapter lda-1b \
  --host 127.0.0.1 \
  --port 50051
```

## 3. Preflight

Run this before launching expensive jobs or debugging remote services. It is a
static check by default: no model server, simulator server, native WebSocket
server, training, or setup code is launched.

```bash
./eval_system/scripts/run/preflight.sh \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --model-adapter lda-1b \
  --sim-adapter robotwin
```

When servers are already running, add `--model-endpoint`, `--sim-endpoint`, and
`--check-endpoints` to POST `/health` to both API services.

## 4. Run Evaluator

```bash
./eval_system/scripts/run/run_eval.sh \
  --model-endpoint grpc://127.0.0.1:50051 \
  --sim-endpoint grpc://127.0.0.1:50052 \
  --benchmark libero \
  --task libero_spatial \
  --episodes 3 \
  --output-dir results/eval
```

The evaluator remains dependency-light. The simulator server runs in the
simulator conda env, the model API server runs in the model conda env, and the
evaluator passes one `TaskSpec` to both endpoints. Use
`metadata.simulators.libero` for LIBERO-only settings and
`metadata.models.lingbot-va` or `metadata.models.lda-1b` for model-only
observation/action settings.

For LDA-1B on RoboTwin, start the RoboTwin simulator with a task profile such as
`robotwin_place_empty_cup`. That profile requires 16D actions and uses the
environment-provided instruction so constant generic prompts do not mask task
conditioning:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
export ROBOTWIN_ROOT="${API_ROOT}/simulators/RoboTwin"
./eval_system/scripts/run/start_sim_server.sh \
  --env .conda-envs/api-robotwin \
  --adapter robotwin \
  --host 127.0.0.1 \
  --port 50052

./eval_system/scripts/run/run_eval.sh \
  --model-endpoint grpc://127.0.0.1:50051 \
  --sim-endpoint grpc://127.0.0.1:50052 \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --episodes 3 \
  --output-dir results/eval
```

The resulting `report.json` includes action diagnostics: observed action dims,
repeat rate, finite-value rate, and action norm statistics. Use these before
watching every video when diagnosing collapse.

## 5. Prepare Training Command

Training is also launched through the model API server, so the actual training
process inherits the model conda env. The default is dry-run:

```bash
./eval_system/scripts/run/run_train.sh \
  --model-endpoint grpc://127.0.0.1:50051 \
  --model lingbot-va \
  --benchmark libero \
  --task libero_spatial \
  --dataset-path /path/to/libero/lerobot_dataset \
  --output-dir results/train \
  --param steps=5000 \
  --param lr=0.00001 \
  --param batch_size=1
```

Add `--launch` only when you want the model server to start training.

## Remote Access

Prefer binding services to `127.0.0.1` and running all three processes on the same remote machine. If you need to access a server from your laptop, use SSH port forwarding:

```bash
ssh -L 50051:127.0.0.1:50051 -L 50052:127.0.0.1:50052 user@remote
```

Only bind with `--host 0.0.0.0` if the remote network is trusted and firewall rules are set correctly.
