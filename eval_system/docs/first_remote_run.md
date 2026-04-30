# First Remote Run Checklist

Use this when the repo is pulled onto the A800 machine.

1. Confirm paths without installing or launching:

```bash
cd /path/to/API
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"
python -m eval_system.diagnostics.preflight \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --model-adapter lda-1b \
  --sim-adapter robotwin
```

2. Start each long-lived process in a separate `tmux` pane:

```bash
./eval_system/scripts/run/start_sim_server.sh \
  --env env-robotwin \
  --adapter robotwin \
  --host 127.0.0.1 \
  --port 50052

./eval_system/scripts/run/start_model_server.sh \
  --env env-lda-1b \
  --adapter lda-1b \
  --host 127.0.0.1 \
  --port 50051
```

The LDA API adapter expects LDA's native WebSocket policy server to already be
running in the LDA environment. Put the checkpoint under
`${DOWNLOAD_ROOT}/checkpoints/lda-1b/` or set `DOWNLOAD_ROOT` to the mounted disk
that already contains the checkpoint tree.

3. Health-check the API boundary:

```bash
./eval_system/scripts/run/preflight.sh \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --model-adapter lda-1b \
  --sim-adapter robotwin \
  --model-endpoint grpc://127.0.0.1:50051 \
  --sim-endpoint grpc://127.0.0.1:50052 \
  --check-endpoints
```

4. Run a short evaluation first:

```bash
./eval_system/scripts/run/run_eval.sh \
  --model-endpoint grpc://127.0.0.1:50051 \
  --sim-endpoint grpc://127.0.0.1:50052 \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --episodes 1 \
  --horizon 20 \
  --output-dir results/eval/robotwin_smoke
```

5. Inspect `report.json` before watching all videos. If
`action_collapse_suspected` is true, check `action_dim_set`,
`action_repeat_rate`, `avg_action_l2_norm`, and the per-step action metadata.
For a standalone video, use:

```bash
python -m eval_system.diagnostics.video_probe path/to/rollout.mp4
```

For LDA, a RoboTwin run should use a checkpoint/action head whose action space
matches RoboTwin's 16D `qpos` contract, or an explicit, semantically justified
`action_slice`/`action_indices` projection plus matching normalization stats.
