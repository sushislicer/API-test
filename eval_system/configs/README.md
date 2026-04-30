# Configs

Task files under `tasks/` define the shared `TaskSpec` sent to both API
services. Keep simulator-only settings under `metadata.simulators.<adapter>` and
model-only settings under `metadata.models.<adapter>` so each conda/container
environment can ignore settings it does not understand.

Before a first run on a remote GPU machine, run:

```bash
./eval_system/scripts/run/preflight.sh \
  --benchmark robotwin \
  --task robotwin_place_empty_cup \
  --model-adapter lda-1b \
  --sim-adapter robotwin
```

Training/evaluation outputs default to the repo-root `results/` tree:

```text
results/
  eval/       evaluator episode logs and reports
  train/      model-server training dry-runs and launch roots
  preflight/  static contract reports
```

