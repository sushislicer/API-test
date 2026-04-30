Simulator-specific configs can be added here without changing evaluator logic.

Native simulator adapters are configured through `TaskSpec.metadata`.
When the same task config also carries model settings, put simulator settings
under `metadata.simulators.<adapter-name>` so only the simulator server consumes
them. The evaluator sends the same `TaskSpec` to the simulator endpoint and the
model endpoint; each process runs in its own conda environment and reads only
its scoped metadata.

LIBERO example:

```json
{
  "benchmark": "libero",
  "task": "libero_spatial",
  "metadata": {
    "simulators": {
      "libero": {
        "repo_path": "simulators/LIBERO",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "camera_height": 128,
        "camera_width": 128
      }
    }
  }
}
```

Relative `repo_path` values are resolved from the API checkout root. If
`repo_path` is omitted, the LIBERO adapter checks `LIBERO_ROOT`, then falls back
to `simulators/LIBERO` inside this API checkout.

SimplerEnv example:

```json
{
  "benchmark": "simplerenv",
  "task": "google_robot_pick_coke_can",
  "metadata": {
    "env_name": "google_robot_pick_coke_can",
    "backend": "simpler_env"
  }
}
```

RoboTwin example:

```json
{
  "benchmark": "robotwin",
  "task": "place_empty_cup",
  "metadata": {
    "simulators": {
      "robotwin": {
        "repo_path": "simulators/RoboTwin",
        "task_name": "place_empty_cup",
        "task_config": "demo_clean",
        "action_type": "qpos",
        "strict_action_dim": true,
        "instruction_source": "env"
      }
    }
  }
}
```

Relative `repo_path` values are resolved from the API checkout root. If
`repo_path` is omitted, the RoboTwin adapter first checks `ROBOTWIN_ROOT`, then
falls back to `simulators/RoboTwin` inside this API checkout.
`strict_action_dim` is enabled by default so an LDA checkpoint trained for a
different embodiment cannot silently send 29D/138D actions into a 16D RoboTwin
controller.

The canonical observation carries native images under `images` and `depth_images` as `raw_base64` array payloads. Model adapters can decode them with `eval_system.common.arrays.decode_array` without importing simulator packages.
