Model-specific configs can be added here without changing evaluator logic.

When a task config carries both simulator and model settings, put model settings
under `TaskSpec.metadata.models.<adapter-name>`. The evaluator sends the same
task spec to both endpoints; the simulator server and model server remain in
separate conda environments and read only their scoped settings.

## LingBot-VA

`lingbot-va` is a WebSocket client adapter. It expects LingBot-VA's native
server to already be running, typically from `models/lingbot-va`:

```bash
export API_ROOT="$(pwd)"
export LINGBOT_VA_ROOT="${API_ROOT}/models/lingbot-va"
export LINGBOT_VA_HOST=127.0.0.1
export LINGBOT_VA_PORT=29056
python -m eval_system.model_server.server --adapter lingbot-va --port 50051
```

Task metadata can override:

```json
{
  "metadata": {
    "models": {
      "lingbot-va": {
        "repo_path": "models/lingbot-va",
        "host": "127.0.0.1",
        "port": 29056,
        "env_type": "libero",
        "action_type": "delta_ee_pose"
      }
    }
  }
}
```

The same adapter also exposes a training launcher through the model API:

```python
from eval_system.common.schemas import TrainingSpec
from eval_system.model_server.client import WorldModelClient

client = WorldModelClient("http://127.0.0.1:50051")
response = client.train(
    TrainingSpec(
        benchmark="robotwin",
        dataset_path="/path/to/your/dataset",
        output_dir="/path/to/train_out",
        dry_run=True,
        metadata={
            "num_gpus": 8,
            "config_overrides": {
                "num_steps": 50000,
                "gradient_accumulation_steps": 1
            }
        },
    )
)
print(response["command_text"])
```

`dry_run=True` is the default and only returns the prepared command. Set
`dry_run=False` to launch LingBot-VA post-training as a detached process from
the model API server, then call `client.training_status(response["job_id"])`.

LingBot-VA uses different attention modes for each workflow: set the checkpoint
`transformer/config.json` `attn_mode` to `"flex"` before training, and to
`"torch"` or `"flashattn"` before evaluation/inference.

## LDA-1B

`lda-1b` is also a WebSocket client adapter. It expects LDA's native
`deployment/model_server/server_policy.py` server to already be running:

```bash
export API_ROOT="$(pwd)"
export LDA_1B_ROOT="${API_ROOT}/models/LDA-1B"
export LDA_1B_HOST=127.0.0.1
export LDA_1B_PORT=10093
python -m eval_system.model_server.server --adapter lda-1b --port 50051
```

Common metadata:

```json
{
  "metadata": {
    "models": {
      "lda-1b": {
        "repo_path": "models/LDA-1B",
        "host": "127.0.0.1",
        "port": 10093,
        "payload_mode": "examples",
        "image_keys": ["agentview_image", "robot0_eye_in_hand_image"],
        "state_keys": ["robot0_eef_pos", "robot0_gripper_qpos"],
        "embodiment_id": 0,
        "action_type": "delta_ee_pose"
      }
    }
  }
}
```

LDA variants often return `normalized_actions`. If your simulator expects
physical action units, provide `action_min` and `action_max` metadata so the API
adapter can unnormalize the returned chunk before sending it to the simulator.
The adapter can also mirror LDA's RoboCasa bridge and read stats from the
checkpoint run directory: set `checkpoint_path`/`policy_ckpt_path`, or set
`norm_stats_path`/`dataset_statistics_path` directly. If the statistics file
contains multiple datasets, also set `unnorm_key`. By default, the adapter
refuses to pass `normalized_actions` through without stats; set
`allow_normalized_actions: true` only when the target simulator intentionally
consumes normalized action units.

For RoboTwin, use an explicit action contract. The included
`robotwin_place_empty_cup` task profile expects 16D dual-arm actions and will
fail fast if a checkpoint returns a different shape, such as a 29D RoboCasa
action or a 138D GR1/Galbot action:

```json
{
  "metadata": {
    "models": {
      "lda-1b": {
        "profile": "robotwin",
        "image_keys": [
          "observation.head_camera.rgb",
          "observation.left_camera.rgb",
          "observation.right_camera.rgb"
        ],
        "state_keys": ["joint_action.vector"],
        "expected_state_dim": 16,
        "expected_image_count": 3,
        "action_type": "qpos",
        "expected_action_dim": 16,
        "allow_normalized_actions": false
      }
    }
  }
}
```

If your LDA checkpoint emits a larger structured vector and only a slice belongs
to RoboTwin, configure that deliberately with `action_slice: [START, END]` or
`action_indices: [...]`; the adapter will not silently truncate.

The `lda-1b` adapter also exposes a native training launcher. It mirrors the
repository's `accelerate launch lda/training/train_LDA.py` workflow and remains
dry-run by default:

```python
from eval_system.common.schemas import TrainingSpec
from eval_system.model_server.client import WorldModelClient

client = WorldModelClient("http://127.0.0.1:50051")
response = client.train(
    TrainingSpec(
        benchmark="robocasa",
        dataset_path="/path/to/dataset",
        output_dir="/path/to/training/results",
        dry_run=True,
        metadata={
            "base_vlm": "/path/to/pretrained/VLM",
            "vision_encoder_path": "/path/to/pretrained/vision/encoder",
            "data_mix": "demo_data",
            "run_id": "experiment_name",
            "num_gpus": 8,
            "config_overrides": {
                "trainer.max_train_steps": 200000,
                "trainer.save_interval": 10000
            }
        },
    )
)
print(response["command_text"])
```

Set `dry_run=False` to launch it from the model API process, then poll
`client.training_status(response["job_id"])`.

The same training metadata can live in the task config under
`metadata.training.<model>.<simulator>`. The launcher accepts user-facing aliases
such as `steps`, `lr`, and `batch_size`; adapters translate them to their native
training arguments. You can override those from the CLI without editing JSON:

```bash
./eval_system/scripts/run/run_train.sh \
  --model lda-1b \
  --benchmark libero \
  --task libero_spatial \
  --dataset-path /path/to/libero/dataset \
  --output-dir /path/to/lda_out \
  --param steps=200000 \
  --param lr=0.00004 \
  --param batch_size=64
```

## Motus and fastWAM

`motus` and `fastwam` are currently placeholder policy adapters in this API
checkout. Evaluation works through the shared dummy policy behavior. Because no
native Motus or fastWAM training repository is checked out here, training is
available through the generic external-command launcher:

```python
response = client.train(
    TrainingSpec(
        output_dir="/path/to/train_out",
        dry_run=True,
        metadata={
            "cwd": "/path/to/model/repo",
            "training_command": ["python", "-m", "your_package.train"],
            "extra_args": ["--config", "config.yaml"],
            "env": {"CUDA_VISIBLE_DEVICES": "0"}
        },
    )
)
```

If `training_command` is omitted, these adapters return `501 Not Implemented`
for `/train` because there is no model-specific training entrypoint to infer.
