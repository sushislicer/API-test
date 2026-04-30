# Environment Setup Scripts

These scripts create conda environments from upstream repos you already have on
disk. They do not run automatically, download checkpoints, download simulator
assets, clone repos, or start servers.

Use separate environments for heavy stacks:

```bash
# Model repos in this checkout
bash eval_system/scripts/envs/create_lingbot_va_env.sh --env env-lingbot-va --repo models/lingbot-va
bash eval_system/scripts/envs/create_lda_1b_env.sh --env env-lda-1b --repo models/LDA-1B

# Simulator repos
bash eval_system/scripts/envs/create_robotwin_env.sh --env env-robotwin --repo simulators/RoboTwin
bash eval_system/scripts/envs/create_libero_env.sh --env env-libero --repo /path/to/LIBERO
bash eval_system/scripts/envs/create_simplerenv_env.sh --env env-simpler --repo /path/to/SimplerEnv
```

Each script links this API checkout into the target environment with a `.pth`
file, so `python -m eval_system...` works from that env without packaging this
repo.

## LingBot-VA

Default command:

```bash
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env env-lingbot-va \
  --repo models/lingbot-va
```

The default install follows the LingBot-VA README inference dependencies:

```bash
conda create -n env-lingbot-va python=3.10.16 -y
conda activate env-lingbot-va
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
pip install --no-deps -e models/lingbot-va
```

Useful options:

```bash
# Use a different CUDA/PyTorch wheel index
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env env-lingbot-va \
  --repo models/lingbot-va \
  --torch-index https://download.pytorch.org/whl/cu126

# Skip torch if the remote machine already has the right GPU build installed
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env env-lingbot-va \
  --repo models/lingbot-va \
  --skip-torch

# Install LingBot's requirements.txt exactly instead of the README package list
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env env-lingbot-va \
  --repo models/lingbot-va \
  --requirements

# Add the post-training extras from the LingBot README
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env env-lingbot-va \
  --repo models/lingbot-va \
  --post-training
```

The script does not edit model checkpoint config. Before inference, follow the
LingBot README note and set the checkpoint `transformer/config.json`
`attn_mode` to `"torch"` or `"flashattn"`, not `"flex"`.

Run LingBot's native server from the LingBot env, then run this API adapter as a
client:

```bash
conda activate env-lingbot-va
cd /home/yangc/Lab/API/models/lingbot-va
START_PORT=29056 bash evaluation/robotwin/launch_server.sh
```

In a second terminal:

```bash
cd /home/yangc/Lab/API
LINGBOT_VA_ROOT=/home/yangc/Lab/API/models/lingbot-va \
LINGBOT_VA_HOST=127.0.0.1 \
LINGBOT_VA_PORT=29056 \
./eval_system/scripts/run/start_model_server.sh \
  --env env-lingbot-va \
  --adapter lingbot-va \
  --port 50051
```

## LDA-1B

Default command:

```bash
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env env-lda-1b \
  --repo models/LDA-1B
```

The default install follows the LDA-1B README:

```bash
conda create -n env-lda-1b python=3.10 -y
conda activate env-lda-1b
pip install -r models/LDA-1B/requirements.txt
pip install flash-attn --no-build-isolation
pip install --no-deps -e models/LDA-1B
```

Useful options:

```bash
# Skip requirements when you are repairing an existing env
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env env-lda-1b \
  --repo models/LDA-1B \
  --skip-requirements

# Skip flash-attn if your PyTorch/CUDA build cannot compile it yet
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env env-lda-1b \
  --repo models/LDA-1B \
  --skip-flash-attn
```

The script does not download LDA checkpoints, Qwen checkpoints, or DINO
checkpoints. Put those on disk following the LDA README, then start the native
LDA server from the LDA env:

```bash
conda activate env-lda-1b
cd /home/yangc/Lab/API/models/LDA-1B
python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/your/lda/checkpoint \
  --port 10093 \
  --use_bf16
```

In a second terminal:

```bash
cd /home/yangc/Lab/API
LDA_1B_ROOT=/home/yangc/Lab/API/models/LDA-1B \
LDA_1B_HOST=127.0.0.1 \
LDA_1B_PORT=10093 \
./eval_system/scripts/run/start_model_server.sh \
  --env env-lda-1b \
  --adapter lda-1b \
  --port 50051
```

LDA's native policy server usually returns `normalized_actions`. If the target
simulator needs physical actions, point the adapter at the checkpoint run's
`dataset_statistics.json` or pass equivalent normalization stats in task
metadata:

```json
{
  "metadata": {
    "models": {
      "lda-1b": {
        "checkpoint_path": "/path/to/run/checkpoints/steps_200000_pytorch_model.pt",
        "unnorm_key": "robotwin",
        "action_type": "qpos",
        "expected_action_dim": 16
      }
    }
  }
}
```

For LDA checkpoints that require an embodiment id or specific observation keys,
set them through metadata:

```json
{
  "metadata": {
    "payload_mode": "examples",
    "image_keys": ["primary"],
    "state_keys": ["agent_pos"],
    "embodiment_id": 0,
    "state_transform": "none"
  }
}
```

Use `state_transform: "sin_cos"` for LDA variants trained with sin/cos state
features.

## RoboTwin

Default command:

```bash
bash eval_system/scripts/envs/create_robotwin_env.sh \
  --env env-robotwin \
  --repo simulators/RoboTwin
```

Default mode follows the official RoboTwin installer:

```bash
conda create -n env-robotwin python=3.10 -y
conda activate env-robotwin
cd simulators/RoboTwin
bash script/_install.sh
```

Assets are not downloaded by default. Add `--download-assets` only when you
want the script to run `script/_download_assets.sh`.

Manual fallback:

```bash
bash eval_system/scripts/envs/create_robotwin_env.sh \
  --env env-robotwin \
  --repo simulators/RoboTwin \
  --manual
```

Run the simulator server:

```bash
ROBOTWIN_ROOT=/home/yangc/Lab/API/simulators/RoboTwin \
./eval_system/scripts/run/start_sim_server.sh \
  --env env-robotwin \
  --adapter robotwin \
  --port 50052
```

## LIBERO

The LIBERO script follows the official install order: requirements, the
official torch/cu113 wheel set, then editable install.

```bash
bash eval_system/scripts/envs/create_libero_env.sh \
  --env env-libero \
  --repo /home/yangc/Lab/API/simulators/LIBERO
```

Skip the torch install if you already installed a GPU-specific PyTorch build:

```bash
bash eval_system/scripts/envs/create_libero_env.sh \
  --env env-libero \
  --repo /home/yangc/Lab/API/simulators/LIBERO \
  --skip-torch
```

Datasets are not downloaded by the script. Run LIBERO's official dataset
download command separately if needed.

## SimplerEnv

The SimplerEnv script expects a checkout with the `ManiSkill2_real2sim`
submodule:

```text
SimplerEnv/
  ManiSkill2_real2sim/
```

Install it with:

```bash
bash eval_system/scripts/envs/create_simplerenv_env.sh \
  --env env-simpler \
  --repo /path/to/SimplerEnv
```

Install optional full requirements:

```bash
bash eval_system/scripts/envs/create_simplerenv_env.sh \
  --env env-simpler \
  --repo /path/to/SimplerEnv \
  --full-requirements
```

## Notes

These scripts are intentionally path-based. They assume the upstream repos are
already on disk, which avoids accidental large downloads. Checkpoints, model
weights, datasets, and simulator assets should be downloaded explicitly by you
following each upstream project's current documentation.
