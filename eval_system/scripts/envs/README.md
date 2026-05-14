# Environment Setup Scripts

These scripts create conda environments from upstream repos you already have on
disk. They do not run automatically, download checkpoints, download simulator
assets, clone repos, or start servers.

Heavy artifacts should live under `downloads/` by default:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}"
```

On the remote A800 machine, point `DOWNLOAD_ROOT` at a larger mounted disk if
needed. The setup scripts only print this path; they do not download checkpoints
or assets unless an explicit option such as `--download-assets` is used.

By default, the scripts create checkout-local conda prefix envs under
`.conda-envs/`. This keeps the API test setup independent from pre-existing
remote envs such as `LDA` or `RoboTwin2`.

The scripts also default conda and pip caches to checkout-local paths:

```bash
export CONDA_PKGS_DIRS="${API_ROOT}/.conda-pkgs"
export PIP_CACHE_DIR="${API_ROOT}/.pip-cache"
```

This avoids corrupting or racing the remote machine's shared conda package
cache. On remote machines where the checkout is on a slow shared mount, set
`API_CONDA_PKGS_DIR` and `API_PIP_CACHE_DIR` to a local scratch disk before
running setup. If conda reports missing `.conda.partial` rename targets, clean
the API envs and package cache before retrying:

```bash
bash eval_system/scripts/envs/cleanup_api_envs.sh --yes --package-cache
```

Create the default API env set sequentially:

```bash
bash eval_system/scripts/envs/create_api_envs.sh
```

That creates:

```text
.conda-envs/api-lingbot-va
.conda-envs/api-lda-1b
.conda-envs/api-robotwin
```

If an earlier debug run created old named envs under the global conda env root
such as `api-lda-1b`, remove them before recreating the prefix envs:

```bash
bash eval_system/scripts/envs/cleanup_api_envs.sh --yes --named-only
```

To delete both the old named envs and any partial checkout-local `.conda-envs`
prefixes, run:

```bash
bash eval_system/scripts/envs/cleanup_api_envs.sh --yes
```

You can also fold named-env cleanup into setup:

```bash
bash eval_system/scripts/envs/create_api_envs.sh --clean-stale-named
```

You can also create or repair one env at a time. `--env` accepts either a conda
name or a prefix path:

```bash
# Model repos in this checkout
bash eval_system/scripts/envs/create_lingbot_va_env.sh --env .conda-envs/api-lingbot-va --repo models/lingbot-va
bash eval_system/scripts/envs/create_lda_1b_env.sh --env .conda-envs/api-lda-1b --repo models/LDA-1B

# Simulator repos
bash eval_system/scripts/envs/create_robotwin_env.sh --env .conda-envs/api-robotwin --repo simulators/RoboTwin
bash eval_system/scripts/envs/create_libero_env.sh --env .conda-envs/api-libero --repo /path/to/LIBERO
bash eval_system/scripts/envs/create_simplerenv_env.sh --env .conda-envs/api-simplerenv --repo /path/to/SimplerEnv
```

Each script links this API checkout into the target environment with a `.pth`
file, so `python -m eval_system...` works from that env without packaging this
repo.

## LingBot-VA

Default command:

```bash
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env .conda-envs/api-lingbot-va \
  --repo models/lingbot-va
```

The default install follows the LingBot-VA README inference dependencies:

```bash
conda create -p .conda-envs/api-lingbot-va python=3.10.16 -y
conda activate .conda-envs/api-lingbot-va
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
pip install --no-deps -e models/lingbot-va
```

Useful options:

```bash
# Use a different CUDA/PyTorch wheel index
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env .conda-envs/api-lingbot-va \
  --repo models/lingbot-va \
  --torch-index https://download.pytorch.org/whl/cu126

# Skip torch if the remote machine already has the right GPU build installed
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env .conda-envs/api-lingbot-va \
  --repo models/lingbot-va \
  --skip-torch

# Install LingBot's requirements.txt exactly instead of the README package list
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env .conda-envs/api-lingbot-va \
  --repo models/lingbot-va \
  --requirements

# Add the post-training extras from the LingBot README
bash eval_system/scripts/envs/create_lingbot_va_env.sh \
  --env .conda-envs/api-lingbot-va \
  --repo models/lingbot-va \
  --post-training
```

The script does not edit model checkpoint config. Before inference, follow the
LingBot README note and set the checkpoint `transformer/config.json`
`attn_mode` to `"torch"` or `"flashattn"`, not `"flex"`.

Run LingBot's native server from the LingBot env, then run this API adapter as a
client:

```bash
conda activate .conda-envs/api-lingbot-va
export API_ROOT=/path/to/API
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
cd "${API_ROOT}/models/lingbot-va"
START_PORT=29056 bash evaluation/robotwin/launch_server.sh
```

In a second terminal:

```bash
cd "${API_ROOT}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}" \
LINGBOT_VA_ROOT="${API_ROOT}/models/lingbot-va" \
LINGBOT_VA_HOST=127.0.0.1 \
LINGBOT_VA_PORT=29056 \
./eval_system/scripts/run/start_model_server.sh \
  --env .conda-envs/api-lingbot-va \
  --adapter lingbot-va \
  --port 50051
```

## LDA-1B

Default command:

```bash
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B
```

The default install follows the LDA-1B README:

```bash
conda create -p .conda-envs/api-lda-1b python=3.10 -y
conda activate .conda-envs/api-lda-1b
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r models/LDA-1B/requirements.txt
pip install flash-attn --no-build-isolation
pip install --no-deps -e models/LDA-1B
```

The explicit torch install uses the same versions pinned by
`models/LDA-1B/requirements.txt`; it just makes the remote CUDA wheel selection
and the failure point deterministic before the larger requirements install.

Useful options:

```bash
# Use a different CUDA/PyTorch wheel index
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --torch-index https://download.pytorch.org/whl/cu124

# Skip torch only if the existing env already has the correct LDA torch build
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --skip-torch

# Skip requirements when you are repairing an existing env
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --skip-requirements

# Skip flash-attn if your PyTorch/CUDA build cannot compile it yet
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --skip-flash-attn

# Re-run only the setup validation after a remote install
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --validate-only

# Repair a partial env that has torch but is missing later LDA install steps
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --repair-requirements

# On a GPU node, also fail validation if torch cannot see CUDA
bash eval_system/scripts/envs/create_lda_1b_env.sh \
  --env .conda-envs/api-lda-1b \
  --repo models/LDA-1B \
  --validate-only \
  --require-cuda
```

The script does not download LDA checkpoints, Qwen checkpoints, or DINO
checkpoints. Put those on disk following the LDA README, then start the native
LDA server from the LDA env:

```bash
conda activate .conda-envs/api-lda-1b
export API_ROOT=/path/to/API
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
cd "${API_ROOT}/models/LDA-1B"
python -m deployment.model_server.server_policy \
  --ckpt_path "${DOWNLOAD_ROOT}/checkpoints/lda-1b/path/to/checkpoint.pt" \
  --port 10093 \
  --use_bf16
```

In a second terminal:

```bash
cd "${API_ROOT}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${API_ROOT}/downloads}" \
LDA_1B_ROOT="${API_ROOT}/models/LDA-1B" \
LDA_1B_HOST=127.0.0.1 \
LDA_1B_PORT=10093 \
./eval_system/scripts/run/start_model_server.sh \
  --env .conda-envs/api-lda-1b \
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
  --env .conda-envs/api-robotwin \
  --repo simulators/RoboTwin
```

Default mode follows the official RoboTwin installer:

```bash
conda create -p .conda-envs/api-robotwin python=3.10 -y
conda activate .conda-envs/api-robotwin
cd simulators/RoboTwin
bash script/_install.sh
```

Assets are not downloaded by default. Add `--download-assets` only when you
want the script to run `script/_download_assets.sh`.

Manual fallback:

```bash
bash eval_system/scripts/envs/create_robotwin_env.sh \
  --env .conda-envs/api-robotwin \
  --repo simulators/RoboTwin \
  --manual
```

Run the simulator server:

```bash
export API_ROOT=/path/to/API
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
ROBOTWIN_ROOT="${API_ROOT}/simulators/RoboTwin" \
./eval_system/scripts/run/start_sim_server.sh \
  --env .conda-envs/api-robotwin \
  --adapter robotwin \
  --port 50052
```

## LIBERO

The LIBERO script follows the official install order: requirements, the
official torch/cu113 wheel set, then editable install.

```bash
bash eval_system/scripts/envs/create_libero_env.sh \
  --env .conda-envs/api-libero \
  --repo simulators/LIBERO
```

Skip the torch install if you already installed a GPU-specific PyTorch build:

```bash
bash eval_system/scripts/envs/create_libero_env.sh \
  --env .conda-envs/api-libero \
  --repo simulators/LIBERO \
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
  --env .conda-envs/api-simplerenv \
  --repo /path/to/SimplerEnv
```

Install optional full requirements:

```bash
bash eval_system/scripts/envs/create_simplerenv_env.sh \
  --env .conda-envs/api-simplerenv \
  --repo /path/to/SimplerEnv \
  --full-requirements
```

## Notes

These scripts are intentionally path-based. They assume the upstream repos are
already on disk, which avoids accidental large downloads. Checkpoints, model
weights, datasets, and simulator assets should be downloaded explicitly by you
following each upstream project's current documentation.
