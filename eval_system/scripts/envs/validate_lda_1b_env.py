#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


MODULES = [
    "torch",
    "torchvision",
    "accelerate",
    "transformers",
    "diffusers",
    "websockets",
    "msgpack",
    "deployment.model_server.server_policy",
    "eval_system",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an LDA-1B API conda environment")
    parser.add_argument("--api-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    api_root = Path(args.api_root).resolve()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(api_root))
    sys.path.insert(0, str(repo))

    missing = []
    for module in MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            missing.append(exc.name or module)

    if missing:
        print("missing LDA-1B environment modules: " + ", ".join(sorted(set(missing))), file=sys.stderr)
        print("repair with:", file=sys.stderr)
        print(
            f"  bash eval_system/scripts/envs/create_lda_1b_env.sh --env {args.env} --repo {repo} --repair-requirements",
            file=sys.stderr,
        )
        return 1

    import torch

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={cuda_available}")
    print(f"torch.cuda.device_count={device_count}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    if device_count:
        try:
            print(f"torch.cuda.device0={torch.cuda.get_device_name(0)}")
        except Exception as exc:  # pragma: no cover - diagnostic best effort
            print(f"torch.cuda.device0_error={type(exc).__name__}: {exc}", file=sys.stderr)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode == 0:
            print("nvidia-smi=" + proc.stdout.strip().replace("\n", " | "))
        else:
            print(f"nvidia-smi failed: {proc.stderr.strip()}", file=sys.stderr)
    else:
        print("nvidia-smi=<not found>")

    if args.require_cuda and not cuda_available:
        print("CUDA validation failed.", file=sys.stderr)
        if torch.version.cuda is None:
            print("The installed torch build is CPU-only. Reinstall the LDA torch CUDA wheel set:", file=sys.stderr)
            print(
                f"  bash eval_system/scripts/envs/create_lda_1b_env.sh --env {args.env} --repo {repo} "
                "--skip-requirements --skip-flash-attn --no-editable --require-cuda",
                file=sys.stderr,
            )
        else:
            print(
                "Torch has CUDA support, but no CUDA device is visible. Check that this shell is on a GPU node, "
                "the NVIDIA driver is mounted, and CUDA_VISIBLE_DEVICES is not empty.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
