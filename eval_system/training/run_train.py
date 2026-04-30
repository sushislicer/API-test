from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common.metadata import deep_update
from ..common.schemas import TrainingSpec
from ..evaluator.run_eval import load_task_spec
from ..model_server.client import WorldModelClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or launch model training through the model API")
    parser.add_argument("--model_endpoint", required=True)
    parser.add_argument("--model", required=True, help="Model adapter name, e.g. lingbot-va or lda-1b")
    parser.add_argument("--benchmark", required=True, help="Simulator/benchmark name, e.g. libero")
    parser.add_argument("--task", required=True, help="Task config name under eval_system/configs/tasks")
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--output_dir", default="results/train")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--config_name", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metadata_json", default="", help="Extra JSON object merged into TrainingSpec.metadata")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Training parameter override scoped to this model and benchmark. Repeat as needed.",
    )
    parser.add_argument("--launch", action="store_true", help="Launch training. Default is dry-run only.")
    args = parser.parse_args()

    config_root = Path(__file__).resolve().parent.parent / "configs"
    task_spec = load_task_spec(config_root, args.benchmark, args.task, horizon=0, seed=args.seed)
    metadata = dict(task_spec.metadata)
    metadata["simulator"] = args.benchmark

    if args.metadata_json:
        deep_update(metadata, json.loads(args.metadata_json))

    if args.param:
        scoped_training = metadata.setdefault("training", {})
        model_training = scoped_training.setdefault(args.model, {})
        benchmark_training = model_training.setdefault(args.benchmark, {})
        for item in args.param:
            key, value = _parse_param(item)
            benchmark_training[key] = value

    spec = TrainingSpec(
        benchmark=args.benchmark,
        dataset_path=args.dataset_path,
        config_name=args.config_name,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        dry_run=not args.launch,
        metadata=metadata,
    )
    response = WorldModelClient(args.model_endpoint).train(spec)
    print(json.dumps(response, indent=2))


def _parse_param(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected --param in KEY=VALUE form, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Expected non-empty parameter name in: {raw}")
    return key, _parse_value(value)


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    main()
