from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.metadata import deep_update
from ..common.schemas import EvalSpec, TaskSpec
from ..model_server.client import WorldModelClient
from ..sim_server.client import SimulatorClient
from .logging import StructuredLogger
from .report import build_report
from .rollout import run_episode


def load_task_spec(config_root: Path, benchmark: str, task: str, horizon: int, seed: int) -> TaskSpec:
    candidate = config_root / "tasks" / f"{task}.json"
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload.setdefault("benchmark", benchmark)
        payload.setdefault("task", task)
        payload.setdefault("episode_horizon", horizon)
        payload.setdefault("seed", seed)
        return TaskSpec.from_dict(payload)
    return TaskSpec(
        benchmark=benchmark,
        task=task,
        episode_horizon=horizon,
        seed=seed,
        instruction=f"Solve task {task} in benchmark {benchmark}.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoupled evaluation")
    parser.add_argument("--model_endpoint", required=True)
    parser.add_argument("--sim_endpoint", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="policy")
    parser.add_argument("--predict_next", action="store_true")
    parser.add_argument("--output_dir", default="results/eval")
    parser.add_argument("--metadata_json", "--metadata-json", default="", help="JSON object merged into TaskSpec.metadata")
    args = parser.parse_args()

    spec = EvalSpec(
        model_endpoint=args.model_endpoint,
        sim_endpoint=args.sim_endpoint,
        benchmark=args.benchmark,
        task=args.task,
        episodes=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
        mode=args.mode,
        predict_next=args.predict_next,
        output_dir=args.output_dir,
    )

    config_root = Path(__file__).resolve().parent.parent / "configs"
    task_spec = load_task_spec(config_root, args.benchmark, args.task, args.horizon, args.seed)
    if args.metadata_json:
        metadata = dict(task_spec.metadata)
        deep_update(metadata, json.loads(args.metadata_json))
        task_spec = TaskSpec(
            benchmark=task_spec.benchmark,
            task=task_spec.task,
            episode_horizon=task_spec.episode_horizon,
            seed=task_spec.seed,
            instruction=task_spec.instruction,
            metadata=metadata,
        )
    model_client = WorldModelClient(spec.model_endpoint)
    sim_client = SimulatorClient(spec.sim_endpoint)
    logger = StructuredLogger(spec.output_dir)

    episodes = []
    for episode_index in range(spec.episodes):
        episode_task = TaskSpec(
            benchmark=task_spec.benchmark,
            task=task_spec.task,
            episode_horizon=task_spec.episode_horizon,
            seed=task_spec.seed + episode_index,
            instruction=task_spec.instruction,
            metadata=task_spec.metadata,
        )
        episode_log = run_episode(episode_index, spec, episode_task, model_client, sim_client)
        logger.write_episode(episode_log)
        episodes.append(episode_log)

    sim_client.close()
    report = build_report(spec.benchmark, spec.task, episodes)
    logger.write_report(report)
    logger.write_summary_csv(report)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
