from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.metadata import deep_update, scoped_metadata
from ..common.serialization import ensure_dir, normalize_endpoint
from ..common.schemas import Action, Observation, TaskSpec
from ..evaluator.run_eval import load_task_spec
from ..model_server.client import WorldModelClient
from ..sim_server.client import SimulatorClient


@dataclass
class Issue:
    severity: str
    message: str
    hint: str = ""
    path: str = ""


@dataclass
class PreflightReport:
    ok: bool
    generated_at: str
    benchmark: str
    task: str
    model_adapter: str
    simulator_adapter: str
    issues: list[Issue] = field(default_factory=list)
    contract: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static API preflight checks without launching model/simulator code")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-adapter", "--model_adapter", default="", help="Model adapter name, e.g. lda-1b")
    parser.add_argument("--sim-adapter", "--sim_adapter", default="", help="Simulator adapter name, e.g. robotwin")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-root", default="")
    parser.add_argument("--metadata-json", "--metadata_json", default="", help="JSON object merged into TaskSpec.metadata")
    parser.add_argument("--model-endpoint", "--model_endpoint", default="")
    parser.add_argument("--sim-endpoint", "--sim_endpoint", default="")
    parser.add_argument("--check-endpoints", action="store_true", help="Call /health on configured endpoints")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()

    report = run_preflight(args)
    print(json.dumps(report.to_dict(), indent=2))
    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("results") / "preflight" / f"{args.benchmark}_{args.task}_{stamp}.json"
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if not report.ok:
        raise SystemExit(1)


def run_preflight(args: argparse.Namespace) -> PreflightReport:
    model_adapter = args.model_adapter or _default_model_adapter(args.benchmark)
    sim_adapter = args.sim_adapter or args.benchmark
    issues: list[Issue] = []
    checks: dict[str, Any] = {}
    contract: dict[str, Any] = {}

    config_root = Path(args.config_root).expanduser().resolve() if args.config_root else Path(__file__).resolve().parents[1] / "configs"
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

    _check_registry(model_adapter, sim_adapter, issues)
    _check_transport_roundtrip(task_spec, checks, issues)
    _check_endpoints(args, checks, issues)
    _check_contract(task_spec, model_adapter, sim_adapter, contract, issues)

    ok = not any(issue.severity == "error" for issue in issues)
    return PreflightReport(
        ok=ok,
        generated_at=datetime.now(timezone.utc).isoformat(),
        benchmark=args.benchmark,
        task=args.task,
        model_adapter=model_adapter,
        simulator_adapter=sim_adapter,
        issues=issues,
        contract=contract,
        checks=checks,
    )


def _check_registry(model_adapter: str, sim_adapter: str, issues: list[Issue]) -> None:
    from ..model_server import adapters as _model_adapters  # noqa: F401
    from ..common.registry import MODEL_ADAPTERS, SIM_ADAPTERS
    from ..sim_server import adapters as _sim_adapters  # noqa: F401

    if model_adapter not in MODEL_ADAPTERS:
        issues.append(
            Issue(
                "error",
                f"Unknown model adapter '{model_adapter}'",
                f"Available model adapters: {sorted(MODEL_ADAPTERS)}",
                "model_adapter",
            )
        )
    if sim_adapter not in SIM_ADAPTERS:
        issues.append(
            Issue(
                "error",
                f"Unknown simulator adapter '{sim_adapter}'",
                f"Available simulator adapters: {sorted(SIM_ADAPTERS)}",
                "sim_adapter",
            )
        )


def _check_transport_roundtrip(task_spec: TaskSpec, checks: dict[str, Any], issues: list[Issue]) -> None:
    try:
        roundtrip = TaskSpec.from_dict(json.loads(json.dumps(task_spec.to_dict())))
        checks["task_spec_roundtrip"] = roundtrip.task == task_spec.task
        action = Action(vector=[0.0] * 7)
        action_roundtrip = Action.from_dict(json.loads(json.dumps(action.to_dict())))
        observation = Observation(images={"dummy": [0]}, state={"state": [0.0, 1.0]})
        observation_roundtrip = Observation.from_dict(json.loads(json.dumps(observation.to_dict())))
        checks["action_roundtrip_dim"] = len(action_roundtrip.to_vector())
        checks["observation_roundtrip_images"] = sorted(observation_roundtrip.images.keys())
    except Exception as exc:
        issues.append(Issue("error", f"Schema JSON roundtrip failed: {type(exc).__name__}: {exc}", path="schemas"))


def _check_endpoints(args: argparse.Namespace, checks: dict[str, Any], issues: list[Issue]) -> None:
    for name, endpoint in (("model_endpoint", args.model_endpoint), ("sim_endpoint", args.sim_endpoint)):
        if not endpoint:
            continue
        try:
            checks[f"{name}_normalized"] = normalize_endpoint(endpoint)
        except ValueError as exc:
            issues.append(Issue("error", str(exc), path=name))

    if not args.check_endpoints:
        return
    if args.model_endpoint:
        try:
            checks["model_health"] = WorldModelClient(args.model_endpoint).health()
        except Exception as exc:
            issues.append(Issue("error", f"Model endpoint health check failed: {type(exc).__name__}: {exc}", path="model_endpoint"))
    if args.sim_endpoint:
        try:
            checks["sim_health"] = SimulatorClient(args.sim_endpoint).health()
        except Exception as exc:
            issues.append(Issue("error", f"Simulator endpoint health check failed: {type(exc).__name__}: {exc}", path="sim_endpoint"))


def _check_contract(
    task_spec: TaskSpec,
    model_adapter: str,
    sim_adapter: str,
    contract: dict[str, Any],
    issues: list[Issue],
) -> None:
    model_metadata = scoped_metadata(task_spec.metadata, model_adapter, "models", "model")
    sim_metadata = scoped_metadata(task_spec.metadata, sim_adapter, "simulators", "simulator")
    if model_adapter == "lda-1b":
        from ..model_server.adapters.lda_1b import _load_action_norm_stats, _with_profile_defaults

        model_metadata = _with_profile_defaults(model_metadata, task_spec.benchmark)
        try:
            stats = _load_action_norm_stats(model_metadata)
        except Exception as exc:
            stats = None
            issues.append(
                Issue(
                    "error",
                    f"LDA action normalization stats could not be loaded: {type(exc).__name__}: {exc}",
                    "Set checkpoint_path/policy_ckpt_path, norm_stats_path/dataset_statistics_path, and unnorm_key correctly.",
                    "metadata.models.lda-1b",
                )
            )
    else:
        stats = None

    contract["model_metadata_keys"] = _contract_keys(model_metadata)
    contract["sim_metadata_keys"] = _contract_keys(sim_metadata)

    if sim_adapter == "robotwin":
        _check_robotwin_contract(task_spec, model_adapter, model_metadata, sim_metadata, stats, contract, issues)


def _check_robotwin_contract(
    task_spec: TaskSpec,
    model_adapter: str,
    model_metadata: dict[str, Any],
    sim_metadata: dict[str, Any],
    stats: dict[str, Any] | None,
    contract: dict[str, Any],
    issues: list[Issue],
) -> None:
    sim_action_type = str(sim_metadata.get("action_type", "qpos"))
    sim_action_dim = _optional_int(sim_metadata.get("expected_action_dim") or sim_metadata.get("action_dim")) or 16
    model_action_type = str(model_metadata.get("action_type", "qpos" if model_adapter == "lda-1b" else ""))
    model_expected_dim = _optional_int(model_metadata.get("expected_action_dim") or model_metadata.get("action_dim"))
    action_indices = _int_list(model_metadata.get("action_indices"))
    action_slice = _optional_slice(model_metadata.get("action_slice") or model_metadata.get("action_range"))
    projected_dim = _projected_dim(action_slice, action_indices)
    if projected_dim is not None:
        model_expected_dim = projected_dim

    image_keys = _string_list(model_metadata.get("image_keys") or model_metadata.get("image_key"))
    state_keys = _string_list(model_metadata.get("state_keys") or model_metadata.get("state_key"))
    expected_image_count = _optional_int(model_metadata.get("expected_image_count") or model_metadata.get("image_count"))
    expected_state_dim = _optional_int(model_metadata.get("expected_state_dim") or model_metadata.get("state_dim"))
    state_transform = str(model_metadata.get("state_transform", "none"))
    likely_input_state_dim = 16 * 2 if state_transform == "sin_cos" and state_keys == ["joint_action.vector"] else 16
    stats_dim = _stats_dim(stats, model_metadata)

    contract.update(
        {
            "sim_action_type": sim_action_type,
            "sim_action_dim": sim_action_dim,
            "model_action_type": model_action_type,
            "model_expected_action_dim": model_expected_dim,
            "model_projected_action_dim": projected_dim,
            "normalization_stats_dim": stats_dim,
            "image_keys": image_keys,
            "expected_image_count": expected_image_count,
            "state_keys": state_keys,
            "state_transform": state_transform,
            "expected_state_dim": expected_state_dim,
            "likely_robotwin_state_dim": likely_input_state_dim,
        }
    )

    if model_action_type and model_action_type != sim_action_type:
        issues.append(
            Issue(
                "error",
                f"Model action_type '{model_action_type}' does not match RoboTwin simulator action_type '{sim_action_type}'",
                "Use qpos for RoboTwin 16D joint actions unless you have explicitly configured an ee controller path.",
                "metadata.models",
            )
        )
    if model_expected_dim is None:
        issues.append(
            Issue(
                "warning",
                "Model expected_action_dim is not set",
                "Set expected_action_dim/action_dim so oversized LDA chunks cannot silently reach RoboTwin.",
                "metadata.models",
            )
        )
    elif model_expected_dim != sim_action_dim:
        issues.append(
            Issue(
                "error",
                f"Model projects to {model_expected_dim} action values but RoboTwin expects {sim_action_dim}",
                "Train/fine-tune with the RoboTwin action contract or configure a deliberate action_slice/action_indices projection.",
                "metadata.models",
            )
        )

    if stats_dim is None:
        action_output_key = str(model_metadata.get("action_output_key", ""))
        allow_normalized = _bool_value(model_metadata.get("allow_normalized_actions"), False)
        if model_adapter == "lda-1b" and not allow_normalized and action_output_key in {"", "normalized_actions"}:
            issues.append(
                Issue(
                    "warning",
                    "LDA usually returns normalized_actions but no action normalization stats were found",
                    "Set checkpoint_path/policy_ckpt_path or norm_stats_path/dataset_statistics_path with unnorm_key.",
                    "metadata.models.lda-1b",
                )
            )
    elif model_expected_dim is not None:
        if action_indices:
            max_index = max(action_indices)
            if max_index >= stats_dim:
                issues.append(Issue("error", f"action_indices reference index {max_index} but stats/action output dim is {stats_dim}", path="metadata.models"))
        elif action_slice is not None:
            start, end = action_slice
            if start is not None and start < 0:
                issues.append(Issue("error", "Negative action_slice start is not supported by static preflight", path="metadata.models"))
            if end is not None and end > stats_dim:
                issues.append(Issue("error", f"action_slice end {end} exceeds stats/action output dim {stats_dim}", path="metadata.models"))
        elif stats_dim != model_expected_dim:
            issues.append(
                Issue(
                    "error",
                    f"Normalization stats have dim {stats_dim} but model expected_action_dim is {model_expected_dim}",
                    "Do not rely on implicit truncation. Configure action_slice/action_indices only if the semantics are known.",
                    "metadata.models.lda-1b",
                )
            )

    if expected_image_count is not None and image_keys and len(image_keys) != expected_image_count:
        issues.append(
            Issue(
                "error",
                f"expected_image_count={expected_image_count} but {len(image_keys)} image_keys are configured",
                "Align image_keys with the checkpoint training views.",
                "metadata.models",
            )
        )
    if expected_state_dim is not None and state_keys == ["joint_action.vector"] and expected_state_dim != likely_input_state_dim:
        issues.append(
            Issue(
                "error",
                f"Expected state_dim={expected_state_dim}, but RoboTwin joint_action.vector with state_transform={state_transform} is likely {likely_input_state_dim}",
                "Use state_transform=none for 16D qpos state, or expected_state_dim=32 when applying sin_cos.",
                "metadata.models",
            )
        )

    instruction_source = str(sim_metadata.get("instruction_source", "env")).lower()
    require_prompt = _bool_value(model_metadata.get("require_prompt"), True)
    if require_prompt and instruction_source != "env" and not (task_spec.instruction or model_metadata.get("prompt") or sim_metadata.get("instruction")):
        issues.append(
            Issue(
                "error",
                "No non-empty prompt is configured and simulator instruction_source is not env",
                "Use RoboTwin's env instruction or set task instruction/model prompt explicitly.",
                "metadata",
            )
        )


def _stats_dim(stats: dict[str, Any] | None, metadata: dict[str, Any]) -> int | None:
    action_min = stats.get("min") if stats else metadata.get("action_min") or metadata.get("action_low")
    action_max = stats.get("max") if stats else metadata.get("action_max") or metadata.get("action_high")
    min_dim = _last_dim(action_min)
    max_dim = _last_dim(action_max)
    if min_dim is None and max_dim is None:
        return None
    if min_dim != max_dim:
        return int(max(filter(lambda item: item is not None, (min_dim, max_dim))))
    return min_dim


def _last_dim(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        shape = value.shape
        return int(shape[-1]) if shape else None
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, (list, tuple)):
            return _last_dim(first)
        return len(value)
    return None


def _projected_dim(action_slice: tuple[int | None, int | None] | None, action_indices: list[int]) -> int | None:
    if action_indices:
        return len(action_indices)
    if action_slice is None:
        return None
    start, end = action_slice
    if start is None or end is None:
        return None
    return max(0, end - start)


def _default_model_adapter(benchmark: str) -> str:
    if benchmark.lower() == "robotwin":
        return "lda-1b"
    return "dummy-model"


def _contract_keys(metadata: dict[str, Any]) -> list[str]:
    return sorted(key for key in metadata if key not in {"models", "simulators", "training"})


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _optional_slice(value: Any) -> tuple[int | None, int | None] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("action_slice must be START:END or [START, END]")
        return (_slice_part(parts[0]), _slice_part(parts[1]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_slice_part(value[0]), _slice_part(value[1]))
    raise ValueError("action_slice must be START:END or [START, END]")


def _slice_part(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


if __name__ == "__main__":
    main()
