from __future__ import annotations

from ..common.metrics import aggregate_episode_metrics
from ..common.schemas import EpisodeLog, EvalReport


def build_report(benchmark: str, task: str, episodes: list[EpisodeLog]) -> EvalReport:
    returns = [episode.episode_return for episode in episodes]
    lengths = [episode.steps for episode in episodes]
    successes = [episode.success for episode in episodes]
    summary = aggregate_episode_metrics(returns, lengths, successes)
    errors = [episode.avg_prediction_error for episode in episodes if episode.avg_prediction_error is not None]
    avg_prediction_error = sum(errors) / len(errors) if errors else None
    action_metrics = _action_metrics(episodes)
    return EvalReport(
        benchmark=benchmark,
        task=task,
        episodes=len(episodes),
        success_rate=summary["success_rate"],
        avg_return=summary["avg_return"],
        avg_episode_length=summary["avg_episode_length"],
        avg_prediction_error=avg_prediction_error,
        metrics={
            "episodes_completed": len(episodes),
            "prediction_enabled": bool(errors),
            **action_metrics,
        },
        episode_summaries=[episode.to_dict() for episode in episodes],
    )


def _action_metrics(episodes: list[EpisodeLog]) -> dict:
    total_actions = 0
    repeated_actions = 0
    nonfinite_actions = 0
    norms: list[float] = []
    dims: set[int] = set()
    for episode in episodes:
        for record in episode.records:
            diagnostics = record.get("action_diagnostics") if isinstance(record, dict) else None
            if not isinstance(diagnostics, dict):
                continue
            total_actions += 1
            dim = diagnostics.get("dim")
            if isinstance(dim, int):
                dims.add(dim)
            norm = diagnostics.get("l2_norm")
            if isinstance(norm, (int, float)):
                norms.append(float(norm))
            if diagnostics.get("repeated_previous_action"):
                repeated_actions += 1
            if diagnostics.get("all_finite") is False:
                nonfinite_actions += 1

    if total_actions == 0:
        return {"action_diagnostics_available": False}
    repeat_rate = repeated_actions / total_actions
    avg_norm = sum(norms) / len(norms) if norms else None
    max_norm = max(norms) if norms else None
    collapse_suspected = repeat_rate >= 0.9 or (max_norm is not None and max_norm <= 1e-8)
    return {
        "action_diagnostics_available": True,
        "action_dim_set": sorted(dims),
        "action_repeat_rate": round(repeat_rate, 6),
        "nonfinite_action_rate": round(nonfinite_actions / total_actions, 6),
        "avg_action_l2_norm": round(avg_norm, 8) if avg_norm is not None else None,
        "min_action_l2_norm": round(min(norms), 8) if norms else None,
        "max_action_l2_norm": round(max_norm, 8) if max_norm is not None else None,
        "action_collapse_suspected": collapse_suspected,
    }
