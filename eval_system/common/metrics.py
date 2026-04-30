from __future__ import annotations

from math import sqrt

from .schemas import Observation


def l2_distance(values_a: list[float], values_b: list[float]) -> float:
    length = min(len(values_a), len(values_b))
    if length == 0:
        return 0.0
    total = 0.0
    for a, b in zip(values_a[:length], values_b[:length]):
        total += (a - b) ** 2
    return sqrt(total)


def observation_error(predicted: Observation, actual: Observation) -> float:
    proprio_error = l2_distance(predicted.proprio, actual.proprio)
    gripper_pred = predicted.gripper if predicted.gripper is not None else 0.0
    gripper_actual = actual.gripper if actual.gripper is not None else 0.0
    gripper_error = abs(gripper_pred - gripper_actual)
    return proprio_error + gripper_error


def aggregate_episode_metrics(episode_returns: list[float], episode_lengths: list[int], successes: list[bool]) -> dict:
    count = max(len(episode_returns), 1)
    return {
        "success_rate": sum(1 for item in successes if item) / count,
        "avg_return": sum(episode_returns) / count,
        "avg_episode_length": sum(episode_lengths) / count,
    }
