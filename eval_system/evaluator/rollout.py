from __future__ import annotations

import math

from ..common.metrics import observation_error
from ..common.schemas import EpisodeLog, EvalSpec, TaskSpec
from ..model_server.client import WorldModelClient
from ..sim_server.client import SimulatorClient


def run_episode(
    episode_index: int,
    spec: EvalSpec,
    task_spec: TaskSpec,
    model_client: WorldModelClient,
    sim_client: SimulatorClient,
) -> EpisodeLog:
    initial_observation = sim_client.reset(task_spec)
    model_client.reset(task_spec)
    observation = initial_observation
    total_reward = 0.0
    prediction_errors: list[float] = []
    records: list[dict] = []
    success = False
    previous_action_vector: list[float] | None = None

    for step_index in range(spec.horizon):
        action = model_client.act(observation)
        action_diagnostics = _action_diagnostics(action.to_vector(), previous_action_vector)
        previous_action_vector = action.to_vector()
        predicted_observation = None
        if spec.predict_next:
            predicted_observation = model_client.predict_next(observation, action)
        step_result = sim_client.step(action)
        total_reward += step_result.reward
        prediction_error = None
        if predicted_observation is not None:
            prediction_error = observation_error(predicted_observation, step_result.observation)
            prediction_errors.append(prediction_error)
        record = {
            "step_index": step_index,
            "observation": observation.to_dict(),
            "action": action.to_dict(),
            "predicted_observation": predicted_observation.to_dict() if predicted_observation else None,
            "sim_result": step_result.to_dict(),
            "prediction_error": prediction_error,
            "action_diagnostics": action_diagnostics,
        }
        records.append(record)
        observation = step_result.observation
        success = step_result.success
        if step_result.done:
            return EpisodeLog(
                episode_index=episode_index,
                task=task_spec.task,
                steps=step_index + 1,
                episode_return=round(total_reward, 4),
                success=success,
                avg_prediction_error=round(sum(prediction_errors) / len(prediction_errors), 6)
                if prediction_errors
                else None,
                records=records,
            )

    return EpisodeLog(
        episode_index=episode_index,
        task=task_spec.task,
        steps=spec.horizon,
        episode_return=round(total_reward, 4),
        success=success,
        avg_prediction_error=round(sum(prediction_errors) / len(prediction_errors), 6)
        if prediction_errors
        else None,
        records=records,
    )


def _action_diagnostics(vector: list[float], previous: list[float] | None) -> dict:
    norm = math.sqrt(sum(value * value for value in vector))
    if previous is None or len(previous) != len(vector):
        delta_norm = None
        repeated = False
    else:
        delta_norm = math.sqrt(sum((value - prev) ** 2 for value, prev in zip(vector, previous)))
        repeated = delta_norm <= 1e-8
    return {
        "dim": len(vector),
        "l2_norm": round(norm, 8),
        "delta_from_previous_l2": round(delta_norm, 8) if delta_norm is not None else None,
        "repeated_previous_action": repeated,
        "all_finite": all(math.isfinite(value) for value in vector),
    }
