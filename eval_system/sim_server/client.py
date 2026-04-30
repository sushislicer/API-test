from __future__ import annotations

from ..common.http import post_json
from ..common.schemas import Action, Observation, StepResult, TaskSpec


class SimulatorClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def reset(self, task: TaskSpec) -> Observation:
        payload = post_json(self.endpoint, "/reset", task.to_dict())
        return Observation.from_dict(payload)

    def health(self) -> dict:
        return post_json(self.endpoint, "/health", {})

    def step(self, action: Action) -> StepResult:
        payload = post_json(self.endpoint, "/step", action.to_dict())
        return StepResult.from_dict(payload)

    def get_state(self) -> dict:
        return post_json(self.endpoint, "/get_state", {})

    def close(self) -> dict:
        return post_json(self.endpoint, "/close", {})
