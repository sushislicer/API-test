from __future__ import annotations

from ..common.http import post_json
from ..common.schemas import Action, Observation, TaskSpec, TrainingSpec, TransitionQuery


class WorldModelClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def reset(self, task: TaskSpec) -> dict:
        return post_json(self.endpoint, "/reset", task.to_dict())

    def health(self) -> dict:
        return post_json(self.endpoint, "/health", {})

    def act(self, observation: Observation) -> Action:
        payload = post_json(self.endpoint, "/act", observation.to_dict())
        return Action.from_dict(payload)

    def predict_next(self, observation: Observation, action: Action) -> Observation:
        query = TransitionQuery(observation=observation, action=action)
        payload = post_json(self.endpoint, "/predict_next", query.to_dict())
        return Observation.from_dict(payload)

    def train(self, spec: TrainingSpec) -> dict:
        return post_json(self.endpoint, "/train", spec.to_dict())

    def training_status(self, job_id: str | None = None) -> dict:
        payload = {"job_id": job_id} if job_id else {}
        return post_json(self.endpoint, "/training_status", payload)
