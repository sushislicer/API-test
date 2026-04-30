from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import Action, Observation, StepResult, TaskSpec, TrainingSpec


class ModelAdapter(ABC):
    """Canonical interface for a world model service."""

    @abstractmethod
    def reset(self, task: TaskSpec) -> dict:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    @abstractmethod
    def predict_next(self, observation: Observation, action: Action) -> Observation:
        raise NotImplementedError

    def train(self, spec: TrainingSpec) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support training")

    def training_status(self, job_id: str | None = None) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support training jobs")

    def rollout(self, observation: Observation, action_sequence: list[Action]) -> list[Observation]:
        predictions: list[Observation] = []
        current = observation
        for action in action_sequence:
            current = self.predict_next(current, action)
            predictions.append(current)
        return predictions


class SimulatorAdapter(ABC):
    """Canonical interface for a simulator service."""

    @abstractmethod
    def reset(self, task: TaskSpec) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> dict:
        raise NotImplementedError
