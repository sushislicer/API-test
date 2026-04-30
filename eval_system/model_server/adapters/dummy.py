from __future__ import annotations

from pathlib import Path

from ...common.base import ModelAdapter
from ...common.registry import register_model
from ...common.schemas import Action, Observation, TaskSpec, TrainingSpec
from ._training import TrainingJobManager, bool_value, external_training_request


class DummyModelAdapter(ModelAdapter):
    def __init__(self, model_name: str = "dummy-model") -> None:
        self.model_name = model_name
        self.task: TaskSpec | None = None
        self.step_count = 0

    def reset(self, task: TaskSpec) -> dict:
        self.task = task
        self.step_count = 0
        return {"model_name": self.model_name, "task": task.task, "seed": task.seed}

    def act(self, observation: Observation) -> Action:
        self.step_count += 1
        base = observation.proprio[:3] if len(observation.proprio) >= 3 else [0.0, 0.0, 0.0]
        translation = [
            round(-0.1 * base[0], 4),
            round(-0.1 * base[1], 4),
            round(-0.1 * base[2], 4),
        ]
        gripper = 1.0 if observation.gripper is not None and observation.gripper < 0.5 else 0.0
        return Action(
            type="delta_ee_pose",
            action_type="delta_ee_pose",
            vector=[*translation, 0.0, 0.0, 0.0, gripper],
            translation=translation,
            rotation=[0.0, 0.0, 0.0],
            gripper=gripper,
            metadata={"policy_step": self.step_count, "model_name": self.model_name},
        )

    def predict_next(self, observation: Observation, action: Action) -> Observation:
        proprio = list(observation.proprio)
        while len(proprio) < 3:
            proprio.append(0.0)
        next_proprio = [
            round(proprio[0] + action.translation[0], 4),
            round(proprio[1] + action.translation[1], 4),
            round(proprio[2] + action.translation[2], 4),
        ]
        return Observation(
            rgb=observation.rgb,
            depth=observation.depth,
            images=observation.images,
            depth_images=observation.depth_images,
            proprio=next_proprio,
            gripper=action.gripper,
            language_instruction=observation.language_instruction,
            objects=observation.objects,
            state=observation.state,
            metadata={
                **observation.metadata,
                "predicted_by": self.model_name,
            },
        )


class ExternalTrainingDummyAdapter(DummyModelAdapter):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self.training_jobs = TrainingJobManager(model_name)

    def train(self, spec: TrainingSpec) -> dict:
        request = external_training_request(
            self.model_name,
            spec,
            default_cwd=_api_root(),
            default_log_name=f"{self.model_name}_train.log",
        )
        return self.training_jobs.handle_train(request, bool_value(spec.dry_run, True))

    def training_status(self, job_id: str | None = None) -> dict:
        return self.training_jobs.status(job_id)


def _api_root() -> Path:
    return Path(__file__).resolve().parents[3]


@register_model("dummy-model")
class RegisteredDummyModelAdapter(DummyModelAdapter):
    def __init__(self) -> None:
        super().__init__(model_name="dummy-model")
