from __future__ import annotations

from ...common.base import SimulatorAdapter
from ...common.registry import register_sim
from ...common.schemas import Action, Observation, StepResult, TaskSpec


class DummySimulatorAdapter(SimulatorAdapter):
    def __init__(self, simulator_name: str = "dummy-simulator") -> None:
        self.simulator_name = simulator_name
        self.task: TaskSpec | None = None
        self.position = [1.0, 1.0, 1.0]
        self.step_count = 0
        self.closed = False

    def reset(self, task: TaskSpec) -> Observation:
        self.task = task
        self.position = [1.0, 1.0, 1.0]
        self.step_count = 0
        self.closed = False
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        vector = action.to_vector()
        translation = vector[:3] if len(vector) >= 3 else action.translation
        self.position = [
            round(self.position[0] + translation[0], 4),
            round(self.position[1] + translation[1], 4),
            round(self.position[2] + translation[2], 4),
        ]
        distance = sum(abs(value) for value in self.position)
        success = distance < 0.15
        done = success or (self.task is not None and self.step_count >= self.task.episode_horizon)
        reward = 1.0 if success else -0.05 * distance
        info = {
            "simulator_name": self.simulator_name,
            "distance_to_goal_l1": round(distance, 4),
            "step_count": self.step_count,
        }
        return StepResult(
            observation=self._observation(gripper=action.gripper),
            reward=round(reward, 4),
            done=done,
            success=success,
            info=info,
        )

    def get_state(self) -> dict:
        return {
            "simulator_name": self.simulator_name,
            "position": self.position,
            "step_count": self.step_count,
            "task": self.task.to_dict() if self.task else None,
            "closed": self.closed,
        }

    def close(self) -> dict:
        self.closed = True
        return {"closed": True, "simulator_name": self.simulator_name}

    def _observation(self, gripper: float = 0.0) -> Observation:
        instruction = self.task.instruction if self.task else ""
        task_name = self.task.task if self.task else "unknown"
        return Observation(
            rgb=f"sim://{self.simulator_name}/{task_name}/step-{self.step_count}.png",
            depth=None,
            images={},
            depth_images={},
            proprio=list(self.position),
            gripper=gripper,
            language_instruction=instruction,
            objects={"goal": [0.0, 0.0, 0.0]},
            state={"position": list(self.position)},
            metadata={
                "simulator_name": self.simulator_name,
                "task": task_name,
                "step_count": self.step_count,
            },
        )


@register_sim("dummy-simulator")
class RegisteredDummySimulatorAdapter(DummySimulatorAdapter):
    def __init__(self) -> None:
        super().__init__(simulator_name="dummy-simulator")
