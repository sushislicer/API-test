from __future__ import annotations

from importlib import import_module
from typing import Any

from ...common.base import SimulatorAdapter
from ...common.arrays import encode_array
from ...common.registry import register_sim
from ...common.schemas import Action, Observation, StepResult, TaskSpec
from ._native_utils import action_vector, extract_success, jsonable, native_observation, scalar_float


@register_sim("simplerenv")
class SimplerEnvAdapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.env: Any | None = None
        self.task: TaskSpec | None = None
        self.env_name = ""
        self.step_count = 0
        self.language_instruction = ""
        self.backend = "simpler_env"

    def reset(self, task: TaskSpec) -> Observation:
        self.close()
        self.task = task
        self.step_count = 0

        metadata = task.metadata
        self.env_name = str(metadata.get("env_name") or metadata.get("gym_id") or task.task)
        self.backend = str(metadata.get("backend", "simpler_env"))
        env_kwargs = dict(metadata.get("env_kwargs", {}))

        if self.backend == "gymnasium":
            registration_module = metadata.get("registration_module")
            if registration_module:
                import_module(str(registration_module))
            import gymnasium as gym

            self.env = gym.make(self.env_name, **env_kwargs)
        else:
            import simpler_env

            self.env = simpler_env.make(self.env_name, **env_kwargs)

        native_obs, reset_info = self._reset_native_env(task.seed, metadata.get("reset_options"))
        self.language_instruction = task.instruction or self._get_language_instruction()
        return self._observation(native_obs, {"reset_info": jsonable(reset_info)})

    def step(self, action: Action) -> StepResult:
        if self.env is None:
            raise RuntimeError("SimplerEnv environment has not been reset")
        native_action = action_vector(action, expected_length=7)
        result = self.env.step(native_action)
        native_obs, reward, terminated, truncated, info = self._parse_step_result(result)
        self.step_count += 1
        self.language_instruction = self._get_language_instruction() or self.language_instruction
        reward_value = scalar_float(reward)
        done = bool(terminated or truncated)
        success = extract_success(info, reward=reward_value)
        return StepResult(
            observation=self._observation(native_obs),
            reward=reward_value,
            done=done,
            success=success,
            info={
                "simulator_name": "simplerenv",
                "env_name": self.env_name,
                "step_count": self.step_count,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "native_info": jsonable(info),
            },
        )

    def get_state(self) -> dict:
        return {
            "simulator_name": "simplerenv",
            "task": self.task.to_dict() if self.task else None,
            "env_name": self.env_name,
            "backend": self.backend,
            "step_count": self.step_count,
            "language_instruction": self.language_instruction,
            "has_env": self.env is not None,
        }

    def close(self) -> dict:
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None
        return {"closed": True, "simulator_name": "simplerenv"}

    def _reset_native_env(self, seed: int, options: Any) -> tuple[Any, Any]:
        assert self.env is not None
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            try:
                result = self.env.reset(seed=seed)
            except TypeError:
                result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return result, {}

    def _parse_step_result(self, result: Any) -> tuple[Any, float, bool, bool, Any]:
        if not isinstance(result, tuple):
            raise TypeError(f"Expected env.step to return a tuple, got {type(result)!r}")
        if len(result) == 5:
            native_obs, reward, terminated, truncated, info = result
            return native_obs, reward, _boolish(terminated), _boolish(truncated), info
        if len(result) == 4:
            native_obs, reward, done, info = result
            return native_obs, reward, _boolish(done), False, info
        raise ValueError(f"Expected 4 or 5 return values from env.step, got {len(result)}")

    def _get_language_instruction(self) -> str:
        if self.env is None:
            return ""
        target = getattr(self.env, "unwrapped", self.env)
        for owner in (target, self.env):
            if hasattr(owner, "get_language_instruction"):
                instruction = owner.get_language_instruction()
                if isinstance(instruction, (list, tuple)):
                    return str(instruction[0]) if instruction else ""
                return str(instruction)
        return ""

    def _observation(self, native_obs: Any, metadata: dict | None = None) -> Observation:
        return native_observation(
            native_obs,
            simulator_name="simplerenv",
            language_instruction=self.language_instruction,
            metadata={
                "env_name": self.env_name,
                "backend": self.backend,
                "step_count": self.step_count,
                **(metadata or {}),
            },
            extra_images=self._extracted_images(native_obs),
        )

    def _extracted_images(self, native_obs: Any) -> dict:
        if self.env is None:
            return {}
        for name in ("get_image_from_maniskill2_obs_dict", "get_image_from_maniskill3_obs_dict"):
            try:
                module = import_module("simpler_env.utils.env.observation_utils")
                extractor = getattr(module, name)
            except (ImportError, AttributeError):
                continue
            try:
                image = extractor(self.env, native_obs)
            except Exception:
                continue
            image = _to_numpy(image)
            if hasattr(image, "dtype") and hasattr(image, "shape") and hasattr(image, "tobytes"):
                return {"primary": encode_array(image)}
        return {}


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return value


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except ValueError:
            pass
    if hasattr(value, "any"):
        any_value = value.any()
        if hasattr(any_value, "item"):
            return bool(any_value.item())
        return bool(any_value)
    return bool(value)
