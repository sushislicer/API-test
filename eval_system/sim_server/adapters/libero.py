from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ...common.base import SimulatorAdapter
from ...common.metadata import scoped_metadata
from ...common.registry import register_sim
from ...common.schemas import Action, Observation, StepResult, TaskSpec
from ._native_utils import action_vector, extract_success, jsonable, native_observation, scalar_float


@register_sim("libero")
class LIBEROAdapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.env: Any | None = None
        self.task: TaskSpec | None = None
        self.task_suite: Any | None = None
        self.task_id = 0
        self.step_count = 0
        self.language_instruction = ""
        self.task_name = ""

    def reset(self, task: TaskSpec) -> Observation:
        metadata = scoped_metadata(task.metadata, "libero", "simulators", "simulator")
        _prepare_libero_import_path(metadata)

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.close()
        self.task = task
        self.step_count = 0

        suite_name = str(metadata.get("task_suite") or task.task)
        self.task_id = int(metadata.get("task_id", 0))
        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO task suite: {suite_name}")
        self.task_suite = benchmark_dict[suite_name]()
        libero_task = self.task_suite.get_task(self.task_id)
        self.task_name = getattr(libero_task, "name", f"{suite_name}:{self.task_id}")
        self.language_instruction = task.instruction or getattr(libero_task, "language", "")

        bddl_file_name = metadata.get("bddl_file_name")
        if not bddl_file_name:
            bddl_file_name = os.path.join(
                get_libero_path("bddl_files"),
                libero_task.problem_folder,
                libero_task.bddl_file,
            )

        env_kwargs = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": int(metadata.get("camera_height", metadata.get("camera_heights", 128))),
            "camera_widths": int(metadata.get("camera_width", metadata.get("camera_widths", 128))),
        }
        env_kwargs.update(metadata.get("env_kwargs", {}))
        self.env = OffScreenRenderEnv(**env_kwargs)
        self.env.seed(task.seed)
        native_obs = self.env.reset()

        init_states = self.task_suite.get_task_init_states(self.task_id)
        init_state_id = None
        if init_states is not None and len(init_states) > 0:
            init_state_id = int(metadata.get("init_state_id", task.seed % len(init_states)))
            set_state_result = self.env.set_init_state(init_states[init_state_id])
            if set_state_result is not None:
                native_obs = set_state_result
            elif hasattr(self.env, "get_observation"):
                native_obs = self.env.get_observation()

        return self._observation(native_obs, {"init_state_id": init_state_id, "suite": suite_name})

    def step(self, action: Action) -> StepResult:
        if self.env is None:
            raise RuntimeError("LIBERO environment has not been reset")
        native_action = action_vector(action, expected_length=7)
        native_obs, reward, done, info = self.env.step(native_action)
        self.step_count += 1
        reward_value = scalar_float(reward)
        success = extract_success(info, reward=reward_value)
        return StepResult(
            observation=self._observation(native_obs),
            reward=reward_value,
            done=bool(done),
            success=success,
            info={
                "simulator_name": "libero",
                "task_name": self.task_name,
                "task_id": self.task_id,
                "step_count": self.step_count,
                "native_info": jsonable(info),
            },
        )

    def get_state(self) -> dict:
        return {
            "simulator_name": "libero",
            "task": self.task.to_dict() if self.task else None,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "has_env": self.env is not None,
        }

    def close(self) -> dict:
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"closed": True, "simulator_name": "libero"}

    def _observation(self, native_obs: Any, metadata: dict | None = None) -> Observation:
        return native_observation(
            native_obs,
            simulator_name="libero",
            language_instruction=self.language_instruction,
            metadata={
                "task_name": self.task_name,
                "task_id": self.task_id,
                "step_count": self.step_count,
                **(metadata or {}),
            },
        )


def _prepare_libero_import_path(metadata: dict) -> None:
    raw_path = metadata.get("repo_path") or os.environ.get("LIBERO_ROOT")
    if raw_path is None:
        default_path = Path(__file__).resolve().parents[3] / "simulators" / "LIBERO"
        raw_path = str(default_path) if default_path.exists() else None
    if raw_path is None:
        return

    repo_root = Path(str(raw_path)).expanduser().resolve()
    for candidate in (repo_root, repo_root / "libero"):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
