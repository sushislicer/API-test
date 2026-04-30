from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from ...common.base import SimulatorAdapter
from ...common.metadata import scoped_metadata
from ...common.paths import api_root, resolve_path
from ...common.registry import register_sim
from ...common.schemas import Action, Observation, StepResult, TaskSpec
from ._native_utils import action_type, action_vector, jsonable, native_observation


@register_sim("robotwin")
class RoboTwinAdapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.env: Any | None = None
        self.task: TaskSpec | None = None
        self.repo_root: Path | None = None
        self.task_name = ""
        self.task_config = ""
        self.step_count = 0
        self.language_instruction = ""
        self.default_action_type = "qpos"
        self.strict_action_dim = True
        self.expected_action_dim: int | None = None
        self.args: dict[str, Any] = {}

    def reset(self, task: TaskSpec) -> Observation:
        self.close()
        self.task = task
        self.step_count = 0

        metadata = scoped_metadata(task.metadata, "robotwin", "simulators", "simulator")
        self.task_name = str(metadata.get("task_name") or task.task)
        self.task_config = str(metadata.get("task_config", "demo_clean"))
        self.default_action_type = str(metadata.get("action_type", "qpos"))
        self.strict_action_dim = _bool_value(metadata.get("strict_action_dim"), True)
        self.expected_action_dim = _optional_int(metadata.get("expected_action_dim") or metadata.get("action_dim"))
        self.repo_root = self._resolve_repo_root(metadata)
        if self.repo_root is not None and str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        self.args = self._build_args(task, metadata)
        module = import_module(f"envs.{self.task_name}")
        env_cls = getattr(module, self.task_name)
        self.env = env_cls()
        self.env.setup_demo(now_ep_num=int(metadata.get("episode_index", 0)), seed=task.seed, is_test=True, **self.args)

        self.language_instruction = self._resolve_instruction(task, metadata)
        if self.language_instruction and hasattr(self.env, "set_instruction"):
            self.env.set_instruction(instruction=self.language_instruction)

        native_obs = self.env.get_obs()
        return self._observation(native_obs)

    def step(self, action: Action) -> StepResult:
        if self.env is None:
            raise RuntimeError("RoboTwin environment has not been reset")
        native_action = action_vector(action)
        native_action_type = action_type(action, self.default_action_type)
        if native_action_type == "delta_ee_pose":
            native_action_type = self.default_action_type
        self._validate_action(native_action, native_action_type)

        self.env.take_action(native_action, action_type=native_action_type)
        self.step_count += 1
        native_obs = self.env.get_obs()
        success = bool(getattr(self.env, "eval_success", False))
        if not success and hasattr(self.env, "check_success"):
            success = bool(self.env.check_success())
        step_limit = getattr(self.env, "step_lim", None)
        take_action_count = int(getattr(self.env, "take_action_cnt", self.step_count))
        done = success or (step_limit is not None and take_action_count >= int(step_limit))
        reward = 1.0 if success else 0.0
        return StepResult(
            observation=self._observation(native_obs),
            reward=reward,
            done=done,
            success=success,
            info={
                "simulator_name": "robotwin",
                "task_name": self.task_name,
                "task_config": self.task_config,
                "step_count": self.step_count,
                "take_action_count": take_action_count,
                "step_limit": step_limit,
                "action_type": native_action_type,
                "action_dim": len(native_action),
                "expected_action_dim": self._expected_action_dim(native_action_type),
            },
        )

    def get_state(self) -> dict:
        return {
            "simulator_name": "robotwin",
            "task": self.task.to_dict() if self.task else None,
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "task_name": self.task_name,
            "task_config": self.task_config,
            "step_count": self.step_count,
            "language_instruction": self.language_instruction,
            "has_env": self.env is not None,
        }

    def close(self) -> dict:
        if self.env is not None:
            if hasattr(self.env, "close_env"):
                self.env.close_env()
            elif hasattr(self.env, "close"):
                self.env.close()
        self.env = None
        return {"closed": True, "simulator_name": "robotwin"}

    def _build_args(self, task: TaskSpec, metadata: dict) -> dict[str, Any]:
        repo_root = self.repo_root
        if repo_root is None:
            raise RuntimeError("RoboTwin requires task.metadata['repo_path'] or ROBOTWIN_ROOT")
        yaml = _import_yaml()
        config_path = repo_root / "task_config" / f"{self.task_config}.yml"
        with config_path.open("r", encoding="utf-8") as handle:
            args = yaml.safe_load(handle) or {}
        _deep_update(args, metadata.get("config_overrides", {}))

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config
        args["eval_mode"] = True
        args["save_data"] = bool(metadata.get("save_data", False))
        args["render_freq"] = int(metadata.get("render_freq", args.get("render_freq", 0)))
        args["save_path"] = str(metadata.get("save_path", repo_root / "data"))
        args["eval_video_save_dir"] = metadata.get("eval_video_save_dir")

        embodiment_config_path = repo_root / "task_config" / "_embodiment_config.yml"
        with embodiment_config_path.open("r", encoding="utf-8") as handle:
            embodiment_types = yaml.safe_load(handle) or {}
        camera_config_path = repo_root / "task_config" / "_camera_config.yml"
        with camera_config_path.open("r", encoding="utf-8") as handle:
            camera_config = yaml.safe_load(handle) or {}

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]

        embodiment_type = args.get("embodiment", [])
        if len(embodiment_type) == 1:
            args["left_robot_file"] = self._embodiment_file(repo_root, embodiment_types, embodiment_type[0])
            args["right_robot_file"] = self._embodiment_file(repo_root, embodiment_types, embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = self._embodiment_file(repo_root, embodiment_types, embodiment_type[0])
            args["right_robot_file"] = self._embodiment_file(repo_root, embodiment_types, embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("RoboTwin embodiment config must contain 1 or 3 items")

        args["left_embodiment_config"] = self._load_robot_config(args["left_robot_file"])
        args["right_embodiment_config"] = self._load_robot_config(args["right_robot_file"])
        return args

    def _resolve_repo_root(self, metadata: dict) -> Path | None:
        raw_path = metadata.get("repo_path") or os.environ.get("ROBOTWIN_ROOT")
        if raw_path:
            return resolve_path(raw_path)
        local_repo = api_root() / "simulators" / "RoboTwin"
        if local_repo.exists():
            return local_repo.resolve()
        return None

    def _embodiment_file(self, repo_root: Path, embodiment_types: dict, embodiment_name: str) -> str:
        raw_path = embodiment_types[embodiment_name]["file_path"]
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        return str(path.resolve())

    def _load_robot_config(self, robot_file: str) -> dict:
        yaml = _import_yaml()
        with (Path(robot_file) / "config.yml").open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _observation(self, native_obs: Any) -> Observation:
        return native_observation(
            native_obs,
            simulator_name="robotwin",
            language_instruction=self.language_instruction,
            metadata={
                "repo_root": str(self.repo_root) if self.repo_root else None,
                "task_name": self.task_name,
                "task_config": self.task_config,
                "step_count": self.step_count,
                "native_args": jsonable(
                    {
                        "camera": self.args.get("camera"),
                        "data_type": self.args.get("data_type"),
                        "embodiment": self.args.get("embodiment"),
                    }
                ),
            },
        )

    def _resolve_instruction(self, task: TaskSpec, metadata: dict[str, Any]) -> str:
        env_instruction = ""
        if self.env is not None and hasattr(self.env, "get_instruction"):
            env_instruction = self.env.get_instruction() or ""
        metadata_instruction = str(metadata.get("instruction") or metadata.get("prompt") or "")
        source = str(metadata.get("instruction_source", "env")).lower()
        if source == "metadata":
            return metadata_instruction or env_instruction or task.instruction
        if source == "task":
            return task.instruction or metadata_instruction or env_instruction
        return env_instruction or metadata_instruction or task.instruction

    def _validate_action(self, action: list[float], action_type: str) -> None:
        if not self.strict_action_dim:
            return
        expected = self._expected_action_dim(action_type)
        if expected is None:
            return
        if len(action) != expected:
            raise ValueError(
                f"RoboTwin expected {expected} action values for action_type='{action_type}', "
                f"got {len(action)}. Configure model metadata expected_action_dim/action_slice, "
                "or set simulator metadata strict_action_dim=false if this mismatch is intentional."
            )

    def _expected_action_dim(self, action_type: str) -> int | None:
        if self.expected_action_dim is not None:
            return self.expected_action_dim
        if self.env is None or not hasattr(self.env, "robot"):
            return None
        if action_type == "ee":
            return 16
        if action_type == "qpos":
            left_jointstate = self.env.robot.get_left_arm_jointState()
            right_jointstate = self.env.robot.get_right_arm_jointState()
            return len(left_jointstate) + len(right_jointstate)
        return None


def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("RoboTwin adapter requires PyYAML in the RoboTwin environment") from exc
    return yaml


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
