from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...common.arrays import decode_array
from ...common.base import ModelAdapter
from ...common.metadata import apply_aliases, scoped_metadata, training_metadata
from ...common.paths import api_root, resolve_path
from ...common.registry import register_model
from ...common.schemas import Action, Observation, TaskSpec, TrainingSpec


@register_model("lingbot-va")
class LingBotVAAdapter(ModelAdapter):
    """Adapter for LingBot-VA's native WebSocket inference server.

    This adapter deliberately does not load LingBot-VA weights. Start the
    LingBot-VA server from the LingBot environment, then run this API model
    server in an environment that has `websockets`, `msgpack`, and `numpy`.
    """

    def __init__(self) -> None:
        self.task: TaskSpec | None = None
        self.repo_root = _default_lingbot_repo()
        self.host = "127.0.0.1"
        self.port = 29056
        self.api_key: str | None = None
        self.connect_timeout_s = 30.0
        self.env_type = "robotwin"
        self.prompt = ""
        self.save_visualization = False
        self.video_guidance_scale = 5.0
        self.action_guidance_scale = 1.0
        self.keyframes_per_chunk = 4
        self.flip_libero_images = True
        self.default_action_type = "ee"

        self.client: _LingBotWebSocketClient | None = None
        self.step_count = 0
        self.pending_actions: list[Action] = []
        self.pending_keyframe_flags: list[bool] = []
        self.last_action_was_keyframe = False
        self.keyframe_observations: list[dict[str, Any]] = []
        self.last_raw_action_chunk: Any | None = None
        self.first_chunk = True
        self.initial_eef_pose: list[float] | None = None
        self.training_jobs: dict[str, dict[str, Any]] = {}

    def reset(self, task: TaskSpec) -> dict:
        self.task = task
        self.step_count = 0
        self.pending_actions = []
        self.pending_keyframe_flags = []
        self.last_action_was_keyframe = False
        self.keyframe_observations = []
        self.last_raw_action_chunk = None
        self.first_chunk = True
        self.initial_eef_pose = None

        metadata = scoped_metadata(task.metadata, "lingbot-va", "models", "model")
        self.repo_root = resolve_path(
            metadata.get("repo_path")
            or os.environ.get("LINGBOT_VA_ROOT")
            or self.repo_root
        )
        self.host = str(metadata.get("host") or os.environ.get("LINGBOT_VA_HOST") or self.host)
        self.port = int(metadata.get("port") or os.environ.get("LINGBOT_VA_PORT") or self.port)
        self.api_key = metadata.get("api_key") or os.environ.get("LINGBOT_VA_API_KEY")
        self.connect_timeout_s = float(metadata.get("connect_timeout_s", self.connect_timeout_s))
        self.env_type = str(metadata.get("env_type") or _infer_env_type(task.benchmark))
        self.prompt = str(metadata.get("prompt") or task.instruction or task.task)
        self.save_visualization = bool(metadata.get("save_visualization", self.save_visualization))
        self.video_guidance_scale = float(metadata.get("video_guidance_scale", self.video_guidance_scale))
        self.action_guidance_scale = float(metadata.get("action_guidance_scale", self.action_guidance_scale))
        self.keyframes_per_chunk = int(metadata.get("keyframes_per_chunk", self.keyframes_per_chunk))
        self.flip_libero_images = bool(metadata.get("flip_libero_images", self.flip_libero_images))
        self.default_action_type = str(metadata.get("action_type", "delta_ee_pose" if self.env_type == "libero" else "ee"))

        self.client = _LingBotWebSocketClient(
            repo_root=self.repo_root,
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            connect_timeout_s=self.connect_timeout_s,
        )
        self.client.infer(
            {
                "reset": True,
                "prompt": self.prompt,
                "save_visualization": self.save_visualization,
            }
        )
        return {
            "model_name": "lingbot-va",
            "mode": "websocket-client",
            "host": self.host,
            "port": self.port,
            "repo_root": str(self.repo_root),
            "env_type": self.env_type,
            "prompt": self.prompt,
        }

    def act(self, observation: Observation) -> Action:
        self._remember_post_step_observation(observation)
        if not self.pending_actions:
            self._flush_kv_cache_if_ready()
            raw_chunk = self._infer_action_chunk(observation)
            self._queue_actions(raw_chunk, observation)

        if not self.pending_actions:
            raise RuntimeError("LingBot-VA returned an empty action chunk")

        action = self.pending_actions.pop(0)
        self.last_action_was_keyframe = self.pending_keyframe_flags.pop(0)
        self.step_count += 1
        action.metadata.update(
            {
                "model_name": "lingbot-va",
                "policy_step": self.step_count,
                "env_type": self.env_type,
                "chunk_queue_remaining": len(self.pending_actions),
            }
        )
        return action

    def predict_next(self, observation: Observation, action: Action) -> Observation:
        return Observation(
            rgb=observation.rgb,
            depth=observation.depth,
            images=observation.images,
            depth_images=observation.depth_images,
            proprio=observation.proprio,
            gripper=observation.gripper,
            language_instruction=observation.language_instruction,
            objects=observation.objects,
            state=observation.state,
            metadata={
                **observation.metadata,
                "predicted_by": "lingbot-va",
                "prediction_unavailable": True,
            },
        )

    def train(self, spec: TrainingSpec) -> dict:
        request = self._training_request(spec)
        if _bool_value(spec.dry_run, True):
            return {
                **self._public_training_request(request),
                "status": "dry_run",
                "message": "Training command prepared but not launched. Set dry_run=false to start it.",
            }

        process_env = os.environ.copy()
        process_env.update(request["env"])
        log_handle = None
        log_path = request.get("log_path")
        if log_path:
            log_file = Path(log_path).expanduser()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_file.open("ab")
            request["log_path"] = str(log_file)

        try:
            process = subprocess.Popen(
                request["command"],
                cwd=request["cwd"],
                env=process_env,
                stdout=log_handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            if log_handle is not None:
                log_handle.close()

        job_id = str(request["job_id"])
        self.training_jobs[job_id] = {
            "process": process,
            "request": self._public_training_request(request),
            "started_at": time.time(),
        }
        return {
            **self.training_status(job_id),
            "message": "LingBot-VA training launched.",
        }

    def training_status(self, job_id: str | None = None) -> dict:
        if job_id is None:
            jobs = [self._training_job_status(job_id, job) for job_id, job in self.training_jobs.items()]
            return {"status": "ok", "jobs": jobs}
        job = self.training_jobs.get(job_id)
        if job is None:
            return {"status": "not_found", "job_id": job_id}
        return self._training_job_status(job_id, job)

    def _training_request(self, spec: TrainingSpec) -> dict[str, Any]:
        simulator_name = str(spec.metadata.get("simulator") or spec.benchmark or "")
        metadata = training_metadata(spec.metadata, "lingbot-va", simulator_name)
        metadata = apply_aliases(
            metadata,
            {
                "steps": "num_steps",
                "num_train_steps": "num_steps",
                "max_steps": "num_steps",
                "lr": "learning_rate",
                "batchsize": "batch_size",
                "batch": "batch_size",
                "grad_accum_steps": "gradient_accumulation_steps",
                "gradient_accumulation": "gradient_accumulation_steps",
                "save_steps": "save_interval",
                "workers": "load_worker",
                "num_workers": "load_worker",
            },
        )
        repo_root = resolve_path(
            metadata.get("repo_path")
            or os.environ.get("LINGBOT_VA_ROOT")
            or self.repo_root
        )
        train_module = repo_root / "wan_va" / "train.py"
        if not train_module.exists():
            raise RuntimeError(f"LingBot-VA training module was not found at {train_module}")

        config_name = str(
            spec.config_name
            or metadata.get("config_name")
            or _training_config_for_benchmark(spec.benchmark)
        )
        num_gpus = int(metadata.get("num_gpus") or metadata.get("ngpu") or os.environ.get("LINGBOT_VA_NGPU") or 8)
        master_port = int(metadata.get("master_port") or os.environ.get("LINGBOT_VA_MASTER_PORT") or 29501)
        log_rank = int(metadata.get("log_rank") or os.environ.get("LINGBOT_VA_LOG_RANK") or 0)
        torchft_lighthouse = str(
            metadata.get("torchft_lighthouse")
            or os.environ.get("TORCHFT_LIGHTHOUSE")
            or "http://localhost:29510"
        )
        python_executable = str(metadata.get("python") or sys.executable)
        save_root = str(metadata.get("save_root") or spec.output_dir or "")

        overrides = dict(metadata.get("config_overrides", {}))
        if spec.dataset_path:
            dataset_path = str(Path(spec.dataset_path).expanduser())
            overrides.setdefault("dataset_path", dataset_path)
            overrides.setdefault("empty_emb_path", str(Path(dataset_path) / "empty_emb.pt"))
        if spec.checkpoint_path:
            overrides.setdefault("resume_from", str(Path(spec.checkpoint_path).expanduser()))
        if metadata.get("pretrained_model_path"):
            overrides.setdefault("wan22_pretrained_model_name_or_path", str(metadata["pretrained_model_path"]))
        for key in (
            "num_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "save_interval",
            "load_worker",
            "cfg_prob",
        ):
            if key in metadata:
                overrides.setdefault(key, metadata[key])

        notes = [
            "Before training, set the checkpoint transformer/config.json attn_mode to flex.",
            "Before evaluation/inference, set attn_mode to torch or flashattn; flex is training-only.",
        ]
        if "enable_wandb" in metadata:
            overrides.setdefault("enable_wandb", _bool_value(metadata["enable_wandb"], False))
        elif not _wandb_configured(metadata):
            overrides.setdefault("enable_wandb", False)
            notes.append("WandB environment variables were not configured, so enable_wandb=false was added.")

        command = [
            python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--local-ranks-filter={log_rank}",
            "--master_port",
            str(master_port),
            "--tee",
            "3",
            "-m",
            "wan_va.train",
            "--config-name",
            config_name,
        ]
        if save_root:
            command.extend(["--save-root", save_root])
        for key, value in overrides.items():
            command.extend(["--set", f"{key}={_format_override_value(value)}"])
        command.extend(_string_list(metadata.get("extra_args")))

        env = {
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHFT_LIGHTHOUSE": torchft_lighthouse,
        }
        for key in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_TEAM_NAME", "WANDB_PROJECT"):
            value = metadata.get(key) or os.environ.get(key)
            if value is not None:
                env[key] = str(value)
        env.update(_string_dict(metadata.get("env", {})))

        job_id = str(metadata.get("job_id") or f"lingbot-va-{time.time_ns()}")
        log_path = metadata.get("log_path")
        if log_path is None and save_root:
            log_path = str(Path(save_root).expanduser() / "lingbot_va_train.log")

        return {
            "model_name": "lingbot-va",
            "mode": "training-launcher",
            "job_id": job_id,
            "command": command,
            "command_text": shlex.join(command),
            "cwd": str(repo_root),
            "env": env,
            "env_preview": _mask_env(env),
            "config_name": config_name,
            "save_root": save_root,
            "log_path": str(log_path) if log_path else None,
            "notes": notes,
        }

    def _public_training_request(self, request: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in request.items() if key not in {"env", "env_preview"}}
        public["env"] = request["env_preview"]
        return public

    def _training_job_status(self, job_id: str, job: dict[str, Any]) -> dict:
        process = job["process"]
        returncode = process.poll()
        if returncode is None:
            status = "running"
        elif returncode == 0:
            status = "succeeded"
        else:
            status = "failed"
        return {
            **job["request"],
            "status": status,
            "job_id": job_id,
            "pid": process.pid,
            "returncode": returncode,
            "started_at": job["started_at"],
        }

    def _infer_action_chunk(self, observation: Observation) -> Any:
        if self.client is None:
            raise RuntimeError("LingBot-VA adapter has not been reset")
        payload = {
            "obs": self._format_observation(observation),
            "prompt": self.prompt,
            "save_visualization": self.save_visualization,
            "video_guidance_scale": self.video_guidance_scale,
            "action_guidance_scale": self.action_guidance_scale,
        }
        response = self.client.infer(payload)
        if "action" not in response:
            raise RuntimeError(f"LingBot-VA response did not include an action: {response.keys()}")
        self.last_raw_action_chunk = response["action"]
        return response["action"]

    def _flush_kv_cache_if_ready(self) -> None:
        if self.client is None or self.last_raw_action_chunk is None or not self.keyframe_observations:
            return
        self.client.infer(
            {
                "obs": self.keyframe_observations,
                "compute_kv_cache": True,
                "imagine": False,
                "save_visualization": self.save_visualization,
                "state": self.last_raw_action_chunk,
            }
        )
        self.keyframe_observations = []

    def _remember_post_step_observation(self, observation: Observation) -> None:
        if not self.last_action_was_keyframe:
            return
        self.keyframe_observations.append(self._format_observation(observation))
        self.last_action_was_keyframe = False

    def _format_observation(self, observation: Observation) -> dict[str, Any]:
        np = _import_numpy()
        if self.env_type == "libero":
            return {
                "observation.images.agentview_rgb": self._image(
                    observation,
                    [
                        "observation.images.agentview_rgb",
                        "agentview_image",
                        "agentview_rgb",
                    ],
                    flip=self.flip_libero_images,
                ),
                "observation.images.eye_in_hand_rgb": self._image(
                    observation,
                    [
                        "observation.images.eye_in_hand_rgb",
                        "robot0_eye_in_hand_image",
                        "eye_in_hand_rgb",
                    ],
                    flip=self.flip_libero_images,
                ),
                "task": self.prompt,
            }
        return {
            "observation.images.cam_high": self._image(
                observation,
                [
                    "observation.images.cam_high",
                    "observation.head_camera.rgb",
                    "head_camera.rgb",
                ],
            ),
            "observation.images.cam_left_wrist": self._image(
                observation,
                [
                    "observation.images.cam_left_wrist",
                    "observation.left_camera.rgb",
                    "left_camera.rgb",
                ],
            ),
            "observation.images.cam_right_wrist": self._image(
                observation,
                [
                    "observation.images.cam_right_wrist",
                    "observation.right_camera.rgb",
                    "right_camera.rgb",
                ],
            ),
            "observation.state": np.asarray(
                self._state_vector(
                    observation,
                    ["joint_action.vector", "observation.state", "qpos"],
                    fallback=observation.proprio,
                ),
                dtype=np.float32,
            ),
            "task": self.prompt,
        }

    def _queue_actions(self, raw_chunk: Any, observation: Observation) -> None:
        np = _import_numpy()
        chunk = np.asarray(raw_chunk)
        if chunk.ndim == 1:
            raw_steps = [(chunk, True)]
        elif chunk.ndim == 2:
            raw_steps = [(chunk[:, index], True) for index in range(chunk.shape[1])]
        elif chunk.ndim == 3:
            action_per_frame = max(chunk.shape[2] // max(self.keyframes_per_chunk, 1), 1)
            start_frame = 1 if self.first_chunk and chunk.shape[1] > 1 else 0
            raw_steps = []
            for frame_idx in range(start_frame, chunk.shape[1]):
                for action_idx in range(chunk.shape[2]):
                    raw_steps.append((chunk[:, frame_idx, action_idx], (action_idx + 1) % action_per_frame == 0))
            self.first_chunk = False
        else:
            raise RuntimeError(f"Unsupported LingBot-VA action shape: {chunk.shape}")

        self.pending_actions = [self._to_action(np.asarray(step).reshape(-1), observation) for step, _ in raw_steps]
        self.pending_keyframe_flags = [is_keyframe for _, is_keyframe in raw_steps]

    def _to_action(self, raw_step: Any, observation: Observation) -> Action:
        values = [float(value) for value in raw_step.tolist()]
        if self.env_type == "robotwin":
            vector = self._robotwin_action_vector(values, observation)
            action_type = "ee"
        else:
            vector = values
            action_type = self.default_action_type

        translation = vector[:3] if len(vector) >= 3 else [0.0, 0.0, 0.0]
        if action_type == "ee" and len(vector) >= 8:
            rotation = vector[3:7]
            gripper = vector[7]
        else:
            rotation = vector[3:6] if len(vector) >= 6 else [0.0, 0.0, 0.0]
            gripper = vector[6] if len(vector) >= 7 else 0.0
        return Action(
            type=action_type,
            action_type=action_type,
            vector=vector,
            translation=translation,
            rotation=rotation,
            gripper=gripper,
            metadata={"raw_action_dim": len(values)},
        )

    def _robotwin_action_vector(self, values: list[float], observation: Observation) -> list[float]:
        if len(values) == 14:
            return [
                *values[:3],
                *_euler_to_quat(values[3], values[4], values[5]),
                values[6],
                *values[7:10],
                *_euler_to_quat(values[10], values[11], values[12]),
                values[13],
            ]
        if len(values) == 16:
            initial_pose = self._initial_eef_pose(observation)
            return _add_initial_pose(values, initial_pose)
        return values

    def _initial_eef_pose(self, observation: Observation) -> list[float]:
        if self.initial_eef_pose is not None:
            return self.initial_eef_pose
        left_pose = self._state_vector(observation, ["endpose.left_endpose"], required=False)
        right_pose = self._state_vector(observation, ["endpose.right_endpose"], required=False)
        left_gripper = self._state_scalar(observation, ["endpose.left_gripper"])
        right_gripper = self._state_scalar(observation, ["endpose.right_gripper"])
        if len(left_pose) < 7 or len(right_pose) < 7 or left_gripper is None or right_gripper is None:
            raise RuntimeError("LingBot-VA returned 16D RoboTwin actions, but the observation lacks initial end-effector pose fields")
        self.initial_eef_pose = [*left_pose[:7], left_gripper, *right_pose[:7], right_gripper]
        return self.initial_eef_pose

    def _image(self, observation: Observation, aliases: list[str], flip: bool = False) -> Any:
        np = _import_numpy()
        for key in aliases:
            if key not in observation.images:
                continue
            value = observation.images[key]
            image = decode_array(value) if isinstance(value, dict) and value.get("encoding") == "raw_base64" else np.asarray(value)
            if flip:
                image = image[::-1]
            return np.ascontiguousarray(image)
        available = ", ".join(sorted(observation.images.keys()))
        raise KeyError(f"Missing LingBot-VA image. Tried {aliases}; available images: {available}")

    def _state_vector(
        self,
        observation: Observation,
        aliases: list[str],
        fallback: list[float] | None = None,
        required: bool = True,
    ) -> list[float]:
        for key in aliases:
            value = observation.state.get(key)
            vector = _numeric_vector(value)
            if vector:
                return vector
        if fallback:
            return [float(value) for value in fallback]
        if required:
            raise KeyError(f"Missing LingBot-VA state vector. Tried {aliases}")
        return []

    def _state_scalar(self, observation: Observation, aliases: list[str]) -> float | None:
        vector = self._state_vector(observation, aliases, required=False)
        return vector[0] if vector else None


class _LingBotWebSocketClient:
    def __init__(
        self,
        *,
        repo_root: Path,
        host: str,
        port: int,
        api_key: str | None,
        connect_timeout_s: float,
    ) -> None:
        self.repo_root = repo_root
        self.host = host
        self.port = port
        self.api_key = api_key
        self.connect_timeout_s = connect_timeout_s
        self.ws: Any | None = None
        self.packer: Any | None = None
        self.unpackb: Any | None = None
        self._connect()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None or self.packer is None or self.unpackb is None:
            raise RuntimeError("LingBot-VA WebSocket client is not connected")
        self.ws.send(self.packer.pack(payload))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot-VA server returned an error string: {response}")
        return self.unpackb(response)

    def _connect(self) -> None:
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        try:
            import websockets.sync.client
            from evaluation.robotwin.msgpack_numpy import Packer, unpackb
        except ImportError as exc:
            raise RuntimeError(
                "LingBot-VA adapter requires the local lingbot-va repo plus its websocket/msgpack dependencies"
            ) from exc

        uri = f"ws://{self.host}:{self.port}"
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        deadline = time.monotonic() + self.connect_timeout_s
        last_error: Exception | None = None
        while True:
            try:
                self.ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                self.packer = Packer()
                self.unpackb = unpackb
                self.unpackb(self.ws.recv())
                return
            except Exception as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(1)
        raise RuntimeError(f"Could not connect to LingBot-VA server at {uri}") from last_error


def _default_lingbot_repo() -> Path:
    return api_root() / "models" / "lingbot-va"


def _infer_env_type(benchmark: str) -> str:
    value = benchmark.lower()
    if "libero" in value:
        return "libero"
    return "robotwin"


def _training_config_for_benchmark(benchmark: str) -> str:
    value = benchmark.lower()
    if "libero" in value:
        return "libero_train"
    return "robotwin_train"


def _import_numpy() -> Any:
    import numpy as np

    return np


def _numeric_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, dict) and value.get("encoding") == "raw_base64":
        array = decode_array(value)
        return [float(item) for item in array.reshape(-1).tolist()]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_numeric_vector(item))
        return flattened
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [float(item) for item in value.reshape(-1).tolist()]
    return []


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return str(value)


def _wandb_configured(metadata: dict[str, Any]) -> bool:
    env = metadata.get("env", {})
    if not isinstance(env, dict):
        env = {}
    return all(env.get(key) or metadata.get(key) or os.environ.get(key) for key in ("WANDB_API_KEY", "WANDB_BASE_URL"))


def _mask_env(env: dict[str, str]) -> dict[str, str]:
    masked = {}
    for key, value in env.items():
        if "KEY" in key or "TOKEN" in key or "SECRET" in key:
            masked[key] = "***"
        else:
            masked[key] = value
    return masked


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return _normalize_quat(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def _quat_multiply(lhs: list[float], rhs: list[float]) -> list[float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return _normalize_quat(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def _normalize_quat(quat: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [value / norm for value in quat]


def _add_eef_pose(delta_pose: list[float], initial_pose: list[float]) -> list[float]:
    translation = [delta_pose[index] + initial_pose[index] for index in range(3)]
    rotation = _quat_multiply(initial_pose[3:7], delta_pose[3:7])
    return [*translation, *rotation, delta_pose[7]]


def _add_initial_pose(action: list[float], initial_pose: list[float]) -> list[float]:
    left_pose = _add_eef_pose(action[:8], initial_pose[:8])
    right_pose = _add_eef_pose(action[8:], initial_pose[8:])
    return [*left_pose, *right_pose]
