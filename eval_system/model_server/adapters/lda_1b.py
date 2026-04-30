from __future__ import annotations

import json
import math
import os
import shlex
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
from ._training import TrainingJobManager, bool_value as _training_bool_value, mask_env, string_dict, string_list


@register_model("lda-1b")
class LDA1BAdapter(ModelAdapter):
    """Adapter for LDA-1B's native WebSocket policy server.

    The adapter is intentionally a thin bridge. It does not load checkpoints or
    import simulator code; the LDA repository owns model startup, and this API
    process only formats canonical observations into the LDA inference payload.
    """

    def __init__(self) -> None:
        self.task: TaskSpec | None = None
        self.repo_root = _default_lda_repo()
        self.host = "127.0.0.1"
        self.port = 10093
        self.api_key: str | None = None
        self.connect_timeout_s = 30.0
        self.prompt = ""
        self.payload_mode = "examples"
        self.default_action_type = "delta_ee_pose"
        self.use_ddim = True
        self.num_ddim_steps = 10
        self.do_sample = False
        self.wrap_state_history = True
        self.require_state = False
        self.state_transform = "none"
        self.image_keys: list[str] = []
        self.state_keys: list[str] = []
        self.action_output_key = ""
        self.action_layout = "time_first"
        self.action_dict_order: list[str] = []
        self.extra_example_fields: dict[str, Any] = {}
        self.extra_payload_fields: dict[str, Any] = {}
        self.embodiment_id: Any | None = None
        self.include_history_action = False
        self.history_action: Any | None = None
        self.action_min: Any | None = None
        self.action_max: Any | None = None
        self.action_mask: Any | None = None
        self.action_norm_stats_source: str | None = None
        self.clip_normalized_actions = True
        self.allow_normalized_actions = False
        self.expected_action_dim: int | None = None
        self.action_slice: tuple[int | None, int | None] | None = None
        self.action_indices: list[int] = []
        self.require_prompt = True
        self.expected_state_dim: int | None = None
        self.expected_image_count: int | None = None
        self.last_payload_metadata: dict[str, Any] = {}

        self.client: _LDAWebSocketClient | None = None
        self.step_count = 0
        self.pending_actions: list[Action] = []
        self.server_metadata: dict[str, Any] = {}
        self.training_jobs = TrainingJobManager("lda-1b")

    def reset(self, task: TaskSpec) -> dict:
        self.task = task
        self.step_count = 0
        self.pending_actions = []

        metadata = scoped_metadata(task.metadata, "lda-1b", "models", "model")
        metadata = _with_profile_defaults(metadata, task.benchmark)
        self.repo_root = resolve_path(
            metadata.get("repo_path")
            or os.environ.get("LDA_1B_ROOT")
            or _default_lda_repo()
        )
        self.host = str(metadata.get("host") or os.environ.get("LDA_1B_HOST") or "127.0.0.1")
        self.port = int(metadata.get("port") or os.environ.get("LDA_1B_PORT") or 10093)
        self.api_key = metadata.get("api_key") or os.environ.get("LDA_1B_API_KEY")
        self.connect_timeout_s = float(metadata.get("connect_timeout_s", 30.0))
        self.prompt = str(metadata.get("prompt") or task.instruction or task.task)
        self.payload_mode = str(metadata.get("payload_mode", "examples"))
        self.default_action_type = str(
            metadata.get("action_type")
            or os.environ.get("LDA_1B_ACTION_TYPE")
            or "delta_ee_pose"
        )
        self.use_ddim = _bool_value(metadata.get("use_ddim"), True)
        self.num_ddim_steps = int(metadata.get("num_ddim_steps", 10))
        self.do_sample = _bool_value(metadata.get("do_sample"), False)
        self.wrap_state_history = _bool_value(metadata.get("wrap_state_history"), True)
        self.require_state = _bool_value(metadata.get("require_state"), False)
        self.state_transform = str(metadata.get("state_transform", "none"))
        self.image_keys = _string_list(metadata.get("image_keys") or metadata.get("image_key"))
        self.state_keys = _string_list(metadata.get("state_keys") or metadata.get("state_key"))
        self.action_output_key = str(metadata.get("action_output_key", ""))
        self.action_layout = str(metadata.get("action_layout", "time_first"))
        self.action_dict_order = _string_list(metadata.get("action_dict_order"))
        self.extra_example_fields = dict(metadata.get("extra_example_fields", {}))
        self.extra_payload_fields = dict(metadata.get("extra_payload_fields", {}))
        self.embodiment_id = metadata.get("embodiment_id", os.environ.get("LDA_1B_EMBODIMENT_ID"))
        self.include_history_action = _bool_value(metadata.get("include_history_action"), False)
        self.history_action = metadata.get("history_action")
        self.clip_normalized_actions = _bool_value(
            metadata.get("clip_normalized_actions"),
            True,
        )
        self.allow_normalized_actions = _bool_value(metadata.get("allow_normalized_actions"), False)
        self.expected_action_dim = _optional_int(metadata.get("expected_action_dim") or metadata.get("action_dim"))
        self.action_slice = _optional_slice(metadata.get("action_slice") or metadata.get("action_range"))
        self.action_indices = _int_list(metadata.get("action_indices"))
        self.require_prompt = _bool_value(metadata.get("require_prompt"), True)
        self.expected_state_dim = _optional_int(metadata.get("expected_state_dim") or metadata.get("state_dim"))
        self.expected_image_count = _optional_int(metadata.get("expected_image_count") or metadata.get("image_count"))

        np = _import_numpy()
        self.action_min = _optional_array(np, metadata.get("action_min") or metadata.get("action_low"))
        self.action_max = _optional_array(np, metadata.get("action_max") or metadata.get("action_high"))
        self.action_mask = _optional_array(np, metadata.get("action_mask"))
        self.action_norm_stats_source = None
        if self.action_min is None or self.action_max is None:
            action_stats = _load_action_norm_stats(metadata)
            if action_stats is not None:
                self.action_min = _optional_array(np, action_stats.get("min"))
                self.action_max = _optional_array(np, action_stats.get("max"))
                self.action_mask = _optional_array(np, action_stats.get("mask"))
                self.action_norm_stats_source = action_stats.get("_source")

        self.client = _LDAWebSocketClient(
            repo_root=self.repo_root,
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            connect_timeout_s=self.connect_timeout_s,
        )
        self.server_metadata = self.client.server_metadata
        return {
            "model_name": "lda-1b",
            "mode": "websocket-client",
            "host": self.host,
            "port": self.port,
            "repo_root": str(self.repo_root),
            "payload_mode": self.payload_mode,
            "prompt": self.prompt,
            "image_keys": self.image_keys,
            "state_keys": self.state_keys,
            "action_type": self.default_action_type,
            "expected_action_dim": self.expected_action_dim,
            "expected_state_dim": self.expected_state_dim,
            "expected_image_count": self.expected_image_count,
            "action_norm_stats_source": self.action_norm_stats_source,
            "server_metadata": self.server_metadata,
        }

    def act(self, observation: Observation) -> Action:
        if not self.pending_actions:
            response = self._infer_action_chunk(observation)
            actions, source_key = self._extract_action_array(response)
            self._queue_actions(actions, source_key)

        if not self.pending_actions:
            raise RuntimeError("LDA-1B returned an empty action chunk")

        action = self.pending_actions.pop(0)
        self.step_count += 1
        action.metadata.update(
            {
                "model_name": "lda-1b",
                "policy_step": self.step_count,
                "chunk_queue_remaining": len(self.pending_actions),
                "payload": self.last_payload_metadata,
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
                "predicted_by": "lda-1b",
                "prediction_unavailable": True,
            },
        )

    def train(self, spec: TrainingSpec) -> dict:
        request = self._training_request(spec)
        return self.training_jobs.handle_train(request, _training_bool_value(spec.dry_run, True))

    def training_status(self, job_id: str | None = None) -> dict:
        return self.training_jobs.status(job_id)

    def _training_request(self, spec: TrainingSpec) -> dict[str, Any]:
        simulator_name = str(spec.metadata.get("simulator") or spec.benchmark or "")
        metadata = training_metadata(spec.metadata, "lda-1b", simulator_name)
        metadata = apply_aliases(
            metadata,
            {
                "steps": "max_train_steps",
                "num_steps": "max_train_steps",
                "num_train_steps": "max_train_steps",
                "lr": "learning_rate",
                "batchsize": "per_device_batch_size",
                "batch_size": "per_device_batch_size",
                "batch": "per_device_batch_size",
                "save_steps": "save_interval",
                "eval_steps": "eval_interval",
                "logging_steps": "logging_frequency",
            },
        )
        repo_root = resolve_path(
            metadata.get("repo_path")
            or os.environ.get("LDA_1B_ROOT")
            or self.repo_root
        )
        train_script = repo_root / "lda" / "training" / "train_LDA.py"
        if not train_script.exists():
            raise RuntimeError(f"LDA-1B training module was not found at {train_script}")

        run_id = str(metadata.get("run_id") or spec.config_name or f"api-{time.time_ns()}")
        run_root_dir = str(metadata.get("run_root_dir") or metadata.get("save_root") or spec.output_dir or "train_out")
        data_root_dir = str(spec.dataset_path or metadata.get("data_root_dir") or "playground/demo_data")
        pretrained_checkpoint = str(spec.checkpoint_path or metadata.get("pretrained_checkpoint") or "null")
        num_processes = int(
            metadata.get("num_processes")
            or metadata.get("num_gpus")
            or metadata.get("ngpu")
            or os.environ.get("LDA_1B_NUM_PROCESSES")
            or 8
        )

        command = [
            str(metadata.get("accelerate") or os.environ.get("LDA_1B_ACCELERATE") or "accelerate"),
            "launch",
            "--config_file",
            str(metadata.get("accelerate_config") or metadata.get("deepspeed_config") or "lda/config/deepseeds/deepspeed_zero2.yaml"),
            "--num_processes",
            str(num_processes),
            "lda/training/train_LDA.py",
            "--config_yaml",
            str(metadata.get("config_yaml") or "lda/config/training/LDA_pretrain.yaml"),
        ]
        default_args = {
            "framework.name": metadata.get("framework_name", "QwenMMDiT"),
            "framework.qwenvl.base_vlm": metadata.get("base_vlm") or os.environ.get("LDA_1B_BASE_VLM") or "/path/to/pretrained/VLM",
            "framework.action_model.vision_encoder_path": metadata.get("vision_encoder_path")
            or os.environ.get("LDA_1B_VISION_ENCODER_PATH")
            or "/path/to/pretrained/vision/encoder",
            "framework.action_model.action_model_type": metadata.get("dit_type", "DiT-L"),
            "framework.action_model.max_num_embodiments": metadata.get("max_num_embodiments", 1),
            "framework.action_model.state_dim": metadata.get("state_dim", None),
            "framework.action_model.action_dim": metadata.get("action_dim", 138),
            "framework.action_model.obs_horizon": metadata.get("obs_horizon", 1),
            "framework.action_model.future_obs_index": metadata.get("future_obs_index", 5),
            "framework.action_model.only_policy": metadata.get("only_policy", False),
            "framework.action_model.policy_and_video_gen": metadata.get("policy_and_video_gen", False),
            "framework.action_model.only_wo_video_gen": metadata.get("only_wo_video_gen", False),
            "framework.action_model.diffusion_model_cfg.positional_embeddings": metadata.get("positional_embeddings", None),
            "datasets.vla_data.use_delta_action": metadata.get("use_delta_action", False),
            "datasets.vla_data.data_root_dir": data_root_dir,
            "datasets.vla_data.training_task_weights": metadata.get("training_task_weights", [1, 1, 1, 1]),
            "datasets.vla_data.data_mix": metadata.get("data_mix", "demo_data"),
            "datasets.vla_data.per_device_batch_size": metadata.get("per_device_batch_size", metadata.get("batch_size", 64)),
            "datasets.vla_data.return_vlm_inputs": metadata.get("return_vlm_inputs", False),
            "trainer.freeze_modules": metadata.get("freeze_modules", "qwen_vl_interface,action_model.vision_encoder"),
            "trainer.max_train_steps": metadata.get("max_train_steps", metadata.get("num_steps", 200000)),
            "trainer.save_interval": metadata.get("save_interval", 10000),
            "trainer.logging_frequency": metadata.get("logging_frequency", 10),
            "trainer.eval_interval": metadata.get("eval_interval", 100),
            "trainer.repeated_diffusion_steps": metadata.get("repeated_diffusion_steps", 1),
            "trainer.learning_rate.base": metadata.get("learning_rate", 4e-5),
            "trainer.pretrained_checkpoint": pretrained_checkpoint,
        }
        _append_dot_args(command, default_args)
        _append_dot_args(command, dict(metadata.get("config_overrides", {})))
        command.extend(
            [
                "--run_root_dir",
                run_root_dir,
                "--run_id",
                run_id,
                "--wandb_project",
                str(metadata.get("wandb_project", "lda")),
                "--wandb_entity",
                str(metadata.get("wandb_entity", "your/wandb/entity")),
                "--is_debug",
                _format_cli_value(metadata.get("is_debug", False)),
            ]
        )
        command.extend(string_list(metadata.get("extra_args")))

        env = {
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_TIMEOUT": "1000",
            "WANDB_MODE": str(metadata.get("wandb_mode", "disabled")),
        }
        for key in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_TEAM_NAME", "WANDB_PROJECT"):
            value = metadata.get(key) or os.environ.get(key)
            if value is not None:
                env[key] = str(value)
        env.update(string_dict(metadata.get("env", {})))

        log_path = metadata.get("log_path")
        if log_path is None and run_root_dir:
            log_path = str(Path(run_root_dir).expanduser() / run_id / "lda_1b_train.log")

        return {
            "model_name": "lda-1b",
            "mode": "training-launcher",
            "job_id": str(metadata.get("job_id") or f"lda-1b-{time.time_ns()}"),
            "command": command,
            "command_text": shlex.join(command),
            "cwd": str(repo_root),
            "env": env,
            "env_preview": mask_env(env),
            "run_root_dir": run_root_dir,
            "run_id": run_id,
            "data_root_dir": data_root_dir,
            "log_path": str(log_path) if log_path else None,
            "notes": [
                "LDA-1B training uses the repository's accelerate-based train_LDA.py entrypoint.",
                "Set base_vlm, vision_encoder_path, data_root_dir/data_mix, run_root_dir, and run_id before launching real training.",
            ],
        }

    def _infer_action_chunk(self, observation: Observation) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LDA-1B adapter has not been reset")
        return self.client.infer(self._payload(observation))

    def _payload(self, observation: Observation) -> dict[str, Any]:
        images = self._images(observation)
        prompt = observation.language_instruction or self.prompt
        if self.require_prompt and not prompt.strip():
            raise RuntimeError("LDA-1B requires a non-empty language prompt; configure task instruction or simulator instruction")
        state = self._state(observation)
        self._validate_conditions(images, state)
        self.last_payload_metadata = {
            "prompt": prompt,
            "image_keys": self.image_keys or _default_image_keys(observation.images),
            "image_count": len(images),
            "state_shape": _shape_of(state),
            "state_keys": self.state_keys,
        }

        if self.payload_mode == "batch_images":
            payload: dict[str, Any] = {
                "batch_images": [images],
                "instructions": [prompt],
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
                "do_sample": self.do_sample,
            }
            if state is not None:
                payload["state"] = [state]
        elif self.payload_mode == "example":
            payload = self._example(images, prompt, state)
        else:
            payload = {
                "examples": [self._example(images, prompt, state)],
                "do_sample": self.do_sample,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }

        payload.update(self.extra_payload_fields)
        return payload

    def _validate_conditions(self, images: list[Any], state: Any | None) -> None:
        if self.expected_image_count is not None and len(images) != self.expected_image_count:
            raise RuntimeError(
                f"LDA-1B expected {self.expected_image_count} image views, got {len(images)}. "
                "Configure image_keys/expected_image_count to match the checkpoint."
            )
        if self.expected_state_dim is None:
            return
        if state is None:
            raise RuntimeError(f"LDA-1B expected state_dim={self.expected_state_dim}, but no state was provided")
        shape = _shape_of(state)
        if not shape or shape[-1] != self.expected_state_dim:
            raise RuntimeError(
                f"LDA-1B expected state_dim={self.expected_state_dim}, got state_shape={shape}. "
                "Configure state_keys/state_transform/expected_state_dim to match the checkpoint."
            )

    def _example(self, images: list[Any], prompt: str, state: Any | None) -> dict[str, Any]:
        example: dict[str, Any] = {
            "image": images,
            "lang": prompt,
        }
        if state is not None:
            example["state"] = state
        if self.embodiment_id is not None:
            example["embodiment_id"] = int(self.embodiment_id) if _looks_int(self.embodiment_id) else self.embodiment_id
        if self.include_history_action or self.history_action is not None:
            example["history_action"] = self.history_action
        example.update(self.extra_example_fields)
        return example

    def _images(self, observation: Observation) -> list[Any]:
        keys = self.image_keys or _default_image_keys(observation.images)
        images = [self._image(observation, key) for key in keys]
        if not images:
            available = ", ".join(sorted(observation.images.keys()))
            raise KeyError(f"LDA-1B requires at least one image; available images: {available}")
        return images

    def _image(self, observation: Observation, key: str) -> Any:
        if key not in observation.images:
            available = ", ".join(sorted(observation.images.keys()))
            raise KeyError(f"Missing LDA-1B image key '{key}'. Available images: {available}")
        return _as_array(observation.images[key])

    def _state(self, observation: Observation) -> Any | None:
        np = _import_numpy()
        if len(self.state_keys) == 1 and self.state_transform == "none":
            key = self.state_keys[0]
            state = _state_array(observation.state.get(key))
            if state is None:
                raise KeyError(f"Missing LDA-1B state key '{key}'")
            state = np.asarray(state, dtype=np.float32)
            if self.wrap_state_history and state.ndim == 1:
                state = state[None, :]
            return state

        vectors: list[list[float]] = []
        if self.state_keys:
            for key in self.state_keys:
                vector = _numeric_vector(observation.state.get(key))
                if not vector:
                    raise KeyError(f"Missing LDA-1B state key '{key}'")
                vectors.append(vector)
        elif observation.proprio:
            vectors.append([float(value) for value in observation.proprio])
        else:
            vector = _first_state_vector(observation)
            if vector:
                vectors.append(vector)

        if not vectors:
            if self.require_state:
                raise KeyError("LDA-1B requires state, but no state vector was found in the observation")
            return None

        state_parts: list[float] = []
        for vector in vectors:
            if self.state_transform == "sin_cos":
                state_parts.extend(math.sin(value) for value in vector)
                state_parts.extend(math.cos(value) for value in vector)
            else:
                state_parts.extend(vector)

        state = np.asarray(state_parts, dtype=np.float32)
        if self.wrap_state_history and state.ndim == 1:
            state = state[None, :]
        return state

    def _extract_action_array(self, response: Any) -> tuple[Any, str]:
        if isinstance(response, dict):
            if response.get("status") == "error" or response.get("ok") is False:
                raise RuntimeError(f"LDA-1B inference failed: {response.get('error', response)}")
            data = response.get("data", response)
        else:
            data = response

        source_key = self.action_output_key
        if isinstance(data, dict):
            if source_key:
                if source_key not in data:
                    raise KeyError(f"LDA-1B response missing configured action_output_key '{source_key}'")
                value = data[source_key]
            else:
                source_key, value = _first_present(
                    data,
                    (
                        "actions",
                        "action",
                        "raw_actions",
                        "normalized_actions",
                        "pred_actions",
                    ),
                )
        else:
            value = data
            source_key = source_key or "response"

        actions = self._action_value_to_array(value)
        actions, unnormalized = self._maybe_unnormalize(actions, source_key)
        self.last_payload_metadata["action_source_key"] = source_key
        self.last_payload_metadata["action_output_shape"] = _shape_of(actions)
        self.last_payload_metadata["action_unnormalized"] = unnormalized
        return actions, source_key

    def _action_value_to_array(self, value: Any) -> Any:
        np = _import_numpy()
        if isinstance(value, dict):
            keys = self.action_dict_order or sorted(value.keys())
            arrays = [np.asarray(_as_array(value[key])) for key in keys]
            try:
                return np.concatenate(arrays, axis=-1)
            except ValueError:
                flattened = [array.reshape(-1) for array in arrays]
                return np.concatenate(flattened, axis=0)
        return np.asarray(_as_array(value))

    def _maybe_unnormalize(self, actions: Any, source_key: str) -> tuple[Any, bool]:
        if source_key != "normalized_actions":
            return actions, False
        if self.action_min is None or self.action_max is None:
            if self.allow_normalized_actions:
                return actions, False
            raise RuntimeError(
                "LDA-1B returned normalized_actions but no action_min/action_max metadata was provided. "
                "Provide normalization stats, configure action_output_key for raw actions, or set "
                "allow_normalized_actions=true if the simulator intentionally expects normalized values."
            )
        np = _import_numpy()
        normalized = np.asarray(actions, dtype=np.float32)
        if self.clip_normalized_actions:
            normalized = np.clip(normalized, -1.0, 1.0)
        action_min = np.asarray(self.action_min, dtype=np.float32)
        action_max = np.asarray(self.action_max, dtype=np.float32)
        if action_min.shape[-1] != normalized.shape[-1] or action_max.shape[-1] != normalized.shape[-1]:
            raise RuntimeError(
                "LDA-1B action_min/action_max dimensions do not match normalized action output: "
                f"min={action_min.shape}, max={action_max.shape}, actions={normalized.shape}"
            )
        unnormalized = (normalized + 1.0) / 2.0 * (action_max - action_min) + action_min
        if self.action_mask is not None:
            mask = np.asarray(self.action_mask, dtype=bool)
            if mask.shape[-1] != normalized.shape[-1]:
                raise RuntimeError(
                    "LDA-1B action_mask dimension does not match normalized action output: "
                    f"mask={mask.shape}, actions={normalized.shape}"
                )
            unnormalized = np.where(mask, unnormalized, normalized)
        return unnormalized, True

    def _queue_actions(self, actions: Any, source_key: str) -> None:
        np = _import_numpy()
        array = np.asarray(actions)
        steps = self._action_steps(array)
        self.pending_actions = [self._to_action(np.asarray(step).reshape(-1), source_key) for step in steps]

    def _action_steps(self, array: Any) -> list[Any]:
        if array.ndim == 0:
            raise RuntimeError("LDA-1B action output was scalar; expected an action vector or chunk")
        if array.ndim == 1:
            return [array]
        if array.ndim == 2:
            if self.action_layout == "dim_first":
                return [array[:, index] for index in range(array.shape[1])]
            if self.action_layout == "single":
                return [array.reshape(-1)]
            return [array[index] for index in range(array.shape[0])]
        selected = array[0]
        if selected.ndim == 1:
            return [selected]
        return [step for step in selected.reshape(-1, selected.shape[-1])]

    def _to_action(self, raw_step: Any, source_key: str) -> Action:
        raw_values = [float(value) for value in raw_step.tolist()]
        values = self._project_action(raw_values)
        self._validate_action_dim(values, len(raw_values))
        translation = values[:3] if len(values) >= 3 else [0.0, 0.0, 0.0]
        if self.default_action_type == "ee" and len(values) >= 8:
            rotation = values[3:7]
            gripper = values[7]
        else:
            rotation = values[3:6] if len(values) >= 6 else [0.0, 0.0, 0.0]
            gripper = values[6] if len(values) >= 7 else 0.0
        return Action(
            type=self.default_action_type,
            action_type=self.default_action_type,
            vector=values,
            translation=translation,
            rotation=rotation,
            gripper=gripper,
            metadata={
                "raw_action_dim": len(values),
                "model_raw_action_dim": len(raw_values),
                "source_key": source_key,
                "normalized": source_key == "normalized_actions"
                and (self.action_min is None or self.action_max is None),
            },
        )

    def _project_action(self, values: list[float]) -> list[float]:
        if self.action_indices:
            return [values[index] for index in self.action_indices]
        if self.action_slice is not None:
            start, end = self.action_slice
            return values[start:end]
        return values

    def _validate_action_dim(self, values: list[float], raw_dim: int) -> None:
        if self.expected_action_dim is None:
            return
        if len(values) != self.expected_action_dim:
            projection = " after action projection" if len(values) != raw_dim else ""
            raise RuntimeError(
                f"LDA-1B produced {raw_dim} action values and {len(values)} values{projection}; "
                f"expected {self.expected_action_dim}. Configure expected_action_dim, action_slice, "
                "or action_indices to match the simulator contract."
            )


class _LDAWebSocketClient:
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
        self.server_metadata: dict[str, Any] = {}
        self._connect()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None or self.packer is None or self.unpackb is None:
            raise RuntimeError("LDA-1B WebSocket client is not connected")
        self.ws.send(self.packer.pack(payload))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LDA-1B server returned an error string: {response}")
        return self.unpackb(response)

    def _connect(self) -> None:
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        try:
            import websockets.sync.client
            from deployment.model_server.tools import msgpack_numpy
        except ImportError as exc:
            raise RuntimeError(
                "LDA-1B adapter requires the local LDA-1B repo plus its websocket/msgpack dependencies"
            ) from exc

        uri = f"ws://{self.host}:{self.port}"
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        deadline = time.monotonic() + self.connect_timeout_s
        last_error: Exception | None = None
        for proxy_key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(proxy_key, None)
        while True:
            try:
                self.ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                )
                self.packer = msgpack_numpy.Packer()
                self.unpackb = msgpack_numpy.unpackb
                metadata = self.unpackb(self.ws.recv())
                self.server_metadata = metadata if isinstance(metadata, dict) else {"raw": metadata}
                return
            except Exception as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(1)
        raise RuntimeError(f"Could not connect to LDA-1B server at {uri}") from last_error


def _default_lda_repo() -> Path:
    return api_root() / "models" / "LDA-1B"


def _with_profile_defaults(metadata: dict[str, Any], benchmark: str) -> dict[str, Any]:
    value = benchmark.lower()
    profile = str(metadata.get("profile") or metadata.get("simulator") or benchmark).lower()
    if "robotwin" not in value and "robotwin" not in profile:
        return metadata

    merged = dict(metadata)
    merged.setdefault(
        "image_keys",
        [
            "observation.head_camera.rgb",
            "observation.left_camera.rgb",
            "observation.right_camera.rgb",
        ],
    )
    merged.setdefault("state_keys", ["joint_action.vector"])
    merged.setdefault("require_state", True)
    merged.setdefault("expected_state_dim", 16)
    merged.setdefault("expected_image_count", 3)
    merged.setdefault("action_type", "qpos")
    merged.setdefault("expected_action_dim", 16)
    merged.setdefault("allow_normalized_actions", False)
    return merged


def _append_dot_args(command: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        command.extend([f"--{key}", _format_cli_value(value)])


def _format_cli_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def _import_numpy() -> Any:
    import numpy as np

    return np


def _as_array(value: Any) -> Any:
    np = _import_numpy()
    if isinstance(value, dict) and value.get("encoding") == "raw_base64":
        return decode_array(value)
    return np.asarray(value)


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


def _state_array(value: Any) -> Any | None:
    if value is None:
        return None
    np = _import_numpy()
    if isinstance(value, dict) and value.get("encoding") == "raw_base64":
        return decode_array(value)
    if isinstance(value, (int, float, list, tuple)):
        array = np.asarray(value, dtype=np.float32)
        return array if array.size else None
    if hasattr(value, "astype") and hasattr(value, "shape"):
        return value
    return None


def _first_state_vector(observation: Observation) -> list[float]:
    preferred_suffixes = (
        "proprio",
        "agent_pos",
        "eef_pos",
        "robot0_eef_pos",
        "joint_pos",
        "joint_states",
        "robot_state",
        "state",
        "qpos",
    )
    for suffix in preferred_suffixes:
        for key, value in observation.state.items():
            if key.lower().endswith(suffix):
                vector = _numeric_vector(value)
                if vector:
                    return vector
    for value in observation.state.values():
        vector = _numeric_vector(value)
        if vector:
            return vector
    return []


def _default_image_keys(images: dict[str, Any]) -> list[str]:
    priority = (
        "primary",
        "image",
        "agentview_image",
        "agentview_rgb",
        "observation.images.agentview_rgb",
        "robot0_eye_in_hand_image",
        "eye_in_hand_rgb",
        "observation.images.eye_in_hand_rgb",
        "observation.images.cam_high",
        "observation.head_camera.rgb",
        "head_camera.rgb",
        "observation.images.cam_left_wrist",
        "observation.left_camera.rgb",
        "left_camera.rgb",
        "observation.images.cam_right_wrist",
        "observation.right_camera.rgb",
        "right_camera.rgb",
    )
    selected: list[str] = [key for key in priority if key in images]
    selected.extend(key for key in sorted(images.keys()) if key not in selected)
    return selected


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any]:
    for key in keys:
        if key in data:
            return key, data[key]
    raise KeyError(f"LDA-1B response did not include an action key. Available keys: {sorted(data.keys())}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _looks_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _load_action_norm_stats(metadata: dict[str, Any]) -> dict[str, Any] | None:
    direct_stats = metadata.get("action_norm_stats")
    unnorm_key = metadata.get("unnorm_key") or metadata.get("dataset_key") or os.environ.get("LDA_1B_UNNORM_KEY")
    if direct_stats:
        stats = _select_action_stats(direct_stats, unnorm_key, "metadata.action_norm_stats")
        stats["_source"] = "metadata.action_norm_stats"
        return stats

    stats_path = _resolve_norm_stats_path(metadata)
    if stats_path is None:
        return None
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = _select_action_stats(payload, unnorm_key, str(stats_path))
    stats["_source"] = str(stats_path)
    return stats


def _resolve_norm_stats_path(metadata: dict[str, Any]) -> Path | None:
    raw_stats_path = (
        metadata.get("norm_stats_path")
        or metadata.get("dataset_statistics_path")
        or metadata.get("dataset_stats_path")
    )
    if raw_stats_path:
        path = resolve_path(raw_stats_path)
        if not path.exists():
            raise FileNotFoundError(f"LDA-1B action norm stats file does not exist: {path}")
        return path

    raw_checkpoint = (
        metadata.get("checkpoint_path")
        or metadata.get("policy_ckpt_path")
        or metadata.get("pretrained_checkpoint")
        or metadata.get("checkpoint")
        or os.environ.get("LDA_1B_CHECKPOINT")
    )
    if not raw_checkpoint:
        return None

    checkpoint_path = resolve_path(raw_checkpoint)
    candidates: list[Path] = []
    if checkpoint_path.is_file():
        candidates.append(checkpoint_path.parent / "dataset_statistics.json")
        if len(checkpoint_path.parents) > 1:
            candidates.append(checkpoint_path.parents[1] / "dataset_statistics.json")
    elif checkpoint_path.is_dir():
        candidates.append(checkpoint_path / "dataset_statistics.json")
    else:
        raise FileNotFoundError(f"LDA-1B checkpoint path does not exist: {checkpoint_path}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find LDA-1B dataset_statistics.json near checkpoint path "
        f"{checkpoint_path}. Configure norm_stats_path/dataset_statistics_path explicitly."
    )


def _select_action_stats(stats: Any, unnorm_key: Any, source: str) -> dict[str, Any]:
    if not isinstance(stats, dict):
        raise ValueError(f"LDA-1B action normalization stats from {source} must be a JSON object")
    if _is_action_stats(stats):
        return dict(stats)
    if isinstance(stats.get("action"), dict):
        return _validate_action_stats(dict(stats["action"]), source)

    if unnorm_key:
        key = str(unnorm_key)
        if key not in stats:
            raise KeyError(f"LDA-1B unnorm_key '{key}' was not found in {source}; available keys: {sorted(stats.keys())}")
        return _action_stats_for_dataset(stats[key], f"{source}:{key}")

    if len(stats) == 1:
        key, value = next(iter(stats.items()))
        return _action_stats_for_dataset(value, f"{source}:{key}")

    raise ValueError(
        f"LDA-1B normalization stats from {source} contain multiple datasets; configure unnorm_key. "
        f"Available keys: {sorted(stats.keys())}"
    )


def _action_stats_for_dataset(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"LDA-1B dataset stats at {source} must be a JSON object")
    if isinstance(value.get("action"), dict):
        return _validate_action_stats(dict(value["action"]), source)
    return _validate_action_stats(dict(value), source)


def _is_action_stats(value: dict[str, Any]) -> bool:
    return "min" in value and "max" in value


def _validate_action_stats(value: dict[str, Any], source: str) -> dict[str, Any]:
    if "min" not in value or "max" not in value:
        raise ValueError(f"LDA-1B action normalization stats at {source} must include 'min' and 'max'")
    return value


def _optional_array(np: Any, value: Any) -> Any | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_slice(value: Any) -> tuple[int | None, int | None] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("action_slice must be START:END or [START, END]")
        return (_slice_part(parts[0]), _slice_part(parts[1]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_slice_part(value[0]), _slice_part(value[1]))
    raise ValueError("action_slice must be START:END or [START, END]")


def _slice_part(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _shape_of(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None
