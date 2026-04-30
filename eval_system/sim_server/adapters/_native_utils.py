from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...common.arrays import encode_array
from ...common.schemas import Action, JsonDict, Observation


def action_vector(action: Action, expected_length: int | None = None) -> list[float]:
    vector = [float(value) for value in action.to_vector()]
    if expected_length is not None and len(vector) != expected_length:
        raise ValueError(f"Expected {expected_length} action values, got {len(vector)}")
    return vector


def action_type(action: Action, default: str) -> str:
    if action.metadata.get("action_type"):
        return str(action.metadata["action_type"])
    if action.action_type and action.action_type != "delta_ee_pose":
        return action.action_type
    if action.type and action.type != "delta_ee_pose":
        return action.type
    return default


def jsonable(value: Any, max_depth: int = 8) -> Any:
    if max_depth < 0:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _is_numpy_scalar(value):
        return value.item()
    if _is_numpy_array(value):
        return encode_array(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item, max_depth - 1) for key, item in value.items()}
    if _is_sequence(value):
        return [jsonable(item, max_depth - 1) for item in value]
    return repr(value)


def native_observation(
    native: Any,
    *,
    simulator_name: str,
    language_instruction: str = "",
    metadata: JsonDict | None = None,
    extra_images: JsonDict | None = None,
) -> Observation:
    images: JsonDict = {}
    depth_images: JsonDict = {}
    state: JsonDict = {}
    proprio: list[float] = []
    objects: JsonDict = {}

    _collect_native_fields(native, "", images, depth_images, state)
    if extra_images:
        images.update(extra_images)

    proprio = _first_numeric_vector(
        state,
        preferred_suffixes=(
            "joint_action.vector",
            "proprio",
            "agent_pos",
            "eef_pos",
            "robot0_eef_pos",
            "joint_pos",
            "joint_states",
            "robot_state",
            "state",
            "qpos",
            "endpose",
        ),
    )

    gripper = _first_gripper_value(state)
    primary_rgb = next(iter(images), None)
    primary_depth = next(iter(depth_images), None)

    return Observation(
        rgb=f"native://{simulator_name}/{primary_rgb}" if primary_rgb else None,
        depth=f"native://{simulator_name}/{primary_depth}" if primary_depth else None,
        images=images,
        depth_images=depth_images,
        proprio=proprio,
        gripper=gripper,
        language_instruction=language_instruction,
        objects=objects,
        state=state,
        metadata={
            "simulator_name": simulator_name,
            **(metadata or {}),
        },
    )


def extract_success(info: Any, reward: float | None = None, fallback: bool = False) -> bool:
    if isinstance(info, Mapping):
        for key in ("success", "eval_success", "is_success"):
            if key in info:
                return bool(_scalar_value(info[key]))
        episode_stats = info.get("episode_stats")
        if isinstance(episode_stats, Mapping):
            for key in ("success", "is_success"):
                if key in episode_stats:
                    return bool(_scalar_value(episode_stats[key]))
    if reward is not None and reward >= 1.0:
        return True
    return fallback


def scalar_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if _is_numpy_scalar(value):
        return float(value.item())
    if _is_numpy_array(value):
        return float(value.reshape(-1)[0].item())
    if hasattr(value, "detach"):
        return scalar_float(value.detach().cpu().numpy())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _collect_native_fields(native: Any, path: str, images: JsonDict, depth_images: JsonDict, state: JsonDict) -> None:
    if isinstance(native, Mapping):
        for key, value in native.items():
            next_path = f"{path}.{key}" if path else str(key)
            _collect_native_fields(value, next_path, images, depth_images, state)
        return

    if _is_numpy_array(native):
        encoded = encode_array(native)
        lower_path = path.lower()
        shape = [int(dim) for dim in native.shape]
        if "depth" in lower_path:
            depth_images[path] = encoded
        elif _looks_like_image(lower_path, shape):
            images[path] = encoded
        else:
            state[path] = encoded
        return

    if _is_sequence(native):
        flattened = _flatten_numeric(native)
        if flattened:
            state[path] = [float(value) for value in flattened]
        else:
            state[path] = jsonable(native)
        return

    if native is not None and path:
        state[path] = jsonable(native)


def _first_numeric_vector(state: JsonDict, preferred_suffixes: tuple[str, ...]) -> list[float]:
    for suffix in preferred_suffixes:
        for key, value in state.items():
            if key.lower().endswith(suffix):
                vector = _numeric_vector(value)
                if len(vector) >= 3:
                    return vector
    for value in state.values():
        vector = _numeric_vector(value)
        if len(vector) >= 3:
            return vector
    return []


def _first_gripper_value(state: JsonDict) -> float | None:
    for key, value in state.items():
        if "gripper" not in key.lower():
            continue
        vector = _numeric_vector(value)
        if vector:
            return float(vector[-1])
    return None


def _numeric_vector(value: Any) -> list[float]:
    if isinstance(value, Mapping) and value.get("encoding") == "raw_base64":
        return []
    flattened = _flatten_numeric(value)
    return [float(item) for item in flattened]


def _flatten_numeric(value: Any) -> list[float]:
    if value is None or isinstance(value, (str, bytes, bytearray, bool)):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if _is_numpy_scalar(value):
        return [float(value.item())]
    if _is_numpy_array(value):
        if len(value.shape) > 1 and _looks_like_image("", [int(dim) for dim in value.shape]):
            return []
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Mapping):
        flattened: list[float] = []
        for item in value.values():
            flattened.extend(_flatten_numeric(item))
        return flattened
    if _is_sequence(value):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return []


def _looks_like_image(path: str, shape: list[int]) -> bool:
    if "rgb" in path or "image" in path:
        return True
    if "camera" in path and len(shape) >= 2:
        return True
    return len(shape) >= 3 and shape[-1] in {1, 3, 4}


def _scalar_value(value: Any) -> Any:
    if _is_numpy_scalar(value):
        return value.item()
    if _is_numpy_array(value) and value.shape == ():
        return value.item()
    return value


def _is_numpy_array(value: Any) -> bool:
    return hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes")


def _is_numpy_scalar(value: Any) -> bool:
    return hasattr(value, "item") and type(value).__module__.startswith("numpy")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
