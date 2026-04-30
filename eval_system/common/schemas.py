from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class Observation:
    rgb: str | None = None
    depth: str | None = None
    images: JsonDict = field(default_factory=dict)
    depth_images: JsonDict = field(default_factory=dict)
    proprio: list[float] = field(default_factory=list)
    gripper: float | None = None
    language_instruction: str = ""
    objects: JsonDict = field(default_factory=dict)
    state: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Observation":
        return cls(**data)


@dataclass
class Action:
    type: str = "delta_ee_pose"
    action_type: str = "delta_ee_pose"
    vector: list[float] = field(default_factory=list)
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gripper: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Action":
        payload = dict(data)
        payload.setdefault("action_type", payload.get("type", "delta_ee_pose"))
        return cls(**payload)

    def to_vector(self) -> list[float]:
        if self.vector:
            return list(self.vector)
        return [*self.translation, *self.rotation, self.gripper]


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    success: bool = False
    info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["observation"] = self.observation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: JsonDict) -> "StepResult":
        return cls(
            observation=Observation.from_dict(data["observation"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
            success=bool(data.get("success", False)),
            info=data.get("info", {}),
        )


@dataclass
class TaskSpec:
    benchmark: str
    task: str
    episode_horizon: int = 100
    seed: int = 0
    instruction: str = ""
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TaskSpec":
        return cls(**data)


@dataclass
class TransitionQuery:
    observation: Observation
    action: Action

    def to_dict(self) -> JsonDict:
        return {
            "observation": self.observation.to_dict(),
            "action": self.action.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TransitionQuery":
        return cls(
            observation=Observation.from_dict(data["observation"]),
            action=Action.from_dict(data["action"]),
        )


@dataclass
class TrainingSpec:
    benchmark: str = ""
    dataset_path: str = ""
    config_name: str = ""
    output_dir: str = ""
    checkpoint_path: str = ""
    seed: int = 0
    dry_run: bool = True
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TrainingSpec":
        return cls(**data)


@dataclass
class EvalSpec:
    model_endpoint: str
    sim_endpoint: str
    benchmark: str
    task: str
    episodes: int = 1
    horizon: int = 100
    seed: int = 0
    mode: str = "policy"
    predict_next: bool = True
    output_dir: str = "out"
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "EvalSpec":
        return cls(**data)


@dataclass
class EpisodeLog:
    episode_index: int
    task: str
    steps: int
    episode_return: float
    success: bool
    avg_prediction_error: float | None = None
    records: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class EvalReport:
    benchmark: str
    task: str
    episodes: int
    success_rate: float
    avg_return: float
    avg_episode_length: float
    avg_prediction_error: float | None = None
    metrics: JsonDict = field(default_factory=dict)
    episode_summaries: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)
