from __future__ import annotations

from typing import Callable

from .base import ModelAdapter, SimulatorAdapter


MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {}
SIM_ADAPTERS: dict[str, type[SimulatorAdapter]] = {}


def register_model(name: str) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        MODEL_ADAPTERS[name] = cls
        return cls

    return decorator


def register_sim(name: str) -> Callable[[type[SimulatorAdapter]], type[SimulatorAdapter]]:
    def decorator(cls: type[SimulatorAdapter]) -> type[SimulatorAdapter]:
        SIM_ADAPTERS[name] = cls
        return cls

    return decorator
