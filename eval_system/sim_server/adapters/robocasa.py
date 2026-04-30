from __future__ import annotations

from ...common.registry import register_sim
from .dummy import DummySimulatorAdapter


@register_sim("robocasa")
class RoboCasaAdapter(DummySimulatorAdapter):
    def __init__(self) -> None:
        super().__init__(simulator_name="robocasa")
