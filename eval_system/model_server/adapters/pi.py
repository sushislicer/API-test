from __future__ import annotations

from ...common.registry import register_model
from .dummy import DummyModelAdapter


@register_model("pi0.7")
class PiAdapter(DummyModelAdapter):
    def __init__(self) -> None:
        super().__init__(model_name="pi0.7")
