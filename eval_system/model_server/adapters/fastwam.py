from __future__ import annotations

from ...common.registry import register_model
from .dummy import ExternalTrainingDummyAdapter


@register_model("fastwam")
class FastWAMAdapter(ExternalTrainingDummyAdapter):
    def __init__(self) -> None:
        super().__init__(model_name="fastwam")
