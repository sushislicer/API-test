from __future__ import annotations

import base64
from typing import Any

from .schemas import JsonDict


def encode_array(value: Any) -> JsonDict:
    return {
        "encoding": "raw_base64",
        "dtype": str(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "data": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def decode_array(payload: JsonDict) -> Any:
    if payload.get("encoding") != "raw_base64":
        raise ValueError(f"Unsupported array encoding: {payload.get('encoding')}")
    import numpy as np

    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])
