from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


def dumps_json(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def loads_json(raw: bytes | str) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw) if raw else {}


def normalize_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme in {"grpc", "http"}:
        return f"http://{parsed.netloc}{parsed.path}"
    if parsed.scheme == "":
        return f"http://{endpoint}"
    raise ValueError(f"Unsupported endpoint scheme: {parsed.scheme}")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
