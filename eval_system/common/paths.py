from __future__ import annotations

from pathlib import Path
from typing import Any


def api_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: Any, *, base: str | Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path(base).expanduser().resolve() if base is not None else api_root()
    return (root / path).resolve()


def resolve_optional_path(value: Any, *, base: str | Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    return resolve_path(value, base=base)

