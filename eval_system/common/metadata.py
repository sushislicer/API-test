from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def scoped_metadata(
    metadata: dict[str, Any],
    component_name: str,
    group_key: str,
    generic_key: str | None = None,
) -> dict[str, Any]:
    """Merge top-level metadata with optional component-specific overrides."""

    merged = dict(metadata)
    if generic_key and isinstance(metadata.get(generic_key), Mapping):
        deep_update(merged, dict(metadata[generic_key]))

    group = metadata.get(group_key)
    if isinstance(group, Mapping):
        default = _lookup_mapping(group, "default")
        if default:
            deep_update(merged, default)
        component = _lookup_mapping(group, component_name)
        if component:
            deep_update(merged, component)

    return merged


def training_metadata(
    metadata: dict[str, Any],
    model_name: str,
    simulator_name: str | None = None,
) -> dict[str, Any]:
    """Merge generic, model, and model/simulator training overrides."""

    merged = scoped_metadata(metadata, model_name, "models", "model")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        return merged

    default = _lookup_mapping(training, "default")
    if default:
        deep_update(merged, default)
    if simulator_name:
        simulator_default = _lookup_mapping(training, simulator_name)
        if simulator_default:
            deep_update(merged, simulator_default)

    model_training = _lookup_mapping(training, model_name)
    if model_training:
        model_default = _lookup_mapping(model_training, "default")
        if model_default:
            deep_update(merged, model_default)
        if simulator_name:
            model_simulator = _lookup_mapping(model_training, simulator_name)
            if model_simulator:
                deep_update(merged, model_simulator)
        direct_values = {
            key: value
            for key, value in model_training.items()
            if not isinstance(value, Mapping)
            or _normalize_key(key) not in {_normalize_key("default"), _normalize_key(simulator_name or "")}
        }
        deep_update(merged, direct_values)

    return merged


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = dict(base[key])
            deep_update(nested, value)
            base[key] = nested
        else:
            base[key] = value
    return base


def apply_aliases(metadata: dict[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    updated = dict(metadata)
    normalized_keys = {_normalize_key(key): key for key in updated}
    for alias, canonical in aliases.items():
        alias_key = normalized_keys.get(_normalize_key(alias))
        if alias_key is None:
            continue
        if canonical not in updated:
            updated[canonical] = updated[alias_key]
    return updated


def _lookup_mapping(container: Mapping[str, Any], key: str) -> dict[str, Any]:
    normalized = _normalize_key(key)
    for item_key, value in container.items():
        if _normalize_key(str(item_key)) == normalized and isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalize_key(value: str) -> str:
    return value.lower().replace("-", "_").replace(".", "_")
