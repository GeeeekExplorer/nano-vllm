from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from nanovllm.avg.constants import CORE_STATE_KEYS


def collect_unknown_fields(payload: Mapping[str, Any], known_fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in known_fields}


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    raise TypeError("timestamp must be a datetime, ISO string, or unix timestamp")


def split_known_state_values(values: Mapping[str, Any] | None) -> tuple[dict[str, float], dict[str, Any]]:
    known: dict[str, float] = {}
    unknown: dict[str, Any] = {}
    if not values:
        return known, unknown
    for key, value in values.items():
        if key in CORE_STATE_KEYS:
            known[key] = float(value)
        else:
            unknown[key] = value
    return known, unknown


def merge_metadata(
    payload: Mapping[str, Any],
    known_fields: set[str],
    unknown_groups: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    metadata = dict(payload.get("metadata") or {})
    extras = collect_unknown_fields(payload, known_fields)
    for key, values in unknown_groups.items():
        if values:
            metadata[key] = values
    if extras:
        metadata["extra_fields"] = extras
    return metadata or None
