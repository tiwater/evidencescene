"""Finite canonical JSON and content hashing shared by runtime artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping


def finite_json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return finite_json_value(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return 0.0 if value == 0.0 else value
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return finite, sorted, ASCII canonical JSON without whitespace."""

    return json.dumps(
        finite_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

