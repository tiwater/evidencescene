"""Immutable camera-local point-frame and point-track primitive contracts.

Placed in ``evidencescene.primitives`` so both ``evidencescene.surface`` and
``evidencescene.tracking`` depend strictly downward on ``primitives`` with zero
circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from evidencescene.primitives.content_hash import content_sha256


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _frozen_array(
    value: object,
    *,
    dtype: np.dtype,
    name: str,
    shape: tuple[int, ...] | None = None,
    last_dimension: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{name} must be a {dtype.str} ndarray")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if last_dimension is not None and (
        value.ndim < 1 or value.shape[-1] != last_dimension
    ):
        raise ValueError(f"{name} has an invalid shape")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    result = np.ndarray(
        value.shape,
        dtype=dtype,
        buffer=np.ascontiguousarray(value).tobytes(),
    )
    result.flags.writeable = False
    return result


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _diagnostics(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostics must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or any(token in key.lower() for token in ("world", "pose"))
        ):
            raise ValueError("world or pose diagnostics are forbidden")
        if isinstance(item, bool) or item is None or isinstance(item, (str, int)):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        else:
            raise ValueError("diagnostics must contain finite JSON scalars")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class CausalPointFrame:
    """Immutable current RGB observation; it contains no world-state input."""

    source_sha256: str
    color_rgb_sha256: str
    camera_id: str
    timestamp_ns: int
    color_rgb: np.ndarray
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_sha256 = _sha256(self.source_sha256, "source hash")
        color_sha256 = _sha256(self.color_rgb_sha256, "color hash")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        color = _frozen_array(
            self.color_rgb,
            dtype=np.dtype("u1"),
            name="color_rgb",
            last_dimension=3,
        )
        if color.ndim != 3 or min(color.shape[:2]) <= 0:
            raise ValueError("color_rgb must be a non-empty HxWx3 image")
        if _array_sha256(color) != color_sha256:
            raise ValueError("color hash does not match actual bytes")
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "color_rgb_sha256", color_sha256)
        object.__setattr__(self, "color_rgb", color)
        object.__setattr__(
            self,
            "content_hash",
            content_sha256(
                {
                    "schema": "ticos.causal-point-frame.v1",
                    "source_sha256": source_sha256,
                    "color_rgb_sha256": color_sha256,
                    "camera_id": self.camera_id,
                    "timestamp_ns": self.timestamp_ns,
                    "shape": list(color.shape),
                }
            ),
        )

    @classmethod
    def from_rgb(
        cls,
        *,
        source_sha256: str,
        camera_id: str,
        timestamp_ns: int,
        color_rgb: np.ndarray,
    ) -> "CausalPointFrame":
        return cls(
            source_sha256=source_sha256,
            color_rgb_sha256=_array_sha256(color_rgb),
            camera_id=camera_id,
            timestamp_ns=timestamp_ns,
            color_rgb=color_rgb,
        )


@dataclass(frozen=True)
class CausalPointTracks:
    """Current camera-local point observations with immutable query identity."""

    status: Literal["candidate", "not_observable"]
    reason: str
    frame_content_hash: str
    model_sha256: str
    camera_id: str
    timestamp_ns: int
    query_ids: np.ndarray
    positions_xy: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    inference_latency_ms: float
    diagnostics: Mapping[str, Any]
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_hash = _sha256(self.frame_content_hash, "frame content hash")
        model_hash = _sha256(self.model_sha256, "model hash")
        if self.status not in {"candidate", "not_observable"}:
            raise ValueError("invalid point-track status")
        if not isinstance(self.reason, str):
            raise ValueError("point-track reason must be text")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id is required")
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        query_ids = _frozen_array(
            self.query_ids, dtype=np.dtype("<i8"), name="query_ids"
        )
        positions = _frozen_array(
            self.positions_xy,
            dtype=np.dtype("<f8"),
            name="positions_xy",
            last_dimension=2,
        )
        visible = _frozen_array(
            self.visible, dtype=np.dtype("?"), name="visible"
        )
        confidence = _frozen_array(
            self.confidence, dtype=np.dtype("<f8"), name="confidence"
        )
        if (
            query_ids.ndim != 1
            or positions.ndim != 2
            or visible.ndim != 1
            or confidence.ndim != 1
            or not len(query_ids) == len(positions) == len(visible) == len(confidence)
        ):
            raise ValueError("point-track arrays must share a one-dimensional query axis")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("point confidence must be in [0,1]")
        if not isinstance(self.inference_latency_ms, (int, float)) or not math.isfinite(
            float(self.inference_latency_ms)
        ) or float(self.inference_latency_ms) < 0.0:
            raise ValueError("inference latency must be finite and nonnegative")
        diagnostics = _diagnostics(self.diagnostics)
        if self.status == "candidate":
            if self.reason or not len(query_ids) or not np.any(visible):
                raise ValueError("candidate tracks require current visible observations")
        elif not self.reason:
            raise ValueError("not_observable tracks require a reason")
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "positions_xy", positions)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "inference_latency_ms", float(self.inference_latency_ms))
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(
            self,
            "content_hash",
            content_sha256(
                {
                    "schema": "ticos.causal-point-tracks.v1",
                    "status": self.status,
                    "reason": self.reason,
                    "frame_content_hash": frame_hash,
                    "model_sha256": model_hash,
                    "camera_id": self.camera_id,
                    "timestamp_ns": self.timestamp_ns,
                    "query_ids": query_ids.tolist(),
                    "positions_xy": positions.tolist(),
                    "visible": visible.tolist(),
                    "confidence": confidence.tolist(),
                    "inference_latency_ms": float(self.inference_latency_ms),
                    "diagnostics": dict(diagnostics),
                }
            ),
        )
