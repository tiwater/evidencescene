"""Fail-closed camera-local boundary for causal long-range point trackers."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from evidencescene.primitives.point_tracks import (
    CausalPointFrame,
    CausalPointTracks,
    _array_sha256,
    _diagnostics,
    _frozen_array,
    _sha256,
)


PointTrackerRunner = Callable[
    [np.ndarray, np.ndarray | None, object | None], Mapping[str, Any]
]


class CausalPointTrackerProvider:
    """Own one camera's recurrent state and reject replay/future/identity changes."""

    def __init__(
        self,
        *,
        model_sha256: str,
        runner: PointTrackerRunner,
        maximum_query_points: int = 1024,
        minimum_visible_points: int = 1,
    ) -> None:
        self._model_sha256 = _sha256(model_sha256, "model hash")
        if maximum_query_points <= 0:
            raise ValueError("maximum_query_points must be positive")
        if minimum_visible_points <= 0 or minimum_visible_points > maximum_query_points:
            raise ValueError("minimum_visible_points is outside the query budget")
        self._runner = runner
        self._maximum_query_points = int(maximum_query_points)
        self._minimum_visible_points = int(minimum_visible_points)
        self._camera_id: str | None = None
        self._query_ids: np.ndarray | None = None
        self._state: object | None = None
        self._last_timestamp_ns: int | None = None

    def start(
        self, frame: CausalPointFrame, query_points_xy: np.ndarray
    ) -> CausalPointTracks:
        if self._query_ids is not None:
            raise ValueError("causal point provider is already initialized")
        points = _frozen_array(
            query_points_xy,
            dtype=np.dtype("<f8"),
            name="query_points_xy",
            last_dimension=2,
        )
        if points.ndim != 2 or not 0 < len(points) <= self._maximum_query_points:
            raise ValueError("query point count is outside the frozen budget")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("query points must be unique")
        height, width = frame.color_rgb.shape[:2]
        if np.any(points[:, 0] < 0.0) or np.any(points[:, 0] >= width) or np.any(
            points[:, 1] < 0.0
        ) or np.any(points[:, 1] >= height):
            raise ValueError("query points must lie inside the current image")
        self._camera_id = frame.camera_id
        self._query_ids = np.arange(len(points), dtype=np.int64)
        self._query_ids.flags.writeable = False
        return self._run(frame, points)

    def track(self, frame: CausalPointFrame) -> CausalPointTracks:
        if self._query_ids is None:
            raise ValueError("causal point provider must be initialized")
        return self._run(frame, None)

    def _run(
        self, frame: CausalPointFrame, query_points_xy: np.ndarray | None
    ) -> CausalPointTracks:
        if not isinstance(frame, CausalPointFrame):
            raise ValueError("frame must be CausalPointFrame")
        if self._camera_id is not None and frame.camera_id != self._camera_id:
            raise ValueError("one provider cannot mix physical cameras")
        if self._last_timestamp_ns is not None and frame.timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("causal point frames must be strictly increasing")
        self._last_timestamp_ns = frame.timestamp_ns
        assert self._query_ids is not None
        try:
            raw = self._runner(frame.color_rgb, query_points_xy, self._state)
            if not isinstance(raw, Mapping) or set(raw) != {
                "positions_xy",
                "visible",
                "confidence",
                "state",
                "inference_latency_ms",
                "diagnostics",
            }:
                raise ValueError("point runner output has unsupported fields")
            if any(
                token in str(key).lower()
                for key in raw
                for token in ("world", "pose")
            ):
                raise ValueError("point runner cannot emit world or pose state")
            positions = np.asarray(raw["positions_xy"], dtype=np.float64)
            visible = np.asarray(raw["visible"], dtype=np.bool_)
            confidence = np.asarray(raw["confidence"], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1:] != (2,):
                raise ValueError("point positions must have shape Nx2")
            if visible.ndim != 1 or confidence.ndim != 1:
                raise ValueError("point visibility and confidence must be vectors")
            if not len(positions) == len(visible) == len(confidence) == len(
                self._query_ids
            ):
                raise ValueError("point runner changed query identity cardinality")
            if not np.isfinite(positions).all() or not np.isfinite(confidence).all():
                raise ValueError("point runner output must be finite")
            if np.any(confidence < 0.0) or np.any(confidence > 1.0):
                raise ValueError("point confidence must be in [0,1]")
            latency_ms = float(raw["inference_latency_ms"])
            if not math.isfinite(latency_ms) or latency_ms < 0.0:
                raise ValueError("point latency must be finite and nonnegative")
            diagnostics = _diagnostics(raw["diagnostics"])
            height, width = frame.color_rgb.shape[:2]
            in_bounds = (
                (positions[:, 0] >= 0.0)
                & (positions[:, 0] < width)
                & (positions[:, 1] >= 0.0)
                & (positions[:, 1] < height)
            )
            visible = visible & in_bounds
            if int(np.sum(visible)) < self._minimum_visible_points:
                self._state = raw["state"]
                return self._unavailable(frame, "insufficient_visible_points")
            candidate = CausalPointTracks(
                status="candidate",
                reason="",
                frame_content_hash=frame.content_hash,
                model_sha256=self._model_sha256,
                camera_id=frame.camera_id,
                timestamp_ns=frame.timestamp_ns,
                query_ids=self._query_ids,
                positions_xy=positions,
                visible=visible,
                confidence=confidence,
                inference_latency_ms=latency_ms,
                diagnostics=diagnostics,
            )
            self._state = raw["state"]
            return candidate
        except (IndexError, KeyError, TypeError, ValueError):
            return self._unavailable(frame, "provider_output_invalid")

    def _unavailable(
        self, frame: CausalPointFrame, reason: str
    ) -> CausalPointTracks:
        assert self._query_ids is not None
        count = len(self._query_ids)
        return CausalPointTracks(
            status="not_observable",
            reason=reason,
            frame_content_hash=frame.content_hash,
            model_sha256=self._model_sha256,
            camera_id=frame.camera_id,
            timestamp_ns=frame.timestamp_ns,
            query_ids=self._query_ids,
            positions_xy=np.zeros((count, 2), dtype=np.float64),
            visible=np.zeros(count, dtype=np.bool_),
            confidence=np.zeros(count, dtype=np.float64),
            inference_latency_ms=0.0,
            diagnostics={},
        )
