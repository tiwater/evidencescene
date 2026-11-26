"""Causal persistent object-point memory with an independent short-track verifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evidencescene.primitives.point_tracks import CausalPointTracks


def _frozen(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class CausalSurfaceTrackMemoryConfig:
    minimum_common_visible_points: int = 6
    minimum_confidence: float = 0.5
    maximum_verifier_disagreement_px: float = 2.5
    required_consecutive_agreement_frames: int = 2

    def __post_init__(self) -> None:
        if self.minimum_common_visible_points <= 0:
            raise ValueError("minimum common visible points must be positive")
        if not 0.0 < self.minimum_confidence <= 1.0:
            raise ValueError("minimum confidence must be in (0,1]")
        if (
            not np.isfinite(self.maximum_verifier_disagreement_px)
            or self.maximum_verifier_disagreement_px <= 0.0
        ):
            raise ValueError("maximum verifier disagreement must be positive")
        if self.required_consecutive_agreement_frames <= 0:
            raise ValueError("required consecutive agreement must be positive")


@dataclass(frozen=True)
class CausalSurfaceTrackMemoryOutput:
    status: str
    camera_id: str
    timestamp_ns: int
    landmark_ids: np.ndarray
    positions_xy: np.ndarray
    confidence: np.ndarray
    object_memory_valid: bool
    correction_authorized: bool
    request_short_reanchor: bool
    agreement_streak: int
    common_visible_count: int
    verifier_disagreement_p50_px: float | None

    def __post_init__(self) -> None:
        if self.status not in {
            "verified",
            "verification_pending",
            "memory_only",
            "disagreement",
            "unavailable",
        }:
            raise ValueError("unsupported surface-track memory status")
        ids = _frozen(self.landmark_ids, dtype=np.dtype("<i8"))
        positions = _frozen(self.positions_xy, dtype=np.dtype("<f8"))
        confidence = _frozen(self.confidence, dtype=np.dtype("<f8"))
        if ids.ndim != 1 or positions.shape != (len(ids), 2):
            raise ValueError("surface-track memory output shape is invalid")
        if confidence.shape != (len(ids),):
            raise ValueError("surface-track memory confidence shape is invalid")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("surface-track memory identities must be unique")
        if not np.isfinite(positions).all() or not np.isfinite(confidence).all():
            raise ValueError("surface-track memory output must be finite")
        object.__setattr__(self, "landmark_ids", ids)
        object.__setattr__(self, "positions_xy", positions)
        object.__setattr__(self, "confidence", confidence)


class CausalSurfaceTrackMemory:
    """Keep persistent identities primary; use re-anchors only as a verifier."""

    def __init__(self, config: CausalSurfaceTrackMemoryConfig) -> None:
        self._config = config
        self._last_timestamp_ns: int | None = None
        self._agreement_streak = 0

    def step(
        self,
        persistent: CausalPointTracks,
        verifier: CausalPointTracks | None,
    ) -> CausalSurfaceTrackMemoryOutput:
        if not isinstance(persistent, CausalPointTracks):
            raise ValueError("persistent track must be CausalPointTracks")
        if (
            self._last_timestamp_ns is not None
            and persistent.timestamp_ns <= self._last_timestamp_ns
        ):
            raise ValueError("surface-track memory timestamps must increase")
        self._last_timestamp_ns = persistent.timestamp_ns
        if verifier is not None and (
            not isinstance(verifier, CausalPointTracks)
            or verifier.camera_id != persistent.camera_id
            or verifier.timestamp_ns != persistent.timestamp_ns
        ):
            raise ValueError("short verifier must match persistent camera and time")

        persistent_valid = self._valid_rows(persistent)
        if not persistent_valid:
            self._agreement_streak = 0
            return self._output(
                persistent,
                status="unavailable",
                rows={},
                memory_valid=False,
                authorized=False,
                request_reanchor=True,
                common_count=0,
                disagreement=None,
            )
        if verifier is None or verifier.status != "candidate":
            self._agreement_streak = 0
            return self._output(
                persistent,
                status="memory_only",
                rows=persistent_valid,
                memory_valid=True,
                authorized=False,
                request_reanchor=True,
                common_count=0,
                disagreement=None,
            )

        verifier_valid = self._valid_rows(verifier)
        common_ids = sorted(persistent_valid.keys() & verifier_valid.keys())
        if len(common_ids) < self._config.minimum_common_visible_points:
            self._agreement_streak = 0
            return self._output(
                persistent,
                status="memory_only",
                rows=persistent_valid,
                memory_valid=True,
                authorized=False,
                request_reanchor=True,
                common_count=len(common_ids),
                disagreement=None,
            )
        disagreement = float(
            np.quantile(
                [
                    np.linalg.norm(
                        persistent_valid[landmark_id][0]
                        - verifier_valid[landmark_id][0]
                    )
                    for landmark_id in common_ids
                ],
                0.50,
            )
        )
        if disagreement > self._config.maximum_verifier_disagreement_px:
            self._agreement_streak = 0
            return self._output(
                persistent,
                status="disagreement",
                rows=persistent_valid,
                memory_valid=True,
                authorized=False,
                request_reanchor=True,
                common_count=len(common_ids),
                disagreement=disagreement,
            )

        self._agreement_streak += 1
        authorized = (
            self._agreement_streak
            >= self._config.required_consecutive_agreement_frames
        )
        return self._output(
            persistent,
            status="verified" if authorized else "verification_pending",
            rows=persistent_valid,
            memory_valid=True,
            authorized=authorized,
            request_reanchor=False,
            common_count=len(common_ids),
            disagreement=disagreement,
        )

    def _valid_rows(
        self, tracks: CausalPointTracks
    ) -> dict[int, tuple[np.ndarray, float]]:
        if tracks.status != "candidate":
            return {}
        return {
            int(landmark_id): (position, float(confidence))
            for landmark_id, position, visible, confidence in zip(
                tracks.query_ids,
                tracks.positions_xy,
                tracks.visible,
                tracks.confidence,
                strict=True,
            )
            if bool(visible) and float(confidence) >= self._config.minimum_confidence
        }

    def _output(
        self,
        tracks: CausalPointTracks,
        *,
        status: str,
        rows: dict[int, tuple[np.ndarray, float]],
        memory_valid: bool,
        authorized: bool,
        request_reanchor: bool,
        common_count: int,
        disagreement: float | None,
    ) -> CausalSurfaceTrackMemoryOutput:
        ids = sorted(rows)
        return CausalSurfaceTrackMemoryOutput(
            status=status,
            camera_id=tracks.camera_id,
            timestamp_ns=tracks.timestamp_ns,
            landmark_ids=np.asarray(ids, dtype=np.int64),
            positions_xy=np.asarray(
                [rows[landmark_id][0] for landmark_id in ids],
                dtype=np.float64,
            ).reshape((-1, 2)),
            confidence=np.asarray(
                [rows[landmark_id][1] for landmark_id in ids],
                dtype=np.float64,
            ),
            object_memory_valid=memory_valid,
            correction_authorized=authorized,
            request_short_reanchor=request_reanchor,
            agreement_streak=self._agreement_streak,
            common_visible_count=common_count,
            verifier_disagreement_p50_px=disagreement,
        )
