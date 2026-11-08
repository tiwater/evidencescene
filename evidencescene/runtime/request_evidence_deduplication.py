"""Request-scoped evidence deduplication and exact run-interval RLE arithmetic.

This module is an opt-in, fail-closed adapter over the existing perception and
world-model owners:
- ``FoundationMaskCacheDetector`` and ``_position_detections_for_observation``
  in ``ticos_vision.perception.detector``
- ``register_rgbd_center_frame``, ``RegisteredRgbdFrame.select_surface``,
  ``extract_registered_rgbd_mask_surface_sparse``, and
  ``select_mask_centroid_depth_component`` in ``ticos_vision.geometry.rgbd_surface``
- ``observe_coplanar_rgbd_tag_corners`` and ``_fit_shared_plane`` in
  ``ticos_vision.estimation.coplanar_rgbd_tag_observations``
- ``update_teleops_world_model_from_observations`` in ``ticos_vision.inference.run_live``
- ``observation_digest_scope`` in ``ticos_vision.runtime.observation_computation_cache``

No new detector, tracker, world model, coordinate frame, or calibration path is
introduced. All cached values are scoped strictly to a single synchronous
request on the originating thread and are cleared immediately upon request exit.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import functools
import gc
import inspect
import math
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Callable, Iterator

import numpy as np

from evidencescene.primitives.geometry import (
    CameraInfoRecord,
    MaskDepthComponentSelection,
    RegisteredRgbdFrame,
    RegisteredRgbdSurface,
    observation_digest_scope,
)

_VALID_MODES = frozenset({"off", "control", "dedup", "validate"})
_ACTIVE_SCOPE: ContextVar["_ActiveRequestScope | None"] = ContextVar(
    "request_evidence_deduplication_scope",
    default=None,
)
_RANSAC_TRIPLET_CACHE: dict[tuple[int, int], np.ndarray] = {}
_RANSAC_TRIPLET_LOCK = Lock()


@dataclass
class RequestDeduplicationStats:
    """Diagnostic counters for a single request or cumulative session."""

    mode: str = "off"
    requests_entered: int = 0
    rle_box_centroid_fast_calls: int = 0
    rle_box_centroid_cache_hits: int = 0
    rle_array_fast_calls: int = 0
    rle_decode_cache_hits: int = 0
    rle_decode_cache_misses: int = 0
    record_cache_hits: int = 0
    record_cache_misses: int = 0
    depth_decode_cache_hits: int = 0
    depth_decode_cache_misses: int = 0
    registered_frame_recorded: int = 0
    registered_surface_reuses: int = 0
    registered_surface_fallbacks: int = 0
    centroid_component_cache_hits: int = 0
    centroid_component_cache_misses: int = 0
    coplanar_plane_fast_calls: int = 0
    gc_gen0_during_requests: int = 0
    gc_gen1_during_requests: int = 0
    gc_gen2_during_requests: int = 0
    validation_checks: int = 0
    validation_mismatches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requests_entered": self.requests_entered,
            "rle_box_centroid_fast_calls": self.rle_box_centroid_fast_calls,
            "rle_box_centroid_cache_hits": self.rle_box_centroid_cache_hits,
            "rle_array_fast_calls": self.rle_array_fast_calls,
            "rle_decode_cache_hits": self.rle_decode_cache_hits,
            "rle_decode_cache_misses": self.rle_decode_cache_misses,
            "record_cache_hits": self.record_cache_hits,
            "record_cache_misses": self.record_cache_misses,
            "depth_decode_cache_hits": self.depth_decode_cache_hits,
            "depth_decode_cache_misses": self.depth_decode_cache_misses,
            "registered_frame_recorded": self.registered_frame_recorded,
            "registered_surface_reuses": self.registered_surface_reuses,
            "registered_surface_fallbacks": self.registered_surface_fallbacks,
            "centroid_component_cache_hits": self.centroid_component_cache_hits,
            "centroid_component_cache_misses": self.centroid_component_cache_misses,
            "coplanar_plane_fast_calls": self.coplanar_plane_fast_calls,
            "gc_gen0_during_requests": self.gc_gen0_during_requests,
            "gc_gen1_during_requests": self.gc_gen1_during_requests,
            "gc_gen2_during_requests": self.gc_gen2_during_requests,
            "validation_checks": self.validation_checks,
            "validation_mismatches": self.validation_mismatches,
        }


def rle_mask_box_and_centroid(
    mask_rle: object,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], tuple[float, float]] | None:
    """Compute exact (min_x, min_y, max_x, max_y) and (mean_x, mean_y) in O(runs).

    Uses arbitrary-precision Python integer arithmetic over row-major runs so
    both the bounding box and floating-point centroid division are bit-for-bit
    identical to pixel-by-pixel enumeration.
    """
    if not isinstance(mask_rle, dict):
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and int(size[0]) == height
        and int(size[1]) == width
        and isinstance(counts, list)
        and len(counts) % 2 == 0
    ):
        return None
    if width <= 0 or height <= 0:
        return None
    total_limit = width * height
    total_pixels = 0
    sum_y = 0
    sum_index = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for start, length in zip(counts[0::2], counts[1::2], strict=False):
        if not isinstance(start, int) or not isinstance(length, int) or length <= 0:
            return None
        end = start + length - 1
        if start < 0 or end >= total_limit:
            return None
        y0, x0 = divmod(start, width)
        y1, x1 = divmod(end, width)
        if y0 < min_y:
            min_y = y0
        if y1 > max_y:
            max_y = y1
        if y0 == y1:
            if x0 < min_x:
                min_x = x0
            if x1 > max_x:
                max_x = x1
            sum_y += length * y0
        else:
            min_x = 0
            max_x = width - 1
            n0 = width - x0
            n1 = x1 + 1
            m = y1 - y0 - 1
            sum_y += n0 * y0 + width * m * (y0 + y1) // 2 + n1 * y1
        sum_index += length * start + length * (length - 1) // 2
        total_pixels += length
    if total_pixels == 0:
        return None
    sum_x = sum_index - width * sum_y
    return (
        (min_x, min_y, max_x, max_y),
        (float(sum_x / total_pixels), float(sum_y / total_pixels)),
    )


def rle_mask_box(
    mask_rle: object,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Compute exact (min_x, min_y, max_x, max_y) in O(runs)."""
    result = rle_mask_box_and_centroid(mask_rle, width=width, height=height)
    return result[0] if result is not None else None


def rle_mask_centroid(
    mask_rle: object,
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Compute exact (mean_x, mean_y) in O(runs)."""
    result = rle_mask_box_and_centroid(mask_rle, width=width, height=height)
    return result[1] if result is not None else None


def rle_mask_array(
    mask_rle: object,
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Decode row-major RLE into a boolean array via flat slice assignment."""
    if not isinstance(mask_rle, dict):
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and int(size[0]) == int(height)
        and int(size[1]) == int(width)
        and isinstance(counts, list)
        and len(counts) % 2 == 0
    ):
        return None
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        return None
    total_limit = w * h
    mask = np.zeros((h, w), dtype=bool)
    flat = mask.reshape(-1)
    for start, length in zip(counts[0::2], counts[1::2], strict=False):
        if not isinstance(start, int) or not isinstance(length, int) or length <= 0:
            return None
        end = start + length
        if start < 0 or end > total_limit:
            return None
        flat[start:end] = True
    return mask


def _ransac_triplets(n: int, iterations: int) -> np.ndarray:
    key = (n, iterations)
    with _RANSAC_TRIPLET_LOCK:
        cached = _RANSAC_TRIPLET_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(0)
    triplets = np.empty((iterations, 3), dtype=np.intp)
    for idx in range(iterations):
        triplets[idx] = rng.choice(n, size=3, replace=False)
    triplets.setflags(write=False)
    with _RANSAC_TRIPLET_LOCK:
        if len(_RANSAC_TRIPLET_CACHE) >= 64:
            _RANSAC_TRIPLET_CACHE.clear()
        _RANSAC_TRIPLET_CACHE[key] = triplets
    return triplets


def fast_fit_shared_plane(
    points: np.ndarray,
    *,
    ransac_iterations: int = 240,
    plane_distance_threshold_m: float = 0.004,
    minimum_plane_points: int = 80,
) -> tuple[np.ndarray, float, int, float] | None:
    """Bit-exact vectorized RANSAC plane fit matching _fit_shared_plane."""
    n = len(points)
    if n < max(3, minimum_plane_points):
        return None
    triplets = _ransac_triplets(n, ransac_iterations)
    sampled = points[triplets]
    firsts = sampled[:, 0, :]
    normals = np.cross(sampled[:, 1, :] - firsts, sampled[:, 2, :] - firsts)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return None
    valid_indices = np.flatnonzero(valid)
    valid_normals = normals[valid_indices] / norms[valid_indices, None]
    valid_firsts = firsts[valid_indices]
    offsets = np.einsum("ij,ij->i", valid_firsts, valid_normals)
    thresh = float(plane_distance_threshold_m)

    # Exact float64 BLAS projection screening + exact (points - first) @ normal
    # verification of contenders.
    proj = points @ valid_normals.T
    tol_hi = thresh + 1e-12
    upper_counts = np.count_nonzero(
        (proj >= (offsets[None, :] - tol_hi))
        & (proj <= (offsets[None, :] + tol_hi)),
        axis=0,
    )
    if int(np.max(upper_counts)) < minimum_plane_points:
        return None

    order = np.argsort(-upper_counts, kind="stable")
    best_count = 0
    best_col_idx = len(valid_indices)
    best: np.ndarray | None = None
    for col in order:
        col_idx = int(col)
        upper = int(upper_counts[col_idx])
        if upper < best_count:
            break
        if upper == best_count and col_idx >= best_col_idx:
            continue
        inliers = (
            np.abs((points - valid_firsts[col_idx]) @ valid_normals[col_idx]) <= thresh
        )
        c = int(np.count_nonzero(inliers))
        if c > best_count or (c == best_count and col_idx < best_col_idx):
            best_count = c
            best_col_idx = col_idx
            best = inliers

    count = int(best_count)
    if count < minimum_plane_points or best is None:
        return None
    inliers = points[best]
    center = np.mean(inliers, axis=0)
    _, _, right_transpose = np.linalg.svd(inliers - center, full_matrices=False)
    normal = right_transpose[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ center)
    residuals = np.abs(inliers @ normal + offset)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    if not math.isfinite(rmse) or rmse > thresh:
        return None
    return normal, offset, count, rmse


def _rle_cache_key(
    mask_rle: object,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int, tuple[int, ...]] | None:
    if not isinstance(mask_rle, dict):
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and isinstance(counts, list)
        and len(counts) % 2 == 0
    ):
        return None
    try:
        s0 = int(size[0])
        s1 = int(size[1])
        counts_tuple = tuple(counts)
    except (TypeError, ValueError):
        return None
    if any(not isinstance(v, int) for v in counts_tuple):
        return None
    return (int(width), int(height), s0, s1, counts_tuple)


def _registered_frame_key(
    *,
    depth_mm: np.ndarray,
    depth_info: CameraInfoRecord | None,
    color_info: CameraInfoRecord | None,
    color_from_depth: np.ndarray | None,
    registration_status: str,
    color_timestamp_ns: int | None,
    depth_timestamp_ns: int | None,
    registration_max_reprojection_error_px: float | None,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[Any, ...] | None:
    if depth_info is None or color_info is None or color_from_depth is None:
        return None
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if transform.shape != (4, 4):
        return None
    return (
        id(depth_mm),
        depth_mm.shape,
        depth_info,
        color_info,
        transform.tobytes(),
        str(registration_status),
        color_timestamp_ns,
        depth_timestamp_ns,
        registration_max_reprojection_error_px,
        float(min_depth_m),
        float(max_depth_m),
    )


class _ActiveRequestScope:
    """Thread-local request cache active for one synchronous observation update."""

    def __init__(self, *, mode: str, stats: RequestDeduplicationStats) -> None:
        self.thread_id = get_ident()
        self.mode = mode
        self.stats = stats
        self.active = True
        self.box_centroid_cache: dict[
            tuple[int, int, int, int, tuple[int, ...]],
            tuple[tuple[int, int, int, int], tuple[float, float]] | None,
        ] = {}
        self.rle_array_cache: dict[
            tuple[int, int, int, int, tuple[int, ...]],
            np.ndarray | None,
        ] = {}
        self.record_cache: dict[str, dict[str, Any] | None] = {}
        self.depth_cache: dict[str, Any] = {}
        self.registered_frames: list[
            tuple[tuple[Any, ...], np.ndarray, RegisteredRgbdFrame]
        ] = []
        self.centroid_component_cache: list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[int, int] | None,
                tuple[Any, ...],
                MaskDepthComponentSelection,
            ]
        ] = []

    def clear(self) -> None:
        self.active = False
        self.box_centroid_cache.clear()
        self.rle_array_cache.clear()
        self.record_cache.clear()
        self.depth_cache.clear()
        self.registered_frames.clear()
        self.centroid_component_cache.clear()


def _current_scope() -> _ActiveRequestScope | None:
    scope = _ACTIVE_SCOPE.get()
    if scope is None or not scope.active or scope.thread_id != get_ident():
        return None
    if scope.mode not in {"dedup", "validate"}:
        return None
    return scope


@contextmanager
def request_evidence_deduplication_scope(
    *,
    mode: str = "dedup",
    stats: RequestDeduplicationStats | None = None,
) -> Iterator[RequestDeduplicationStats]:
    """Activate request-scoped evidence deduplication for one synchronous request."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported request deduplication mode: {mode!r}")
    target_stats = stats if stats is not None else RequestDeduplicationStats(mode=mode)
    target_stats.mode = mode
    target_stats.requests_entered += 1
    scope = _ActiveRequestScope(mode=mode, stats=target_stats)
    token = _ACTIVE_SCOPE.set(scope)
    digest_enabled = mode in {"dedup", "validate"}
    try:
        with observation_digest_scope(enabled=digest_enabled, capacity=4):
            yield target_stats
    finally:
        scope.clear()
        _ACTIVE_SCOPE.reset(token)


def deduplicated_read_record(
    path: Path,
    fallback: Callable[[Path], dict[str, Any] | None],
) -> dict[str, Any] | None:
    scope = _current_scope()
    if scope is None:
        return fallback(path)
    key = str(path)
    if key in scope.record_cache:
        scope.stats.record_cache_hits += 1
        cached = scope.record_cache[key]
        if scope.mode == "validate":
            expected = fallback(path)
            scope.stats.validation_checks += 1
            if cached != expected:
                scope.stats.validation_mismatches += 1
                raise AssertionError(f"Foundation mask record cache mismatch for {path}")
        return cached
    scope.stats.record_cache_misses += 1
    record = fallback(path)
    scope.record_cache[key] = record
    return record


def deduplicated_mask_box(
    detector: Any,
    mask_rle: object,
    *,
    width: int,
    height: int,
    fallback: Callable[..., tuple[int, int, int, int] | None],
) -> tuple[int, int, int, int] | None:
    scope = _current_scope()
    if scope is None:
        return fallback(detector, mask_rle, width=width, height=height)
    key = _rle_cache_key(mask_rle, width=width, height=height)
    if key is not None and key in scope.box_centroid_cache:
        scope.stats.rle_box_centroid_cache_hits += 1
        cached = scope.box_centroid_cache[key]
        box = cached[0] if cached is not None else None
    else:
        scope.stats.rle_box_centroid_fast_calls += 1
        computed = rle_mask_box_and_centroid(mask_rle, width=width, height=height)
        if key is not None:
            scope.box_centroid_cache[key] = computed
        box = computed[0] if computed is not None else None
    if scope.mode == "validate":
        expected = fallback(detector, mask_rle, width=width, height=height)
        scope.stats.validation_checks += 1
        if box != expected:
            scope.stats.validation_mismatches += 1
            raise AssertionError(f"rle_mask_box mismatch: {box} != {expected}")
    return box


def deduplicated_mask_centroid(
    detector: Any,
    mask_rle: object,
    *,
    width: int,
    height: int,
    fallback: Callable[..., tuple[float, float] | None],
) -> tuple[float, float] | None:
    scope = _current_scope()
    if scope is None:
        return fallback(detector, mask_rle, width=width, height=height)
    key = _rle_cache_key(mask_rle, width=width, height=height)
    if key is not None and key in scope.box_centroid_cache:
        scope.stats.rle_box_centroid_cache_hits += 1
        cached = scope.box_centroid_cache[key]
        centroid = cached[1] if cached is not None else None
    else:
        scope.stats.rle_box_centroid_fast_calls += 1
        computed = rle_mask_box_and_centroid(mask_rle, width=width, height=height)
        if key is not None:
            scope.box_centroid_cache[key] = computed
        centroid = computed[1] if computed is not None else None
    if scope.mode == "validate":
        expected = fallback(detector, mask_rle, width=width, height=height)
        scope.stats.validation_checks += 1
        if centroid != expected:
            scope.stats.validation_mismatches += 1
            raise AssertionError(f"rle_mask_centroid mismatch: {centroid} != {expected}")
    return centroid


def deduplicated_decode_mask_rle(
    payload: Any,
    *,
    width: int,
    height: int,
    fallback: Callable[..., np.ndarray | None],
) -> np.ndarray | None:
    scope = _current_scope()
    if scope is None:
        return fallback(payload, width=width, height=height)
    key = _rle_cache_key(payload, width=width, height=height)
    if key is not None and key in scope.rle_array_cache:
        scope.stats.rle_decode_cache_hits += 1
        cached = scope.rle_array_cache[key]
    else:
        scope.stats.rle_decode_cache_misses += 1
        scope.stats.rle_array_fast_calls += 1
        cached = rle_mask_array(payload, width=width, height=height)
        if cached is not None:
            cached.setflags(write=False)
        if key is not None:
            scope.rle_array_cache[key] = cached
    if scope.mode == "validate":
        expected = fallback(payload, width=width, height=height)
        scope.stats.validation_checks += 1
        equal = (
            (cached is None and expected is None)
            or (
                cached is not None
                and expected is not None
                and np.array_equal(cached, expected)
            )
        )
        if not equal:
            scope.stats.validation_mismatches += 1
            raise AssertionError("decode_mask_rle mismatch against baseline")
    return cached


def deduplicated_mask_array(
    detector: Any,
    mask_rle: object,
    *,
    width: int,
    height: int,
    fallback: Callable[..., np.ndarray | None],
) -> np.ndarray | None:
    return deduplicated_decode_mask_rle(
        mask_rle,
        width=width,
        height=height,
        fallback=lambda payload, *, width, height: fallback(
            detector, payload, width=width, height=height
        ),
    )


def deduplicated_decode_compressed_depth(
    payload_base64: str,
    fallback: Callable[[str], Any],
) -> Any:
    scope = _current_scope()
    if scope is None or not isinstance(payload_base64, str):
        return fallback(payload_base64)
    if payload_base64 in scope.depth_cache:
        scope.stats.depth_decode_cache_hits += 1
        cached = scope.depth_cache[payload_base64]
        if scope.mode == "validate":
            expected = fallback(payload_base64)
            scope.stats.validation_checks += 1
            if not np.array_equal(cached.values_mm, expected.values_mm):
                scope.stats.validation_mismatches += 1
                raise AssertionError("decode_compressed_depth cache mismatch")
        return cached
    scope.stats.depth_decode_cache_misses += 1
    decoded = fallback(payload_base64)
    if hasattr(decoded, "values_mm") and isinstance(decoded.values_mm, np.ndarray):
        decoded.values_mm.setflags(write=False)
    if len(scope.depth_cache) < 8:
        scope.depth_cache[payload_base64] = decoded
    return decoded


def record_registered_rgbd_center_frame(
    *args: Any,
    fallback: Callable[..., RegisteredRgbdFrame],
    **kwargs: Any,
) -> RegisteredRgbdFrame:
    frame = fallback(*args, **kwargs)
    scope = _current_scope()
    if scope is not None and frame.ready:
        depth_mm = kwargs.get("depth_mm")
        depth_info = kwargs.get("depth_info")
        color_info = kwargs.get("color_info")
        color_from_depth = kwargs.get("color_from_depth")
        registration_status = kwargs.get("registration_status", "validated")
        color_timestamp_ns = kwargs.get("color_timestamp_ns")
        depth_timestamp_ns = kwargs.get("depth_timestamp_ns")
        registration_max_reprojection_error_px = kwargs.get(
            "registration_max_reprojection_error_px"
        )
        min_depth_m = kwargs.get("min_depth_m", 0.08)
        max_depth_m = kwargs.get("max_depth_m", 2.50)
        if depth_mm is not None:
            depth_arr = np.asarray(depth_mm)
            transform = (
                np.asarray(color_from_depth, dtype=np.float64)
                if color_from_depth is not None
                else None
            )
            key = _registered_frame_key(
                depth_mm=depth_arr,
                depth_info=depth_info,
                color_info=color_info,
                color_from_depth=transform,
                registration_status=registration_status,
                color_timestamp_ns=color_timestamp_ns,
                depth_timestamp_ns=depth_timestamp_ns,
                registration_max_reprojection_error_px=registration_max_reprojection_error_px,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            if key is not None:
                scope.registered_frames.append((key, depth_arr, frame))
                scope.stats.registered_frame_recorded += 1
    return frame


def deduplicated_extract_registered_rgbd_mask_surface_sparse(
    *args: Any,
    fallback: Callable[..., RegisteredRgbdSurface],
    **kwargs: Any,
) -> RegisteredRgbdSurface:
    scope = _current_scope()
    roi_margin_px = int(kwargs.get("roi_margin_px", 8))
    max_sync_skew_ms = float(kwargs.get("max_sync_skew_ms", 20.0))
    depth_mm = kwargs.get("depth_mm")
    color_mask = kwargs.get("color_mask")
    min_points = int(kwargs.get("min_points", 50))
    if (
        scope is not None
        and depth_mm is not None
        and color_mask is not None
        and roi_margin_px >= 8
        and max_sync_skew_ms == 20.0
    ):
        depth_arr = np.asarray(depth_mm)
        color_from_depth = kwargs.get("color_from_depth")
        transform = (
            np.asarray(color_from_depth, dtype=np.float64)
            if color_from_depth is not None
            else None
        )
        key = _registered_frame_key(
            depth_mm=depth_arr,
            depth_info=kwargs.get("depth_info"),
            color_info=kwargs.get("color_info"),
            color_from_depth=transform,
            registration_status=kwargs.get("registration_status", "validated"),
            color_timestamp_ns=kwargs.get("color_timestamp_ns"),
            depth_timestamp_ns=kwargs.get("depth_timestamp_ns"),
            registration_max_reprojection_error_px=kwargs.get(
                "registration_max_reprojection_error_px"
            ),
            min_depth_m=kwargs.get("min_depth_m", 0.08),
            max_depth_m=kwargs.get("max_depth_m", 2.50),
        )
        if key is not None:
            matched_frame: RegisteredRgbdFrame | None = None
            for saved_key, saved_depth, saved_frame in scope.registered_frames:
                if saved_key == key and saved_depth is depth_arr:
                    matched_frame = saved_frame
                    break
            if matched_frame is not None and matched_frame.ready:
                mask = np.asarray(color_mask, dtype=bool)
                if mask.shape == (
                    matched_frame.color_image_size[1],
                    matched_frame.color_image_size[0],
                ) and bool(np.any(mask)):
                    selected = matched_frame.select_surface(mask, min_points=min_points)
                    if selected.ready:
                        reused = replace(
                            selected,
                            source="validated_color_depth_sparse_center_registration",
                        )
                        scope.stats.registered_surface_reuses += 1
                        if scope.mode == "validate":
                            expected = fallback(*args, **kwargs)
                            scope.stats.validation_checks += 1
                            exact = (
                                reused.ready == expected.ready
                                and reused.reason == expected.reason
                                and reused.source == expected.source
                                and np.array_equal(
                                    reused.points_depth_m, expected.points_depth_m
                                )
                                and np.array_equal(
                                    reused.depth_pixels_uv, expected.depth_pixels_uv
                                )
                                and np.array_equal(
                                    reused.color_pixels_uv, expected.color_pixels_uv
                                )
                                and np.array_equal(
                                    reused.color_depth_m, expected.color_depth_m
                                )
                            )
                            if not exact:
                                scope.stats.validation_mismatches += 1
                                raise AssertionError(
                                    "RegisteredRgbdFrame.select_surface differed from sparse extraction"
                                )
                        return reused
        scope.stats.registered_surface_fallbacks += 1
    return fallback(*args, **kwargs)


def deduplicated_select_mask_centroid_depth_component(
    surface: RegisteredRgbdSurface,
    *args: Any,
    fallback: Callable[..., MaskDepthComponentSelection],
    **kwargs: Any,
) -> MaskDepthComponentSelection:
    scope = _current_scope()
    color_mask = kwargs.get("color_mask")
    depth_image_shape = kwargs.get("depth_image_shape")
    if (
        scope is not None
        and not args
        and isinstance(surface, RegisteredRgbdSurface)
        and surface.ready
        and color_mask is not None
    ):
        pts = surface.points_depth_m
        dpix = surface.depth_pixels_uv
        cpix = surface.color_pixels_uv
        mask_arr = np.asarray(color_mask, dtype=bool)
        shape_key = (
            (int(depth_image_shape[0]), int(depth_image_shape[1]))
            if depth_image_shape is not None
            else None
        )
        params_key = (
            surface.color_image_size,
            int(kwargs.get("min_points", 50)),
            float(kwargs.get("local_depth_tolerance_floor_m", 0.005)),
            float(kwargs.get("maximum_anchor_distance_fraction", 0.25)),
        )
        for (
            saved_pts,
            saved_dpix,
            saved_cpix,
            saved_mask,
            saved_shape,
            saved_params,
            saved_sel,
        ) in scope.centroid_component_cache:
            if (
                saved_shape == shape_key
                and saved_params == params_key
                and (saved_mask is mask_arr or np.array_equal(saved_mask, mask_arr))
                and (saved_pts is pts or np.array_equal(saved_pts, pts))
                and (saved_dpix is dpix or np.array_equal(saved_dpix, dpix))
                and (saved_cpix is cpix or np.array_equal(saved_cpix, cpix))
            ):
                scope.stats.centroid_component_cache_hits += 1
                if scope.mode == "validate":
                    expected = fallback(surface, *args, **kwargs)
                    scope.stats.validation_checks += 1
                    exact = (
                        saved_sel.ready == expected.ready
                        and saved_sel.reason == expected.reason
                        and saved_sel.support_point_count == expected.support_point_count
                        and saved_sel.local_window_point_count
                        == expected.local_window_point_count
                        and (
                            (
                                saved_sel.point_depth_m is None
                                and expected.point_depth_m is None
                            )
                            or (
                                saved_sel.point_depth_m is not None
                                and expected.point_depth_m is not None
                                and np.array_equal(
                                    saved_sel.point_depth_m, expected.point_depth_m
                                )
                            )
                        )
                    )
                    if not exact:
                        scope.stats.validation_mismatches += 1
                        raise AssertionError(
                            "select_mask_centroid_depth_component cache mismatch"
                        )
                return saved_sel
        scope.stats.centroid_component_cache_misses += 1
        result = fallback(surface, *args, **kwargs)
        scope.centroid_component_cache.append(
            (pts, dpix, cpix, mask_arr, shape_key, params_key, result)
        )
        return result
    return fallback(surface, *args, **kwargs)


def deduplicated_fit_shared_plane(
    points: np.ndarray,
    fallback: Callable[[np.ndarray], tuple[np.ndarray, float, int, float] | None],
) -> tuple[np.ndarray, float, int, float] | None:
    scope = _current_scope()
    if scope is None:
        return fallback(points)
    scope.stats.coplanar_plane_fast_calls += 1
    result = fast_fit_shared_plane(points)
    if scope.mode == "validate":
        expected = fallback(points)
        scope.stats.validation_checks += 1
        exact = (result is None and expected is None) or (
            result is not None
            and expected is not None
            and np.array_equal(result[0], expected[0])
            and result[1] == expected[1]
            and result[2] == expected[2]
            and result[3] == expected[3]
        )
        if not exact:
            scope.stats.validation_mismatches += 1
            raise AssertionError("fast_fit_shared_plane mismatch against _fit_shared_plane")
    return result


@contextmanager
def install_request_evidence_deduplication(
    *,
    mode: str = "dedup",
    manage_cyclic_gc: bool = True,
) -> Iterator[RequestDeduplicationStats]:
    """Install request-scoped evidence deduplication hooks on standalone owners."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported request deduplication mode: {mode!r}")

    from evidencescene.primitives import geometry as geom_mod
    from evidencescene.primitives import image_mask as image_mask_mod
    from evidencescene.runtime import coplanar_rgbd_tag_observations as coplanar_mod

    stats = RequestDeduplicationStats(mode=mode)

    orig_image_decode_rle = image_mask_mod.decode_mask_rle
    orig_depth_decode = getattr(geom_mod, "decode_compressed_depth", None)
    orig_surface_register = getattr(geom_mod, "register_rgbd_center_frame", None)
    orig_surface_sparse = getattr(
        geom_mod, "extract_registered_rgbd_mask_surface_sparse", None
    )
    orig_surface_centroid = getattr(
        geom_mod, "select_mask_centroid_depth_component", None
    )
    orig_fit_shared_plane = coplanar_mod._fit_shared_plane

    def _gc_cb(phase: str, info: dict[str, Any]) -> None:
        if phase == "start" and _ACTIVE_SCOPE.get() is not None:
            gen = int(info.get("generation", -1))
            if gen == 0:
                stats.gc_gen0_during_requests += 1
            elif gen == 1:
                stats.gc_gen1_during_requests += 1
            elif gen == 2:
                stats.gc_gen2_during_requests += 1

    gc.callbacks.append(_gc_cb)
    gc_was_enabled = gc.isenabled()
    if manage_cyclic_gc and mode in {"dedup", "validate"}:
        gc.collect()
        gc.disable()

    def wrapped_decode_rle(
        payload: Any, *, width: int, height: int
    ) -> np.ndarray | None:
        return deduplicated_decode_mask_rle(
            payload, width=width, height=height, fallback=orig_image_decode_rle
        )

    def wrapped_fit_plane(
        points: np.ndarray,
    ) -> tuple[np.ndarray, float, int, float] | None:
        return deduplicated_fit_shared_plane(points, orig_fit_shared_plane)

    image_mask_mod.decode_mask_rle = wrapped_decode_rle
    coplanar_mod._fit_shared_plane = wrapped_fit_plane

    if orig_depth_decode is not None:
        geom_mod.decode_compressed_depth = lambda payload_base64: (  # type: ignore[attr-defined]
            deduplicated_decode_compressed_depth(payload_base64, orig_depth_decode)
        )
    if orig_surface_register is not None:
        geom_mod.register_rgbd_center_frame = lambda *args, **kwargs: (  # type: ignore[attr-defined]
            record_registered_rgbd_center_frame(
                *args, fallback=orig_surface_register, **kwargs
            )
        )
    if orig_surface_sparse is not None:
        geom_mod.extract_registered_rgbd_mask_surface_sparse = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: deduplicated_extract_registered_rgbd_mask_surface_sparse(
                *args, fallback=orig_surface_sparse, **kwargs
            )
        )
    if orig_surface_centroid is not None:
        geom_mod.select_mask_centroid_depth_component = (  # type: ignore[attr-defined]
            lambda surface, *args, **kwargs: deduplicated_select_mask_centroid_depth_component(
                surface, *args, fallback=orig_surface_centroid, **kwargs
            )
        )

    try:
        with request_evidence_deduplication_scope(mode=mode, stats=stats):
            yield stats
    finally:
        image_mask_mod.decode_mask_rle = orig_image_decode_rle
        coplanar_mod._fit_shared_plane = orig_fit_shared_plane
        if orig_depth_decode is not None:
            geom_mod.decode_compressed_depth = orig_depth_decode  # type: ignore[attr-defined]
        if orig_surface_register is not None:
            geom_mod.register_rgbd_center_frame = orig_surface_register  # type: ignore[attr-defined]
        if orig_surface_sparse is not None:
            geom_mod.extract_registered_rgbd_mask_surface_sparse = orig_surface_sparse  # type: ignore[attr-defined]
        if orig_surface_centroid is not None:
            geom_mod.select_mask_centroid_depth_component = orig_surface_centroid  # type: ignore[attr-defined]

        if _gc_cb in gc.callbacks:
            gc.callbacks.remove(_gc_cb)
        if manage_cyclic_gc and mode in {"dedup", "validate"}:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
