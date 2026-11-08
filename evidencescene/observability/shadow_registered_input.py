"""Adapt existing registered object inputs for offline shadow tracker replay.

No tracker, object association, live state, publication, or frame estimation is
created here. Inputs must already come from the existing perception owner.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
from typing import Any

import cv2
import numpy as np

from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def prepare_shadow_registered_inputs(
    frames: Sequence[RegisteredBundleTrackInput],
    *,
    object_keys: Sequence[str | int],
    max_sync_skew_ms: float = 20.0,
) -> dict[str, Any]:
    """Convert one same-observation set into registered RGB-D/mask arrays and source receipt.

    Order is supplied by the caller, who remains responsible for identity. The
    adapter rejects absent or empty masks; it cannot decide how to advance a
    tracker when an object is unavailable. ``wire_authorized`` is always false.
    """
    if isinstance(max_sync_skew_ms, bool):
        raise ValueError("max_sync_skew_ms must be finite and positive")
    try:
        skew_ms = float(max_sync_skew_ms)
        max_skew_ns = int(skew_ms * 1_000_000)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_sync_skew_ms must be finite and positive") from None
    if not math.isfinite(skew_ms) or skew_ms <= 0 or max_skew_ns <= 0:
        raise ValueError("max_sync_skew_ms must be finite and positive")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must be a nonempty ordered list")
    if not isinstance(object_keys, (list, tuple)) or len(object_keys) != len(frames):
        raise ValueError("one explicit object key is required per frame")
    if any(not ((type(k) is str and bool(k)) or type(k) is int) for k in object_keys):
        raise ValueError("object keys must be nonempty strings or integers")
    if len({(type(k), k) for k in object_keys}) != len(object_keys):
        raise ValueError("object keys must be unique")
    if any(not isinstance(f, RegisteredBundleTrackInput) for f in frames):
        raise ValueError("existing registered inputs are required")

    first = frames[0]
    masks, evidence = [], []
    common = None
    for key, frame in zip(object_keys, frames):
        stream = frame.stream_key
        if (
            not isinstance(stream, tuple)
            or len(stream) != 8
            or type(stream[6]) is not type(key)
            or stream[6] != key
        ):
            raise ValueError("registered object identity does not match its explicit slot")
        camera = stream[:6] + stream[7:]
        if common is None:
            common = camera
        elif camera != common:
            raise ValueError("camera, source epoch or calibration differs between objects")
        if (
            type(frame.color_timestamp_ns) is not int
            or frame.color_timestamp_ns <= 0
            or type(frame.depth_timestamp_ns) is not int
            or frame.depth_timestamp_ns <= 0
            or frame.color_timestamp_ns != first.color_timestamp_ns
            or frame.depth_timestamp_ns != first.depth_timestamp_ns
            or abs(frame.color_timestamp_ns - frame.depth_timestamp_ns) > max_skew_ns
        ):
            raise ValueError("registered sensor clocks differ or are invalid")
        rgb, depth, mask, k = frame.color_rgb, frame.depth_mm, frame.mask, frame.intrinsics
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or depth.dtype != np.uint16
            or depth.ndim != 2
            or depth.shape != rgb.shape[:2]
            or mask.dtype != np.bool_
            or mask.shape != depth.shape
            or not depth.size
            or k.shape != (3, 3)
            or k.dtype.kind not in "fiu"
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], (0, 0, 1), atol=1e-6, rtol=0)
        ):
            raise ValueError("invalid registered RGB-D, mask or intrinsics")
        if (
            not np.array_equal(rgb, first.color_rgb)
            or not np.array_equal(depth, first.depth_mm)
            or not np.array_equal(k, first.intrinsics)
        ):
            raise ValueError("object slots do not share exact registered RGB-D and intrinsics")
        if (
            not isinstance(frame.source_evidence_hash, str)
            or len(frame.source_evidence_hash) != 64
            or any(c not in "0123456789abcdef" for c in frame.source_evidence_hash)
        ):
            raise ValueError("invalid existing source evidence hash")
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
        if not np.any(eroded):
            raise ValueError("registered object mask is empty after erosion")
        masks.append(eroded)
        evidence.append(
            {
                "object_key": key,
                "stream_key": list(stream),
                "registered_input_sha256": frame.evidence_hash,
                "source_evidence_sha256": frame.source_evidence_hash,
                "eroded_mask_u8_sha256": hashlib.sha256(eroded.tobytes()).hexdigest(),
            }
        )
    depth_m = first.depth_mm.astype(np.float32) / np.float32(1000)
    return {
        "rgb": _readonly_copy(first.color_rgb),
        "depth_m": _readonly_copy(depth_m),
        "masks": _readonly_copy(np.stack(masks)[:, None]),
        "intrinsics": _readonly_copy(first.intrinsics),
        "timestamp_seconds": first.color_timestamp_ns / 1e9,
        "receipt": {
            "color_timestamp_ns": str(first.color_timestamp_ns),
            "depth_timestamp_ns": str(first.depth_timestamp_ns),
            "objects": evidence,
            "wire_authorized": False,
        },
    }
