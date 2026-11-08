"""Serialize a narrowly evidenced SAM2 no-mask observation for shadow scoring.

SAM2ObservationSession and the registered RGB-D producer remain owners;
this stateless adapter neither runs a tracker nor creates/publishes geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from evidencescene.observability.shadow_pose_owner_ab import _registered_input_identity
from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput


def make_missing_mask_shadow_tick(
    *,
    frame_index: int,
    pre_sam_registered_input: RegisteredBundleTrackInput,
    sam_diagnostic: Mapping[str, Any],
    sam_visual_mask_evidence: object,
    max_sync_skew_ms: float = 20.0,
) -> dict[str, Any]:
    """Return a no-pose tick only for an explicit same-tick SAM2 no-mask result.

    This does not handle depth-insufficient or other ambiguous failures. The
    caller is responsible for providing the owner's real diagnostic; the tick
    does not authenticate producer identity, model execution, or freshness.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if isinstance(max_sync_skew_ms, bool):
        raise ValueError("max_sync_skew_ms must be finite and positive")
    try:
        skew_ms = float(max_sync_skew_ms)
        max_skew_ns = int(skew_ms * 1_000_000)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_sync_skew_ms must be finite and positive") from None
    if not math.isfinite(skew_ms) or skew_ms <= 0 or max_skew_ns <= 0:
        raise ValueError("max_sync_skew_ms must be finite and positive")
    if not isinstance(pre_sam_registered_input, RegisteredBundleTrackInput):
        raise ValueError("existing registered input required")
    frame = pre_sam_registered_input
    if not isinstance(frame.stream_key, tuple) or len(frame.stream_key) != 8:
        raise ValueError("invalid registered stream key")
    tick = {
        "frame_index": frame_index,
        "stream_key": list(frame.stream_key),
        "object_key": frame.stream_key[6],
        "color_timestamp_ns": frame.color_timestamp_ns,
        "depth_timestamp_ns": frame.depth_timestamp_ns,
        "input_evidence_hash": frame.evidence_hash,
    }
    identity = _registered_input_identity(tick)
    if (
        identity is None
        or type(frame.color_timestamp_ns) is not int
        or type(frame.depth_timestamp_ns) is not int
        or min(frame.color_timestamp_ns, frame.depth_timestamp_ns) <= 0
        or abs(frame.color_timestamp_ns - frame.depth_timestamp_ns) > max_skew_ns
    ):
        raise ValueError("invalid or unsynchronized registered identity")
    if (
        not isinstance(sam_diagnostic, Mapping)
        or sam_diagnostic.get("reason") != "sam_mask_unavailable"
        or type(sam_diagnostic.get("depth_supported_pixels")) is not int
        or sam_diagnostic["depth_supported_pixels"] != 0
        or type(sam_diagnostic.get("mask_pixel_count")) is not int
        or sam_diagnostic["mask_pixel_count"] != 0
        or sam_diagnostic.get("source_color_timestamp_ns") != str(frame.color_timestamp_ns)
        or sam_diagnostic.get("source_depth_timestamp_ns") != str(frame.depth_timestamp_ns)
        or sam_visual_mask_evidence is not None
    ):
        raise ValueError("same-tick unambiguous SAM2 no-mask evidence required")
    tick.update(
        status="not_observable",
        reason="sam_mask_unavailable",
        mask_available=False,
        source_mask_evidence_sha256=None,
        mask_diagnostic={
            "reason": "sam_mask_unavailable",
            "depth_supported_pixels": 0,
            "mask_pixel_count": 0,
            "source_color_timestamp_ns": str(frame.color_timestamp_ns),
            "source_depth_timestamp_ns": str(frame.depth_timestamp_ns),
        },
        camera_from_object=None,
        wire_authorized=False,
    )
    return tick
