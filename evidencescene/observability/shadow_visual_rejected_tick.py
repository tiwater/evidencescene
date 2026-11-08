"""Score an existing SAM2 visual mask whose registered depth was rejected.

This stateless serializer does not advance SAM2, estimate a pose, rectify a
mask or publish world/assist state. The caller-supplied owner evidence is not
independent authentication of source, tracker execution, or freshness.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from evidencescene.observability.shadow_pose_owner_ab import _registered_input_identity
from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput
from evidencescene.primitives.image_mask import decode_mask_rle, encode_mask_rle
from evidencescene.primitives.sam2_observation_session import _mask_evidence_sha256


def make_visual_rejected_shadow_tick(
    *,
    frame_index: int,
    pre_sam_registered_input: RegisteredBundleTrackInput,
    sam_diagnostic: Mapping[str, Any],
    sam_visual_mask_evidence: Mapping[str, Any],
    max_sync_skew_ms: float = 20.0,
) -> dict[str, Any]:
    """Report a present visual mask with no pose after a specific depth reject.

    Only the two named owner rejection reasons are supported. No input here
    certifies that the caller used the producer's actual outputs, and no
    independent scoreboard blocker is satisfied by this serialization.
    """
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if isinstance(max_sync_skew_ms, (bool, np.bool_)):
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
    color, depth, mask, intrinsics = frame.color_rgb, frame.depth_mm, frame.mask, frame.intrinsics
    if (
        not isinstance(color, np.ndarray)
        or color.dtype != np.uint8
        or color.ndim != 3
        or color.shape[2] != 3
        or not isinstance(depth, np.ndarray)
        or depth.dtype != np.uint16
        or depth.ndim != 2
        or color.shape[:2] != depth.shape
        or not depth.size
        or not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != depth.shape
        or not isinstance(intrinsics, np.ndarray)
        or intrinsics.shape != (3, 3)
        or intrinsics.dtype.kind not in "fiu"
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], (0, 0, 1), atol=1e-6, rtol=0)
    ):
        raise ValueError("registered RGB-D or intrinsics are invalid")
    key = frame.stream_key
    if not isinstance(key, tuple) or len(key) != 8:
        raise ValueError("invalid registered stream key")
    tick = {
        "frame_index": frame_index,
        "stream_key": list(key),
        "object_key": key[6],
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
    if not isinstance(sam_diagnostic, Mapping):
        raise ValueError("same-tick SAM2 depth rejection required")
    reason = sam_diagnostic.get("reason")
    support, pixels = sam_diagnostic.get("depth_supported_pixels"), sam_diagnostic.get("mask_pixel_count")
    if (
        reason not in ("sam_depth_support_insufficient", "sam_depth_far_contaminated")
        or type(support) is not int
        or type(pixels) is not int
        or pixels <= 0
        or support < 0
        or support > pixels
        or (reason == "sam_depth_support_insufficient" and support >= 40)
        or (reason == "sam_depth_far_contaminated" and support < 40)
        or sam_diagnostic.get("source_color_timestamp_ns") != str(frame.color_timestamp_ns)
        or sam_diagnostic.get("source_depth_timestamp_ns") != str(frame.depth_timestamp_ns)
    ):
        raise ValueError("same-tick SAM2 depth rejection is invalid")
    if not isinstance(sam_visual_mask_evidence, Mapping):
        raise ValueError("same-tick SAM2 visual mask evidence required")
    e = sam_visual_mask_evidence
    if (
        e.get("source") != "original_sam2_video_memory"
        or e.get("stream_key") != list(key)
        or e.get("color_timestamp_ns") != str(frame.color_timestamp_ns)
        or e.get("depth_timestamp_ns") != str(frame.depth_timestamp_ns)
        or e.get("coordinate_frame") != "sensor_color_image"
        or not isinstance(e.get("mask_rle"), dict)
    ):
        raise ValueError("SAM2 visual evidence identity is invalid")
    rgb = frame.sam_color_rgb if frame.sam_color_rgb is not None else frame.color_rgb
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("sensor image is invalid")
    height, width = rgb.shape[:2]
    rle = e["mask_rle"]
    size, counts = rle.get("size"), rle.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int for v in size)
        or size != [height, width]
        or not isinstance(counts, list)
        or any(type(v) is not int for v in counts)
    ):
        raise ValueError("SAM2 visual mask is invalid")
    mask = decode_mask_rle(rle, width=width, height=height)
    if mask is None or not np.any(mask) or encode_mask_rle(mask) != rle:
        raise ValueError("SAM2 visual mask is invalid")
    expected = _mask_evidence_sha256(key, frame.color_timestamp_ns, frame.depth_timestamp_ns, rle)
    if e.get("mask_evidence_sha256") != expected:
        raise ValueError("SAM2 evidence digest mismatch")
    if frame.sensor_calibration is None:
        if (
            not isinstance(frame.depth_mm, np.ndarray)
            or frame.depth_mm.dtype != np.uint16
            or frame.depth_mm.shape != mask.shape
            or int(np.count_nonzero(mask)) != pixels
            or int(np.count_nonzero(mask & (frame.depth_mm > 0))) != support
        ):
            raise ValueError("SAM2 mask-depth diagnostic disagrees with input")
        if reason == "sam_depth_far_contaminated":
            far = int(np.count_nonzero(mask & (frame.depth_mm > 3000)))
            near = int(np.count_nonzero(mask & (frame.depth_mm >= 150) & (frame.depth_mm <= 1500)))
            if far / support < 0.30 or near >= 40:
                raise ValueError("SAM2 far-depth diagnostic disagrees with input")
    tick.update(
        status="not_observable",
        reason=reason,
        mask_available=True,
        source_mask_evidence_sha256=expected,
        camera_from_object=None,
        wire_authorized=False,
        mask_diagnostic={
            "reason": reason,
            "depth_supported_pixels": support,
            "mask_pixel_count": pixels,
            "source_color_timestamp_ns": str(frame.color_timestamp_ns),
            "source_depth_timestamp_ns": str(frame.depth_timestamp_ns),
        },
    )
    return tick
