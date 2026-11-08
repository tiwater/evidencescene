"""Identity-bound causal owner of the original SAM2 video-memory tracker.

Produces current dense-mask proposals for BundleTrack, not world poses. The
owner must serialize calls and supply either a frame budget or a tracker with
explicit bounded forward-memory retention.
"""
from dataclasses import replace
import hashlib
import json
import math
import os
import time

import numpy as np

from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput


def _same_tracker_new_sensor_epoch(old: tuple, new: tuple) -> bool:
    return (
        isinstance(old, tuple)
        and isinstance(new, tuple)
        and len(old) == len(new)
        and len(old) > 2
        and old[0] == new[0]
        and old[2:] == new[2:]
        and old[1] != new[1]
    )


def _mask_evidence_sha256(stream_key, color_timestamp_ns, depth_timestamp_ns, mask_rle) -> str:
    """Identify one exact SAM2 mask and its sensor observation."""
    payload = json.dumps(
        {
            "source": "original_sam2_video_memory",
            "stream_key": list(stream_key),
            "color_timestamp_ns": str(color_timestamp_ns),
            "depth_timestamp_ns": str(depth_timestamp_ns),
            "mask_rle": mask_rle,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SAM2ObservationSession:
    def __init__(self, tracker, *, maximum_frames: int | None,
                 maximum_sync_skew_ms: float = 20.0, visual_initialization: bool = False):
        if maximum_frames is None:
            if getattr(tracker, "bounded_forward_memory", False) is not True:
                raise ValueError("continuous SAM session requires bounded forward memory")
        elif type(maximum_frames) is not int or maximum_frames <= 0:
            raise ValueError("explicit positive SAM frame budget required")
        self.tracker = tracker
        self.maximum_frames = maximum_frames
        if (not math.isfinite(float(maximum_sync_skew_ms))
                or float(maximum_sync_skew_ms) <= 0.0):
            raise ValueError("maximum_sync_skew_ms must be finite and positive")
        self.maximum_sync_skew_ns = int(float(maximum_sync_skew_ms) * 1_000_000.0)
        if self.maximum_sync_skew_ns <= 0:
            raise ValueError("maximum_sync_skew_ms must be positive")
        self.frame_count = 0
        self._key = None
        self._color_ns = None
        self._depth_ns = None
        self._closed = False
        self._failed = False
        self.last_diagnostic = None
        self.current_sensor_mask_evidence = None
        self.current_visual_mask_evidence = None
        self.current_visual_mask = None
        self.visual_initialization = visual_initialization

    def advance(self, frame: RegisteredBundleTrackInput, *, initialize: bool = False):
        self.current_sensor_mask_evidence = None
        self.current_visual_mask_evidence = None
        self.current_visual_mask = None
        if self._closed or self._failed:
            raise RuntimeError("SAM observation session closed or failed")
        if not isinstance(frame, RegisteredBundleTrackInput):
            raise ValueError("identity-bound registered input required")
        if (type(frame.color_timestamp_ns) is not int or type(frame.depth_timestamp_ns) is not int
                or min(frame.color_timestamp_ns, frame.depth_timestamp_ns) <= 0
                or abs(frame.color_timestamp_ns - frame.depth_timestamp_ns) > self.maximum_sync_skew_ns):
            raise ValueError(
                "SAM requires positive matching sensor timestamps within "
                f"{self.maximum_sync_skew_ns / 1_000_000.0:g} ms"
            )
        if type(initialize) is not bool or initialize != (self.frame_count == 0):
            raise ValueError("a measured anchor is required exactly once at initialization")
        new_epoch = self._key is not None and _same_tracker_new_sensor_epoch(self._key, frame.stream_key)
        if self._key is not None and frame.stream_key != self._key and not new_epoch:
            raise ValueError("SAM camera, object, calibration or source epoch changed")
        if (self._color_ns is not None and (frame.color_timestamp_ns <= self._color_ns
                or frame.depth_timestamp_ns <= self._depth_ns) and not new_epoch):
            raise ValueError("SAM cannot reuse or reorder sensor evidence")
        if self.maximum_frames is not None and self.frame_count >= self.maximum_frames:
            raise RuntimeError("SAM memory budget exhausted; new measured initialization required")
        if initialize and not self.visual_initialization and np.count_nonzero(frame.mask & (frame.depth_mm > 0)) < 40:
            raise ValueError("SAM anchor has insufficient current depth support")
        # The anchor's final immutable evidence digest is independent of SAM2
        # output. Start it while the existing video-memory owner is on GPU;
        # non-anchor frames are prefetched below after their current mask is
        # known, immediately before BundleTrack consumes them.
        if initialize:
            prefetch = getattr(frame, "prefetch_evidence_hash", None)
            if callable(prefetch):
                prefetch()
        tracking_started = time.perf_counter()
        source_rgb = frame.sam_color_rgb
        image = source_rgb if isinstance(source_rgb, np.ndarray) else frame.color_rgb
        if initialize:
            source_anchor = frame.sam_anchor_mask
            anchor_mask = source_anchor if isinstance(source_anchor, np.ndarray) else frame.mask
            if not np.any(anchor_mask):
                raise ValueError("nonempty explicit visual anchor required")
        else:
            anchor_mask = None
        try:
            # SAM2's official transform accepts RGB ndarrays directly and
            # produces the same tensor as the PIL route. Avoid wrapping every
            # registered frame in a temporary PIL object. When the color
            # camera is distorted, ``image`` is the sensor RGB the anchor was
            # drawn on; the mask is warped onto the rectified grid below.
            tracked = self.tracker.track(image=image, anchor_mask=anchor_mask)
        except Exception:
            # Upstream state may already have advanced. Never retry it as if
            # no observation had been consumed.
            self._failed = True
            raise
        self._key = frame.stream_key
        self._color_ns, self._depth_ns = frame.color_timestamp_ns, frame.depth_timestamp_ns
        self.frame_count += 1
        empty_reason = getattr(self.tracker, "_last_empty_reason", None) or "sam_mask_unavailable"
        self.last_diagnostic = dict(reason=empty_reason, depth_supported_pixels=0,
            mask_pixel_count=0,
            source_color_timestamp_ns=str(frame.color_timestamp_ns),
            source_depth_timestamp_ns=str(frame.depth_timestamp_ns),
            tracking_latency_ms=round((time.perf_counter() - tracking_started) * 1000.0, 3))
        if tracked is None:
            return None
        sensor_mask = tracked.mask
        if (isinstance(sensor_mask, np.ndarray) and sensor_mask.dtype == np.bool_
                and sensor_mask.shape == image.shape[:2] and np.any(sensor_mask)):
            from evidencescene.primitives.image_mask import encode_mask_rle

            self.current_visual_mask = sensor_mask.copy()
            visual_mask_rle = encode_mask_rle(sensor_mask)
            self.current_visual_mask_evidence = {
                "source": "original_sam2_video_memory",
                "stream_key": list(frame.stream_key),
                "color_timestamp_ns": str(frame.color_timestamp_ns),
                "depth_timestamp_ns": str(frame.depth_timestamp_ns),
                "mask_rle": visual_mask_rle,
                "mask_evidence_sha256": _mask_evidence_sha256(
                    frame.stream_key,
                    frame.color_timestamp_ns,
                    frame.depth_timestamp_ns,
                    visual_mask_rle,
                ),
                "coordinate_frame": "sensor_color_image",
                "scope": "image_only_not_metric_pose",
            }
        mask = tracked.mask
        if frame.sensor_calibration is not None and isinstance(source_rgb, np.ndarray):
            from evidencescene.primitives.registered_rgbd import warp_sensor_mask_to_rectified

            if (
                isinstance(mask, np.ndarray)
                and mask.shape == (int(frame.sensor_calibration[1]), int(frame.sensor_calibration[0]))
            ):
                mask = warp_sensor_mask_to_rectified(mask, frame.sensor_calibration)
        if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                or mask.shape != frame.depth_mm.shape):
            self.last_diagnostic["reason"] = "sam_mask_grid_invalid"
            return None
        mask_pixels = int(np.count_nonzero(mask))
        positive = mask & (frame.depth_mm > 0)
        supported = int(np.count_nonzero(positive))
        far = int(np.count_nonzero(positive & (frame.depth_mm > 3000)))
        far_fraction = (far / supported) if supported else 0.0
        if os.getenv("TICOS_SAM_MASK_DUMP") == "1":
            _dump_tracked_mask(
                self.frame_count,
                frame.color_rgb,
                mask,
                frame.depth_mm,
                supported=supported,
                far_fraction=far_fraction,
            )
        # A window mask is mostly >3 m and has no carton-range depth. Drop it
        # from SAM memory and do not publish it.
        near = positive & (frame.depth_mm >= 150) & (frame.depth_mm <= 1500)
        near_count = int(np.count_nonzero(near))
        near_median = float(np.median(frame.depth_mm[near])) if near_count else 0.0
        if not initialize and supported >= 40 and far_fraction >= 0.30 and near_count < 40:
            discard = getattr(self.tracker, "discard_last_forward_frame", None)
            if callable(discard) and not self.visual_initialization:
                discard()
            self.last_diagnostic.update(
                depth_supported_pixels=supported,
                mask_pixel_count=mask_pixels,
                depth_far_fraction=round(far_fraction, 4),
                reason="sam_depth_far_contaminated",
            )
            return None
        # Window-range samples inside a carton mask are depth bleed. Clear
        # them so BundleTrack cannot lock onto the window. The carton-range
        # samples stay. Hole filling, when the matcher has no depth at its
        # pixels, happens inside the tracker and only after that match fails.
        if not initialize and far_fraction >= 0.30 and near_count >= 40:
            depth_mm = np.array(frame.depth_mm, copy=True)
            depth_mm[mask & (frame.depth_mm > 1500)] = 0
            frame = replace(frame, depth_mm=np.ascontiguousarray(depth_mm))
            positive = mask & (depth_mm > 0)
            supported = int(np.count_nonzero(positive))
            self.last_diagnostic["depth_far_zeroed"] = True
            self.last_diagnostic["depth_near_median_mm"] = int(near_median)
        self.last_diagnostic.update(depth_supported_pixels=supported,
            mask_pixel_count=mask_pixels,
            reason="sam_depth_support_insufficient" if supported < 40 else "current_mask_candidate")
        if supported < 40:
            return None
        # Retain the accepted mask on the sensor RGB grid, not the rectified
        # tracker grid. This is the same SAM output, not another estimator.
        from evidencescene.primitives.image_mask import encode_mask_rle

        sensor_mask = tracked.mask
        if (
            not self.last_diagnostic.get("depth_far_zeroed", False)
            and isinstance(sensor_mask, np.ndarray)
            and sensor_mask.dtype == np.bool_
            and sensor_mask.shape == image.shape[:2]
        ):
            sensor_mask_rle = encode_mask_rle(sensor_mask)
            self.current_sensor_mask_evidence = {
                "source": "original_sam2_video_memory",
                "stream_key": list(frame.stream_key),
                "color_timestamp_ns": str(frame.color_timestamp_ns),
                "depth_timestamp_ns": str(frame.depth_timestamp_ns),
                "mask_rle": sensor_mask_rle,
                "mask_evidence_sha256": _mask_evidence_sha256(
                    frame.stream_key,
                    frame.color_timestamp_ns,
                    frame.depth_timestamp_ns,
                    sensor_mask_rle,
                ),
                "coordinate_frame": "sensor_color_image",
            }
        current = replace(frame, mask=mask)
        prefetch = getattr(current, "prefetch_evidence_hash", None)
        if callable(prefetch):
            prefetch()
        return current

    def close(self):
        if not self._closed:
            self.tracker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _dump_tracked_mask(frame_count, color, mask, depth_mm, *, supported, far_fraction):
    """Write one overlay so a far or unmatched mask can be checked by eye."""
    import json
    from pathlib import Path

    import cv2

    root = Path("/tmp/ticos-edge-live-ab/mask-dump")
    root.mkdir(parents=True, exist_ok=True)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        bbox = None
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    near = mask & (depth_mm > 0) & (depth_mm <= 3000)
    far = mask & (depth_mm > 3000)
    near_n = int(np.count_nonzero(near))
    near_median = int(np.median(depth_mm[near])) if near_n else None
    image = np.ascontiguousarray(color)
    if image.ndim == 3 and image.shape[2] == 3:
        drawn = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        drawn = image.copy()
    tint = np.zeros_like(drawn)
    tint[mask] = (0, 255, 0)
    tint[far] = (0, 0, 255)
    drawn = cv2.addWeighted(drawn, 0.65, tint, 0.35, 0)
    stem = f"{int(frame_count):06d}"
    cv2.imwrite(str(root / f"{stem}.jpg"), drawn)
    (root / f"{stem}.json").write_text(json.dumps({
        "frame_count": int(frame_count),
        "bbox_xyxy": bbox,
        "mask_pixels": int(np.count_nonzero(mask)),
        "depth_supported_pixels": int(supported),
        "far_fraction": round(float(far_fraction), 4),
        "near_pixels": near_n,
        "near_median_mm": near_median,
    }), encoding="utf-8")

