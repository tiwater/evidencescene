"""Runtime-only RGB-D rectification shared by the live BundleTrack route.

This module intentionally has no replay/export or training imports.  The
production scene-runtime image can therefore feed the existing persistent
SAM2 + BundleTrack world model without copying the research/training package
into the image.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np


def rectify_registered_rgbd(
    *,
    color: np.ndarray,
    color_mask: np.ndarray,
    color_info: Any,
    registered: Any,
    source_label: str = "validated_k_d_color_from_depth_registered_then_rectified",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Rectify one registered RGB-D observation without inventing depth.

    This is the same operation used by the original offline BundleTrack
    exporter: RGB and mask use linear/nearest remapping, while metric depth is
    always sampled with nearest-neighbour interpolation.
    """

    if not registered.ready:
        raise ValueError(f"RGB-D registration failed: {registered.reason}")
    image = np.asarray(color, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color must be an HxWx3 image")
    height, width = image.shape[:2]
    mask = np.asarray(color_mask, dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError("semantic mask does not match color geometry")
    distorted_depth_m = registered.color_depth_image()
    rectified_k, map_x, map_y = _cached_rectification(
        width,
        height,
        float(color_info.fx),
        float(color_info.fy),
        float(color_info.cx),
        float(color_info.cy),
        tuple(float(value) for value in color_info.d),
    )
    rectified_rgb = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # Nearest-neighbour preserves actual metric samples.  Bilinear depth
    # would fabricate a surface between measurements.
    rectified_depth_m = cv2.remap(
        distorted_depth_m,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    # Temporal SAM frames deliberately carry an explicit empty mask. Remapping
    # an all-zero image cannot create support, so skip that full-frame OpenCV
    # pass while retaining the exact empty-mask contract.
    if not np.any(mask):
        rectified_mask = np.zeros((height, width), dtype=bool)
    else:
        rectified_mask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
    rectified_depth_mm = np.clip(
        np.rint(rectified_depth_m * 1000.0),
        0.0,
        float(np.iinfo(np.uint16).max),
    ).astype(np.uint16)
    supported = rectified_mask & (rectified_depth_mm > 0)
    diagnostics = {
        "source": source_label,
        "raw_depth_backed": True,
        "metric_depth_interpolation": "none_nearest_sample_only",
        "registered_point_count": int(len(registered.points_depth_m)),
        "mask_pixel_count": int(np.count_nonzero(rectified_mask)),
        "mask_depth_supported_pixel_count": int(np.count_nonzero(supported)),
        "color_depth_sync_skew_ms": registered.sync_skew_ms,
    }
    return (
        rectified_rgb,
        rectified_depth_mm,
        rectified_mask,
        np.asarray(rectified_k, dtype=np.float64),
        diagnostics,
    )


def warp_sensor_mask_to_rectified(
    mask: np.ndarray,
    calibration: tuple,
) -> np.ndarray:
    """Map a sensor-grid mask through the same undistort used for RGB-D.

    SAM tracks the original color image. BundleTrack consumes the rectified
    depth grid, so the mask has to cross that same map before the depth gate.
    """
    width, height, fx, fy, cx, cy, distortion = calibration
    _rectified_k, map_x, map_y = _cached_rectification(
        int(width),
        int(height),
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        tuple(float(value) for value in distortion),
    )
    warped = cv2.remap(
        np.asarray(mask, dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return warped.astype(bool)


def _camera_matrix(info: Any) -> np.ndarray:
    return np.asarray(
        (
            (float(info.fx), 0.0, float(info.cx)),
            (0.0, float(info.fy), float(info.cy)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


@lru_cache(maxsize=16)
def _cached_rectification(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache calibration-only maps while keeping pixel evidence per-frame.

    Camera intrinsics and distortion are stream calibration, not observation
    state.  Reusing their OpenCV maps removes repeated calibration setup from
    every RGB-D tick without changing the registered samples or interpolation
    policy.  The small bounded cache avoids retaining arbitrary reconnect
    calibrations indefinitely.
    """
    k = np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float64)
    d = np.asarray(distortion, dtype=np.float64)
    rectified_k, _roi = cv2.getOptimalNewCameraMatrix(
        k,
        d,
        (width, height),
        0.0,
        (width, height),
        centerPrincipalPoint=False,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        rectified_k,
        (width, height),
        cv2.CV_32FC1,
    )
    for array in (rectified_k, map_x, map_y):
        array.setflags(write=False)
    return rectified_k, map_x, map_y

