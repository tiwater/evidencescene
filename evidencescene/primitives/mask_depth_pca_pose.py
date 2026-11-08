"""Mask+depth camera-local pose helpers for shadow scoring and BT recovery.

``advance_mask_depth_pca_pose`` remains shadow-only (full PCA axes).

``recover_camera_from_object_with_prior`` is the production recovery adapter:
when BundleTrack misses but SAM2 mask + depth support remain, keep the last
BundleTrack rotation and update translation from the mask-depth centroid,
clipped to the camera-local jump gate. It is not a second world model.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

# Match the existing BundleTrack / SAM depth-support gate used on the live path.
DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS = 100
# Recovery may sample with stride; SAM can still report ≥100 raw support
# while strided backprojection falls just under 100 during pour motion.
DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS = 50
DEFAULT_MINIMUM_DEPTH_M = 0.15
DEFAULT_MAXIMUM_DEPTH_M = 3.0
DEFAULT_PIXEL_STRIDE = 2
# Recovery and its native-owner re-entry must respect the configured pouring
# per-frame camera-local maximum jump (40 mm), including across dropped poses.
DEFAULT_MAXIMUM_TRANSLATION_STEP_M = 0.04
MASK_DEPTH_RECOVERY_CENTROID_ESTIMATORS = ("mean", "depth_mad_consensus")
DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR = "mean"
DEPTH_MAD_CONSENSUS_SIGMA = 2.5
DEPTH_MAD_CONSENSUS_MINIMUM_BAND_M = 0.010
DEFAULT_MASK_DEPTH_RECOVERY_METHOD = "mask_centroid"
MASK_DEPTH_RECOVERY_METHODS = (
    DEFAULT_MASK_DEPTH_RECOVERY_METHOD,
    "causal_surface_translation_icp",
)
SURFACE_ICP_MATCH_DISTANCE_M = 0.020
SURFACE_ICP_MAXIMUM_ITERATIONS = 8


def _finite_intrinsics(camera_matrix: object) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3x3 array")
    return matrix


def backproject_masked_depth_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    minimum_depth_m: float = DEFAULT_MINIMUM_DEPTH_M,
    maximum_depth_m: float = DEFAULT_MAXIMUM_DEPTH_M,
) -> np.ndarray:
    """Return Nx3 camera-frame points from a boolean mask and metric depth."""
    depth = np.asarray(depth_m)
    selected = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth_m and mask must share the same 2D shape")
    if type(pixel_stride) is not int or pixel_stride <= 0:
        raise ValueError("pixel_stride must be a positive int")
    intrinsics = _finite_intrinsics(camera_matrix)
    sampled_depth = depth[::pixel_stride, ::pixel_stride]
    sampled_mask = selected[::pixel_stride, ::pixel_stride]
    metric = np.asarray(sampled_depth, dtype=np.float64)
    valid = (
        sampled_mask
        & np.isfinite(metric)
        & (metric >= float(minimum_depth_m))
        & (metric <= float(maximum_depth_m))
        & (metric > 0.0)
    )
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rows = rows * pixel_stride
    cols = cols * pixel_stride
    z = np.asarray(depth[rows, cols], dtype=np.float64)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z))


def estimate_mask_depth_centroid_camera(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    minimum_depth_support_pixels: int = DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
) -> tuple[np.ndarray | None, int]:
    """Return the current metric mask centroid when registered depth supports it."""
    centroid, support, _diagnostics = estimate_mask_depth_centroid_camera_with_diagnostics(
        depth_m=depth_m,
        mask=mask,
        camera_matrix=camera_matrix,
        minimum_depth_support_pixels=minimum_depth_support_pixels,
        pixel_stride=pixel_stride,
        estimator=DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
    )
    return centroid, support


def estimate_mask_depth_centroid_camera_with_diagnostics(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    minimum_depth_support_pixels: int = DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    estimator: str = DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
) -> tuple[np.ndarray | None, int, dict[str, Any]]:
    """Estimate a measured mask centroid with an explicit depth-consensus policy.

    Producer/state owner: ``ObjectCentricObservationSession.advance`` under
    ``OriginalPersistentWorldRegistry``. This remains a translation adapter on
    that owner's current SAM2 mask and BundleTrack rotation; it adds no tracker,
    object frame, or world estimate. ``mean`` preserves the existing behavior.
    ``depth_mad_consensus`` rejects depth samples outside a robust global depth
    band and fails closed unless the band retains minimum support and a strict
    majority of the depth samples. The consensus position is still only a
    camera-local visible-surface measurement.
    """
    if estimator not in MASK_DEPTH_RECOVERY_CENTROID_ESTIMATORS:
        raise ValueError("unsupported mask-depth recovery centroid estimator")
    if type(minimum_depth_support_pixels) is not int or minimum_depth_support_pixels <= 0:
        raise ValueError("minimum_depth_support_pixels must be a positive integer")
    points = backproject_masked_depth_points(
        depth_m,
        mask,
        camera_matrix,
        pixel_stride=pixel_stride,
    )
    support = int(points.shape[0])
    sampled_stride = int(pixel_stride)
    if support < int(minimum_depth_support_pixels) and int(pixel_stride) > 1:
        points = backproject_masked_depth_points(
            depth_m,
            mask,
            camera_matrix,
            pixel_stride=1,
        )
        support = int(points.shape[0])
        sampled_stride = 1
    if support < int(minimum_depth_support_pixels):
        return None, support, {
            "centroid_estimator": estimator,
            "depth_supported_pixels": support,
            "centroid_support_pixels": 0,
            "centroid_depth_consensus_fraction": 0.0,
            "centroid_sample_stride": sampled_stride,
            "centroid_rejection_reason": "sam_depth_support_insufficient",
        }

    selected = points
    depth_median = None
    depth_mad = None
    depth_tolerance = None
    if estimator == "depth_mad_consensus":
        z = points[:, 2]
        depth_median = float(np.median(z))
        depth_mad = float(np.median(np.abs(z - depth_median)))
        order = np.argsort(z, kind="stable")
        sorted_z = z[order]
        gaps = np.diff(sorted_z)
        split_at = np.flatnonzero(gaps > DEPTH_MAD_CONSENSUS_MINIMUM_BAND_M) + 1
        layers = np.split(order, split_at)
        layer_index = int(np.argmax([len(layer) for layer in layers]))
        selected_indices = layers[layer_index]
        selected = points[selected_indices]
        selected_support = int(len(selected))
        consensus_fraction = selected_support / float(support)
        depth_tolerance = (
            float(sorted_z[selected_indices[-1]] - sorted_z[selected_indices[0]])
            if selected_support
            else 0.0
        )
        diagnostics = {
            "centroid_estimator": estimator,
            "depth_supported_pixels": support,
            "centroid_support_pixels": selected_support,
            "centroid_depth_consensus_fraction": consensus_fraction,
            "centroid_depth_median_m": depth_median,
            "centroid_depth_mad_m": depth_mad,
            "centroid_depth_layer_span_m": depth_tolerance,
            "centroid_depth_layer_count": len(layers),
            "centroid_sample_stride": sampled_stride,
        }
        if selected_support < int(minimum_depth_support_pixels):
            return None, selected_support, {
                **diagnostics,
                "centroid_rejection_reason": "sam_depth_consensus_support_insufficient",
            }
        if consensus_fraction <= 0.5:
            return None, selected_support, {
                **diagnostics,
                "centroid_rejection_reason": "sam_depth_consensus_not_majority",
            }
    else:
        selected_support = support
        consensus_fraction = 1.0
        diagnostics = {
            "centroid_estimator": estimator,
            "depth_supported_pixels": support,
            "centroid_support_pixels": selected_support,
            "centroid_depth_consensus_fraction": consensus_fraction,
            "centroid_sample_stride": sampled_stride,
        }
    return (
        np.asarray(selected.mean(axis=0), dtype=np.float64),
        selected_support,
        diagnostics,
    )


def _orthonormal_basis_from_covariance(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = np.asarray(eigenvectors[:, order], dtype=np.float64)
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, -1] *= -1.0
    # Keep a right-handed frame after possible reflection.
    if float(np.linalg.det(basis)) < 0.0:
        basis[:, 0] *= -1.0
    return basis


def estimate_camera_from_object_pca(
    points_camera_m: np.ndarray,
    *,
    minimum_points: int = DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
) -> np.ndarray | None:
    """Build a camera_from_object SE3 with origin at the point cloud centroid."""
    points = np.asarray(points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_m must be an Nx3 array")
    if points.shape[0] < int(minimum_points):
        return None
    if not np.isfinite(points).all():
        return None
    centroid = points.mean(axis=0)
    rotation = _orthonormal_basis_from_covariance(points)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = centroid
    return pose


def recover_camera_from_object_with_prior(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    prior_camera_from_object: np.ndarray,
    prior_mask_centroid_camera: np.ndarray,
    maximum_translation_step_m: float = DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    minimum_depth_support_pixels: int = DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    centroid_estimator: str = DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """Recover a camera_from_object pose using prior rotation + mask centroid.

    Returns ``(pose, reason, diagnostics)``. On success ``reason`` is None.
    Translation is clipped so consecutive recovery steps cannot exceed the jump
    gate; rotation is copied from the last BundleTrack candidate.
    """
    prior = np.asarray(prior_camera_from_object, dtype=np.float64)
    if prior.shape != (4, 4) or not np.isfinite(prior).all():
        return None, "recovery_prior_pose_invalid", {"depth_supported_pixels": 0}
    if not np.allclose(prior[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        return None, "recovery_prior_pose_invalid", {"depth_supported_pixels": 0}
    rotation = prior[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
        return None, "recovery_prior_pose_invalid", {"depth_supported_pixels": 0}
    prior_centroid = np.asarray(prior_mask_centroid_camera, dtype=np.float64)
    if prior_centroid.shape != (3,) or not np.isfinite(prior_centroid).all():
        return (
            None,
            "recovery_prior_mask_centroid_invalid",
            {"depth_supported_pixels": 0},
        )
    if (isinstance(maximum_translation_step_m, (bool, np.bool_))
            or not np.isfinite(maximum_translation_step_m) or float(maximum_translation_step_m) <= 0.0):
        raise ValueError("maximum_translation_step_m must be finite and positive")
    centroid, support, centroid_diagnostics = estimate_mask_depth_centroid_camera_with_diagnostics(
        depth_m=depth_m,
        mask=mask,
        camera_matrix=camera_matrix,
        minimum_depth_support_pixels=minimum_depth_support_pixels,
        pixel_stride=pixel_stride,
        estimator=centroid_estimator,
    )

    if centroid is None:
        rejection_reason = str(
            centroid_diagnostics.get("centroid_rejection_reason")
            or "sam_depth_support_insufficient"
        )
        return (
            None,
            rejection_reason,
            dict(centroid_diagnostics, depth_supported_pixels=support),
        )
    prior_translation = prior[:3, 3]
    # The mask centroid is not the BundleTrack object origin. Use its motion
    # relative to the prior observation, then apply that displacement to the
    # BundleTrack-gauge origin instead of snapping the origin to the centroid.
    delta = centroid - prior_centroid
    distance = float(np.linalg.norm(delta))
    if distance > float(maximum_translation_step_m):
        delta = delta * (float(maximum_translation_step_m) / distance)
        clipped = True
    else:
        clipped = False
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = prior[:3, :3]
    pose[:3, 3] = prior_translation + delta
    return (
        pose,
        None,
        {
            "depth_supported_pixels": support,
            **centroid_diagnostics,
            "translation_step_m": float(np.linalg.norm(delta)),
            "translation_clipped": clipped,
            "raw_centroid_step_m": distance,
            "mask_centroid_camera_m": centroid.tolist(),
            "pose_source": "mask_depth_recovery",
        },
    )


def recover_camera_from_object_with_surface_prior(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    prior_camera_from_object: np.ndarray,
    prior_mask_centroid_camera: np.ndarray | None,
    surface_points_object_m: np.ndarray,
    maximum_translation_step_m: float = DEFAULT_MAXIMUM_TRANSLATION_STEP_M,
    minimum_depth_support_pixels: int = DEFAULT_RECOVERY_MINIMUM_DEPTH_SUPPORT_PIXELS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    maximum_surface_match_distance_m: float = SURFACE_ICP_MATCH_DISTANCE_M,
    maximum_iterations: int = SURFACE_ICP_MAXIMUM_ITERATIONS,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """Recover translation by fitting current mask depths to the owner's surface.

    The state owner remains ``OriginalPersistentWorldRegistry`` through its
    ``ObjectCentricObservationSession.advance`` path.  The sole geometry prior
    is the already accumulated ``CausalObjectSurface`` in BundleTrack's gauge;
    this function keeps the last BundleTrack rotation and only proposes a
    clipped camera-local translation.  It is an adapter on that owner, not a
    second tracker or world model.  The caller must supply a causal snapshot
    from before the current frame is fused.

    A depth sample supports translation only when its 3D point agrees with
    the existing object-gauge surface.  Background rays inside an instance
    mask therefore cannot move the pose unless they also fit that prior.
    Ambiguous or unsupported matches fail closed.
    """
    prior = np.asarray(prior_camera_from_object, dtype=np.float64)
    if (
        prior.shape != (4, 4)
        or not np.isfinite(prior).all()
        or not np.allclose(prior[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5)
    ):
        return None, "recovery_prior_pose_invalid", {"depth_supported_pixels": 0}
    rotation = prior[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        return None, "recovery_prior_pose_invalid", {"depth_supported_pixels": 0}
    if (
        isinstance(maximum_translation_step_m, (bool, np.bool_))
        or not np.isfinite(maximum_translation_step_m)
        or float(maximum_translation_step_m) <= 0.0
        or not np.isfinite(maximum_surface_match_distance_m)
        or float(maximum_surface_match_distance_m) <= 0.0
        or type(maximum_iterations) is not int
        or maximum_iterations <= 0
    ):
        raise ValueError("surface recovery limits must be finite and positive")

    points = backproject_masked_depth_points(
        depth_m,
        mask,
        camera_matrix,
        pixel_stride=pixel_stride,
    )
    sampled_stride = int(pixel_stride)
    if len(points) < int(minimum_depth_support_pixels) and pixel_stride > 1:
        points = backproject_masked_depth_points(
            depth_m,
            mask,
            camera_matrix,
            pixel_stride=1,
        )
        sampled_stride = 1
    depth_support = int(len(points))
    if depth_support < int(minimum_depth_support_pixels):
        return None, "sam_depth_support_insufficient", {
            "depth_supported_pixels": depth_support,
            "surface_prior_point_count": 0,
            "surface_match_support_pixels": 0,
            "centroid_sample_stride": sampled_stride,
        }

    surface = np.asarray(surface_points_object_m, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1:] != (3,):
        return None, "causal_surface_prior_invalid", {
            "depth_supported_pixels": depth_support,
            "surface_prior_point_count": 0,
            "surface_match_support_pixels": 0,
            "centroid_sample_stride": sampled_stride,
        }
    surface = surface[np.isfinite(surface).all(axis=1)]
    if len(surface) < int(minimum_depth_support_pixels):
        return None, "causal_surface_prior_support_insufficient", {
            "depth_supported_pixels": depth_support,
            "surface_prior_point_count": int(len(surface)),
            "surface_match_support_pixels": 0,
            "centroid_sample_stride": sampled_stride,
        }

    anchor, _anchor_support, anchor_diagnostics = (
        estimate_mask_depth_centroid_camera_with_diagnostics(
            depth_m=depth_m,
            mask=mask,
            camera_matrix=camera_matrix,
            minimum_depth_support_pixels=minimum_depth_support_pixels,
            pixel_stride=pixel_stride,
            estimator=DEFAULT_MASK_DEPTH_RECOVERY_CENTROID_ESTIMATOR,
        )
    )
    previous_anchor = (
        np.asarray(prior_mask_centroid_camera, dtype=np.float64)
        if prior_mask_centroid_camera is not None
        else np.empty((0,), dtype=np.float64)
    )
    if anchor is None:
        return None, "sam_depth_support_insufficient", {
            "depth_supported_pixels": depth_support,
            "surface_prior_point_count": int(len(surface)),
            "surface_match_support_pixels": 0,
            "centroid_sample_stride": sampled_stride,
        }

    # Do not seed ICP from the mask centroid: a hollow object can make that
    # centroid belong to the background visible through its opening. Start at
    # the last owner pose, permit one coarse correspondence pass spanning the
    # existing jump gate, then require the fitted points to meet the tighter
    # surface residual gate.
    raw_anchor_step = (
        float(np.linalg.norm(anchor - previous_anchor))
        if previous_anchor.shape == (3,) and np.isfinite(previous_anchor).all()
        else None
    )
    prior_translation = prior[:3, 3]
    translation = prior_translation.copy()
    tree = cKDTree(surface)
    match_support = 0
    match_fraction = 0.0
    residual_median = None
    residual_p95 = None
    converged = False
    clipped = False
    for _iteration in range(maximum_iterations):
        points_object = (points - translation[None, :]) @ rotation
        residuals, nearest = tree.query(points_object, k=1, workers=1)
        match_limit = float(maximum_surface_match_distance_m)
        if _iteration == 0:
            match_limit += float(maximum_translation_step_m)
        inlier = residuals <= match_limit
        match_support = int(np.count_nonzero(inlier))
        match_fraction = match_support / float(depth_support)
        if match_support < int(minimum_depth_support_pixels):
            return None, "causal_surface_match_support_insufficient", {
                "depth_supported_pixels": depth_support,
                "surface_prior_point_count": int(len(surface)),
                "surface_match_support_pixels": match_support,
                "surface_match_fraction": match_fraction,
                "surface_match_distance_m": float(maximum_surface_match_distance_m),
                "centroid_sample_stride": sampled_stride,
                **anchor_diagnostics,
            }
        matched_surface_camera = surface[nearest[inlier]] @ rotation.T
        translation_candidates = points[inlier] - matched_surface_camera
        proposed = np.median(translation_candidates, axis=0)
        raw_step = proposed - prior_translation
        step_length = float(np.linalg.norm(raw_step))
        if step_length > float(maximum_translation_step_m):
            proposed = prior_translation + raw_step * (
                float(maximum_translation_step_m) / step_length
            )
            clipped = True
        converged = float(np.linalg.norm(proposed - translation)) <= 5e-4
        translation = proposed
        if converged:
            break

    points_object = (points - translation[None, :]) @ rotation
    residuals, _nearest = tree.query(points_object, k=1, workers=1)
    final_inlier = residuals <= float(maximum_surface_match_distance_m)
    match_support = int(np.count_nonzero(final_inlier))
    match_fraction = match_support / float(depth_support)
    if match_support < int(minimum_depth_support_pixels):
        return None, "causal_surface_match_support_insufficient", {
            "depth_supported_pixels": depth_support,
            "surface_prior_point_count": int(len(surface)),
            "surface_match_support_pixels": match_support,
            "surface_match_fraction": match_fraction,
            "surface_match_distance_m": float(maximum_surface_match_distance_m),
            "centroid_sample_stride": sampled_stride,
            **anchor_diagnostics,
        }
    selected_residuals = residuals[final_inlier]
    residual_median = float(np.median(selected_residuals))
    residual_p95 = float(np.percentile(selected_residuals, 95.0))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose, None, {
        "depth_supported_pixels": depth_support,
        "surface_prior_point_count": int(len(surface)),
        "surface_match_support_pixels": match_support,
        "surface_match_fraction": match_fraction,
        "surface_match_distance_m": float(maximum_surface_match_distance_m),
        "surface_match_residual_median_m": residual_median,
        "surface_match_residual_p95_m": residual_p95,
        "surface_match_converged": converged,
        "centroid_sample_stride": sampled_stride,
        "raw_anchor_step_m": raw_anchor_step,
        "translation_step_m": float(np.linalg.norm(translation - prior_translation)),
        "translation_clipped": clipped,
        "mask_centroid_camera_m": anchor.tolist(),
        "pose_source": "mask_depth_surface_recovery",
        **anchor_diagnostics,
    }

def advance_mask_depth_pca_pose(
    *,
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    color_timestamp_ns: int,
    depth_timestamp_ns: int,
    minimum_depth_support_pixels: int = DEFAULT_MINIMUM_DEPTH_SUPPORT_PIXELS,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
) -> dict[str, Any]:
    """Advance one shadow tick. Never authorizes world/table publication."""
    started = time.perf_counter()
    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth_m and mask must share the same 2D shape")
    mask_pixels = int(selected.sum())
    depth_support = int(np.count_nonzero(selected & np.isfinite(depth) & (depth > 0)))
    if mask_pixels < int(minimum_depth_support_pixels):
        return {
            "status": "not_observable",
            "reason": "sam_depth_support_insufficient",
            "color_timestamp_ns": str(color_timestamp_ns),
            "depth_timestamp_ns": str(depth_timestamp_ns),
            "camera_from_object": None,
            "mask_available": mask_pixels > 0,
            "mask_pixel_count": mask_pixels,
            "depth_supported_pixels": depth_support,
            "sampled_depth_point_count": None,
            "world_pose_authorized": False,
            "wire_authorized": False,
            "runtime_timing_ms": {
                "pose_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }
    points = backproject_masked_depth_points(
        depth_m,
        mask,
        camera_matrix,
        pixel_stride=pixel_stride,
    )
    pose = estimate_camera_from_object_pca(
        points,
        minimum_points=minimum_depth_support_pixels,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if pose is None:
        return {
            "status": "not_observable",
            "reason": "mask_depth_pca_insufficient_points",
            "color_timestamp_ns": str(color_timestamp_ns),
            "depth_timestamp_ns": str(depth_timestamp_ns),
            "camera_from_object": None,
            "mask_available": True,
            "mask_pixel_count": mask_pixels,
            "depth_supported_pixels": depth_support,
            "sampled_depth_point_count": int(points.shape[0]),
            "world_pose_authorized": False,
            "wire_authorized": False,
            "runtime_timing_ms": {"pose_ms": elapsed_ms, "total_ms": elapsed_ms},
        }
    return {
        "status": "candidate",
        "reason": None,
        "color_timestamp_ns": str(color_timestamp_ns),
        "depth_timestamp_ns": str(depth_timestamp_ns),
        "camera_from_object": pose.tolist(),
        "mask_available": True,
        "mask_pixel_count": mask_pixels,
        "depth_supported_pixels": depth_support,
        "sampled_depth_point_count": int(points.shape[0]),
        "world_pose_authorized": False,
        "wire_authorized": False,
        "runtime_timing_ms": {"pose_ms": elapsed_ms, "total_ms": elapsed_ms},
    }


def tick_is_mask_available(tick: Mapping[str, Any]) -> bool:
    """Return whether a tick should count in the mask-available denominator."""
    if tick.get("mask_available") is True:
        return True
    reason = tick.get("reason")
    if reason == "sam_mask_unavailable":
        return False
    diagnostic = tick.get("mask_diagnostic")
    if isinstance(diagnostic, Mapping):
        diagnostic_reason = diagnostic.get("reason")
        if diagnostic_reason == "sam_mask_unavailable":
            return False
        support = diagnostic.get("depth_supported_pixels")
        if type(support) is int and support > 0:
            return True
    support = tick.get("depth_supported_pixels")
    return type(support) is int and support > 0

