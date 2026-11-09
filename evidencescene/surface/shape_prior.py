"""Preparation-stage generative 3D shape prior anchoring and evidence-first fusion.

In the EvidenceScene pipeline, heavy single-image 3D generative priors (such as
Meta's SAM 3D Objects) operate in **Stage 0 (Preparation / Cold-Start Stage)**
before real-time manipulation begins:

1. **Stage 0 — Metric Depth Anchoring (`anchor_shape_prior_to_depth`)**:
   Given a complete 360-degree 3D object prior in an arbitrary monocular gauge
   and calibrated multi-camera RGB-D front-surface measurements at t=0,
   estimates a deterministic Sim(3) similarity transform (scale s > 0, rotation
   R in SO(3), translation t in R^3) via Umeyama registration followed by
   trimmed front-surface closest-point refinement.

2. **Stage 1/2 — Evidence-First Surface Fusion (`fuse_prior_with_measured_surface`)**:
   During real-time 30 Hz manipulation, the metric-anchored 360-degree prior
   moves rigidly with the observability-gated 6-DoF object pose. Within every
   observed coarse cell (12 mm), multi-view corroborated sensor surfels from
   `RevisableSurfelMap` (support >= 2) supersede prior points, while unobserved
   and self-occluded back-side regions retain the metric-anchored prior so the
   scene representation remains complete without fabricating sensor authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Sim3AnchorResult:
    """Metric anchoring result mapping an unscaled 3D shape prior to object gauge."""

    scale: float
    rotation: np.ndarray
    translation_m: np.ndarray
    similarity_matrix: np.ndarray
    rigid_matrix: np.ndarray
    inlier_count: int
    inlier_rmse_m: float
    trimmed_fraction: float


@dataclass(frozen=True)
class PreparedSceneObjectPrior:
    """Complete 360-degree scene object asset anchored during Stage-0 preparation."""

    object_id: str
    prior_source: str
    points_object_m: np.ndarray
    colors_rgb: np.ndarray | None
    faces: np.ndarray | None
    anchor: Sim3AnchorResult
    gauge_id: str = "object_gauge"


def _umeyama_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    allow_scale: bool = True,
    scale_bounds: tuple[float, float] = (0.05, 20.0),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Umeyama similarity registration from source_points to target_points."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or dst.shape != src.shape or len(src) < 3:
        raise ValueError("source and target must be matching Nx3 arrays with N >= 3")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    var_src = float(np.mean(np.sum(src_centered**2, axis=1)))
    if var_src <= 1e-15:
        raise ValueError("degenerate source point cloud with near-zero variance")

    cov = (dst_centered.T @ src_centered) / float(len(src))
    u_mat, singular_values, vt_mat = np.linalg.svd(cov)
    d_sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0.0:
        d_sign[2] = -1.0

    rotation = u_mat @ np.diag(d_sign) @ vt_mat
    if allow_scale:
        raw_scale = float(np.sum(singular_values * d_sign) / var_src)
        scale = float(np.clip(raw_scale, scale_bounds[0], scale_bounds[1]))
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def anchor_shape_prior_to_depth(
    prior_points: Any,
    measured_front_points: Any,
    *,
    correspondences: np.ndarray | None = None,
    trim_fraction: float = 0.85,
    max_iterations: int = 20,
    tolerance_m: float = 1e-6,
    allow_scale: bool = True,
    scale_bounds: tuple[float, float] = (0.05, 20.0),
) -> tuple[np.ndarray, Sim3AnchorResult]:
    """Anchor a Stage-0 generative 3D shape prior (e.g. SAM 3D Objects) to metric depth.

    Parameters
    ----------
    prior_points:
        Nx3 array of complete 3D object prior coordinates in arbitrary monocular gauge.
    measured_front_points:
        Mx3 array of calibrated multi-camera RGB-D points measured on the visible surface.
    correspondences:
        Optional Kx2 integer array of (prior_index, measured_index) initial pairs.
        When omitted, initializes scale and translation from bounding-box / centroid
        statistics on the visible front hemisphere and refines via trimmed Sim(3) ICP.
    trim_fraction:
        Fraction of closest measured front-surface points retained per iteration
        (in (0.2, 1.0]) so self-occluded back-side prior vertices do not bias front fit.
    """
    prior = np.asarray(prior_points, dtype=np.float64)
    measured = np.asarray(measured_front_points, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != 3 or len(prior) < 3 or not np.isfinite(prior).all():
        raise ValueError("prior_points must be finite Nx3 with N >= 3")
    if (
        measured.ndim != 2
        or measured.shape[1] != 3
        or len(measured) < 3
        or not np.isfinite(measured).all()
    ):
        raise ValueError("measured_front_points must be finite Mx3 with M >= 3")
    if not (0.2 <= float(trim_fraction) <= 1.0):
        raise ValueError("trim_fraction must be in [0.2, 1.0]")
    if (
        len(scale_bounds) != 2
        or scale_bounds[0] <= 0.0
        or scale_bounds[0] > scale_bounds[1]
    ):
        raise ValueError("scale_bounds must be a positive (min_scale, max_scale) pair")

    if correspondences is not None:
        pairs = np.asarray(correspondences, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 3:
            raise ValueError("correspondences must be Kx2 integer indices with K >= 3")
        scale, rotation, translation = _umeyama_sim3(
            prior[pairs[:, 0]],
            measured[pairs[:, 1]],
            allow_scale=allow_scale,
            scale_bounds=scale_bounds,
        )
    else:
        mu_p = prior.mean(axis=0)
        mu_m = measured.mean(axis=0)
        span_p = float(np.linalg.norm(np.ptp(prior, axis=0)))
        span_m = float(np.linalg.norm(np.ptp(measured, axis=0)))
        if span_p <= 1e-12:
            raise ValueError("degenerate prior_points with zero bounding span")
        scale = (
            float(np.clip(span_m / span_p, scale_bounds[0], scale_bounds[1]))
            if allow_scale
            else 1.0
        )
        rotation = np.eye(3, dtype=np.float64)
        translation = mu_m - scale * mu_p

    keep_count = max(3, int(math.ceil(float(trim_fraction) * len(measured))))
    prev_rmse = float("inf")
    inlier_rmse = 0.0

    for _ in range(max(1, int(max_iterations))):
        transformed_prior = scale * (prior @ rotation.T) + translation
        tree = cKDTree(transformed_prior)
        dists, nearest_prior_idx = tree.query(measured, k=1)
        dists = np.asarray(dists, dtype=np.float64)
        nearest_prior_idx = np.asarray(nearest_prior_idx, dtype=np.int64)

        if keep_count < len(measured):
            order = np.argsort(dists, kind="stable")[:keep_count]
        else:
            order = np.arange(len(measured), dtype=np.int64)

        matched_prior = prior[nearest_prior_idx[order]]
        matched_measured = measured[order]
        scale, rotation, translation = _umeyama_sim3(
            matched_prior,
            matched_measured,
            allow_scale=allow_scale,
            scale_bounds=scale_bounds,
        )
        aligned_subset = scale * (matched_prior @ rotation.T) + translation
        residuals = np.linalg.norm(aligned_subset - matched_measured, axis=1)
        inlier_rmse = float(np.sqrt(np.mean(residuals**2)))
        if abs(prev_rmse - inlier_rmse) <= tolerance_m:
            break
        prev_rmse = inlier_rmse

    anchored_points = scale * (prior @ rotation.T) + translation
    sim_mat = np.eye(4, dtype=np.float64)
    sim_mat[:3, :3] = scale * rotation
    sim_mat[:3, 3] = translation

    rigid_mat = np.eye(4, dtype=np.float64)
    rigid_mat[:3, :3] = rotation
    rigid_mat[:3, 3] = translation

    result = Sim3AnchorResult(
        scale=float(scale),
        rotation=rotation,
        translation_m=translation,
        similarity_matrix=sim_mat,
        rigid_matrix=rigid_mat,
        inlier_count=int(keep_count),
        inlier_rmse_m=float(inlier_rmse),
        trimmed_fraction=float(keep_count) / float(len(measured)),
    )
    return anchored_points, result


def transfer_measured_rgb_to_prior(
    prior_points_m: Any,
    measured_points_m: Any,
    measured_rgb: Any,
    *,
    k_neighbors: int = 3,
) -> np.ndarray:
    """Transfer calibrated sensor RGB appearance onto a metric-anchored 3D shape prior."""
    prior = np.asarray(prior_points_m, dtype=np.float64)
    measured = np.asarray(measured_points_m, dtype=np.float64)
    colors = np.asarray(measured_rgb, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != 3 or not np.isfinite(prior).all():
        raise ValueError("prior_points_m must be finite Nx3")
    if (
        measured.ndim != 2
        or measured.shape[1] != 3
        or len(measured) == 0
        or colors.shape != measured.shape
    ):
        raise ValueError("measured_points_m and measured_rgb must be matching non-empty Mx3")
    if len(prior) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    k = max(1, min(int(k_neighbors), len(measured)))
    tree = cKDTree(measured)
    dists, idx = tree.query(prior, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dists, 1e-4)
    weights /= np.sum(weights, axis=1, keepdims=True)
    blended = np.sum(colors[idx] * weights[:, :, None], axis=1)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def build_prepared_scene_object_prior(
    object_id: str,
    prior_points: Any,
    measured_front_points: Any,
    *,
    prior_source: str = "sam3d_objects",
    measured_rgb: Any | None = None,
    faces: Any | None = None,
    correspondences: np.ndarray | None = None,
    trim_fraction: float = 0.85,
    allow_scale: bool = True,
    gauge_id: str = "object_gauge",
) -> PreparedSceneObjectPrior:
    """Construct a Stage-0 metric-anchored complete 3D object prior for online tracking."""
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("non-empty object_id required")
    if not isinstance(prior_source, str) or not prior_source.strip():
        raise ValueError("non-empty prior_source required")

    anchored_pts, anchor = anchor_shape_prior_to_depth(
        prior_points,
        measured_front_points,
        correspondences=correspondences,
        trim_fraction=trim_fraction,
        allow_scale=allow_scale,
    )
    colors_out = None
    if measured_rgb is not None:
        colors_out = transfer_measured_rgb_to_prior(
            anchored_pts, measured_front_points, measured_rgb
        )
    faces_out = None
    if faces is not None:
        faces_arr = np.asarray(faces, dtype=np.int64)
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("faces must be Fx3 triangle indices")
        faces_out = faces_arr

    return PreparedSceneObjectPrior(
        object_id=object_id.strip(),
        prior_source=prior_source.strip(),
        points_object_m=anchored_pts,
        colors_rgb=colors_out,
        faces=faces_out,
        anchor=anchor,
        gauge_id=gauge_id,
    )


def fuse_prior_with_measured_surface(
    measured_points: Any,
    measured_support: Any,
    measured_cameras: Any,
    prior: PreparedSceneObjectPrior | np.ndarray,
    *,
    replace_radius_m: float = 0.012,
    minimum_measured_support: int = 2,
    maximum_point_count: int = 1500,
) -> dict[str, Any]:
    """Fuse live RevisableSurfelMap surfels with a Stage-0 metric-anchored 3D prior.

    Evidence-First Replacement Rule:
      1. Every measured surfel with ``support >= minimum_measured_support`` is
         admitted first and defines an observed ball of radius ``replace_radius_m``
         (default 12 mm, matching the coarse12mm cell diameter).
      2. Prior points inside any observed ball are superseded by the real multi-view
         sensor surfels.
      3. Prior points outside observed balls (unseen back side, self-occluded walls)
         are retained with ``observation_support = 0``, ``camera_support = 0``, and
         ``is_prior = True`` so the 3D scene remains complete while downstream
         estimators can strictly distinguish measured evidence from Stage-0 prior.
    """
    if not math.isfinite(replace_radius_m) or replace_radius_m <= 0.0:
        raise ValueError("replace_radius_m must be finite and positive")
    if type(minimum_measured_support) is not int or minimum_measured_support < 1:
        raise ValueError("minimum_measured_support must be a positive integer")
    if type(maximum_point_count) is not int or maximum_point_count < 1:
        raise ValueError("maximum_point_count must be a positive integer")

    meas_xyz = np.asarray(measured_points, dtype=np.float64)
    meas_sup = np.asarray(measured_support, dtype=np.int64)
    meas_cam = np.asarray(measured_cameras, dtype=np.int64)
    if (
        meas_xyz.ndim != 2
        or meas_xyz.shape[1] != 3
        or meas_sup.shape != (len(meas_xyz),)
        or meas_cam.shape != (len(meas_xyz),)
    ):
        raise ValueError("measured_points, measured_support, and measured_cameras must align")

    prior_xyz = (
        prior.points_object_m
        if isinstance(prior, PreparedSceneObjectPrior)
        else np.asarray(prior, dtype=np.float64)
    )
    if prior_xyz.ndim != 2 or prior_xyz.shape[1] != 3 or not np.isfinite(prior_xyz).all():
        raise ValueError("prior must supply finite Nx3 points")

    valid_meas = meas_sup >= minimum_measured_support
    kept_meas_xyz = meas_xyz[valid_meas]
    kept_meas_sup = meas_sup[valid_meas]
    kept_meas_cam = meas_cam[valid_meas]

    # Order measured points by (-cameras, -support)
    if len(kept_meas_xyz) > 0:
        order = np.lexsort((-kept_meas_sup, -kept_meas_cam))
        kept_meas_xyz = kept_meas_xyz[order]
        kept_meas_sup = kept_meas_sup[order]
        kept_meas_cam = kept_meas_cam[order]

    if len(kept_meas_xyz) >= maximum_point_count:
        kept_meas_xyz = kept_meas_xyz[:maximum_point_count]
        kept_meas_sup = kept_meas_sup[:maximum_point_count]
        kept_meas_cam = kept_meas_cam[:maximum_point_count]
        retained_prior_xyz = np.empty((0, 3), dtype=np.float64)
        superseded_count = len(prior_xyz)
    elif len(kept_meas_xyz) == 0:
        retained_prior_xyz = prior_xyz[:maximum_point_count].copy()
        superseded_count = 0
    else:
        tree = cKDTree(kept_meas_xyz)
        dists, _ = tree.query(prior_xyz, k=1, distance_upper_bound=replace_radius_m)
        unobserved_mask = np.isinf(dists)
        superseded_count = int(np.count_nonzero(~unobserved_mask))
        unobserved_prior = prior_xyz[unobserved_mask]
        remaining_budget = maximum_point_count - len(kept_meas_xyz)
        if len(unobserved_prior) > remaining_budget:
            # Uniformly stride unobserved prior points to cover the unseen back surface
            idx = np.linspace(0, len(unobserved_prior) - 1, remaining_budget, dtype=np.int64)
            retained_prior_xyz = unobserved_prior[idx]
        else:
            retained_prior_xyz = unobserved_prior

    n_meas = len(kept_meas_xyz)
    n_prior = len(retained_prior_xyz)
    fused_xyz = (
        np.vstack([kept_meas_xyz, retained_prior_xyz])
        if n_prior > 0 and n_meas > 0
        else kept_meas_xyz.copy()
        if n_meas > 0
        else retained_prior_xyz.copy()
    )
    fused_sup = np.concatenate([kept_meas_sup, np.zeros(n_prior, dtype=np.int64)])
    fused_cam = np.concatenate([kept_meas_cam, np.zeros(n_prior, dtype=np.int64)])
    is_prior_mask = np.concatenate(
        [np.zeros(n_meas, dtype=bool), np.ones(n_prior, dtype=bool)]
    )

    return {
        "points_object_m": fused_xyz,
        "observation_support": fused_sup,
        "camera_support": fused_cam,
        "is_prior_mask": is_prior_mask,
        "measured_point_count": int(n_meas),
        "prior_points_retained": int(n_prior),
        "prior_points_superseded": int(superseded_count),
        "prior_source": (
            prior.prior_source
            if isinstance(prior, PreparedSceneObjectPrior)
            else "custom_prior"
        ),
        "replace_radius_m": float(replace_radius_m),
        "publication_authorized": False,
    }


__all__ = [
    "PreparedSceneObjectPrior",
    "Sim3AnchorResult",
    "anchor_shape_prior_to_depth",
    "build_prepared_scene_object_prior",
    "fuse_prior_with_measured_surface",
    "transfer_measured_rgb_to_prior",
]
