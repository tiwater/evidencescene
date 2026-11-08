"""Bidirectional 3D surface reconstruction metrics (P_eps, C_eps, F_95, R_95, Chamfer).

Implements the 4 mm grid-quantized directional accuracy and coverage evaluation
protocol from EvidenceScene (Section III-A & Section V), without ICP drift masking:
    P_eps = |{s in S : min_{m in M} ||s - m||_2 <= eps}| / |S|
    C_eps = |{m in M : min_{s in S} ||m - s||_2 <= eps}| / |M|
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from evidencescene.surface.object_surface_fusion import observation_voxel_centroids


@dataclass(frozen=True)
class SurfaceReconstructionScorecard:
    """Bidirectional surface reconstruction metrics on a common evaluation grid."""

    reconstructed_cell_count: int
    reference_cell_count: int
    threshold_m: float
    grid_voxel_size_m: float | None
    precision_pct: float
    coverage_pct: float
    f1_pct: float
    forward_p95_m: float
    reverse_p95_m: float
    forward_p95_mm: float
    reverse_p95_mm: float
    forward_mean_m: float
    reverse_mean_m: float
    chamfer_l1_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "reconstructed_cell_count": self.reconstructed_cell_count,
            "reference_cell_count": self.reference_cell_count,
            "threshold_m": self.threshold_m,
            "grid_voxel_size_m": self.grid_voxel_size_m,
            "P_eps_pct": self.precision_pct,
            "C_eps_pct": self.coverage_pct,
            "F1_eps_pct": self.f1_pct,
            "F_95_m": self.forward_p95_m,
            "R_95_m": self.reverse_p95_m,
            "F_95_mm": self.forward_p95_mm,
            "R_95_mm": self.reverse_p95_mm,
            "forward_mean_m": self.forward_mean_m,
            "reverse_mean_m": self.reverse_mean_m,
            "chamfer_l1_m": self.chamfer_l1_m,
        }


def quantize_surface_grid(
    points_m: np.ndarray,
    *,
    voxel_size_m: float = 0.004,
) -> np.ndarray:
    """Quantize a 3D point cloud onto a common voxel centroid grid."""
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("points_m must be a finite Nx3 array")
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    _keys, centroids = observation_voxel_centroids(pts, voxel_size_m)
    return centroids


def evaluate_surface_reconstruction(
    reconstructed_points_m: np.ndarray,
    reference_points_m: np.ndarray,
    *,
    threshold_m: float = 0.010,
    grid_voxel_size_m: float | None = 0.004,
) -> SurfaceReconstructionScorecard:
    """Evaluate directional precision P_eps, coverage C_eps, and p95 tails F_95 / R_95.

    Deleting uncertain points can inflate P_eps while collapsing C_eps; reporting
    both directions simultaneously prevents gaming the metric.
    """
    if not math.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be finite and positive")
    rec_raw = np.asarray(reconstructed_points_m, dtype=np.float64)
    ref_raw = np.asarray(reference_points_m, dtype=np.float64)
    if rec_raw.ndim != 2 or rec_raw.shape[1] != 3 or not np.isfinite(rec_raw).all():
        raise ValueError("reconstructed_points_m must be a finite Nx3 array")
    if ref_raw.ndim != 2 or ref_raw.shape[1] != 3 or len(ref_raw) == 0 or not np.isfinite(ref_raw).all():
        raise ValueError("reference_points_m must be a non-empty finite Mx3 array")

    if grid_voxel_size_m is not None:
        if not math.isfinite(grid_voxel_size_m) or grid_voxel_size_m <= 0.0:
            raise ValueError("grid_voxel_size_m must be finite and positive when provided")
        S = quantize_surface_grid(rec_raw, voxel_size_m=grid_voxel_size_m)
        M = quantize_surface_grid(ref_raw, voxel_size_m=grid_voxel_size_m)
    else:
        S = rec_raw.copy()
        M = ref_raw.copy()

    if len(S) == 0:
        return SurfaceReconstructionScorecard(
            reconstructed_cell_count=0,
            reference_cell_count=len(M),
            threshold_m=threshold_m,
            grid_voxel_size_m=grid_voxel_size_m,
            precision_pct=0.0,
            coverage_pct=0.0,
            f1_pct=0.0,
            forward_p95_m=math.inf,
            reverse_p95_m=math.inf,
            forward_p95_mm=math.inf,
            reverse_p95_mm=math.inf,
            forward_mean_m=math.inf,
            reverse_mean_m=math.inf,
            chamfer_l1_m=math.inf,
        )

    tree_m = cKDTree(M)
    tree_s = cKDTree(S)
    d_s_to_m, _ = tree_m.query(S, k=1)
    d_m_to_s, _ = tree_s.query(M, k=1)

    p_eps = float(np.mean(d_s_to_m <= threshold_m)) * 100.0
    c_eps = float(np.mean(d_m_to_s <= threshold_m)) * 100.0
    f1_eps = (2.0 * p_eps * c_eps / (p_eps + c_eps)) if (p_eps + c_eps) > 0.0 else 0.0

    f_95 = float(np.quantile(d_s_to_m, 0.95))
    r_95 = float(np.quantile(d_m_to_s, 0.95))
    f_mean = float(np.mean(d_s_to_m))
    r_mean = float(np.mean(d_m_to_s))
    chamfer = 0.5 * (f_mean + r_mean)

    return SurfaceReconstructionScorecard(
        reconstructed_cell_count=len(S),
        reference_cell_count=len(M),
        threshold_m=threshold_m,
        grid_voxel_size_m=grid_voxel_size_m,
        precision_pct=p_eps,
        coverage_pct=c_eps,
        f1_pct=f1_eps,
        forward_p95_m=f_95,
        reverse_p95_m=r_95,
        forward_p95_mm=f_95 * 1000.0,
        reverse_p95_mm=r_95 * 1000.0,
        forward_mean_m=f_mean,
        reverse_mean_m=r_mean,
        chamfer_l1_m=chamfer,
    )
