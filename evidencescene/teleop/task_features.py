"""3D functional task-feature geometry, rim circle consensus, and SE(3) covariance propagation.

Provides first-principles geometric fitting and uncertainty propagation for:
- Point task features (e.g., ``pour_outlet`` / spout tip position and pouring axis).
- Circular aperture features (e.g., ``container_opening`` / target cup rim center,
  plane normal, inner radius, and azimuthal sector coverage).
- First-order SE(3) Jacobian covariance propagation from object frame to world/table frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _unit_vector(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValueError(f"{name} must have nonzero norm")
    out = arr / norm
    out.flags.writeable = False
    return out


def _orthonormal_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _unit_vector(normal, "normal")
    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, anchor))) > 0.85:
        anchor = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, anchor)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))
    return u, v


@dataclass(frozen=True)
class MeasuredPointFeature3D:
    """A 3D functional point feature (such as ``pour_outlet``) with uncertainty."""

    feature_id: str
    point_m: np.ndarray
    direction: np.ndarray | None
    covariance_3x3_m2: np.ndarray
    direction_std_rad: float
    support_pixel_count: int
    timestamp_ns: int
    coordinate_frame: str = "table"

    @property
    def position_std_m(self) -> float:
        eigvals = np.linalg.eigvalsh(self.covariance_3x3_m2)
        return float(math.sqrt(max(0.0, float(np.max(eigvals)))))


@dataclass(frozen=True)
class MeasuredCircularAperture3D:
    """A 3D circular container opening (such as ``container_opening``) with uncertainty."""

    feature_id: str
    center_m: np.ndarray
    normal: np.ndarray
    radius_m: float
    radius_std_m: float
    center_covariance_3x3_m2: np.ndarray
    plane_inlier_rmse_m: float
    sector_coverage_ratio: float
    support_point_count: int
    timestamp_ns: int
    coordinate_frame: str = "table"
    ransac_consensus_agreed: bool = True

    @property
    def center_std_m(self) -> float:
        eigvals = np.linalg.eigvalsh(self.center_covariance_3x3_m2)
        return float(math.sqrt(max(0.0, float(np.max(eigvals)))))


@dataclass(frozen=True)
class PointToApertureRelation3D:
    """Relative geometric state between a functional point feature and a circular aperture."""

    signed_height_above_aperture_m: float
    planar_radial_offset_m: float
    radial_clearance_margin_m: float
    combined_radial_uncertainty_95_m: float
    worst_case_clearance_margin_95_m: float
    direction_alignment_deg: float | None
    projected_point_on_aperture_plane_m: np.ndarray
    planar_offset_vector_m: np.ndarray


def fit_robust_plane_3d(
    points_m: np.ndarray,
    *,
    reference_up: Sequence[float] = (0.0, 0.0, 1.0),
    inlier_threshold_m: float = 0.008,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a robust 3D plane (centroid, unit normal, inlier mask, inlier RMSE)."""
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("points_m must be a finite Nx3 array with at least 3 points")
    up = _unit_vector(reference_up, "reference_up")

    weights = np.ones(len(pts), dtype=np.float64)
    centroid = pts.mean(axis=0)
    normal = up.copy()

    for _ in range(max(1, int(iterations))):
        w_sum = float(np.sum(weights))
        if w_sum <= 1e-12:
            break
        centroid = np.sum(pts * weights[:, None], axis=0) / w_sum
        centered = pts - centroid
        cov = (centered * weights[:, None]).T @ centered / w_sum
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        if float(np.dot(normal, up)) < 0.0:
            normal = -normal
        residuals = np.abs((pts - centroid) @ normal)
        weights = np.where(
            residuals <= inlier_threshold_m,
            1.0,
            inlier_threshold_m / np.maximum(residuals, 1e-9),
        )

    residuals = np.abs((pts - centroid) @ normal)
    inliers = residuals <= inlier_threshold_m
    if int(np.count_nonzero(inliers)) >= 3:
        inlier_pts = pts[inliers]
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if float(np.dot(normal, up)) < 0.0:
            normal = -normal
        residuals = np.abs((pts - centroid) @ normal)
        inliers = residuals <= inlier_threshold_m

    inlier_res = residuals[inliers] if np.any(inliers) else residuals
    rmse = float(math.sqrt(float(np.mean(np.square(inlier_res)))))
    return centroid, _unit_vector(normal, "normal"), inliers, rmse


def _kasa_circle_fit_2d(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic Kasa circle fit in 2D returning (center_xy, radius, radial_rmse)."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("xy must have at least 3 2D points")
    design = np.column_stack([2.0 * pts[:, 0], 2.0 * pts[:, 1], np.ones(len(pts))])
    rhs = np.sum(np.square(pts), axis=1)
    sol, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3 or not np.isfinite(sol).all():
        raise ValueError("collinear or degenerate 2D points for circle fit")
    center = sol[:2]
    rad_sq = float(sol[2] + np.dot(center, center))
    if rad_sq <= 1e-10:
        raise ValueError("nonpositive fitted circle radius")
    radius = math.sqrt(rad_sq)
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    rmse = float(math.sqrt(float(np.mean(np.square(dists - radius)))))
    return center, radius, rmse


@lru_cache(maxsize=32)
def _circle_consensus_triplets(n: int, max_hypotheses: int) -> np.ndarray:
    """Deterministic triplets for the rim consensus check.

    The original loop drew one ``Generator.choice`` per hypothesis.  Caching
    the resulting stream preserves that exact sequence while avoiding the
    per-call Python RNG overhead on repeated aperture fits.
    """
    rng = np.random.default_rng(seed=1729)
    count = max(1, int(max_hypotheses))
    indices = np.empty((count, 3), dtype=np.int64)
    for i in range(count):
        indices[i] = rng.choice(n, size=3, replace=False)
    indices.flags.writeable = False
    return indices


def compare_circle_consensus_3d(
    points_2d: np.ndarray,
    *,
    fitted_center_2d: np.ndarray,
    fitted_radius_m: float,
    inlier_threshold_m: float = 0.006,
    max_hypotheses: int = 256,
    max_center_discrepancy_m: float = 0.015,
    max_radius_discrepancy_m: float = 0.015,
) -> tuple[bool, np.ndarray, float]:
    """Deterministic 3-point RANSAC circle consensus check against algebraic fit.

    Hypotheses are evaluated in blocks so the residual matrix stays bounded
    for large rims.  The triplet stream, inlier test, strict tie-breaking and
    per-hypothesis RMSE are identical to the original scalar loop.
    """
    pts = np.asarray(points_2d, dtype=np.float64)
    n = len(pts)
    if n < 6:
        return False, fitted_center_2d, fitted_radius_m

    fitted_center = np.asarray(fitted_center_2d, dtype=np.float64)
    fitted_radius = float(fitted_radius_m)
    hypotheses = max(1, int(max_hypotheses))
    triplets = _circle_consensus_triplets(n, hypotheses)
    best_inliers = -1
    best_rmse = math.inf
    best_center: np.ndarray | None = None
    best_radius: float | None = None

    px = np.ascontiguousarray(pts[:, 0])
    py = np.ascontiguousarray(pts[:, 1])
    block_size = 256 if n <= 1024 else 64
    for block_start in range(0, hypotheses, block_size):
        idx = triplets[block_start:block_start + block_size]
        i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]
        ax, ay = px[i0], py[i0]
        bx, by = px[i1], py[i1]
        cx, cy = px[i2], py[i2]
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        valid = np.abs(d) >= 1e-10
        if not np.any(valid):
            continue
        # Drop degenerate triples before division so the vectorized circumcenter
        # uses the same arithmetic as the scalar path.
        ax, ay = ax[valid], ay[valid]
        bx, by = bx[valid], by[valid]
        cx, cy = cx[valid], cy[valid]
        d = d[valid]
        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        radii = np.sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy))
        keep = (radii >= 0.01) & (radii <= 0.35)
        if not np.any(keep):
            continue
        ux = ux[keep]
        uy = uy[keep]
        radii = radii[keep]
        dx = px[None, :] - ux[:, None]
        dy = py[None, :] - uy[:, None]
        residuals = np.abs(np.sqrt(dx * dx + dy * dy) - radii[:, None])
        inliers = residuals <= inlier_threshold_m
        counts = np.count_nonzero(inliers, axis=1)
        block_max = int(counts.max())
        if block_max < best_inliers:
            continue
        target_count = max(best_inliers, block_max)
        for j in np.flatnonzero(counts == target_count):
            j_idx = int(j)
            count = int(counts[j_idx])
            row = residuals[j_idx][inliers[j_idx]]
            rmse = float(np.mean(row)) if count > 0 else math.inf
            if count > best_inliers or (count == best_inliers and rmse < best_rmse):
                best_inliers = count
                best_rmse = rmse
                best_center = np.array([ux[j_idx], uy[j_idx]], dtype=np.float64)
                best_radius = float(radii[j_idx])

    if best_center is None:
        # No admissible hypothesis: preserve the original fitted values and
        # the original agreement check (zero discrepancy).
        best_center = fitted_center.copy()
        best_radius = fitted_radius

    center_diff = float(np.linalg.norm(best_center - fitted_center))
    radius_diff = abs(float(best_radius) - fitted_radius)
    agreed = (
        center_diff <= max_center_discrepancy_m
        and radius_diff <= max_radius_discrepancy_m
    )
    return agreed, best_center, best_radius

def fit_circular_aperture_3d(
    points_m: np.ndarray,
    *,
    feature_id: str = "container_opening",
    timestamp_ns: int = 1,
    coordinate_frame: str = "table",
    reference_up: Sequence[float] = (0.0, 0.0, 1.0),
    num_sectors: int = 8,
    minimum_sector_coverage_ratio: float = 0.5,
    minimum_radius_m: float = 0.015,
    maximum_radius_m: float = 0.25,
) -> MeasuredCircularAperture3D:
    """Fit a 3D circular aperture (center, normal, radius, uncertainty) from rim points."""
    centroid, normal, plane_inliers, plane_rmse = fit_robust_plane_3d(
        points_m, reference_up=reference_up
    )
    pts = np.asarray(points_m, dtype=np.float64)
    inlier_pts = pts[plane_inliers] if int(np.count_nonzero(plane_inliers)) >= 6 else pts
    if len(inlier_pts) < 6:
        raise ValueError("insufficient inlier points to fit circular aperture")

    u_axis, v_axis = _orthonormal_plane_basis(normal)
    rel = inlier_pts - centroid[None, :]
    xy = np.column_stack([rel @ u_axis, rel @ v_axis])

    # Sector-wise outer-rim quantile selection to suppress interior depth points
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    radii0 = np.linalg.norm(xy, axis=1)
    sector_edges = np.linspace(-math.pi, math.pi, num_sectors + 1)
    selected_xy_list: list[np.ndarray] = []
    occupied_sectors = 0
    for s_idx in range(num_sectors):
        in_sec = (angles >= sector_edges[s_idx]) & (
            angles < sector_edges[s_idx + 1] if s_idx + 1 < num_sectors else angles <= sector_edges[s_idx + 1]
        )
        if int(np.count_nonzero(in_sec)) >= 1:
            occupied_sectors += 1
            sec_pts = xy[in_sec]
            sec_r = radii0[in_sec]
            q70 = float(np.quantile(sec_r, 0.70))
            selected_xy_list.append(sec_pts[sec_r >= q70])

    coverage_ratio = occupied_sectors / float(num_sectors)
    if coverage_ratio < minimum_sector_coverage_ratio:
        raise ValueError(
            f"aperture azimuthal sector coverage {coverage_ratio:.2f} below {minimum_sector_coverage_ratio:.2f}"
        )

    rim_xy = np.concatenate(selected_xy_list, axis=0) if selected_xy_list else xy
    if len(rim_xy) < 6:
        rim_xy = xy

    center_2d, radius_m, radial_rmse = _kasa_circle_fit_2d(rim_xy)
    if not (minimum_radius_m <= radius_m <= maximum_radius_m):
        raise ValueError(f"fitted aperture radius {radius_m:.4f} m outside [{minimum_radius_m}, {maximum_radius_m}]")

    agreed, _, _ = compare_circle_consensus_3d(
        rim_xy,
        fitted_center_2d=center_2d,
        fitted_radius_m=radius_m,
    )

    center_3d = centroid + center_2d[0] * u_axis + center_2d[1] * v_axis
    center_3d.flags.writeable = False

    # Construct 3x3 center covariance in 3D coordinates
    in_plane_var = max((radial_rmse ** 2) / max(len(rim_xy), 4), 1e-6)
    normal_var = max(plane_rmse ** 2, 1e-6)
    basis_3x3 = np.column_stack([u_axis, v_axis, normal])
    cov_local = np.diag([in_plane_var, in_plane_var, normal_var])
    center_cov = basis_3x3 @ cov_local @ basis_3x3.T
    center_cov.flags.writeable = False

    radius_std = float(max(radial_rmse, 0.0015))
    return MeasuredCircularAperture3D(
        feature_id=feature_id,
        center_m=center_3d,
        normal=normal,
        radius_m=float(radius_m),
        radius_std_m=radius_std,
        center_covariance_3x3_m2=center_cov,
        plane_inlier_rmse_m=plane_rmse,
        sector_coverage_ratio=coverage_ratio,
        support_point_count=len(inlier_pts),
        timestamp_ns=int(timestamp_ns),
        coordinate_frame=coordinate_frame,
        ransac_consensus_agreed=agreed,
    )


def propagate_feature_uncertainty_se3(
    *,
    feature_object_frame: MeasuredPointFeature3D,
    world_from_object: np.ndarray,
    pose_covariance_6x6: np.ndarray,
    target_coordinate_frame: str = "table",
) -> MeasuredPointFeature3D:
    """Propagate a 3D point feature and its covariance from object frame to world/table frame.

    Uses the first-order right-trivialized SE(3) Jacobian:
        p_W = R * p_O + t
        J_p = [ -R * [p_O]_x,  R ]
        Sigma_{p,W} = J_p * Sigma_xi * J_p^T + R * Sigma_{p,O} * R^T
    """
    T = np.asarray(world_from_object, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("world_from_object must be a finite 4x4 SE(3) matrix")
    cov_xi = np.asarray(pose_covariance_6x6, dtype=np.float64)
    if cov_xi.shape != (6, 6) or not np.isfinite(cov_xi).all():
        raise ValueError("pose_covariance_6x6 must be a finite 6x6 covariance matrix")

    R = T[:3, :3]
    t = T[:3, 3]
    p_obj = np.asarray(feature_object_frame.point_m, dtype=np.float64)
    p_world = R @ p_obj + t
    p_world.flags.writeable = False

    J_p = np.zeros((3, 6), dtype=np.float64)
    J_p[:, :3] = -R @ _skew(p_obj)
    J_p[:, 3:] = R

    cov_world = (
        J_p @ cov_xi @ J_p.T
        + R @ np.asarray(feature_object_frame.covariance_3x3_m2, dtype=np.float64) @ R.T
    )
    cov_world = 0.5 * (cov_world + cov_world.T)
    cov_world.flags.writeable = False

    dir_world: np.ndarray | None = None
    dir_std = float(feature_object_frame.direction_std_rad)
    if feature_object_frame.direction is not None:
        d_obj = _unit_vector(feature_object_frame.direction, "direction")
        dir_world = _unit_vector(R @ d_obj, "direction_world")
        rot_var_trace = float(np.trace(cov_xi[:3, :3])) / 3.0
        dir_std = float(math.sqrt(dir_std * dir_std + max(0.0, rot_var_trace)))

    return MeasuredPointFeature3D(
        feature_id=feature_object_frame.feature_id,
        point_m=p_world,
        direction=dir_world,
        covariance_3x3_m2=cov_world,
        direction_std_rad=dir_std,
        support_pixel_count=feature_object_frame.support_pixel_count,
        timestamp_ns=feature_object_frame.timestamp_ns,
        coordinate_frame=target_coordinate_frame,
    )


def evaluate_point_to_aperture_relation(
    point_feature: MeasuredPointFeature3D,
    aperture: MeasuredCircularAperture3D,
) -> PointToApertureRelation3D:
    """Compute exact 3D clearance and alignment between a point feature and a circular aperture."""
    if point_feature.coordinate_frame != aperture.coordinate_frame:
        raise ValueError(
            f"coordinate frame mismatch: {point_feature.coordinate_frame} vs {aperture.coordinate_frame}"
        )
    diff = np.asarray(point_feature.point_m, dtype=np.float64) - np.asarray(
        aperture.center_m, dtype=np.float64
    )
    n = _unit_vector(aperture.normal, "aperture.normal")
    signed_height = float(np.dot(diff, n))
    planar_vec = diff - signed_height * n
    planar_vec.flags.writeable = False
    proj_point = np.asarray(aperture.center_m, dtype=np.float64) + planar_vec
    proj_point.flags.writeable = False

    radial_offset = float(np.linalg.norm(planar_vec))
    clearance_margin = float(aperture.radius_m - radial_offset)

    # Project combined 3D covariance onto the aperture tangent plane
    u_axis, v_axis = _orthonormal_plane_basis(n)
    P_uv = np.column_stack([u_axis, v_axis])  # 3x2
    cov_total_3d = (
        np.asarray(point_feature.covariance_3x3_m2, dtype=np.float64)
        + np.asarray(aperture.center_covariance_3x3_m2, dtype=np.float64)
    )
    cov_2d = P_uv.T @ cov_total_3d @ P_uv
    eigvals_2d = np.linalg.eigvalsh(cov_2d)
    # 95% confidence in 2D chi-square (df=2): sqrt(5.99146) = 2.4477468
    chi2_2d_95 = 2.447746830680816
    radial_unc_95 = float(
        chi2_2d_95 * math.sqrt(max(0.0, float(np.max(eigvals_2d))))
        + 1.959963984540054 * aperture.radius_std_m
    )
    worst_case_margin_95 = float(clearance_margin - radial_unc_95)

    alignment_deg: float | None = None
    if point_feature.direction is not None:
        d = _unit_vector(point_feature.direction, "point_feature.direction")
        cos_angle = float(np.clip(np.dot(d, -n), -1.0, 1.0))
        alignment_deg = math.degrees(math.acos(cos_angle))

    return PointToApertureRelation3D(
        signed_height_above_aperture_m=signed_height,
        planar_radial_offset_m=radial_offset,
        radial_clearance_margin_m=clearance_margin,
        combined_radial_uncertainty_95_m=radial_unc_95,
        worst_case_clearance_margin_95_m=worst_case_margin_95,
        direction_alignment_deg=alignment_deg,
        projected_point_on_aperture_plane_m=proj_point,
        planar_offset_vector_m=planar_vec,
    )
