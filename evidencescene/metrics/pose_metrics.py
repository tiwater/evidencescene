"""Standard 6D object pose tracking evaluation metrics (ADD, ADD-S, AUC, determinism spread).

Implements the Frame-0 gauge-aligned, failure-inclusive evaluation protocol from
EvidenceScene (Section III-A & Section V).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PoseSequenceScorecard:
    """Comprehensive failure-inclusive 6D pose tracking scorecard."""

    total_frames: int
    denominator_excluding_initial: int
    available_post_initial_count: int
    candidate_duty_cycle: float
    mean_add_m: float | None
    mean_adds_m: float | None
    mean_translation_error_m: float | None
    mean_rotation_error_deg: float | None
    add_le_threshold_count: int
    adds_le_threshold_count: int
    strict_threshold_count: int
    add_recall_fraction: float
    adds_recall_fraction: float
    strict_recall_fraction: float
    add_auc_0_to_max: float
    adds_auc_0_to_max: float
    threshold_m: float
    strict_rotation_deg: float
    auc_max_threshold_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "denominator_excluding_initial": self.denominator_excluding_initial,
            "available_post_initial_count": self.available_post_initial_count,
            "candidate_duty_cycle": self.candidate_duty_cycle,
            "mean_add_m": self.mean_add_m,
            "mean_adds_m": self.mean_adds_m,
            "mean_translation_error_m": self.mean_translation_error_m,
            "mean_rotation_error_deg": self.mean_rotation_error_deg,
            "add_le_10mm_count": self.add_le_threshold_count,
            "adds_le_10mm_count": self.adds_le_threshold_count,
            "strict_10mm_10deg_count": self.strict_threshold_count,
            "add_recall_fraction": self.add_recall_fraction,
            "adds_recall_fraction": self.adds_recall_fraction,
            "strict_recall_fraction": self.strict_recall_fraction,
            "add_recall_auc_0_to_100mm": self.add_auc_0_to_max,
            "recall_auc_0_to_100mm": self.adds_auc_0_to_max,
            "threshold_m": self.threshold_m,
            "strict_rotation_deg": self.strict_rotation_deg,
            "auc_max_threshold_m": self.auc_max_threshold_m,
        }


@dataclass(frozen=True)
class TrajectoryDeterminismSpread:
    """Run-to-run trajectory spread across repeated tracker executions on identical inputs."""

    run_count: int
    frame_count: int
    mutually_available_frames: int
    availability_mismatch_frames: int
    max_translation_spread_mm: float
    mean_translation_spread_mm: float
    max_rotation_spread_deg: float
    mean_rotation_spread_deg: float
    is_deterministic: bool


def _valid_se3(pose: object) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        arr = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        arr.shape != (4, 4)
        or not np.isfinite(arr).all()
        or not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0.0)
        or not np.allclose(arr[:3, :3].T @ arr[:3, :3], np.eye(3), atol=1e-4, rtol=0.0)
        or not math.isclose(float(np.linalg.det(arr[:3, :3])), 1.0, abs_tol=1e-4)
    ):
        return None
    return arr


def _invert_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r_t = pose[:3, :3].T
    inv[:3, :3] = r_t
    inv[:3, 3] = -r_t @ pose[:3, 3]
    return inv


def align_initial_gauge(
    initial_estimated_pose: np.ndarray,
    initial_reference_pose: np.ndarray,
) -> np.ndarray:
    """Compute the Frame-0 right-multiplying gauge alignment G_0 = T_est,0^{-1} T_gt,0."""
    est = _valid_se3(initial_estimated_pose)
    ref = _valid_se3(initial_reference_pose)
    if est is None or ref is None:
        raise ValueError("initial_estimated_pose and initial_reference_pose must be valid 4x4 SE(3)")
    return _invert_se3(est) @ ref


def compute_add(
    pose_estimated: np.ndarray,
    pose_reference: np.ndarray,
    model_points_m: np.ndarray,
) -> float:
    """Compute Average Distance of Model Points (ADD) in meters."""
    est = _valid_se3(pose_estimated)
    ref = _valid_se3(pose_reference)
    if est is None or ref is None:
        raise ValueError("pose_estimated and pose_reference must be valid 4x4 SE(3)")
    pts = np.asarray(model_points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError("model_points_m must be a non-empty finite Nx3 array")
    pts_est = pts @ est[:3, :3].T + est[:3, 3]
    pts_ref = pts @ ref[:3, :3].T + ref[:3, 3]
    return float(np.mean(np.linalg.norm(pts_est - pts_ref, axis=1)))


def compute_adds(
    pose_estimated: np.ndarray,
    pose_reference: np.ndarray,
    model_points_m: np.ndarray,
) -> float:
    """Compute Average Closest Point Distance (ADD-S) in meters."""
    est = _valid_se3(pose_estimated)
    ref = _valid_se3(pose_reference)
    if est is None or ref is None:
        raise ValueError("pose_estimated and pose_reference must be valid 4x4 SE(3)")
    pts = np.asarray(model_points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError("model_points_m must be a non-empty finite Nx3 array")
    pts_est = pts @ est[:3, :3].T + est[:3, 3]
    pts_ref = pts @ ref[:3, :3].T + ref[:3, 3]
    tree = cKDTree(pts_ref)
    dists, _ = tree.query(pts_est, k=1)
    return float(np.mean(dists))


def compute_geodesic_rotation_error_deg(
    pose_estimated: np.ndarray,
    pose_reference: np.ndarray,
) -> float:
    """Compute geodesic rotation error on SO(3) in degrees."""
    est = _valid_se3(pose_estimated)
    ref = _valid_se3(pose_reference)
    if est is None or ref is None:
        raise ValueError("pose_estimated and pose_reference must be valid 4x4 SE(3)")
    cos_val = (float(np.trace(est[:3, :3].T @ ref[:3, :3])) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def compute_translation_error_m(
    pose_estimated: np.ndarray,
    pose_reference: np.ndarray,
) -> float:
    """Compute Euclidean translation error in meters."""
    est = _valid_se3(pose_estimated)
    ref = _valid_se3(pose_reference)
    if est is None or ref is None:
        raise ValueError("pose_estimated and pose_reference must be valid 4x4 SE(3)")
    return float(np.linalg.norm(est[:3, 3] - ref[:3, 3]))


def compute_failure_inclusive_recall_auc(
    valid_errors_m: Sequence[float] | np.ndarray,
    *,
    denominator: int,
    max_threshold_m: float = 0.100,
) -> float:
    """Compute exact continuous failure-inclusive recall AUC over [0, max_threshold_m].

    Unavailable frames are accounted for in ``denominator`` and contribute zero
    recall at all thresholds.
    """
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    if not math.isfinite(max_threshold_m) or max_threshold_m <= 0.0:
        raise ValueError("max_threshold_m must be finite and positive")
    errs = np.asarray(valid_errors_m, dtype=np.float64)
    if errs.ndim != 1:
        raise ValueError("valid_errors_m must be a 1D sequence")
    if len(errs) > denominator:
        raise ValueError("len(valid_errors_m) cannot exceed denominator")
    finite = errs[np.isfinite(errs) & (errs >= 0.0) & (errs <= max_threshold_m)]
    if finite.size == 0:
        return 0.0
    contributions = 1.0 - (finite / float(max_threshold_m))
    return float(np.sum(contributions) / float(denominator))


def evaluate_pose_sequence(
    estimated_poses: Sequence[np.ndarray | None],
    reference_poses: Sequence[np.ndarray],
    model_points_m: np.ndarray,
    *,
    align_frame_zero: bool = True,
    exclude_frame_zero: bool = True,
    threshold_m: float = 0.010,
    strict_rotation_deg: float = 10.0,
    auc_max_threshold_m: float = 0.100,
) -> PoseSequenceScorecard:
    """Evaluate a 6D pose trajectory against reference poses using the EvidenceScene protocol."""
    if len(estimated_poses) != len(reference_poses):
        raise ValueError("estimated_poses and reference_poses must have equal length")
    n_total = len(reference_poses)
    min_len = 2 if exclude_frame_zero else 1
    if n_total < min_len:
        raise ValueError(f"sequence requires at least {min_len} frames")

    pts = np.asarray(model_points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0 or not np.isfinite(pts).all():
        raise ValueError("model_points_m must be a non-empty finite Nx3 array")

    gauge = np.eye(4, dtype=np.float64)
    if align_frame_zero:
        est0 = _valid_se3(estimated_poses[0])
        ref0 = _valid_se3(reference_poses[0])
        if est0 is None or ref0 is None:
            raise ValueError("Frame 0 must have valid estimated and reference SE(3) poses for gauge alignment")
        gauge = align_initial_gauge(est0, ref0)

    start_idx = 1 if exclude_frame_zero else 0
    denominator = n_total - start_idx

    add_errors: list[float] = []
    adds_errors: list[float] = []
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    strict_hits = 0

    # Build model KD-tree once in object coordinates using SE(3) Euclidean isometry:
    # ||(R_est p_i + t_est) - (R_ref p_j + t_ref)||_2 == ||(R_rel p_i + t_rel) - p_j||_2
    tree_model = cKDTree(pts)

    for idx in range(start_idx, n_total):
        ref = _valid_se3(reference_poses[idx])
        if ref is None:
            raise ValueError(f"reference_poses[{idx}] is not a valid SE(3) matrix")
        est_raw = _valid_se3(estimated_poses[idx])
        if est_raw is None:
            continue
        est = est_raw @ gauge
        r_ref_t = ref[:3, :3].T
        dt_cam = est[:3, 3] - ref[:3, 3]
        r_rel = r_ref_t @ est[:3, :3]
        t_rel = r_ref_t @ dt_cam
        pts_rel = pts @ r_rel.T + t_rel
        diff = pts_rel - pts
        add_val = float(np.mean(np.sqrt(np.einsum("ij,ij->i", diff, diff))))
        dists, _ = tree_model.query(pts_rel, k=1)
        adds_val = float(np.mean(dists))
        t_err = float(np.linalg.norm(dt_cam))
        cos_val = (float(np.trace(r_rel)) - 1.0) * 0.5
        r_err = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
        add_errors.append(add_val)
        adds_errors.append(adds_val)
        trans_errors.append(t_err)
        rot_errors.append(r_err)
        if t_err <= threshold_m and r_err <= strict_rotation_deg:
            strict_hits += 1

    avail_count = len(add_errors)
    add_hits = sum(1 for e in add_errors if e <= threshold_m)
    adds_hits = sum(1 for e in adds_errors if e <= threshold_m)

    add_auc = compute_failure_inclusive_recall_auc(
        add_errors, denominator=denominator, max_threshold_m=auc_max_threshold_m
    )
    adds_auc = compute_failure_inclusive_recall_auc(
        adds_errors, denominator=denominator, max_threshold_m=auc_max_threshold_m
    )

    return PoseSequenceScorecard(
        total_frames=n_total,
        denominator_excluding_initial=denominator,
        available_post_initial_count=avail_count,
        candidate_duty_cycle=float(avail_count) / float(denominator),
        mean_add_m=float(np.mean(add_errors)) if add_errors else None,
        mean_adds_m=float(np.mean(adds_errors)) if adds_errors else None,
        mean_translation_error_m=float(np.mean(trans_errors)) if trans_errors else None,
        mean_rotation_error_deg=float(np.mean(rot_errors)) if rot_errors else None,
        add_le_threshold_count=add_hits,
        adds_le_threshold_count=adds_hits,
        strict_threshold_count=strict_hits,
        add_recall_fraction=float(add_hits) / float(denominator),
        adds_recall_fraction=float(adds_hits) / float(denominator),
        strict_recall_fraction=float(strict_hits) / float(denominator),
        add_auc_0_to_max=add_auc,
        adds_auc_0_to_max=adds_auc,
        threshold_m=threshold_m,
        strict_rotation_deg=strict_rotation_deg,
        auc_max_threshold_m=auc_max_threshold_m,
    )


def compute_trajectory_determinism_spread(
    trajectory_runs: Sequence[Sequence[np.ndarray | None]],
    *,
    tolerance_mm: float = 1e-6,
    tolerance_deg: float = 1e-6,
) -> TrajectoryDeterminismSpread:
    """Measure run-to-run translational (mm) and rotational (deg) spread across repeated runs."""
    if len(trajectory_runs) < 2:
        raise ValueError("at least 2 trajectory runs are required to measure determinism spread")
    n_frames = len(trajectory_runs[0])
    if n_frames == 0 or any(len(run) != n_frames for run in trajectory_runs):
        raise ValueError("all trajectory runs must be non-empty and share the same frame count")

    trans_spreads_mm: list[float] = []
    rot_spreads_deg: list[float] = []
    mutually_available = 0
    mismatch_frames = 0

    for f_idx in range(n_frames):
        valid_poses = [_valid_se3(run[f_idx]) for run in trajectory_runs]
        present = [p for p in valid_poses if p is not None]
        if len(present) == 0:
            continue
        if len(present) != len(trajectory_runs):
            mismatch_frames += 1
            continue
        mutually_available += 1
        max_t_mm = 0.0
        max_r_deg = 0.0
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                t_mm = compute_translation_error_m(present[i], present[j]) * 1000.0
                r_deg = compute_geodesic_rotation_error_deg(present[i], present[j])
                if t_mm > max_t_mm:
                    max_t_mm = t_mm
                if r_deg > max_r_deg:
                    max_r_deg = r_deg
        trans_spreads_mm.append(max_t_mm)
        rot_spreads_deg.append(max_r_deg)

    max_t = float(max(trans_spreads_mm)) if trans_spreads_mm else 0.0
    mean_t = float(np.mean(trans_spreads_mm)) if trans_spreads_mm else 0.0
    max_r = float(max(rot_spreads_deg)) if rot_spreads_deg else 0.0
    mean_r = float(np.mean(rot_spreads_deg)) if rot_spreads_deg else 0.0
    deterministic = (
        mismatch_frames == 0
        and max_t <= tolerance_mm
        and max_r <= tolerance_deg
    )

    return TrajectoryDeterminismSpread(
        run_count=len(trajectory_runs),
        frame_count=n_frames,
        mutually_available_frames=mutually_available,
        availability_mismatch_frames=mismatch_frames,
        max_translation_spread_mm=max_t,
        mean_translation_spread_mm=mean_t,
        max_rotation_spread_deg=max_r,
        mean_rotation_spread_deg=mean_r,
        is_deterministic=deterministic,
    )
