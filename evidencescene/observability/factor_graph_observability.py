"""Causal factor-graph observability gates (F, G, R, M) for 6-DoF pose tracking.

Implements the mathematical formulations from EvidenceScene (Sections III-IV):
- Correctness Baseline (F): Frame-scoped pose attribution (f_kf == f_t) and
  solver-return consistency between refined inliers and returned SE(3) pose.
- Prior-Connected Graph Admission (G): Connected-component reachability from
  the anchored initial gauge prior X_0 before committing frontend map updates.
- Nuisance-Eliminated Marginal Rank & Schur-Complement Gate (R): Orthogonal
  left-null-space landmark elimination
      A_bar_x = (I - A_y A_y^+) A_x
  column-deletion rank difference
      r_i = rank(A_bar_x) - rank(A_bar_{x,-i}) in {0, ..., 6}
  across multi-scale SVD tolerances, plus the 3x3 marginal translational Schur
  complement precision block R_i.
- Feature-Birth Observation Ledger (M): Immutable ledger of camera-frame 3D
  birth measurements (obj_id, f_birth, f_admit, track_id, z_cam) with
  deduplicated birth factor emission upon landmark instantiation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CausalAttributionVerdict:
    """Result of the F-fork temporal attribution and solver-return check."""

    admitted: bool
    reason: str
    frame_index: int
    keyframe_index: int
    selected_pose: np.ndarray | None
    used_refined_pose: bool


@dataclass(frozen=True)
class PriorConnectivityReport:
    """Result of the G-fork prior-connected graph component analysis."""

    candidate_pose_id: int
    candidate_connected_to_prior: bool
    anchored_landmark_count_for_candidate: int
    admitted: bool
    reason: str
    anchored_pose_ids: tuple[int, ...]
    unanchored_pose_ids: tuple[int, ...]
    anchored_landmark_ids: tuple[int, ...]
    unanchored_landmark_ids: tuple[int, ...]


@dataclass(frozen=True)
class SchurObservabilityReport:
    """Result of the R-fork null-space rank and Schur-complement precision gate."""

    candidate_pose_index: int
    marginal_pose_rank: int
    rank_by_tolerance_scale: Mapping[str, int]
    full_reduced_rank: int
    reduced_rank_without_candidate: int
    translational_schur_block_3x3: np.ndarray
    translational_min_singular_value: float
    translational_condition_number: float
    pose_marginal_information_6x6: np.ndarray
    pose_min_singular_value: float
    pose_condition_number: float
    admitted: bool
    reason: str


@dataclass(frozen=True)
class BirthObservationRecord:
    """Immutable 3D sensor measurement captured when a feature track is born (M)."""

    object_id: str
    birth_frame_index: int
    track_id: int
    point_camera_m: tuple[float, float, float]
    uncertainty_m: float


@dataclass(frozen=True)
class BirthFactorCommit:
    """Deduplicated birth factor emitted when a landmark is instantiated (M)."""

    object_id: str
    birth_frame_index: int
    admit_frame_index: int
    track_id: int
    landmark_id: int
    point_camera_m: tuple[float, float, float]
    uncertainty_m: float


def _validate_se3(matrix: object, name: str = "pose") -> np.ndarray | None:
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        arr.shape != (4, 4)
        or not np.isfinite(arr).all()
        or not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0)
        or not np.allclose(arr[:3, :3].T @ arr[:3, :3], np.eye(3), atol=1e-5, rtol=0.0)
        or not math.isclose(float(np.linalg.det(arr[:3, :3])), 1.0, abs_tol=1e-5)
    ):
        return None
    return arr.copy()


def verify_causal_pose_attribution(
    *,
    current_frame_index: int,
    promoted_keyframe_index: int,
    raw_pose: object,
    refined_pose: object | None = None,
    raw_inlier_count: int = 0,
    refined_inlier_count: int = 0,
    raw_mean_residual_m: float = math.inf,
    refined_mean_residual_m: float = math.inf,
) -> CausalAttributionVerdict:
    """Enforce Fork F: frame-scoped pose attribution and refined-pose return consistency.

    1. Rejects historical keyframe promotions (f_kf != f_t) from overwriting the
       current frame's live pose estimate.
    2. Selects refined_pose when local refinement improved or maintained inlier
       support with non-increasing residual, eliminating the raw/refined return
       mismatch.
    """
    if type(current_frame_index) is not int or current_frame_index < 0:
        raise ValueError("current_frame_index must be a non-negative int")
    if type(promoted_keyframe_index) is not int or promoted_keyframe_index < 0:
        raise ValueError("promoted_keyframe_index must be a non-negative int")

    if promoted_keyframe_index != current_frame_index:
        return CausalAttributionVerdict(
            admitted=False,
            reason="historical_keyframe_attribution_mismatch",
            frame_index=current_frame_index,
            keyframe_index=promoted_keyframe_index,
            selected_pose=None,
            used_refined_pose=False,
        )

    valid_raw = _validate_se3(raw_pose, "raw_pose")
    valid_refined = _validate_se3(refined_pose, "refined_pose") if refined_pose is not None else None

    if valid_refined is not None and (
        refined_inlier_count > raw_inlier_count
        or (
            refined_inlier_count == raw_inlier_count
            and refined_mean_residual_m <= raw_mean_residual_m
        )
    ):
        return CausalAttributionVerdict(
            admitted=True,
            reason="admitted_refined_pose",
            frame_index=current_frame_index,
            keyframe_index=promoted_keyframe_index,
            selected_pose=valid_refined,
            used_refined_pose=True,
        )

    if valid_raw is not None:
        return CausalAttributionVerdict(
            admitted=True,
            reason="admitted_raw_pose",
            frame_index=current_frame_index,
            keyframe_index=promoted_keyframe_index,
            selected_pose=valid_raw,
            used_refined_pose=False,
        )

    return CausalAttributionVerdict(
        admitted=False,
        reason="invalid_se3_pose_hypothesis",
        frame_index=current_frame_index,
        keyframe_index=promoted_keyframe_index,
        selected_pose=None,
        used_refined_pose=False,
    )


def evaluate_prior_connected_admission(
    *,
    pose_ids: Sequence[int],
    landmark_ids: Sequence[int],
    prior_pose_ids: Sequence[int],
    pose_landmark_edges: Sequence[tuple[int, int]],
    pose_pose_edges: Sequence[tuple[int, int]] = (),
    candidate_pose_id: int,
    minimum_anchored_landmarks: int = 3,
) -> PriorConnectivityReport:
    """Enforce Fork G: prior-connected component admission on the pose-landmark graph.

    Finds the connected component C_0 containing the anchored initial pose prior(s)
    X_0 and checks whether candidate_pose_id belongs to C_0 and observes at least
    minimum_anchored_landmarks landmarks in C_0.
    """
    if minimum_anchored_landmarks < 0:
        raise ValueError("minimum_anchored_landmarks must be non-negative")
    pose_set = {int(p) for p in pose_ids}
    landmark_set = {int(l) for l in landmark_ids}
    prior_set = {int(p) for p in prior_pose_ids if int(p) in pose_set}
    cand = int(candidate_pose_id)

    pose_to_landmarks: dict[int, set[int]] = {p: set() for p in pose_set}
    landmark_to_poses: dict[int, set[int]] = {l: set() for l in landmark_set}
    pose_neighbors: dict[int, set[int]] = {p: set() for p in pose_set}

    for p_raw, l_raw in pose_landmark_edges:
        p, l = int(p_raw), int(l_raw)
        if p in pose_set and l in landmark_set:
            pose_to_landmarks[p].add(l)
            landmark_to_poses[l].add(p)

    for pa_raw, pb_raw in pose_pose_edges:
        pa, pb = int(pa_raw), int(pb_raw)
        if pa in pose_set and pb in pose_set and pa != pb:
            pose_neighbors[pa].add(pb)
            pose_neighbors[pb].add(pa)

    anchored_poses: set[int] = set(prior_set)
    anchored_landmarks: set[int] = set()
    queue: deque[tuple[str, int]] = deque(("pose", p) for p in sorted(prior_set))

    while queue:
        kind, node = queue.popleft()
        if kind == "pose":
            for nb_pose in pose_neighbors.get(node, ()):
                if nb_pose not in anchored_poses:
                    anchored_poses.add(nb_pose)
                    queue.append(("pose", nb_pose))
            for nb_lm in pose_to_landmarks.get(node, ()):
                if nb_lm not in anchored_landmarks:
                    anchored_landmarks.add(nb_lm)
                    queue.append(("landmark", nb_lm))
        else:
            for nb_pose in landmark_to_poses.get(node, ()):
                if nb_pose not in anchored_poses:
                    anchored_poses.add(nb_pose)
                    queue.append(("pose", nb_pose))

    connected = cand in anchored_poses
    cand_landmarks = pose_to_landmarks.get(cand, set())
    anchored_shared_count = len(cand_landmarks & anchored_landmarks)

    if not prior_set:
        admitted = False
        reason = "no_prior_anchored_poses"
    elif not connected:
        admitted = False
        reason = "disconnected_from_initial_prior_gauge"
    elif cand not in prior_set and anchored_shared_count < minimum_anchored_landmarks:
        admitted = False
        reason = "insufficient_anchored_landmark_support"
    else:
        admitted = True
        reason = "admitted_prior_connected"

    return PriorConnectivityReport(
        candidate_pose_id=cand,
        candidate_connected_to_prior=connected,
        anchored_landmark_count_for_candidate=anchored_shared_count,
        admitted=admitted,
        reason=reason,
        anchored_pose_ids=tuple(sorted(anchored_poses)),
        unanchored_pose_ids=tuple(sorted(pose_set - anchored_poses)),
        anchored_landmark_ids=tuple(sorted(anchored_landmarks)),
        unanchored_landmark_ids=tuple(sorted(landmark_set - anchored_landmarks)),
    )


def _numerical_rank(singular_values: np.ndarray, tol: float) -> int:
    if singular_values.size == 0:
        return 0
    return int(np.sum(singular_values > tol))


def evaluate_schur_observability_gate(
    *,
    pose_jacobian_ax: np.ndarray,
    landmark_jacobian_ay: np.ndarray | None = None,
    candidate_pose_index: int,
    required_pose_rank: int = 6,
    tolerance_scales: Sequence[float] = (0.1, 1.0, 10.0),
    minimum_translational_singular_value: float = 1e-4,
    maximum_translational_condition_number: float = 1e6,
) -> SchurObservabilityReport:
    """Enforce Fork R: nuisance-eliminated marginal pose rank and Schur precision gate.

    Given whitened linear system ||A_x dx + A_y dy - b||_2^2 with 6 columns per
    pose in A_x (ordered [rot_3, trans_3] or [trans_3, rot_3]; translation occupies
    columns 3:6 of each 6-column pose block, matching SE(3) [omega, v] convention):
    1. Projects A_x onto the left null space of A_y:
           A_bar_x = (I - A_y A_y^+) A_x
    2. Computes the column-deletion rank difference for pose i:
           r_i = rank(A_bar_x) - rank(A_bar_{x,-i}) in {0, ..., 6}
       across the specified tolerance scales {0.1x, 1x, 10x} * eps * max(shape) * sigma_max.
    3. Computes the 6x6 marginal pose information block S_i and 3x3 marginal
       translational Schur complement precision block R_i.
    """
    ax = np.asarray(pose_jacobian_ax, dtype=np.float64)
    if ax.ndim != 2 or ax.shape[1] == 0 or ax.shape[1] % 6 != 0 or not np.isfinite(ax).all():
        raise ValueError("pose_jacobian_ax must be a finite 2D array with 6*N_p columns")
    num_poses = ax.shape[1] // 6
    if type(candidate_pose_index) is not int or not 0 <= candidate_pose_index < num_poses:
        raise ValueError(f"candidate_pose_index must be in [0, {num_poses - 1}]")
    if not 1 <= required_pose_rank <= 6:
        raise ValueError("required_pose_rank must be in [1, 6]")

    # Step 1: Orthogonal left-null-space projection to eliminate uncertain landmarks
    if landmark_jacobian_ay is not None and np.asarray(landmark_jacobian_ay).size > 0:
        ay = np.asarray(landmark_jacobian_ay, dtype=np.float64)
        if ay.ndim != 2 or ay.shape[0] != ax.shape[0] or not np.isfinite(ay).all():
            raise ValueError("landmark_jacobian_ay must be finite with the same row count as pose_jacobian_ax")
        u_y, s_y, _vt_y = np.linalg.svd(ay, full_matrices=False)
        eps = float(np.finfo(np.float64).eps)
        tol_y = eps * max(ay.shape) * (float(s_y[0]) if s_y.size > 0 else 0.0)
        rank_y = _numerical_rank(s_y, tol_y)
        if rank_y > 0:
            u_active = u_y[:, :rank_y]
            a_bar_x = ax - u_active @ (u_active.T @ ax)
        else:
            a_bar_x = ax.copy()
    else:
        a_bar_x = ax.copy()

    # Step 2: Column-deletion rank difference r_i = rank(A_bar_x) - rank(A_bar_{x,-i})
    col_start = 6 * candidate_pose_index
    col_end = col_start + 6
    keep_cols = np.r_[0:col_start, col_end : a_bar_x.shape[1]]
    a_bar_without_i = a_bar_x[:, keep_cols]

    s_full = np.linalg.svd(a_bar_x, compute_uv=False) if a_bar_x.size > 0 else np.zeros(0)
    s_without = (
        np.linalg.svd(a_bar_without_i, compute_uv=False)
        if a_bar_without_i.size > 0
        else np.zeros(0)
    )
    sigma_max = float(s_full[0]) if s_full.size > 0 else 0.0
    base_tol = float(np.finfo(np.float64).eps) * max(a_bar_x.shape) * sigma_max

    rank_by_scale: dict[str, int] = {}
    default_full_rank = 0
    default_without_rank = 0
    for scale in tolerance_scales:
        scale_f = float(scale)
        tol = scale_f * base_tol
        r_full = _numerical_rank(s_full, tol)
        r_wo = _numerical_rank(s_without, tol)
        r_diff = max(0, min(6, r_full - r_wo))
        label = f"{scale_f:g}x"
        rank_by_scale[label] = r_diff
        if math.isclose(scale_f, 1.0, rel_tol=1e-9):
            default_full_rank = r_full
            default_without_rank = r_wo

    if "1x" not in rank_by_scale:
        default_full_rank = _numerical_rank(s_full, base_tol)
        default_without_rank = _numerical_rank(s_without, base_tol)
        rank_by_scale["1x"] = max(0, min(6, default_full_rank - default_without_rank))

    marginal_pose_rank = rank_by_scale["1x"]

    # Step 3: Marginal 6x6 pose information and 3x3 translational Schur complement R_i
    # Partition A_bar_x into candidate pose columns A_i (m x 6) and other pose columns A_{-i}
    a_i = a_bar_x[:, col_start:col_end]
    if a_bar_without_i.shape[1] > 0 and a_bar_without_i.shape[0] > 0:
        u_o, s_o, _vt_o = np.linalg.svd(a_bar_without_i, full_matrices=False)
        tol_o = float(np.finfo(np.float64).eps) * max(a_bar_without_i.shape) * (
            float(s_o[0]) if s_o.size > 0 else 0.0
        )
        rank_o = _numerical_rank(s_o, tol_o)
        if rank_o > 0:
            u_o_act = u_o[:, :rank_o]
            a_i_marginal = a_i - u_o_act @ (u_o_act.T @ a_i)
        else:
            a_i_marginal = a_i
    else:
        a_i_marginal = a_i

    h_ii_6x6 = a_i_marginal.T @ a_i_marginal
    h_ii_6x6 = 0.5 * (h_ii_6x6 + h_ii_6x6.T)

    # Extract 3x3 marginal translational precision block R_i (eliminating candidate rotation 0:3)
    h_rr = h_ii_6x6[:3, :3]
    h_rt = h_ii_6x6[:3, 3:6]
    h_tr = h_ii_6x6[3:6, :3]
    h_tt = h_ii_6x6[3:6, 3:6]
    r_i_3x3 = h_tt - h_tr @ np.linalg.pinv(h_rr, rcond=1e-12) @ h_rt
    r_i_3x3 = 0.5 * (r_i_3x3 + r_i_3x3.T)

    sv_t = np.linalg.svd(r_i_3x3, compute_uv=False)
    trans_min_sv = float(sv_t[-1])
    trans_max_sv = float(sv_t[0])
    trans_cond = (
        trans_max_sv / trans_min_sv
        if trans_min_sv > 1e-15
        else math.inf
    )

    sv_pose = np.linalg.svd(h_ii_6x6, compute_uv=False)
    pose_min_sv = float(sv_pose[-1])
    pose_max_sv = float(sv_pose[0])
    pose_cond = (
        pose_max_sv / pose_min_sv
        if pose_min_sv > 1e-15
        else math.inf
    )

    all_scales_full_rank = all(r >= required_pose_rank for r in rank_by_scale.values())
    if not all_scales_full_rank:
        admitted = False
        reason = f"rank_deficient_pose_update(r_i={marginal_pose_rank}<{required_pose_rank})"
    elif trans_min_sv < minimum_translational_singular_value:
        admitted = False
        reason = "translational_schur_singular_value_below_threshold"
    elif trans_cond > maximum_translational_condition_number:
        admitted = False
        reason = "translational_schur_condition_number_exceeded"
    else:
        admitted = True
        reason = "admitted_full_rank_observable"

    return SchurObservabilityReport(
        candidate_pose_index=candidate_pose_index,
        marginal_pose_rank=marginal_pose_rank,
        rank_by_tolerance_scale=rank_by_scale,
        full_reduced_rank=default_full_rank,
        reduced_rank_without_candidate=default_without_rank,
        translational_schur_block_3x3=r_i_3x3,
        translational_min_singular_value=trans_min_sv,
        translational_condition_number=trans_cond,
        pose_marginal_information_6x6=h_ii_6x6,
        pose_min_singular_value=pose_min_sv,
        pose_condition_number=pose_cond,
        admitted=admitted,
        reason=reason,
    )


def _skew3(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def build_se3_landmark_jacobians(
    *,
    world_from_camera_poses: Mapping[int, np.ndarray],
    landmarks_world_m: Mapping[int, np.ndarray],
    pose_landmark_observations: Sequence[tuple[int, int, float]],
    prior_pose_ids: Sequence[int] = (0,),
    pose_pose_observations: Sequence[tuple[int, int, float]] = (),
    prior_information_weight: float = 1e3,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Construct whitened pose and landmark Jacobians (A_x, A_y) for an SE(3) factor graph.

    Parameters
    ----------
    world_from_camera_poses:
        Mapping from pose_id to 4x4 SE(3) matrix T_{W,C}.
    landmarks_world_m:
        Mapping from landmark_id to 3D point p_W in R^3.
    pose_landmark_observations:
        Sequence of (pose_id, landmark_id, isotropic_sigma_m) for 3D metric
        bearing-range factors h(T_m, p_n) = R_m^T (p_n - t_m).
    prior_pose_ids:
        Pose IDs anchored by a 6-DoF gauge prior (e.g. X_0).
    pose_pose_observations:
        Optional sequence of (pose_id_a, pose_id_b, isotropic_sigma) for
        6-DoF relative BetweenFactors.

    Returns
    -------
    (A_x, A_y, ordered_pose_ids, ordered_landmark_ids)
    """
    ordered_poses = tuple(sorted(int(k) for k in world_from_camera_poses))
    ordered_landmarks = tuple(sorted(int(k) for k in landmarks_world_m))
    pose_idx = {pid: idx for idx, pid in enumerate(ordered_poses)}
    lm_idx = {lid: idx for idx, lid in enumerate(ordered_landmarks)}

    poses_valid: dict[int, np.ndarray] = {}
    for pid in ordered_poses:
        mat = _validate_se3(world_from_camera_poses[pid], f"pose[{pid}]")
        if mat is None:
            raise ValueError(f"world_from_camera_poses[{pid}] is not a valid SE(3) matrix")
        poses_valid[pid] = mat

    lms_valid: dict[int, np.ndarray] = {}
    for lid in ordered_landmarks:
        pt = np.asarray(landmarks_world_m[lid], dtype=np.float64)
        if pt.shape != (3,) or not np.isfinite(pt).all():
            raise ValueError(f"landmarks_world_m[{lid}] must be a finite 3-vector")
        lms_valid[lid] = pt

    prior_list = [int(p) for p in prior_pose_ids if int(p) in pose_idx]
    num_rows = 6 * len(prior_list) + 3 * len(pose_landmark_observations) + 6 * len(pose_pose_observations)
    ax = np.zeros((num_rows, 6 * len(ordered_poses)), dtype=np.float64)
    ay = np.zeros((num_rows, 3 * len(ordered_landmarks)), dtype=np.float64)

    row = 0
    for pid in prior_list:
        p_col = 6 * pose_idx[pid]
        ax[row : row + 6, p_col : p_col + 6] = np.eye(6, dtype=np.float64) * float(prior_information_weight)
        row += 6

    for pid_raw, lid_raw, sigma_raw in pose_landmark_observations:
        pid, lid = int(pid_raw), int(lid_raw)
        sigma = float(sigma_raw)
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError("observation sigma must be finite and positive")
        weight = 1.0 / sigma
        T_wc = poses_valid[pid]
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        p_w = lms_valid[lid]
        q_c = R_wc.T @ (p_w - t_wc)

        # Jacobian of h(T_m, p_n) = R_m^T (p_n - t_m) w.r.t. right-trivialized
        # pose tangent [delta_omega (3), delta_v (3)] and landmark delta_p (3):
        # dh / d(delta_omega) = [q_c]_x
        # dh / d(delta_v)     = -I_3
        # dh / d(delta_p)     = R_wc^T
        p_col = 6 * pose_idx[pid]
        l_col = 3 * lm_idx[lid]
        ax[row : row + 3, p_col : p_col + 3] = weight * _skew3(q_c)
        ax[row : row + 3, p_col + 3 : p_col + 6] = -weight * np.eye(3, dtype=np.float64)
        ay[row : row + 3, l_col : l_col + 3] = weight * R_wc.T
        row += 3

    for pa_raw, pb_raw, sigma_raw in pose_pose_observations:
        pa, pb = int(pa_raw), int(pb_raw)
        sigma = float(sigma_raw)
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError("between-factor sigma must be finite and positive")
        weight = 1.0 / sigma
        T_wa = poses_valid[pa]
        T_wb = poses_valid[pb]
        T_ba = np.linalg.inv(T_wb) @ T_wa
        R_ba = T_ba[:3, :3]
        t_ba = T_ba[:3, 3]
        # Adjoint of T_ba maps perturbation on pose_a to pose_b frame
        adj_ba = np.zeros((6, 6), dtype=np.float64)
        adj_ba[:3, :3] = R_ba
        adj_ba[3:6, :3] = _skew3(t_ba) @ R_ba
        adj_ba[3:6, 3:6] = R_ba

        ca = 6 * pose_idx[pa]
        cb = 6 * pose_idx[pb]
        ax[row : row + 6, ca : ca + 6] = -weight * adj_ba
        ax[row : row + 6, cb : cb + 6] = weight * np.eye(6, dtype=np.float64)
        row += 6

    return ax, ay, ordered_poses, ordered_landmarks


class FeatureBirthLedger:
    """Enforce Fork M: immutable feature-birth 3D observation ledger and deduplicated emission.

    Records the original camera-frame 3D sensor measurement when each 2D point
    track is first observed with valid depth (f_birth), and emits at most one
    deduplicated birth BearingRangeFactor commit when that track is later
    instantiated as a 3D landmark in the pose graph (f_admit >= f_birth).
    """

    def __init__(self, *, object_id: str) -> None:
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("object_id must be a non-empty string")
        self.object_id = object_id
        self._birth_records: dict[int, BirthObservationRecord] = {}
        self._committed_tracks: set[int] = set()

    @property
    def recorded_track_count(self) -> int:
        return len(self._birth_records)

    @property
    def committed_factor_count(self) -> int:
        return len(self._committed_tracks)

    def record_birth_observation(
        self,
        *,
        birth_frame_index: int,
        track_id: int,
        point_camera_m: Sequence[float] | np.ndarray,
        uncertainty_m: float = 0.005,
    ) -> BirthObservationRecord:
        """Record the immutable birth 3D measurement for a track if not already recorded."""
        if type(birth_frame_index) is not int or birth_frame_index < 0:
            raise ValueError("birth_frame_index must be a non-negative int")
        if type(track_id) is not int or track_id < 0:
            raise ValueError("track_id must be a non-negative int")
        if track_id in self._birth_records:
            return self._birth_records[track_id]
        pt = np.asarray(point_camera_m, dtype=np.float64)
        if pt.shape != (3,) or not np.isfinite(pt).all() or float(pt[2]) <= 0.0:
            raise ValueError("point_camera_m must be a finite 3D point with positive depth z > 0")
        unc = float(uncertainty_m)
        if not math.isfinite(unc) or unc <= 0.0:
            raise ValueError("uncertainty_m must be finite and positive")
        rec = BirthObservationRecord(
            object_id=self.object_id,
            birth_frame_index=birth_frame_index,
            track_id=track_id,
            point_camera_m=(float(pt[0]), float(pt[1]), float(pt[2])),
            uncertainty_m=unc,
        )
        self._birth_records[track_id] = rec
        return rec

    def emit_birth_factors(
        self,
        *,
        admit_frame_index: int,
        instantiated_track_to_landmark: Mapping[int, int],
        retained_keyframe_indices: Sequence[int] | None = None,
    ) -> list[BirthFactorCommit]:
        """Emit deduplicated birth factors for newly instantiated landmarks."""
        if type(admit_frame_index) is not int or admit_frame_index < 0:
            raise ValueError("admit_frame_index must be a non-negative int")
        retained_set = (
            {int(k) for k in retained_keyframe_indices}
            if retained_keyframe_indices is not None
            else None
        )
        commits: list[BirthFactorCommit] = []
        for track_raw, lm_raw in sorted(instantiated_track_to_landmark.items()):
            tid, lid = int(track_raw), int(lm_raw)
            if tid in self._committed_tracks:
                continue
            birth = self._birth_records.get(tid)
            if birth is None or birth.birth_frame_index > admit_frame_index:
                continue
            if retained_set is not None and birth.birth_frame_index not in retained_set:
                continue
            commit = BirthFactorCommit(
                object_id=self.object_id,
                birth_frame_index=birth.birth_frame_index,
                admit_frame_index=admit_frame_index,
                track_id=tid,
                landmark_id=lid,
                point_camera_m=birth.point_camera_m,
                uncertainty_m=birth.uncertainty_m,
            )
            self._committed_tracks.add(tid)
            commits.append(commit)
        return commits
