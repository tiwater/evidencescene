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

from evidencescene._accel import NATIVE


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
    if arr.shape != (4, 4):
        return None
    if NATIVE is not None:
        arr_c = np.ascontiguousarray(arr, dtype=np.float64)
        if not NATIVE.is_valid_se3(arr_c, 1e-6, 1e-5, 1e-5):
            return None
        return arr_c.copy()
    if (
        not np.isfinite(arr).all()
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
    if not pose_pose_edges and len(pose_ids) <= 2:
        pose_set = {int(p) for p in pose_ids}
        landmark_set = {int(l) for l in landmark_ids}
        prior_set = {int(p) for p in prior_pose_ids if int(p) in pose_set}
        cand = int(candidate_pose_id)
        if not prior_set:
            return PriorConnectivityReport(
                candidate_pose_id=cand,
                candidate_connected_to_prior=False,
                anchored_landmark_count_for_candidate=0,
                admitted=False,
                reason="no_prior_anchored_poses",
                anchored_pose_ids=(),
                unanchored_pose_ids=tuple(sorted(pose_set)),
                anchored_landmark_ids=(),
                unanchored_landmark_ids=tuple(sorted(landmark_set)),
            )
        lms_by_pose: dict[int, set[int]] = {p: set() for p in pose_set}
        for p_raw, l_raw in pose_landmark_edges:
            p, l = int(p_raw), int(l_raw)
            if p in lms_by_pose and l in landmark_set:
                lms_by_pose[p].add(l)
        if len(pose_set) == 1:
            p0 = next(iter(pose_set))
            anchored_poses = {p0}
            anchored_landmarks = lms_by_pose[p0]
        else:
            p0, p1 = tuple(pose_set)
            if len(prior_set) == 2:
                anchored_poses = {p0, p1}
                anchored_landmarks = lms_by_pose[p0] | lms_by_pose[p1]
            else:
                pr = next(iter(prior_set))
                other = p1 if pr == p0 else p0
                l_pr = lms_by_pose[pr]
                l_other = lms_by_pose[other]
                if l_pr & l_other:
                    anchored_poses = {p0, p1}
                    anchored_landmarks = l_pr | l_other
                else:
                    anchored_poses = {pr}
                    anchored_landmarks = l_pr
        connected = cand in anchored_poses
        cand_landmarks = lms_by_pose.get(cand, set())
        anchored_shared_count = len(cand_landmarks & anchored_landmarks)
        if not connected:
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
    if ax.ndim != 2 or ax.shape[1] == 0 or ax.shape[1] % 6 != 0:
        raise ValueError("pose_jacobian_ax must be a finite 2D array with 6*N_p columns")
    num_poses = ax.shape[1] // 6
    if type(candidate_pose_index) is not int or not 0 <= candidate_pose_index < num_poses:
        raise ValueError(f"candidate_pose_index must be in [0, {num_poses - 1}]")
    if not 1 <= required_pose_rank <= 6:
        raise ValueError("required_pose_rank must be in [1, 6]")

    ay_arr: np.ndarray | None = None
    if landmark_jacobian_ay is not None:
        ay_cand = np.asarray(landmark_jacobian_ay, dtype=np.float64)
        if ay_cand.size > 0:
            if ay_cand.ndim != 2 or ay_cand.shape[0] != ax.shape[0]:
                raise ValueError(
                    "landmark_jacobian_ay must be finite with the same row count as pose_jacobian_ax"
                )
            ay_arr = ay_cand

    native_res = None
    if NATIVE is not None and ax.shape[1] <= 192 and ax.shape[0] >= ax.shape[1]:
        ax_c = ax if ax.flags.c_contiguous else np.ascontiguousarray(ax, dtype=np.float64)
        ay_c = (
            (ay_arr if ay_arr.flags.c_contiguous else np.ascontiguousarray(ay_arr, dtype=np.float64))
            if ay_arr is not None
            else None
        )
        candidate = NATIVE.evaluate_schur_gate_fast(ax_c, ay_c, candidate_pose_index)
        if candidate is not NotImplemented:
            native_res = candidate

    if native_res is not None:
        s_full, s_without, h_ii_6x6, r_i_3x3, trans_min_sv, trans_max_sv, pose_min_sv, pose_max_sv = native_res
    else:
        if not np.isfinite(ax).all():
            raise ValueError("pose_jacobian_ax must be a finite 2D array with 6*N_p columns")

        # Step 1: Orthogonal left-null-space projection to eliminate uncertain landmarks
        if ay_arr is not None:
            ay = ay_arr
            if not np.isfinite(ay).all():
                raise ValueError(
                    "landmark_jacobian_ay must be finite with the same row count as pose_jacobian_ax"
                )
            eps = float(np.finfo(np.float64).eps)
            projected_disjoint = False
            if ay.shape[1] >= 18 and ay.shape[1] % 3 == 0 and ay.shape[0] >= 18:
                num_lm = ay.shape[1] // 3
                block_active = np.any(ay.reshape(ay.shape[0], num_lm, 3) != 0.0, axis=2)
                row_counts = np.count_nonzero(block_active, axis=1)
                if int(row_counts.max()) <= 1:
                    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                    max_s = 0.0
                    for l_idx in range(num_lm):
                        r_mask = block_active[:, l_idx]
                        if np.any(r_mask):
                            y_blk = ay[r_mask, 3 * l_idx : 3 * l_idx + 3]
                            u_b, s_b, _ = np.linalg.svd(y_blk, full_matrices=False)
                            if s_b.size > 0 and float(s_b[0]) > max_s:
                                max_s = float(s_b[0])
                            blocks.append((r_mask, u_b, s_b))
                    tol_y = eps * max(ay.shape) * max_s
                    a_bar_x = ax.copy()
                    for r_mask, u_b, s_b in blocks:
                        r_b = _numerical_rank(s_b, tol_y)
                        if r_b > 0:
                            u_act = u_b[:, :r_b]
                            sub_x = a_bar_x[r_mask]
                            a_bar_x[r_mask] = sub_x - u_act @ (u_act.T @ sub_x)
                    projected_disjoint = True
            if not projected_disjoint:
                u_y, s_y, _vt_y = np.linalg.svd(ay, full_matrices=False)
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
        n_ax_cols = a_bar_x.shape[1]
        if col_start == 0:
            a_bar_without_i = a_bar_x[:, col_end:]
        elif col_end == n_ax_cols:
            a_bar_without_i = a_bar_x[:, :col_start]
        else:
            a_bar_without_i = np.concatenate(
                (a_bar_x[:, :col_start], a_bar_x[:, col_end:]), axis=1
            )

        s_full = np.linalg.svd(a_bar_x, compute_uv=False) if a_bar_x.size > 0 else np.zeros(0)
        have_wo_svd = a_bar_without_i.shape[1] > 0 and a_bar_without_i.shape[0] > 0
        if have_wo_svd:
            u_o, s_o, _vt_o = np.linalg.svd(a_bar_without_i, full_matrices=False)
            s_without = s_o
        else:
            u_o = np.empty((a_bar_x.shape[0], 0), dtype=np.float64)
            s_o = np.zeros(0)
            s_without = s_o

        # Step 3: Marginal 6x6 pose information and 3x3 translational Schur complement R_i
        a_i = a_bar_x[:, col_start:col_end]
        if have_wo_svd:
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

        h_rr = h_ii_6x6[:3, :3]
        h_rt = h_ii_6x6[:3, 3:6]
        h_tr = h_ii_6x6[3:6, :3]
        h_tt = h_ii_6x6[3:6, 3:6]
        r_i_3x3 = h_tt - h_tr @ np.linalg.pinv(h_rr, rcond=1e-12) @ h_rt
        r_i_3x3 = 0.5 * (r_i_3x3 + r_i_3x3.T)

        sv_t = np.linalg.svd(r_i_3x3, compute_uv=False)
        trans_min_sv = float(sv_t[-1])
        trans_max_sv = float(sv_t[0])

        sv_pose = np.linalg.svd(h_ii_6x6, compute_uv=False)
        pose_min_sv = float(sv_pose[-1])
        pose_max_sv = float(sv_pose[0])

    sigma_max = float(s_full[0]) if s_full.size > 0 else 0.0
    base_tol = float(np.finfo(np.float64).eps) * max(ax.shape) * sigma_max

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
    trans_cond = (
        trans_max_sv / trans_min_sv
        if trans_min_sv > 1e-15
        else math.inf
    )
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

    if NATIVE is not None:
        poses_arr = (
            np.ascontiguousarray(
                np.stack([poses_valid[pid] for pid in ordered_poses], axis=0),
                dtype=np.float64,
            )
            if ordered_poses
            else np.empty((0, 4, 4), dtype=np.float64)
        )
        lms_arr = (
            np.ascontiguousarray(
                np.stack([lms_valid[lid] for lid in ordered_landmarks], axis=0),
                dtype=np.float64,
            )
            if ordered_landmarks
            else np.empty((0, 3), dtype=np.float64)
        )
        prior_slots = np.array(
            [pose_idx[pid] for pid in prior_list], dtype=np.int64
        )
        n_obs = len(pose_landmark_observations)
        obs_ps = np.empty(n_obs, dtype=np.int64)
        obs_ls = np.empty(n_obs, dtype=np.int64)
        obs_w = np.empty(n_obs, dtype=np.float64)
        for i, (pid_raw, lid_raw, sigma_raw) in enumerate(pose_landmark_observations):
            sigma = float(sigma_raw)
            if sigma <= 0.0 or not math.isfinite(sigma):
                raise ValueError("observation sigma must be finite and positive")
            obs_ps[i] = pose_idx[int(pid_raw)]
            obs_ls[i] = lm_idx[int(lid_raw)]
            obs_w[i] = 1.0 / sigma
        n_btw = len(pose_pose_observations)
        btw_pa = np.empty(n_btw, dtype=np.int64)
        btw_pb = np.empty(n_btw, dtype=np.int64)
        btw_w = np.empty(n_btw, dtype=np.float64)
        for i, (pa_raw, pb_raw, sigma_raw) in enumerate(pose_pose_observations):
            sigma = float(sigma_raw)
            if sigma <= 0.0 or not math.isfinite(sigma):
                raise ValueError("between-factor sigma must be finite and positive")
            btw_pa[i] = pose_idx[int(pa_raw)]
            btw_pb[i] = pose_idx[int(pb_raw)]
            btw_w[i] = 1.0 / sigma
        ax, ay = NATIVE.build_se3_landmark_jacobians_fast(
            poses_arr,
            lms_arr,
            prior_slots,
            float(prior_information_weight),
            obs_ps,
            obs_ls,
            obs_w,
            btw_pa,
            btw_pb,
            btw_w,
        )
        return ax, ay, ordered_poses, ordered_landmarks

    num_rows = 6 * len(prior_list) + 3 * len(pose_landmark_observations) + 6 * len(pose_pose_observations)
    ax = np.zeros((num_rows, 6 * len(ordered_poses)), dtype=np.float64)
    ay = np.zeros((num_rows, 3 * len(ordered_landmarks)), dtype=np.float64)

    eye3 = np.eye(3, dtype=np.float64)
    eye6 = np.eye(6, dtype=np.float64)
    prior_diag = eye6 * float(prior_information_weight)
    row = 0
    for pid in prior_list:
        p_col = 6 * pose_idx[pid]
        ax[row : row + 6, p_col : p_col + 6] = prior_diag
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
        ax[row : row + 3, p_col + 3 : p_col + 6] = -weight * eye3
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
        ax[row : row + 6, cb : cb + 6] = weight * eye6
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


@dataclass(frozen=True)
class SchurSubspaceRegularizationReport:
    """Result of subspace-projected Schur observability regularization on SE(3).

    Projects the relative pose tangent ``xi = Log(T_prior^{-1} T_cand)`` onto
    the observable eigenspace of the 6x6 marginal pose information matrix
    ``H_ii`` so rank-deficient directions (e.g., cylindrical spin or sliding
    along a featureless edge) remain anchored to ``T_prior`` while observable
    DoFs are updated smoothly.
    """

    mode: str
    regularized_pose: np.ndarray
    observable_dof_count: int
    clamped_dof_count: int
    eigenvalues_6: np.ndarray
    modal_gains_6: np.ndarray
    unobservable_basis: np.ndarray
    raw_tangent_6: np.ndarray
    regularized_tangent_6: np.ndarray
    clamped_tangent_norm: float
    prior_fallback_triggered: bool
    reason: str


def _so3_exp_mat(omega: np.ndarray) -> np.ndarray:
    vx, vy, vz = float(omega[0]), float(omega[1]), float(omega[2])
    theta2 = vx * vx + vy * vy + vz * vz
    if theta2 < 1e-8:
        a = 1.0 - theta2 * (1.0 / 6.0) + theta2 * theta2 * (1.0 / 120.0)
        b = 0.5 - theta2 * (1.0 / 24.0) + theta2 * theta2 * (1.0 / 720.0)
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta2
    K = _skew3(omega)
    return np.eye(3, dtype=np.float64) + a * K + b * (K @ K)


def _so3_log_vec(R: np.ndarray) -> np.ndarray:
    cos_theta = 0.5 * (float(R[0, 0]) + float(R[1, 1]) + float(R[2, 2]) - 1.0)
    cos_c = max(-1.0, min(1.0, cos_theta))
    rx = float(R[2, 1]) - float(R[1, 2])
    ry = float(R[0, 2]) - float(R[2, 0])
    rz = float(R[1, 0]) - float(R[0, 1])
    if cos_c >= 1.0 - 1e-8:
        sin2 = 0.25 * (rx * rx + ry * ry + rz * rz)
        factor = 0.5 + sin2 * (1.0 / 12.0) + sin2 * sin2 * (3.0 / 80.0)
        return np.array([factor * rx, factor * ry, factor * rz], dtype=np.float64)
    if cos_c <= -1.0 + 1e-7:
        # Near pi: extract unit axis from (R + I)/2 diagonal/columns.
        M = 0.5 * (R + np.eye(3, dtype=np.float64))
        idx = int(np.argmax(np.diag(M)))
        axis = M[:, idx].copy()
        nrm = float(np.linalg.norm(axis))
        if nrm > 1e-12:
            axis /= nrm
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        theta = math.acos(cos_c)
        return axis * theta
    theta = math.acos(cos_c)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_c * cos_c))
    factor = theta / (2.0 * sin_theta)
    return np.array([factor * rx, factor * ry, factor * rz], dtype=np.float64)


def _se3_left_jacobian(omega: np.ndarray) -> np.ndarray:
    theta2 = float(np.dot(omega, omega))
    K = _skew3(omega)
    if theta2 < 1e-8:
        c1 = 0.5 - theta2 * (1.0 / 24.0) + theta2 * theta2 * (1.0 / 720.0)
        c2 = (1.0 / 6.0) - theta2 * (1.0 / 120.0) + theta2 * theta2 * (1.0 / 5040.0)
    else:
        theta = math.sqrt(theta2)
        c1 = (1.0 - math.cos(theta)) / theta2
        c2 = (theta - math.sin(theta)) / (theta2 * theta)
    return np.eye(3, dtype=np.float64) + c1 * K + c2 * (K @ K)


def _se3_left_jacobian_inv(omega: np.ndarray) -> np.ndarray:
    theta2 = float(np.dot(omega, omega))
    K = _skew3(omega)
    if theta2 < 1e-8:
        c2 = (1.0 / 12.0) + theta2 * (1.0 / 720.0)
    else:
        theta = math.sqrt(theta2)
        half = 0.5 * theta
        # 1/theta^2 * (1 - (theta/2) / tan(theta/2))
        c2 = (1.0 - (half * math.cos(half)) / max(1e-15, math.sin(half))) / theta2
    return np.eye(3, dtype=np.float64) - 0.5 * K + c2 * (K @ K)


def _se3_log(T: np.ndarray) -> np.ndarray:
    omega = _so3_log_vec(T[:3, :3])
    v = _se3_left_jacobian_inv(omega) @ T[:3, 3]
    return np.concatenate([omega, v])


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    omega = xi[:3]
    v = xi[3:6]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _so3_exp_mat(omega)
    out[:3, 3] = _se3_left_jacobian(omega) @ v
    return out


def _se3_inv(T: np.ndarray) -> np.ndarray:
    R_t = T[:3, :3].T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_t
    out[:3, 3] = -(R_t @ T[:3, 3])
    return out


def project_schur_observable_subspace(
    *,
    candidate_pose: object,
    prior_pose: object,
    schur_report: SchurObservabilityReport | None = None,
    pose_marginal_information_6x6: np.ndarray | None = None,
    mode: str = "tikhonov",
    prior_information_weight: float = 1.0,
    relative_eigenvalue_tolerance: float = 1e-6,
    minimum_eigenvalue_threshold: float = 1e-6,
) -> SchurSubspaceRegularizationReport:
    """Regularize a candidate SE(3) pose update onto the Schur-observable subspace.

    Preserves the binary :func:`evaluate_schur_observability_gate` contract while
    enabling smooth subspace-conditioned updates for symmetric or partially
    observed objects (e.g., cylindrical vessels with 5 observable DoFs).

    Given right-trivialized relative tangent ``xi = Log(T_prior^{-1} T_cand)``
    in ``[delta_omega (3), delta_v (3)]`` coordinates and symmetric PSD marginal
    information ``H_ii = Q diag(lambda) Q^T``:

    - ``mode="tikhonov"`` solves the closed-form MAP objective
      ``min_d (d - xi)^T H_ii (d - xi) + lambda_prior ||d||_2^2``, applying
      modal gain ``g_k = lambda_k^+ / (lambda_k^+ + lambda_prior)`` for
      observable modes (``lambda_k > tol``) and ``0`` for unobservable null-space
      modes (``lambda_k <= tol``).
    - ``mode="hard_subspace"`` applies an exact orthogonal projector with gain
      ``1.0`` on observable eigenmodes and ``0.0`` on unobservable modes,
      passing full-rank updates through unchanged while clamping null-space drift
      strictly to ``prior_pose``.
    """
    if mode not in ("tikhonov", "hard_subspace"):
        raise ValueError("mode must be 'tikhonov' or 'hard_subspace'")
    if (
        isinstance(prior_information_weight, bool)
        or not math.isfinite(prior_information_weight)
        or prior_information_weight < 0.0
    ):
        raise ValueError("prior_information_weight must be finite and nonnegative")
    if (
        isinstance(relative_eigenvalue_tolerance, bool)
        or not math.isfinite(relative_eigenvalue_tolerance)
        or relative_eigenvalue_tolerance < 0.0
    ):
        raise ValueError("relative_eigenvalue_tolerance must be finite and nonnegative")
    if (
        isinstance(minimum_eigenvalue_threshold, bool)
        or not math.isfinite(minimum_eigenvalue_threshold)
        or minimum_eigenvalue_threshold < 0.0
    ):
        raise ValueError("minimum_eigenvalue_threshold must be finite and nonnegative")

    T_cand = _validate_se3(candidate_pose, "candidate_pose")
    if T_cand is None:
        raise ValueError("candidate_pose must be a valid finite 4x4 SE(3) matrix")
    T_prior = _validate_se3(prior_pose, "prior_pose")
    if T_prior is None:
        raise ValueError("prior_pose must be a valid finite 4x4 SE(3) matrix")

    if schur_report is not None and pose_marginal_information_6x6 is not None:
        raise ValueError("Provide either schur_report or pose_marginal_information_6x6, not both")
    if schur_report is not None:
        if not isinstance(schur_report, SchurObservabilityReport):
            raise ValueError("schur_report must be a SchurObservabilityReport")
        H_raw = schur_report.pose_marginal_information_6x6
    elif pose_marginal_information_6x6 is not None:
        H_raw = pose_marginal_information_6x6
    else:
        raise ValueError("Either schur_report or pose_marginal_information_6x6 must be provided")

    H = np.asarray(H_raw, dtype=np.float64)
    if H.shape != (6, 6) or not np.isfinite(H).all():
        raise ValueError("pose_marginal_information_6x6 must be a finite 6x6 matrix")
    H_sym = 0.5 * (H + H.T)

    eigvals, eigvecs = np.linalg.eigh(H_sym)
    eigvals_clamped = np.maximum(eigvals, 0.0)
    lam_max = float(eigvals_clamped[-1])
    eig_tol = max(
        float(minimum_eigenvalue_threshold),
        float(relative_eigenvalue_tolerance) * lam_max,
    )
    obs_mask = eigvals_clamped > eig_tol
    obs_count = int(np.count_nonzero(obs_mask))
    clamped_count = 6 - obs_count

    T_delta = _se3_inv(T_prior) @ T_cand
    raw_xi = _se3_log(T_delta)

    if obs_count == 0:
        gains = np.zeros(6, dtype=np.float64)
        reg_xi = np.zeros(6, dtype=np.float64)
        T_reg = T_prior.copy()
        prior_fallback = True
        reason = "unobservable_fallback_to_prior"
    else:
        if mode == "hard_subspace":
            gains = np.where(obs_mask, 1.0, 0.0).astype(np.float64)
        else:
            lam_prior = float(prior_information_weight)
            gains = np.where(
                obs_mask,
                eigvals_clamped / (eigvals_clamped + lam_prior),
                0.0,
            ).astype(np.float64)
        if obs_count == 6 and mode == "hard_subspace":
            reg_xi = raw_xi.copy()
            T_reg = T_cand.copy()
            reason = "full_rank_passthrough"
        else:
            modal_coords = eigvecs.T @ raw_xi
            reg_xi = eigvecs @ (gains * modal_coords)
            T_reg = T_prior @ _se3_exp(reg_xi)
            reason = (
                "full_rank_tikhonov_regularized"
                if obs_count == 6
                else f"subspace_projected({obs_count}/6_observable)"
            )
        prior_fallback = False

    unobs_basis = eigvecs[:, ~obs_mask].copy()
    clamped_norm = float(np.linalg.norm(raw_xi - reg_xi))

    for arr in (T_reg, eigvals, gains, unobs_basis, raw_xi, reg_xi):
        arr.setflags(write=False)

    return SchurSubspaceRegularizationReport(
        mode=mode,
        regularized_pose=T_reg,
        observable_dof_count=obs_count,
        clamped_dof_count=clamped_count,
        eigenvalues_6=eigvals,
        modal_gains_6=gains,
        unobservable_basis=unobs_basis,
        raw_tangent_6=raw_xi,
        regularized_tangent_6=reg_xi,
        clamped_tangent_norm=clamped_norm,
        prior_fallback_triggered=prior_fallback,
        reason=reason,
    )
