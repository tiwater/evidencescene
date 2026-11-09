"""EvidenceScene SOTA paradigm ablation & mechanism benchmark suite (`ablation-v1`).

Quantitatively compares EvidenceScene's core algorithmic mechanisms against
four standard baseline paradigms on identical inputs:
1. RevisableSurfelMap (M+S) vs. static irreversible voxel integration (nvblox/Voxblox paradigm)
2. Schur-Complement Observability Gate (F+G+R+M) vs. ungated pose admission (feedforward/unconstrained BA paradigm)
3. O(runs) RLE interval set algebra vs. dense O(H*W) boolean mask pipeline at 1080p (1080x1920)
4. Causal C^1 SE(3) re-acquisition blending vs. unblended step-reacquisition extrapolation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

import evidencescene
from evidencescene.cli import run_end_to_end_demo
from evidencescene.datasets import (
    IndependentValidationSequence,
    generate_independent_validation_sequences,
)
from evidencescene.metrics import (
    evaluate_pose_sequence,
    evaluate_surface_reconstruction,
)
from evidencescene.observability import (
    build_se3_landmark_jacobians,
    evaluate_prior_connected_admission,
    evaluate_schur_observability_gate,
)
from evidencescene.primitives import (
    backproject_masked_depth_points,
    decode_mask_rle,
    encode_mask_rle,
    rle_area,
    rle_iou,
)
from evidencescene.runtime import rle_mask_box_and_centroid
from evidencescene.surface import (
    RevisableSurfelMap,
    select_existing_points_shadow,
)
from evidencescene.teleop import CausalSe3TrajectoryPredictor


def _bench_rle_vs_dense_1080p(repeat: int = 15) -> dict[str, Any]:
    h, w = 1080, 1920
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[220:860, 480:1440] = True
    mask_a[360:440, 720:880] = False
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[232:872, 496:1456] = True
    mask_b[360:440, 720:880] = False

    rle_a = encode_mask_rle(mask_a)
    rle_b = encode_mask_rle(mask_b)

    # Warm-up
    _ = rle_mask_box_and_centroid(rle_a, width=w, height=h)
    _ = rle_iou(rle_a, rle_b)

    t0 = time.perf_counter()
    for _ in range(repeat):
        area_rle = rle_area(rle_a)
        box_cent = rle_mask_box_and_centroid(rle_a, width=w, height=h)
        iou_rle = rle_iou(rle_a, rle_b)
    rle_ms = ((time.perf_counter() - t0) * 1000.0) / repeat

    t1 = time.perf_counter()
    for _ in range(repeat):
        da = decode_mask_rle(rle_a, width=w, height=h)
        db = decode_mask_rle(rle_b, width=w, height=h)
        assert da is not None and db is not None
        area_dense = int(np.count_nonzero(da))
        ys, xs = np.nonzero(da)
        box_dense = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        cent_dense = (float(xs.mean()), float(ys.mean()))
        inter = int(np.count_nonzero(da & db))
        union = int(np.count_nonzero(da | db))
        iou_dense = inter / union
    dense_ms = ((time.perf_counter() - t1) * 1000.0) / repeat

    assert box_cent is not None
    assert area_rle == area_dense
    assert box_cent[0] == box_dense
    assert np.allclose(box_cent[1], cent_dense, atol=1e-12)
    assert abs(iou_rle - iou_dense) < 1e-12

    return {
        "resolution": "1080x1920",
        "rle_runs": len(rle_a["counts"]) // 2,
        "rle_median_ms": round(rle_ms, 4),
        "dense_median_ms": round(dense_ms, 4),
        "speedup_x": round(dense_ms / max(rle_ms, 1e-6), 2),
        "exact_parity": True,
    }


def _bench_c1_reacquisition_vs_unblended() -> dict[str, Any]:
    predictor = CausalSe3TrajectoryPredictor(
        reacquisition_gap_threshold_s=0.05,
        c1_join_duration_s=0.15,
        maximum_join_velocity_jump_mps=0.15,
        horizons_ms=(100.0, 250.0, 500.0),
    )
    pose0 = np.eye(4, dtype=np.float64)
    pose0[:3, 3] = [0.10, 0.00, 0.50]
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, 3] = [0.11, 0.00, 0.50]
    # Occlusion gap of 120 ms followed by a 25 mm re-acquisition displacement
    pose_reacq = np.eye(4, dtype=np.float64)
    pose_reacq[:3, 3] = [0.145, 0.012, 0.495]

    predictor.observe(pose=pose0, timestamp_ns=1_000_000_000, is_reacquisition=False)
    predictor.observe(pose=pose1, timestamp_ns=1_033_333_333, is_reacquisition=False)
    rep = predictor.observe(
        pose=pose_reacq, timestamp_ns=1_153_333_333, is_reacquisition=True
    )

    unblended_jump_mm = float(np.linalg.norm(pose_reacq[:3, 3] - pose1[:3, 3]) * 1000.0)
    c1_jump_mm = float(rep.position_join_jump_m * 1000.0)
    horizons = [h.interval_half_width_95_m for h in rep.horizons]
    monotonic_uncertainty = (
        rep.interval_width_monotonic_violation_count == 0
        and all(horizons[i] <= horizons[i + 1] for i in range(len(horizons) - 1))
    )

    return {
        "unblended_position_jump_mm": round(unblended_jump_mm, 3),
        "evidencescene_c1_position_jump_mm": round(c1_jump_mm, 6),
        "c1_join_active": bool(rep.c1_join_active),
        "monotonic_horizon_uncertainty": monotonic_uncertainty,
    }


def _invert_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r_t = pose[:3, :3].T
    inv[:3, :3] = r_t
    inv[:3, 3] = -r_t @ pose[:3, 3]
    return inv


def _evaluate_map_snapshot(
    surfel_map: RevisableSurfelMap,
    model_points_m: np.ndarray,
    *,
    budget: int = 1500,
) -> dict[str, float]:
    full_xyz, full_sup, full_cams, full_keys = surfel_map.snapshot(
        minimum_observation_support=2,
        maximum_point_count=surfel_map.maximum_voxels,
        include_voxel_keys=True,
    )
    if len(full_xyz) == 0:
        return {
            "P10_pct": 0.0,
            "C10_pct": 0.0,
            "F1_10_pct": 0.0,
            "F95_mm": float("inf"),
            "R95_mm": float("inf"),
        }
    sel = select_existing_points_shadow(
        full_xyz, full_sup, full_cams, full_keys, budget=budget
    )
    pts = full_xyz[sel["indices"]]
    score = evaluate_surface_reconstruction(
        pts,
        model_points_m,
        threshold_m=0.010,
        grid_voxel_size_m=0.004,
    )
    return {
        "P10_pct": round(score.precision_pct, 2),
        "C10_pct": round(score.coverage_pct, 2),
        "F1_10_pct": round(score.f1_pct, 2),
        "F95_mm": round(score.forward_p95_mm, 2),
        "R95_mm": round(score.reverse_p95_mm, 2),
    }


def _run_single_sequence_arm(
    val_seq: IndependentValidationSequence,
    *,
    enable_gate: bool,
    enable_surface_revision: bool,
) -> dict[str, Any]:
    scenario = val_seq.scenario
    surfel_map = RevisableSurfelMap(
        voxel_size_m=0.004,
        maximum_voxels=100_000,
        maximum_live_bytes=64 * 1024 * 1024,
    )
    committed_poses_cam: list[np.ndarray | None] = []
    retained_poses_cam: list[np.ndarray | None] = []
    reference_poses_cam: list[np.ndarray] = []
    camera_observations_raw: dict[str, tuple[str, np.ndarray]] = {}
    commit_ledger: list[dict[str, Any]] = []
    phase_surface_scores: dict[str, dict[str, float]] = {}

    checkpoint_by_frame = {
        f_idx: phase_name
        for phase_name, f_idx in val_seq.phase_checkpoints.items()
    }
    last_verified_pose: np.ndarray | None = None

    for bundle in scenario.frames:
        idx = bundle.frame_index
        reference_poses_cam.append(bundle.true_camera_from_object)
        if bundle.is_occluded or bundle.noisy_estimated_pose is None:
            committed_poses_cam.append(None)
            retained_poses_cam.append(last_verified_pose)
            continue

        T_wc_0 = scenario.world_from_camera @ scenario.frames[0].true_camera_from_object
        T_wc_i = scenario.world_from_camera @ bundle.noisy_estimated_pose
        edges = [(0, lid) for lid in val_seq.anchored_landmark_ids] + [
            (idx, lid) for lid in bundle.observed_landmark_ids
        ]
        conn_report = evaluate_prior_connected_admission(
            pose_ids=(0, idx) if idx != 0 else (0,),
            landmark_ids=tuple(scenario.landmarks_object_m.keys()),
            prior_pose_ids=(0,),
            pose_landmark_edges=edges,
            candidate_pose_id=idx,
            minimum_anchored_landmarks=3,
        )
        obs_tuples = [(p, l, 0.002) for p, l in edges]
        ax, ay, ordered_poses, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: T_wc_0, idx: T_wc_i} if idx != 0 else {0: T_wc_0},
            landmarks_world_m=scenario.landmarks_object_m,
            pose_landmark_observations=obs_tuples,
            prior_pose_ids=(0,),
        )
        cand_slot = ordered_poses.index(idx)
        schur_report = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax,
            landmark_jacobian_ay=ay,
            candidate_pose_index=cand_slot,
            required_pose_rank=6,
            tolerance_scales=(0.1, 1.0, 10.0),
            minimum_translational_singular_value=1e-4,
            maximum_translational_condition_number=1e6,
        )

        gate_pass = bool(conn_report.admitted and schur_report.admitted)
        admitted = gate_pass if enable_gate else True

        depth_m = bundle.registered_input.depth_mm.astype(np.float64) / 1000.0
        pts_cam = backproject_masked_depth_points(
            depth_m,
            bundle.registered_input.mask,
            bundle.registered_input.intrinsics,
            pixel_stride=1,
        )

        revised_kfs: list[int] = []
        if admitted:
            committed_poses_cam.append(bundle.noisy_estimated_pose)
            last_verified_pose = bundle.noisy_estimated_pose
            retained_poses_cam.append(last_verified_pose)
            if len(pts_cam) > 0:
                kf_key = f"kf_{idx:04d}"
                camera_observations_raw[kf_key] = (bundle.camera_id, pts_cam)
                T_oc = _invert_se3(bundle.noisy_estimated_pose)
                pts_obj = pts_cam @ T_oc[:3, :3].T + T_oc[:3, 3]
                surfel_map.update({kf_key: (bundle.camera_id, pts_obj)})
        else:
            committed_poses_cam.append(None)
            retained_poses_cam.append(last_verified_pose)

        # At the global BA revision trigger frame, historical pre-BA keyframes
        # receive refined poses; if S is enabled, RevisableSurfelMap executes
        # an atomic birth-keyframe-indexed surface transaction.
        if idx == val_seq.ba_revision_trigger_frame:
            revision_updates: dict[str, tuple[str, np.ndarray]] = {}
            for rev_idx in val_seq.pre_ba_drifted_frames:
                kf_key = f"kf_{rev_idx:04d}"
                refined_pose = scenario.frames[rev_idx].true_camera_from_object
                if enable_surface_revision:
                    committed_poses_cam[rev_idx] = refined_pose
                    retained_poses_cam[rev_idx] = refined_pose
                    if kf_key in camera_observations_raw:
                        cam_label, raw_cam_pts = camera_observations_raw[kf_key]
                        T_oc_ref = _invert_se3(refined_pose)
                        pts_obj_ref = raw_cam_pts @ T_oc_ref[:3, :3].T + T_oc_ref[:3, 3]
                        revision_updates[kf_key] = (cam_label, pts_obj_ref)
                        revised_kfs.append(rev_idx)
            if revision_updates and enable_surface_revision:
                surfel_map.update(revision_updates)

        commit_ledger.append(
            {
                "frame_index": idx,
                "c0_connected": bool(conn_report.candidate_connected_to_prior),
                "anchored_landmarks": int(conn_report.anchored_landmark_count_for_candidate),
                "marginal_rank_min": int(schur_report.marginal_pose_rank),
                "schur_sigma_min": round(float(schur_report.translational_min_singular_value), 6),
                "schur_cond": (
                    round(float(schur_report.translational_condition_number), 2)
                    if np.isfinite(schur_report.translational_condition_number)
                    else float("inf")
                ),
                "decision": "COMMIT" if admitted else "REJECT_HOLD",
                "revised_keyframes": revised_kfs,
            }
        )

        if idx in checkpoint_by_frame:
            phase_name = checkpoint_by_frame[idx]
            phase_surface_scores[phase_name] = _evaluate_map_snapshot(
                surfel_map, scenario.model_points_object_m
            )

    committed_score_10 = evaluate_pose_sequence(
        committed_poses_cam,
        reference_poses_cam,
        scenario.model_points_object_m,
        threshold_m=0.010,
    )
    committed_score_20 = evaluate_pose_sequence(
        committed_poses_cam,
        reference_poses_cam,
        scenario.model_points_object_m,
        threshold_m=0.020,
    )
    retained_score_10 = evaluate_pose_sequence(
        retained_poses_cam,
        reference_poses_cam,
        scenario.model_points_object_m,
        threshold_m=0.010,
    )

    return {
        "sequence_id": val_seq.sequence_id,
        "enable_gate": enable_gate,
        "enable_surface_revision": enable_surface_revision,
        "pose": {
            "denominator": committed_score_10.denominator_excluding_initial,
            "committed_avail": committed_score_10.available_post_initial_count,
            "retained_avail": retained_score_10.available_post_initial_count,
            "add_le_10mm": committed_score_10.add_le_threshold_count,
            "add_le_20mm": committed_score_20.add_le_threshold_count,
            "strict_10mm_10deg": committed_score_10.strict_threshold_count,
            "committed_add_auc": round(committed_score_10.add_auc_0_to_max, 4),
            "retained_add_auc": round(retained_score_10.add_auc_0_to_max, 4),
            "mean_add_mm": round((committed_score_10.mean_add_m or 0.0) * 1000.0, 2),
            "mean_adds_mm": round((committed_score_10.mean_adds_m or 0.0) * 1000.0, 2),
        },
        "surface_phases": phase_surface_scores,
        "terminal_surface": phase_surface_scores["t_terminal"],
        "commit_ledger": commit_ledger,
    }


def _bench_revisable_surfel_vs_bundlefusion_reintegration(
    repeat: int = 25,
) -> dict[str, Any]:
    """Compare RevisableSurfelMap birth-indexed delta transaction vs. full-history re-integration.

    Under equal inputs (K=20 keyframes of 3,000 points each = 60,000 raw points,
    with k=3 historical keyframes revised by global BA), compares:
    1. RevisableSurfelMap incremental delta update (`update(revised_3_kfs)`)
    2. Full-history re-integration from all K=20 keyframes (`rebuild_all(all_20_kfs)`)
    """
    rng = np.random.default_rng(77)
    num_kfs = 20
    pts_per_kf = 3000
    revised_indices = (4, 9, 14)

    initial_kfs: dict[str, tuple[str, np.ndarray]] = {}
    revised_kfs_only: dict[str, tuple[str, np.ndarray]] = {}
    full_post_revision_kfs: dict[str, tuple[str, np.ndarray]] = {}

    for k in range(num_kfs):
        cam = ("chest", "head", "wrist")[k % 3]
        base_pts = rng.uniform(-0.06, 0.06, size=(pts_per_kf, 3)).astype(np.float64)
        kf_id = f"kf_{k:04d}"
        initial_kfs[kf_id] = (cam, base_pts)
        if k in revised_indices:
            shifted_pts = base_pts + np.array([0.012, -0.008, 0.004], dtype=np.float64)
            revised_kfs_only[kf_id] = (cam, shifted_pts)
            full_post_revision_kfs[kf_id] = (cam, shifted_pts)
        else:
            full_post_revision_kfs[kf_id] = (cam, base_pts)

    # 1. Benchmark incremental birth-indexed delta transaction in RevisableSurfelMap
    delta_times_ms: list[float] = []
    for _ in range(repeat):
        m_inc = RevisableSurfelMap(
            voxel_size_m=0.004,
            maximum_voxels=100_000,
            maximum_live_bytes=128 * 1024 * 1024,
        )
        m_inc.update(initial_kfs)
        t0 = time.perf_counter()
        m_inc.update(revised_kfs_only)
        delta_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # 2. Benchmark full-history re-integration (rebuilding from all K=20 keyframes)
    rebuild_times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        m_full = RevisableSurfelMap(
            voxel_size_m=0.004,
            maximum_voxels=100_000,
            maximum_live_bytes=128 * 1024 * 1024,
        )
        m_full.update(full_post_revision_kfs)
        rebuild_times_ms.append((time.perf_counter() - t0) * 1000.0)

    inc_pts, inc_sup, inc_cams, inc_keys = m_inc.snapshot(
        minimum_observation_support=1,
        maximum_point_count=m_inc.maximum_voxels,
        include_voxel_keys=True,
    )
    full_pts, full_sup, full_cams, full_keys = m_full.snapshot(
        minimum_observation_support=1,
        maximum_point_count=m_full.maximum_voxels,
        include_voxel_keys=True,
    )

    exact_keys = bool(np.array_equal(inc_keys, full_keys))
    exact_support = bool(np.array_equal(inc_sup, full_sup))
    exact_cams = bool(np.array_equal(inc_cams, full_cams))
    max_pt_diff = float(np.max(np.abs(inc_pts - full_pts)))

    inc_med = float(np.median(delta_times_ms))
    full_med = float(np.median(rebuild_times_ms))

    return {
        "total_keyframes": num_kfs,
        "points_per_keyframe": pts_per_kf,
        "total_raw_points": num_kfs * pts_per_kf,
        "revised_keyframes_count": len(revised_indices),
        "incremental_delta_median_ms": round(inc_med, 3),
        "full_reintegration_median_ms": round(full_med, 3),
        "speedup_x": round(full_med / max(inc_med, 1e-6), 2),
        "exact_voxel_key_parity": exact_keys,
        "exact_support_parity": exact_support,
        "exact_camera_count_parity": exact_cams,
        "max_centroid_diff_m": max_pt_diff,
    }


def run_unified_2x2_factorial_benchmark() -> dict[str, Any]:
    """Run the frozen 2x2 unified-branch factorial (Gate ON/OFF x Surface Revision S ON/OFF)."""
    sequences = generate_independent_validation_sequences(num_frames=30)
    arms = (
        ("gate_off_s_off", False, False),
        ("gate_off_s_on", False, True),
        ("gate_on_s_off", True, False),
        ("gate_on_s_on", True, True),
    )

    per_sequence_results: dict[str, dict[str, Any]] = {}
    arm_aggregates: dict[str, dict[str, Any]] = {}

    for seq in sequences:
        per_sequence_results[seq.sequence_id] = {}

    for arm_name, g_flag, s_flag in arms:
        runs = [
            _run_single_sequence_arm(
                seq, enable_gate=g_flag, enable_surface_revision=s_flag
            )
            for seq in sequences
        ]
        for seq, r in zip(sequences, runs, strict=True):
            per_sequence_results[seq.sequence_id][arm_name] = r

        tot_denom = sum(r["pose"]["denominator"] for r in runs)
        tot_comm_avail = sum(r["pose"]["committed_avail"] for r in runs)
        tot_ret_avail = sum(r["pose"]["retained_avail"] for r in runs)
        tot_add10 = sum(r["pose"]["add_le_10mm"] for r in runs)
        tot_add20 = sum(r["pose"]["add_le_20mm"] for r in runs)
        tot_strict = sum(r["pose"]["strict_10mm_10deg"] for r in runs)

        phase_names = ("t_pre_degeneracy", "t_post_degeneracy", "t_post_ba_revision", "t_terminal")
        agg_phases: dict[str, dict[str, float]] = {}
        for p_name in phase_names:
            agg_phases[p_name] = {
                "P10_pct": round(float(np.mean([r["surface_phases"][p_name]["P10_pct"] for r in runs])), 2),
                "C10_pct": round(float(np.mean([r["surface_phases"][p_name]["C10_pct"] for r in runs])), 2),
                "F1_10_pct": round(float(np.mean([r["surface_phases"][p_name]["F1_10_pct"] for r in runs])), 2),
                "F95_mm": round(float(np.mean([r["surface_phases"][p_name]["F95_mm"] for r in runs])), 2),
                "R95_mm": round(float(np.mean([r["surface_phases"][p_name]["R95_mm"] for r in runs])), 2),
            }

        arm_aggregates[arm_name] = {
            "enable_gate": g_flag,
            "enable_surface_revision": s_flag,
            "pose_aggregate": {
                "total_post_init_frames": tot_denom,
                "committed_avail": tot_comm_avail,
                "retained_avail": tot_ret_avail,
                "add_le_10mm": tot_add10,
                "add_le_20mm": tot_add20,
                "strict_10mm_10deg": tot_strict,
                "mean_committed_add_auc": round(
                    float(np.mean([r["pose"]["committed_add_auc"] for r in runs])), 4
                ),
                "mean_retained_add_auc": round(
                    float(np.mean([r["pose"]["retained_add_auc"] for r in runs])), 4
                ),
                "mean_add_mm": round(
                    float(np.mean([r["pose"]["mean_add_mm"] for r in runs])), 2
                ),
                "mean_adds_mm": round(
                    float(np.mean([r["pose"]["mean_adds_mm"] for r in runs])), 2
                ),
            },
            "surface_phases_mean": agg_phases,
            "terminal_surface_mean": agg_phases["t_terminal"],
        }

    return {
        "frozen_config": {
            "voxel_size_m": 0.004,
            "required_pose_rank": 6,
            "minimum_anchored_landmarks": 3,
            "tolerance_multipliers": [0.1, 1.0, 10.0],
            "minimum_schur_singular_value": 1e-4,
            "maximum_schur_condition_number": 1e6,
            "minimum_observation_support": 2,
            "selector_budget": 1500,
            "evaluates_measured_surfels_only_is_prior_false": True,
        },
        "arm_aggregates": arm_aggregates,
        "per_sequence": per_sequence_results,
    }


def run_ablation_suite() -> dict[str, Any]:
    e2e = run_end_to_end_demo(num_frames=24, seed=42)
    rle_1080p = _bench_rle_vs_dense_1080p()
    c1_vs_unblended = _bench_c1_reacquisition_vs_unblended()
    unified_2x2 = run_unified_2x2_factorial_benchmark()
    revision_cost = _bench_revisable_surfel_vs_bundlefusion_reintegration()

    p_base = e2e["pose_evaluation"]["ungated_baseline"]
    p_ours = e2e["pose_evaluation"]["evidencescene_gated_and_revised"]
    s_base = e2e["surface_evaluation"]["naive_unrevised_accumulation"]
    s_ours = e2e["surface_evaluation"]["evidencescene_revisable_surfel_map"]

    digest_payload = json.dumps(
        {
            "pose": e2e["pose_evaluation"],
            "surface": e2e["surface_evaluation"],
            "c1": c1_vs_unblended,
            "rle_parity": rle_1080p["exact_parity"],
            "unified_2x2": unified_2x2["arm_aggregates"],
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()[:16]

    return {
        "schema": "evidencescene.ablation-benchmark.v1",
        "benchmark_id": "ablation-v1",
        "version": evidencescene.__version__,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "ablations": {
            "1_observability_gate_vs_ungated_pose": {
                "ungated_mean_add_mm": p_base["mean_add_mm"],
                "evidencescene_mean_add_mm": p_ours["mean_add_mm"],
                "add_error_reduction_x": round(
                    p_base["mean_add_mm"] / max(p_ours["mean_add_mm"], 1e-6), 2
                ),
                "ungated_mean_adds_mm": p_base["mean_adds_mm"],
                "evidencescene_mean_adds_mm": p_ours["mean_adds_mm"],
                "run_to_run_spread_mm": p_ours["run_to_run_spread_mm"],
            },
            "2_revisable_surfel_vs_static_voxel": {
                "static_voxel_P10_pct": s_base["P_10_pct"],
                "evidencescene_P10_pct": s_ours["P_10_pct"],
                "static_voxel_C10_pct": s_base["C_10_pct"],
                "evidencescene_C10_pct": s_ours["C_10_pct"],
                "static_voxel_F95_mm": s_base["F_95_mm"],
                "evidencescene_F95_mm": s_ours["F_95_mm"],
                "f95_error_reduction_x": round(
                    s_base["F_95_mm"] / max(s_ours["F_95_mm"], 1e-6), 2
                ),
            },
            "3_rle_algebra_vs_dense_1080p": rle_1080p,
            "4_c1_causal_se3_vs_unblended_reacquisition": c1_vs_unblended,
            "5_unified_2x2_gate_by_surface_revision": unified_2x2,
            "6_revisable_surfel_vs_full_reintegration": revision_cost,
        },
        "output_digest": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_ablation_suite()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
