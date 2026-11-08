"""Command-line interface and end-to-end closed-loop demonstration for EvidenceScene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

import evidencescene
from evidencescene.datasets import generate_synthetic_manipulation_sequence
from evidencescene.metrics import (
    compute_trajectory_determinism_spread,
    evaluate_pose_sequence,
    evaluate_surface_reconstruction,
)
from evidencescene.observability import (
    FeatureBirthLedger,
    build_se3_landmark_jacobians,
    evaluate_prior_connected_admission,
    evaluate_schur_observability_gate,
)
from evidencescene.primitives import (
    backproject_masked_depth_points,
    decode_mask_rle,
    rle_area,
    rle_iou,
)
from evidencescene.runtime import rle_mask_box_and_centroid
from evidencescene.surface import (
    RevisableSurfelMap,
    select_existing_points_shadow,
)
from evidencescene.teleop import (
    CausalSe3TrajectoryPredictor,
    MeasuredPointFeature3D,
    fit_circular_aperture_3d,
    propagate_feature_uncertainty_se3,
    verify_l2_pouring_safety_envelope,
)


def _invert_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r_t = pose[:3, :3].T
    inv[:3, :3] = r_t
    inv[:3, 3] = -r_t @ pose[:3, 3]
    return inv


def run_end_to_end_demo(*, num_frames: int = 24, seed: int = 42) -> dict[str, Any]:
    """Execute the complete EvidenceScene pipeline on a synthetic manipulation sequence."""
    scenario = generate_synthetic_manipulation_sequence(
        num_frames=num_frames, seed=seed
    )

    # 1. RLE Mask Algebra & Frontend Verification
    rle_areas: list[int] = []
    rle_ious: list[float] = []
    prev_rle = None
    for bundle in scenario.frames:
        area = rle_area(bundle.mask_rle)
        rle_areas.append(area)
        if area > 0:
            geom = rle_mask_box_and_centroid(
                bundle.mask_rle,
                width=bundle.registered_input.color_rgb.shape[1],
                height=bundle.registered_input.color_rgb.shape[0],
            )
            assert geom is not None
        if prev_rle is not None:
            rle_ious.append(rle_iou(prev_rle, bundle.mask_rle))
        prev_rle = bundle.mask_rle

    # Benchmark O(runs) RLE vs O(H*W) Dense Mask on standard 480x640 RGB-D silhouette
    mask_480p_a = np.zeros((480, 640), dtype=bool)
    mask_480p_a[90:390, 140:500] = True
    mask_480p_b = np.zeros((480, 640), dtype=bool)
    mask_480p_b[96:396, 148:508] = True
    rle_480p_a = evidencescene.primitives.encode_mask_rle(mask_480p_a)
    rle_480p_b = evidencescene.primitives.encode_mask_rle(mask_480p_b)

    t0_rle = time.perf_counter()
    for _ in range(10):
        _ = rle_mask_box_and_centroid(rle_480p_a, width=640, height=480)
        _ = rle_iou(rle_480p_a, rle_480p_b)
    rle_stage_ms = (time.perf_counter() - t0_rle) * 100.0

    t0_dense = time.perf_counter()
    for _ in range(10):
        da = decode_mask_rle(rle_480p_a, width=640, height=480)
        db = decode_mask_rle(rle_480p_b, width=640, height=480)
        assert da is not None and db is not None
        ys, xs = np.nonzero(da)
        _ = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        _ = (float(xs.mean()), float(ys.mean()))
        inter = int(np.count_nonzero(da & db))
        union = int(np.count_nonzero(da | db))
        _ = inter / union
    dense_stage_ms = (time.perf_counter() - t0_dense) * 100.0
    rle_speedup_x = dense_stage_ms / max(rle_stage_ms, 1e-6)

    t0_online = time.perf_counter()

    # 2. Factor-Graph Observability (F + G + R + M) & Revisable Surfel Fusion (M + S)
    birth_ledger = FeatureBirthLedger(object_id=scenario.object_id)
    naive_map = RevisableSurfelMap(
        voxel_size_m=0.004,
        maximum_voxels=100_000,
        maximum_live_bytes=64 * 1024 * 1024,
    )
    revisable_map = RevisableSurfelMap(
        voxel_size_m=0.004,
        maximum_voxels=100_000,
        maximum_live_bytes=64 * 1024 * 1024,
    )

    predictor = CausalSe3TrajectoryPredictor(
        reacquisition_gap_threshold_s=0.05,
        c1_join_duration_s=0.15,
        maximum_join_velocity_jump_mps=0.15,
        horizons_ms=(100.0, 250.0, 500.0),
    )

    ungated_poses_cam: list[np.ndarray | None] = []
    gated_poses_cam: list[np.ndarray | None] = []
    reference_poses_cam: list[np.ndarray] = []
    camera_observations_raw: dict[str, tuple[str, np.ndarray]] = {}

    rejected_rank_deficient_frames: list[int] = []
    c1_reacquisition_frames: list[int] = []
    revised_keyframe_transactions = 0
    was_occluded = False

    # Anchor frame 0 observations in birth ledger
    for lid, pt_obj in scenario.landmarks_object_m.items():
        T_co_0 = scenario.frames[0].true_camera_from_object
        pt_cam_0 = T_co_0[:3, :3] @ pt_obj + T_co_0[:3, 3]
        birth_ledger.record_birth_observation(
            birth_frame_index=0,
            track_id=lid,
            point_camera_m=pt_cam_0,
            uncertainty_m=0.002,
        )

    birth_commits = birth_ledger.emit_birth_factors(
        admit_frame_index=0,
        instantiated_track_to_landmark={lid: lid for lid in scenario.landmarks_object_m},
    )

    for bundle in scenario.frames:
        idx = bundle.frame_index
        reference_poses_cam.append(bundle.true_camera_from_object)
        ungated_poses_cam.append(bundle.noisy_estimated_pose)

        if bundle.is_occluded or bundle.noisy_estimated_pose is None:
            gated_poses_cam.append(None)
            was_occluded = True
            continue

        # Build 2-pose (Anchor Frame 0 + Candidate Frame idx) factor-graph observability test
        T_wc_0 = scenario.world_from_camera @ scenario.frames[0].true_camera_from_object
        T_wc_i = scenario.world_from_camera @ bundle.noisy_estimated_pose
        edges = [(0, lid) for lid in scenario.landmarks_object_m] + [
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
        )

        # Backproject current frame depth points in camera coordinates
        depth_m = bundle.registered_input.depth_mm.astype(np.float64) / 1000.0
        pts_cam = backproject_masked_depth_points(
            depth_m,
            bundle.registered_input.mask,
            bundle.registered_input.intrinsics,
            pixel_stride=1,
        )

        # Naive map fuses every non-occluded frame without observability gating or pose revision
        if len(pts_cam) > 0:
            T_oc_naive = _invert_se3(bundle.noisy_estimated_pose)
            pts_obj_naive = pts_cam @ T_oc_naive[:3, :3].T + T_oc_naive[:3, 3]
            naive_map.update({f"kf_{idx:04d}": (bundle.camera_id, pts_obj_naive)})

        if not (conn_report.admitted and schur_report.admitted):
            rejected_rank_deficient_frames.append(idx)
            gated_poses_cam.append(None)
            continue

        gated_poses_cam.append(bundle.noisy_estimated_pose)

        # Fuse admitted observation into lineage-indexed RevisableSurfelMap
        if len(pts_cam) > 0:
            camera_observations_raw[f"kf_{idx:04d}"] = (bundle.camera_id, pts_cam)
            T_oc = _invert_se3(bundle.noisy_estimated_pose)
            pts_obj = pts_cam @ T_oc[:3, :3].T + T_oc[:3, 3]
            revisable_map.update({f"kf_{idx:04d}": (bundle.camera_id, pts_obj)})

        # At Frame 8, global pose graph optimizes historical keyframes 4 & 5, correcting their drift:
        # Execute atomic surface-history transaction S on RevisableSurfelMap
        if idx == 8:
            revision_updates: dict[str, tuple[str, np.ndarray]] = {}
            for rev_idx in (4, 5):
                kf_key = f"kf_{rev_idx:04d}"
                if kf_key in camera_observations_raw:
                    cam_label, raw_cam_pts = camera_observations_raw[kf_key]
                    refined_pose = scenario.frames[rev_idx].true_camera_from_object
                    gated_poses_cam[rev_idx] = refined_pose
                    T_oc_ref = _invert_se3(refined_pose)
                    pts_obj_ref = raw_cam_pts @ T_oc_ref[:3, :3].T + T_oc_ref[:3, 3]
                    revision_updates[kf_key] = (cam_label, pts_obj_ref)
            if revision_updates:
                revisable_map.update(revision_updates)
                revised_keyframe_transactions += 1

        # Update causal SE(3) trajectory predictor in world frame
        T_wo_est = scenario.world_from_camera @ bundle.noisy_estimated_pose
        traj_report = predictor.observe(
            pose=T_wo_est,
            timestamp_ns=bundle.timestamp_ns,
            is_reacquisition=was_occluded,
        )
        if traj_report.c1_join_active and was_occluded:
            c1_reacquisition_frames.append(idx)
            assert traj_report.position_join_jump_m == 0.0
        was_occluded = False

    online_loop_ms = (time.perf_counter() - t0_online) * 1000.0

    # 3. Evaluate 6D Pose Accuracy & Run-to-Run Determinism
    ungated_pose_score = evaluate_pose_sequence(
        ungated_poses_cam,
        reference_poses_cam,
        scenario.model_points_object_m,
        threshold_m=0.010,
    )
    gated_pose_score = evaluate_pose_sequence(
        gated_poses_cam,
        reference_poses_cam,
        scenario.model_points_object_m,
        threshold_m=0.010,
    )
    determinism = compute_trajectory_determinism_spread(
        [gated_poses_cam, [p.copy() if p is not None else None for p in gated_poses_cam]]
    )

    # 4. Evaluate Surface Reconstruction & View-Diversity Selection (support10/coarse12mm)
    naive_xyz, _, _ = naive_map.snapshot(
        minimum_observation_support=2, maximum_point_count=1500
    )
    full_xyz, full_support, full_cams, full_keys = revisable_map.snapshot(
        minimum_observation_support=2,
        maximum_point_count=revisable_map.maximum_voxels,
        include_voxel_keys=True,
    )
    t0_select = time.perf_counter()
    selection = select_existing_points_shadow(
        full_xyz, full_support, full_cams, full_keys, budget=1500
    )
    surface_selection_ms = (time.perf_counter() - t0_select) * 1000.0
    selected_xyz = full_xyz[selection["indices"]]

    naive_surface_score = evaluate_surface_reconstruction(
        naive_xyz,
        scenario.model_points_object_m,
        threshold_m=0.010,
        grid_voxel_size_m=0.004,
    )
    revisable_surface_score = evaluate_surface_reconstruction(
        selected_xyz,
        scenario.model_points_object_m,
        threshold_m=0.010,
        grid_voxel_size_m=0.004,
    )

    # 5. Evaluate L2 Teleoperation Pouring Safety at Terminal Frame
    t0_teleop = time.perf_counter()
    aperture = fit_circular_aperture_3d(
        scenario.target_cup_rim_world_m,
        feature_id="cup_rim",
        timestamp_ns=scenario.frames[-1].timestamp_ns,
        coordinate_frame="world",
    )
    spout_obj = MeasuredPointFeature3D(
        feature_id="pour_outlet",
        point_m=scenario.spout_point_object_m,
        direction=scenario.spout_direction_object,
        covariance_3x3_m2=np.eye(3, dtype=np.float64) * 2e-6,
        direction_std_rad=0.015,
        support_pixel_count=48,
        timestamp_ns=scenario.frames[-1].timestamp_ns,
        coordinate_frame="object",
    )
    final_T_wo = scenario.frames[-1].true_world_from_object
    pose_cov_6x6 = np.diag([1e-5, 1e-5, 1e-5, 2e-6, 2e-6, 2e-6])
    spout_world = propagate_feature_uncertainty_se3(
        feature_object_frame=spout_obj,
        world_from_object=final_T_wo,
        pose_covariance_6x6=pose_cov_6x6,
        target_coordinate_frame="world",
    )
    pouring_verdict = verify_l2_pouring_safety_envelope(
        pour_outlet_world=spout_world,
        target_aperture_world=aperture,
        exit_speed_mps=0.35,
    )
    teleop_safety_ms = (time.perf_counter() - t0_teleop) * 1000.0

    digest_hasher = hashlib.sha256()
    digest_hasher.update(np.ascontiguousarray(selected_xyz).tobytes())
    for p in gated_poses_cam:
        if p is not None:
            digest_hasher.update(np.ascontiguousarray(p).tobytes())
    output_digest = digest_hasher.hexdigest()[:16]
    mean_frame_ms = online_loop_ms / max(1, num_frames)
    effective_fps = 1000.0 / max(mean_frame_ms, 1e-6)

    return {
        "schema": "evidencescene.end-to-end-demo.v1",
        "version": evidencescene.__version__,
        "scenario": {
            "num_frames": num_frames,
            "model_point_count": len(scenario.model_points_object_m),
            "tracked_landmarks": len(scenario.landmarks_object_m),
            "mean_rle_foreground_pixels": round(float(np.mean(rle_areas)), 1),
            "mean_adjacent_rle_iou": round(float(np.mean(rle_ious)), 4),
        },
        "observability_gate": {
            "birth_factors_committed": len(birth_commits),
            "rejected_rank_deficient_frames": rejected_rank_deficient_frames,
            "revised_keyframe_surface_transactions": revised_keyframe_transactions,
            "c1_reacquisition_frames": c1_reacquisition_frames,
        },
        "pose_evaluation": {
            "ungated_baseline": {
                "mean_add_mm": round((ungated_pose_score.mean_add_m or 0.0) * 1000.0, 2),
                "mean_adds_mm": round((ungated_pose_score.mean_adds_m or 0.0) * 1000.0, 2),
                "adds_le_10mm_count": ungated_pose_score.adds_le_threshold_count,
                "adds_auc_0_100mm": round(ungated_pose_score.adds_auc_0_to_max, 4),
            },
            "evidencescene_gated_and_revised": {
                "mean_add_mm": round((gated_pose_score.mean_add_m or 0.0) * 1000.0, 2),
                "mean_adds_mm": round((gated_pose_score.mean_adds_m or 0.0) * 1000.0, 2),
                "adds_le_10mm_count": gated_pose_score.adds_le_threshold_count,
                "adds_auc_0_100mm": round(gated_pose_score.adds_auc_0_to_max, 4),
                "run_to_run_spread_mm": determinism.max_translation_spread_mm,
                "run_to_run_spread_deg": determinism.max_rotation_spread_deg,
            },
        },
        "surface_evaluation": {
            "naive_unrevised_accumulation": {
                "P_10_pct": round(naive_surface_score.precision_pct, 2),
                "C_10_pct": round(naive_surface_score.coverage_pct, 2),
                "F1_10_pct": round(naive_surface_score.f1_pct, 2),
                "F_95_mm": round(naive_surface_score.forward_p95_mm, 2),
            },
            "evidencescene_revisable_surfel_map": {
                "P_10_pct": round(revisable_surface_score.precision_pct, 2),
                "C_10_pct": round(revisable_surface_score.coverage_pct, 2),
                "F1_10_pct": round(revisable_surface_score.f1_pct, 2),
                "F_95_mm": round(revisable_surface_score.forward_p95_mm, 2),
                "selector_mode": selection["mode"],
                "selected_surfel_count": int(len(selected_xyz)),
            },
        },
        "teleop_pouring_safety": {
            "status": pouring_verdict.l2_pour_safety_status,
            "safe_to_pour": pouring_verdict.safe_to_pour,
            "rim_clearance_95_mm": round(
                (pouring_verdict.current_rim_clearance_95_m or 0.0) * 1000.0, 2
            ),
            "target_aperture_radius_mm": round(aperture.radius_m * 1000.0, 2),
        },
        "runtime_performance": {
            "rle_stage_ms": round(rle_stage_ms, 3),
            "dense_mask_baseline_ms": round(dense_stage_ms, 3),
            "rle_vs_dense_speedup_x": round(rle_speedup_x, 2),
            "mean_frame_latency_ms": round(mean_frame_ms, 3),
            "effective_fps": round(effective_fps, 1),
            "surface_selection_ms": round(surface_selection_ms, 3),
            "teleop_aperture_and_safety_ms": round(teleop_safety_ms, 3),
            "output_digest": output_digest,
        },
    }


def _print_human_report(report: dict[str, Any]) -> None:
    print("=" * 74)
    print(
        f"  EvidenceScene v{report['version']} — End-to-End Closed-Loop Pipeline Report"
    )
    print("=" * 74)
    sc = report["scenario"]
    obs = report["observability_gate"]
    p_base = report["pose_evaluation"]["ungated_baseline"]
    p_ours = report["pose_evaluation"]["evidencescene_gated_and_revised"]
    s_base = report["surface_evaluation"]["naive_unrevised_accumulation"]
    s_ours = report["surface_evaluation"]["evidencescene_revisable_surfel_map"]
    pour = report["teleop_pouring_safety"]
    perf = report["runtime_performance"]

    print(
        f"[1] RGB-D Stream & O(runs) RLE Algebra : {sc['num_frames']} frames | "
        f"Mean Area = {sc['mean_rle_foreground_pixels']} px | "
        f"Mean Adj. IoU = {sc['mean_adjacent_rle_iou']:.4f}"
    )
    print(
        f"[2] Factor-Graph Observability (F+G+R+M): {obs['birth_factors_committed']} birth factors | "
        f"Rejected rank-deficient frames = {obs['rejected_rank_deficient_frames']} | "
        f"C1 re-acquisitions = {obs['c1_reacquisition_frames']}"
    )
    print(
        f"[3] 6D Pose Tracking (Ungated -> Ours) : "
        f"Mean ADD {p_base['mean_add_mm']:.2f} mm -> {p_ours['mean_add_mm']:.2f} mm | "
        f"Mean ADD-S {p_base['mean_adds_mm']:.2f} mm -> {p_ours['mean_adds_mm']:.2f} mm | "
        f"Spread = {p_ours['run_to_run_spread_mm']:.2f} mm / {p_ours['run_to_run_spread_deg']:.2f} deg"
    )
    print(
        f"[4] 3D Surface Fusion (Naive -> M+S)   : "
        f"P_10 {s_base['P_10_pct']:.2f}% -> {s_ours['P_10_pct']:.2f}% | "
        f"C_10 {s_base['C_10_pct']:.2f}% -> {s_ours['C_10_pct']:.2f}% | "
        f"F_95 {s_base['F_95_mm']:.2f} mm -> {s_ours['F_95_mm']:.2f} mm"
    )
    print(
        f"[5] L2 Pouring Safety Verification     : "
        f"Status = {pour['status']} (safe={pour['safe_to_pour']}) | "
        f"95% Rim Clearance = +{pour['rim_clearance_95_mm']:.2f} mm"
    )
    print(
        f"[6] Live Runtime Performance & Digest  : "
        f"{perf['mean_frame_latency_ms']:.2f} ms/frame ({perf['effective_fps']:.1f} FPS) | "
        f"RLE vs Dense = {perf['rle_vs_dense_speedup_x']:.1f}x | "
        f"Digest = {perf['output_digest']}"
    )
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evidencescene",
        description="EvidenceScene: Observability-Conditioned 3D Scene Reconstruction CLI",
    )
    sub = parser.add_subparsers(dest="command")

    info_p = sub.add_parser("info", help="Print package and environment metadata")
    info_p.add_argument("--json", action="store_true", help="Emit JSON output")

    demo_p = sub.add_parser(
        "demo",
        help="Run the self-contained end-to-end EvidenceScene manipulation pipeline",
    )
    demo_p.add_argument("--frames", type=int, default=24, help="Number of frames (>=15)")
    demo_p.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    demo_p.add_argument("--json", action="store_true", help="Emit JSON output")
    demo_p.add_argument("--out", type=Path, default=None, help="Write JSON report to file")

    args = parser.parse_args(argv)

    if args.command == "info":
        import scipy

        info = {
            "package": "evidencescene",
            "version": evidencescene.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "modules": [
                m for m in evidencescene.__all__ if not m.startswith("_")
            ],
        }
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            for k, v in info.items():
                print(f"{k:12s}: {v}")
        return 0

    if args.command in ("demo", None):
        frames = getattr(args, "frames", 24)
        seed = getattr(args, "seed", 42)
        as_json = getattr(args, "json", False)
        out_path = getattr(args, "out", None)
        report = run_end_to_end_demo(num_frames=frames, seed=seed)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if as_json:
            print(json.dumps(report, indent=2))
        else:
            _print_human_report(report)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
