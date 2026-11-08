"""Unit tests for standalone factor-graph observability gates (F, G, R, M),
6D pose & 3D surface evaluation metrics, O(R_A + R_B) RLE interval algebra,
dataset readers, standalone runtime deduplication, and the CLI entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evidencescene.cli import main as cli_main
from evidencescene.datasets import (
    FolderRgbdSequenceReader,
    generate_synthetic_manipulation_sequence,
)
from evidencescene.metrics import (
    align_initial_gauge,
    compute_add,
    compute_adds,
    compute_failure_inclusive_recall_auc,
    compute_geodesic_rotation_error_deg,
    compute_trajectory_determinism_spread,
    compute_translation_error_m,
    evaluate_pose_sequence,
    evaluate_surface_reconstruction,
    quantize_surface_grid,
)
from evidencescene.observability import (
    FeatureBirthLedger,
    build_se3_landmark_jacobians,
    evaluate_prior_connected_admission,
    evaluate_schur_observability_gate,
    verify_causal_pose_attribution,
)
from evidencescene.primitives import image_mask as image_mask_mod
from evidencescene.primitives.image_mask import (
    decode_mask_rle,
    encode_mask_rle,
    rle_area,
    rle_difference_runs,
    rle_intersection_area,
    rle_intersection_runs,
    rle_iou,
    rle_union_runs,
)
from evidencescene.runtime.request_evidence_deduplication import (
    install_request_evidence_deduplication,
)


class TestFactorGraphObservabilityGates:
    """Tests for the four-stage factor-graph observability gates (F, G, R, M)."""

    def test_gate_f_causal_pose_attribution(self) -> None:
        raw_pose = np.eye(4, dtype=np.float64)
        refined_pose = np.eye(4, dtype=np.float64)
        refined_pose[0, 3] = 0.002

        decision_ok = verify_causal_pose_attribution(
            current_frame_index=5,
            promoted_keyframe_index=5,
            raw_pose=raw_pose,
            refined_pose=refined_pose,
            raw_inlier_count=12,
            refined_inlier_count=16,
            raw_mean_residual_m=0.004,
            refined_mean_residual_m=0.002,
        )
        assert decision_ok.admitted is True
        assert decision_ok.used_refined_pose is True
        assert decision_ok.reason == "admitted_refined_pose"
        assert decision_ok.selected_pose is not None
        assert np.allclose(decision_ok.selected_pose, refined_pose)

        decision_mismatch = verify_causal_pose_attribution(
            current_frame_index=5,
            promoted_keyframe_index=4,
            raw_pose=raw_pose,
            refined_pose=refined_pose,
            raw_inlier_count=12,
            refined_inlier_count=16,
        )
        assert decision_mismatch.admitted is False
        assert decision_mismatch.reason == "historical_keyframe_attribution_mismatch"
        assert decision_mismatch.selected_pose is None

    def test_gate_g_prior_connected_admission(self) -> None:
        decision_conn = evaluate_prior_connected_admission(
            pose_ids=(0, 1, 2, 3, 4),
            landmark_ids=(10, 11, 12, 20, 21, 22),
            prior_pose_ids=(0,),
            pose_landmark_edges=[
                (0, 10),
                (0, 11),
                (0, 12),
                (1, 10),
                (1, 11),
                (1, 12),
                (2, 10),
                (2, 11),
                (2, 12),
                # Disconnected island on poses 3, 4 observing landmarks 20, 21, 22
                (3, 20),
                (3, 21),
                (3, 22),
                (4, 20),
                (4, 21),
                (4, 22),
            ],
            candidate_pose_id=2,
            minimum_anchored_landmarks=3,
        )
        assert decision_conn.admitted is True
        assert decision_conn.candidate_connected_to_prior is True
        assert decision_conn.anchored_pose_ids == (0, 1, 2)
        assert decision_conn.unanchored_pose_ids == (3, 4)

        decision_floating = evaluate_prior_connected_admission(
            pose_ids=(0, 1, 2, 3, 4),
            landmark_ids=(10, 11, 12, 20, 21, 22),
            prior_pose_ids=(0,),
            pose_landmark_edges=[
                (0, 10),
                (0, 11),
                (0, 12),
                (3, 20),
                (3, 21),
                (3, 22),
                (4, 20),
                (4, 21),
                (4, 22),
            ],
            candidate_pose_id=4,
            minimum_anchored_landmarks=3,
        )
        assert decision_floating.admitted is False
        assert decision_floating.candidate_connected_to_prior is False
        assert decision_floating.reason == "disconnected_from_initial_prior_gauge"

    def test_gate_r_schur_rank_theorem_0_1_2_3_landmarks(self) -> None:
        """Verify theoretical SE(3) landmark observability rank r_i in {0, 3, 5, 6}."""
        pose0 = np.eye(4, dtype=np.float64)
        pose1 = np.eye(4, dtype=np.float64)
        pose1[:3, 3] = [0.05, -0.02, 0.55]

        lms = {
            0: np.array([-0.04, 0.0, 0.0], dtype=np.float64),
            1: np.array([0.00, 0.0, 0.0], dtype=np.float64),
            2: np.array([0.04, 0.0, 0.0], dtype=np.float64),
            3: np.array([0.00, 0.05, 0.02], dtype=np.float64),
        }
        anchor_obs = [(0, lid, 0.002) for lid in lms]

        # 0 landmarks observed by candidate pose 1 -> rank 0
        ax0, ay0, _, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=lms,
            pose_landmark_observations=anchor_obs,
            prior_pose_ids=(0,),
        )
        res0 = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax0,
            landmark_jacobian_ay=ay0,
            candidate_pose_index=1,
        )
        assert res0.admitted is False
        assert res0.marginal_pose_rank == 0

        # 1 landmark observed by candidate pose 1 -> rank 3 (translation only)
        ax1, ay1, _, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=lms,
            pose_landmark_observations=anchor_obs + [(1, 0, 0.002)],
            prior_pose_ids=(0,),
        )
        res1 = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax1,
            landmark_jacobian_ay=ay1,
            candidate_pose_index=1,
        )
        assert res1.admitted is False
        assert res1.marginal_pose_rank == 3

        # 2 distinct landmarks observed by candidate pose 1 -> rank 5
        ax2, ay2, _, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=lms,
            pose_landmark_observations=anchor_obs + [(1, 0, 0.002), (1, 1, 0.002)],
            prior_pose_ids=(0,),
        )
        res2 = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax2,
            landmark_jacobian_ay=ay2,
            candidate_pose_index=1,
        )
        assert res2.admitted is False
        assert res2.marginal_pose_rank == 5

        # 3 collinear landmarks (0, 1, 2) -> rank 5 < 6 (rejected by Schur gate!)
        ax3_col, ay3_col, _, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=lms,
            pose_landmark_observations=anchor_obs
            + [(1, 0, 0.002), (1, 1, 0.002), (1, 2, 0.002)],
            prior_pose_ids=(0,),
        )
        res3_col = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax3_col,
            landmark_jacobian_ay=ay3_col,
            candidate_pose_index=1,
        )
        assert res3_col.admitted is False
        assert res3_col.marginal_pose_rank == 5
        assert res3_col.reason.startswith("rank_deficient_pose_update")

        # 3 non-collinear landmarks (0, 2, 3) -> full rank 6 across all tolerance scales
        ax3_ok, ay3_ok, _, _ = build_se3_landmark_jacobians(
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=lms,
            pose_landmark_observations=anchor_obs
            + [(1, 0, 0.002), (1, 2, 0.002), (1, 3, 0.002)],
            prior_pose_ids=(0,),
        )
        res3_ok = evaluate_schur_observability_gate(
            pose_jacobian_ax=ax3_ok,
            landmark_jacobian_ay=ay3_ok,
            candidate_pose_index=1,
        )
        assert res3_ok.admitted is True
        assert res3_ok.marginal_pose_rank == 6
        assert dict(res3_ok.rank_by_tolerance_scale) == {"0.1x": 6, "1x": 6, "10x": 6}
        assert res3_ok.translational_min_singular_value > 0.0
        assert res3_ok.pose_marginal_information_6x6.shape == (6, 6)
        assert res3_ok.translational_schur_block_3x3.shape == (3, 3)

    def test_gate_m_monotonic_feature_birth_ledger(self) -> None:
        ledger = FeatureBirthLedger(object_id="mug")
        rec10 = ledger.record_birth_observation(
            birth_frame_index=1,
            track_id=10,
            point_camera_m=[0.01, -0.02, 0.50],
            uncertainty_m=0.003,
        )
        assert rec10.birth_frame_index == 1
        assert ledger.recorded_track_count == 1

        # Re-recording the same track_id preserves the original birth observation
        rec10_dup = ledger.record_birth_observation(
            birth_frame_index=4,
            track_id=10,
            point_camera_m=[0.09, 0.09, 0.70],
            uncertainty_m=0.010,
        )
        assert rec10_dup == rec10
        assert ledger.recorded_track_count == 1

        commits = ledger.emit_birth_factors(
            admit_frame_index=3,
            instantiated_track_to_landmark={10: 100},
        )
        assert len(commits) == 1
        assert commits[0].birth_frame_index == 1
        assert commits[0].admit_frame_index == 3
        assert commits[0].landmark_id == 100
        assert ledger.committed_factor_count == 1

        # Emitting again for the same track is deduplicated (returns empty list)
        commits_dup = ledger.emit_birth_factors(
            admit_frame_index=5,
            instantiated_track_to_landmark={10: 100},
        )
        assert commits_dup == []


class TestPoseAndSurfaceMetrics:
    """Tests for 6D pose (ADD, ADD-S, AUC, spread) and 3D surface metrics."""

    def test_pose_metrics_and_gauge_alignment(self) -> None:
        pts = np.array(
            [
                [-0.03, -0.02, -0.01],
                [0.03, -0.02, 0.01],
                [0.00, 0.04, -0.02],
                [-0.01, 0.02, 0.05],
            ],
            dtype=np.float64,
        )
        gt0 = np.eye(4, dtype=np.float64)
        gt0[:3, 3] = [0.1, -0.05, 0.6]
        gt1 = np.eye(4, dtype=np.float64)
        gt1[:3, 3] = [0.12, -0.04, 0.61]

        # Right-multiplying gauge offset G_delta on object frame
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = [0.05, -0.02, 0.01]
        pred0 = gt0 @ delta
        pred1 = gt1 @ delta

        gauge = align_initial_gauge(pred0, gt0)
        aligned1 = pred1 @ gauge
        assert np.allclose(aligned1, gt1, atol=1e-12)

        assert compute_add(aligned1, gt1, pts) < 1e-12
        assert compute_adds(aligned1, gt1, pts) < 1e-12
        assert compute_geodesic_rotation_error_deg(aligned1, gt1) < 1e-7
        assert compute_translation_error_m(aligned1, gt1) < 1e-12

        summary = evaluate_pose_sequence(
            estimated_poses=[pred0, pred1, None],
            reference_poses=[gt0, gt1, gt1],
            model_points_m=pts,
            align_frame_zero=True,
            exclude_frame_zero=True,
        )
        assert summary.total_frames == 3
        assert summary.denominator_excluding_initial == 2
        assert summary.available_post_initial_count == 1
        assert abs(summary.candidate_duty_cycle - 0.5) < 1e-9
        assert abs(summary.add_auc_0_to_max - 0.5) < 1e-6
        assert abs(summary.adds_auc_0_to_max - 0.5) < 1e-6

    def test_failure_inclusive_auc_and_determinism_spread(self) -> None:
        auc_perfect = compute_failure_inclusive_recall_auc(
            [0.0, 0.0], denominator=2, max_threshold_m=0.1
        )
        assert abs(auc_perfect - 1.0) < 1e-9

        auc_half_missing = compute_failure_inclusive_recall_auc(
            [0.0], denominator=2, max_threshold_m=0.1
        )
        assert abs(auc_half_missing - 0.5) < 1e-9

        traj = [np.eye(4, dtype=np.float64) for _ in range(5)]
        spread = compute_trajectory_determinism_spread(
            [traj, [p.copy() for p in traj], [p.copy() for p in traj]]
        )
        assert spread.max_translation_spread_mm == 0.0
        assert spread.max_rotation_spread_deg == 0.0
        assert spread.is_deterministic is True

    def test_surface_reconstruction_metrics(self) -> None:
        rng = np.random.default_rng(7)
        gt_pts = rng.uniform(-0.04, 0.04, size=(400, 3))
        quantized = quantize_surface_grid(gt_pts, voxel_size_m=0.004)
        assert 0 < quantized.shape[0] <= gt_pts.shape[0]

        res_exact = evaluate_surface_reconstruction(
            gt_pts, gt_pts, threshold_m=0.005, grid_voxel_size_m=0.004
        )
        assert res_exact.precision_pct == 100.0
        assert res_exact.coverage_pct == 100.0
        assert res_exact.f1_pct == 100.0
        assert res_exact.forward_p95_mm == 0.0
        assert res_exact.reverse_p95_mm == 0.0
        assert res_exact.chamfer_l1_m == 0.0

        outliers = rng.uniform(0.20, 0.30, size=(100, 3))
        noisy_pts = np.vstack([gt_pts, outliers])
        res_noisy = evaluate_surface_reconstruction(
            noisy_pts, gt_pts, threshold_m=0.005, grid_voxel_size_m=0.004
        )
        assert res_noisy.coverage_pct == 100.0
        assert res_noisy.precision_pct < 90.0
        assert res_noisy.forward_p95_mm > 100.0
        assert res_noisy.reverse_p95_mm == 0.0


class TestRleIntervalSetAlgebra:
    """Verify O(R_A + R_B) RLE set algebra matches dense boolean operations exactly."""

    def test_rle_set_operations_match_dense_masks(self) -> None:
        rng = np.random.default_rng(1234)
        for _ in range(10):
            h, w = 24, 32
            mask_a = rng.random((h, w)) > 0.55
            mask_b = rng.random((h, w)) > 0.55

            rle_a = encode_mask_rle(mask_a)
            rle_b = encode_mask_rle(mask_b)

            assert rle_area(rle_a) == int(np.count_nonzero(mask_a))
            assert rle_area(rle_b) == int(np.count_nonzero(mask_b))

            inter_dense = mask_a & mask_b
            union_dense = mask_a | mask_b
            diff_dense = mask_a & (~mask_b)

            inter_rle = rle_intersection_runs(rle_a, rle_b)
            union_rle = rle_union_runs(rle_a, rle_b)
            diff_rle = rle_difference_runs(rle_a, rle_b)

            dec_inter = decode_mask_rle(inter_rle, width=w, height=h)
            dec_union = decode_mask_rle(union_rle, width=w, height=h)
            dec_diff = decode_mask_rle(diff_rle, width=w, height=h)
            assert dec_inter is not None and np.array_equal(dec_inter, inter_dense)
            assert dec_union is not None and np.array_equal(dec_union, union_dense)
            assert dec_diff is not None and np.array_equal(dec_diff, diff_dense)

            inter_area = int(np.count_nonzero(inter_dense))
            union_area = int(np.count_nonzero(union_dense))
            expected_iou = inter_area / union_area if union_area > 0 else 1.0

            assert rle_intersection_area(rle_a, rle_b) == inter_area
            assert abs(rle_iou(rle_a, rle_b) - expected_iou) < 1e-12


class TestDatasetsRuntimeAndCli:
    """Tests for synthetic & folder datasets, runtime deduplication, and CLI."""

    def test_synthetic_sequence_and_folder_reader(self, tmp_path: Path) -> None:
        seq = generate_synthetic_manipulation_sequence(num_frames=16, seed=42)
        assert len(seq.frames) == 16
        assert seq.model_points_object_m.shape[1] == 3
        assert seq.frames[6].is_rank_deficient_frame is True

        rgb_dir = tmp_path / "rgb"
        depth_dir = tmp_path / "depth"
        mask_dir = tmp_path / "masks"
        for d in (rgb_dir, depth_dir, mask_dir):
            d.mkdir(parents=True)

        k_mat = np.array([[300.0, 0.0, 32.0], [0.0, 300.0, 24.0], [0.0, 0.0, 1.0]])
        np.savetxt(tmp_path / "cam_K.txt", k_mat)

        for idx in range(2):
            stem = f"{idx:06d}"
            rgb = np.full((48, 64, 3), 120 + idx * 10, dtype=np.uint8)
            depth_mm = np.full((48, 64), 550 + idx * 5, dtype=np.uint16)
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[12:36, 16:48] = 255

            Image.fromarray(rgb).save(rgb_dir / f"{stem}.png")
            Image.fromarray(depth_mm).save(depth_dir / f"{stem}.png")
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")

        reader = FolderRgbdSequenceReader(tmp_path)
        assert len(reader) == 2
        frames = list(reader)
        assert len(frames) == 2
        assert frames[0].color_rgb.shape == (48, 64, 3)
        assert int(frames[0].depth_mm[20, 20]) == 550
        assert bool(frames[0].mask[20, 20]) is True

    def test_standalone_request_evidence_deduplication(self) -> None:
        mask = np.zeros((32, 48), dtype=bool)
        mask[8:24, 10:30] = True
        rle = encode_mask_rle(mask)

        with install_request_evidence_deduplication(mode="validate") as stats:
            m1 = image_mask_mod.decode_mask_rle(rle, width=48, height=32)
            m2 = image_mask_mod.decode_mask_rle(rle, width=48, height=32)
            assert m1 is not None and m2 is not None
            assert np.array_equal(m1, mask)
            assert np.array_equal(m2, mask)
            assert stats.rle_decode_cache_misses == 1
            assert stats.rle_decode_cache_hits == 1
            assert stats.validation_mismatches == 0

    def test_cli_demo_and_info(self, tmp_path: Path) -> None:
        out_json = tmp_path / "demo_summary.json"
        rc_demo = cli_main(["demo", "--frames", "16", "--out", str(out_json)])
        assert rc_demo == 0
        assert out_json.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["observability_gate"]["rejected_rank_deficient_frames"] == [6]
        assert (
            payload["surface_evaluation"]["evidencescene_revisable_surfel_map"]["P_10_pct"]
            > payload["surface_evaluation"]["naive_unrevised_accumulation"]["P_10_pct"]
        )
        assert payload["teleop_pouring_safety"]["safe_to_pour"] is True
        perf = payload["runtime_performance"]
        assert perf["effective_fps"] > 0.0
        assert perf["rle_vs_dense_speedup_x"] > 0.0
        assert len(perf["output_digest"]) == 16

        rc_info = cli_main(["info", "--json"])
        assert rc_info == 0
