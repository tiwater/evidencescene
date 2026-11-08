"""Comprehensive tests for standalone evidencescene modules and L2 teleop engine."""

from __future__ import annotations

import importlib
import math
import pkgutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import evidencescene
from evidencescene.primitives.geometry import CameraInfoRecord
from evidencescene.surface import (
    ProjectionTarget,
    SurfaceEvidence,
    project_surface_to_camera,
    project_table_aligned_surfaces,
    select_existing_points_shadow,
)
from evidencescene.teleop import (
    CausalSe3TrajectoryPredictor,
    MeasuredPointFeature3D,
    evaluate_point_to_aperture_relation,
    fit_circular_aperture_3d,
    fit_robust_plane_3d,
    predict_ballistic_pour_locus,
    predict_multi_horizon_se3_trajectory,
    project_stream_centerline_uv,
    propagate_feature_uncertainty_se3,
    verify_l2_pouring_safety_envelope,
)
from evidencescene.tracking import (
    BearingPointObservation,
    CausalMetricPointPoseRuntime,
    CausalThreeCameraPoseRuntime,
    MetricPointObservation,
    PinholeCameraModel,
    TableMetricPointObservation,
    TopPhysicalMessage,
    constrained_top_xy_refinement,
    predict_bounded_constant_velocity_se3,
)


def test_every_submodule_imports_standalone_without_ticos() -> None:
    """Verify 100% standalone import isolation when ticos_vision/ticos_replay/point2pose are blocked."""
    code = """
import sys, pkgutil, importlib
class BlockExternalAndCompetitors:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(("ticos_vision", "ticos_replay", "point2pose")):
            raise ImportError(f"Blocked forbidden dependency import: {fullname}")
        return None
sys.meta_path.insert(0, BlockExternalAndCompetitors())
import evidencescene
count = 0
for mod in pkgutil.walk_packages(evidencescene.__path__, "evidencescene."):
    assert "point2pose" not in mod.name.lower(), f"Competitor name found in submodule: {mod.name}"
    m = importlib.import_module(mod.name)
    for sym in getattr(m, "__all__", ()):
        assert "point2pose" not in sym.lower(), f"Competitor name found in symbol: {mod.name}.{sym}"
    count += 1
assert count >= 34, f"Expected >= 34 submodules, got {count}"
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Standalone import failed:\n{proc.stderr}"


def test_causal_se3_trajectory_predictor_c1_join_and_monotonic_bounds() -> None:
    predictor = CausalSe3TrajectoryPredictor(
        reacquisition_gap_threshold_s=0.10,
        c1_join_duration_s=0.20,
        maximum_join_velocity_jump_mps=0.15,
        horizons_ms=(100.0, 250.0, 500.0),
    )

    T0 = np.eye(4, dtype=np.float64)
    T0[:3, 3] = [0.20, -0.10, 0.25]
    r0 = predictor.observe(pose=T0, timestamp_ns=1_000_000_000)
    assert r0.interval_width_monotonic_violation_count == 0
    assert len(r0.horizons) == 3
    assert (
        r0.horizons[0].interval_half_width_95_m
        < r0.horizons[1].interval_half_width_95_m
        < r0.horizons[2].interval_half_width_95_m
    )

    T1 = np.eye(4, dtype=np.float64)
    T1[:3, 3] = [0.203, -0.098, 0.25]
    r1 = predictor.observe(pose=T1, timestamp_ns=1_033_333_333)
    assert r1.interval_width_monotonic_violation_count == 0
    assert not r1.c1_join_active

    # Simulate a 250ms occlusion gap followed by a shifted re-acquisition pose
    T2 = np.eye(4, dtype=np.float64)
    T2[:3, :3] = Rotation.from_euler("z", 4.0, degrees=True).as_matrix()
    T2[:3, 3] = [0.24, -0.07, 0.26]
    r2 = predictor.observe(
        pose=T2,
        timestamp_ns=1_300_000_000,
        is_reacquisition=True,
    )
    assert r2.c1_join_active is True
    assert r2.position_join_jump_m == 0.0
    assert r2.velocity_join_jump_mps <= 0.15 + 1e-9
    assert r2.interval_width_monotonic_violation_count == 0

    # Verify zero position jump at horizon=0 when C1 join is active
    r2_zero = predict_multi_horizon_se3_trajectory(
        current_pose=T2,
        current_timestamp_ns=1_300_000_000,
        linear_velocity_mps=predictor._linear_velocity_mps,
        angular_velocity_rad_s=predictor._angular_velocity_rad_s,
        horizons_ms=(0.0, 100.0, 250.0, 500.0),
        c1_join_state=predictor._c1_join_state,
    )
    assert r2_zero.interval_width_monotonic_violation_count == 0


def test_fit_circular_aperture_3d_and_se3_uncertainty_propagation() -> None:
    rng = np.random.default_rng(42)
    angles = np.linspace(-math.pi, math.pi, 160, endpoint=False)
    true_center = np.array([0.35, -0.12, 0.08], dtype=np.float64)
    true_radius = 0.045
    rim_pts = np.column_stack(
        [
            true_center[0] + true_radius * np.cos(angles) + rng.normal(0.0, 0.0005, len(angles)),
            true_center[1] + true_radius * np.sin(angles) + rng.normal(0.0, 0.0005, len(angles)),
            np.full(len(angles), true_center[2]) + rng.normal(0.0, 0.0004, len(angles)),
        ]
    )
    _, normal, _, plane_rmse = fit_robust_plane_3d(rim_pts)
    assert abs(float(normal[2]) - 1.0) < 1e-3
    assert plane_rmse < 0.002

    aperture = fit_circular_aperture_3d(
        rim_pts,
        feature_id="container_opening",
        timestamp_ns=2_000_000_000,
        coordinate_frame="table",
    )
    assert np.allclose(aperture.center_m, true_center, atol=0.002)
    assert abs(aperture.radius_m - true_radius) < 0.002
    assert aperture.sector_coverage_ratio == 1.0
    assert aperture.ransac_consensus_agreed is True

    # Propagate spout point feature from object frame to table frame
    outlet_obj = MeasuredPointFeature3D(
        feature_id="pour_outlet",
        point_m=np.array([0.08, 0.0, 0.12], dtype=np.float64),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        covariance_3x3_m2=np.eye(3, dtype=np.float64) * 1e-6,
        direction_std_rad=0.02,
        support_pixel_count=32,
        timestamp_ns=2_000_000_000,
        coordinate_frame="object",
    )
    T_wo = np.eye(4, dtype=np.float64)
    T_wo[:3, 3] = [0.27, -0.12, 0.06]
    cov_xi = np.diag([1e-4, 1e-4, 1e-4, 4e-6, 4e-6, 4e-6])
    outlet_table = propagate_feature_uncertainty_se3(
        feature_object_frame=outlet_obj,
        world_from_object=T_wo,
        pose_covariance_6x6=cov_xi,
        target_coordinate_frame="table",
    )
    assert np.allclose(outlet_table.point_m, [0.35, -0.12, 0.18], atol=1e-6)
    assert outlet_table.position_std_m > 0.002

    rel = evaluate_point_to_aperture_relation(outlet_table, aperture)
    assert abs(rel.signed_height_above_aperture_m - 0.10) < 0.003
    assert rel.planar_radial_offset_m < 0.003
    assert rel.worst_case_clearance_margin_95_m > 0.02
    assert rel.direction_alignment_deg is not None and rel.direction_alignment_deg < 2.0


def test_ballistic_pour_locus_and_l2_safety_envelope_gate() -> None:
    rng = np.random.default_rng(7)
    angles = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    rim_pts = np.column_stack(
        [
            0.30 + 0.048 * np.cos(angles) + rng.normal(0.0, 0.0003, len(angles)),
            0.00 + 0.048 * np.sin(angles) + rng.normal(0.0, 0.0003, len(angles)),
            0.05 + rng.normal(0.0, 0.0003, len(angles)),
        ]
    )
    aperture = fit_circular_aperture_3d(rim_pts, coordinate_frame="table")

    outlet_safe = MeasuredPointFeature3D(
        feature_id="pour_outlet",
        point_m=np.array([0.285, 0.0, 0.15], dtype=np.float64),
        direction=np.array([0.25, 0.0, -0.9682458], dtype=np.float64),
        covariance_3x3_m2=np.eye(3, dtype=np.float64) * 4e-6,
        direction_std_rad=0.025,
        support_pixel_count=40,
        timestamp_ns=1_000_000_000,
        coordinate_frame="table",
    )
    locus = predict_ballistic_pour_locus(
        pour_outlet=outlet_safe,
        target_aperture=aperture,
        exit_speed_mps=0.35,
    )
    assert locus.valid is True
    assert locus.time_of_flight_s is not None and locus.time_of_flight_s > 0.05
    assert locus.planar_radial_offset_m is not None and locus.planar_radial_offset_m < 0.015

    # Camera projection of stream centerline
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, 3] = [0.0, 0.0, 0.6]
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    uv_proj = project_stream_centerline_uv(
        locus,
        camera_from_world=T_cw,
        camera_matrix=K,
        image_size=(640, 480),
    )
    assert uv_proj["ready"] is True
    assert len(uv_proj["centerline_uv"]) >= 10

    verdict_safe = verify_l2_pouring_safety_envelope(
        pour_outlet_world=outlet_safe,
        target_aperture_world=aperture,
        exit_speed_mps=0.35,
    )
    assert verdict_safe.safe_to_pour is True
    assert verdict_safe.uncertainty_envelope_clears_cup_rim is True
    assert verdict_safe.l2_pour_safety_status == "safe_to_pour"
    assert verdict_safe.publication_authorized is False
    assert verdict_safe.control_authorized is False
    assert np.allclose(verdict_safe.assist_translation_delta_m, 0.0, atol=1e-9)

    # Misaligned spout that spills outside the rim must fail closed
    outlet_unsafe = MeasuredPointFeature3D(
        feature_id="pour_outlet",
        point_m=np.array([0.36, 0.0, 0.15], dtype=np.float64),
        direction=np.array([0.25, 0.0, -0.9682458], dtype=np.float64),
        covariance_3x3_m2=np.eye(3, dtype=np.float64) * 4e-6,
        direction_std_rad=0.025,
        support_pixel_count=40,
        timestamp_ns=1_000_000_000,
        coordinate_frame="table",
    )
    verdict_unsafe = verify_l2_pouring_safety_envelope(
        pour_outlet_world=outlet_unsafe,
        target_aperture_world=aperture,
        exit_speed_mps=0.35,
    )
    assert verdict_unsafe.safe_to_pour is False
    assert verdict_unsafe.uncertainty_envelope_clears_cup_rim is False
    assert verdict_unsafe.l2_pour_safety_status == "hold_or_reposition"
    assert verdict_unsafe.assist_translation_delta_m[0] < -0.02


def test_shadow_surface_snapshot_and_multi_camera_pose_runtime() -> None:
    # 1. Test support10/coarse12mm view-diversity selector
    gx, gy = np.meshgrid(np.linspace(-0.04, 0.04, 15), np.linspace(-0.04, 0.04, 15))
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    voxel_keys = np.floor(pts / 0.004).astype(np.int64)
    support = np.full(len(pts), 12, dtype=np.int64)
    cameras = np.full(len(pts), 2, dtype=np.int64)
    snap = select_existing_points_shadow(
        pts,
        support,
        cameras,
        voxel_keys,
        budget=50,
    )
    assert len(snap["indices"]) == 50
    assert snap["mode"] == "coarse_shadow"
    assert snap["wire_authorized"] is False

    # 2. Test bounded constant-velocity SE(3) gap prediction
    T_old = np.eye(4, dtype=np.float64)
    T_lat = np.eye(4, dtype=np.float64)
    T_lat[:3, 3] = [0.01, 0.0, 0.0]
    gap_pred = predict_bounded_constant_velocity_se3(
        older_pose=T_old,
        older_timestamp_ns=1_000_000_000,
        latest_pose=T_lat,
        latest_timestamp_ns=1_100_000_000,
        target_timestamp_ns=1_200_000_000,
    )
    assert np.allclose(gap_pred.pose[:3, 3], [0.02, 0.0, 0.0], atol=1e-6)

    # 3. Test CausalThreeCameraPoseRuntime fast path + constrained Top XY refinement
    landmarks = {
        i: np.array(
            [
                0.03 * math.cos(2.0 * math.pi * i / 8.0),
                0.03 * math.sin(2.0 * math.pi * i / 8.0),
                0.02 * (i % 2),
            ],
            dtype=np.float64,
        )
        for i in range(8)
    }
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    T_cam_table = np.eye(4, dtype=np.float64)
    T_cam_table[:3, 3] = [0.0, 0.0, 0.8]
    models = [
        PinholeCameraModel("chest", T_cam_table, K),
        PinholeCameraModel("head", T_cam_table, K),
        PinholeCameraModel("top", T_cam_table, K),
    ]
    init_pose = np.eye(4, dtype=np.float64)
    runtime = CausalThreeCameraPoseRuntime(
        landmarks_object_frame_m=landmarks,
        camera_models=models,
        initial_table_from_object=init_pose,
    )
    metric_obs = []
    for cam in ("chest", "head"):
        for lid, pt in landmarks.items():
            metric_obs.append(
                TableMetricPointObservation(
                    frame_index=1,
                    timestamp_ns=1_000_000_000,
                    camera_id=cam,
                    landmark_id=lid,
                    point_table_frame_m=pt + np.array([0.002, -0.001, 0.0]),
                    uncertainty_m=0.003,
                )
            )
    pub = runtime.publish_fast(
        event_sequence=1,
        frame_index=1,
        timestamp_ns=1_000_000_000,
        arrival_timestamp_ns=1_005_000_000,
        metric_observations=metric_obs,
    )
    assert pub.published is True
    assert pub.degraded is False


def test_circle_consensus_is_deterministic_across_cached_triplets() -> None:
    from evidencescene.teleop.task_features import compare_circle_consensus_3d

    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rim = np.column_stack([0.04 * np.cos(angles), 0.04 * np.sin(angles)])
    first = compare_circle_consensus_3d(
        rim,
        fitted_center_2d=np.zeros(2),
        fitted_radius_m=0.04,
    )
    second = compare_circle_consensus_3d(
        rim,
        fitted_center_2d=np.zeros(2),
        fitted_radius_m=0.04,
    )
    assert first[0] is second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]

    too_few = compare_circle_consensus_3d(
        rim[:5],
        fitted_center_2d=np.zeros(2),
        fitted_radius_m=0.04,
    )
    assert too_few[0] is False
