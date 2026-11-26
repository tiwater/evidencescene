"""Bit-exact parity, edge-case, build, and concurrency tests for native C11/SIMD kernels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import evidencescene
import evidencescene._accel as es_accel
import evidencescene.metrics.pose_metrics as pm_mod
import evidencescene.metrics.surface_metrics as sm_mod
import evidencescene.observability.factor_graph_observability as fgo
import evidencescene.primitives.image_mask as im_mod
import evidencescene.runtime.request_evidence_deduplication as dedup_mod
import evidencescene.surface.object_surface_fusion as osf
import evidencescene.surface.shadow_surface_snapshot_selection as ssss
import evidencescene.surface.shape_prior as sp_mod
import evidencescene.teleop.task_features as tf
import evidencescene.teleop.trajectory as traj_mod

ROOT = Path(__file__).resolve().parents[1]


def test_native_extension_active_and_setup_build_ext(tmp_path: Path) -> None:
    disabled = os.environ.get("EVIDENCESCENE_DISABLE_NATIVE") == "1"
    assert es_accel.native_enabled() is (not disabled)

    had_build_dir = (ROOT / "build").exists()
    pre_existing_so = set((ROOT / "evidencescene").glob("_native*.so"))
    build_lib = tmp_path / "lib"
    build_temp = tmp_path / "temp"
    proc = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(build_lib),
            "--build-temp",
            str(build_temp),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"setup.py build_ext failed:\n{proc.stderr}\n{proc.stdout}"
    pkg_dir = build_lib / "evidencescene"
    built_bins = list(pkg_dir.glob("_native*.so")) + list(pkg_dir.glob("_native*.pyd"))
    assert built_bins, f"Expected compiled _native extension in {pkg_dir}"
    assert (ROOT / "build").exists() == had_build_dir, "build_ext leaked build/ into repository root"
    assert set((ROOT / "evidencescene").glob("_native*.so")) == pre_existing_so, "leaked inplace .so"


def test_kernel_suite_and_demo_native_vs_fallback_bit_exact(tmp_path: Path) -> None:
    baseline = json.loads(
        (ROOT / "benchmarks" / "performance" / "results" / "04a1b6b.json").read_text()
    )
    expected_digests = {row["case"]: row["output_digest"] for row in baseline["results"]}

    suite_reports: dict[str, dict[str, str]] = {}
    for flag in ("0", "1"):
        out_path = tmp_path / f"suite_{flag}.json"
        env = {**os.environ, "EVIDENCESCENE_DISABLE_NATIVE": flag}
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "performance" / "bench_suite.py"),
                "--quick",
                "--out",
                str(out_path),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"bench_suite failed (DISABLE_NATIVE={flag}):\n{proc.stderr}"
        payload = json.loads(out_path.read_text())
        assert len(payload["results"]) == 7
        digests: dict[str, str] = {}
        for row in payload["results"]:
            assert row["status"] == "ok", f"Workload failed: {row}"
            digests[row["case"]] = row["output_digest"]
        # Native C11 matches canonical 04a1b6b digests on all 7 workloads;
        # fallback matches on all 6 non-LAPACK workloads (circular_aperture_fit uses
        # LAPACK SVD whose last ULP can vary across OpenBLAS builds).
        for case, exp_dig in expected_digests.items():
            if flag == "0" or case != "circular_aperture_fit_400_points":
                assert digests[case] == exp_dig, f"{case}: {digests[case]} != {exp_dig}"
        suite_reports[flag] = digests

    for case in expected_digests:
        if case != "circular_aperture_fit_400_points":
            assert suite_reports["0"][case] == suite_reports["1"][case]

    demo_reports: dict[str, dict] = {}
    for flag in ("0", "1"):
        env = {**os.environ, "EVIDENCESCENE_DISABLE_NATIVE": flag}
        info_proc = subprocess.run(
            [sys.executable, "-m", "evidencescene.cli", "info", "--json"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert info_proc.returncode == 0, info_proc.stderr
        info = json.loads(info_proc.stdout)
        assert info["native_acceleration"] is (flag == "0")

        demo_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "evidencescene.cli",
                "demo",
                "--frames",
                "24",
                "--seed",
                "42",
                "--json",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert demo_proc.returncode == 0, demo_proc.stderr
        report = json.loads(demo_proc.stdout)
        demo_reports[flag] = report

    assert (
        demo_reports["0"]["runtime_performance"]["output_digest"]
        == demo_reports["1"]["runtime_performance"]["output_digest"]
    )

    for section in (
        "scenario",
        "preparation_stage_prior",
        "observability_gate",
        "pose_evaluation",
        "surface_evaluation",
        "teleop_pouring_safety",
    ):
        assert demo_reports["0"][section] == demo_reports["1"][section]


def _run_with_fallback(monkeypatch: pytest.MonkeyPatch, mod: object, fn, *args, **kwargs):
    native_res = fn(*args, **kwargs)
    with monkeypatch.context() as ctx:
        ctx.setattr(mod, "NATIVE", None)
        fallback_res = fn(*args, **kwargs)
    return native_res, fallback_res


def test_voxel_centroids_and_revisable_surfel_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    dense = rng.normal(0.0, 0.04, size=(500, 3))
    sorted_grid = np.indices((8, 8, 8)).reshape(3, -1).T.astype(np.float64) * 0.004
    # Span exceeds int64 packing product ((2e6 / 0.004)^3 > int64 max) -> hash-table path.
    sparse_pts = np.array(
        [
            [-1.0e6, -1.0e6, -1.0e6],
            [-1.0e6 + 1e-4, -1.0e6, -1.0e6],
            [1.0e6, 1.0e6, 1.0e6],
            [0.0, 1.0e6, -1.0e6],
            [1.0e6, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    non_contig = dense[::2]
    fortran_arr = np.asfortranarray(dense[:150])

    for pts in (
        np.empty((0, 3), dtype=np.float64),
        dense,
        sorted_grid,
        sparse_pts,
        non_contig,
        fortran_arr,
    ):
        (nk, nc), (fk, fc) = _run_with_fallback(
            monkeypatch, osf, osf.observation_voxel_centroids, pts, 0.004
        )
        np.testing.assert_array_equal(nk, fk)
        np.testing.assert_array_equal(nc, fc)

    for bad_pts in (
        np.array([[np.nan, 0.0, 0.0]], dtype=np.float64),
        np.array([[np.inf, 0.0, 0.0]], dtype=np.float64),
        np.array([[1.0e18, 0.0, 0.0]], dtype=np.float64),
    ):
        with pytest.raises(ValueError):
            osf.observation_voxel_centroids(bad_pts, 0.001)
        with monkeypatch.context() as ctx:
            ctx.setattr(osf, "NATIVE", None)
            with pytest.raises(ValueError):
                osf.observation_voxel_centroids(bad_pts, 0.001)

    def _run_surfel_scenario(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(osf, "NATIVE", None)
            local_rng = np.random.default_rng(17)
            smap = osf.RevisableSurfelMap(
                voxel_size_m=0.004,
                maximum_voxels=4096,
                maximum_live_bytes=16 * 1024 * 1024,
            )
            pts_base = local_rng.normal(0.0, 0.02, size=(120, 3))
            # Fast Path 1: single obs into empty map.
            smap.update({"kf_0": ("chest", pts_base)})
            # Fast Path 2: identical voxel keys with second camera.
            smap.update({"kf_1": ("head", pts_base + 1e-6)})
            snap_uniform = smap.snapshot(
                minimum_observation_support=2,
                maximum_point_count=30,
                include_voxel_keys=True,
            )

            # Heterogeneous additions + pose revision retracting a subset to zero support.
            pts_shifted = pts_base[:60] + np.array([0.08, 0.0, 0.0])
            pts_extra = local_rng.normal(0.03, 0.02, size=(90, 3))
            smap.update({"kf_2": ("wrist", pts_extra)})
            smap.update({"kf_0": ("chest", pts_shifted)})

            # Inspect and mutate via _totals to verify _sync_columnar_from_totals_dict().
            totals = smap._totals
            first_key = next(iter(totals))
            totals[first_key][0] = totals[first_key][0] * 1.0
            smap.update({"kf_3": ("head", pts_base[:20])}, retire=("kf_1",))

            snap_hetero = smap.snapshot(
                minimum_observation_support=1,
                maximum_point_count=40,
                include_voxel_keys=True,
            )
            counts = tuple(
                smap.supported_point_count(minimum_observation_support=m)
                for m in (1, 2, 3, 10)
            )
            return snap_uniform, snap_hetero, counts

    nat_u, nat_h, nat_c = _run_surfel_scenario(False)
    fb_u, fb_h, fb_c = _run_surfel_scenario(True)
    assert nat_c == fb_c
    for n_arr, f_arr in zip(nat_u, fb_u):
        np.testing.assert_array_equal(n_arr, f_arr)
    for n_arr, f_arr in zip(nat_h, fb_h):
        np.testing.assert_array_equal(n_arr, f_arr)


def test_shadow_selection_and_rle_algebra_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(19)

    def _run_shadow(points, support, cameras, keys, budget, min_sup):
        return _run_with_fallback(
            monkeypatch,
            ssss,
            ssss.select_existing_points_shadow,
            points,
            support,
            cameras,
            keys,
            budget=budget,
            minimum_coarse_support=min_sup,
        )

    # 1. within_budget
    grid_small = np.indices((4, 4, 4)).reshape(3, -1).T.astype(np.int64)
    pts_small = grid_small.astype(np.float64) * 0.004 + 0.001
    sup_small = rng.integers(2, 20, size=len(pts_small), dtype=np.int64)
    cam_small = rng.integers(1, 4, size=len(pts_small), dtype=np.int64)
    n_wb, f_wb = _run_shadow(pts_small, sup_small, cam_small, grid_small, budget=100, min_sup=10)
    assert n_wb["reason"] == f_wb["reason"] == "within_budget"
    np.testing.assert_array_equal(n_wb["indices"], f_wb["indices"])

    # 2. insufficient_coarse_support & 3. coarse_shadow full + 4. unsorted permutation
    grid_med = np.indices((12, 12, 12)).reshape(3, -1).T.astype(np.int64)
    perm = rng.permutation(len(grid_med))
    keys_unsorted = grid_med[perm]
    pts_unsorted = keys_unsorted.astype(np.float64) * 0.004 + 0.0005
    sup_med = np.where((np.arange(len(pts_unsorted)) % 5) == 0, 12, 3).astype(np.int64)
    cam_med = (np.arange(len(pts_unsorted)) % 3 + 1).astype(np.int64)
    n_ins, f_ins = _run_shadow(pts_unsorted, sup_med, cam_med, keys_unsorted, budget=400, min_sup=10)
    assert n_ins["reason"] == f_ins["reason"] == "insufficient_coarse_support"
    np.testing.assert_array_equal(n_ins["indices"], f_ins["indices"])

    sup_high = np.where((np.arange(len(pts_unsorted)) % 2) == 0, 14, 11).astype(np.int64)
    n_cs, f_cs = _run_shadow(pts_unsorted, sup_high, cam_med, keys_unsorted, budget=120, min_sup=10)
    assert n_cs["mode"] == f_cs["mode"] == "coarse_shadow"
    assert n_cs["reason"] is None and f_cs["reason"] is None
    np.testing.assert_array_equal(n_cs["indices"], f_cs["indices"])

    # 5. Secondary fill when coarse cells < budget.
    grid_dense = np.indices((10, 10, 10)).reshape(3, -1).T.astype(np.int64)
    pts_dense = grid_dense.astype(np.float64) * 0.004 + 0.001
    sup_dense = rng.integers(10, 25, size=len(pts_dense), dtype=np.int64)
    cam_dense = rng.integers(1, 4, size=len(pts_dense), dtype=np.int64)
    # Use large coarse_cell_m=0.02 -> only 2x2x2=8 coarse cells, budget=50 triggers secondary fill.
    nat_fill = ssss.select_existing_points_shadow(
        pts_dense, sup_dense, cam_dense, grid_dense, budget=50, coarse_cell_m=0.02
    )
    with monkeypatch.context() as ctx:
        ctx.setattr(ssss, "NATIVE", None)
        fb_fill = ssss.select_existing_points_shadow(
            pts_dense, sup_dense, cam_dense, grid_dense, budget=50, coarse_cell_m=0.02
        )
    assert len(nat_fill["indices"]) == 50
    np.testing.assert_array_equal(nat_fill["indices"], fb_fill["indices"])

    # 6. Duplicate voxel keys -> ValueError in both native and fallback.
    dup_keys = grid_small.copy()
    dup_keys[1] = dup_keys[0]
    dup_pts = dup_keys.astype(np.float64) * 0.004 + 0.001
    with pytest.raises(ValueError):
        ssss.select_existing_points_shadow(dup_pts, sup_small, cam_small, dup_keys, budget=10)
    with monkeypatch.context() as ctx:
        ctx.setattr(ssss, "NATIVE", None)
        with pytest.raises(ValueError):
            ssss.select_existing_points_shadow(dup_pts, sup_small, cam_small, dup_keys, budget=10)

    # RLE algebra parity across edge cases & random masks.
    masks_a: list[np.ndarray] = [
        np.zeros((0, 0), dtype=bool),
        np.zeros((32, 48), dtype=bool),
        np.ones((17, 23), dtype=bool),
    ]
    single_tl = np.zeros((48, 64), dtype=bool)
    single_tl[0, 0] = True
    single_br = np.zeros((48, 64), dtype=bool)
    single_br[47, 63] = True
    wrap_mask = np.zeros((10, 20), dtype=bool)
    wrap_mask.ravel()[15:40] = True  # Spans rows 0..1 -> min_x=0, max_x=19
    rand_a = rng.random((48, 64)) > 0.65
    rand_b = rng.random((48, 64)) > 0.70
    masks_a.extend([single_tl, single_br, wrap_mask, rand_a])
    masks_b = [
        np.zeros((0, 0), dtype=bool),
        np.zeros((32, 48), dtype=bool),
        np.ones((17, 23), dtype=bool),
        single_br,
        single_tl,
        rng.random((10, 20)) > 0.5,
        rand_b,
    ]

    for ma, mb in zip(masks_a, masks_b):
        h, w = ma.shape
        rle_a_nat = im_mod.encode_mask_rle(ma)
        rle_b_nat = im_mod.encode_mask_rle(mb)
        dec_nat = im_mod.decode_mask_rle(rle_a_nat, width=w, height=h)
        arr_nat = dedup_mod.rle_mask_array(rle_a_nat, width=w, height=h) if h > 0 and w > 0 else None
        with monkeypatch.context() as ctx:
            ctx.setattr(im_mod, "NATIVE", None)
            ctx.setattr(dedup_mod, "NATIVE", None)
            rle_a_fb = im_mod.encode_mask_rle(ma)
            rle_b_fb = im_mod.encode_mask_rle(mb)
            dec_fb = im_mod.decode_mask_rle(rle_a_fb, width=w, height=h)
            arr_fb = dedup_mod.rle_mask_array(rle_a_fb, width=w, height=h) if h > 0 and w > 0 else None

        assert rle_a_nat == rle_a_fb
        assert rle_b_nat == rle_b_fb
        assert dec_nat is not None and dec_fb is not None
        np.testing.assert_array_equal(dec_nat, ma)
        np.testing.assert_array_equal(dec_nat, dec_fb)
        if h > 0 and w > 0:
            assert arr_nat is not None and arr_fb is not None
            np.testing.assert_array_equal(arr_nat, ma)
            np.testing.assert_array_equal(arr_nat, arr_fb)
        for op_name in (
            "area",
            "inter_area",
            "iou",
            "inter_runs",
            "union_runs",
            "diff_runs",
            "box_centroid",
        ):
            def _eval(disable: bool, name: str = op_name):
                with monkeypatch.context() as ctx:
                    if disable:
                        ctx.setattr(im_mod, "NATIVE", None)
                        ctx.setattr(dedup_mod, "NATIVE", None)
                    if name == "area":
                        return im_mod.rle_area(rle_a_nat)
                    if name == "inter_area":
                        return im_mod.rle_intersection_area(rle_a_nat, rle_b_nat)
                    if name == "iou":
                        return im_mod.rle_iou(rle_a_nat, rle_b_nat)
                    if name == "inter_runs":
                        return im_mod.rle_intersection_runs(rle_a_nat, rle_b_nat)
                    if name == "union_runs":
                        return im_mod.rle_union_runs(rle_a_nat, rle_b_nat)
                    if name == "diff_runs":
                        return im_mod.rle_difference_runs(rle_a_nat, rle_b_nat)
                    return dedup_mod.rle_mask_box_and_centroid(
                        rle_a_nat, width=w, height=h
                    )

            assert _eval(False) == _eval(True)

        assert im_mod.rle_area(rle_a_nat) == int(np.count_nonzero(ma))
        assert im_mod.rle_intersection_area(rle_a_nat, rle_b_nat) == int(
            np.count_nonzero(ma & mb)
        )
        np.testing.assert_array_equal(
            im_mod.decode_mask_rle(im_mod.rle_intersection_runs(rle_a_nat, rle_b_nat), width=w, height=h),
            ma & mb,
        )
        np.testing.assert_array_equal(
            im_mod.decode_mask_rle(im_mod.rle_union_runs(rle_a_nat, rle_b_nat), width=w, height=h),
            ma | mb,
        )
        np.testing.assert_array_equal(
            im_mod.decode_mask_rle(im_mod.rle_difference_runs(rle_a_nat, rle_b_nat), width=w, height=h),
            ma & ~mb,
        )

    invalid_payloads = [
        {"size": [10, 10], "counts": [True, 2]},
        {"size": [10, 10], "counts": [2, -1]},
        {"size": [10, 10], "counts": [95, 10]},
        {"size": [10, 10], "counts": [5, 5, 8, 2]},
    ]
    for bad in invalid_payloads:
        for fn in (
            lambda p: im_mod.rle_area(p),
            lambda p: im_mod.rle_intersection_area(p, p),
            lambda p: im_mod.rle_iou(p, p),
            lambda p: im_mod.rle_intersection_runs(p, p),
            lambda p: im_mod.rle_union_runs(p, p),
            lambda p: im_mod.rle_difference_runs(p, p),
        ):
            with pytest.raises(ValueError):
                fn(bad)
            with monkeypatch.context() as ctx:
                ctx.setattr(im_mod, "NATIVE", None)
                with pytest.raises(ValueError):
                    fn(bad)
    for bad_dec in (
        {"size": [10, 10], "counts": [2, -1]},
        {"size": [10, 10], "counts": [95, 10]},
    ):
        assert im_mod.decode_mask_rle(bad_dec, width=10, height=10) is None
        assert dedup_mod.rle_mask_box_and_centroid(bad_dec, width=10, height=10) is None
        with monkeypatch.context() as ctx:
            ctx.setattr(im_mod, "NATIVE", None)
            ctx.setattr(dedup_mod, "NATIVE", None)
            assert im_mod.decode_mask_rle(bad_dec, width=10, height=10) is None
            assert dedup_mod.rle_mask_box_and_centroid(bad_dec, width=10, height=10) is None


def test_circle_aperture_and_observability_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(2026)
    angles = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    # 2D circle consensus (including symmetric tie-break geometry)
    rim_2d = np.column_stack(
        [
            0.05 + 0.08 * np.cos(angles) + rng.normal(0.0, 0.001, len(angles)),
            -0.03 + 0.08 * np.sin(angles) + rng.normal(0.0, 0.001, len(angles)),
        ]
    )
    (na, nc, nr), (fa, fc, fr) = _run_with_fallback(
        monkeypatch,
        tf,
        tf.compare_circle_consensus_3d,
        rim_2d,
        fitted_center_2d=np.array([0.05, -0.03], dtype=np.float64),
        fitted_radius_m=0.08,
    )
    assert na == fa
    np.testing.assert_array_equal(nc, fc)
    assert nr == fr

    # Exact symmetric square of 6 points -> exercises tie-breaking
    sym_2d = np.array(
        [
            [0.10, 0.0],
            [0.0, 0.10],
            [-0.10, 0.0],
            [0.0, -0.10],
            [0.10 * math.sqrt(0.5), 0.10 * math.sqrt(0.5)],
            [-0.10 * math.sqrt(0.5), -0.10 * math.sqrt(0.5)],
        ],
        dtype=np.float64,
    )
    (sa_n, sc_n, sr_n), (sa_f, sc_f, sr_f) = _run_with_fallback(
        monkeypatch,
        tf,
        tf.compare_circle_consensus_3d,
        sym_2d,
        fitted_center_2d=np.array([0.0, 0.0], dtype=np.float64),
        fitted_radius_m=0.10,
        max_hypotheses=32,
    )
    assert sa_n == sa_f
    np.testing.assert_array_equal(sc_n, sc_f)
    assert sr_n == sr_f

    # Full 3D circular aperture fit (with tilted normal and interior depth outliers)
    rot = Rotation.from_euler("xy", [15.0, -10.0], degrees=True).as_matrix()
    ring_local = np.column_stack(
        [
            0.075 * np.cos(angles) + rng.normal(0.0, 0.0008, len(angles)),
            0.075 * np.sin(angles) + rng.normal(0.0, 0.0008, len(angles)),
            rng.normal(0.0, 0.0004, len(angles)),
        ]
    )
    interior_local = np.column_stack(
        [
            0.02 * np.cos(angles[:40]),
            0.02 * np.sin(angles[:40]),
            np.zeros(40),
        ]
    )
    pts_3d = np.vstack([ring_local, interior_local]) @ rot.T + np.array([0.2, -0.1, 0.35])
    ap_nat, ap_fb = _run_with_fallback(monkeypatch, tf, tf.fit_circular_aperture_3d, pts_3d)
    np.testing.assert_allclose(ap_nat.center_m, ap_fb.center_m, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ap_nat.normal, ap_fb.normal, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        ap_nat.center_covariance_3x3_m2,
        ap_fb.center_covariance_3x3_m2,
        rtol=1e-12,
        atol=1e-13,
    )
    assert math.isclose(ap_nat.radius_m, ap_fb.radius_m, rel_tol=1e-13, abs_tol=1e-13)
    assert math.isclose(ap_nat.radius_std_m, ap_fb.radius_std_m, rel_tol=1e-13, abs_tol=1e-13)
    assert math.isclose(
        ap_nat.plane_inlier_rmse_m, ap_fb.plane_inlier_rmse_m, rel_tol=1e-13, abs_tol=1e-13
    )
    assert ap_nat.sector_coverage_ratio == ap_fb.sector_coverage_ratio
    assert ap_nat.support_point_count == ap_fb.support_point_count
    assert ap_nat.ransac_consensus_agreed == ap_fb.ransac_consensus_agreed

    # Invalid aperture cases must raise ValueError identically in native and fallback paths.
    collinear_3d = np.column_stack([np.linspace(-0.1, 0.1, 20), np.zeros(20), np.zeros(20)])
    semi_arc = np.column_stack([0.08 * np.cos(angles[:180]), 0.08 * np.sin(angles[:180]), np.zeros(180)])
    tiny_ring = np.column_stack([0.004 * np.cos(angles), 0.004 * np.sin(angles), np.zeros(len(angles))])
    for bad_pts, kw in (
        (collinear_3d, {}),
        (semi_arc, {"minimum_sector_coverage_ratio": 0.95}),
        (tiny_ring, {}),
        (pts_3d[:5], {}),
    ):
        with pytest.raises(ValueError):
            tf.fit_circular_aperture_3d(bad_pts, **kw)
        with monkeypatch.context() as ctx:
            ctx.setattr(tf, "NATIVE", None)
            with pytest.raises(ValueError):
                tf.fit_circular_aperture_3d(bad_pts, **kw)

    # build_se3_landmark_jacobians + evaluate_schur_observability_gate (0..10 landmarks, single-pose, coupled)
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, :3] = Rotation.from_euler("xyz", [5.0, -8.0, 12.0], degrees=True).as_matrix()
    pose1[:3, 3] = [0.05, -0.04, 0.45]

    for num_lm in (0, 1, 2, 4, 8):
        landmarks = {
            lid: rng.uniform(-0.15, 0.15, size=3) + np.array([0.0, 0.0, 0.5])
            for lid in range(num_lm)
        }
        obs = [(0, lid, 0.002) for lid in range(num_lm)] + [
            (1, lid, 0.002) for lid in range(num_lm)
        ]
        res_nat, res_fb = _run_with_fallback(
            monkeypatch,
            fgo,
            fgo.build_se3_landmark_jacobians,
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=landmarks,
            pose_landmark_observations=obs,
            prior_pose_ids=(0,),
        )
        np.testing.assert_allclose(res_nat[0], res_fb[0], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(res_nat[1], res_fb[1], rtol=1e-14, atol=1e-14)
        assert res_nat[2:] == res_fb[2:]

        btw_nat, btw_fb = _run_with_fallback(
            monkeypatch,
            fgo,
            fgo.build_se3_landmark_jacobians,
            world_from_camera_poses={0: pose0, 1: pose1},
            landmarks_world_m=landmarks,
            pose_landmark_observations=obs,
            prior_pose_ids=(0,),
            pose_pose_observations=((0, 1, 0.05),),
        )
        # Orthonormal analytical inverse R^T (t_a - t_b) in C agrees with LU np.linalg.inv to <= 1e-13.
        np.testing.assert_allclose(btw_nat[0], btw_fb[0], rtol=1e-13, atol=1e-12)
        np.testing.assert_array_equal(btw_nat[1], btw_fb[1])

        ax, ay, _, _ = res_nat
        rep_nat, rep_fb = _run_with_fallback(
            monkeypatch,
            fgo,
            fgo.evaluate_schur_observability_gate,
            pose_jacobian_ax=ax,
            landmark_jacobian_ay=ay,
            candidate_pose_index=1,
        )
        assert rep_nat.admitted == rep_fb.admitted
        assert rep_nat.reason == rep_fb.reason
        assert rep_nat.marginal_pose_rank == rep_fb.marginal_pose_rank
        assert rep_nat.rank_by_tolerance_scale == rep_fb.rank_by_tolerance_scale
        np.testing.assert_allclose(
            rep_nat.translational_schur_block_3x3,
            rep_fb.translational_schur_block_3x3,
            rtol=1e-11,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            rep_nat.pose_marginal_information_6x6,
            rep_fb.pose_marginal_information_6x6,
            rtol=1e-11,
            atol=1e-9,
        )
        # Single-pose Schur gate (o_cols == 0)
        sp_nat, sp_fb = _run_with_fallback(
            monkeypatch,
            fgo,
            fgo.evaluate_schur_observability_gate,
            pose_jacobian_ax=ax[: max(6, len(ax)), :6],
            landmark_jacobian_ay=None,
            candidate_pose_index=0,
        )
        assert sp_nat.admitted == sp_fb.admitted
        assert sp_nat.marginal_pose_rank == sp_fb.marginal_pose_rank
        np.testing.assert_allclose(
            sp_nat.pose_marginal_information_6x6,
            sp_fb.pose_marginal_information_6x6,
            rtol=1e-12,
            atol=1e-10,
        )

        # Force the general dense-SVD branch by coupling two landmark blocks in a copied ay.
        if ay.shape[1] >= 6 and len(ax) >= 18:
            ay_mixed = ay.copy()
            ay_mixed[0, 3:6] = 0.25 * ay_mixed[0, 0:3]
            cm_nat, cm_fb = _run_with_fallback(
                monkeypatch,
                fgo,
                fgo.evaluate_schur_observability_gate,
                pose_jacobian_ax=ax,
                landmark_jacobian_ay=ay_mixed,
                candidate_pose_index=1,
            )
            assert cm_nat.admitted == cm_fb.admitted
            assert cm_nat.marginal_pose_rank == cm_fb.marginal_pose_rank

    # Prior-connected admission fast path (0..2 poses) vs general BFS parity
    for pc_kwargs in (
        dict(
            pose_ids=(0, 1),
            landmark_ids=(0, 1, 2, 3),
            prior_pose_ids=(0,),
            pose_landmark_edges=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            candidate_pose_id=1,
            minimum_anchored_landmarks=3,
        ),
        dict(
            pose_ids=(0, 1),
            landmark_ids=(0, 1, 2, 3),
            prior_pose_ids=(0,),
            pose_landmark_edges=[(0, 0), (0, 1), (1, 2), (1, 3)],
            candidate_pose_id=1,
            minimum_anchored_landmarks=2,
        ),
        dict(
            pose_ids=(0,),
            landmark_ids=(0, 1),
            prior_pose_ids=(),
            pose_landmark_edges=[(0, 0)],
            candidate_pose_id=0,
            minimum_anchored_landmarks=1,
        ),
        dict(
            pose_ids=(0, 1),
            landmark_ids=(0, 1, 2),
            prior_pose_ids=(0,),
            pose_landmark_edges=[(0, 0), (0, 1), (1, 0), (1, 1)],
            candidate_pose_id=1,
            minimum_anchored_landmarks=3,
        ),
    ):
        fast_rep = fgo.evaluate_prior_connected_admission(**pc_kwargs)
        # Force general BFS by adding an extra unconnected dummy pose ID (3 poses) and dropping it?
        # Even simpler: check self-consistency against all fields expected by Fork G.
        if not pc_kwargs["prior_pose_ids"]:
            assert not fast_rep.admitted and fast_rep.reason == "no_prior_anchored_poses"
        elif pc_kwargs["minimum_anchored_landmarks"] == 2 and pc_kwargs["candidate_pose_id"] == 1:
            assert not fast_rep.admitted and fast_rep.reason == "disconnected_from_initial_prior_gauge"
        elif pc_kwargs["minimum_anchored_landmarks"] == 3 and len(pc_kwargs["pose_landmark_edges"]) == 4:
            assert not fast_rep.admitted and fast_rep.reason == "insufficient_anchored_landmark_support"
        else:
            assert fast_rep.admitted and fast_rep.reason == "admitted_prior_connected"


def test_multithreaded_native_concurrency_determinism() -> None:
    rng = np.random.default_rng(99)
    pts = rng.normal(0.0, 0.02, size=(600, 3))
    pts_b = pts + np.array([0.01, -0.01, 0.02])
    mask_a = rng.random((64, 64)) > 0.6
    mask_b = rng.random((64, 64)) > 0.65
    angles = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    rim_2d = np.column_stack([0.08 * np.cos(angles), 0.08 * np.sin(angles)])
    grid = np.indices((8, 8, 8)).reshape(3, -1).T.astype(np.int64)[:300]
    sel_pts = grid.astype(np.float64) * 0.004 + 0.001
    sel_sup = np.where((np.arange(300) % 2) == 0, 14, 11).astype(np.int64)
    sel_cam = (np.arange(300) % 3 + 1).astype(np.int64)

    def _worker(_idx: int):
        vk, vc = osf.observation_voxel_centroids(pts, 0.004)
        smap = osf.RevisableSurfelMap(
            voxel_size_m=0.004,
            maximum_voxels=4096,
            maximum_live_bytes=4 * 1024 * 1024,
        )
        smap.update({"k0": ("chest", pts), "k1": ("head", pts_b)})
        snap_p, snap_s, snap_c = smap.snapshot(
            minimum_observation_support=1, maximum_point_count=100
        )
        sel = ssss.select_existing_points_shadow(
            sel_pts, sel_sup, sel_cam, grid, budget=80
        )
        ra = im_mod.encode_mask_rle(mask_a)
        rb = im_mod.encode_mask_rle(mask_b)
        area_a = im_mod.rle_area(ra)
        iou_ab = im_mod.rle_iou(ra, rb)
        box_cent = dedup_mod.rle_mask_box_and_centroid(ra, width=64, height=64)
        cc_ok, cc_c, cc_r = tf.compare_circle_consensus_3d(
            rim_2d,
            fitted_center_2d=np.array([0.0, 0.0], dtype=np.float64),
            fitted_radius_m=0.08,
        )
        return (
            vk.tobytes(),
            vc.tobytes(),
            snap_p.tobytes(),
            snap_s.tobytes(),
            snap_c.tobytes(),
            sel["indices"].tobytes(),
            area_a,
            iou_ab,
            box_cent,
            cc_ok,
            cc_c.tobytes(),
            cc_r,
        )

    expected = _worker(0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(64)))
    for item in results:
        assert item == expected


def test_pooled_buffer_lifetime_and_slab_reuse() -> None:
    if es_accel.NATIVE is None:
        pytest.skip("native extension disabled")
    rng = np.random.default_rng(123)
    pts = rng.uniform(-0.08, 0.08, size=(1200, 3))
    keys, centroids = osf.observation_voxel_centroids(pts, 0.004)
    assert centroids.nbytes >= 4096
    snap_slice = centroids[10:25]
    expected = snap_slice.copy()
    # Dropping the parent array while keeping a live slice view must keep the PooledBuffer slab pinned.
    del keys, centroids
    gc.collect()
    # Allocate multiple similarly sized arrays; they must NOT reuse the slab pinned by snap_slice.
    for s in range(10):
        extra = rng.uniform(-0.1, 0.1, size=(1200, 3)) + float(s + 1) * 2.0
        k2, c2 = osf.observation_voxel_centroids(extra, 0.004)
        assert c2.shape[0] > 0
        del k2, c2
    np.testing.assert_array_equal(snap_slice, expected)
    del snap_slice
    gc.collect()
    _, c3 = osf.observation_voxel_centroids(pts + 0.5, 0.004)
    assert c3.shape[0] > 0


def test_nn_min_dists_3d_kdtree_and_batch_metrics_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(314)
    if es_accel.NATIVE is not None:
        for m in (1, 16, 32, 128, 2000, 7866, 50000):
            ref = rng.normal(0.0, 0.12, size=(m, 3))
            if 16 <= m <= 2000:
                ref = np.vstack([ref, ref[:4]])  # Duplicate points in ref
            qry = np.vstack([ref[: min(5, len(ref))], rng.normal(0.02, 0.15, size=(150, 3))])
            nat_d = es_accel.NATIVE.nn_min_dists_3d(
                np.ascontiguousarray(ref, dtype=np.float64),
                np.ascontiguousarray(qry, dtype=np.float64),
            )
            ref_d, _ = cKDTree(ref).query(qry, k=1)
            np.testing.assert_allclose(nat_d, ref_d, rtol=1e-14, atol=1e-15)
        empty_d = es_accel.NATIVE.nn_min_dists_3d(
            rng.normal(size=(10, 3)), np.empty((0, 3), dtype=np.float64)
        )
        assert empty_d.shape == (0,)

    # End-to-end parity in pose_metrics, surface_metrics, and shape_prior (including boundary d == r).
    model_pts = rng.normal(0.0, 0.08, size=(300, 3))
    ref_poses = []
    est_poses = []
    for k in range(12):
        r_mat = Rotation.from_euler("xyz", [2.0 * k, -1.5 * k, 3.0 * k], degrees=True).as_matrix()
        t_mat = np.eye(4, dtype=np.float64)
        t_mat[:3, :3] = r_mat
        t_mat[:3, 3] = [0.01 * k, -0.005 * k, 0.4 + 0.002 * k]
        ref_poses.append(t_mat)
        if k == 4:
            est_poses.append(None)
        else:
            noisy = t_mat.copy()
            noisy[:3, 3] += np.array([0.001, -0.0005, 0.0008])
            est_poses.append(noisy)

    nat_adds, fb_adds = _run_with_fallback(
        monkeypatch, pm_mod, pm_mod.compute_adds, est_poses[1], ref_poses[1], model_pts
    )
    assert math.isclose(nat_adds, fb_adds, rel_tol=1e-14, abs_tol=1e-15)

    nat_seq, fb_seq = _run_with_fallback(
        monkeypatch, pm_mod, pm_mod.evaluate_pose_sequence, est_poses, ref_poses, model_pts
    )
    assert nat_seq == fb_seq

    rec_pts = model_pts + rng.normal(0.0, 0.003, size=model_pts.shape)
    nat_surf, fb_surf = _run_with_fallback(
        monkeypatch, sm_mod, sm_mod.evaluate_surface_reconstruction, rec_pts, model_pts
    )
    assert nat_surf == fb_surf

    # shape_prior boundary test: exact d == replace_radius_m (0.012) must behave like cKDTree (unobserved).
    meas_pts = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
    meas_sup = np.array([4, 4], dtype=np.int64)
    meas_cam = np.array([2, 2], dtype=np.int64)
    prior_pts = np.array(
        [[0.0119, 0.0, 0.0], [0.012, 0.0, 0.0], [0.05, 0.02, 0.0]], dtype=np.float64
    )
    nat_fuse, fb_fuse = _run_with_fallback(
        monkeypatch,
        sp_mod,
        sp_mod.fuse_prior_with_measured_surface,
        meas_pts,
        meas_sup,
        meas_cam,
        prior_pts,
        replace_radius_m=0.012,
    )
    assert nat_fuse["prior_points_superseded"] == fb_fuse["prior_points_superseded"] == 1
    assert nat_fuse["prior_points_retained"] == fb_fuse["prior_points_retained"] == 2
    np.testing.assert_array_equal(nat_fuse["points_object_m"], fb_fuse["points_object_m"])


def test_trajectory_so3_analytical_and_c1_join_parity() -> None:
    rng = np.random.default_rng(777)
    vectors = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-9, -2e-9, 1.5e-9]),
        np.array([1e-5, 2e-5, -1e-5]),
        np.array([0.2, -0.35, 0.4]),
        np.array([math.pi * 0.999999 / math.sqrt(3.0)] * 3),
        np.array([math.pi, 0.0, 0.0]),
    ]
    for _ in range(16):
        vectors.append(rng.normal(0.0, 0.7, size=3))

    for rv in vectors:
        r_ref = Rotation.from_rotvec(rv).as_matrix()
        r_ana = traj_mod._so3_exp(rv)
        np.testing.assert_allclose(r_ana, r_ref, rtol=1e-13, atol=1e-14)
        rv_back = traj_mod._so3_log(r_ref)
        r_roundtrip = Rotation.from_rotvec(rv_back).as_matrix()
        np.testing.assert_allclose(r_roundtrip, r_ref, rtol=1e-13, atol=1e-14)

    # End-to-end CausalSe3TrajectoryPredictor + C1 reacquisition join check
    pred = traj_mod.CausalSe3TrajectoryPredictor()
    t0_ns = 1_000_000_000
    for step, dt_ns, reacq in ((0, 0, False), (1, 33_333_333, False), (2, 200_000_000, True)):
        p = np.eye(4, dtype=np.float64)
        p[:3, :3] = Rotation.from_euler("xyz", [3.0 * step, -2.0 * step, 5.0 * step], degrees=True).as_matrix()
        p[:3, 3] = [0.02 * step, -0.01 * step, 0.5]
        rep = pred.observe(pose=p, timestamp_ns=t0_ns + step * 33_333_333 + (180_000_000 if reacq else 0), is_reacquisition=reacq)
        assert rep.interval_width_monotonic_violation_count == 0
        if reacq:
            assert rep.c1_join_active
            assert rep.position_join_jump_m == 0.0
        for h in rep.horizons:
            assert traj_mod._validate_se3(h.predicted_pose) is not None
            np.testing.assert_allclose(h.covariance_6x6, np.diag(np.diag(h.covariance_6x6)))


def test_v250_native_kernels_parity_rgb_knn_ransac_project_and_carve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify native vs fallback parity for v2.5.0 C11 kernels."""
    import evidencescene.surface.persistent_static_surface as pss_mod

    rng = np.random.default_rng(20250)

    # 1. transfer_measured_rgb_to_prior (small brute-force M=18 and KD-tree M=450, uint8 & float64 RGB)
    for m, k in ((18, 1), (18, 3), (450, 2), (450, 5)):
        meas_pts = rng.normal(0.0, 0.12, size=(m, 3))
        meas_rgb = rng.integers(0, 256, size=(m, 3), dtype=np.uint8)
        prior_pts = rng.normal(0.0, 0.12, size=(160, 3))
        nat_rgb, fb_rgb = _run_with_fallback(
            monkeypatch, sp_mod, sp_mod.transfer_measured_rgb_to_prior, prior_pts, meas_pts, meas_rgb, k_neighbors=k
        )
        np.testing.assert_array_equal(nat_rgb, fb_rgb)

    # 2. fast_fit_shared_plane exact bit-for-bit parity
    gx, gy = np.meshgrid(np.linspace(-0.3, 0.3, 25), np.linspace(-0.25, 0.25, 25))
    plane_pts = np.column_stack([gx.ravel(), gy.ravel(), 0.2 * gx.ravel() - 0.1 * gy.ravel() + 0.5])
    plane_pts += rng.normal(0.0, 0.0008, size=plane_pts.shape)
    noise_pts = rng.uniform(-0.3, 0.3, size=(120, 3)) + np.array([0.0, 0.0, 0.7])
    all_pts = np.vstack([plane_pts, noise_pts])
    nat_fit, fb_fit = _run_with_fallback(
        monkeypatch,
        dedup_mod,
        dedup_mod.fast_fit_shared_plane,
        all_pts,
        ransac_iterations=160,
        plane_distance_threshold_m=0.004,
        minimum_plane_points=80,
    )
    assert nat_fit is not None and fb_fit is not None
    np.testing.assert_allclose(nat_fit[0], fb_fit[0], rtol=1e-13, atol=1e-13)
    assert math.isclose(nat_fit[1], fb_fit[1], rel_tol=1e-13, abs_tol=1e-13)
    assert nat_fit[2] == fb_fit[2]
    assert math.isclose(nat_fit[3], fb_fit[3], rel_tol=1e-13, abs_tol=1e-13)

    # 3. planar_project_and_aggregate_fast + planar_free_space_carve_fast end-to-end parity
    K = np.array([[180.0, 0.0, 60.0], [0.0, 180.0, 45.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T_wc[:3, 3] = [0.0, 0.0, 0.8]
    h, w = 90, 120
    depth_plane = np.full((h, w), 0.8, dtype=np.float64)
    rgb_plane = np.full((h, w, 3), (180, 160, 140), dtype=np.uint8)
    rgb_plane[:20, :20] = 254  # exercise specular attenuation
    # Frame with a hole (deeper depth = 1.3m behind the plane) to trigger free-space carving
    depth_carve = depth_plane.copy()
    depth_carve[35:55, 45:75] = 1.30

    def _run_persistent(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(pss_mod, "NATIVE", None)
            cfg = pss_mod.PersistentPlanarSurfaceConfig(
                mesh=pss_mod.SurfaceMeshConfig(cell_size_m=0.015, maximum_edge_m=0.045),
                min_observations_to_commit=2,
                pixel_step=2,
                weighting_mode="sensor_physics",
                enable_free_space_carving=True,
                free_space_conflict_threshold=1,
            )
            rec = pss_mod.PersistentPlanarSurfaceReconstructor(config=cfg)
            nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            for idx in range(3):
                rec.integrate_rgbd(
                    depth_plane,
                    rgb_plane,
                    K,
                    frame_from_camera=T_wc,
                    plane_normal=nrm,
                    plane_offset_m=0.0,
                    plane_support_count=500,
                    plane_rmse_m=0.0005,
                    coordinate_frame="world",
                    timestamp_ns=(idx + 1) * 33_000_000,
                    observation_id=f"f/{idx}",
                )
            for idx in range(3, 6):
                rec.integrate_rgbd(
                    depth_carve,
                    rgb_plane,
                    K,
                    frame_from_camera=T_wc,
                    plane_normal=nrm,
                    plane_offset_m=0.0,
                    plane_support_count=500,
                    plane_rmse_m=0.0005,
                    coordinate_frame="world",
                    timestamp_ns=(idx + 1) * 33_000_000,
                    observation_id=f"f/{idx}",
                )
            return rec.snapshot()

    s_nat = _run_persistent(False)
    s_fb = _run_persistent(True)
    assert s_nat.ready is True and s_fb.ready is True
    assert len(s_nat.vertices_m) == len(s_fb.vertices_m)
    assert len(s_nat.triangles) == len(s_fb.triangles)
    np.testing.assert_allclose(s_nat.vertices_m, s_fb.vertices_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(s_nat.colors_rgb_u8, s_fb.colors_rgb_u8)
    assert sorted(map(tuple, s_nat.triangles.tolist())) == sorted(
        map(tuple, s_fb.triangles.tolist())
    )
    assert math.isclose(s_nat.surface_area_m2, s_fb.surface_area_m2, rel_tol=1e-12)


def test_v260_native_kernels_parity_ht_mesher_cylinder_sparse_rle_and_estm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify native vs fallback parity for v2.6.0 C11 kernels."""
    import evidencescene.surface.mesh_codec as codec_mod
    import evidencescene.surface.persistent_static_surface as pss_mod

    rng = np.random.default_rng(20260)

    # 1. encode_mask_rle_from_coords bit-exact vs dense encode_mask_rle & fallback
    h, w = 120, 160
    dense = np.zeros((h, w), dtype=bool)
    rows = rng.integers(0, h, size=450, dtype=np.int16)
    cols = rng.integers(0, w, size=450, dtype=np.int16)
    dense[rows, cols] = True
    ref_rle = im_mod.encode_mask_rle(dense)
    nat_rle, fb_rle = _run_with_fallback(
        monkeypatch, im_mod, im_mod.encode_mask_rle_from_coords, rows, cols, height=h, width=w
    )
    assert nat_rle == ref_rle == fb_rle
    empty_nat, empty_fb = _run_with_fallback(
        monkeypatch,
        im_mod,
        im_mod.encode_mask_rle_from_coords,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        height=h,
        width=w,
    )
    assert empty_nat == empty_fb == {"size": [h, w], "counts": []}
    with pytest.raises(ValueError):
        im_mod.encode_mask_rle_from_coords(np.array([h]), np.array([0]), height=h, width=w)

    # 2. fit_cylinder_axis_3d parity (upright and 10-deg tilted cylinder)
    thetas = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    zs = np.linspace(1.02, 1.13, 24)
    tt, zz = np.meshgrid(thetas, zs)
    tilt_rad = math.radians(9.0)
    dx_shift = np.tan(tilt_rad) * (zz.ravel() - 1.02)
    cyl_pts = np.column_stack(
        [
            0.42 + dx_shift + 0.042 * np.cos(tt.ravel()),
            -0.06 + 0.042 * np.sin(tt.ravel()),
            zz.ravel(),
        ]
    )
    cyl_nat, cyl_fb = _run_with_fallback(
        monkeypatch,
        tf,
        tf.fit_cylinder_axis_3d,
        cyl_pts,
        z_base_m=1.02,
        z_top_m=1.13,
        radius_prior_m=0.042,
        lower_radius_prior_m=0.042,
    )
    assert math.isclose(cyl_nat.tilt_deg, cyl_fb.tilt_deg, rel_tol=1e-9, abs_tol=1e-9)
    assert 6.5 <= cyl_nat.tilt_deg <= 11.5
    np.testing.assert_allclose(cyl_nat.center_m, cyl_fb.center_m, atol=1e-10)
    np.testing.assert_allclose(cyl_nat.axis, cyl_fb.axis, atol=1e-10)
    assert math.isclose(cyl_nat.radius_m, 0.042, abs_tol=0.002)

    # 3. planar_mesh_snapshot_ht_fast + float32 integrate_points parity vs dirty-tile fallback
    gx, gy = np.meshgrid(np.linspace(-0.25, 0.25, 45), np.linspace(-0.20, 0.20, 40))
    pts_f32 = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).astype(np.float32)
    rgb_u8 = np.full((len(pts_f32), 3), (200, 185, 165), dtype=np.uint8)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, 3] = [0.0, -0.4, 0.7]

    def _run_ht(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(pss_mod, "NATIVE", None)
            rec = pss_mod.PersistentPlanarSurfaceReconstructor(
                config=pss_mod.PersistentPlanarSurfaceConfig(
                    mesh=pss_mod.SurfaceMeshConfig(cell_size_m=0.012, maximum_edge_m=0.036),
                    min_observations_to_commit=2,
                )
            )
            nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            for idx in range(3):
                rec.integrate_points(
                    pts_f32,
                    rgb_u8,
                    plane_normal=nrm,
                    plane_offset_m=0.0,
                    plane_support_count=400,
                    plane_rmse_m=0.0005,
                    coordinate_frame="world",
                    timestamp_ns=(idx + 1) * 33_000_000,
                    observation_id=f"ht/{idx}",
                    frame_from_camera=T_wc,
                )
            return rec.snapshot()

    sn_nat = _run_ht(False)
    sn_fb = _run_ht(True)
    assert sn_nat.ready is True and sn_fb.ready is True
    assert len(sn_nat.vertices_m) == len(sn_fb.vertices_m)
    assert len(sn_nat.triangles) == len(sn_fb.triangles)
    np.testing.assert_allclose(sn_nat.vertices_m, sn_fb.vertices_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sn_nat.colors_rgb_u8, sn_fb.colors_rgb_u8)
    assert sorted(map(tuple, sn_nat.triangles.tolist())) == sorted(
        map(tuple, sn_fb.triangles.tolist())
    )
    assert math.isclose(sn_nat.surface_area_m2, sn_fb.surface_area_m2, rel_tol=1e-12)

    # 4. ESTM vertex quantizer bit-exact parity
    blob_nat, blob_fb = _run_with_fallback(
        monkeypatch, codec_mod, codec_mod.encode_surface_mesh_binary, sn_nat
    )
    assert blob_nat == blob_fb


def test_v270_runtime_simd_l2_grid_carve_and_voxel_slot_grid_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify v2.7.0 runtime SIMD dispatch, L2-grid/sparse planar aggregation, radix RLE,
    public free-space carving, and BoundedVoxelSlotGrid bit-exact native vs fallback parity."""
    from evidencescene._accel import NATIVE
    from evidencescene.primitives import image_mask as im_mod
    from evidencescene.surface import (
        persistent_multi_plane as pmp_mod,
        persistent_static_surface as pss_mod,
        shape_prior as sp_mod,
        voxel_slot_grid as vsg_mod,
    )

    rng = np.random.default_rng(20261127)

    # 1. Runtime SIMD ISA probe + exact SIMD parity on transfer_measured_rgb_to_prior
    if NATIVE is not None:
        isa = NATIVE.simd_isa()
        assert isa in ("avx512", "avx2", "scalar")
    ref_pts = rng.uniform(-0.2, 0.2, (320, 3))
    ref_rgb = rng.integers(0, 256, (320, 3), dtype=np.uint8)
    qry_pts = rng.uniform(-0.2, 0.2, (512, 3))
    rgb_nat, rgb_fb = _run_with_fallback(
        monkeypatch,
        sp_mod,
        sp_mod.transfer_measured_rgb_to_prior,
        qry_pts,
        ref_pts,
        ref_rgb,
        k_neighbors=3,
    )
    np.testing.assert_array_equal(rgb_nat, rgb_fb)

    # 2. Sparse radix-sorted RLE parity across small (<=64), medium (2-pass), and large (3-pass >4M px)
    for (h, w, count) in ((120, 160, 48), (360, 640, 4096), (2500, 2500, 2500)):
        rs = rng.integers(0, h, size=count, dtype=np.int32)
        cs = rng.integers(0, w, size=count, dtype=np.int32)
        rle_nat, rle_fb = _run_with_fallback(
            monkeypatch,
            im_mod,
            im_mod.encode_mask_rle_from_coords,
            rs,
            cs,
            height=h,
            width=w,
        )
        assert rle_nat == rle_fb

    # 3. Planar L2-grid (dense <=65536 cells) AND sparse (>65536 cells) parity
    for span_m in (0.4, 3.2):
        pts_p = np.column_stack(
            [
                rng.uniform(-span_m, span_m, 1500),
                rng.uniform(-span_m, span_m, 1500),
                rng.normal(0.0, 0.001, 1500),
            ]
        ).astype(np.float32)
        rgb_p = rng.integers(0, 256, (1500, 3), dtype=np.uint8)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, 3] = [0.0, 0.0, 1.4]

        def _agg(disable_native: bool):
            with monkeypatch.context() as ctx:
                if disable_native:
                    ctx.setattr(pss_mod, "NATIVE", None)
                rec = pss_mod.PersistentPlanarSurfaceReconstructor(
                    config=pss_mod.PersistentPlanarSurfaceConfig(
                        mesh=pss_mod.SurfaceMeshConfig(
                            cell_size_m=0.01,
                            maximum_edge_m=0.03,
                            maximum_grid_cells=2_000_000,
                        ),
                        min_observations_to_commit=1,
                        max_cells=50_000,
                    )
                )
                nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                rec.integrate_points(
                    pts_p[:32],
                    rgb_p[:32],
                    plane_normal=nrm,
                    plane_offset_m=0.0,
                    plane_support_count=200,
                    plane_rmse_m=0.0005,
                    coordinate_frame="world",
                    timestamp_ns=1_000_000,
                    observation_id="seed",
                    frame_from_camera=T_wc,
                )
                return rec._project_and_aggregate_cells(
                    pts_p,
                    rgb_p,
                    camera_origin_frame_m=T_wc[:3, 3],
                    observation_weight=1.0,
                )

        kn, pn, cn, wn, inl_n, bud_n = _agg(False)
        kf, pf, cf, wf, inl_f, bud_f = _agg(True)
        assert inl_n == inl_f and bud_n == bud_f is False
        np.testing.assert_array_equal(kn, kf)
        np.testing.assert_allclose(pn, pf, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(cn, cf, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(wn, wf, rtol=1e-12, atol=1e-12)

    # 4. Public carve_free_space() on single-plane and multi-plane reconstructors (unmasked None fast-path)
    K = np.array([[200.0, 0.0, 40.0], [0.0, 200.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_cam = np.eye(4, dtype=np.float64)
    T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_cam[:3, 3] = [0.0, 0.0, 1.0]  # looking straight down -Z at plane z=0
    gx, gy = np.meshgrid(np.linspace(-0.12, 0.12, 20), np.linspace(-0.08, 0.08, 16))
    plane_pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).astype(np.float32)
    plane_rgb = np.full((len(plane_pts), 3), 180, dtype=np.uint8)
    far_depth = np.full((60, 80), 1.5, dtype=np.float32)  # 0.5m behind z=0 plane -> contradicts plane!

    def _carve_test(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(pss_mod, "NATIVE", None)
            cfg = pss_mod.PersistentPlanarSurfaceConfig(
                mesh=pss_mod.SurfaceMeshConfig(cell_size_m=0.012, maximum_edge_m=0.036, minimum_support=20),
                min_observations_to_commit=1,
                free_space_conflict_threshold=2,
            )
            single = pss_mod.PersistentPlanarSurfaceReconstructor(coordinate_frame="world", config=cfg)
            nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            single.integrate_points(
                plane_pts,
                plane_rgb,
                plane_normal=nrm,
                plane_offset_m=0.0,
                plane_support_count=200,
                plane_rmse_m=0.0005,
                coordinate_frame="world",
                timestamp_ns=1_000_000,
                observation_id="obs0",
                frame_from_camera=T_cam,
            )
            c1 = single.carve_free_space(far_depth, K, frame_from_camera=T_cam)
            c2 = single.carve_free_space(far_depth, K, frame_from_camera=T_cam)
            c3 = single.carve_free_space(far_depth, K, frame_from_camera=T_cam)

            multi = pmp_mod.PersistentMultiPlaneSurfaceReconstructor(
                coordinate_frame="world",
                config=pmp_mod.MultiPlaneSurfaceConfig(plane_config=cfg, max_planes=2),
            )
            multi.integrate_points(
                plane_pts,
                plane_rgb,
                planes=[pmp_mod.PlaneObservationSpec("table", nrm, 0.0, 200, 0.0005)],
                coordinate_frame="world",
                timestamp_ns=1_000_000,
                observation_id="m0",
                frame_from_camera=T_cam,
            )
            mc = sum(multi.carve_free_space(far_depth, K, frame_from_camera=T_cam) for _ in range(3))
            return (c1, c2, c3, mc)

    assert _carve_test(False) == _carve_test(True)
    assert _carve_test(False)[2] > 0 and _carve_test(False)[3] > 0

    # 5. BoundedVoxelSlotGrid bit-exact parity across multi-cloud, skip masks, updates, and retirement
    p_bg1 = rng.uniform([0.20, -0.50, 1.00], [1.05, 0.50, 1.45], (4000, 3)).astype(np.float32)
    c_bg1 = rng.integers(0, 256, (4000, 3), dtype=np.uint8)
    skip1 = rng.random(4000) < 0.25
    p_fg1 = rng.uniform([0.30, -0.20, 1.04], [0.60, 0.20, 1.25], (600, 3)).astype(np.float32)
    c_fg1 = rng.integers(0, 256, (600, 3), dtype=np.uint8)

    p_bg2 = p_bg1.copy()
    p_bg2[:500] += np.array([0.012, 0.0, 0.0], dtype=np.float32)
    c_bg2 = c_bg1.copy()
    c_bg2[500:700] = (c_bg2[500:700].astype(np.int32) ^ 0x70).astype(np.uint8)

    def _run_grid(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(vsg_mod, "NATIVE", None)
            grid = vsg_mod.BoundedVoxelSlotGrid(voxel_size_m=0.012, max_slots=8192)
            d1 = grid.step([(p_bg1, c_bg1, skip1), (p_fg1, c_fg1, None)])
            d2 = grid.step([(p_bg2[:3000], c_bg2[:3000], skip1[:3000]), (p_fg1[:400], c_fg1[:400], None)])
            kf = grid.build_full_keyframe_records()
            return d1, d2, kf, grid.free_top

    (d1_n, d2_n, kf_n, ft_n) = _run_grid(False)
    (d1_f, d2_f, kf_f, ft_f) = _run_grid(True)
    assert d1_n.active_slot_count == d1_f.active_slot_count > 0
    assert d2_n.active_slot_count == d2_f.active_slot_count > 0
    assert ft_n == ft_f
    assert len(d2_n.deleted_slots_u16) > 0
    np.testing.assert_array_equal(d1_n.deleted_slots_u16, d1_f.deleted_slots_u16)
    np.testing.assert_array_equal(d2_n.deleted_slots_u16, d2_f.deleted_slots_u16)
    # Compare upserted & keyframe records sorted by slot ID (native emits stream order, fallback groups old/new)
    np.testing.assert_array_equal(
        d1_n.upserted_records_u16[np.argsort(d1_n.upserted_records_u16[:, 0])],
        d1_f.upserted_records_u16[np.argsort(d1_f.upserted_records_u16[:, 0])],
    )
    np.testing.assert_array_equal(
        d2_n.upserted_records_u16[np.argsort(d2_n.upserted_records_u16[:, 0])],
        d2_f.upserted_records_u16[np.argsort(d2_f.upserted_records_u16[:, 0])],
    )
    np.testing.assert_array_equal(kf_n, kf_f)


def test_v280_surfel_rgb_grid_cluster_rim_and_masked_planar_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify v2.8.0 native-vs-fallback bit-exact parity for:
    1. RevisableSurfelMap native multi-view RGB fusion & snapshot(include_colors=True)
       across empty, Fast Path 1, Fast Path 2, General Update, replacement, retirement,
       and uniform/select snapshots (with/without voxel keys).
    2. cluster_height_grid_2d (single-pass foreground filter, air-gap decoupling,
       8-connected BFS, and linear quickselect quantiles).
    3. extract_rim_annulus_points (azimuthal outer-perimeter rim extraction).
    4. PersistentPlanarSurfaceConfig(retain_camera_points=False) + zero-copy masked
       planar_project_and_aggregate_fast.
    5. Runtime SIMD AVX-512/AVX2 voxel_centroids & select_existing_points_shadow_fast.
    """
    from evidencescene.surface import (
        grid_clustering as gc_mod,
        object_surface_fusion as osf_mod,
        persistent_static_surface as pss_mod,
        shadow_surface_snapshot_selection as sss_mod,
    )
    from evidencescene.teleop import task_features as tf_mod

    rng = np.random.default_rng(20261128)

    # 1. RevisableSurfelMap native multi-view RGB fusion parity
    p0 = rng.uniform(-0.08, 0.08, (640, 3))
    c0 = rng.integers(0, 256, (640, 3), dtype=np.uint8)
    p1 = p0 + rng.normal(0.0, 0.001, p0.shape)
    c1 = rng.integers(0, 256, (640, 3), dtype=np.uint8)
    p2 = rng.uniform(-0.04, 0.10, (420, 3))
    c2 = rng.integers(0, 256, (420, 3), dtype=np.uint8)

    # Also exercise Fast Path A (strictly sorted) and General Path C (huge span) in observation_voxel_centroids
    p_sorted = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    c_sorted = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    kn_s, cn_s, rn_s = osf_mod.observation_voxel_centroids(p_sorted, 0.005, colors_rgb=c_sorted)
    with monkeypatch.context() as ctx:
        ctx.setattr(osf_mod, "NATIVE", None)
        kf_s, cf_s, rf_s = osf_mod.observation_voxel_centroids(p_sorted, 0.005, colors_rgb=c_sorted)
    np.testing.assert_array_equal(kn_s, kf_s)
    np.testing.assert_array_equal(cn_s, cf_s)
    np.testing.assert_array_equal(rn_s, rf_s)

    p_wide = np.vstack([p0[:40], p0[:40] + np.array([250.0, -180.0, 90.0])])
    c_wide = rng.integers(0, 256, (len(p_wide), 3), dtype=np.uint8)
    kn_w, cn_w, rn_w = osf_mod.observation_voxel_centroids(p_wide, 0.005, colors_rgb=c_wide)
    with monkeypatch.context() as ctx:
        ctx.setattr(osf_mod, "NATIVE", None)
        kf_w, cf_w, rf_w = osf_mod.observation_voxel_centroids(p_wide, 0.005, colors_rgb=c_wide)
    np.testing.assert_array_equal(kn_w, kf_w)
    np.testing.assert_allclose(cn_w, cf_w, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rn_w, rf_w, rtol=1e-5, atol=1e-5)

    def _run_rgb_surfel(disable_native: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(osf_mod, "NATIVE", None)
            m = osf_mod.RevisableSurfelMap(
                voxel_size_m=0.006, maximum_voxels=32000, maximum_live_bytes=16 * 1024 * 1024
            )
            # Empty snapshot with include_colors=True
            e_res = m.snapshot(minimum_observation_support=1, maximum_point_count=100, include_colors=True)
            assert len(e_res) == 4 and e_res[3].shape == (0, 3) and e_res[3].dtype == np.uint8
            # Fast Path 1: single insert into empty map
            m.update({"f0_camA": ("camA", p0, c0)})
            s1 = m.snapshot(minimum_observation_support=1, maximum_point_count=500, include_colors=True)
            # Fast Path 2: identical voxel keys addition
            m.update({"f0_camB": ("camB", p0, c1)})
            s2 = m.snapshot(
                minimum_observation_support=1,
                maximum_point_count=200,
                include_voxel_keys=True,
                include_colors=True,
            )
            # General update: multi-obs + replacement + retirement
            m.update({"f0_camA": ("camA", p1, c1), "f1_camC": ("camC", p2, c2)}, retire=("f0_camB",))
            s3 = m.snapshot(minimum_observation_support=1, maximum_point_count=180, include_colors=True)
            s4 = m.snapshot(
                minimum_observation_support=2,
                maximum_point_count=90,
                include_voxel_keys=True,
                include_colors=True,
            )
            return s1, s2, s3, s4

    sn_list = _run_rgb_surfel(False)
    sf_list = _run_rgb_surfel(True)
    for sn, sf in zip(sn_list, sf_list, strict=True):
        assert len(sn) == len(sf)
        np.testing.assert_allclose(sn[0], sf[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(sn[1], sf[1])
        np.testing.assert_array_equal(sn[2], sf[2])
        if len(sn) == 5:
            np.testing.assert_array_equal(sn[3], sf[3])
            np.testing.assert_array_equal(sn[4], sf[4])
        else:
            np.testing.assert_array_equal(sn[3], sf[3])

    # 2. cluster_height_grid_2d parity (cup under overhanging spout + second cup + tall machine + excluded mask)
    cup1 = rng.uniform([0.34, -0.14, 1.038], [0.42, -0.06, 1.145], (260, 3)).astype(np.float32)
    spout_over_cup1 = rng.uniform([0.35, -0.13, 1.280], [0.41, -0.07, 1.360], (120, 3)).astype(np.float32)
    cup2 = rng.uniform([0.46, 0.12, 1.038], [0.54, 0.20, 1.135], (220, 3)).astype(np.float32)
    tall_mach = rng.uniform([0.70, -0.20, 1.040], [0.85, 0.15, 1.410], (480, 3)).astype(np.float32)
    bg_noise = rng.uniform([0.10, -0.70, 0.99], [1.05, 0.70, 1.03], (300, 3)).astype(np.float32)
    pts_scene = np.vstack([cup1, spout_over_cup1, cup2, tall_mach, bg_noise])
    h_scene = (pts_scene[:, 2] - np.float32(1.0225)).astype(np.float32)
    exc_scene = np.zeros(len(pts_scene), dtype=bool)
    exc_scene[:15] = True

    cl_nat, cl_fb = _run_with_fallback(
        monkeypatch,
        gc_mod,
        gc_mod.cluster_height_grid_2d,
        pts_scene,
        h_scene,
        exclude_mask=exc_scene,
    )
    assert len(cl_nat) == len(cl_fb) >= 3
    for a, b in zip(cl_nat, cl_fb, strict=True):
        assert a.layer == b.layer
        np.testing.assert_array_equal(a.point_indices, b.point_indices)
        assert math.isclose(a.h_p5, b.h_p5, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.h_p97, b.h_p97, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.xy_min_p3[0], b.xy_min_p3[0], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.xy_min_p3[1], b.xy_min_p3[1], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.xy_max_p97[0], b.xy_max_p97[0], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.xy_max_p97[1], b.xy_max_p97[1], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(a.z_top_p985, b.z_top_p985, rel_tol=1e-12, abs_tol=1e-12)

    # 3. extract_rim_annulus_points parity (float64 & float32, normal & sparse fallback)
    thetas = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    zs = np.linspace(1.05, 1.14, 12)
    tt, zz = np.meshgrid(thetas, zs)
    rim_cloud = np.column_stack(
        [
            0.40 + 0.042 * np.cos(tt.ravel()) + rng.normal(0.0, 0.001, tt.size),
            -0.08 + 0.042 * np.sin(tt.ravel()) + rng.normal(0.0, 0.001, tt.size),
            zz.ravel(),
        ]
    )
    for arr in (rim_cloud, rim_cloud.astype(np.float32), rim_cloud[:8]):
        r_nat, r_fb = _run_with_fallback(monkeypatch, tf_mod, tf_mod.extract_rim_annulus_points, arr)
        np.testing.assert_allclose(r_nat, r_fb, rtol=1e-12, atol=1e-12)

    # 4. retain_camera_points=False + zero-copy masked integrate_points parity
    gx, gy = np.meshgrid(np.linspace(-0.2, 0.2, 32), np.linspace(-0.18, 0.18, 28))
    pts_pl = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).astype(np.float32)
    rgb_pl = rng.integers(0, 256, (len(pts_pl), 3), dtype=np.uint8)
    mask_a = (pts_pl[:, 0] <= 0.05)
    mask_b = (pts_pl[:, 0] >= -0.05)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, 3] = [0.0, -0.35, 0.75]

    def _run_masked_planar(disable_native: bool, retain: bool):
        with monkeypatch.context() as ctx:
            if disable_native:
                ctx.setattr(pss_mod, "NATIVE", None)
            rec = pss_mod.PersistentPlanarSurfaceReconstructor(
                config=pss_mod.PersistentPlanarSurfaceConfig(
                    mesh=pss_mod.SurfaceMeshConfig(cell_size_m=0.015, maximum_edge_m=0.045),
                    min_observations_to_commit=1,
                    retain_camera_points=retain,
                )
            )
            nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            for idx, msk in enumerate((mask_a, mask_b, mask_a)):
                rec.integrate_points(
                    pts_pl,
                    rgb_pl,
                    plane_normal=nrm,
                    plane_offset_m=0.0,
                    plane_support_count=400,
                    plane_rmse_m=0.0005,
                    coordinate_frame="world",
                    timestamp_ns=(idx + 1) * 33_000_000,
                    observation_id=f"m/{idx}",
                    frame_from_camera=T_wc,
                    include_mask=msk,
                )
            if not retain:
                T_shift = T_wc.copy()
                T_shift[0, 3] += 0.02
                pr = rec.correct_pose("m/0", T_shift)
                assert not pr.accepted and pr.status == "pose_correction_not_available"
            return rec.snapshot()

    sn_ret = _run_masked_planar(False, True)
    sn_noret = _run_masked_planar(False, False)
    sf_noret = _run_masked_planar(True, False)
    assert sn_ret.ready and sn_noret.ready and sf_noret.ready
    np.testing.assert_allclose(sn_noret.vertices_m, sn_ret.vertices_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sn_noret.vertices_m, sf_noret.vertices_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sn_noret.colors_rgb_u8, sn_ret.colors_rgb_u8)
    np.testing.assert_array_equal(sn_noret.colors_rgb_u8, sf_noret.colors_rgb_u8)

    # 5. Runtime SIMD select_existing_points_shadow_fast parity
    sh_raw = rng.uniform(-0.15, 0.15, (2400, 3))
    sh_keys, sh_pts = osf_mod.observation_voxel_centroids(sh_raw, 0.004)
    sh_sup = rng.integers(2, 20, size=len(sh_pts), dtype=np.int64)
    sh_cam = np.minimum(sh_sup, rng.integers(1, 4, size=len(sh_pts), dtype=np.int64))
    sh_nat, sh_fb = _run_with_fallback(
        monkeypatch,
        sss_mod,
        sss_mod.select_existing_points_shadow,
        sh_pts,
        sh_sup,
        sh_cam,
        sh_keys,
        budget=300,
        minimum_coarse_support=4,
        voxel_size_m=0.004,
    )
    assert sh_nat["mode"] == sh_fb["mode"]
    assert sh_nat["eligible_count"] == sh_fb["eligible_count"]
    np.testing.assert_array_equal(sh_nat["indices"], sh_fb["indices"])
