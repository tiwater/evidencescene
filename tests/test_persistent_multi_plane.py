"""Tests for PersistentMultiPlaneSurfaceReconstructor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pytest

from evidencescene.surface import (
    BINARY_MESH_MAGIC,
    MultiPlaneSurfaceConfig,
    PersistentMultiPlaneSurfaceReconstructor,
    PersistentPlanarSurfaceConfig,
    PlaneObservationSpec,
    SurfaceMeshConfig,
    decode_surface_mesh_binary,
    encode_surface_mesh_binary,
    extract_measured_planes,
    reconstruct_planar_surface,
)


def _camera_pose(eye: tuple[float, float, float], target: tuple[float, float, float]) -> np.ndarray:
    e = np.array(eye, dtype=np.float64)
    t = np.array(target, dtype=np.float64)
    z_axis = t - e
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = e
    return T


def _render_l_shaped_scene(
    T_wc: np.ndarray,
    K: np.ndarray,
    *,
    width: int = 160,
    height: int = 120,
    table_z: float = 0.0,
    wall_y: float = 0.25,
    table_bounds: tuple[float, float, float, float] = (-0.25, 0.25, -0.25, 0.25),
    wall_z_max: float = 0.30,
    occluder_box: tuple[float, float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast a horizontal table (z=table_z) and vertical backsplash (y=wall_y)."""
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    dirs_c = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
    R = T_wc[:3, :3]
    origin = T_wc[:3, 3]
    dirs_w = dirs_c @ R.T

    depth = np.full((height, width), np.nan, dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Intersect horizontal table z = table_z (for z_dir != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_tbl = (table_z - origin[2]) / dirs_w[..., 2]
    hit_tbl = np.isfinite(t_tbl) & (t_tbl > 0.15) & (t_tbl < 2.5)
    p_tbl = origin[None, None, :] + t_tbl[..., None] * dirs_w
    hit_tbl &= (
        (p_tbl[..., 0] >= table_bounds[0])
        & (p_tbl[..., 0] <= table_bounds[1])
        & (p_tbl[..., 1] >= table_bounds[2])
        & (p_tbl[..., 1] <= table_bounds[3])
    )

    # Intersect vertical backsplash y = wall_y (normal [0, -1, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = (wall_y - origin[1]) / dirs_w[..., 1]
    hit_wall = np.isfinite(t_wall) & (t_wall > 0.15) & (t_wall < 2.5)
    p_wall = origin[None, None, :] + t_wall[..., None] * dirs_w
    hit_wall &= (
        (p_wall[..., 0] >= table_bounds[0])
        & (p_wall[..., 0] <= table_bounds[1])
        & (p_wall[..., 2] >= table_z)
        & (p_wall[..., 2] <= wall_z_max)
    )

    # Choose closer valid hit
    tbl_wins = hit_tbl & (~hit_wall | (t_tbl <= t_wall))
    wall_wins = hit_wall & (~hit_tbl | (t_wall < t_tbl))
    depth[tbl_wins] = t_tbl[tbl_wins]
    rgb[tbl_wins] = np.array([210, 190, 160], dtype=np.uint8)
    depth[wall_wins] = t_wall[wall_wins]
    rgb[wall_wins] = np.array([90, 130, 195], dtype=np.uint8)

    # Optional foreground occluder floating above the table
    if occluder_box is not None:
        ox0, ox1, oy0, oy1, oz = occluder_box
        with np.errstate(divide="ignore", invalid="ignore"):
            t_occ = (oz - origin[2]) / dirs_w[..., 2]
        p_occ = origin[None, None, :] + t_occ[..., None] * dirs_w
        hit_occ = (
            np.isfinite(t_occ)
            & (t_occ > 0.15)
            & (p_occ[..., 0] >= ox0)
            & (p_occ[..., 0] <= ox1)
            & (p_occ[..., 1] >= oy0)
            & (p_occ[..., 1] <= oy1)
            & (~np.isfinite(depth) | (t_occ < depth))
        )
        depth[hit_occ] = t_occ[hit_occ]
        rgb[hit_occ] = np.array([20, 200, 40], dtype=np.uint8)

    return depth, rgb


def test_l_shaped_counter_and_backsplash_multiview_fusion_and_seam_arbitration() -> None:
    """Fuse horizontal tabletop + vertical backsplash simultaneously without seam double-counting."""
    K = np.array([[180.0, 0.0, 80.0], [0.0, 180.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    specs = [
        PlaneObservationSpec(
            plane_id="backsplash",
            plane_normal=np.array([0.0, -1.0, 0.0], dtype=np.float64),
            plane_offset_m=0.25,
            plane_support_count=400,
            plane_rmse_m=0.001,
        ),
        PlaneObservationSpec(
            plane_id="table",
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            plane_offset_m=0.0,
            plane_support_count=500,
            plane_rmse_m=0.001,
        ),
    ]

    recon = PersistentMultiPlaneSurfaceReconstructor(
        config=MultiPlaneSurfaceConfig(
            plane_config=PersistentPlanarSurfaceConfig(
                mesh=SurfaceMeshConfig(cell_size_m=0.015, maximum_edge_m=0.045),
                min_observations_to_commit=2,
                pixel_step=1,
            ),
            max_planes=4,
        )
    )

    views = [
        _camera_pose((0.0, -0.55, 0.48), (0.0, 0.08, 0.08)),
        _camera_pose((-0.18, -0.52, 0.45), (0.0, 0.08, 0.08)),
        _camera_pose((0.18, -0.52, 0.45), (0.0, 0.08, 0.08)),
    ]

    for idx, pose in enumerate(views):
        depth, rgb = _render_l_shaped_scene(pose, K)
        res = recon.integrate_rgbd(
            depth,
            rgb,
            K,
            frame_from_camera=pose,
            planes=specs,
            coordinate_frame="workspace",
            timestamp_ns=(idx + 1) * 33_000_000,
            observation_id=f"cam/{idx}",
            camera_id="head",
        )
        assert res.accepted is True
        assert res.status == "integrated"
        assert res.active_plane_ids == ("backsplash", "table")

    snap = recon.snapshot(reference_timestamp_ns=100_000_000)
    assert snap.ready is True
    assert snap.reason == "ok"
    assert snap.plane_ids == ("backsplash", "table")
    assert snap.coordinate_frame == "workspace"
    assert snap.plane_snapshots["table"].ready is True
    assert snap.plane_snapshots["backsplash"].ready is True

    # Unified snapshot arrays must be read-only and match exact plane concatenation.
    with pytest.raises(ValueError):
        snap.vertices_m[0, 0] = 999.0
    n_wall = len(snap.plane_snapshots["backsplash"].vertices_m)
    n_tbl = len(snap.plane_snapshots["table"].vertices_m)
    assert len(snap.vertices_m) == n_wall + n_tbl
    assert int(np.count_nonzero(snap.vertex_plane_indices == 0)) == n_wall
    assert int(np.count_nonzero(snap.vertex_plane_indices == 1)) == n_tbl
    assert snap.triangles.min() >= 0 and snap.triangles.max() < len(snap.vertices_m)

    # Check table vertices stay on z ~= 0 and backsplash vertices stay on y ~= 0.25.
    wall_v = snap.vertices_m[snap.vertex_plane_indices == 0]
    tbl_v = snap.vertices_m[snap.vertex_plane_indices == 1]
    assert float(np.max(np.abs(wall_v[:, 1] - 0.25))) < 1e-3
    assert float(np.max(np.abs(tbl_v[:, 2]))) < 1e-3
    assert snap.surface_area_m2 == pytest.approx(
        snap.plane_snapshots["backsplash"].surface_area_m2
        + snap.plane_snapshots["table"].surface_area_m2,
        rel=1e-9,
    )

    # Occluding part of the table on a later frame must retain previously committed table geometry.
    area_tbl_before = snap.plane_snapshots["table"].surface_area_m2
    depth_occ, rgb_occ = _render_l_shaped_scene(
        views[0], K, occluder_box=(-0.10, 0.10, -0.10, 0.10, 0.12)
    )
    res_occ = recon.integrate_rgbd(
        depth_occ,
        rgb_occ,
        K,
        frame_from_camera=views[0],
        planes=specs,
        coordinate_frame="workspace",
        timestamp_ns=140_000_000,
        observation_id="cam/3_occluded",
        camera_id="head",
    )
    assert res_occ.accepted is True
    snap_after = recon.snapshot()
    assert snap_after.plane_snapshots["table"].surface_area_m2 >= area_tbl_before * 0.95


def test_multi_plane_seam_tie_break_determinism_and_integrate_points() -> None:
    """Exact seam points equidistant from two planes deterministically go to lexicographically first plane_id."""
    specs = [
        PlaneObservationSpec("plane_b", np.array([0.0, 0.0, 1.0]), 0.0, 100, 0.001),
        PlaneObservationSpec("plane_a", np.array([0.0, 1.0, 0.0]), 0.0, 100, 0.001),
    ]
    recon = PersistentMultiPlaneSurfaceReconstructor(
        config=MultiPlaneSurfaceConfig(
            plane_config=PersistentPlanarSurfaceConfig(
                mesh=SurfaceMeshConfig(cell_size_m=0.01, maximum_edge_m=0.03),
                min_observations_to_commit=1,
            ),
            seam_tie_tolerance_m=1e-6,
            enable_cross_plane_exclusion=True,
        )
    )
    # Build points strictly on the seam line y=0, z=0 (residual 0 to both planes),
    # plus dedicated interior points on each plane.
    xs = np.linspace(-0.05, 0.05, 11)
    seam_pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    gx, gy = np.meshgrid(xs, np.linspace(0.02, 0.08, 7))
    only_b = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    gx2, gz2 = np.meshgrid(xs, np.linspace(0.02, 0.08, 7))
    only_a = np.column_stack([gx2.ravel(), np.zeros(gx2.size), gz2.ravel()])

    pts = np.vstack([seam_pts, only_a, only_b])
    colors = np.full((len(pts), 3), 180, dtype=np.uint8)

    res = recon.integrate_points(
        pts,
        colors,
        planes=specs,  # intentionally unsorted input order
        coordinate_frame="world",
        timestamp_ns=1_000_000,
        observation_id="pts/0",
    )
    assert res.accepted is True
    # Total inliers across both planes must equal len(pts) with ZERO double counting!
    assert res.total_inlier_sample_count == len(pts)
    assert res.plane_results["plane_a"].inlier_sample_count == len(seam_pts) + len(only_a)
    assert res.plane_results["plane_b"].inlier_sample_count == len(only_b)


def test_multi_plane_pose_revision_resets_max_planes_and_concurrency() -> None:
    """Verify pose revision broadcast, per-plane & global reset, max_planes cap, and thread safety."""
    K = np.array([[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pose0 = _camera_pose((0.0, -0.55, 0.45), (0.0, 0.05, 0.05))
    pose_wrong = pose0.copy()
    pose_wrong[:3, 3] += np.array([0.02, 0.0, 0.0])

    specs = [
        PlaneObservationSpec("backsplash", np.array([0.0, -1.0, 0.0]), 0.25, 300, 0.001),
        PlaneObservationSpec("table", np.array([0.0, 0.0, 1.0]), 0.0, 300, 0.001),
    ]
    recon = PersistentMultiPlaneSurfaceReconstructor(
        config=MultiPlaneSurfaceConfig(
            plane_config=PersistentPlanarSurfaceConfig(
                mesh=SurfaceMeshConfig(cell_size_m=0.02, maximum_edge_m=0.06),
                min_observations_to_commit=1,
                pixel_step=2,
            ),
            max_planes=2,
        )
    )

    depth, rgb = _render_l_shaped_scene(pose0, K)
    r0 = recon.integrate_rgbd(
        depth,
        rgb,
        K,
        frame_from_camera=pose_wrong,
        planes=specs,
        coordinate_frame="world",
        timestamp_ns=10_000_000,
        observation_id="obs/0",
    )
    assert r0.accepted is True

    # Adding a 3rd plane when max_planes=2 must fail closed without corrupting existing 2 planes.
    spec_extra = PlaneObservationSpec("shelf", np.array([0.0, 0.0, 1.0]), -0.4, 200, 0.001)
    r_cap = recon.integrate_rgbd(
        depth,
        rgb,
        K,
        frame_from_camera=pose0,
        planes=[spec_extra],
        coordinate_frame="world",
        timestamp_ns=20_000_000,
        observation_id="obs/cap",
    )
    assert r_cap.accepted is False
    assert r_cap.status == "max_planes_exceeded"
    assert recon.plane_ids == ("backsplash", "table")

    # Broadcast pose correction across both planes.
    r_fix = recon.correct_pose("obs/0", pose0, timestamp_ns=25_000_000)
    assert r_fix.accepted is True
    assert r_fix.status == "pose_corrected"
    assert r_fix.plane_results["table"].accepted is True
    assert r_fix.plane_results["backsplash"].accepted is True

    # Coordinate frame mismatch fails closed under default policy="reject".
    r_frame = recon.integrate_rgbd(
        depth,
        rgb,
        K,
        frame_from_camera=pose0,
        planes=specs,
        coordinate_frame="other_frame",
        timestamp_ns=30_000_000,
        observation_id="obs/bad_frame",
    )
    assert r_frame.accepted is False
    assert r_frame.status == "coordinate_frame_changed"

    # Per-plane reset clears only backsplash while keeping table ready.
    epoch_before = recon.epoch
    r_sub_reset = recon.reset(plane_id="backsplash")
    assert r_sub_reset.accepted is True
    assert r_sub_reset.status == "plane_reset"
    assert recon.epoch == epoch_before
    snap_half = recon.snapshot()
    assert snap_half.plane_snapshots["table"].ready is True
    assert snap_half.plane_snapshots["backsplash"].ready is False

    # Unknown plane reset fails closed.
    r_unknown = recon.reset(plane_id="nonexistent")
    assert r_unknown.accepted is False
    assert r_unknown.status == "unknown_plane"

    # Global reset clears all planes and increments epoch.
    r_glob = recon.reset()
    assert r_glob.accepted is True
    assert r_glob.epoch == epoch_before + 1
    assert recon.plane_ids == ()
    assert recon.snapshot().ready is False

    # Thread-safe concurrent integration & snapshotting.
    def _writer(i: int) -> None:
        recon.integrate_rgbd(
            depth,
            rgb,
            K,
            frame_from_camera=pose0,
            planes=specs,
            coordinate_frame="world",
            timestamp_ns=(i + 1) * 10_000_000,
            observation_id=f"conc/{i}",
        )

    def _reader(_: int) -> bool:
        s = recon.snapshot()
        assert len(s.vertices_m) == len(s.vertex_plane_indices)
        return s.ready

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_writer, i) for i in range(6)]
        rfuts = [pool.submit(_reader, i) for i in range(12)]
        for f in futs:
            f.result()
        for f in rfuts:
            f.result()
    assert recon.snapshot().ready is True


def test_extract_measured_planes_and_binary_mesh_codec_roundtrip() -> None:
    """Verify deterministic multi-plane extraction and ESTM binary codec across mesh types."""
    rng = np.random.default_rng(2026)
    xs = np.linspace(-0.25, 0.25, 36)
    ys = np.linspace(-0.20, 0.20, 30)
    gx, gy = np.meshgrid(xs, ys)
    tbl = np.column_stack(
        [gx.ravel(), gy.ravel(), rng.normal(0.0, 0.0005, gx.size)]
    )
    gx2, gz2 = np.meshgrid(xs, np.linspace(0.02, 0.28, 24))
    wall = np.column_stack(
        [gx2.ravel(), np.full(gx2.size, 0.22) + rng.normal(0.0, 0.0005, gx2.size), gz2.ravel()]
    )
    outliers = rng.uniform([-0.2, -0.2, 0.05], [0.2, 0.15, 0.25], size=(80, 3))
    cloud = np.vstack([tbl, wall, outliers])
    colors = np.vstack(
        [
            np.full((len(tbl), 3), (210, 195, 170), dtype=np.uint8),
            np.full((len(wall), 3), (90, 130, 190), dtype=np.uint8),
            np.full((len(outliers), 3), (40, 200, 60), dtype=np.uint8),
        ]
    )

    specs = extract_measured_planes(
        cloud,
        max_planes=3,
        ransac_iterations=180,
        plane_distance_threshold_m=0.005,
        minimum_plane_points=200,
        min_inter_plane_angle_deg=20.0,
    )
    assert len(specs) == 2
    # Extracted normals must be approximately orthogonal (table + vertical backsplash).
    dot_01 = abs(float(specs[0].plane_normal @ specs[1].plane_normal))
    assert dot_01 < 0.05
    assert all(s.plane_rmse_m < 0.002 for s in specs)

    multi = PersistentMultiPlaneSurfaceReconstructor(
        config=MultiPlaneSurfaceConfig(
            plane_config=PersistentPlanarSurfaceConfig(
                mesh=SurfaceMeshConfig(cell_size_m=0.015, maximum_edge_m=0.045),
                min_observations_to_commit=1,
            )
        )
    )
    res = multi.integrate_points(
        cloud,
        colors,
        planes=specs,
        coordinate_frame="workspace",
        timestamp_ns=123_000_000,
        observation_id="auto/0",
    )
    assert res.accepted is True
    msnap = multi.snapshot()
    assert msnap.ready is True and len(msnap.plane_ids) == 2

    # Round-trip MultiPlaneSurfaceSnapshot through ESTM binary codec (preserving plane indices).
    blob_m = encode_surface_mesh_binary(msnap)
    assert blob_m[:4] == BINARY_MESH_MAGIC
    dec_m = decode_surface_mesh_binary(blob_m)
    assert dec_m["ready"] is True
    assert dec_m["plane_ids"] == list(msnap.plane_ids)
    assert np.max(np.abs(dec_m["vertices_m"] - msnap.vertices_m)) < 1e-4
    np.testing.assert_array_equal(dec_m["colors_rgb_u8"], msnap.colors_rgb_u8)
    np.testing.assert_array_equal(dec_m["triangles"], msnap.triangles)
    np.testing.assert_array_equal(dec_m["vertex_plane_indices"], msnap.vertex_plane_indices)
    np.testing.assert_array_equal(dec_m["triangle_plane_indices"], msnap.triangle_plane_indices)

    # Round-trip single PersistentSurfaceSnapshot and MeasuredSurfaceMesh.
    psnap = msnap.plane_snapshots[msnap.plane_ids[0]]
    dec_p = decode_surface_mesh_binary(encode_surface_mesh_binary(psnap))
    assert dec_p["ready"] is True
    assert np.max(np.abs(dec_p["vertices_m"] - psnap.vertices_m)) < 1e-4
    np.testing.assert_array_equal(dec_p["triangles"], psnap.triangles)

    single_mesh = reconstruct_planar_surface(
        tbl,
        colors[: len(tbl)],
        plane_normal=specs[0].plane_normal,
        plane_offset_m=specs[0].plane_offset_m,
        plane_support_count=specs[0].plane_support_count,
        plane_rmse_m=specs[0].plane_rmse_m,
        coordinate_frame="workspace",
        timestamp_ns=123_000_000,
        source_id="single/0",
        config=SurfaceMeshConfig(cell_size_m=0.015, maximum_edge_m=0.045),
    )
    dec_s = decode_surface_mesh_binary(encode_surface_mesh_binary(single_mesh))
    assert dec_s["ready"] is True and dec_s["source_id"] == "single/0"
    assert np.max(np.abs(dec_s["vertices_m"] - single_mesh.vertices_m)) < 1e-4

    # Unready / empty snapshot round-trips cleanly.
    empty_snap = PersistentMultiPlaneSurfaceReconstructor().snapshot()
    dec_e = decode_surface_mesh_binary(encode_surface_mesh_binary(empty_snap))
    assert dec_e["ready"] is False and len(dec_e["vertices_m"]) == 0
