"""Standalone tests for persistent measured planar static surface reconstruction."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evidencescene.surface import (
    PersistentPlanarSurfaceConfig,
    PersistentPlanarSurfaceReconstructor,
    SurfaceMeshConfig,
    reconstruct_planar_surface,
)


def _look_at(camera_pos, target_pos, world_up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Return 4x4 frame_from_camera (OpenCV optical frame: +z forward, +x right, +y down)."""
    cam = np.asarray(camera_pos, dtype=np.float64)
    tgt = np.asarray(target_pos, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    z_axis = tgt - cam
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, up)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(z_axis, np.array([1.0, 0.0, 0.0]))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = cam
    return T


def _render_planar_rgbd(
    *,
    frame_from_camera: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int] = (96, 128),
    plane_normal=(0.0, 0.0, 1.0),
    plane_offset_m: float = 0.0,
    x_bounds=(-0.45, 0.45),
    y_bounds=(-0.30, 0.30),
    gap_x_bounds: tuple[float, float] | None = None,
    occluder_box: tuple[float, float, float, float, float] | None = None,
    background_z: float | None = None,
    noise_std_m: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a calibrated registered RGB-D frame of a bounded textured plane.

    Returns ``(depth_m, rgb_u8, foreground_mask)`` where ``foreground_mask`` marks
    pixels belonging to ``occluder_box = (xmin, xmax, ymin, ymax, z_height)``.
    """
    h, w = shape
    n = np.asarray(plane_normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays_c = np.stack(((uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)), axis=-1)
    rot = frame_from_camera[:3, :3]
    cam_pos = frame_from_camera[:3, 3]
    rays_w = rays_c @ rot.T

    depth = np.full((h, w), np.nan, dtype=np.float64)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fg_mask = np.zeros((h, w), dtype=bool)

    denom = rays_w @ n
    valid_ray = np.abs(denom) > 1e-6
    t_plane = np.full((h, w), np.nan, dtype=np.float64)
    t_plane[valid_ray] = -(float(cam_pos @ n) + plane_offset_m) / denom[valid_ray]
    hit_w = cam_pos[None, None, :] + t_plane[..., None] * rays_w

    # Express local coordinates on plane for bounds / texture.
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    u_vec = axis - n * float(np.dot(axis, n))
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(n, u_vec)
    p0 = -plane_offset_m * n
    rel = hit_w - p0[None, None, :]
    lu = rel @ u_vec
    lv = rel @ v_vec

    on_table = (
        valid_ray
        & (t_plane > 0.15)
        & (t_plane < 3.0)
        & (lu >= x_bounds[0])
        & (lu <= x_bounds[1])
        & (lv >= y_bounds[0])
        & (lv <= y_bounds[1])
    )
    if gap_x_bounds is not None:
        on_table &= ~((lu >= gap_x_bounds[0]) & (lu <= gap_x_bounds[1]))

    depth[on_table] = t_plane[on_table]
    # Deterministic spatial RGB texture.
    rgb[..., 0] = np.clip(((lu - x_bounds[0]) / max(x_bounds[1] - x_bounds[0], 1e-6)) * 220 + 20, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(((lv - y_bounds[0]) / max(y_bounds[1] - y_bounds[0], 1e-6)) * 220 + 20, 0, 255).astype(np.uint8)
    rgb[..., 2] = 110
    rgb[~on_table] = 0

    if background_z is not None:
        t_bg = np.full((h, w), np.nan, dtype=np.float64)
        t_bg[valid_ray] = -(float(cam_pos @ n) + (plane_offset_m - background_z)) / denom[valid_ray]
        bg = (~on_table) & valid_ray & (t_bg > 0.15) & (t_bg < 3.0)
        depth[bg] = t_bg[bg]
        rgb[bg] = (30, 30, 40)

    if occluder_box is not None:
        xmin, xmax, ymin, ymax, z_h = occluder_box
        t_occ = np.full((h, w), np.nan, dtype=np.float64)
        t_occ[valid_ray] = -(float(cam_pos @ n) + (plane_offset_m - z_h)) / denom[valid_ray]
        hit_occ = cam_pos[None, None, :] + t_occ[..., None] * rays_w
        rel_occ = hit_occ - p0[None, None, :]
        ou = rel_occ @ u_vec
        ov = rel_occ @ v_vec
        occ = (
            valid_ray
            & (t_occ > 0.15)
            & (t_occ < 3.0)
            & (ou >= xmin)
            & (ou <= xmax)
            & (ov >= ymin)
            & (ov <= ymax)
            & (~np.isfinite(depth) | (t_occ < depth))
        )
        depth[occ] = t_occ[occ]
        rgb[occ] = (240, 20, 20)
        fg_mask[occ] = True

    if noise_std_m > 0.0:
        rng = np.random.default_rng(seed)
        finite = np.isfinite(depth)
        depth[finite] += rng.normal(0.0, noise_std_m, size=int(np.count_nonzero(finite)))

    return depth, rgb, fg_mask


def _default_K(shape=(96, 128)) -> np.ndarray:
    h, w = shape
    return np.array([[140.0, 0.0, w / 2.0], [0.0, 140.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _config(**kwargs) -> PersistentPlanarSurfaceConfig:
    mesh_cfg = kwargs.pop(
        "mesh",
        SurfaceMeshConfig(cell_size_m=0.02, maximum_edge_m=0.06, maximum_plane_residual_m=0.008, minimum_support=16),
    )
    defaults = dict(mesh=mesh_cfg, pixel_step=2, min_observations_to_commit=2)
    defaults.update(kwargs)
    return PersistentPlanarSurfaceConfig(**defaults)


def test_multiview_fusion_gains_coverage_retains_through_occlusion_and_keeps_gaps_open():
    K = _default_K()
    recon = PersistentPlanarSurfaceReconstructor(config=_config())

    # View 1 (left side) observed twice to commit; gap in the middle [-0.05, 0.05] is NEVER observed.
    T_left = _look_at((-0.25, -0.45, 0.60), (-0.22, 0.0, 0.0))
    for i in range(2):
        d, c, _ = _render_planar_rgbd(
            frame_from_camera=T_left, K=K, x_bounds=(-0.45, 0.0), gap_x_bounds=(-0.05, 0.05)
        )
        res = recon.integrate_rgbd(
            d, c, K,
            frame_from_camera=T_left,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=500, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=1_000_000_000 + i * 100_000_000,
            observation_id=f"left-{i}", camera_id="cam0",
        )
        assert res.accepted and res.status == "integrated"

    snap_left = recon.snapshot()
    assert snap_left.ready
    area_left = snap_left.surface_area_m2
    assert area_left > 0.15

    # View 2 (right side) observed twice -> coverage must grow significantly!
    T_right = _look_at((0.25, -0.45, 0.60), (0.22, 0.0, 0.0))
    for i in range(2):
        d, c, _ = _render_planar_rgbd(
            frame_from_camera=T_right, K=K, x_bounds=(0.0, 0.45), gap_x_bounds=(-0.05, 0.05)
        )
        res = recon.integrate_rgbd(
            d, c, K,
            frame_from_camera=T_right,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=500, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=2_000_000_000 + i * 100_000_000,
            observation_id=f"right-{i}", camera_id="cam0",
        )
        assert res.accepted

    snap_fused = recon.snapshot()
    assert snap_fused.ready
    assert snap_fused.surface_area_m2 > area_left * 1.6

    # Unobserved gap [-0.05, 0.05] must NOT be bridged by any triangle!
    tri = snap_fused.vertices_m[snap_fused.triangles]
    bridges_gap = (tri[:, :, 0].min(axis=1) < -0.04) & (tri[:, :, 0].max(axis=1) > 0.04)
    assert not np.any(bridges_gap)

    # Full occlusion frame (large raised slab covering the entire table):
    T_center = _look_at((0.0, -0.55, 0.65), (0.0, 0.0, 0.0))
    d_occ, c_occ, fg_occ = _render_planar_rgbd(
        frame_from_camera=T_center, K=K,
        x_bounds=(-0.45, 0.45), gap_x_bounds=(-0.05, 0.05),
        occluder_box=(-0.50, 0.50, -0.35, 0.35, 0.12),
    )
    res_occ = recon.integrate_rgbd(
        d_occ, c_occ, K,
        frame_from_camera=T_center,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=500, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=5_000_000_000,
        observation_id="occ-0", camera_id="cam0", exclude_mask=fg_occ,
    )
    assert res_occ.accepted and res_occ.status == "no_inlier_samples"

    # Fused mesh retains 100% of previously supported surface through occlusion!
    snap_occ = recon.snapshot()
    assert snap_occ.ready
    assert snap_occ.surface_area_m2 == pytest.approx(snap_fused.surface_area_m2)
    assert len(snap_occ.triangles) == len(snap_fused.triangles)
    # No red occluder pixels leaked into the static surface!
    assert not np.any((snap_occ.colors_rgb_u8[:, 0] > 235) & (snap_occ.colors_rgb_u8[:, 1] < 25))
    # Snapshot age vs observation freshness are clearly distinguishable:
    assert snap_occ.latest_observation_timestamp_ns == 5_000_000_000
    assert snap_occ.newest_contributing_timestamp_ns == 2_100_000_000
    assert snap_occ.oldest_contributing_timestamp_ns == 1_100_000_000
    assert snap_occ.newest_observation_lag_ns == 2_900_000_000
    assert snap_occ.oldest_vertex_lag_ns == 3_900_000_000
    assert snap_occ.snapshot_age_ns() < 1_000_000_000
    lags = snap_occ.observation_freshness_ns(6_000_000_000)
    assert lags["latest_observation_lag_ns"] == 1_000_000_000
    assert lags["newest_contributing_lag_ns"] == 3_900_000_000
    assert lags["oldest_contributing_lag_ns"] == 4_900_000_000


@pytest.mark.parametrize("angles", [(45, 0, 0), (0, 90, 0), (30, -50, 20)])
def test_arbitrary_oriented_planes_supported(angles):
    rot = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    normal = rot @ np.array([0.0, 0.0, 1.0])
    offset = -float(normal @ (rot @ np.array([0.1, -0.2, 0.3])))
    K = _default_K()
    # Place camera along +normal looking at plane center.
    p_center = rot @ np.array([0.1, -0.2, 0.3])
    cam_pos = p_center + 0.7 * normal - 0.15 * (rot @ np.array([0.0, 1.0, 0.0]))
    T_fc = _look_at(cam_pos, p_center, world_up=rot @ np.array([0.0, 1.0, 0.0]))

    recon = PersistentPlanarSurfaceReconstructor(config=_config())
    for i in range(2):
        d, c, _ = _render_planar_rgbd(
            frame_from_camera=T_fc, K=K, plane_normal=normal, plane_offset_m=offset,
            x_bounds=(-0.3, 0.3), y_bounds=(-0.25, 0.25),
        )
        res = recon.integrate_rgbd(
            d, c, K,
            frame_from_camera=T_fc,
            plane_normal=normal, plane_offset_m=offset, plane_support_count=400, plane_rmse_m=0.001,
            coordinate_frame="world/tilted", timestamp_ns=1_000_000 * (i + 1),
            observation_id=f"tilted-{i}",
        )
        assert res.accepted

    snap = recon.snapshot()
    assert snap.ready
    assert snap.surface_area_m2 > 0.1
    assert np.allclose(snap.face_normals @ normal, 1.0, atol=1e-4)
    assert np.max(np.abs(snap.vertices_m @ normal + offset)) < 1e-4


def test_transient_foreground_and_free_space_carving_leave_no_ghosts():
    K = _default_K()
    T = _look_at((0.0, -0.45, 0.65), (0.0, 0.0, 0.0))
    recon = PersistentPlanarSurfaceReconstructor(
        config=_config(min_observations_to_commit=2, tentative_ttl_updates=3, free_space_conflict_threshold=2)
    )

    # Establish real static table on x in [-0.30, 0.10] with floor background behind the edge at z = -0.25.
    for i in range(2):
        d, c, _ = _render_planar_rgbd(
            frame_from_camera=T, K=K, x_bounds=(-0.30, 0.10), y_bounds=(-0.20, 0.20), background_z=-0.25
        )
        recon.integrate_rgbd(
            d, c, K, frame_from_camera=T,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=(i + 1) * 100_000_000, observation_id=f"base-{i}",
        )

    # Frame 2: unmasked moving foreground flat patch appears for ONE frame past the table edge at x in [0.18, 0.35], z=0.002.
    d_trans, c_trans, _ = _render_planar_rgbd(
        frame_from_camera=T, K=K, x_bounds=(-0.30, 0.10), y_bounds=(-0.20, 0.20),
        occluder_box=(0.18, 0.35, -0.10, 0.10, 0.002), background_z=-0.25,
    )
    recon.integrate_rgbd(
        d_trans, c_trans, K, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=300_000_000, observation_id="transient-1",
    )
    snap_after_trans = recon.snapshot()
    assert snap_after_trans.ready
    # Single-frame transient must NOT enter the committed mesh!
    assert float(snap_after_trans.vertices_m[:, 0].max()) < 0.13

    # Even if an unmasked flat ghost lingered for 2 frames and committed, once it leaves and the camera sees
    # the deeper floor behind it, free-space carving removes the ghost while preserving the real table!
    recon.integrate_rgbd(
        d_trans, c_trans, K, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=400_000_000, observation_id="transient-2",
    )
    assert float(recon.snapshot().vertices_m[:, 0].max()) > 0.25  # ghost temporarily committed

    for j in range(4):
        d_clean, c_clean, _ = _render_planar_rgbd(
            frame_from_camera=T, K=K, x_bounds=(-0.30, 0.10), y_bounds=(-0.20, 0.20), background_z=-0.25
        )
        recon.integrate_rgbd(
            d_clean, c_clean, K, frame_from_camera=T,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=(5 + j) * 100_000_000, observation_id=f"clean-{j}",
        )

    snap_carved = recon.snapshot()
    assert snap_carved.ready
    # Ghost beyond x=0.10 is carved away; real table on [-0.30, 0.10] remains intact!
    assert float(snap_carved.vertices_m[:, 0].max()) < 0.13
    assert float(snap_carved.vertices_m[:, 0].min()) < -0.22


def test_duplicate_observations_and_pose_corrections():
    K = _default_K()
    T_true = _look_at((0.0, -0.45, 0.65), (0.0, 0.0, 0.0))
    T_wrong = T_true.copy()
    T_wrong[0, 3] += 0.20  # 20 cm lateral pose error

    recon = PersistentPlanarSurfaceReconstructor(
        config=_config(min_observations_to_commit=1, max_retained_observations=3)
    )
    d, c, _ = _render_planar_rgbd(frame_from_camera=T_true, K=K, x_bounds=(-0.20, 0.20), y_bounds=(-0.15, 0.15))

    r0 = recon.integrate_rgbd(
        d, c, K, frame_from_camera=T_wrong,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=1_000, observation_id="obs-0",
    )
    assert r0.accepted and r0.status == "integrated"
    v_after_0 = r0.version
    snap_wrong = recon.snapshot()
    assert snap_wrong.ready
    assert float(snap_wrong.vertices_m[:, 0].mean()) > 0.15  # shifted by +20cm

    # Exact duplicate is idempotent no-op.
    r_dup = recon.integrate_rgbd(
        d, c, K, frame_from_camera=T_wrong,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=1_000, observation_id="obs-0",
    )
    assert not r_dup.accepted and r_dup.status == "duplicate_ignored"
    assert r_dup.version == v_after_0
    assert int(recon.snapshot().vertex_observation_counts.max()) == 1

    # Correct pose -> old shifted cells removed, corrected cells centered around 0!
    r_corr = recon.correct_pose("obs-0", T_true)
    assert r_corr.accepted and r_corr.status == "pose_corrected"
    assert r_corr.version == v_after_0 + 1
    snap_corr = recon.snapshot()
    assert snap_corr.ready
    assert abs(float(snap_corr.vertices_m[:, 0].mean())) < 0.03
    assert float(snap_corr.vertices_m[:, 0].max()) < 0.22
    assert snap_corr.contributing_observations[0].pose_revision == 1

    # Calling correct_pose with the same pose is a no-op.
    r_same = recon.correct_pose("obs-0", T_true)
    assert not r_same.accepted and r_same.status == "pose_unchanged"
    assert r_same.version == r_corr.version

    # Push obs-0 out of the 3-observation retention window.
    for i in range(1, 5):
        recon.integrate_rgbd(
            d, c, K, frame_from_camera=T_true,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=1_000 + i * 100, observation_id=f"obs-{i}",
        )
    # Out-of-window pose correction fails closed cleanly.
    r_old = recon.correct_pose("obs-0", T_wrong)
    assert not r_old.accepted and r_old.status == "pose_correction_out_of_window"
    # Unknown observation fails closed cleanly.
    r_unk = recon.correct_pose("obs-does-not-exist", T_true)
    assert not r_unk.accepted and r_unk.status == "pose_correction_unknown_observation"


def test_frame_calibration_plane_changes_and_explicit_reset():
    K0 = _default_K((96, 128))
    K1 = np.array([[180.0, 0.0, 80.0], [0.0, 180.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T = _look_at((0.0, -0.45, 0.65), (0.0, 0.0, 0.0))
    d0, c0, _ = _render_planar_rgbd(frame_from_camera=T, K=K0, shape=(96, 128))
    d1, c1, _ = _render_planar_rgbd(frame_from_camera=T, K=K1, shape=(120, 160))

    # Default policies = "reject": incompatible changes never corrupt existing surface!
    recon = PersistentPlanarSurfaceReconstructor(config=_config(min_observations_to_commit=1))
    assert recon.integrate_rgbd(
        d0, c0, K0, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=100, observation_id="f0", camera_id="head",
    ).accepted

    # Distinct camera_id with different K/resolution is allowed (multi-camera fusion!).
    assert recon.integrate_rgbd(
        d1, c1, K1, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=200, observation_id="f1", camera_id="chest",
    ).accepted

    snap_before = recon.snapshot()
    verts_before = snap_before.vertices_m.copy()

    # Same camera_id ("head") with changed calibration -> rejected without mutating state.
    r_cal = recon.integrate_rgbd(
        d1, c1, K1, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=300, observation_id="f2", camera_id="head",
    )
    assert not r_cal.accepted and r_cal.status == "camera_calibration_changed"

    # Coordinate frame change -> rejected without mutating state.
    r_frame = recon.integrate_rgbd(
        d0, c0, K0, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="robot_base", timestamp_ns=400, observation_id="f3", camera_id="head",
    )
    assert not r_frame.accepted and r_frame.status == "coordinate_frame_changed"

    # Plane shift > tolerance -> rejected without mutating state.
    r_plane = recon.integrate_rgbd(
        d0, c0, K0, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.10, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=500, observation_id="f4", camera_id="head",
    )
    assert not r_plane.accepted and r_plane.status == "plane_changed"
    np.testing.assert_array_equal(recon.snapshot().vertices_m, verts_before)

    # Explicit reset clears map, bumps epoch and version, and allows new frame/reused IDs.
    r_rst = recon.reset()
    assert r_rst.accepted and r_rst.status == "reset" and r_rst.epoch == 1
    snap_empty = recon.snapshot()
    assert not snap_empty.ready and snap_empty.reason == "empty_after_reset"
    assert len(snap_empty.vertices_m) == 0 and snap_empty.version == r_rst.version

    # Reset-policy reconstructor automatically resets on frame/calibration/plane change.
    auto = PersistentPlanarSurfaceReconstructor(
        config=_config(
            min_observations_to_commit=1,
            frame_change_policy="reset",
            calibration_change_policy="reset",
            plane_change_policy="reset",
        )
    )
    auto.integrate_rgbd(
        d0, c0, K0, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=100, observation_id="a0", camera_id="head",
    )
    r_auto_frame = auto.integrate_rgbd(
        d0, c0, K0, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="world", timestamp_ns=200, observation_id="a1", camera_id="head",
    )
    assert r_auto_frame.accepted and r_auto_frame.reset_triggered and r_auto_frame.epoch == 1
    assert auto.snapshot().coordinate_frame == "world"


def test_long_session_memory_and_bounds_stay_flat():
    K = _default_K((64, 80))
    T = _look_at((0.0, -0.45, 0.65), (0.0, 0.0, 0.0))
    d, c, _ = _render_planar_rgbd(frame_from_camera=T, K=K, shape=(64, 80))

    cfg = _config(
        max_cells=500,
        max_retained_observations=8,
        max_seen_observation_ids=32,
        max_provenance_observations=16,
        max_cell_weight=10.0,
        min_observations_to_commit=2,
    )
    recon = PersistentPlanarSurfaceReconstructor(config=cfg)
    mem_samples = []
    for i in range(120):
        # Slightly shift view to exercise cell cap & compaction.
        Ti = T.copy()
        Ti[0, 3] = 0.08 * math.sin(i * 0.2)
        res = recon.integrate_rgbd(
            d, c, K, frame_from_camera=Ti,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=300, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=(i + 1) * 33_000_000, observation_id=f"long-{i}",
        )
        assert res.accepted
        if i in (20, 60, 119):
            mem_samples.append(recon.estimated_memory_bytes())

    snap = recon.snapshot()
    assert snap.ready
    assert snap.support_count <= cfg.max_cells
    assert len(snap.contributing_observations) <= cfg.max_provenance_observations
    assert snap.compacted_observation_count == 120 - cfg.max_retained_observations
    # Memory stays flat after the retention window fills!
    assert mem_samples[-1] <= mem_samples[0] * 1.25


def test_integrate_points_and_plane_reuse_and_immutability():
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 25), np.linspace(-0.2, 0.2, 25))
    pts = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    rgb = np.full((len(pts), 3), (80, 160, 220), dtype=np.uint8)
    exc = pts[:, 0] > 0.12

    recon = PersistentPlanarSurfaceReconstructor(config=_config(min_observations_to_commit=2))
    r0 = recon.integrate_points(
        pts, rgb,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=400, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=1_000, observation_id="p0", exclude_mask=exc,
    )
    assert r0.accepted
    # Second frame reuses latched plane by passing plane_normal=None, plane_offset_m=None.
    r1 = recon.integrate_points(
        pts, rgb,
        coordinate_frame="table", timestamp_ns=2_000, observation_id="p1", exclude_mask=exc,
    )
    assert r1.accepted
    snap = recon.snapshot()
    assert snap.ready
    assert float(snap.vertices_m[:, 0].max()) <= 0.121
    assert not snap.vertices_m.flags.writeable
    assert not snap.colors_rgb_u8.flags.writeable
    assert not snap.triangles.flags.writeable
    assert not snap.vertex_last_observed_ns.flags.writeable


def test_malformed_inputs_raise_value_error_without_corrupting_state():
    K = _default_K((32, 32))
    T = _look_at((0.0, -0.45, 0.65), (0.0, 0.0, 0.0))
    d, c, _ = _render_planar_rgbd(frame_from_camera=T, K=K, shape=(32, 32))
    recon = PersistentPlanarSurfaceReconstructor(config=_config(min_observations_to_commit=1))
    recon.integrate_rgbd(
        d, c, K, frame_from_camera=T,
        plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=200, plane_rmse_m=0.001,
        coordinate_frame="table", timestamp_ns=100, observation_id="good-0",
    )
    v0 = recon.version

    bad_T = np.diag([2.0, 2.0, 2.0, 1.0])  # scaling, not SE(3)
    with pytest.raises(ValueError, match="rotation"):
        recon.integrate_rgbd(
            d, c, K, frame_from_camera=bad_T,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=200, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=200, observation_id="bad-T",
        )
    with pytest.raises(ValueError, match="intrinsics"):
        recon.integrate_rgbd(
            d, c, K * 0.0, frame_from_camera=T,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=200, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=200, observation_id="bad-K",
        )
    with pytest.raises(ValueError, match="exclude_mask"):
        recon.integrate_rgbd(
            d, c, K, frame_from_camera=T,
            plane_normal=(0, 0, 1), plane_offset_m=0.0, plane_support_count=200, plane_rmse_m=0.001,
            coordinate_frame="table", timestamp_ns=200, observation_id="bad-mask",
            exclude_mask=np.zeros(d.shape, dtype=np.int32),
        )
    with pytest.raises(ValueError, match="policy"):
        PersistentPlanarSurfaceConfig(frame_change_policy="ignore")  # type: ignore[arg-type]

    assert recon.version == v0
