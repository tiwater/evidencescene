"""Validate persistent planar static surface reconstruction on calibrated RGB-D NPZ sequences.

Each NPZ frame supplies registered RGB-D and calibration in a common frame:
  depth_m (HxW float32/64), rgb_u8 (HxWx3 uint8), K (3x3 float64),
  frame_from_camera (4x4 float64 SE3), plane_normal (3,), plane_offset_m (float),
  plane_support_count (int), plane_rmse_m (float), timestamp_ns (int),
  observation_id (str), camera_id (str), coordinate_frame (str),
  optional exclude_mask (HxW bool), optional phase (str: e.g. 'initial', 'occlusion', 'new_view').

Optional ``reference_table_points.npz`` in the sequence directory supplies a
held-out measured reference cloud (``points_m`` Nx3) for geometric error metrics.

Raw sensor recordings are never embedded in the scorecard. Compact PLY mesh
artifacts are written for independent inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
from scipy.spatial import cKDTree

from evidencescene import __version__
from evidencescene.primitives.geometry import _backproject_masked_depth
from evidencescene.surface import (
    PersistentPlanarSurfaceConfig,
    PersistentPlanarSurfaceReconstructor,
    SurfaceMeshConfig,
    reconstruct_planar_surface,
)


def write_ply_mesh(path: Path, vertices_m: np.ndarray, colors_rgb_u8: np.ndarray, triangles: np.ndarray) -> int:
    """Write a binary little-endian PLY triangle mesh with per-vertex RGB colors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices_m, dtype=np.float32)
    c = np.asarray(colors_rgb_u8, dtype=np.uint8)
    f = np.asarray(triangles, dtype=np.int32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(f)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    v_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v_buf = np.empty(len(v), dtype=v_dtype)
    if len(v):
        v_buf["x"], v_buf["y"], v_buf["z"] = v[:, 0], v[:, 1], v[:, 2]
        v_buf["r"], v_buf["g"], v_buf["b"] = c[:, 0], c[:, 1], c[:, 2]

    f_dtype = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    f_buf = np.empty(len(f), dtype=f_dtype)
    if len(f):
        f_buf["n"] = 3
        f_buf["i0"], f_buf["i1"], f_buf["i2"] = f[:, 0], f[:, 1], f[:, 2]

    payload = header + v_buf.tobytes() + f_buf.tobytes()
    path.write_bytes(payload)
    return len(payload)


def snapshot_array_bytes(snap) -> int:
    return int(
        snap.vertices_m.nbytes
        + snap.colors_rgb_u8.nbytes
        + snap.triangles.nbytes
        + snap.face_normals.nbytes
        + snap.vertex_observation_counts.nbytes
        + snap.vertex_last_observed_ns.nbytes
    )


def baseline_planar_from_rgbd(
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    K: np.ndarray,
    frame_from_camera: np.ndarray,
    *,
    plane_normal: np.ndarray,
    plane_offset_m: float,
    plane_support_count: int,
    plane_rmse_m: float,
    coordinate_frame: str,
    timestamp_ns: int,
    source_id: str,
    exclude_mask: np.ndarray | None,
    pixel_step: int,
    mesh_config: SurfaceMeshConfig,
):
    selected = np.ones(depth_m.shape, dtype=bool)
    if exclude_mask is not None:
        selected &= ~exclude_mask
    sampled = depth_m[::pixel_step, ::pixel_step]
    valid = selected[::pixel_step, ::pixel_step] & np.isfinite(sampled) & (sampled >= 0.15) & (sampled <= 3.0)
    vv, uu = np.nonzero(valid)
    vv, uu = vv * pixel_step, uu * pixel_step
    if len(vv) == 0:
        pts_frame = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)
    else:
        pts_cam = _backproject_masked_depth(
            depth_m, selected, K, pixel_stride=pixel_step, minimum_depth_m=0.15, maximum_depth_m=3.0
        )
        pts_frame = pts_cam @ frame_from_camera[:3, :3].T + frame_from_camera[:3, 3][None, :]
        colors = rgb_u8[vv, uu]
    return reconstruct_planar_surface(
        pts_frame,
        colors,
        plane_normal=plane_normal,
        plane_offset_m=plane_offset_m,
        plane_support_count=plane_support_count,
        plane_rmse_m=plane_rmse_m,
        coordinate_frame=coordinate_frame,
        timestamp_ns=timestamp_ns,
        source_id=source_id,
        config=mesh_config,
    )


def evaluate_against_reference(vertices_m: np.ndarray, ref_points_m: np.ndarray) -> dict[str, float]:
    if len(vertices_m) == 0 or len(ref_points_m) == 0:
        return {}
    tree = cKDTree(ref_points_m)
    dists, idx = tree.query(vertices_m, k=1)
    # Vertices inside the reference camera's footprint (within 2cm XY of a held-out reference point)
    # measure geometric error separately from newly discovered out-of-footprint area.
    xy_tree = cKDTree(ref_points_m[:, :2])
    xy_dists, xy_idx = xy_tree.query(vertices_m[:, :2], k=1)
    in_fp = xy_dists <= 0.02
    in_fp_dists = dists[in_fp]
    in_fp_z_err = np.abs(vertices_m[in_fp, 2] - ref_points_m[xy_idx[in_fp], 2])
    rev_tree = cKDTree(vertices_m)
    rev_dists, _ = rev_tree.query(ref_points_m, k=1)
    return {
        "in_reference_footprint_vertices": int(np.count_nonzero(in_fp)),
        "in_reference_footprint_3d_rmse_m": float(np.sqrt(np.mean(in_fp_dists**2))) if np.any(in_fp) else 0.0,
        "in_reference_footprint_3d_median_m": float(np.median(in_fp_dists)) if np.any(in_fp) else 0.0,
        "in_reference_footprint_z_rmse_m": float(np.sqrt(np.mean(in_fp_z_err**2))) if np.any(in_fp) else 0.0,
        "out_of_reference_footprint_new_vertices": int(np.count_nonzero(~in_fp)),
        "ref_covered_within_20mm_ratio": float(np.mean(rev_dists <= 0.02)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="Directory of exported calibrated RGB-D NPZ frames")
    parser.add_argument("output", type=Path, help="Path to write JSON scorecard")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory to write inspectable PLY meshes")
    parser.add_argument("--cell-size-m", type=float, default=0.01)
    parser.add_argument("--max-edge-m", type=float, default=0.032)
    parser.add_argument("--max-plane-residual-m", type=float, default=0.008)
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--min-observations", type=int, default=2)
    args = parser.parse_args()

    frame_paths = sorted(
        p for p in args.samples.glob("*.npz") if p.name != "reference_table_points.npz"
    )
    assert frame_paths, f"No NPZ frames found in {args.samples}"

    ref_path = args.samples / "reference_table_points.npz"
    ref_points_m = None
    ref_sha256 = None
    if ref_path.is_file():
        ref_sha256 = hashlib.sha256(ref_path.read_bytes()).hexdigest()
        with np.load(ref_path, allow_pickle=False) as rf:
            ref_points_m = np.asarray(rf["points_m"], dtype=np.float64)

    mesh_cfg = SurfaceMeshConfig(
        cell_size_m=args.cell_size_m,
        maximum_edge_m=args.max_edge_m,
        maximum_plane_residual_m=args.max_plane_residual_m,
        maximum_normal_step_m=args.max_plane_residual_m,
        minimum_support=80,
    )
    cfg = PersistentPlanarSurfaceConfig(
        mesh=mesh_cfg,
        min_observations_to_commit=args.min_observations,
        pixel_step=args.pixel_step,
        max_retained_observations=64,
    )
    recon = PersistentPlanarSurfaceReconstructor(config=cfg)

    tracemalloc.start()
    rows = []
    update_ms_list = []
    snapshot_ms_list = []
    baseline_ms_list = []

    best_single_view_row = None
    best_single_view_mesh = None
    worst_occlusion_row = None
    worst_occlusion_baseline_mesh = None
    worst_occlusion_persistent_snap = None
    last_frame_data = None
    first_camera_id = None
    first_K = None
    first_frame_name = None

    for idx, path in enumerate(frame_paths):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as f:
            depth_m = np.asarray(f["depth_m"], dtype=np.float64)
            rgb_u8 = np.asarray(f["rgb_u8"], dtype=np.uint8)
            K = np.asarray(f["K"], dtype=np.float64)
            T_fc = np.asarray(f["frame_from_camera"], dtype=np.float64)
            plane_normal = np.asarray(f["plane_normal"], dtype=np.float64)
            plane_offset_m = float(f["plane_offset_m"])
            plane_support_count = int(f["plane_support_count"])
            plane_rmse_m = float(f["plane_rmse_m"])
            timestamp_ns = int(f["timestamp_ns"])
            observation_id = str(f["observation_id"])
            camera_id = str(f["camera_id"])
            coordinate_frame = str(f["coordinate_frame"])
            exclude_mask = np.asarray(f["exclude_mask"], dtype=bool) if "exclude_mask" in f else None
            phase = str(f["phase"]) if "phase" in f else "stream"

        if first_camera_id is None:
            first_camera_id, first_K, first_frame_name = camera_id, K.copy(), coordinate_frame
        last_frame_data = (
            depth_m, rgb_u8, K, T_fc, plane_normal, plane_offset_m,
            plane_support_count, plane_rmse_m, coordinate_frame, timestamp_ns, observation_id, camera_id, exclude_mask,
        )

        # Baseline per-observation planar mesh (v2.1.0).
        t0 = time.perf_counter()
        base_mesh = baseline_planar_from_rgbd(
            depth_m, rgb_u8, K, T_fc,
            plane_normal=plane_normal, plane_offset_m=plane_offset_m,
            plane_support_count=plane_support_count, plane_rmse_m=plane_rmse_m,
            coordinate_frame=coordinate_frame, timestamp_ns=timestamp_ns,
            source_id=observation_id, exclude_mask=exclude_mask,
            pixel_step=args.pixel_step, mesh_config=mesh_cfg,
        )
        base_ms = (time.perf_counter() - t0) * 1000.0
        baseline_ms_list.append(base_ms)

        # Persistent reconstruction update.
        t1 = time.perf_counter()
        upd = recon.integrate_rgbd(
            depth_m, rgb_u8, K,
            frame_from_camera=T_fc,
            plane_normal=plane_normal, plane_offset_m=plane_offset_m,
            plane_support_count=plane_support_count, plane_rmse_m=plane_rmse_m,
            coordinate_frame=coordinate_frame, timestamp_ns=timestamp_ns,
            observation_id=observation_id, camera_id=camera_id,
            exclude_mask=exclude_mask, pixel_step=args.pixel_step,
        )
        upd_ms = (time.perf_counter() - t1) * 1000.0
        update_ms_list.append(upd_ms)

        t2 = time.perf_counter()
        snap = recon.snapshot(reference_timestamp_ns=timestamp_ns)
        snap_ms = (time.perf_counter() - t2) * 1000.0
        snapshot_ms_list.append(snap_ms)

        assert upd.accepted, (path.name, upd)
        if snap.ready:
            residuals = np.abs(snap.vertices_m @ plane_normal + plane_offset_m)
            pers_rmse = float(np.sqrt(np.mean(residuals**2)))
            pers_max = float(np.max(residuals))
            assert pers_max <= mesh_cfg.maximum_plane_residual_m + 1e-6
            assert snap.maximum_edge_m <= mesh_cfg.maximum_edge_m + 1e-6
        else:
            pers_rmse = None
            pers_max = None

        row = {
            "index": idx,
            "sample": path.name,
            "input_sha256": sha,
            "observation_id": observation_id,
            "camera_id": camera_id,
            "phase": phase,
            "timestamp_ns": str(timestamp_ns),
            "baseline": {
                "ready": base_mesh.ready,
                "reason": base_mesh.reason,
                "vertices": len(base_mesh.vertices_m),
                "triangles": len(base_mesh.triangles),
                "area_m2": base_mesh.surface_area_m2,
                "processing_ms": base_ms,
            },
            "persistent": {
                "ready": snap.ready,
                "reason": snap.reason,
                "version": snap.version,
                "vertices": len(snap.vertices_m),
                "triangles": len(snap.triangles),
                "area_m2": snap.surface_area_m2,
                "maximum_edge_m": snap.maximum_edge_m,
                "plane_rmse_m": pers_rmse,
                "plane_max_error_m": pers_max,
                "newest_observation_lag_ns": snap.newest_observation_lag_ns,
                "oldest_vertex_lag_ns": snap.oldest_vertex_lag_ns,
                "snapshot_array_bytes": snapshot_array_bytes(snap),
                "estimated_memory_bytes": snap.estimated_memory_bytes,
                "update_ms": upd_ms,
                "snapshot_ms": snap_ms,
            },
        }
        rows.append(row)

        if base_mesh.ready and phase == "initial":
            if best_single_view_row is None or base_mesh.surface_area_m2 > best_single_view_row["baseline"]["area_m2"]:
                best_single_view_row = row
                best_single_view_mesh = base_mesh

        if phase == "occlusion" and snap.ready:
            if worst_occlusion_row is None or base_mesh.surface_area_m2 < worst_occlusion_row["baseline"]["area_m2"]:
                worst_occlusion_row = row
                worst_occlusion_baseline_mesh = base_mesh
                worst_occlusion_persistent_snap = snap

    final_snap = recon.snapshot()
    assert final_snap.ready, final_snap.reason
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Lifecycle verification (duplicate, pose correction, frame/calibration change, reset).
    (
        l_d, l_c, l_K, l_T, l_pn, l_po, l_ps, l_pr, l_frame, l_ts, l_oid, l_cid, l_exc
    ) = last_frame_data
    dup_res = recon.integrate_rgbd(
        l_d, l_c, l_K, frame_from_camera=l_T,
        plane_normal=l_pn, plane_offset_m=l_po, plane_support_count=l_ps, plane_rmse_m=l_pr,
        coordinate_frame=l_frame, timestamp_ns=l_ts, observation_id=l_oid, camera_id=l_cid, exclude_mask=l_exc,
    )
    assert not dup_res.accepted and dup_res.status == "duplicate_ignored"

    T_perturbed = l_T.copy()
    T_perturbed[0, 3] += 0.015
    corr_res = recon.correct_pose(l_oid, T_perturbed)
    assert corr_res.accepted and corr_res.status == "pose_corrected"
    restore_res = recon.correct_pose(l_oid, l_T)
    assert restore_res.accepted and restore_res.status == "pose_corrected"

    frame_chg = recon.integrate_rgbd(
        l_d, l_c, l_K, frame_from_camera=l_T,
        plane_normal=l_pn, plane_offset_m=l_po, plane_support_count=l_ps, plane_rmse_m=l_pr,
        coordinate_frame=f"{l_frame}_other", timestamp_ns=l_ts + 1, observation_id="lifecycle-frame-chg", camera_id=l_cid,
    )
    assert not frame_chg.accepted and frame_chg.status == "coordinate_frame_changed"

    bad_K = first_K.copy()
    bad_K[0, 0] *= 1.25
    cal_chg = recon.integrate_rgbd(
        l_d, l_c, bad_K, frame_from_camera=l_T,
        plane_normal=l_pn, plane_offset_m=l_po, plane_support_count=l_ps, plane_rmse_m=l_pr,
        coordinate_frame=first_frame_name, timestamp_ns=l_ts + 2, observation_id="lifecycle-cal-chg", camera_id=first_camera_id,
    )
    assert not cal_chg.accepted and cal_chg.status == "camera_calibration_changed"

    rst = recon.reset()
    snap_after_rst = recon.snapshot()
    assert rst.accepted and rst.status == "reset" and rst.epoch == 1
    assert not snap_after_rst.ready and snap_after_rst.reason == "empty_after_reset"
    assert len(snap_after_rst.vertices_m) == 0 and len(snap_after_rst.triangles) == 0

    # Write inspectable PLY mesh artifacts.
    artifacts = {}
    if best_single_view_mesh is not None:
        p = args.artifacts_dir / "recorded_baseline_single_view.ply"
        nbytes = write_ply_mesh(p, best_single_view_mesh.vertices_m, best_single_view_mesh.colors_rgb_u8, best_single_view_mesh.triangles)
        artifacts["baseline_single_view"] = {
            "path": str(p.name),
            "sample": best_single_view_row["sample"],
            "vertices": len(best_single_view_mesh.vertices_m),
            "triangles": len(best_single_view_mesh.triangles),
            "area_m2": best_single_view_mesh.surface_area_m2,
            "ply_bytes": nbytes,
        }
    if worst_occlusion_baseline_mesh is not None and worst_occlusion_persistent_snap is not None:
        p_bo = args.artifacts_dir / "recorded_baseline_during_occlusion.ply"
        n_bo = write_ply_mesh(
            p_bo, worst_occlusion_baseline_mesh.vertices_m,
            worst_occlusion_baseline_mesh.colors_rgb_u8, worst_occlusion_baseline_mesh.triangles,
        )
        p_po = args.artifacts_dir / "recorded_persistent_during_occlusion.ply"
        n_po = write_ply_mesh(
            p_po, worst_occlusion_persistent_snap.vertices_m,
            worst_occlusion_persistent_snap.colors_rgb_u8, worst_occlusion_persistent_snap.triangles,
        )
        artifacts["baseline_during_occlusion"] = {
            "path": str(p_bo.name),
            "sample": worst_occlusion_row["sample"],
            "vertices": len(worst_occlusion_baseline_mesh.vertices_m),
            "triangles": len(worst_occlusion_baseline_mesh.triangles),
            "area_m2": worst_occlusion_baseline_mesh.surface_area_m2,
            "ply_bytes": n_bo,
        }
        artifacts["persistent_during_occlusion"] = {
            "path": str(p_po.name),
            "sample": worst_occlusion_row["sample"],
            "vertices": len(worst_occlusion_persistent_snap.vertices_m),
            "triangles": len(worst_occlusion_persistent_snap.triangles),
            "area_m2": worst_occlusion_persistent_snap.surface_area_m2,
            "newest_observation_lag_ns": worst_occlusion_persistent_snap.newest_observation_lag_ns,
            "oldest_vertex_lag_ns": worst_occlusion_persistent_snap.oldest_vertex_lag_ns,
            "ply_bytes": n_po,
        }

    p_final = args.artifacts_dir / "recorded_persistent_final.ply"
    n_final = write_ply_mesh(p_final, final_snap.vertices_m, final_snap.colors_rgb_u8, final_snap.triangles)
    artifacts["persistent_final"] = {
        "path": str(p_final.name),
        "vertices": len(final_snap.vertices_m),
        "triangles": len(final_snap.triangles),
        "area_m2": final_snap.surface_area_m2,
        "ply_bytes": n_final,
    }

    initial_rows = [r for r in rows if r["phase"] == "initial" and r["persistent"]["ready"]]
    occ_rows = [r for r in rows if r["phase"] == "occlusion"]
    new_view_rows = [r for r in rows if r["phase"] == "new_view"]
    base_ready_areas = [r["baseline"]["area_m2"] for r in rows if r["baseline"]["ready"]]

    initial_area = initial_rows[-1]["persistent"]["area_m2"] if initial_rows else 0.0
    final_area = final_snap.surface_area_m2
    occ_base_min_area = min((r["baseline"]["area_m2"] for r in occ_rows), default=0.0)
    occ_pers_min_area = min((r["persistent"]["area_m2"] for r in occ_rows), default=0.0)
    occ_retention_ratio = (occ_pers_min_area / initial_area) if initial_area > 0 else 0.0

    ref_eval_persistent = (
        evaluate_against_reference(final_snap.vertices_m, ref_points_m) if ref_points_m is not None else {}
    )
    ref_eval_baseline = (
        evaluate_against_reference(best_single_view_mesh.vertices_m, ref_points_m)
        if (ref_points_m is not None and best_single_view_mesh is not None)
        else {}
    )

    summary = {
        "schema": "evidencescene.persistent-planar-surface-validation.v1",
        "version": __version__,
        "frames": len(rows),
        "cameras": sorted({r["camera_id"] for r in rows}),
        "phases": {
            p: sum(1 for r in rows if r["phase"] == p)
            for p in sorted({r["phase"] for r in rows})
        },
        "config": {
            "cell_size_m": args.cell_size_m,
            "maximum_edge_m": args.max_edge_m,
            "maximum_plane_residual_m": args.max_plane_residual_m,
            "pixel_step": args.pixel_step,
            "min_observations_to_commit": args.min_observations,
            "max_retained_observations": cfg.max_retained_observations,
            "max_cells": cfg.max_cells,
        },
        "coverage_comparison": {
            "baseline_median_area_m2": float(np.median(base_ready_areas)) if base_ready_areas else 0.0,
            "baseline_best_single_view_area_m2": float(max(base_ready_areas)) if base_ready_areas else 0.0,
            "baseline_min_occlusion_area_m2": float(occ_base_min_area),
            "persistent_after_initial_area_m2": float(initial_area),
            "persistent_min_occlusion_area_m2": float(occ_pers_min_area),
            "persistent_occlusion_retention_ratio": float(occ_retention_ratio),
            "persistent_final_area_m2": float(final_area),
            "coverage_gain_over_initial_ratio": float(final_area / initial_area) if initial_area > 0 else 0.0,
            "coverage_gain_over_best_single_view_ratio": (
                float(final_area / max(base_ready_areas)) if base_ready_areas and max(base_ready_areas) > 0 else 0.0
            ),
            "new_view_frames": len(new_view_rows),
        },
        "geometric_error": {
            "final_plane_rmse_m": rows[-1]["persistent"]["plane_rmse_m"],
            "final_plane_max_error_m": rows[-1]["persistent"]["plane_max_error_m"],
            "reference_cloud_sha256": ref_sha256,
            "reference_cloud_points": 0 if ref_points_m is None else int(len(ref_points_m)),
            "persistent_vs_reference": ref_eval_persistent,
            "best_single_view_baseline_vs_reference": ref_eval_baseline,
        },
        "performance_and_memory": {
            "update_latency_p50_ms": float(np.median(update_ms_list)),
            "update_latency_p95_ms": float(np.percentile(update_ms_list, 95)),
            "update_latency_max_ms": float(np.max(update_ms_list)),
            "snapshot_latency_p50_ms": float(np.median(snapshot_ms_list)),
            "snapshot_latency_p95_ms": float(np.percentile(snapshot_ms_list, 95)),
            "baseline_latency_p50_ms": float(np.median(baseline_ms_list)),
            "final_snapshot_vertices": len(final_snap.vertices_m),
            "final_snapshot_triangles": len(final_snap.triangles),
            "final_snapshot_array_bytes": snapshot_array_bytes(final_snap),
            "reconstructor_estimated_memory_bytes": final_snap.estimated_memory_bytes,
            "tracemalloc_current_bytes": int(curr_mem),
            "tracemalloc_peak_bytes": int(peak_mem),
        },
        "lifecycle_checks": {
            "duplicate_ignored": True,
            "pose_corrected_and_restored": True,
            "coordinate_frame_change_rejected": True,
            "camera_calibration_change_rejected": True,
            "explicit_reset_emptied_snapshot": True,
        },
        "artifacts": artifacts,
        "rows": rows,
        "limits": [
            "Recorded multi-camera tabletop teleoperation RGB-D sequence; not a claim of cross-environment generalization.",
            "Reference cloud is held-out measured multi-camera depth from the same physical capture, not laser-scanner CAD ground truth.",
            "Triangle interiors are bounded local interpolation; unknown regions remain open and meshes are not watertight.",
            "Timing and tracemalloc figures reflect a single-machine local replay.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    compact = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
