#!/usr/bin/env python3
"""Score all 9/9 YCBInEOAT & 3/3 Laval sequences with Surface & Block Bootstrap CIs + LaTeX export."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evidencescene.metrics import (  # noqa: E402
    align_initial_gauge,
    compute_moving_block_bootstrap_ci,
    compute_paired_block_bootstrap_diff_ci,
    evaluate_pose_sequence,
    evaluate_surface_reconstruction,
    per_frame_auc_contributions,
)
from evidencescene.primitives import backproject_masked_depth_points  # noqa: E402
from evidencescene.surface import (  # noqa: E402
    RevisableSurfelMap,
    select_existing_points_shadow,
)
from suite_common import (  # noqa: E402
    LAVAL_ACTION_IDS,
    YCBINEOAT_SPECS,
    default_paths,
    erode_mask_3x3,
    load_laval_frames_and_gt,
    load_obj_vertices,
    load_ycb_gt_poses,
    load_ycb_sequence_stems,
    polar_so3,
    sample_mesh_surface_points,
    write_json,
)

HISTORICAL_P2P_REF = {
    "cracker_box_reorient": (365, 0.961),
    "mustard0": (726, 0.946),
    "bleach0": (155, 0.798),
    "sugar_box_yalehand0": (617, 0.915),
    "tomato_soup_can_yalehand0": (1301, 0.957),
    "bleach_hard_00_03_chaitanya": (416, 0.948),
}


def _verify_historical_gt2est_receipt(
    seq_id: str,
    est_poses: list[np.ndarray | None],
    gt_poses: list[np.ndarray],
    verts: np.ndarray,
    exp_cnt: int,
    exp_auc: float,
) -> bool:
    """Verify reverse ref->est ADD-S parity against frozen October 2026 receipts."""
    from scipy.spatial import cKDTree
    from evidencescene._accel import NATIVE

    if est_poses[0] is None:
        raise ValueError(f"Missing initial pose for {seq_id}")
    gauge = align_initial_gauge(est_poses[0], gt_poses[0])
    errs_m: list[float] = []
    for est, gt in zip(est_poses[1:], gt_poses[1:]):
        if est is None:
            continue
        aligned = est @ gauge
        pts_est = np.ascontiguousarray(verts @ aligned[:3, :3].T + aligned[:3, 3], dtype=np.float64)
        pts_ref = np.ascontiguousarray(verts @ gt[:3, :3].T + gt[:3, 3], dtype=np.float64)
        if NATIVE is not None:
            dists = NATIVE.nn_min_dists_3d(pts_est, pts_ref)
            if dists is NotImplemented:
                dists, _ = cKDTree(pts_est).query(pts_ref, k=1)
        else:
            dists, _ = cKDTree(pts_est).query(pts_ref, k=1)
        errs_m.append(float(np.mean(dists)))
    denom = len(gt_poses) - 1
    arr = np.asarray(errs_m, dtype=np.float64)
    gt2est_cnt = int(np.sum(arr <= 0.010))
    fin = arr[arr <= 0.100]
    gt2est_auc = float(np.sum(1.0 - fin / 0.100) / denom)
    assert gt2est_cnt == exp_cnt, (seq_id, gt2est_cnt, exp_cnt)
    assert abs(gt2est_auc - exp_auc) <= 0.001, (seq_id, gt2est_auc, exp_auc)
    return True


def _invert_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r_t = pose[:3, :3].T
    inv[:3, :3] = r_t
    inv[:3, 3] = -r_t @ pose[:3, 3]
    return inv


def _extract_p2p_poses(results_path: Path) -> list[np.ndarray | None]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    out: list[np.ndarray | None] = []
    for row in rows:
        lost = row.get("lost")
        pose = row.get("pose")
        if pose is None or lost is None:
            out.append(None)
            continue
        is_lost = bool(lost[0]) if isinstance(lost, list) else bool(lost)
        if is_lost:
            out.append(None)
            continue
        mat = np.asarray(pose, dtype=np.float64)
        if mat.ndim == 3:
            mat = mat[0]
        out.append(polar_so3(mat))
    return out


def _extract_bt_poses(results_path: Path) -> list[np.ndarray | None]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    out: list[np.ndarray | None] = []
    for row in rows:
        if row.get("status") != "candidate" or row.get("pose") is None:
            out.append(None)
            continue
        out.append(polar_so3(row["pose"]))
    return out


def _extract_fp_poses(results_path: Path) -> list[np.ndarray | None]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    out: list[np.ndarray | None] = []
    for row in rows:
        pose = row.get("camera_from_model")
        out.append(polar_so3(pose))
    return out


def _reconstruct_and_score_surface_ycb(
    *,
    seq_dir: Path,
    stems: list[str],
    est_poses: list[np.ndarray | None],
    gt0: np.ndarray,
    model_samples_obj: np.ndarray,
    mask_mode: str,
    gen_masks_dir: Path | None,
    stride: int = 5,
    budget: int = 1500,
) -> dict[str, Any]:
    """Reconstruct causal Frame-0 gauge-aligned surface from strided views & score against CAD mesh."""
    if est_poses[0] is None:
        return {"P10_pct": 0.0, "C10_pct": 0.0, "F1_10_pct": 0.0, "F95_mm": float("inf"), "R95_mm": float("inf")}
    gauge = align_initial_gauge(est_poses[0], gt0)
    intrinsics = np.loadtxt(seq_dir / "cam_K.txt").reshape(3, 3)

    # Frame-0 GT-aligned reference points in object Frame-0 gauge (identity == gt0 for YCB?
    # Note: aligned_est_i = est_i @ gauge transforms model vertex coords -> camera frame at step i,
    # matching gt_i! Let obj_0 gauge be Frame-0 GT object model frame.)
    # Therefore T_obj_from_cam = inv(aligned_est_i).
    surfel_map = RevisableSurfelMap(
        voxel_size_m=0.004, maximum_voxels=200_000, maximum_live_bytes=128 * 1024 * 1024
    )
    integrated = 0
    for i, stem in enumerate(stems):
        if i % stride != 0:
            continue
        p = est_poses[i]
        if p is None:
            continue
        depth_m = cv2.imread(str(seq_dir / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if mask_mode == "reference":
            raw_m = cv2.imread(str(seq_dir / "gt_mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            m_bool = erode_mask_3x3(np.any(raw_m > 0, axis=2) if raw_m.ndim == 3 else (raw_m > 0))
        else:
            assert gen_masks_dir is not None
            raw_m = cv2.imread(str(gen_masks_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            if raw_m is None:
                continue
            m_bool = erode_mask_3x3(raw_m > 0)
        if not m_bool.any():
            continue
        pts_cam = backproject_masked_depth_points(
            depth_m,
            m_bool,
            intrinsics,
            minimum_depth_m=0.1,
            maximum_depth_m=1.5,
            pixel_stride=1,
        )
        if len(pts_cam) == 0:
            continue
        t_cam_from_obj = p @ gauge
        t_obj_from_cam = _invert_se3(t_cam_from_obj)
        pts_obj = pts_cam @ t_obj_from_cam[:3, :3].T + t_obj_from_cam[:3, 3]
        surfel_map.update({f"kf_{i:05d}": ("cam0", pts_obj)})
        integrated += 1

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
            "integrated_views": integrated,
            "supported_surfels": 0,
        }
    sel = select_existing_points_shadow(full_xyz, full_sup, full_cams, full_keys, budget=budget)
    pts_sel = full_xyz[sel["indices"]]
    sc = evaluate_surface_reconstruction(
        pts_sel, model_samples_obj, threshold_m=0.010, grid_voxel_size_m=0.004
    )
    return {
        "P10_pct": round(sc.precision_pct, 2),
        "C10_pct": round(sc.coverage_pct, 2),
        "F1_10_pct": round(sc.f1_pct, 2),
        "F95_mm": round(sc.forward_p95_mm, 2),
        "R95_mm": round(sc.reverse_p95_mm, 2),
        "integrated_views": integrated,
        "supported_surfels": int(len(full_xyz)),
        "selected_points": int(len(pts_sel)),
    }


def _reconstruct_and_score_surface_laval(
    *,
    frames: list[dict[str, Any]],
    intrinsics: np.ndarray,
    est_poses: list[np.ndarray | None],
    gt0: np.ndarray,
    model_samples_obj: np.ndarray,
    stride: int = 3,
    budget: int = 1500,
) -> dict[str, Any]:
    if est_poses[0] is None:
        return {"P10_pct": 0.0, "C10_pct": 0.0, "F1_10_pct": 0.0, "F95_mm": float("inf"), "R95_mm": float("inf")}
    gauge = align_initial_gauge(est_poses[0], gt0)
    surfel_map = RevisableSurfelMap(
        voxel_size_m=0.004, maximum_voxels=250_000, maximum_live_bytes=128 * 1024 * 1024
    )
    integrated = 0
    for f in frames:
        i = f["index"]
        if i % stride != 0:
            continue
        p = est_poses[i]
        if p is None or f["status"] != "available" or not f["mask"].any():
            continue
        m_bool = erode_mask_3x3(f["mask"])
        if not m_bool.any():
            continue
        pts_cam = backproject_masked_depth_points(
            f["depth_m"],
            m_bool,
            intrinsics,
            minimum_depth_m=0.1,
            maximum_depth_m=2.0,
            pixel_stride=1,
        )
        if len(pts_cam) == 0:
            continue
        t_cam_from_obj = p @ gauge
        t_obj_from_cam = _invert_se3(t_cam_from_obj)
        pts_obj = pts_cam @ t_obj_from_cam[:3, :3].T + t_obj_from_cam[:3, 3]
        surfel_map.update({f"kf_{i:05d}": ("chest", pts_obj)})
        integrated += 1

    full_xyz, full_sup, full_cams, full_keys = surfel_map.snapshot(
        minimum_observation_support=2,
        maximum_point_count=surfel_map.maximum_voxels,
        include_voxel_keys=True,
    )
    if len(full_xyz) == 0:
        return {"P10_pct": 0.0, "C10_pct": 0.0, "F1_10_pct": 0.0, "F95_mm": float("inf"), "R95_mm": float("inf")}
    sel = select_existing_points_shadow(full_xyz, full_sup, full_cams, full_keys, budget=budget)
    sc = evaluate_surface_reconstruction(
        full_xyz[sel["indices"]], model_samples_obj, threshold_m=0.010, grid_voxel_size_m=0.004
    )
    return {
        "P10_pct": round(sc.precision_pct, 2),
        "C10_pct": round(sc.coverage_pct, 2),
        "F1_10_pct": round(sc.f1_pct, 2),
        "F95_mm": round(sc.forward_p95_mm, 2),
        "R95_mm": round(sc.reverse_p95_mm, 2),
        "integrated_views": integrated,
        "supported_surfels": int(len(full_xyz)),
    }


def _score_pose_and_ci(
    est_poses: list[np.ndarray | None],
    gt_poses: list[np.ndarray],
    verts: np.ndarray,
    *,
    align_frame_zero: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    sc_add_and_s = evaluate_pose_sequence(
        est_poses,
        gt_poses,
        verts,
        align_frame_zero=align_frame_zero,
        exclude_frame_zero=True,
        threshold_m=0.010,
        strict_rotation_deg=10.0,
    )
    c_add, c_adds = per_frame_auc_contributions(
        est_poses, gt_poses, verts, align_frame_zero=align_frame_zero, exclude_frame_zero=True
    )
    ci_add = compute_moving_block_bootstrap_ci(c_add, block_length=20, num_resamples=2000, seed=20261121)
    ci_adds = compute_moving_block_bootstrap_ci(c_adds, block_length=20, num_resamples=2000, seed=20261121)
    row = {
        "n_post_init": sc_add_and_s.denominator_excluding_initial,
        "avail": sc_add_and_s.available_post_initial_count,
        "add_le_10mm": sc_add_and_s.add_le_threshold_count,
        "adds_le_10mm": sc_add_and_s.adds_le_threshold_count,
        "strict_10mm_10deg": sc_add_and_s.strict_threshold_count,
        "add_auc": round(sc_add_and_s.add_auc_0_to_max, 4),
        "add_auc_ci95": [round(ci_add.ci_low, 4), round(ci_add.ci_high, 4)],
        "adds_auc": round(sc_add_and_s.adds_auc_0_to_max, 4),
        "adds_auc_ci95": [round(ci_adds.ci_low, 4), round(ci_adds.ci_high, 4)],
    }
    return row, c_add, c_adds


def _latency_from_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8")).get("latency")


def _write_latex_tables(
    ycb_payload: dict[str, Any],
    laval_payload: dict[str, Any],
    papers_dir: Path,
) -> None:
    # 1. Write papers/evidencescene/ycb-full-suite-table.tex
    lines = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\caption{Full 9/9 Official YCBInEOAT Suite ($7{,}440$ Post-Initial Frames) Across Both Reference Masks and Shared Autonomous SAM~2 Video Masks. ES: \textbf{EvidenceScene} (Composed Native BundleTrack + \texttt{RevisableSurfelMap} $1{,}500$-pt budget); P2P: Point2Pose (\texttt{51856226610d}); FP: FoundationPose (CVPR'24, CAD-Assisted Reference Given Exact YCB Mesh). Paired deltas use 95\% Moving-Block Bootstrap CIs ($b=20, B=2{,}000$).}\label{tab:ycb_full_9}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrrrrrrrrrrr@{}}\toprule",
        r"\textbf{Sequence} ($n$) & \multicolumn{3}{c}{\textbf{Ref-Mask ADD-S / ADD AUC}} & \multicolumn{2}{c}{\textbf{Ref $\le 10$ mm}} & \multicolumn{2}{c}{\textbf{SAM~2 Auto ADD-S / ADD AUC}} & \multicolumn{2}{c}{\textbf{Auto $\le 10$ mm}} & \multicolumn{2}{c}{\textbf{Ref Surf. $P_{10}/C_{10}$ ($F_{95}$) / Lat. $p_{50}$}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r" & P2P & \textbf{ES (Ours)} & FP (CAD*) & P2P & \textbf{ES} & P2P & \textbf{ES (Ours)} & P2P & \textbf{ES} & P2P & \textbf{ES (Ours)} \\ \midrule",
    ]
    for row in ycb_payload["sequences"]:
        n = row["n_post_init"]
        p_ref = row["reference_mask"]["point2pose"]["pose"]
        e_ref = row["reference_mask"]["evidencescene"]["pose"]
        f_ref = row["reference_mask"]["foundationpose_cad_f0_aligned"]["pose"]
        p_gen = row["generated_sam2_mask"]["point2pose"]["pose"]
        e_gen = row["generated_sam2_mask"]["evidencescene"]["pose"]
        ps = row["reference_mask"]["point2pose"]["surface"]
        es = row["reference_mask"]["evidencescene"]["surface"]
        lat_es = row["reference_mask"]["evidencescene"].get("latency") or {}
        lat_txt = f"{lat_es.get('p50_ms', 0.0):.1f}" if lat_es else "--"
        lines.append(
            f"{row['short_label']} ({n}) & "
            f"{p_ref['adds_auc']:.3f}/{p_ref['add_auc']:.3f} & "
            f"{e_ref['adds_auc']:.3f}/{e_ref['add_auc']:.3f} & "
            f"{f_ref['adds_auc']:.3f}/{f_ref['add_auc']:.3f} & "
            f"{p_ref['adds_le_10mm']} & {e_ref['adds_le_10mm']} & "
            f"{p_gen['adds_auc']:.3f}/{p_gen['add_auc']:.3f} & "
            f"{e_gen['adds_auc']:.3f}/{e_gen['add_auc']:.3f} & "
            f"{p_gen['adds_le_10mm']} & {e_gen['adds_le_10mm']} & "
            f"{ps['P10_pct']:.1f}/{ps['C10_pct']:.1f} ({ps['F95_mm']:.1f}) & "
            f"{es['P10_pct']:.1f}/{es['C10_pct']:.1f} ({es['F95_mm']:.1f}) / {lat_txt} ms \\\\"
        )

    agg = ycb_payload["aggregate"]
    pr = agg["reference_mask"]["point2pose"]
    er = agg["reference_mask"]["evidencescene"]
    fr = agg["reference_mask"]["foundationpose_cad_f0_aligned"]
    pg = agg["generated_sam2_mask"]["point2pose"]
    eg = agg["generated_sam2_mask"]["evidencescene"]
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{Pooled Frame-Wt.}} (7,440) & "
        f"{pr['pooled_adds_auc']:.4f}/{pr['pooled_add_auc']:.4f} & "
        f"\\textbf{{{er['pooled_adds_auc']:.4f}}}/{er['pooled_add_auc']:.4f} & "
        f"{fr['pooled_adds_auc']:.4f}/{fr['pooled_add_auc']:.4f} & "
        f"{pr['total_adds_le_10mm']} & \\textbf{{{er['total_adds_le_10mm']}}} & "
        f"{pg['pooled_adds_auc']:.4f}/{pg['pooled_add_auc']:.4f} & "
        f"\\textbf{{{eg['pooled_adds_auc']:.4f}}}/\\textbf{{{eg['pooled_add_auc']:.4f}}} & "
        f"{pg['total_adds_le_10mm']} & \\textbf{{{eg['total_adds_le_10mm']}}} & "
        f"{pr['mean_P10_pct']:.1f}/{pr['mean_C10_pct']:.1f} ({pr['mean_F95_mm']:.1f}) & "
        f"\\textbf{{{er['mean_P10_pct']:.1f}}}/\\textbf{{{er['mean_C10_pct']:.1f}}} (\\textbf{{{er['mean_F95_mm']:.1f}}}) / {er['median_sequence_p50_ms']:.1f} ms \\\\"
    )
    dr = agg["reference_mask"]["paired_delta_es_minus_p2p"]
    dg = agg["generated_sam2_mask"]["paired_delta_es_minus_p2p"]
    lines.append(
        f"\\multicolumn{{12}}{{@{{}}l@{{}}}}{{\\emph{{95\\% Block-Bootstrap Paired $\\Delta$ (ES $-$ P2P):}} "
        f"Ref-Mask $\\Delta$ADD-S AUC $= {dr['adds_auc_diff']:+.4f}$ $[{dr['adds_auc_ci95'][0]:+.4f}, {dr['adds_auc_ci95'][1]:+.4f}]$ "
        f"($\\Delta$ADD $= {dr['add_auc_diff']:+.4f}$ $[{dr['add_auc_ci95'][0]:+.4f}, {dr['add_auc_ci95'][1]:+.4f}]$); "
        f"Auto SAM~2 $\\Delta$ADD-S AUC $= {dg['adds_auc_diff']:+.4f}$ $[{dg['adds_auc_ci95'][0]:+.4f}, {dg['adds_auc_ci95'][1]:+.4f}]$ "
        f"($\\Delta$ADD $= {dg['add_auc_diff']:+.4f}$ $[{dg['add_auc_ci95'][0]:+.4f}, {dg['add_auc_ci95'][1]:+.4f}]$). "
        f"*FP receives the exact CAD mesh.}} \\\\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    (papers_dir / "ycb-full-suite-table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. Write papers/evidencescene/laval-real-unified-2x2-table.tex
    u2 = laval_payload["unified_2x2_real"]
    comp = laval_payload["cross_method_laval"]
    sb = u2["schur_gate_benchmark"]
    lines2 = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\caption{Real-Video Public Validation on All Three Vicon-Referenced Laval Actions ($701$ Post-Initial $1080\mathrm{p}$ RGB-D Frames with Saved SAM~2 Masks): Top = Frozen-Threshold Unified-Branch $2\times 2$ Factorial ($\text{Gate }G+R \times \text{Birth-Indexed \texttt{RevisableSurfelMap} }S$, $1{,}500$-pt budget); Bottom = Composed Native EvidenceScene and Scanned-Mesh FoundationPose Comparison.}\label{tab:laval_real_unified}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llrrrrrrr@{}}\toprule",
        r"\textbf{Track / Sequence} & \textbf{Configuration / System} & \textbf{Avail.} & \textbf{$\le 10$ / $\le 20$ mm} & \textbf{ADD AUC} [95\% CI] & \textbf{Ret.\ AUC} & \textbf{$P_{10}/C_{10}$ (\%)} & \textbf{$F_{95}/R_{95}$ (mm)} & \textbf{Notes / Latency} \\ \midrule",
    ]
    arm_labels = (
        ("gate_off_s_off", "Gate OFF, $S$ OFF"),
        ("gate_off_s_on", "Gate OFF, $S$ ON"),
        ("gate_on_s_off", "Gate ON, $S$ OFF"),
        ("gate_on_s_on", "\\textbf{Gate ON, $S$ ON}"),
    )
    for action, label in (("cookiejar", "Cookiejar (295)"), ("clock-hard", "Clock-hard (202)"), ("clock-full", "Clock-full (204)")):
        for ak, alabel in arm_labels:
            arm = u2["per_action"][action][ak]
            po = arm["pose"]
            sf = arm["terminal_surface"]
            ci = po["committed_add_auc_ci95"]
            note = f"$\\Delta$-rev {arm.get('delta_revision_median_ms', 0.0):.2f} ms" if "s_on" in ak else "--"
            lines2.append(
                f"{label} & {alabel} & {po['committed_avail']}/{po['denominator']} & "
                f"{po['add_le_10mm']} / {po['add_le_20mm']} & "
                f"{po['committed_add_auc']:.4f} [{ci[0]:.3f},{ci[1]:.3f}] & "
                f"{po['retained_add_auc']:.4f} & "
                f"{sf['P10_pct']:.2f} / {sf['C10_pct']:.2f} & "
                f"{sf['F95_mm']:.2f} / {sf['R95_mm']:.2f} & {note} \\\\"
            )
        lines2.append(r"\midrule")

    lines2.append(
        r"\multicolumn{9}{@{}l@{}}{\textbf{Cross-Method Reference on Laval (Composed Native Model-Free EvidenceScene vs.\ Scanned-Mesh FoundationPose [CAD*])}} \\"
    )
    lines2.append(
        r"\textbf{Sequence} ($n$) & \textbf{System} & \textbf{Avail.} & \textbf{ADD $\le 10$ / Strict} & \textbf{ADD AUC} [95\% CI] & \textbf{ADD-S AUC} & \textbf{$P_{10}/C_{10}$ (\%)} & \textbf{$F_{95}/R_{95}$ (mm)} & \textbf{Lat.\ $p_{50}$} \\ \midrule"
    )
    for action, label in (("cookiejar", "Cookiejar (295)"), ("clock-hard", "Clock-hard (202)"), ("clock-full", "Clock-full (204)")):
        for mkey, mlabel in (
            ("evidencescene_composed_native", "\\textbf{EvidenceScene} (Composed Native)"),
            ("foundationpose_mesh_f0_aligned", "FoundationPose (Scanned Mesh*)"),
        ):
            m = comp[action][mkey]
            po = m["pose"]
            sf = m["surface"]
            ci = po["add_auc_ci95"]
            lat = m.get("latency") or {}
            lat_txt = f"{lat.get('p50_ms', 0.0):.1f} ms" if lat else "--"
            lines2.append(
                f"{label} & {mlabel} & {po['avail']}/{po['n_post_init']} & "
                f"{po['add_le_10mm']} / {po['strict_10mm_10deg']} & "
                f"{po['add_auc']:.4f} [{ci[0]:.3f},{ci[1]:.3f}] & "
                f"{po['adds_auc']:.4f} & "
                f"{sf['P10_pct']:.2f} / {sf['C10_pct']:.2f} & "
                f"{sf['F95_mm']:.2f} / {sf['R95_mm']:.2f} & {lat_txt} \\\\"
            )
    lines2.append(r"\midrule")
    lines2.append(
        f"\\multicolumn{{9}}{{@{{}}l@{{}}}}{{\\emph{{C11 Schur Gate Benchmark across {sb['connected_noninitial_updates']} real Laval GTSAM updates:}} "
        f"median ${sb['c11_median_us']:.1f}\\,\\mu\\mathrm{{s}}$ vs.\\ ${sb['numpy_median_us']:.1f}\\,\\mu\\mathrm{{s}}$ NumPy SVD "
        f"(\\textbf{{{sb['speedup_vs_numpy_x']:.1f}$\\times$}}, {sb['parity_mismatches']} rank/admission mismatches).}} \\\\"
    )
    lines2.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    (papers_dir / "laval-real-unified-2x2-table.tex").write_text("\n".join(lines2) + "\n", encoding="utf-8")


def main() -> int:
    paths = default_paths()
    ycb_root = paths["ycb_root"]
    prior_root = paths["prior_ycb_root"]
    cache_root = paths["run_cache"]

    # 1. Score all 9/9 YCBInEOAT sequences
    ycb_rows: list[dict[str, Any]] = []
    pooled_ref_c = {"p2p": ([], []), "es": ([], []), "fp": ([], [])}
    pooled_gen_c = {"p2p": ([], []), "es": ([], [])}

    for spec in YCBINEOAT_SPECS:
        seq_id = spec.seq_id
        seq_dir = ycb_root / seq_id
        stems = load_ycb_sequence_stems(seq_dir)
        gt_poses = load_ycb_gt_poses(seq_dir, stems)
        mesh_path = ycb_root / "models" / spec.mesh_relpath
        verts = load_obj_vertices(mesh_path)
        samples = sample_mesh_surface_points(mesh_path, num_samples=100_000, seed=17)

        # Reference mask sources
        p2p_ref_path = (
            (prior_root / spec.prior_p2p_ref)
            if spec.prior_p2p_ref
            else (cache_root / "point2pose" / "ycb-ref" / seq_id / "results.json")
        )
        es_ref_dir = cache_root / "composed-native" / "ycb-reference" / seq_id
        fp_path = (
            (prior_root / spec.prior_fp_cad)
            if spec.prior_fp_cad
            else (cache_root / "foundationpose" / "ycb" / seq_id / "results.json")
        )

        p2p_ref_poses = _extract_p2p_poses(p2p_ref_path)
        es_ref_poses = _extract_bt_poses(es_ref_dir / "results.json")
        fp_poses = _extract_fp_poses(fp_path)

        p_ref_pose, p_ref_c_add, p_ref_c_adds = _score_pose_and_ci(p2p_ref_poses, gt_poses, verts, align_frame_zero=True)
        e_ref_pose, e_ref_c_add, e_ref_c_adds = _score_pose_and_ci(es_ref_poses, gt_poses, verts, align_frame_zero=True)
        f_ref_pose_aln, f_ref_c_add, f_ref_c_adds = _score_pose_and_ci(fp_poses, gt_poses, verts, align_frame_zero=True)
        f_ref_pose_raw, _, _ = _score_pose_and_ci(fp_poses, gt_poses, verts, align_frame_zero=False)

        # Verify reverse ref->est ADD-S parity against frozen October 2026 receipts
        hist_verified = False
        if seq_id in HISTORICAL_P2P_REF:
            exp_cnt, exp_auc = HISTORICAL_P2P_REF[seq_id]
            hist_verified = _verify_historical_gt2est_receipt(
                seq_id, p2p_ref_poses, gt_poses, verts, exp_cnt, exp_auc
            )

        # Generated SAM 2 mask sources
        if spec.prior_p2p_gen and (prior_root / spec.prior_p2p_gen / "results.json").is_file():
            p2p_gen_dir = prior_root / spec.prior_p2p_gen
        else:
            p2p_gen_dir = cache_root / "point2pose" / "ycb-gen" / seq_id
        es_gen_dir = cache_root / "composed-native" / "ycb-generated" / seq_id

        p2p_gen_poses = _extract_p2p_poses(p2p_gen_dir / "results.json")
        es_gen_poses = _extract_bt_poses(es_gen_dir / "results.json")

        p_gen_pose, p_gen_c_add, p_gen_c_adds = _score_pose_and_ci(p2p_gen_poses, gt_poses, verts, align_frame_zero=True)
        e_gen_pose, e_gen_c_add, e_gen_c_adds = _score_pose_and_ci(es_gen_poses, gt_poses, verts, align_frame_zero=True)

        pooled_ref_c["p2p"][0].append(p_ref_c_add)
        pooled_ref_c["p2p"][1].append(p_ref_c_adds)
        pooled_ref_c["es"][0].append(e_ref_c_add)
        pooled_ref_c["es"][1].append(e_ref_c_adds)
        pooled_ref_c["fp"][0].append(f_ref_c_add)
        pooled_ref_c["fp"][1].append(f_ref_c_adds)

        pooled_gen_c["p2p"][0].append(p_gen_c_add)
        pooled_gen_c["p2p"][1].append(p_gen_c_adds)
        pooled_gen_c["es"][0].append(e_gen_c_add)
        pooled_gen_c["es"][1].append(e_gen_c_adds)

        # 3D Surface evaluation (strided every 5th valid frame, support>=2, 1500-pt budget)
        surf_p_ref = _reconstruct_and_score_surface_ycb(
            seq_dir=seq_dir, stems=stems, est_poses=p2p_ref_poses, gt0=gt_poses[0],
            model_samples_obj=samples, mask_mode="reference", gen_masks_dir=None,
        )
        surf_e_ref = _reconstruct_and_score_surface_ycb(
            seq_dir=seq_dir, stems=stems, est_poses=es_ref_poses, gt0=gt_poses[0],
            model_samples_obj=samples, mask_mode="reference", gen_masks_dir=None,
        )
        surf_f_ref = _reconstruct_and_score_surface_ycb(
            seq_dir=seq_dir, stems=stems, est_poses=fp_poses, gt0=gt_poses[0],
            model_samples_obj=samples, mask_mode="reference", gen_masks_dir=None,
        )
        gen_masks_dir = p2p_gen_dir / "masks"
        surf_p_gen = _reconstruct_and_score_surface_ycb(
            seq_dir=seq_dir, stems=stems, est_poses=p2p_gen_poses, gt0=gt_poses[0],
            model_samples_obj=samples, mask_mode="generated", gen_masks_dir=gen_masks_dir,
        )
        surf_e_gen = _reconstruct_and_score_surface_ycb(
            seq_dir=seq_dir, stems=stems, est_poses=es_gen_poses, gt0=gt_poses[0],
            model_samples_obj=samples, mask_mode="generated", gen_masks_dir=gen_masks_dir,
        )

        # Paired sequence-level delta CIs
        d_ref_adds = compute_paired_block_bootstrap_diff_ci(e_ref_c_adds, p_ref_c_adds, block_length=20, num_resamples=2000)
        d_gen_adds = compute_paired_block_bootstrap_diff_ci(e_gen_c_adds, p_gen_c_adds, block_length=20, num_resamples=2000)

        fp_sum_path = cache_root / "foundationpose" / "ycb" / seq_id / "summary.json"
        p2p_ref_sum_path = cache_root / "point2pose" / "ycb-ref" / seq_id / "summary.json"
        ycb_rows.append(
            {
                "seq_id": seq_id,
                "short_label": spec.short_label,
                "ycb_object": spec.ycb_object,
                "total_frames": len(stems),
                "n_post_init": len(stems) - 1,
                "historical_gt2est_receipt_verified": hist_verified,
                "reference_mask": {
                    "point2pose": {"pose": p_ref_pose, "surface": surf_p_ref, "latency": _latency_from_summary(p2p_ref_sum_path)},
                    "evidencescene": {"pose": e_ref_pose, "surface": surf_e_ref, "latency": _latency_from_summary(es_ref_dir / "summary.json")},
                    "foundationpose_cad_f0_aligned": {"pose": f_ref_pose_aln, "surface": surf_f_ref, "latency": _latency_from_summary(fp_sum_path)},
                    "foundationpose_cad_no_align": {"pose": f_ref_pose_raw},
                    "paired_delta_adds_auc": d_ref_adds.as_dict(),
                },
                "generated_sam2_mask": {
                    "point2pose": {"pose": p_gen_pose, "surface": surf_p_gen},
                    "evidencescene": {"pose": e_gen_pose, "surface": surf_e_gen, "latency": _latency_from_summary(es_gen_dir / "summary.json")},
                    "paired_delta_adds_auc": d_gen_adds.as_dict(),
                },
            }
        )
        print(
            f"[Score:YCB:{spec.short_label}] "
            f"Ref ADDS(P2P={p_ref_pose['adds_auc']:.3f}, ES={e_ref_pose['adds_auc']:.3f}, FP={f_ref_pose_aln['adds_auc']:.3f}) | "
            f"Gen ADDS(P2P={p_gen_pose['adds_auc']:.3f}, ES={e_gen_pose['adds_auc']:.3f})",
            flush=True,
        )

    def _agg_ycb_arm(rows, track: str, model: str, c_pair: tuple[list[np.ndarray], list[np.ndarray]]) -> dict[str, Any]:
        c_add = np.concatenate(c_pair[0])
        c_adds = np.concatenate(c_pair[1])
        ci_add = compute_moving_block_bootstrap_ci(c_add, block_length=20, num_resamples=2000, seed=20261121)
        ci_adds = compute_moving_block_bootstrap_ci(c_adds, block_length=20, num_resamples=2000, seed=20261121)
        sub_lats = [
            r[track][model]["latency"]["p50_ms"]
            for r in rows
            if r[track][model].get("latency") and r[track][model]["latency"].get("p50_ms")
        ]
        return {
            "pooled_add_auc": round(float(np.mean(c_add)), 4),
            "pooled_add_auc_ci95": [round(ci_add.ci_low, 4), round(ci_add.ci_high, 4)],
            "pooled_adds_auc": round(float(np.mean(c_adds)), 4),
            "pooled_adds_auc_ci95": [round(ci_adds.ci_low, 4), round(ci_adds.ci_high, 4)],
            "total_adds_le_10mm": sum(r[track][model]["pose"]["adds_le_10mm"] for r in rows),
            "total_add_le_10mm": sum(r[track][model]["pose"]["add_le_10mm"] for r in rows),
            "mean_P10_pct": round(float(np.mean([r[track][model]["surface"]["P10_pct"] for r in rows])), 2),
            "mean_C10_pct": round(float(np.mean([r[track][model]["surface"]["C10_pct"] for r in rows])), 2),
            "mean_F95_mm": round(float(np.mean([r[track][model]["surface"]["F95_mm"] for r in rows])), 2),
            "mean_R95_mm": round(float(np.mean([r[track][model]["surface"]["R95_mm"] for r in rows])), 2),
            "median_sequence_p50_ms": round(float(np.median(sub_lats)), 2) if sub_lats else None,
        }

    d_pool_ref_adds = compute_paired_block_bootstrap_diff_ci(
        np.concatenate(pooled_ref_c["es"][1]), np.concatenate(pooled_ref_c["p2p"][1]), block_length=20, num_resamples=2000
    )
    d_pool_ref_add = compute_paired_block_bootstrap_diff_ci(
        np.concatenate(pooled_ref_c["es"][0]), np.concatenate(pooled_ref_c["p2p"][0]), block_length=20, num_resamples=2000
    )
    d_pool_gen_adds = compute_paired_block_bootstrap_diff_ci(
        np.concatenate(pooled_gen_c["es"][1]), np.concatenate(pooled_gen_c["p2p"][1]), block_length=20, num_resamples=2000
    )
    d_pool_gen_add = compute_paired_block_bootstrap_diff_ci(
        np.concatenate(pooled_gen_c["es"][0]), np.concatenate(pooled_gen_c["p2p"][0]), block_length=20, num_resamples=2000
    )

    ycb_payload = {
        "schema": "evidencescene.ycbineoat-9seq-full.v1",
        "adds_direction": "canonical_est_to_ref",
        "sequence_count": len(ycb_rows),
        "total_frames": sum(r["total_frames"] for r in ycb_rows),
        "total_post_init_frames": sum(r["n_post_init"] for r in ycb_rows),
        "bootstrap_config": {"method": "circular_moving_block", "block_length": 20, "num_resamples": 2000, "seed": 20261121},
        "surface_protocol": {
            "voxel_size_m": 0.004,
            "stride_frames": 5,
            "minimum_observation_support": 2,
            "selector": "support10/coarse12mm",
            "budget": 1500,
            "is_prior": False,
        },
        "aggregate": {
            "reference_mask": {
                "point2pose": _agg_ycb_arm(ycb_rows, "reference_mask", "point2pose", pooled_ref_c["p2p"]),
                "evidencescene": _agg_ycb_arm(ycb_rows, "reference_mask", "evidencescene", pooled_ref_c["es"]),
                "foundationpose_cad_f0_aligned": _agg_ycb_arm(ycb_rows, "reference_mask", "foundationpose_cad_f0_aligned", pooled_ref_c["fp"]),
                "paired_delta_es_minus_p2p": {
                    "adds_auc_diff": round(d_pool_ref_adds.estimate, 4),
                    "adds_auc_ci95": [round(d_pool_ref_adds.ci_low, 4), round(d_pool_ref_adds.ci_high, 4)],
                    "add_auc_diff": round(d_pool_ref_add.estimate, 4),
                    "add_auc_ci95": [round(d_pool_ref_add.ci_low, 4), round(d_pool_ref_add.ci_high, 4)],
                },
            },
            "generated_sam2_mask": {
                "point2pose": _agg_ycb_arm(ycb_rows, "generated_sam2_mask", "point2pose", pooled_gen_c["p2p"]),
                "evidencescene": _agg_ycb_arm(ycb_rows, "generated_sam2_mask", "evidencescene", pooled_gen_c["es"]),
                "paired_delta_es_minus_p2p": {
                    "adds_auc_diff": round(d_pool_gen_adds.estimate, 4),
                    "adds_auc_ci95": [round(d_pool_gen_adds.ci_low, 4), round(d_pool_gen_adds.ci_high, 4)],
                    "add_auc_diff": round(d_pool_gen_add.estimate, 4),
                    "add_auc_ci95": [round(d_pool_gen_add.ci_low, 4), round(d_pool_gen_add.ci_high, 4)],
                },
            },
        },
        "sequences": ycb_rows,
    }

    # 2. Score Laval 3/3 sequences across Composed Native & FoundationPose + Real-Video 2x2
    laval_u2_path = cache_root / "laval_unified_2x2_raw.json"
    laval_u2 = json.loads(laval_u2_path.read_text(encoding="utf-8"))
    cross_laval: dict[str, Any] = {}
    for action in LAVAL_ACTION_IDS:
        frames, intrinsics, gt_poses, model_path = load_laval_frames_and_gt(paths["laval_root"] / action)
        samples = sample_mesh_surface_points(model_path, num_samples=100_000, seed=17)

        es_dir = cache_root / "composed-native" / "laval-sam2" / action
        fp_dir = cache_root / "foundationpose" / "laval" / action

        es_poses = _extract_bt_poses(es_dir / "results.json")
        fp_poses = _extract_fp_poses(fp_dir / "results.json")

        es_pose_sc, _, _ = _score_pose_and_ci(es_poses, gt_poses, samples, align_frame_zero=True)
        fp_pose_sc, _, _ = _score_pose_and_ci(fp_poses, gt_poses, samples, align_frame_zero=True)

        es_surf = _reconstruct_and_score_surface_laval(
            frames=frames, intrinsics=intrinsics, est_poses=es_poses, gt0=gt_poses[0], model_samples_obj=samples
        )
        fp_surf = _reconstruct_and_score_surface_laval(
            frames=frames, intrinsics=intrinsics, est_poses=fp_poses, gt0=gt_poses[0], model_samples_obj=samples
        )

        cross_laval[action] = {
            "evidencescene_composed_native": {
                "pose": es_pose_sc,
                "surface": es_surf,
                "latency": _latency_from_summary(es_dir / "summary.json"),
            },
            "foundationpose_mesh_f0_aligned": {
                "pose": fp_pose_sc,
                "surface": fp_surf,
                "latency": _latency_from_summary(fp_dir / "summary.json"),
            },
        }
        print(
            f"[Score:Laval:{action}] ES ADD={es_pose_sc['add_auc']:.4f} (P10={es_surf['P10_pct']:.1f}%) | "
            f"FP ADD={fp_pose_sc['add_auc']:.4f} (P10={fp_surf['P10_pct']:.1f}%)",
            flush=True,
        )

    laval_payload = {
        "schema": "evidencescene.laval-competitors-and-real-2x2.v1",
        "unified_2x2_real": laval_u2,
        "cross_method_laval": cross_laval,
    }

    res_dir = HERE / "results"
    write_json(res_dir / "ycbineoat_9seq_full_scorecard.json", ycb_payload)
    write_json(res_dir / "laval_3seq_competitors_and_unified_2x2.json", laval_payload)
    _write_latex_tables(ycb_payload, laval_payload, REPO_ROOT / "papers" / "evidencescene")
    print("SCORING AND LATEX FRAGMENT GENERATION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
