#!/usr/bin/env python3
"""Run Real-Video Unified-Branch 2x2 Factorial (Gate G+R x RevisableSurfelMap S) and 71-update C11 Schur bench on Laval."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import evidencescene.observability.factor_graph_observability as fgo_mod  # noqa: E402
from evidencescene.metrics import (  # noqa: E402
    compute_moving_block_bootstrap_ci,
    evaluate_pose_sequence,
    evaluate_surface_reconstruction,
    per_frame_auc_contributions,
)
from evidencescene.observability import evaluate_schur_observability_gate  # noqa: E402
from evidencescene.primitives import backproject_masked_depth_points  # noqa: E402
from evidencescene.surface import (  # noqa: E402
    RevisableSurfelMap,
    select_existing_points_shadow,
)
from suite_common import (  # noqa: E402
    LAVAL_ACTION_IDS,
    default_paths,
    erode_mask_3x3,
    load_laval_frames_and_gt,
    polar_so3,
    sample_mesh_surface_points,
    write_json,
)

EXPECTED_P2P_COMMIT = "51856226610df75e5c06e8de545bd27f7c4ba99c"


def _invert_se3(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    r_t = pose[:3, :3].T
    inv[:3, :3] = r_t
    inv[:3, 3] = -r_t @ pose[:3, 3]
    return inv


def _extract_gtsam_observability(
    graph: Any,
    values: Any,
    candidate_frame_id: int,
) -> dict[str, Any]:
    """Extract C0 prior-connected component and unreduced whitened (Ax, Ay) Jacobians."""
    import gtsam

    adjacency: dict[int, set[int]] = defaultdict(set)
    roots: list[int] = []
    factors = [graph.at(i) for i in range(graph.size())]
    for factor in factors:
        keys = [int(k) for k in factor.keys()]
        if type(factor).__name__ == "PriorFactorPose3":
            roots.extend(keys)
        for k in keys:
            adjacency[k].update(keys)

    if len(roots) != 1:
        return {
            "connected": False,
            "is_root": False,
            "anchored_landmarks": 0,
            "ax": None,
            "ay": None,
            "cand_slot": -1,
            "component_pose_count": 0,
        }

    root = roots[0]
    cand_key = int(gtsam.symbol("x", int(candidate_frame_id)))
    if cand_key == root:
        return {
            "connected": True,
            "is_root": True,
            "anchored_landmarks": 0,
            "ax": None,
            "ay": None,
            "cand_slot": -1,
            "component_pose_count": 1,
        }

    comp = {root}
    queue = [root]
    while queue:
        cur = queue.pop()
        for nxt in adjacency[cur] - comp:
            comp.add(nxt)
            queue.append(nxt)

    if cand_key not in comp:
        return {
            "connected": False,
            "is_root": False,
            "anchored_landmarks": 0,
            "ax": None,
            "ay": None,
            "cand_slot": -1,
            "component_pose_count": sum(1 for k in comp if chr(k >> 56) == "x"),
        }

    pose_keys = sorted(k for k in comp if chr(k >> 56) == "x" and k != root)
    lm_keys = sorted(k for k in comp if chr(k >> 56) == "l")
    pose_offsets = {k: 6 * i for i, k in enumerate(pose_keys)}
    lm_offsets = {k: 3 * i for i, k in enumerate(lm_keys)}
    cand_slot = pose_keys.index(cand_key)

    anchored_lms: set[int] = set()
    rows_ax: list[np.ndarray] = []
    rows_ay: list[np.ndarray] = []

    for factor in factors:
        fkeys = [int(k) for k in factor.keys()]
        if not set(fkeys).issubset(comp):
            continue
        if type(factor).__name__ == "PriorFactorPose3":
            continue
        if cand_key in fkeys:
            for fk in fkeys:
                if chr(fk >> 56) == "l":
                    anchored_lms.add(fk)
        linear = factor.linearize(values)
        lkeys = [int(k) for k in linear.keys()]
        container = gtsam.GaussianFactorGraph()
        container.push_back(linear)
        a_f, _ = container.jacobian(gtsam.Ordering(lkeys))
        ax_f = np.zeros((a_f.shape[0], 6 * len(pose_keys)), dtype=np.float64)
        ay_f = np.zeros((a_f.shape[0], 3 * len(lm_keys)), dtype=np.float64)
        off = 0
        for k in lkeys:
            w = 6 if chr(k >> 56) == "x" else 3
            blk = a_f[:, off : off + w]
            off += w
            if k == root:
                continue
            if k in pose_offsets:
                c0 = pose_offsets[k]
                ax_f[:, c0 : c0 + 6] = blk
            elif k in lm_offsets:
                c0 = lm_offsets[k]
                ay_f[:, c0 : c0 + 3] = blk
        rows_ax.append(ax_f)
        rows_ay.append(ay_f)

    ax = np.vstack(rows_ax) if rows_ax else np.empty((0, 6 * len(pose_keys)), dtype=np.float64)
    ay = np.vstack(rows_ay) if rows_ay else np.empty((0, 3 * len(lm_keys)), dtype=np.float64)
    return {
        "connected": True,
        "is_root": False,
        "anchored_landmarks": len(anchored_lms),
        "ax": ax,
        "ay": ay,
        "cand_slot": cand_slot,
        "component_pose_count": len(pose_keys) + 1,
    }


def _snapshot_surface_metrics(
    surfel_map: RevisableSurfelMap,
    model_points_m: np.ndarray,
    *,
    budget: int = 1500,
) -> dict[str, Any]:
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
            "full_P10_pct": 0.0,
            "full_C10_pct": 0.0,
            "supported_surfels": 0,
            "selected_points": 0,
        }
    sel = select_existing_points_shadow(full_xyz, full_sup, full_cams, full_keys, budget=budget)
    pts_budget = full_xyz[sel["indices"]]
    sc_budget = evaluate_surface_reconstruction(
        pts_budget, model_points_m, threshold_m=0.010, grid_voxel_size_m=0.004
    )
    sc_full = evaluate_surface_reconstruction(
        full_xyz, model_points_m, threshold_m=0.010, grid_voxel_size_m=0.004
    )
    return {
        "P10_pct": round(sc_budget.precision_pct, 2),
        "C10_pct": round(sc_budget.coverage_pct, 2),
        "F1_10_pct": round(sc_budget.f1_pct, 2),
        "F95_mm": round(sc_budget.forward_p95_mm, 2),
        "R95_mm": round(sc_budget.reverse_p95_mm, 2),
        "full_P10_pct": round(sc_full.precision_pct, 2),
        "full_C10_pct": round(sc_full.coverage_pct, 2),
        "full_F95_mm": round(sc_full.forward_p95_mm, 2),
        "supported_surfels": int(len(full_xyz)),
        "selected_points": int(len(pts_budget)),
    }


def _run_action_with_gate(
    *,
    action: str,
    enable_gate: bool,
    paths: dict[str, Path],
    captured_jacobians: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run real Point2Pose (F1+F2) with Gate OFF or ON and parallel S OFF / S ON RevisableSurfelMap."""
    p2p_root = paths["point2pose_root"]
    if str(p2p_root) not in sys.path:
        sys.path.insert(0, str(p2p_root))
    import torch
    from omegaconf import OmegaConf
    from point2pose.data_types.frame import Frame
    from point2pose.pipeline.modular_pipeline import ModularPipeline

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    dbg_dir = paths["run_cache"] / "laval-2x2-debug" / f"{action}-gate-{int(enable_gate)}"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(p2p_root / "configs/ycbinisaac/eccv_final.yaml")
    cfg.pipeline.params.max_num_obj = 1
    cfg.pipeline.params.use_segmenter = False
    cfg.pipeline.params.debug_dir = str(dbg_dir)
    cfg.pipeline.params.pose_save_path = str(dbg_dir / "poses")
    cfg.pipeline.params.meta_data_save_path = str(dbg_dir / "meta")
    cfg.pipeline.params.save_pose = False
    cfg.pipeline.params.save_meta_data = False
    cfg.pipeline.params.save_key_points = False
    cfg.pipeline.params.save_cropped_pcd = False
    cfg.pipeline.params.sdf_mesh_save_every = 0
    cfg.reconstructor.params.sdf_use_gpu = False
    cfg.reconstructor.params.sdf_fuse_color = False
    cfg.tracker.params.checkpoint_path = str(paths["bootstapir_checkpoint"])
    cfg.tracker.params.device = "cuda"
    if hasattr(cfg, "visualization") and hasattr(cfg.visualization, "params"):
        cfg.visualization.params.save_images = False
        cfg.visualization.params.visualize_points = False
        cfg.visualization.params.output_image_dir = str(dbg_dir / "vis")

    frames, intrinsics, gt_poses, model_path = load_laval_frames_and_gt(paths["laval_root"] / action)
    model_samples = sample_mesh_surface_points(model_path, num_samples=100_000, seed=17)
    # Reference mesh in Frame-0 camera coordinates (matching Frame-0 Vicon pose)
    gt0 = gt_poses[0]
    model_pts_frame0 = model_samples @ gt0[:3, :3].T + gt0[:3, 3]

    pipe = ModularPipeline(cfg)

    # F2 repair: make SVDClusterRANSACRegister return refined_T when applied and not reverted
    orig_register = pipe.frontend.register.register

    def fixed_register(*args, **kwargs):
        t_ret, stats = orig_register(*args, **kwargs)
        sdf_info = stats.get("sdf_refine") or {}
        gate_info = stats.get("final_inlier_gate") or {}
        if sdf_info.get("applied") and not gate_info.get("applied"):
            best_idx = int(stats["best_cluster_idx"])
            t_ret = np.asarray(stats["clusters"][best_idx]["T"], dtype=np.float64).copy()
        return t_ret, stats

    pipe.frontend.register.register = fixed_register

    # Dual RevisableSurfelMap instances isolating S OFF vs S ON on identical gate stream
    map_s_off = RevisableSurfelMap(
        voxel_size_m=0.004, maximum_voxels=250_000, maximum_live_bytes=128 * 1024 * 1024
    )
    map_s_on = RevisableSurfelMap(
        voxel_size_m=0.004, maximum_voxels=250_000, maximum_live_bytes=128 * 1024 * 1024
    )

    state: dict[str, Any] = {
        "frame_id": -1,
        "eroded_mask": None,
        "depth_m": None,
        "pre_obj_pose": None,
        "rejected_current_kf": False,
        "raw_kf_obs": {},  # kf_idx -> (frame_id, pts_cam)
        "last_integrated_poses": {},  # kf_idx -> 4x4 camera_from_object
        "delta_revision_times_ms": [],
        "gate_stats": {"updates": 0, "admitted": 0, "rejected_disconnected": 0, "rejected_rank_or_schur": 0},
        "mid_snapshot": {"s_off": None, "s_on": None},
    }

    orig_update = pipe.kf_graph.update
    orig_integrate = pipe.sdf_builder.integrate_keyframe

    def hooked_update(keyframes):
        import gtsam

        if not keyframes:
            return orig_update(keyframes)
        obj = pipe.objects[0]
        opt = pipe.kf_graph._get_optimizer(obj.id)
        snap_graph = gtsam.NonlinearFactorGraph(opt._graph)
        snap_values = gtsam.Values(opt._values)
        snap_ins_poses = set(opt._inserted_poses)
        snap_ins_lms = set(opt._inserted_landmarks)
        snap_ins_lm_ids = list(opt.inserted_landmark_ids)
        snap_init = bool(opt._initialized)
        snap_prev_pose_inv = opt._prev_pose_inv.copy()
        snap_prev_fid = int(opt._prev_frame_id)
        snap_xo = opt.Xo
        snap_last_kf_pose = {k: v.copy() for k, v in pipe.kf_graph._last_kf_pose.items()}
        snap_last_kf_idx = dict(pipe.kf_graph._last_kf_idx)
        snap_kfs_by_frame = {oid: dict(fmap) for oid, fmap in pipe.kf_graph._keyframes_by_obj_frame.items()}

        state["pre_obj_pose"] = obj.pose.copy()
        state["rejected_current_kf"] = False
        # Snapshot existing keyframe poses before optimization
        pre_kf_poses = {
            int(k): np.asarray(kf_obj.pose, dtype=np.float64).copy()
            for k, kf_obj in pipe.kf_graph._keyframes_by_obj_frame.get(obj.id, {}).items()
        }
        pre_new_kf_poses = {id(k): np.asarray(k.pose, dtype=np.float64).copy() for k in keyframes}
        updated_poses, updated_landmarks = orig_update(keyframes)
        state["gate_stats"]["updates"] += len(keyframes)

        for kf in keyframes:
            opt = pipe.kf_graph._get_optimizer(kf.obj_id)
            obs = _extract_gtsam_observability(opt._graph, opt._values, int(kf.frame_id))
            if obs["is_root"]:
                admitted = True
                reason = "root_prior"
            elif not obs["connected"]:
                admitted = False
                reason = "disconnected_from_prior"
            else:
                gate_rep = evaluate_schur_observability_gate(
                    pose_jacobian_ax=obs["ax"],
                    landmark_jacobian_ay=obs["ay"],
                    candidate_pose_index=obs["cand_slot"],
                    required_pose_rank=6,
                    tolerance_scales=(0.1, 1.0, 10.0),
                    minimum_translational_singular_value=1e-4,
                    maximum_translational_condition_number=1e6,
                )
                admitted = bool(gate_rep.admitted) and (obs["anchored_landmarks"] >= 3)
                reason = (
                    gate_rep.reason
                    if gate_rep.admitted and obs["anchored_landmarks"] >= 3
                    else ("insufficient_anchored_landmarks" if obs["anchored_landmarks"] < 3 else gate_rep.reason)
                )
                if not enable_gate:
                    captured_jacobians.append(
                        {
                            "action": action,
                            "frame_id": int(kf.frame_id),
                            "ax": obs["ax"],
                            "ay": obs["ay"],
                            "cand_slot": int(obs["cand_slot"]),
                            "expected_admitted": bool(gate_rep.admitted),
                            "expected_rank": int(gate_rep.marginal_pose_rank),
                        }
                    )

            if admitted:
                state["gate_stats"]["admitted"] += 1
            elif reason == "disconnected_from_prior":
                state["gate_stats"]["rejected_disconnected"] += 1
            else:
                state["gate_stats"]["rejected_rank_or_schur"] += 1

            if enable_gate and not admitted:
                state["rejected_current_kf"] = True
                opt._graph = snap_graph
                opt._values = snap_values
                opt._inserted_poses = snap_ins_poses
                opt._inserted_landmarks = snap_ins_lms
                opt.inserted_landmark_ids = snap_ins_lm_ids
                opt._initialized = snap_init
                opt._prev_pose_inv = snap_prev_pose_inv
                opt._prev_frame_id = snap_prev_fid
                opt.Xo = snap_xo
                pipe.kf_graph._last_kf_pose = snap_last_kf_pose
                pipe.kf_graph._last_kf_idx = snap_last_kf_idx
                pipe.kf_graph._keyframes_by_obj_frame = snap_kfs_by_frame
                for fid_prev, kf_ref in pipe.kf_graph._keyframes_by_obj_frame.get(obj.id, {}).items():
                    if fid_prev in pre_kf_poses:
                        kf_ref.pose = pre_kf_poses[fid_prev].copy()
                        kf_ref.update_kps()
                for k_new in keyframes:
                    if id(k_new) in pre_new_kf_poses:
                        k_new.pose = pre_new_kf_poses[id(k_new)].copy()
                        k_new.update_kps()
                return {}, {}

        # If S ON is running in parallel, apply any revised historical keyframe poses now
        revision_batch: dict[str, tuple[str, np.ndarray]] = {}
        if not (enable_gate and state["rejected_current_kf"]):
            cur_new_kfs = {int(k.kf_idx) for k in keyframes}
            for _fid, kf_ref in pipe.kf_graph._keyframes_by_obj_frame.get(obj.id, {}).items():
                kidx = int(kf_ref.kf_idx)
                if kidx in cur_new_kfs or kidx not in state["raw_kf_obs"]:
                    continue
                new_p = np.asarray(kf_ref.pose, dtype=np.float64)
                old_p = state["last_integrated_poses"].get(kidx)
                if old_p is not None and not np.allclose(old_p, new_p, rtol=0.0, atol=1e-12):
                    pts_cam = state["raw_kf_obs"][kidx]
                    t_oc = _invert_se3(new_p)
                    pts_obj = pts_cam @ t_oc[:3, :3].T + t_oc[:3, 3]
                    revision_batch[f"kf_{kidx:04d}"] = ("chest", pts_obj)
                    state["last_integrated_poses"][kidx] = new_p.copy()
            if revision_batch:
                t_rev0 = time.perf_counter()
                map_s_on.update(revision_batch)
                state["delta_revision_times_ms"].append((time.perf_counter() - t_rev0) * 1000.0)

        return updated_poses, updated_landmarks

    def hooked_integrate(obj, keyframe):
        # F1 repair: preserve obj.pose when promoted keyframe is from an earlier frame
        if int(keyframe.frame_id) != int(state["frame_id"]) and state["pre_obj_pose"] is not None:
            obj.pose = state["pre_obj_pose"].copy()

        if enable_gate and state["rejected_current_kf"]:
            return False

        sdf_ok = orig_integrate(obj=obj, keyframe=keyframe)

        # Integrate into RevisableSurfelMap unified surface branch using promoted keyframe's own RGB-D frame
        kidx = int(keyframe.kf_idx)
        kf_f = frames[int(keyframe.frame_id)]
        if kidx not in state["raw_kf_obs"] and kf_f["status"] == "available" and kf_f["mask"].any():
            kf_eroded = erode_mask_3x3(kf_f["mask"])
            if kf_eroded.any():
                pts_cam = backproject_masked_depth_points(
                    kf_f["depth_m"],
                    kf_eroded,
                    intrinsics,
                    minimum_depth_m=0.1,
                    maximum_depth_m=2.0,
                    pixel_stride=1,
                )
                if len(pts_cam) > 0:
                    pose_kf = np.asarray(keyframe.pose, dtype=np.float64).copy()
                    t_oc = _invert_se3(pose_kf)
                    pts_obj = pts_cam @ t_oc[:3, :3].T + t_oc[:3, 3]
                    kf_name = f"kf_{kidx:04d}"
                    state["raw_kf_obs"][kidx] = pts_cam
                    state["last_integrated_poses"][kidx] = pose_kf
                    map_s_off.update({kf_name: ("chest", pts_obj)})
                    map_s_on.update({kf_name: ("chest", pts_obj)})
        return sdf_ok

    pipe.kf_graph.update = hooked_update
    pipe.sdf_builder.integrate_keyframe = hooked_integrate

    committed_poses: list[np.ndarray | None] = []
    retained_poses: list[np.ndarray | None] = []
    last_verified_pose: np.ndarray | None = None

    for f in frames:
        idx = int(f["index"])
        state["frame_id"] = idx
        state["rejected_current_kf"] = False
        if f["status"] != "available" or not f["mask"].any():
            state["eroded_mask"] = None
            state["depth_m"] = None
            committed_poses.append(None)
            retained_poses.append(last_verified_pose)
        else:
            eroded = erode_mask_3x3(f["mask"])
            if not eroded.any():
                state["eroded_mask"] = None
                state["depth_m"] = None
                committed_poses.append(None)
                retained_poses.append(last_verified_pose)
            else:
                state["eroded_mask"] = eroded
                state["depth_m"] = f["depth_m"]
                frame = Frame(
                    id=idx,
                    rgb=f["rgb"],
                    depth=f["depth_m"],
                    mask=torch.from_numpy(eroded.astype(np.uint8))[None, None].to("cuda"),
                    intrinsics=intrinsics,
                    depth_factor=1.0,
                    timestamp=(idx + 1) / 1e9,
                )
                pose = pipe.step(frame)
                lost = bool(pipe.objects[0].lost)
                pose_mat = polar_so3(np.asarray(pose[0], dtype=np.float64)) if (pose is not None and not lost) else None
                if enable_gate and state["rejected_current_kf"]:
                    committed_poses.append(None)
                    retained_poses.append(last_verified_pose)
                elif pose_mat is not None:
                    committed_poses.append(pose_mat)
                    last_verified_pose = pose_mat
                    retained_poses.append(last_verified_pose)
                else:
                    committed_poses.append(None)
                    retained_poses.append(last_verified_pose)

        if action == "cookiejar" and idx == 150:
            state["mid_snapshot"]["s_off"] = _snapshot_surface_metrics(map_s_off, model_pts_frame0)
            state["mid_snapshot"]["s_on"] = _snapshot_surface_metrics(map_s_on, model_pts_frame0)

    comm_sc_10 = evaluate_pose_sequence(committed_poses, gt_poses, model_samples, threshold_m=0.010)
    comm_sc_20 = evaluate_pose_sequence(committed_poses, gt_poses, model_samples, threshold_m=0.020)
    ret_sc_10 = evaluate_pose_sequence(retained_poses, gt_poses, model_samples, threshold_m=0.010)

    c_add, c_adds = per_frame_auc_contributions(committed_poses, gt_poses, model_samples)
    r_add, r_adds = per_frame_auc_contributions(retained_poses, gt_poses, model_samples)
    ci_comm_add = compute_moving_block_bootstrap_ci(c_add, block_length=20, num_resamples=2000, seed=20261121)
    ci_ret_add = compute_moving_block_bootstrap_ci(r_add, block_length=20, num_resamples=2000, seed=20261121)

    surf_term_off = _snapshot_surface_metrics(map_s_off, model_pts_frame0)
    surf_term_on = _snapshot_surface_metrics(map_s_on, model_pts_frame0)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    pose_summary = {
        "denominator": comm_sc_10.denominator_excluding_initial,
        "committed_avail": comm_sc_10.available_post_initial_count,
        "retained_avail": ret_sc_10.available_post_initial_count,
        "adds_le_10mm": comm_sc_10.adds_le_threshold_count,
        "add_le_10mm": comm_sc_10.add_le_threshold_count,
        "add_le_20mm": comm_sc_20.add_le_threshold_count,
        "strict_10mm_10deg": comm_sc_10.strict_threshold_count,
        "committed_add_auc": round(comm_sc_10.add_auc_0_to_max, 4),
        "committed_add_auc_ci95": [round(ci_comm_add.ci_low, 4), round(ci_comm_add.ci_high, 4)],
        "retained_add_auc": round(ret_sc_10.add_auc_0_to_max, 4),
        "retained_add_auc_ci95": [round(ci_ret_add.ci_low, 4), round(ci_ret_add.ci_high, 4)],
        "committed_adds_auc": round(comm_sc_10.adds_auc_0_to_max, 4),
        "mean_add_mm": round((comm_sc_10.mean_add_m or 0.0) * 1000.0, 2),
    }

    return {
        "action": action,
        "enable_gate": enable_gate,
        "gate_stats": state["gate_stats"],
        "pose": pose_summary,
        "committed_poses": committed_poses,
        "s_off_terminal_surface": surf_term_off,
        "s_on_terminal_surface": surf_term_on,
        "cookiejar_f150_surface": state["mid_snapshot"] if action == "cookiejar" else None,
        "delta_revision_count": len(state["delta_revision_times_ms"]),
        "delta_revision_median_ms": round(float(np.median(state["delta_revision_times_ms"])), 3)
        if state["delta_revision_times_ms"]
        else 0.0,
    }


def _benchmark_captured_schur_updates(captured: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare C11 native vs pure-NumPy evaluate_schur_observability_gate across real Laval updates."""
    if not captured:
        return {"count": 0}
    orig_native = fgo_mod.NATIVE
    try:
        # Verify 100% parity
        mismatches = 0
        fast_count = 0
        for item in captured:
            fgo_mod.NATIVE = orig_native
            rep_nat = evaluate_schur_observability_gate(
                pose_jacobian_ax=item["ax"],
                landmark_jacobian_ay=item["ay"],
                candidate_pose_index=item["cand_slot"],
            )
            if orig_native is not None and item["ax"].shape[1] <= 192 and item["ax"].shape[0] >= item["ax"].shape[1]:
                fast_count += 1
            fgo_mod.NATIVE = None
            rep_py = evaluate_schur_observability_gate(
                pose_jacobian_ax=item["ax"],
                landmark_jacobian_ay=item["ay"],
                candidate_pose_index=item["cand_slot"],
            )
            if rep_nat.admitted != rep_py.admitted or rep_nat.marginal_pose_rank != rep_py.marginal_pose_rank:
                mismatches += 1

        # Benchmark C11 native path
        fgo_mod.NATIVE = orig_native
        nat_times_us: list[float] = []
        for _ in range(5):
            for item in captured:
                evaluate_schur_observability_gate(
                    pose_jacobian_ax=item["ax"],
                    landmark_jacobian_ay=item["ay"],
                    candidate_pose_index=item["cand_slot"],
                )
        for _ in range(25):
            for item in captured:
                t0 = time.perf_counter()
                evaluate_schur_observability_gate(
                    pose_jacobian_ax=item["ax"],
                    landmark_jacobian_ay=item["ay"],
                    candidate_pose_index=item["cand_slot"],
                )
                nat_times_us.append((time.perf_counter() - t0) * 1e6)

        # Benchmark Python/NumPy fallback
        fgo_mod.NATIVE = None
        py_times_us: list[float] = []
        for _ in range(10):
            for item in captured:
                t0 = time.perf_counter()
                evaluate_schur_observability_gate(
                    pose_jacobian_ax=item["ax"],
                    landmark_jacobian_ay=item["ay"],
                    candidate_pose_index=item["cand_slot"],
                )
                py_times_us.append((time.perf_counter() - t0) * 1e6)
    finally:
        fgo_mod.NATIVE = orig_native

    nat_med_us = float(np.median(nat_times_us))
    nat_mean_us = float(np.mean(nat_times_us))
    py_med_us = float(np.median(py_times_us))
    return {
        "connected_noninitial_updates": len(captured),
        "c11_fast_path_eligible_count": fast_count,
        "parity_mismatches": mismatches,
        "c11_median_us": round(nat_med_us, 2),
        "c11_mean_us": round(nat_mean_us, 2),
        "numpy_median_us": round(py_med_us, 2),
        "speedup_vs_numpy_x": round(py_med_us / max(nat_med_us, 1e-6), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", nargs="+", default=list(LAVAL_ACTION_IDS))
    args = parser.parse_args()

    paths = default_paths()
    p2p_root = paths["point2pose_root"]
    head = subprocess.check_output(["git", "-C", str(p2p_root), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_P2P_COMMIT:
        raise RuntimeError(f"Unexpected Point2Pose commit {head} != {EXPECTED_P2P_COMMIT}")

    captured_jacobians: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}

    for action in args.actions:
        print(f"[Laval-2x2] Running {action} (Gate OFF)...", flush=True)
        res_off = _run_action_with_gate(
            action=action, enable_gate=False, paths=paths, captured_jacobians=captured_jacobians
        )
        print(f"[Laval-2x2] Running {action} (Gate ON)...", flush=True)
        res_on = _run_action_with_gate(
            action=action, enable_gate=True, paths=paths, captured_jacobians=captured_jacobians
        )

        raw_results[action] = {
            "gate_off_s_off": {
                "pose": res_off["pose"],
                "terminal_surface": res_off["s_off_terminal_surface"],
                "cookiejar_f150_surface": res_off["cookiejar_f150_surface"]["s_off"]
                if res_off["cookiejar_f150_surface"]
                else None,
                "gate_stats": res_off["gate_stats"],
            },
            "gate_off_s_on": {
                "pose": res_off["pose"],
                "terminal_surface": res_off["s_on_terminal_surface"],
                "cookiejar_f150_surface": res_off["cookiejar_f150_surface"]["s_on"]
                if res_off["cookiejar_f150_surface"]
                else None,
                "gate_stats": res_off["gate_stats"],
                "delta_revision_count": res_off["delta_revision_count"],
                "delta_revision_median_ms": res_off["delta_revision_median_ms"],
            },
            "gate_on_s_off": {
                "pose": res_on["pose"],
                "terminal_surface": res_on["s_off_terminal_surface"],
                "cookiejar_f150_surface": res_on["cookiejar_f150_surface"]["s_off"]
                if res_on["cookiejar_f150_surface"]
                else None,
                "gate_stats": res_on["gate_stats"],
            },
            "gate_on_s_on": {
                "pose": res_on["pose"],
                "terminal_surface": res_on["s_on_terminal_surface"],
                "cookiejar_f150_surface": res_on["cookiejar_f150_surface"]["s_on"]
                if res_on["cookiejar_f150_surface"]
                else None,
                "gate_stats": res_on["gate_stats"],
                "delta_revision_count": res_on["delta_revision_count"],
                "delta_revision_median_ms": res_on["delta_revision_median_ms"],
            },
        }
        # Save per-frame gate OFF/ON poses to cache for downstream cross-method scoring
        pose_cache = paths["run_cache"] / "laval-2x2-poses" / action
        write_json(
            pose_cache / "poses.json",
            {
                "gate_off": [p.tolist() if p is not None else None for p in res_off["committed_poses"]],
                "gate_on": [p.tolist() if p is not None else None for p in res_on["committed_poses"]],
            },
        )

    schur_bench = _benchmark_captured_schur_updates(captured_jacobians)
    aggregates: dict[str, Any] = {}
    for arm in ("gate_off_s_off", "gate_off_s_on", "gate_on_s_off", "gate_on_s_on"):
        arms_list = [raw_results[a][arm] for a in args.actions]
        aggregates[arm] = {
            "total_post_init_frames": sum(x["pose"]["denominator"] for x in arms_list),
            "committed_avail": sum(x["pose"]["committed_avail"] for x in arms_list),
            "retained_avail": sum(x["pose"]["retained_avail"] for x in arms_list),
            "mean_committed_add_auc": round(float(np.mean([x["pose"]["committed_add_auc"] for x in arms_list])), 4),
            "mean_retained_add_auc": round(float(np.mean([x["pose"]["retained_add_auc"] for x in arms_list])), 4),
            "mean_P10_pct": round(float(np.mean([x["terminal_surface"]["P10_pct"] for x in arms_list])), 2),
            "mean_C10_pct": round(float(np.mean([x["terminal_surface"]["C10_pct"] for x in arms_list])), 2),
            "mean_F1_10_pct": round(float(np.mean([x["terminal_surface"]["F1_10_pct"] for x in arms_list])), 2),
            "mean_F95_mm": round(float(np.mean([x["terminal_surface"]["F95_mm"] for x in arms_list])), 2),
            "mean_R95_mm": round(float(np.mean([x["terminal_surface"]["R95_mm"] for x in arms_list])), 2),
        }

    payload = {
        "schema": "evidencescene.laval-real-unified-2x2.v1",
        "actions": list(args.actions),
        "frozen_config": {
            "voxel_size_m": 0.004,
            "required_pose_rank": 6,
            "minimum_anchored_landmarks": 3,
            "tolerance_scales": [0.1, 1.0, 10.0],
            "minimum_translational_singular_value": 1e-4,
            "maximum_translational_condition_number": 1e6,
            "minimum_observation_support": 2,
            "selector_budget": 1500,
            "evaluates_measured_surfels_only_is_prior_false": True,
        },
        "schur_gate_benchmark": schur_bench,
        "arm_aggregates": aggregates,
        "per_action": raw_results,
    }
    write_json(paths["run_cache"] / "laval_unified_2x2_raw.json", payload)
    print(json.dumps({"aggregates": aggregates, "schur_bench": schur_bench}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
