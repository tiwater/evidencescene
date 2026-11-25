#!/usr/bin/env python3
"""Run CAD-assisted FoundationPose (NVlabs a1b694b) on missing YCBInEOAT & all 3 Laval clips."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from suite_common import (  # noqa: E402
    LAVAL_ACTION_IDS,
    YCB_SPEC_BY_ID,
    default_paths,
    erode_mask_3x3,
    load_laval_frames_and_gt,
    load_ycb_sequence_stems,
    p50_p95,
    write_json,
)

EXPECTED_FP_COMMIT = "a1b694b83e633c2cb6115b9063d940a687759392"


def _verify_upstream_repo(fp_root: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(fp_root), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_FP_COMMIT:
        raise RuntimeError(f"Unexpected FoundationPose commit {head} != {EXPECTED_FP_COMMIT}")
    return head


def _run_sequence(
    *,
    seq_name: str,
    mesh_path: Path,
    intrinsics: np.ndarray,
    frame_iter,
    total_frames: int,
    out_dir: Path,
    fp_root: Path,
) -> dict[str, object]:
    out_json = out_dir / "results.json"
    out_meta = out_dir / "summary.json"
    if out_json.is_file() and out_meta.is_file():
        print(f"[FoundationPose:{seq_name}] Cached at {out_dir}", flush=True)
        return {"sequence": seq_name, "cached": True, "frames": total_frames}

    if str(fp_root) not in sys.path:
        sys.path.insert(0, str(fp_root))
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from Utils import set_seed
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    set_seed(0)
    mesh = trimesh.load(mesh_path.resolve(), force="mesh", process=False)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != mesh.vertices.shape:
        normals = np.zeros_like(mesh.vertices, dtype=np.float32)
    dbg_dir = out_dir / "fp_debug"
    shutil.rmtree(dbg_dir, ignore_errors=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(dbg_dir),
        debug=0,
        glctx=dr.RasterizeCudaContext(),
    )

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for idx, rgb, depth_m, mask0 in frame_iter:
            t0 = time.perf_counter()
            if idx == 0:
                if mask0 is None or not mask0.any():
                    raise RuntimeError(f"Empty frame-0 mask on {seq_name}")
                h_ori, w_ori = depth_m.shape[:2]
                if h_ori * w_ori > 640 * 480:
                    scale = 0.5
                    nw, nh = int(round(w_ori * scale)), int(round(h_ori * scale))
                    rgb_reg = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                    dep_reg = cv2.resize(depth_m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                    msk_reg = cv2.resize(mask0.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
                    k_reg = intrinsics.copy()
                    k_reg[:2, :] *= scale
                    pose = est.register(K=k_reg, rgb=rgb_reg, depth=dep_reg, ob_mask=msk_reg, iteration=5)
                else:
                    pose = est.register(K=intrinsics, rgb=rgb, depth=depth_m, ob_mask=mask0, iteration=5)
                torch.cuda.empty_cache()
            else:
                pose = est.track_one(rgb=rgb, depth=depth_m, K=intrinsics, iteration=2)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            mat = np.asarray(pose, dtype=float)
            records.append(
                {
                    "index": idx,
                    "mask_used": idx == 0,
                    "status": "pose_returned_unverified",
                    "camera_from_model": mat.tolist(),
                    "latency_ms": round(dt_ms, 3),
                }
            )
            if (idx + 1) % 200 == 0 or idx + 1 == total_frames:
                print(f"[FoundationPose:{seq_name}] {idx + 1}/{total_frames}", flush=True)

    del est
    gc.collect()
    torch.cuda.empty_cache()

    shutil.rmtree(dbg_dir, ignore_errors=True)
    summary = {
        "system": "FoundationPose",
        "upstream_commit": EXPECTED_FP_COMMIT,
        "sequence": seq_name,
        "frames": len(records),
        "latency": p50_p95(latencies_ms, warmup=5),
    }
    write_json(out_json, records)
    write_json(out_meta, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=[
            "bleach_hard_00_03_chaitanya",
            "cracker_box_yalehand0",
            "mustard_easy_00_02",
            "sugar_box1",
        ],
    )
    parser.add_argument("--laval", nargs="*", default=list(LAVAL_ACTION_IDS))
    parser.add_argument("--include-bleach0-timing", action="store_true")
    args = parser.parse_args()

    paths = default_paths()
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    fp_root = paths["foundationpose_root"]
    _verify_upstream_repo(fp_root)
    ycb_root = paths["ycb_root"]
    laval_root = paths["laval_root"]
    out_root = paths["run_cache"] / "foundationpose"
    out_root.mkdir(parents=True, exist_ok=True)

    seqs = list(args.sequences)
    if args.include_bleach0_timing and "bleach0" not in seqs:
        seqs.insert(0, "bleach0")

    summaries = []
    for seq_id in seqs:
        spec = YCB_SPEC_BY_ID[seq_id]
        seq_dir = ycb_root / seq_id
        stems = load_ycb_sequence_stems(seq_dir)
        intrinsics = np.loadtxt(seq_dir / "cam_K.txt").reshape(3, 3)
        mesh_path = ycb_root / "models" / spec.mesh_relpath

        def _ycb_iter(sdir=seq_dir, f_stems=stems):
            for i, stem in enumerate(f_stems):
                bgr = cv2.imread(str(sdir / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth_m = cv2.imread(str(sdir / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                mask0 = None
                if i == 0:
                    m = cv2.imread(str(sdir / "gt_mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                    mask0 = (np.any(m > 0, axis=2) if m.ndim == 3 else (m > 0))
                yield i, rgb, depth_m, mask0

        summaries.append(
            _run_sequence(
                seq_name=seq_id,
                mesh_path=mesh_path,
                intrinsics=intrinsics,
                frame_iter=_ycb_iter(),
                total_frames=len(stems),
                out_dir=out_root / "ycb" / seq_id,
                fp_root=fp_root,
            )
        )

    for action in args.laval:
        frames, intrinsics, _gt, model_path = load_laval_frames_and_gt(laval_root / action)

        def _laval_iter(f_list=frames):
            for f in f_list:
                yield f["index"], f["rgb"], f["depth_m"], (f["mask"] if f["index"] == 0 else None)

        summaries.append(
            _run_sequence(
                seq_name=f"laval/{action}",
                mesh_path=model_path,
                intrinsics=intrinsics,
                frame_iter=_laval_iter(),
                total_frames=len(frames),
                out_dir=out_root / "laval" / action,
                fp_root=fp_root,
            )
        )

    write_json(out_root / "suite_summary.json", {"runs": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
