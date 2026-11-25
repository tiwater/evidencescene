#!/usr/bin/env python3
"""Run pinned upstream Point2Pose (51856226610d) on new YCBInEOAT reference & SAM 2 generated masks."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from suite_common import (  # noqa: E402
    YCB_SPEC_BY_ID,
    default_paths,
    erode_mask_3x3,
    load_ycb_sequence_stems,
    p50_p95,
    sha256_file,
    write_json,
)

EXPECTED_P2P_COMMIT = "51856226610df75e5c06e8de545bd27f7c4ba99c"


def _run_sequence(
    *,
    seq_id: str,
    mode: str,
    ycb_root: Path,
    p2p_root: Path,
    bootstapir_ckpt: Path,
    sam2_ckpt: Path,
    out_dir: Path,
) -> dict[str, object]:
    out_json = out_dir / "results.json"
    out_meta = out_dir / "summary.json"
    if out_json.is_file() and out_meta.is_file():
        print(f"[Point2Pose:{seq_id}:{mode}] Cached at {out_dir}", flush=True)
        return {"sequence": seq_id, "mode": mode, "cached": True}

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

    dbg_dir = out_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    if mode == "generated":
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(p2p_root / "configs/ycbineoat/eccv_final.yaml")
    cfg.tracker.params.checkpoint_path = str(bootstapir_ckpt)
    cfg.tracker.params.device = "cuda"
    cfg.segmenter.params.checkpoint = str(sam2_ckpt)
    cfg.segmenter.params.device = "cuda"
    cfg.pipeline.params.use_segmenter = mode == "generated"
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
    if hasattr(cfg, "visualization") and hasattr(cfg.visualization, "params"):
        cfg.visualization.params.save_images = False
        cfg.visualization.params.visualize_points = False
        cfg.visualization.params.output_image_dir = str(dbg_dir / "vis")

    seq_dir = ycb_root / seq_id
    stems = load_ycb_sequence_stems(seq_dir)
    intrinsics = np.loadtxt(seq_dir / "cam_K.txt").reshape(3, 3)

    pipeline = ModularPipeline(cfg)
    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    for i, stem in enumerate(stems):
        bgr = cv2.imread(str(seq_dir / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = cv2.imread(str(seq_dir / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        raw_mask = cv2.imread(str(seq_dir / "gt_mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        mask_bool = erode_mask_3x3(np.any(raw_mask > 0, axis=2) if raw_mask.ndim == 3 else (raw_mask > 0))
        ts_ns = int(stem)

        if mode == "reference" and not mask_bool.any():
            if i == 0:
                raise RuntimeError(f"Empty frame-0 reference mask on {seq_id}")
            records.append(
                {
                    "index": i,
                    "timestamp_ns": ts_ns,
                    "pose": None,
                    "lost": None,
                    "status": "no_mask_input",
                    "latency_ms": 0.0,
                }
            )
            continue

        if mode == "generated":
            seed = torch.from_numpy(mask_bool.astype(np.uint8))[None, None].to("cuda") if i == 0 else None
            if i == 0 and not mask_bool.any():
                raise RuntimeError(f"Empty frame-0 seed mask on {seq_id}")
            frame = Frame(
                id=i,
                rgb=rgb,
                depth=depth_m,
                mask=seed,
                intrinsics=intrinsics,
                depth_factor=1.0,
                timestamp=ts_ns / 1e9,
            )
        else:
            frame = Frame(
                id=i,
                rgb=rgb,
                depth=depth_m,
                mask=torch.from_numpy(mask_bool.astype(np.uint8))[None, None].to("cuda"),
                intrinsics=intrinsics,
                depth_factor=1.0,
                timestamp=ts_ns / 1e9,
            )

        t0 = time.perf_counter()
        pose = pipeline.step(frame)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        row: dict[str, object] = {
            "index": i,
            "timestamp_ns": ts_ns,
            "pose": np.asarray(pose).tolist() if pose is not None else None,
            "lost": [bool(x.lost) for x in pipeline.objects],
            "status": "tracker_returned",
            "latency_ms": round(dt_ms, 3),
        }
        if mode == "generated":
            if frame.mask is None:
                produced = np.zeros(mask_bool.shape, dtype=np.uint8)
                mask_state = "missing"
            else:
                produced = (frame.mask[0, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
                mask_state = "returned"
            mask_path = out_dir / "masks" / f"{i:06d}.png"
            if not cv2.imwrite(str(mask_path), produced):
                raise RuntimeError(f"Failed writing {mask_path}")
            row["mask_sha256"] = sha256_file(mask_path)
            row["mask_pixels"] = int(np.count_nonzero(produced))
            row["mask_state"] = mask_state
            row["reference_mask_used"] = i == 0
        records.append(row)

        if (i + 1) % 200 == 0 or i + 1 == len(stems):
            print(f"[Point2Pose:{seq_id}:{mode}] {i + 1}/{len(stems)}", flush=True)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "system": "Point2Pose",
        "upstream_commit": EXPECTED_P2P_COMMIT,
        "sequence": seq_id,
        "mode": mode,
        "frames": len(records),
        "latency": p50_p95(latencies_ms, warmup=5),
    }
    write_json(out_json, records)
    write_json(out_meta, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref-sequences",
        nargs="*",
        default=["cracker_box_yalehand0", "mustard_easy_00_02", "sugar_box1"],
    )
    parser.add_argument(
        "--gen-sequences",
        nargs="*",
        default=[
            "bleach_hard_00_03_chaitanya",
            "cracker_box_yalehand0",
            "mustard_easy_00_02",
            "sugar_box1",
        ],
    )
    args = parser.parse_args()

    paths = default_paths()
    p2p_root = paths["point2pose_root"]
    head = subprocess.check_output(["git", "-C", str(p2p_root), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_P2P_COMMIT:
        raise RuntimeError(f"Unexpected Point2Pose commit {head} != {EXPECTED_P2P_COMMIT}")

    out_root = paths["run_cache"] / "point2pose"
    runs = []
    for seq_id in args.ref_sequences:
        assert seq_id in YCB_SPEC_BY_ID
        runs.append(
            _run_sequence(
                seq_id=seq_id,
                mode="reference",
                ycb_root=paths["ycb_root"],
                p2p_root=p2p_root,
                bootstapir_ckpt=paths["bootstapir_checkpoint"],
                sam2_ckpt=paths["sam2_checkpoint"],
                out_dir=out_root / "ycb-ref" / seq_id,
            )
        )
    for seq_id in args.gen_sequences:
        assert seq_id in YCB_SPEC_BY_ID
        runs.append(
            _run_sequence(
                seq_id=seq_id,
                mode="generated",
                ycb_root=paths["ycb_root"],
                p2p_root=p2p_root,
                bootstapir_ckpt=paths["bootstapir_checkpoint"],
                sam2_ckpt=paths["sam2_checkpoint"],
                out_dir=out_root / "ycb-gen" / seq_id,
            )
        )
    write_json(out_root / "suite_summary.json", {"runs": runs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
