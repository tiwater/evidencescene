#!/usr/bin/env python3
"""Run EvidenceScene Composed Native (BundleTrack + deterministic/NO_BA retirement) across 9/9 YCB & Laval."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from suite_common import (  # noqa: E402
    LAVAL_ACTION_IDS,
    YCBINEOAT_SPECS,
    YCB_SPEC_BY_ID,
    default_paths,
    erode_mask_3x3,
    load_laval_frames_and_gt,
    load_ycb_sequence_stems,
    p50_p95,
    write_json,
)


def _retire_blocked_contiguous(runtime) -> list[str]:
    """Guarded contiguous NO_BA non-keyframe retirement inside BundleTrack."""
    tracker = runtime.session.tracker
    bundler = tracker.bundler
    status_no_ba = runtime.frame_status.Frame.NO_BA
    width = int(tracker.cfg_track["bundle"]["window_size"]) + 1
    frames = list(bundler._frames.values())
    latest = sorted(frames, key=lambda f: int(f._id))[-width:]
    current = bundler._newframe
    cur_id = int(current._id)
    if len(latest) != width or [int(f._id) for f in latest] != list(range(cur_id - width + 1, cur_id + 1)):
        return []
    if any(f._status != status_no_ba for f in latest):
        return []

    protected = (
        {str(f._id_str) for f in bundler._keyframes}
        | {str(current._id_str), str(bundler._firstframe._id_str)}
        | {str(f._id_str) for f in bundler._local_frames}
        | {str(f._id_str) for f in frames if int(f._id) == int(current._ref_frame_id)}
    )
    cache = runtime._pose_snapshot_cache
    eligible = []
    for f in frames:
        name = str(f._id_str)
        c = cache.get(name)
        if name not in protected and f._status == status_no_ba and c is not None and c[1] == status_no_ba and c[2] is None:
            eligible.append(f)

    nonkey = len(frames) - len(bundler._keyframes)
    retired: list[str] = []
    for f in sorted(eligible, key=lambda item: int(item._id)):
        if nonkey <= width:
            break
        name = str(f._id_str)
        if bundler.forgetFrame(f):
            retired.append(name)
            nonkey -= 1
    return retired


def _run_worker(dataset: str, seq_id: str, mask_mode: str) -> int:
    from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput
    import evidencescene.tracking.pinned_bundletrack_runner as pinned

    paths = default_paths()
    out_root = paths["run_cache"] / "composed-native" / f"{dataset}-{mask_mode}" / seq_id
    out_json = out_root / "results.json"
    out_meta = out_root / "summary.json"
    if out_json.is_file() and out_meta.is_file():
        print(f"[ComposedNative:{dataset}:{mask_mode}:{seq_id}] Cached at {out_root}", flush=True)
        return 0

    bt_work_dir = out_root / "native_tmp"
    shutil.rmtree(bt_work_dir, ignore_errors=True)

    orig_configs = pinned._tracking_configs
    if dataset == "laval":
        def _laval_configs(**kwargs):
            tracking, nerf = orig_configs(**kwargs)
            tracking["depth_processing"]["zfar"] = 2.0
            nerf["far"] = 2.0
            tracking["SPDLOG"] = 0
            return tracking, nerf

        pinned._tracking_configs = _laval_configs
    else:
        def _ycb_configs(**kwargs):
            tracking, nerf = orig_configs(**kwargs)
            tracking["SPDLOG"] = 0
            return tracking, nerf

        pinned._tracking_configs = _ycb_configs

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    retired_total = 0

    try:
        with pinned.open_pinned_bundletrack_runtime(
            output_root=bt_work_dir,
            source_root=paths["bundlesdf_source"],
            build_root=paths["bundlesdf_build"],
            dependency_prefix=paths["bundlesdf_prefix"],
            checkpoint_path=paths["loftr_checkpoint"],
            runtime_output_mode="full_audit",
            tracking_only=True,
        ) as runtime:
            orig_snapshot = runtime._retained_pose_snapshot

            def _snapshot_with_retire(*a, **kw):
                nonlocal retired_total
                retired_total += len(_retire_blocked_contiguous(runtime))
                return orig_snapshot(*a, **kw)

            runtime._retained_pose_snapshot = _snapshot_with_retire

            if dataset == "ycb":
                seq_dir = paths["ycb_root"] / seq_id
                stems = load_ycb_sequence_stems(seq_dir)
                intrinsics = np.loadtxt(seq_dir / "cam_K.txt").reshape(3, 3)

                gen_masks_dir = None
                if mask_mode == "generated":
                    spec = YCB_SPEC_BY_ID[seq_id]
                    if spec.prior_p2p_gen and (paths["prior_ycb_root"] / spec.prior_p2p_gen / "masks").is_dir():
                        gen_masks_dir = paths["prior_ycb_root"] / spec.prior_p2p_gen / "masks"
                    else:
                        gen_masks_dir = paths["run_cache"] / "point2pose" / "ycb-gen" / seq_id / "masks"
                    if not gen_masks_dir.is_dir():
                        raise FileNotFoundError(f"Missing generated masks directory for {seq_id}: {gen_masks_dir}")

                total = len(stems)
                for i, stem in enumerate(stems):
                    bgr = cv2.imread(str(seq_dir / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    depth_mm = cv2.imread(str(seq_dir / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16)
                    if mask_mode == "reference":
                        raw_mask = cv2.imread(str(seq_dir / "gt_mask" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                        raw_mask_bool = np.any(raw_mask > 0, axis=2) if raw_mask.ndim == 3 else (raw_mask > 0)
                    else:
                        gen_mask = cv2.imread(str(gen_masks_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
                        raw_mask_bool = (gen_mask > 0) if gen_mask is not None else np.zeros(depth_mm.shape, dtype=bool)
                    ts_ns = int(stem)

                    if not erode_mask_3x3(raw_mask_bool).any():
                        if i == 0:
                            raise RuntimeError(f"Empty frame-0 mask for {seq_id}")
                        records.append(
                            {
                                "index": i,
                                "timestamp_ns": ts_ns,
                                "status": "no_mask_input",
                                "pose": None,
                                "latency_ms": 0.0,
                            }
                        )
                        continue

                    frame = RegisteredBundleTrackInput(
                        stream_key=("ycb", seq_id, "cam", "obj0", "v1", "rgbd", "object0"),
                        color_timestamp_ns=ts_ns,
                        depth_timestamp_ns=ts_ns,
                        color_rgb=rgb,
                        depth_mm=depth_mm,
                        mask=raw_mask_bool,
                        intrinsics=intrinsics,
                        source_evidence_hash=f"ycb:{seq_id}:{mask_mode}:{i}",
                        _take_ownership=True,
                    )
                    t0 = time.perf_counter()
                    res = runtime.track_registered(frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(dt_ms)

                    status = str(res["tracking_status"])
                    pose = res.get("camera_from_object")
                    records.append(
                        {
                            "index": i,
                            "timestamp_ns": ts_ns,
                            "status": status,
                            "pose": pose,
                            "latency_ms": round(dt_ms, 3),
                        }
                    )
                    if (i + 1) % 200 == 0 or i + 1 == total:
                        print(f"[ComposedNative:ycb:{mask_mode}:{seq_id}] {i + 1}/{total}", flush=True)
            else:
                frames, intrinsics, _gt, _model = load_laval_frames_and_gt(paths["laval_root"] / seq_id)
                total = len(frames)
                for f in frames:
                    i = f["index"]
                    if f["status"] != "available" or not erode_mask_3x3(f["mask"]).any():
                        if i == 0:
                            raise RuntimeError(f"Unavailable frame-0 on {seq_id}")
                        records.append({"index": i, "status": "no_mask_input", "pose": None, "latency_ms": 0.0})
                        continue
                    ts_ns = (i + 1) * 33_333_333
                    frame = RegisteredBundleTrackInput(
                        stream_key=("laval", seq_id, "cam", "obj0", "v1", "rgbd", "object0"),
                        color_timestamp_ns=ts_ns,
                        depth_timestamp_ns=ts_ns,
                        color_rgb=f["rgb"],
                        depth_mm=f["depth_mm"],
                        mask=f["mask"],
                        intrinsics=intrinsics,
                        source_evidence_hash=f"laval:{seq_id}:{i}",
                        _take_ownership=True,
                    )
                    t0 = time.perf_counter()
                    res = runtime.track_registered(frame)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(dt_ms)
                    status = str(res["tracking_status"])
                    pose = res.get("camera_from_object")
                    records.append(
                        {
                            "index": i,
                            "status": status,
                            "pose": pose,
                            "latency_ms": round(dt_ms, 3),
                        }
                    )
                print(f"[ComposedNative:laval:{seq_id}] {total}/{total}", flush=True)
    finally:
        pinned._tracking_configs = orig_configs
        shutil.rmtree(bt_work_dir, ignore_errors=True)

    summary = {
        "system": "EvidenceScene Composed Native",
        "dataset": dataset,
        "sequence": seq_id,
        "mask_mode": mask_mode,
        "frames": len(records),
        "retired_blocked_count": retired_total,
        "latency": p50_p95(latencies_ms, warmup=5),
    }
    write_json(out_json, records)
    write_json(out_meta, summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=("ycb", "laval"), default="ycb")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--mask-mode", choices=("reference", "generated", "sam2"), default="reference")
    parser.add_argument("--include-laval", action="store_true")
    parser.add_argument("--include-generated", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if not args.sequence:
            raise ValueError("--sequence required with --worker")
        return _run_worker(args.dataset, args.sequence, args.mask_mode)

    paths = default_paths()
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    lib_prefix = f"{paths['bundlesdf_build']}:{paths['bundlesdf_prefix']}/lib"
    env["LD_LIBRARY_PATH"] = f"{lib_prefix}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else lib_prefix

    jobs: list[tuple[str, str, str]] = []
    for spec in YCBINEOAT_SPECS:
        jobs.append(("ycb", spec.seq_id, "reference"))
    if args.include_generated:
        for spec in YCBINEOAT_SPECS:
            jobs.append(("ycb", spec.seq_id, "generated"))
    if args.include_laval:
        for action in LAVAL_ACTION_IDS:
            jobs.append(("laval", action, "sam2"))

    for ds, seq, mode in jobs:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--dataset",
            ds,
            "--sequence",
            seq,
            "--mask-mode",
            mode,
        ]
        res = subprocess.run(cmd, env=env, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Worker failed ({res.returncode}) on {ds}/{mode}/{seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
