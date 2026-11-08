"""Measure the EvidenceScene native BundleTrack branch on YCBInEOAT bleach0."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # repository root
sys.path.insert(0, str(HERE))
from common import base_metadata, frame_stems, p50_p95, write_result

from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput
from evidencescene.tracking.pinned_bundletrack_runner import (
    open_pinned_bundletrack_runtime,
)

DEFAULT_SEQUENCE = Path(
    os.environ.get(
        "EVIDENCESCENE_YCB_BLEACH0_DIR",
        str(Path.home() / "datasets" / "YCBInEOAT" / "bleach0"),
    )
)
DEFAULT_SOURCE = Path(
    os.environ.get(
        "EVIDENCESCENE_BUNDLESDF_SOURCE",
        str(Path.home() / ".cache" / "evidencescene-bundlesdf" / "source"),
    )
)
DEFAULT_BUILD = Path(
    os.environ.get(
        "EVIDENCESCENE_BUNDLESDF_BUILD",
        str(Path.home() / ".cache" / "evidencescene-bundlesdf" / "build"),
    )
)
DEFAULT_PREFIX = Path(
    os.environ.get(
        "EVIDENCESCENE_BUNDLESDF_PREFIX",
        str(Path.home() / ".cache" / "evidencescene-bundlesdf" / "dependency-prefix-v1"),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--dependency-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or (
        args.source_root / "BundleTrack/LoFTR/weights/outdoor_ds.ckpt"
    )
    stems = frame_stems(args.sequence, args.frames)
    output_root = Path("/tmp/evidencescene-bt-bench")
    if output_root.exists():
        shutil.rmtree(output_root)
    intrinsics = np.loadtxt(args.sequence / "cam_K.txt")
    runtime = open_pinned_bundletrack_runtime(
        output_root=output_root,
        source_root=args.source_root,
        build_root=args.build_root,
        dependency_prefix=args.dependency_prefix,
        checkpoint_path=checkpoint,
        runtime_output_mode="online_pose",
        tracking_only=True,
    )
    samples: list[float] = []
    candidates = 0
    try:
        for index, stem in enumerate(stems):
            rgb = cv2.cvtColor(
                cv2.imread(str(args.sequence / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            depth = cv2.imread(
                str(args.sequence / "depth" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED,
            ).astype(np.uint16)
            mask = cv2.imread(
                str(args.sequence / "gt_mask" / f"{stem}.png"),
                cv2.IMREAD_UNCHANGED,
            ).astype(bool)
            timestamp = int(stem)
            frame = RegisteredBundleTrackInput(
                stream_key=("ycb", "bleach0", "cam", "obj0", "v1", "rgbd", "object0"),
                color_timestamp_ns=timestamp,
                depth_timestamp_ns=timestamp,
                color_rgb=rgb,
                depth_mm=depth,
                mask=mask,
                intrinsics=intrinsics,
                source_evidence_hash=f"ycb-bleach0:{stem}",
            )
            start = time.perf_counter()
            result = runtime.track_registered(frame)
            samples.append((time.perf_counter() - start) * 1000.0)
            candidates += int(result["status"] == "candidate")
            if index == 0:
                print(f"first frame: {samples[0]:.1f} ms", file=sys.stderr)
    finally:
        runtime.close()

    payload = base_metadata(
        protocol="ycbineoat_bleach0_refmask_40f_v1",
        sequence=args.sequence,
        stems=stems,
        method="EvidenceScene native BundleTrack branch",
    )
    payload["metrics"] = {
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
        **p50_p95(samples, warmup=args.warmup),
        "nonlost_frames": candidates,
        "frames": len(stems),
    }
    payload["notes"] = (
        "Pose tracking only; per-frame reference masks; surface integration disabled. "
        "Warm statistics exclude the first --warmup frames."
    )
    write_result(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
