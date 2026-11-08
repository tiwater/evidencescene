"""Measure Point2Pose on the same frozen YCBInEOAT bleach0 frames."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
from common import base_metadata, frame_stems, p50_p95, write_result

DEFAULT_SEQUENCE = Path(
    os.environ.get(
        "EVIDENCESCENE_YCB_BLEACH0_DIR",
        str(Path.home() / "datasets" / "YCBInEOAT" / "bleach0"),
    )
)
DEFAULT_POINT2POSE = Path(
    os.environ.get(
        "EVIDENCESCENE_POINT2POSE_ROOT",
        str(Path.home() / "src" / "point-to-pose"),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--point2pose-root", type=Path, default=DEFAULT_POINT2POSE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.point2pose_root.resolve()
    sys.path.insert(0, str(root))
    import cv2
    import torch
    from omegaconf import OmegaConf
    from point2pose.data_types.frame import Frame
    from point2pose.io.sources.dataset.datareader import YcbineoatReader
    from point2pose.pipeline.modular_pipeline import ModularPipeline

    stems = frame_stems(args.sequence, args.frames)
    checkpoint = args.checkpoint or (
        root / "checkpoints/tapir/causal_bootstapir_checkpoint.pt"
    )
    cfg = OmegaConf.load(root / "configs/ycbineoat/ycbineoat_all.yaml")
    cfg.pipeline.params.use_segmenter = False
    cfg.pipeline.params.device = "cuda"
    cfg.pipeline.params.debug_dir = "/tmp/evidencescene-p2p-bench/debug/"
    cfg.pipeline.params.pose_save_path = "/tmp/evidencescene-p2p-bench/poses/"
    cfg.pipeline.params.save_pose = False
    cfg.pipeline.params.save_meta_data = False
    cfg.pipeline.params.sdf_mesh_save_dir = "/tmp/evidencescene-p2p-bench/mesh/"
    cfg.pipeline.params.sdf_mesh_save_every = 0
    cfg.reconstructor.params.build_sdf_after_global_opt = False
    cfg.reconstructor.params.sdf_use_gpu = False
    cfg.reconstructor.params.sdf_fuse_color = False
    cfg.tracker.params.checkpoint_path = str(checkpoint)
    cfg.tracker.params.device = "cuda"
    shutil.rmtree("/tmp/evidencescene-p2p-bench", ignore_errors=True)
    Path("/tmp/evidencescene-p2p-bench/debug").mkdir(parents=True)
    Path("/tmp/evidencescene-p2p-bench/poses").mkdir(parents=True)

    reader = YcbineoatReader(str(args.sequence))
    pipeline = ModularPipeline(cfg)
    device = torch.device("cuda")
    samples: list[float] = []
    nonlost = 0
    for index, _stem in enumerate(stems):
        rgb = reader.get_color(index)
        depth = reader.get_depth(index)
        mask = torch.from_numpy(reader.get_mask(index)).unsqueeze(0).unsqueeze(0).to(device)
        frame = Frame(
            id=index,
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=reader.K,
            depth_factor=1.0,
            timestamp=time.time(),
        )
        start = time.perf_counter()
        pose = pipeline.step(frame)
        samples.append((time.perf_counter() - start) * 1000.0)
        nonlost += int(pose is not None and np.asarray(pose).size > 0)
        if index == 0:
            print(f"first frame: {samples[0]:.1f} ms", file=sys.stderr)

    payload = base_metadata(
        protocol="ycbineoat_bleach0_refmask_40f_v1",
        sequence=args.sequence,
        stems=stems,
        method="Point2Pose",
    )
    payload["metrics"] = {
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
        **p50_p95(samples, warmup=args.warmup),
        "nonlost_frames": nonlost,
        "frames": len(stems),
    }
    payload["notes"] = (
        "Per-frame reference masks; segmenter disabled; SDF global rebuild disabled. "
        "Frontend + local/global graph remains enabled. Warm statistics exclude the first --warmup frames."
    )
    write_result(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
