"""EvidenceScene SOTA paradigm ablation & mechanism benchmark suite (`ablation-v1`).

Quantitatively compares EvidenceScene's core algorithmic mechanisms against
four standard baseline paradigms on identical inputs:
1. RevisableSurfelMap (M+S) vs. static irreversible voxel integration (nvblox/Voxblox paradigm)
2. Schur-Complement Observability Gate (F+G+R+M) vs. ungated pose admission (feedforward/unconstrained BA paradigm)
3. O(runs) RLE interval set algebra vs. dense O(H*W) boolean mask pipeline at 1080p (1080x1920)
4. Causal C^1 SE(3) re-acquisition blending vs. unblended step-reacquisition extrapolation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))

import evidencescene
from evidencescene.cli import run_end_to_end_demo
from evidencescene.primitives import (
    decode_mask_rle,
    encode_mask_rle,
    rle_area,
    rle_iou,
)
from evidencescene.runtime import rle_mask_box_and_centroid
from evidencescene.teleop import CausalSe3TrajectoryPredictor


def _bench_rle_vs_dense_1080p(repeat: int = 15) -> dict[str, Any]:
    h, w = 1080, 1920
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[220:860, 480:1440] = True
    mask_a[360:440, 720:880] = False
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[232:872, 496:1456] = True
    mask_b[360:440, 720:880] = False

    rle_a = encode_mask_rle(mask_a)
    rle_b = encode_mask_rle(mask_b)

    # Warm-up
    _ = rle_mask_box_and_centroid(rle_a, width=w, height=h)
    _ = rle_iou(rle_a, rle_b)

    t0 = time.perf_counter()
    for _ in range(repeat):
        area_rle = rle_area(rle_a)
        box_cent = rle_mask_box_and_centroid(rle_a, width=w, height=h)
        iou_rle = rle_iou(rle_a, rle_b)
    rle_ms = ((time.perf_counter() - t0) * 1000.0) / repeat

    t1 = time.perf_counter()
    for _ in range(repeat):
        da = decode_mask_rle(rle_a, width=w, height=h)
        db = decode_mask_rle(rle_b, width=w, height=h)
        assert da is not None and db is not None
        area_dense = int(np.count_nonzero(da))
        ys, xs = np.nonzero(da)
        box_dense = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        cent_dense = (float(xs.mean()), float(ys.mean()))
        inter = int(np.count_nonzero(da & db))
        union = int(np.count_nonzero(da | db))
        iou_dense = inter / union
    dense_ms = ((time.perf_counter() - t1) * 1000.0) / repeat

    assert box_cent is not None
    assert area_rle == area_dense
    assert box_cent[0] == box_dense
    assert np.allclose(box_cent[1], cent_dense, atol=1e-12)
    assert abs(iou_rle - iou_dense) < 1e-12

    return {
        "resolution": "1080x1920",
        "rle_runs": len(rle_a["counts"]) // 2,
        "rle_median_ms": round(rle_ms, 4),
        "dense_median_ms": round(dense_ms, 4),
        "speedup_x": round(dense_ms / max(rle_ms, 1e-6), 2),
        "exact_parity": True,
    }


def _bench_c1_reacquisition_vs_unblended() -> dict[str, Any]:
    predictor = CausalSe3TrajectoryPredictor(
        reacquisition_gap_threshold_s=0.05,
        c1_join_duration_s=0.15,
        maximum_join_velocity_jump_mps=0.15,
        horizons_ms=(100.0, 250.0, 500.0),
    )
    pose0 = np.eye(4, dtype=np.float64)
    pose0[:3, 3] = [0.10, 0.00, 0.50]
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, 3] = [0.11, 0.00, 0.50]
    # Occlusion gap of 120 ms followed by a 25 mm re-acquisition displacement
    pose_reacq = np.eye(4, dtype=np.float64)
    pose_reacq[:3, 3] = [0.145, 0.012, 0.495]

    predictor.observe(pose=pose0, timestamp_ns=1_000_000_000, is_reacquisition=False)
    predictor.observe(pose=pose1, timestamp_ns=1_033_333_333, is_reacquisition=False)
    rep = predictor.observe(
        pose=pose_reacq, timestamp_ns=1_153_333_333, is_reacquisition=True
    )

    unblended_jump_mm = float(np.linalg.norm(pose_reacq[:3, 3] - pose1[:3, 3]) * 1000.0)
    c1_jump_mm = float(rep.position_join_jump_m * 1000.0)
    horizons = [h.interval_half_width_95_m for h in rep.horizons]
    monotonic_uncertainty = (
        rep.interval_width_monotonic_violation_count == 0
        and all(horizons[i] <= horizons[i + 1] for i in range(len(horizons) - 1))
    )

    return {
        "unblended_position_jump_mm": round(unblended_jump_mm, 3),
        "evidencescene_c1_position_jump_mm": round(c1_jump_mm, 6),
        "c1_join_active": bool(rep.c1_join_active),
        "monotonic_horizon_uncertainty": monotonic_uncertainty,
    }


def run_ablation_suite() -> dict[str, Any]:
    e2e = run_end_to_end_demo(num_frames=24, seed=42)
    rle_1080p = _bench_rle_vs_dense_1080p()
    c1_vs_unblended = _bench_c1_reacquisition_vs_unblended()

    p_base = e2e["pose_evaluation"]["ungated_baseline"]
    p_ours = e2e["pose_evaluation"]["evidencescene_gated_and_revised"]
    s_base = e2e["surface_evaluation"]["naive_unrevised_accumulation"]
    s_ours = e2e["surface_evaluation"]["evidencescene_revisable_surfel_map"]

    digest_payload = json.dumps(
        {
            "pose": e2e["pose_evaluation"],
            "surface": e2e["surface_evaluation"],
            "c1": c1_vs_unblended,
            "rle_parity": rle_1080p["exact_parity"],
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()[:16]

    return {
        "schema": "evidencescene.ablation-benchmark.v1",
        "benchmark_id": "ablation-v1",
        "version": evidencescene.__version__,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "ablations": {
            "1_observability_gate_vs_ungated_pose": {
                "ungated_mean_add_mm": p_base["mean_add_mm"],
                "evidencescene_mean_add_mm": p_ours["mean_add_mm"],
                "add_error_reduction_x": round(
                    p_base["mean_add_mm"] / max(p_ours["mean_add_mm"], 1e-6), 2
                ),
                "ungated_mean_adds_mm": p_base["mean_adds_mm"],
                "evidencescene_mean_adds_mm": p_ours["mean_adds_mm"],
                "run_to_run_spread_mm": p_ours["run_to_run_spread_mm"],
            },
            "2_revisable_surfel_vs_static_voxel": {
                "static_voxel_P10_pct": s_base["P_10_pct"],
                "evidencescene_P10_pct": s_ours["P_10_pct"],
                "static_voxel_C10_pct": s_base["C_10_pct"],
                "evidencescene_C10_pct": s_ours["C_10_pct"],
                "static_voxel_F95_mm": s_base["F_95_mm"],
                "evidencescene_F95_mm": s_ours["F_95_mm"],
                "f95_error_reduction_x": round(
                    s_base["F_95_mm"] / max(s_ours["F_95_mm"], 1e-6), 2
                ),
            },
            "3_rle_algebra_vs_dense_1080p": rle_1080p,
            "4_c1_causal_se3_vs_unblended_reacquisition": c1_vs_unblended,
        },
        "output_digest": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_ablation_suite()
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
