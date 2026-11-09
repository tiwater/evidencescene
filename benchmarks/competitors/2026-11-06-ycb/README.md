# Controlled YCBInEOAT system benchmark

This is the source of the **controlled same-input** rows on the project page.
It compares the EvidenceScene native pose branch against Point2Pose on the same
40 RGB-D frames, the same per-frame reference masks, the same host and the same
RTX 5090.

## Protocol

- Protocol ID: `ycbineoat_bleach0_refmask_40f_v1`
- Session: `YCBInEOAT/bleach0`
- Frames: first 40 sorted RGB frames, `1578966042136484471` …
  `1578966044363778497`
- Input SHA-256:
  `39ae471e99f341be2296762e4224b9f12449ff49657c780781aceae5bd798b3b`
  (relative path + size + SHA-256 for every RGB/depth/mask frame)
- Masks: per-frame reference masks, not autonomous segmentation
- Hardware: NVIDIA RTX 5090 32 GB, AMD Ryzen 7 9800X3D
- Software: Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.9
- Warm-up: first 5 frames excluded; p95 is nearest-rank over the remaining 35
- Repeats: three independent runs; reported values are the median of run-level p50/p95/RSS
- Latency: wall time around one `track_registered` / `pipeline.step` call,
  including each method's own Python adapter and graph/optimizer work
- Not measured: HTTP/ROS transport, sensor capture, SAM2 segmentation, network,
  full-session sustained throughput

## Results

| System | Scope | p50 | p95 | Throughput | Peak RSS (MB) / frames |
| --- | --- | ---: | ---: | ---: | ---: |
| EvidenceScene native BundleTrack branch | pose tracking only; surface integration disabled | 30.24 ms | 33.05 ms | 33.1 fps | 2544.8 / 40/40 candidates |
| Point2Pose | frontend + local/global graph; segmenter disabled; SDF global rebuild disabled | 41.60 ms | 42.46 ms | 24.0 fps | 2219.9 / 40/40 non-lost |

The machine-readable results are
[`results/bundletrack-ycb-bleach0-40f.json`](results/bundletrack-ycb-bleach0-40f.json)
and
[`results/point2pose-ycb-bleach0-40f.json`](results/point2pose-ycb-bleach0-40f.json).
Both use `evidencescene.system-performance.v1`; `compare_systems.py` accepts
them because protocol, session and input SHA-256 match.

## Reproduce

The scripts use explicit paths so the comparison can be rerun without changing
the package under test.

### EvidenceScene native branch

```bash
# Cached pinned BundleTrack build + dependency-prefix marker
python3 benchmarks/competitors/2026-11-06-ycb/run_bundletrack.py \
  --sequence /path/to/YCBInEOAT/bleach0 \
  --source-root /path/to/bundlesdf-cache/runtime-pinned-20260714 \
  --build-root /path/to/bundlesdf-cache/BundleTrack-build-pinned \
  --dependency-prefix /path/to/bundlesdf-cache/dependency-prefix-v1 \
  --checkpoint /path/to/bundlesdf-cache/runtime-pinned-20260714/BundleTrack/LoFTR/weights/outdoor_ds.ckpt \
  --frames 40 \
  --out benchmarks/competitors/2026-11-06-ycb/results/bundletrack-ycb-bleach0-40f.json
```

The cached build is `ffa67d425240` plus the deterministic EvidenceScene patches;
it is the native pose branch, not a claim of unmodified upstream BundleTrack.

### Point2Pose

```bash
# Python 3.12 environment with torch 2.8, gtsam 4.3, tapnet, LightGlue,
# segment-anything-2-real-time and omegaconf.
python3 benchmarks/competitors/2026-11-06-ycb/run_point2pose.py \
  --sequence /path/to/YCBInEOAT/bleach0 \
  --point2pose-root /path/to/point-to-pose \
  --checkpoint /path/to/point-to-pose/checkpoints/tapir/causal_bootstapir_checkpoint.pt \
  --frames 40 \
  --out benchmarks/competitors/2026-11-06-ycb/results/point2pose-ycb-bleach0-40f.json
```

Pinned Point2Pose commit: `51856226610df75e5c06e8de545bd27f7c4ba99c`.
BootsTAPIR checkpoint SHA-256:
`87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.

### Compare

```bash
python3 benchmarks/performance/compare_systems.py \
  benchmarks/competitors/2026-11-06-ycb/results/bundletrack-ycb-bleach0-40f.json \
  benchmarks/competitors/2026-11-06-ycb/results/point2pose-ycb-bleach0-40f.json \
  --out benchmarks/competitors/2026-11-06-ycb/results/compare.md
```

## Limits

- 40 adjacent frames are correlated; the p95 is not a population estimate.
- The measurement is online tracking, not a full teleop pipeline or HTTP endpoint.
- Both methods receive reference masks; neither result measures autonomous
  segmentation.
- Point2Pose runs without its SAM2 segmenter because the benchmark supplies
  masks; it also disables global SDF rebuild to isolate tracking-stage work.
- Do not compare these numbers to the published/reference rows in
  `docs/performance_leaderboard.json`; those use different protocols.
