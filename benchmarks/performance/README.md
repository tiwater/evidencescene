# Kernel performance benchmark (`kernel-v1`)

This directory is the reproducible **kernel** benchmark for `evidencescene`.
It measures deterministic, output-preserving library workloads so a version
change can be compared quantitatively without conflating it with pose accuracy
or recorded-session endpoint latency.

It is deliberately **not**:

- a recorded RGB-D session replay;
- an end-to-end HTTP/ROS latency measurement;
- a pose or surface accuracy benchmark;
- a comparison of kernel APIs against BundleTrack/Point2Pose system APIs.

Use the system track (see [Competitor comparison](#competitor-comparison))
for cross-system claims.

## Quick start

```sh
# Current package, JSON to stdout
python3 benchmarks/performance/bench_suite.py

# Write a labelled result
python3 benchmarks/performance/bench_suite.py \
    --label "$(git rev-parse --short HEAD)" \
    --out benchmarks/performance/results/current.json

# Compare two result files
python3 benchmarks/performance/compare.py baseline.json candidate.json

# Or run the same suite against a past v2 ref and the current tree
./benchmarks/performance/run_versions.sh 04a1b6b HEAD
```

`make perf` runs the suite; `make benchmark-versions` runs the two-ref
comparison.

## Workloads and metrics

Every workload reports `median_ms`, `p95_ms`, `min_ms`, `repeat`, and an
`output_digest`. Each timing sample includes one full call after warm-up.

| Workload ID | What it measures |
| --- | --- |
| `surface_snapshot_60k_budget_1500` | `RevisableSurfelMap.snapshot` over 60k voxels at the 1,500-point wire budget |
| `surface_update_60k_points` | One `RevisableSurfelMap.update` of a 60k-point observation into a fresh map |
| `voxel_centroids_60k_points` | Per-voxel centroid accumulation for 60k points |
| `rle_box_centroid_480x640` | Exact `O(runs)` RLE bounding box + centroid on a 480×640 mask |
| `surface_selection_60k_budget_1500` | `select_existing_points_shadow` over a 60k-point snapshot at budget 1,500 |
| `circle_consensus_400_points_256_hypotheses` | Seeded 3-point rim consensus on 400 points |
| `circular_aperture_fit_400_points` | Full circular aperture fit (plane + sectors + consensus + covariance) |

All metrics are latency in milliseconds and are **lower-is-better**.

## Why output digests matter

A speed-up is only valid if both versions produced the same output. The suite
hashes the relevant arrays or scalar results for every workload. `compare.py`
marks any mismatch as `**MISMATCH**` and `--require-identical` fails the run.
This prevents a "faster" version that silently changed surface ordering,
centroid sums, consensus decisions, or selection indices.

## Version comparison

`run_versions.sh <baseline-ref> [candidate-ref]`:

1. creates detached git worktrees for both refs;
2. runs the current suite script with `--package-root` pointing at each
   worktree's package;
3. writes `results/<ref>.json`;
4. runs `compare.py --require-identical`;
5. writes a Markdown report to `results/<baseline>_vs_<candidate>.md`.

Refs must contain the `evidencescene` package (v2 rename or later). For older
`causalpose3d` refs, check out the old repo and use its own benchmark copy.

The checked-in `results/04a1b6b_vs_6c0e8f4.md` is the recorded comparison
between the pre-optimization v2.0.0 tree (`04a1b6b`) and the performance tree
(`6c0e8f4`). On the recorded host it reports a **1.68x geometric-mean
speed-up** across the seven workloads, with **all output digests identical**.

## Recorded development results

Same host, Python 3.10.12, NumPy 1.21.5, warm process.

| Case | Baseline (`04a1b6b`) | Previous (`6c0e8f4`) | Current | Speed-up (vs `04a1b6b`) |
| --- | ---: | ---: | ---: | ---: |
| `surface_snapshot_60k_budget_1500` | 20.48 ms | 8.77 ms | **0.22 ms** | **93.59x** |
| `surface_update_60k_points` | 72.99 ms | 62.02 ms | **1.07 ms** | **68.23x** |
| `voxel_centroids_60k_points` | 10.33 ms | 6.06 ms | **4.57 ms** | **2.26x** |
| `rle_box_centroid_480x640` | 0.092 ms | 0.093 ms | **0.092 ms** | **1.00x** |
| `surface_selection_60k_budget_1500` | 15.52 ms | 15.46 ms | **1.05 ms** | **14.77x** |
| `circle_consensus_400_points_256_hypotheses` | 4.45 ms | 1.75 ms | **0.81 ms** | **5.50x** |
| `circular_aperture_fit_400_points` | 4.35 ms | 1.52 ms | **1.17 ms** | **3.72x** |

Across all seven workloads, the current tree achieves an **8.65x geometric-mean speed-up** over `04a1b6b` with **100% bit-identical output digests**.

To also run the four-paradigm SOTA mechanism ablation suite (`ablation-v1`):

```sh
python3 benchmarks/performance/bench_ablation.py
```

These are kernel microbenchmarks on one host. They do not establish end-to-end
latency, live-site acceptance or a production cutover. Report the CPU, Python,
NumPy, git SHA and package version with every comparison; `bench_suite.py`
writes all of them into the JSON automatically.

## Competitor comparison

Do **not** compare BundleTrack/Point2Pose system latencies to these kernel
numbers. They run different pipelines, input contracts and scope.

For cross-system comparison use a recorded-session system result with this
schema:

```json
{
  "schema": "evidencescene.system-performance.v1",
  "system": "EvidenceScene",
  "version": "2.0.0",
  "protocol": "recorded_session_pipeline_v1",
  "session": {
    "id": "teleop_20260625_173830",
    "frames": 40,
    "input_sha256": "<sha256 of the frozen input manifest>"
  },
  "environment": {
    "cpu": "...",
    "gpu": "...",
    "python": "...",
    "numpy": "..."
  },
  "metrics": {
    "p50_ms": 127.8,
    "p95_ms": 153.4,
    "peak_rss_mb": 0.0,
    "nonlost_frames": 34,
    "output_digest": "<sha256 of the frozen output manifest>"
  },
  "notes": "warm HTTP endpoint, 40-frame configured pouring sequence"
}
```

`compare_systems.py` only compares results with the same `protocol` and
`session.input_sha256`; mismatched inputs are rejected. The numbers above are
an illustrative shape, not a new measurement.

### External competitor snapshots

Competitor measurements are kept out of the product benchmark path. They live
under [`../competitors/`](../competitors/) as dated, manually produced snapshots
with frozen inputs and provenance. They are not run by CI and are not package
dependencies. The first snapshot is
[`2026-11-06-ycb`](../competitors/2026-11-06-ycb/).

`compare_systems.py` remains a generic tool for explicitly selected snapshots;
it is not wired into CI. Do not substitute kernel timings for a system
comparison.
