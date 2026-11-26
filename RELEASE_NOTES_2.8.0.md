# EvidenceScene v2.8.0 Release Notes

**Release Date:** 2026-11-26

`evidencescene v2.8.0` completes full runtime AVX-512/AVX2 SIMD dispatch across all native kernels, adds native multi-view per-voxel RGB fusion to `RevisableSurfelMap`, introduces C11 single-pass 2D height-grid foreground clustering (`cluster_height_grid_2d`) and azimuthal outer-rim extraction (`extract_rim_annulus_points`), enables zero-copy masked planar integration (`retain_camera_points=False`), and refreshes the public `kernel-v1` benchmark leaderboard (**`191.31x` geometric-mean speed-up** over baseline `04a1b6b`).

## What's New

### 1. Native Multi-View RGB Fusion in `RevisableSurfelMap`
- `observation_voxel_centroids(points, voxel_size_m, colors_rgb=None)` computes per-voxel RGB means alongside lexicographic centroids in a single C11 pass (across Fast Path A, L3 direct-grid Path B, and hash-table Path C, with exact NumPy fallback).
- `RevisableSurfelMap.update()` accepts either `(camera, points)` or `(camera, points, colors_rgb_u8)` observations, atomically accumulating and retracting per-voxel RGB sums across insertions, pose revisions, and retirements (`surfel_general_update`).
- `RevisableSurfelMap.snapshot(..., include_colors=True)` returns fused `uint8` RGB directly in `snapshot_uniform_fast` and `snapshot_select` — eliminating downstream per-object kNN RGB donor transfers (**`3.19x` faster** end-to-end: `0.078 ms` vs `0.250 ms` per object).

### 2. Single-Pass C11 2D Height-Grid Clustering (`evidencescene.surface.grid_clustering`)
- Added `GridCluster2D` and `cluster_height_grid_2d` backed by C11 `cluster_height_grid_2d_fast` (using zero-allocation `ThreadScratch` buffers):
  - Single-pass foreground bounds filtering and 2D occupancy projection.
  - L1-cache vertical air-gap decoupling via two 3×3 dilations to separate low tabletop objects (layer 0) from overhanging/tall equipment structures (layer 1).
  - Row-major 8-connected BFS connected-component labeling matching `scipy.ndimage.label`.
  - Deterministic C11 3-way quickselect linear quantiles (`h_p5`, `h_p97`, `xy_min_p3`, `xy_max_p97`, `z_top_p985`), cutting 72k-point tabletop clustering time by **`2.02x`** (`1.126 ms` vs `2.278 ms` NumPy/SciPy fallback) with `<= 1e-12` parity.

### 3. Single-Pass C11 Azimuthal Rim Extraction (`evidencescene.teleop`)
- Added `extract_rim_annulus_points` backed by C11 `extract_rim_annulus_fast` for extracting outer-perimeter circular aperture rim rings with linear quickselect quantiles (**`11.18x` faster**: `0.012 ms` vs `0.136 ms` NumPy fallback).

### 4. Zero-Copy Masked Planar Integration & `retain_camera_points=False`
- Added `PersistentPlanarSurfaceConfig.retain_camera_points: bool = True`. When disabled for streaming point-cloud pipelines that do not require historical `correct_pose()`, `integrate_points()` skips per-frame camera-frame back-transforms and passes `include_mask`/`exclude_mask` directly into C11 `planar_project_and_aggregate_fast` (backed by zero-page-fault `ThreadScratch` buffers), cutting masked 72k-point integration latency by **`1.79x`** (`0.799 ms` vs `1.430 ms`).

### 5. Complete Portable Runtime AVX-512/AVX2 Dispatch & Refreshed `kernel-v1` Leaderboard
- Upgraded `voxel_centroids` and `select_existing_points_shadow_fast` from compile-time `#ifdef` guards to portable `__attribute__((target("avx512f,avx512dq,avx512vl")))` / `__attribute__((target("avx2")))` runtime dispatch via `__builtin_cpu_supports`, so standard portable wheels automatically run AVX-512 on supported x86_64 CPUs.
- Refreshed `docs/performance_leaderboard.json` and rendered project pages (`docs/index.html`, `docs/demo/index.html`, `docs/benchmark/index.html`) from live `v2.8.0` C11/AVX-512 measurements:

| `kernel-v1` Workload | Baseline `04a1b6b` (ms) | `v2.8.0` C11/AVX-512 (ms) | Speed-up |
| :--- | ---: | ---: | ---: |
| `surface_snapshot_60k_budget_1500` | `20.478` | **`0.0047`** | **`4,357.02x`** |
| `surface_update_60k_points` | `72.995` | **`0.3337`** | **`218.74x`** |
| `voxel_centroids_60k_points` | `10.329` | **`0.3500`** | **`29.51x`** |
| `rle_box_centroid_480x640` | `0.092` | **`0.0024`** | **`38.33x`** |
| `surface_selection_60k_budget_1500` | `15.517` | **`0.0993`** | **`156.26x`** |
| `circle_consensus_400_points_256_hypotheses` | `4.453` | **`0.0292`** | **`152.50x`** |
| `circular_aperture_fit_400_points` | `4.348` | **`0.0255`** | **`170.51x`** |
| **Geometric Mean (`kernel-v1`)** | — | — | **`191.31x`** |

## Verification
- C11 extension builds with `0` warnings under `-Wall -O3 -std=c11 -fno-fast-math -ffp-contract=off`.
- **82 passed** with native C11/AVX-512 enabled (`pytest -q`).
- **81 passed, 1 skipped** with pure-Python fallback (`EVIDENCESCENE_DISABLE_NATIVE=1 pytest -q`).
- All public pages, leaderboard contracts, and privacy checks verified (`tests/test_performance_leaderboard.py`).
