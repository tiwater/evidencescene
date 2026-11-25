# EvidenceScene 2.3.0 — C11/SIMD native acceleration, dirty-tile meshing, & physics-weighted fusion

EvidenceScene 2.3.0 integrates the optional C11/SIMD deterministic computational
kernel extension (`evidencescene._native`) across the library, extends native
acceleration to measured and persistent planar surface meshing, adds `32×32`
dirty-tile incremental meshing, and introduces sensor-physics fusion weighting
($\cos\theta / z^2$ with specular highlight attenuation) by default.

## Highlights

- **Optional C11/SIMD native extension (`evidencescene._native` & `evidencescene._accel`)**:
  - Strict IEEE-754 determinism (`-O3 -std=c11 -fno-fast-math -ffp-contract=off`),
    zero NumPy C-ABI build dependency (PEP 3118 buffer protocol), slab buffer
    pool, content-addressed JIT fallback for source checkouts, and automatic
    pure-NumPy fallback (`EVIDENCESCENE_DISABLE_NATIVE=1`).
  - Accelerates all 7 core `kernel-v1` workloads (`RevisableSurfelMap` update &
    snapshot, voxel centroids, RLE algebra, circular aperture fit, shadow
    selection, Schur observability gate, 3D KD-tree metrics, and masked-depth
    backprojection).
  - Adds 8 new C11 surface/meshing kernels (`planar_mesh_finish_fast`,
    `planar_tile_quads_fast`, `planar_compact_triangles_fast`,
    `planar_cell_aggregate_fast`, `planar_ht_lookup_fast`,
    `planar_ht_rebuild_fast`, `planar_window_accumulate_fast`, and
    `planar_compact_obs_fast`).
- **`32×32` dirty-tile incremental meshing (`PersistentPlanarSurfaceReconstructor`)**:
  - Re-triangulates only tiles modified since the last snapshot (with 1-cell
    shared border ownership for seamless cross-tile quads) and incrementally
    maintains fused cell positions/colors in C11.
  - Localized wrist-camera updates with active sliding-window compaction drop
    from `35.35 ms/frame` to **`1.75 ms/frame`** (**20.2× speedup**); full
    21.6k-point table updates drop from `17.5 ms/frame` to **`5.84 ms/frame`**
    (**3.0× speedup**); single-frame planar baseline meshing drops from
    `2.89 ms` to **`1.53 ms`** (`p50`); replay `tracemalloc` peak memory drops
    28% (`26.22 MB → 18.82 MB`).
- **Sensor-physics fusion weighting (`weighting_mode="sensor_physics"`, default)**:
  - Weights per-sample depth & color updates by incidence cosine and inverse
    squared viewing distance ($\operatorname{clip}(\cos\theta / \max(z, 0.2)^2, 0.05, 8.0)$)
    when camera pose is supplied, while attenuating near-saturated specular
    glints (`specular_saturation_threshold=250`, `specular_rgb_weight_scale=0.25`).
  - Automatically falls back to uniform weighting when `frame_from_camera` is
    omitted or `weighting_mode="uniform"` is configured.
- **Circular moving-block bootstrap CIs (`evidencescene.metrics`)**:
  - Adds `BlockBootstrapCI`, `compute_moving_block_bootstrap_ci`, and
    `compute_paired_block_bootstrap_diff_ci` for finite-horizon uncertainty
    estimation on temporally correlated video sequences.

## Validation

- **73 standalone tests** pass with C11 native enabled AND with
  `EVIDENCESCENE_DISABLE_NATIVE=1` across Python 3.10, 3.11, and 3.12 CI,
  including bit-exact native vs. fallback parity, multi-tile boundary
  stitching, sensor-physics weighting, concurrency, and `setup.py build_ext`.
- Recorded 37-frame physical 3-camera RGB-D replay
  (`benchmarks/surface_mesh/recorded-persistent-planar-20261125-v230.json`):
  - Persistent final coverage `0.488 m²` (`8,020 V / 15,186 F`, `+41.8%` over
    best single view, `101.6%` retained during heavy occlusion, `4.3×` baseline
    during occlusion, zero foreground ghosts with and without `exclude_mask`).
  - Update p50/p95 `5.32 / 11.69 ms`, snapshot p50/p95 `3.19 / 3.88 ms`,
    single-frame baseline p50 `1.53 ms`, `tracemalloc` peak `18.82 MB`.

## Installation

Install the attached wheel, or from git:

```sh
pip install 'evidencescene @ git+https://github.com/tiwater/evidencescene.git@v2.3.0'
```

This is a GitHub release; no PyPI publication is implied.
