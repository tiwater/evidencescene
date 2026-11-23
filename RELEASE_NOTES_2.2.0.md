# EvidenceScene 2.2.0 — persistent measured static planar surfaces

EvidenceScene now supports bounded, multi-frame persistent static planar surface
reconstruction across calibrated RGB-D streams and pre-registered common-frame
point clouds, preserving previously measured regions across viewpoint changes
and temporary occlusions without filling unobserved gaps or committing transient
foreground ghosts.

## New library APIs

- `PersistentPlanarSurfaceReconstructor`: integrates successive calibrated RGB-D
  frames (`integrate_rgbd`) or common-frame samples (`integrate_points`) in any
  plane orientation, reusing the 2.1.0 masked-depth backprojection and planar
  topology gates (`SurfaceMeshConfig`, `_quads`, `_finish`). All public methods
  and properties are synchronized with an internal `threading.RLock`.
- `PersistentPlanarSurfaceConfig`: metric limits, temporal corroboration
  (`min_observations_to_commit`, `tentative_ttl_updates`), free-space carving
  bounds, fail-closed change policies (`frame_change_policy`,
  `calibration_change_policy`, `plane_change_policy`), and long-session memory
  caps (`max_cells`, `max_cell_weight`, `max_retained_observations`,
  `max_seen_observation_ids`, `max_provenance_observations`).
- `PersistentSurfaceSnapshot` and `ObservationProvenance`: bounded, versioned,
  read-only mesh snapshots (`vertices_m`, `colors_rgb_u8`, `triangles`,
  `face_normals`, `vertex_observation_counts`, `vertex_last_observed_ns`) with
  explicitly separated monotonic snapshot age (`snapshot_created_monotonic_ns`,
  `snapshot_age_ns()`) and sensor-clock observation freshness
  (`latest_observation_timestamp_ns`, `newest_contributing_timestamp_ns`,
  `median_contributing_timestamp_ns`, `oldest_contributing_timestamp_ns`,
  `observation_freshness_ns()`).
- `SurfaceUpdateResult`: explicit status reporting for idempotent duplicates
  (`duplicate_ignored`), exact reversible recent pose corrections
  (`correct_pose`, `correct_poses`, `pose_corrected`, and fail-closed
  `pose_correction_out_of_window`), calibration/frame/plane changes, and
  explicit `reset()`.

## Validation

- 61 standalone library tests pass (Python 3.10, 3.11, 3.12), covering
  multi-viewpoint coverage growth, rotated/vertical planes, occlusion retention,
  open missing-data gaps, transient foreground rejection and free-space carving,
  snapshot age vs. observation freshness, duplicates, pose revisions,
  calibration/frame/plane changes, resets, long-session memory bounds,
  thread-safe concurrent updates/snapshots, and malformed inputs.
- Recorded 3-camera physical RGB-D replay (37 frames across `chest`, moving
  `left_arm`, and moving `right_arm` cameras with temporary bimanual table
  occlusion and moving foreground objects; 8 mm cell / 24 mm max-edge defaults):
  - Best single-view baseline: `0.344 m²` (5,855 V / 10,389 F).
  - During heavy occlusion (`008_occlusion_chest_0272`): per-observation
    baseline drops to `0.090 m²` (1,848 V / 2,695 F), while persistent fusion
    retains `0.385 m²` (6,413 V / 12,020 F; 101.6% of pre-occlusion area,
    4.3× baseline).
  - Final multi-view fused mesh: `0.488 m²` (8,020 V / 15,182 F; +41.8% over
    best single view, adding 1,121 newly measured out-of-initial-footprint
    vertices while preserving 870 open boundary edges).
  - Geometric accuracy: final plane residual RMSE `3.25 mm`, max `7.79 mm`
    (`<= 8.0 mm`, zero foreground ghosts, including in an unmasked-foreground
    ablation with `exclude_mask=None`); against a 25,359-point held-out
    measured reference cloud (in-footprint), 3D RMSE `4.48 mm`, 3D median
    `2.37 mm`, normal-axis RMSE `2.33 mm`, 100% reference coverage within 20 mm.
  - Local replay latency & footprint: update p50/p95 `6.40 / 11.07 ms`,
    snapshot p50/p95 `4.46 / 4.81 ms`, final snapshot arrays `859 KB`
    (`318 KB` binary PLY), reconstructor estimated memory `13.59 MB`
    (`26.22 MB` `tracemalloc` peak).

See `docs/measured-surfaces.md`, `benchmarks/surface_mesh/recorded-persistent-planar-20261123.json`,
and the inspectable PLY artifacts under `benchmarks/surface_mesh/artifacts/` for
contracts, scorecards, and reproducible validation commands. Raw recordings are
not distributed; scorecards include exported-input SHA-256 hashes.

## Scope and installation

Triangle interiors are bounded local interpolation, not independently measured
geometry. Meshes preserve open gaps and are not watertight or guaranteed safe
for collision checking. This release focuses on persistent planar static work
surfaces and does not claim curved multi-view volumetric fusion, watertight
unseen geometry, or cross-environment generalization. Uncommitted local
acceleration experiments are not included in this release.

Install the attached universal wheel, or:

```sh
pip install 'evidencescene @ git+https://github.com/tiwater/evidencescene.git@v2.2.0'
```

This is a GitHub release; no PyPI publication is implied.
