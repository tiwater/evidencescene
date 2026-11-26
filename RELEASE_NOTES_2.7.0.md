# EvidenceScene v2.7.0 Release Notes

## Highlights

EvidenceScene `v2.7.0` unlocks **portable runtime x86_64 SIMD dispatch (`AVX-512` / `AVX2` / scalar)** in standard wheels, introduces an **L2-cache direct 2D grid aggregator** eliminating hash/sort bottlenecks in persistent planar fusion, upgrades sparse coordinate RLE to **O(K) LSD radix sort**, exposes direct **`carve_free_space()`** APIs for point-cloud pipelines, and adds **`BoundedVoxelSlotGrid`** for single-pass C11 incremental 3D surfel streaming.

### 1. Portable Runtime x86_64 SIMD Dispatch (`AVX-512` / `AVX2` / Scalar)
- Compiled into portable wheels without `-march=native`: detects CPU ISA once at module load via `__builtin_cpu_supports` (`evidencescene._accel.NATIVE.simd_isa()` returns `"avx512"`, `"avx2"`, or `"scalar"`).
- Accelerates 3D kNN RGB transfer (`transfer_rgb_knn_3d`), 3D nearest-neighbor Hausdorff queries (`nn_min_dists_3d`), and 2D circle RANSAC consensus (`circle_consensus_2d`) with 8-wide AVX-512 and 4-wide AVX2 leaf kernels while preserving strict IEEE-754 (`-fno-fast-math -ffp-contract=off`) parity with the pure-NumPy reference path.
- `transfer_measured_rgb_to_prior` now avoids redundant `uint8 -> float64` color copies and duplicate Python finiteness pre-scans on the native path.

### 2. L2-Cache Direct 2D Grid Planar Projection & Aggregation (`planar_project_and_aggregate_fast`)
- Replaces per-frame hash-table probing and 72-byte struct `qsort` with a two-pass L2-cache direct 2D grid fast path whenever the inlier bounding box fits within `65,536` cells (`<= 256 KiB`), emitting cell centroids, RGB, and physics weights in exact ascending packed-key order with zero hashing and zero sorting.
- Large sparse grids (`> 65,536` cells) sort compact 16-byte `(key, idx)` pairs instead of 72-byte structs.

### 3. O(K) Radix-Sorted Sparse Coordinate RLE (`encode_coords_rle_fast`)
- Replaces generic `qsort` with 2-pass (`<= 4M` pixels) and 3-pass (`<= 4G` pixels) 11-bit LSD radix sort (and inline insertion sort for `K <= 64`), maintaining bit-exact parity with dense `encode_mask_rle`.

### 4. Public `carve_free_space()` APIs & Zero-Allocation Unmasked Carving
- Added thread-safe `carve_free_space(depth_m, camera_matrix, *, frame_from_camera, include_mask=None, exclude_mask=None) -> int` on both `PersistentPlanarSurfaceReconstructor` and `PersistentMultiPlaneSurfaceReconstructor`.
- `planar_free_space_carve_fast` accepts `selected_mask=None` directly to skip full-frame boolean mask allocations when carving unmasked depth maps.

### 5. Bounded 3D Voxel Slot Delta Grid (`evidencescene.surface.BoundedVoxelSlotGrid`)
- New generic bounded (`max_slots <= 65535`) 3D voxel-indexed surfel slot table (`BoundedVoxelSlotGrid`, `VoxelSlotDelta`, `quantize_points_u16`, `pack_rgb565`) backed by the single-pass GIL-released C11 kernel `voxel_slot_grid_step_fast`.
- Ingests multiple `(pts_f32, rgb_u8, optional_skip_mask)` point clouds without intermediate masking or concatenation copies, deduplicates voxels in O(1) via monotonic observation stamps, quantizes updated slots to 16-bit coordinates + RGB565, emits compact `(M, 5) uint16` upserts and `(D,) uint16` retirements, and matches the pure-NumPy fallback bit-for-bit.

## Verification
- Native C11 (`-Wall` zero compiler warnings) and pure-NumPy fallback (`EVIDENCESCENE_DISABLE_NATIVE=1`) test suites both pass with exact parity across all v2.7.0 kernels and APIs.
- GitHub Actions CI matrix passes across Python 3.10, 3.11, and 3.12 (native ON, fallback OFF, CLI demo, benchmark, and wheel/sdist build).
