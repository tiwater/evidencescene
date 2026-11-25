# EvidenceScene v2.5.0

## Summary

EvidenceScene `v2.5.0` adds four new C11/SIMD computational kernels covering 3D kNN RGB transfer, fused planar projection/physics-weighted aggregation, camera free-space carving, and deterministic RANSAC plane consensus; introduces automatic multi-plane discovery (`extract_measured_planes`); and graduates the 4-byte-aligned `ESTM` compact binary mesh codec into the standalone `evidencescene.surface` package.

## Highlights

### 1. Four New C11/SIMD Native Kernels (`evidencescene/_native.c`)
- **`planar_project_and_aggregate_fast`**: Fuses plane residual gating, tangent `(u, v)` projection, grazing-angle & $\cos\theta/z^2$ physics weighting, specular highlight attenuation, cell key packing, and hash-table aggregation into one zero-intermediate-allocation C11 pass (supporting both `float32` and `float64` point clouds directly).
- **`planar_free_space_carve_fast`**: Single-pass GIL-released camera projection and 3×3 neighborhood free-space visibility check for `PersistentPlanarSurfaceReconstructor._apply_free_space_carving`.
- **`transfer_rgb_knn_3d`**: Reuses the AVX2/AVX-512 median-split 3D KD-tree for exact $k$-nearest neighbor inverse-distance RGB blending in `transfer_measured_rgb_to_prior`, eliminating per-object SciPy `cKDTree` construction overhead.
- **`fit_plane_ransac_fast`**: Deterministic triplet-seeded RANSAC consensus with exact early-exit pruning for `fast_fit_shared_plane` and sequential multi-plane peeling.
- Cleaned up unused `HT_TOMB` static constant so `evidencescene/_native.c` compiles cleanly under `-Wall`.

### 2. Deterministic Multi-Plane Discovery (`extract_measured_planes`)
- Added `extract_measured_planes(...)` in `evidencescene.surface.persistent_multi_plane` (re-exported from `evidencescene.surface`).
- Sequentially peels up to `max_planes` dominant measured planes using `fast_fit_shared_plane`, refines normals and RMSE on full inlier support, enforces optional inter-plane angle diversity (`min_inter_plane_angle_deg`), and emits `PlaneObservationSpec` entries directly consumable by `PersistentMultiPlaneSurfaceReconstructor`.

### 3. Standalone Compact Binary Mesh Wire Codec (`evidencescene.surface.mesh_codec`)
- Added `BINARY_MESH_MAGIC` (`b"ESTM"`), `encode_surface_mesh_binary`, and `decode_surface_mesh_binary` (re-exported from `evidencescene.surface`).
- Encodes `MeasuredSurfaceMesh`, `PersistentSurfaceSnapshot`, `MultiPlaneSurfaceSnapshot` (including `vertex_plane_indices` and `triangle_plane_indices`), or compatible mappings into 4-byte-aligned `uint16` quantized vertices, `uint8` RGB, and `uint16`/`uint32` triangles (~3.7× smaller raw / ~2.2× smaller gzipped than JSON at `0.05 mm` quantization precision).

## Verification

- All 79 unit & parity tests pass with both native C11 enabled and pure-NumPy fallback (`EVIDENCESCENE_DISABLE_NATIVE=1`), verifying exact native/fallback parity across all four new C11 kernels, multi-plane discovery, and binary codec round-trips.
