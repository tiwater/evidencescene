# EvidenceScene v2.6.0

## Summary

EvidenceScene `v2.6.0` introduces a single-pass C11 hash-table surface snapshot mesher, a 3D cylindrical vessel axis & tilt estimator (`fit_cylinder_axis_3d`), an $O(K \log K)$ sparse-coordinate RLE mask encoder (`encode_mask_rle_from_coords`), and C11 single-pass `ESTM` vertex quantization—delivering up to **10× faster** persistent mesh snapshots and **>20× faster** vessel body estimation while eliminating dense mask and `float64` copy overhead.

## Highlights

### 1. Single-Pass C11 Hash-Table Planar Mesher & Zero-Copy `float32` Path (`evidencescene.surface.persistent_static_surface`)
- Added C11 kernel `planar_mesh_snapshot_ht_fast`: probes 3 neighbor cells directly in the open-addressing slot hash table, applies exact edge/normal-step/winding gates, and compacts vertices/colors/triangles/normals in a single GIL-released pass (eliminating per-tile Python set iteration and `np.fromiter` overhead).
- Added zero-copy `float32` support and mask-free fast-path in `PersistentPlanarSurfaceReconstructor.integrate_points`, avoiding unnecessary `float32 -> float64` copies before C11 `planar_project_and_aggregate_fast`.

### 2. 3D Cylindrical Vessel Axis & Tilt Feature Extractor (`evidencescene.teleop.task_features`)
- Added `MeasuredCylinderAxis3D` and `fit_cylinder_axis_3d(...)` backed by C11 kernel `fit_cylinder_axis_fast` (re-exported from `evidencescene.teleop`).
- Complements `fit_circular_aperture_3d` under top-rim occlusion by fitting azimuthally-binned outer-perimeter circles across upper and lower vessel bands via analytical 2×2 IRLS Gauss-Newton normal equations, supporting arbitrary `reference_up` orientations.

### 3. Sparse-Coordinate O(K log K) RLE Mask Encoder (`evidencescene.primitives.image_mask`)
- Added `encode_mask_rle_from_coords(rows, cols, height=..., width=...)` backed by C11 kernel `encode_coords_rle_fast` (re-exported from `evidencescene.primitives`).
- Emits bit-exact canonical row-major RLE dictionaries directly from sparse `(row, col)` pixel indices without allocating or scanning dense `(H, W)` boolean images.

### 4. C11 Single-Pass `ESTM` Vertex Quantizer (`evidencescene.surface.mesh_codec`)
- Added C11 kernel `estm_quantize_vertices_fast` wired into `encode_surface_mesh_binary` for single-pass bounding-box extraction and `uint16` quantization with bit-exact parity against the NumPy fallback.

## Verification

- All 80 standalone unit & parity tests pass with both native C11 enabled (`-Wall` zero compiler warnings) and pure-NumPy fallback (`EVIDENCESCENE_DISABLE_NATIVE=1`).
