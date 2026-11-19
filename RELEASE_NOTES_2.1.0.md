# EvidenceScene 2.1.0 — measured surface reconstruction

EvidenceScene can now turn measured samples into bounded local triangle surfaces,
with explicit observation provenance, metric validity gates, and missing-data holes.

## New library APIs

- `reconstruct_planar_surface`: measured planes in arbitrary orientations; exact
  input vertices and RGB, existing plane support/error gates, no fixed table frame.
- `reconstruct_organized_surface`: curved RGB-D samples linked by pixel adjacency,
  rejecting missing neighbors and excessive 3-D edge lengths.
- `reconstruct_rgbd_surface`: camera-local reconstruction from registered RGB-D,
  reusing the existing masked-depth backprojection primitive.
- `MeasuredSurfaceMesh` and `SurfaceMeshConfig`: source indices, timestamp/frame,
  triangle normals, surface area, edge metrics, and bounded allocations.

## Validation

- 50 standalone library tests pass, including curved/rotated surfaces, depth gaps,
  exact RGB/sample lineage, empty evidence, and allocation guards.
- 20 recorded multi-camera registered-sample frames: all planar meshes ready;
  exact input vertex and RGB lineage preserved.
- 80 recorded camera RGB-D frames: all camera meshes ready; source pixels, RGB,
  and measured depth verified exactly.
- Local short-replay planar topology p50/p95: 8.78 / 9.89 ms; direct camera RGB-D
  p50/p95: 10.05 / 11.98 ms. These are different workloads, not a speedup claim.
- The downstream integration delegates topology to this library and excludes the
  display mesh from control scene snapshots.

See `docs/measured-surfaces.md` and `benchmarks/surface_mesh/` for API contracts,
input schemas, recorded scorecards and reproducible validation commands.
Raw recordings are not distributed; scorecards include exported-input hashes.

## Scope and installation

Triangle interiors are local interpolation, not additional measured evidence.
This release does not claim improved pose accuracy, watertight unseen geometry,
photorealistic textures, or a new pose-owner cutover. Recorded tests use simulation
RGB-D, not physical-store data. Uncommitted local acceleration experiments are
not included in this release.

Install the attached universal wheel, or:

```sh
pip install 'evidencescene @ git+https://github.com/tiwater/evidencescene.git@v2.1.0'
```

This is a GitHub release; no PyPI publication is implied.
