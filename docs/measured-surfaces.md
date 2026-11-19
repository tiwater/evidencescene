# Measured surface meshes (2.1)

EvidenceScene now reconstructs local triangle topology from measured surface
samples. This is a library capability, independent of host applications, robot models,
simulation coordinates, browser rendering, and transport schemas.

## Inputs and ownership

The caller supplies samples from its existing calibrated RGB-D or tracked surface
owner. All APIs preserve input vertex positions and RGB exactly, return their
original `source_indices`, and carry the caller's frame, observation ID, and
sensor timestamp. The APIs do not run a second pose estimator, table detector,
tracker, or persistent world map. They do not promote a shadow pose owner.

`reconstruct_planar_surface` accepts an already measured plane (`n @ p + d = 0`)
and its support/error evidence. It works for arbitrary orientations, including
vertical planes, rather than assuming a fixed table height or robot frame. Its
tangent grid is internal indexing only. It never projects vertices onto an ideal
plane or replaces measured colors with a constant material.

`reconstruct_organized_surface` accepts one camera's already registered and
backprojected samples with original pixel coordinates. It supports curved and
nonplanar surfaces using image adjacency without flattening them. It does not
backproject again or merge unrelated cameras. For dense images, use `pixel_step`
to select a bounded regular lattice; e.g. 4 for a 640×360 source.

`reconstruct_rgbd_surface` is the camera-local convenience API: it reuses the
library's existing masked-depth backprojection for already registered RGB-D and
rectified intrinsics, then applies the organized topology builder. Its source
indices refer to row-major flattened input pixels. It requires no robot/world
transform and does not infer one.

## Usage

```python
from evidencescene.surface import reconstruct_rgbd_surface

mesh = reconstruct_rgbd_surface(
    depth_m, rgb_u8, camera_matrix,
    coordinate_frame="camera/head", timestamp_ns=frame.timestamp_ns,
    source_id="recording/head/42", pixel_step=4,
)
```


```python
from evidencescene.surface import SurfaceMeshConfig, reconstruct_planar_surface

mesh = reconstruct_planar_surface(
    points_m, colors_rgb_u8,
    plane_normal=owner_plane.normal,
    plane_offset_m=owner_plane.offset_m,
    plane_support_count=owner_plane.inlier_count,
    plane_rmse_m=owner_plane.rmse_m,
    coordinate_frame="camera/head",  # same frame as points and plane
    timestamp_ns=frame.timestamp_ns,
    source_id="recording/head/42",
    config=SurfaceMeshConfig(cell_size_m=0.008, maximum_edge_m=0.024),
)
if mesh.ready:
    render(mesh.vertices_m, mesh.colors_rgb_u8, mesh.triangles)
else:
    show_unavailable(mesh.reason)
```

For curved surfaces:

```python
from evidencescene.surface import reconstruct_organized_surface

mesh = reconstruct_organized_surface(
    points_m, colors_rgb_u8, pixels_uv=original_pixel_coordinates,
    image_shape=(360, 640), pixel_step=4,
    coordinate_frame="camera/head", timestamp_ns=frame.timestamp_ns,
    source_id="recording/head/42",
)
```

## Guarantees and limits

- Four occupied neighboring cells/pixels are required for a quad. No convex hull,
  boundary extrapolation, morphological dilation, or hole filling.
- Triangle edges must meet the metric distance limit. Planar meshes additionally
  gate plane residual and normal displacement. Raised objects and depth jumps
  beyond those limits do not join the plane.
- Nonfinite samples are excluded; source indices still refer to the original
  array. Malformed inputs raise `ValueError`. Missing/weak plane evidence,
  insufficient support, or exceeded budgets return an empty mesh with a reason.
- Dense grid and vertex allocations are bounded. Output arrays do not alias the
  input and are read-only. No temporal state is retained.
- Triangle interiors are **local interpolation**, not independently measured
  geometry. Gaps smaller than the sampling cell/stride cannot be resolved by
  topology alone. Meshes are not necessarily watertight or safe for collision
  checking. Curved mesh normals follow pixel winding, not guaranteed outward
  orientation. Clients must handle freshness using the sensor timestamp.
- This release adds topology, normals, area/edge metrics, and provenance. It does
  **not** claim improved pose accuracy, new unseen-surface recovery, photorealistic
  texture reconstruction, or a new production pose-owner cutover.

## Validation

Unit tests cover rotated planes, curved surfaces, exact sample lineage, missing
strips, outliers, discontinuities, allocation bounds, and malformed inputs. The
release scorecard under `benchmarks/surface_mesh/` additionally uses recorded
multi-camera RGB-D. Recorded validation is not a claim of long-duration stability
or generalization to all materials and environments.
