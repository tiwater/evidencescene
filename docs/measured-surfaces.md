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

## Persistent static planar surfaces

`PersistentPlanarSurfaceReconstructor` extends the per-observation planar mesh
primitives to integrate calibrated RGB-D frames (`integrate_rgbd`) or
pre-registered common-frame point clouds (`integrate_points`) across changing
viewpoints and temporary occlusions.

- **Inputs**: `observation_id`, sensor `timestamp_ns`, `coordinate_frame`,
  camera intrinsics + `frame_from_camera` SE(3) transform, measured plane
  evidence, optional `camera_id`, and caller-provided `include_mask` /
  `exclude_mask` for excluding moving objects and robot links.
- **Fusion & anti-ghosting**: Uses a bounded sparse planar tangent-grid cell
  store with multi-observation temporal corroboration (`min_observations_to_commit`),
  unconfirmed tentative cell expiry (`tentative_ttl_updates`), plane residual
  and grazing-angle gating, and free-space carving when later views clearly see
  behind a phantom cell (while preserving occluded static cells where foreground
  depth lies in front of the plane).
- **Snapshot versioning & freshness**: `snapshot()` returns a bounded, immutable
  `PersistentSurfaceSnapshot` with monotonic `version`, reset `epoch`, read-only
  mesh arrays (`vertices_m`, `colors_rgb_u8`, `triangles`, `face_normals`),
  per-vertex `vertex_observation_counts` and `vertex_last_observed_ns`, and
  bounded `contributing_observations` provenance. Snapshot age
  (`snapshot_created_monotonic_ns`, `snapshot_age_ns()`) and sensor-clock
  observation freshness (`latest_observation_timestamp_ns`,
  `newest_contributing_timestamp_ns`, `oldest_contributing_timestamp_ns`,
  `observation_freshness_ns()`) are reported separately.
- **Lifecycle, concurrency, & bounds**: All public methods and properties are
  synchronized with an internal `threading.RLock`, allowing a perception loop to
  integrate observations while another thread reads immutable snapshots. Duplicate
  `observation_id` calls with unchanged pose are idempotent no-ops
  (`duplicate_ignored`). Camera pose corrections within the bounded sliding
  revision window (`max_retained_observations`) subtract and re-integrate that
  observation (`correct_pose`, `correct_poses`); older compacted observations
  fail closed (`pose_correction_out_of_window`). Frame, per-camera calibration,
  and plane changes default to fail-closed `"reject"` (configurable to
  `"reset"`), and `reset()` clears all state while bumping `epoch` and
  `version`. Cell count (`max_cells`), archived weight (`max_cell_weight`),
  seen-ID tracking, and provenance history are all bounded for long sessions.

```python
from evidencescene.surface import (
    PersistentPlanarSurfaceConfig,
    PersistentPlanarSurfaceReconstructor,
    SurfaceMeshConfig,
)

recon = PersistentPlanarSurfaceReconstructor(
    config=PersistentPlanarSurfaceConfig(
        mesh=SurfaceMeshConfig(cell_size_m=0.008, maximum_edge_m=0.024),
        min_observations_to_commit=2,
        pixel_step=4,
    )
)
update = recon.integrate_rgbd(
    depth_m, rgb_u8, camera_matrix,
    frame_from_camera=frame_from_camera,
    plane_normal=owner_plane.normal,
    plane_offset_m=owner_plane.offset_m,
    plane_support_count=owner_plane.inlier_count,
    plane_rmse_m=owner_plane.rmse_m,
    coordinate_frame="table",
    timestamp_ns=frame.timestamp_ns,
    observation_id="head/0042",
    camera_id="head",
    exclude_mask=foreground_and_robot_mask,
)
snap = recon.snapshot(reference_timestamp_ns=frame.timestamp_ns)
if snap.ready:
    age_ns = snap.snapshot_age_ns()
    freshness = snap.observation_freshness_ns(frame.timestamp_ns)
```

## Validation

Unit tests cover rotated planes, curved surfaces, exact sample lineage, missing
strips, outliers, discontinuities, allocation bounds, malformed inputs,
multi-view persistent fusion, occlusion retention, anti-ghosting/free-space
carving, duplicates, pose revisions, calibration/frame changes, resets, and
long-session memory bounds. The release scorecards under
`benchmarks/surface_mesh/` additionally validate recorded multi-camera RGB-D
sequences, including persistent vs. per-observation baseline replays and
inspectable PLY mesh artifacts under `benchmarks/surface_mesh/artifacts/`.
Recorded validation is not a claim of long-duration stability or generalization
to all materials and environments.
