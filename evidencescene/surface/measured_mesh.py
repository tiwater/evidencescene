"""Measured-sample surface reconstruction without a pose or world-state owner.

Builds local topology from an existing measured plane or organized RGB-D sample
adjacency. Never fits a plane, invents a pose, extrapolates a boundary, or fills
missing observations. Triangle interiors are bounded interpolation, not new
measurements. Source indices always index the caller's input sample arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class SurfaceMeshConfig:
    """Metric limits and allocation bounds; independent of any robot or scene."""

    cell_size_m: float = 0.008
    maximum_edge_m: float = 0.024
    maximum_plane_residual_m: float = 0.005
    maximum_normal_step_m: float = 0.005
    minimum_support: int = 80
    maximum_vertices: int = 40_000
    maximum_grid_cells: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("cell_size_m", "maximum_edge_m", "maximum_plane_residual_m", "maximum_normal_step_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_support", "maximum_vertices", "maximum_grid_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MeasuredSurfaceMesh:
    """Topology plus exact measured vertex/RGB lineage in the caller's frame.

    ``ready`` means locally supported topology exists, not that the surface is
    watertight or safe for control. ``source_indices`` refer to the input arrays,
    including when nonfinite points were filtered. No confidence is fabricated.
    """

    ready: bool
    reason: str
    coordinate_frame: str
    timestamp_ns: int
    source_id: str
    vertices_m: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    colors_rgb_u8: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    source_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    face_normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    support_count: int = 0
    surface_area_m2: float = 0.0
    maximum_edge_m: float = 0.0


def _validate_samples(points_m, colors_rgb_u8, coordinate_frame, timestamp_ns, source_id):
    points = np.asarray(points_m, dtype=np.float64)
    colors = np.asarray(colors_rgb_u8)
    if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
        raise ValueError("points and colors must be matching Nx3 arrays")
    if colors.dtype != np.uint8:
        raise ValueError("colors_rgb_u8 must have uint8 dtype")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must name the existing input frame")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must name the input observation")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    return points, colors


def _quads(grid):
    corners = np.stack((grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=-1)
    q = corners[np.all(corners >= 0, axis=-1)]
    return np.concatenate((q[:, [0, 1, 2]], q[:, [1, 3, 2]]))


def _finish(points, colors, ids, faces, config, metadata, *, plane_normal=None):
    if not len(faces):
        return MeasuredSurfaceMesh(False, "no_connected_measured_cells", **metadata)
    vertices = points[ids]
    tri = vertices[faces]
    edges = tri - np.roll(tri, 1, axis=1)
    edge_lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    keep = (edge_lengths.max(axis=1) <= config.maximum_edge_m) & (area2 > 1e-12)
    if plane_normal is not None:
        keep &= (np.abs(edges @ plane_normal).max(axis=1) <= config.maximum_normal_step_m)
        keep &= (cross @ plane_normal) > 1e-12
    faces = faces[keep]
    if not len(faces):
        return MeasuredSurfaceMesh(False, "no_valid_local_triangles", **metadata)
    used, inverse = np.unique(faces, return_inverse=True)
    if len(used) > config.maximum_vertices:
        return MeasuredSurfaceMesh(False, "surface_vertex_budget_exceeded", **metadata)
    arrays = (vertices[used], colors[ids[used]], inverse.reshape(-1, 3).astype(np.int32),
              ids[used], cross[keep] / area2[keep, None])
    for array in arrays:
        array.setflags(write=False)
    return MeasuredSurfaceMesh(True, "measured_vertices_local_interpolation", **metadata,
                               vertices_m=arrays[0], colors_rgb_u8=arrays[1], triangles=arrays[2],
                               source_indices=arrays[3], face_normals=arrays[4], support_count=len(ids),
                               surface_area_m2=float(area2[keep].sum() / 2),
                               maximum_edge_m=float(edge_lengths[keep].max()))


def reconstruct_planar_surface(
    points_m, colors_rgb_u8, *, plane_normal, plane_offset_m: float,
    plane_support_count: int, plane_rmse_m: float, coordinate_frame: str,
    timestamp_ns: int, source_id: str, config: SurfaceMeshConfig | None = None,
) -> MeasuredSurfaceMesh:
    """Mesh an existing measured plane of ANY orientation, preserving sample positions.

    The supplied unit normal and offset describe ``n @ p + offset = 0`` in the
    input frame. A deterministic tangent basis is used only to index local cells;
    it is not published as a new coordinate frame. Four occupied neighbor cells
    are required per quad. Unknown cells and depth discontinuities remain open.
    Bad plane evidence returns an empty result; malformed inputs raise ValueError.
    """
    config = config or SurfaceMeshConfig()
    points, colors = _validate_samples(points_m, colors_rgb_u8, coordinate_frame, timestamp_ns, source_id)
    metadata = dict(coordinate_frame=coordinate_frame, timestamp_ns=int(timestamp_ns), source_id=source_id)
    normal = np.asarray(plane_normal, dtype=np.float64)
    if (normal.shape != (3,) or not np.isfinite(normal).all() or
            not np.isclose(np.linalg.norm(normal), 1, atol=1e-6, rtol=0) or
            not np.isfinite([plane_offset_m, plane_rmse_m]).all() or plane_rmse_m < 0 or
            isinstance(plane_support_count, bool) or not isinstance(plane_support_count, (int, np.integer)) or
            plane_support_count < config.minimum_support or plane_rmse_m > config.maximum_plane_residual_m):
        return MeasuredSurfaceMesh(False, "no_measured_plane", **metadata)
    ids = np.flatnonzero(np.isfinite(points).all(axis=1) &
                         (np.abs(points @ normal + plane_offset_m) <= config.maximum_plane_residual_m))
    if len(ids) < config.minimum_support:
        return MeasuredSurfaceMesh(False, "insufficient_current_support", **metadata)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = axis - normal * np.dot(axis, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    # Recenter in tangent space before quantization to prevent integer overflow
    # for observations whose existing frame happens to have a large translation.
    uv = points[ids] @ np.column_stack((u, v))
    # A sub-nanocell tolerance avoids losing exact lattice neighbors to roundoff.
    scaled = np.floor((uv - uv.min(axis=0)) / config.cell_size_m + 1e-9)
    if not np.isfinite(scaled).all() or np.any(scaled >= config.maximum_grid_cells):
        return MeasuredSurfaceMesh(False, "surface_grid_budget_exceeded", **metadata)
    cells = scaled.astype(np.int64)
    extent = cells.max(axis=0) + 1
    if int(extent[0]) * int(extent[1]) > config.maximum_grid_cells:
        return MeasuredSurfaceMesh(False, "surface_grid_budget_exceeded", **metadata)
    _, first = np.unique(cells[:, 0] * extent[1] + cells[:, 1], return_index=True)
    ids, cells = ids[first], cells[first]
    if len(ids) > config.maximum_vertices:
        return MeasuredSurfaceMesh(False, "surface_vertex_budget_exceeded", **metadata)
    grid = np.full(tuple(extent), -1, dtype=np.int32)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(ids))
    return _finish(points, colors, ids, _quads(grid), config, metadata, plane_normal=normal)


def reconstruct_organized_surface(
    points_m, colors_rgb_u8, *, pixels_uv, image_shape: tuple[int, int],
    coordinate_frame: str, timestamp_ns: int, source_id: str,
    pixel_step: int = 1, config: SurfaceMeshConfig | None = None,
) -> MeasuredSurfaceMesh:
    """Mesh one camera's existing calibrated/filtered RGB-D samples, including curves.

    Pixels index the original color-registered depth image. No backprojection,
    transform, pose inference or multi-camera merging is performed. Samples may
    be in camera-local or an existing aligned frame; source indices are retained.
    Only four directly adjacent valid samples form a quad. Holes and rejected
    3-D edges are never bridged. ``pixel_step`` selects a regular sampling lattice
    and should be set to the producer's sampling step (or an integer multiple).
    Triangle winding follows image adjacency, not guaranteed outward orientation.
    """
    config = config or SurfaceMeshConfig()
    points, colors = _validate_samples(points_m, colors_rgb_u8, coordinate_frame, timestamp_ns, source_id)
    metadata = dict(coordinate_frame=coordinate_frame, timestamp_ns=int(timestamp_ns), source_id=source_id)
    pixels = np.asarray(pixels_uv)
    if pixels.shape != (len(points), 2) or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixels_uv must be an integer Nx2 array")
    if (len(image_shape) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 for x in image_shape)
            or isinstance(pixel_step, bool) or not isinstance(pixel_step, (int, np.integer)) or pixel_step <= 0):
        raise ValueError("image_shape and pixel_step must be positive integers")
    h, w = image_shape
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= w) or np.any(pixels[:, 1] >= h):
        raise ValueError("pixels must lie inside the source image")
    # Bound the allocation before building indices, even for a sparse input.
    shape = ((w - 1) // pixel_step + 1, (h - 1) // pixel_step + 1)
    if shape[0] * shape[1] > config.maximum_grid_cells:
        return MeasuredSurfaceMesh(False, "surface_grid_budget_exceeded", **metadata)
    ids = np.flatnonzero(np.isfinite(points).all(axis=1) & np.all(pixels % pixel_step == 0, axis=1))
    if len(ids) < config.minimum_support:
        return MeasuredSurfaceMesh(False, "insufficient_current_support", **metadata)
    if len(ids) > config.maximum_vertices:
        return MeasuredSurfaceMesh(False, "surface_vertex_budget_exceeded", **metadata)
    cells = pixels[ids] // pixel_step
    keys = cells[:, 0] * shape[1] + cells[:, 1]
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate source pixels: provide a single camera observation")
    grid = np.full(shape, -1, dtype=np.int32)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(ids))
    return _finish(points, colors, ids, _quads(grid), config, metadata)


def reconstruct_rgbd_surface(
    depth_m, color_rgb_u8, camera_matrix, *, coordinate_frame: str,
    timestamp_ns: int, source_id: str, mask=None, pixel_step: int = 4,
    minimum_depth_m: float = 0.15, maximum_depth_m: float = 3.0,
    config: SurfaceMeshConfig | None = None,
) -> MeasuredSurfaceMesh:
    """Reconstruct a camera-local mesh from already registered RGB-D.

    Uses the library's existing masked-depth backprojection primitive. RGB and
    depth must share rectified intrinsics and pixel registration. No extrinsic
    transform is inferred. ``source_indices`` index the original flattened HxW
    depth/RGB images (row-major), not a filtered point array. All other meshing
    and missing-data contracts match ``reconstruct_organized_surface``.
    """
    from dataclasses import replace
    from evidencescene.primitives.geometry import _backproject_masked_depth

    config = config or SurfaceMeshConfig()
    depth = np.asarray(depth_m)
    color = np.asarray(color_rgb_u8)
    K = np.asarray(camera_matrix, dtype=np.float64)
    if depth.ndim != 2 or color.shape != (*depth.shape, 3) or color.dtype != np.uint8:
        raise ValueError("registered depth HxW and uint8 RGB HxWx3 must match")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0 or
            not np.allclose(K[2], [0,0,1]) or K[0,1] != 0 or K[1,0] != 0):
        raise ValueError("camera_matrix must contain valid rectified pinhole intrinsics")
    if (not np.isfinite([minimum_depth_m, maximum_depth_m]).all() or
            not 0 < minimum_depth_m < maximum_depth_m):
        raise ValueError("depth range must be finite, positive and ordered")
    if isinstance(pixel_step, bool) or not isinstance(pixel_step, (int, np.integer)) or pixel_step <= 0:
        raise ValueError("pixel_step must be a positive integer")
    selected = np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask)
    if selected.shape != depth.shape or selected.dtype != bool:
        raise ValueError("mask must be a boolean HxW image matching depth")
    h, w = depth.shape
    if h == 0 or w == 0:
        raise ValueError("depth image must not have empty dimensions")
    # Reuse validation/allocation bounds before the existing backprojection call.
    _validate_samples(np.empty((0,3)), np.empty((0,3), dtype=np.uint8), coordinate_frame, timestamp_ns, source_id)
    if ((h-1)//pixel_step+1) * ((w-1)//pixel_step+1) > config.maximum_grid_cells:
        return MeasuredSurfaceMesh(False, "surface_grid_budget_exceeded", coordinate_frame, int(timestamp_ns), source_id)
    sampled = depth[::pixel_step, ::pixel_step]
    valid = (selected[::pixel_step, ::pixel_step] & np.isfinite(sampled) &
             (sampled >= minimum_depth_m) & (sampled <= maximum_depth_m))
    vv, uu = np.nonzero(valid)
    vv, uu = vv * pixel_step, uu * pixel_step
    if len(vv) > config.maximum_vertices:
        return MeasuredSurfaceMesh(False, "surface_vertex_budget_exceeded", coordinate_frame, int(timestamp_ns), source_id)
    points = _backproject_masked_depth(depth, selected, K, pixel_stride=pixel_step,
                minimum_depth_m=minimum_depth_m, maximum_depth_m=maximum_depth_m)
    mesh = reconstruct_organized_surface(points, color[vv,uu], pixels_uv=np.column_stack((uu,vv)),
        image_shape=(h,w), pixel_step=pixel_step, config=config, coordinate_frame=coordinate_frame,
        timestamp_ns=timestamp_ns, source_id=source_id)
    indices = (vv * w + uu)[mesh.source_indices]
    indices.setflags(write=False)
    return replace(mesh, source_indices=indices)
