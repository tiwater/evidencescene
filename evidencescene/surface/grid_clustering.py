"""Single-pass 2D height-grid foreground clustering with vertical air-gap decoupling.

Separates low tabletop/graspable objects from overhanging tall equipment structures
using a bounded 2D occupancy grid, 8-connected component labeling, and linear
quickselect quantiles — backed by C11 ``cluster_height_grid_2d_fast`` with an exact
NumPy/SciPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label

from evidencescene._accel import NATIVE


@dataclass(frozen=True)
class GridCluster2D:
    """One 8-connected foreground cluster with robust height and bounding quantiles."""

    point_indices: np.ndarray  # (K,) int32 indices into input points_m
    layer: int  # 0 = lower/graspable layer, 1 = elevated/tall layer
    h_p5: float
    h_p97: float
    xy_min_p3: tuple[float, float]
    xy_max_p97: tuple[float, float]
    z_top_p985: float


def cluster_height_grid_2d(
    points_m: np.ndarray,
    height_above_plane_m: np.ndarray,
    *,
    exclude_mask: np.ndarray | None = None,
    x_bounds: Sequence[float] = (0.22, 0.96),
    y_bounds: Sequence[float] = (-0.63, 0.63),
    height_bounds: Sequence[float] = (0.014, 0.45),
    grid_origin_xy: Sequence[float] = (0.20, -0.65),
    grid_span_xy: Sequence[float] = (0.80, 1.30),
    cell_size_m: float = 0.020,
    min_cluster_points: int = 42,
    air_gap_mid_bounds: Sequence[float] = (0.142, 0.225),
    air_gap_high_min: float = 0.225,
    layer0_max_height: float = 0.168,
    layer0_max_abs_y: float = 0.43,
    layer0_max_x: float = 0.65,
) -> list[GridCluster2D]:
    """Cluster 3D foreground points into lower (layer 0) and tall (layer 1) 2D components."""
    pts = np.asarray(points_m)
    h_arr = np.asarray(height_above_plane_m)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_m must be an Nx3 array")
    if h_arr.ndim != 1 or len(h_arr) != len(pts):
        raise ValueError("height_above_plane_m must be a 1-D array matching points_m")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    if len(pts) < int(min_cluster_points):
        return []

    exc: np.ndarray | None = None
    if exclude_mask is not None:
        exc = np.asarray(exclude_mask, dtype=np.bool_)
        if exc.shape != (len(pts),):
            raise ValueError("exclude_mask must be a 1-D boolean array matching points_m")

    if NATIVE is not None and hasattr(NATIVE, "cluster_height_grid_2d_fast"):
        pts_c = (
            pts
            if (pts.flags.c_contiguous and pts.dtype in (np.float32, np.float64))
            else np.ascontiguousarray(pts, dtype=np.float32)
        )
        h_c = (
            h_arr
            if (h_arr.flags.c_contiguous and h_arr.dtype in (np.float32, np.float64))
            else np.ascontiguousarray(h_arr, dtype=np.float32)
        )
        exc_c = (
            None
            if exc is None
            else (exc if exc.flags.c_contiguous else np.ascontiguousarray(exc, dtype=np.bool_))
        )
        raw = NATIVE.cluster_height_grid_2d_fast(
            pts_c,
            h_c,
            exc_c,
            float(x_bounds[0]),
            float(x_bounds[1]),
            float(y_bounds[0]),
            float(y_bounds[1]),
            float(height_bounds[0]),
            float(height_bounds[1]),
            float(grid_origin_xy[0]),
            float(grid_origin_xy[1]),
            float(grid_span_xy[0]),
            float(grid_span_xy[1]),
            float(cell_size_m),
            int(min_cluster_points),
            float(air_gap_mid_bounds[0]),
            float(air_gap_mid_bounds[1]),
            float(air_gap_high_min),
            float(layer0_max_height),
            float(layer0_max_abs_y),
            float(layer0_max_x),
        )
        if raw is not NotImplemented:
            return [
                GridCluster2D(
                    point_indices=idx_arr,
                    layer=int(layer_id),
                    h_p5=float(hp5),
                    h_p97=float(hp97),
                    xy_min_p3=(float(xlo), float(ylo)),
                    xy_max_p97=(float(xhi), float(yhi)),
                    z_top_p985=float(ztop),
                )
                for (idx_arr, layer_id, hp5, hp97, xlo, ylo, xhi, yhi, ztop) in raw
            ]

    # Exact NumPy / SciPy fallback (float64 arithmetic matches C11 double precision)
    pts64 = pts.astype(np.float64, copy=False)
    h64 = h_arr.astype(np.float64, copy=False)
    fg_mask = (
        np.isfinite(pts64[:, 2])
        & (h64 >= float(height_bounds[0]))
        & (h64 <= float(height_bounds[1]))
        & (pts64[:, 0] >= float(x_bounds[0]))
        & (pts64[:, 0] <= float(x_bounds[1]))
        & (pts64[:, 1] >= float(y_bounds[0]))
        & (pts64[:, 1] <= float(y_bounds[1]))
    )
    if exc is not None:
        fg_mask &= ~exc
    fg_indices = np.flatnonzero(fg_mask).astype(np.int32)
    if len(fg_indices) < int(min_cluster_points):
        return []

    fg_pts = pts64[fg_indices]
    fg_h = h64[fg_indices]
    nx_c = int(math.ceil(float(grid_span_xy[0]) / float(cell_size_m))) + 2
    ny_c = int(math.ceil(float(grid_span_xy[1]) / float(cell_size_m))) + 2
    gx = np.clip(np.floor((fg_pts[:, 0] - float(grid_origin_xy[0])) / float(cell_size_m)).astype(np.int32), 0, nx_c - 1)
    gy = np.clip(np.floor((fg_pts[:, 1] - float(grid_origin_xy[1])) / float(cell_size_m)).astype(np.int32), 0, ny_c - 1)

    mid_occ = np.zeros((nx_c, ny_c), dtype=bool)
    hi_occ = np.zeros((nx_c, ny_c), dtype=bool)
    in_mid = (fg_h >= float(air_gap_mid_bounds[0])) & (fg_h <= float(air_gap_mid_bounds[1]))
    in_hi = fg_h > float(air_gap_high_min)
    if np.any(in_mid):
        mid_occ[gx[in_mid], gy[in_mid]] = True
    if np.any(in_hi):
        hi_occ[gx[in_hi], gy[in_hi]] = True
    st = np.ones((3, 3), dtype=bool)
    tall_col_seed = mid_occ & binary_dilation(hi_occ, structure=st)
    tall_equip_grid = binary_dilation(tall_col_seed, structure=st)
    is_layer1 = (
        tall_equip_grid[gx, gy]
        | (fg_h > float(layer0_max_height))
        | (np.abs(fg_pts[:, 1]) > float(layer0_max_abs_y))
        | (fg_pts[:, 0] > float(layer0_max_x))
    )

    out: list[GridCluster2D] = []
    min_pts = int(min_cluster_points)
    st_i = np.ones((3, 3), dtype=np.int32)
    for layer_mask, layer_id in ((~is_layer1, 0), (is_layer1, 1)):
        l_local = np.flatnonzero(layer_mask)
        if len(l_local) < min_pts:
            continue
        occ_l = np.zeros((nx_c, ny_c), dtype=bool)
        occ_l[gx[l_local], gy[l_local]] = True
        lbl_2d, n_cc = ndimage_label(occ_l, structure=st_i)
        pt_lbl = lbl_2d[gx[l_local], gy[l_local]]
        for cc_i in range(1, int(n_cc) + 1):
            sub_l = l_local[pt_lbl == cc_i]
            if len(sub_l) < min_pts:
                continue
            g_idx = fg_indices[sub_l]
            c_pts = pts64[g_idx]
            c_h = h64[g_idx]
            hp5, hp97 = np.percentile(c_h, (5.0, 97.0))
            xy_lo, xy_hi = np.percentile(c_pts[:, :2], (3.0, 97.0), axis=0)
            z_top = float(np.percentile(c_pts[:, 2], 98.5))
            out.append(
                GridCluster2D(
                    point_indices=g_idx,
                    layer=layer_id,
                    h_p5=float(hp5),
                    h_p97=float(hp97),
                    xy_min_p3=(float(xy_lo[0]), float(xy_lo[1])),
                    xy_max_p97=(float(xy_hi[0]), float(xy_hi[1])),
                    z_top_p985=z_top,
                )
            )
    return out


__all__ = ["GridCluster2D", "cluster_height_grid_2d"]
