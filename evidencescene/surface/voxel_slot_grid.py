"""Bounded 3D voxel-indexed surfel slot grid for incremental scene streaming.

Maintains a fixed-capacity table of 3D voxel surfels (``max_slots <= 65535``)
with O(1) flat-grid deduplication, monotonic per-step observation stamping,
16-bit bounding-box coordinate quantization, RGB565 color packing, dirty-slot
delta extraction, and optional automatic retirement of unobserved slots.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from evidencescene._accel import NATIVE


def quantize_points_u16(
    pts_world_m: np.ndarray,
    *,
    bbox_min_m: Sequence[float] | np.ndarray = (-0.20, -0.65, 0.90),
    bbox_span_m: Sequence[float] | np.ndarray = (1.50, 1.30, 0.80),
) -> np.ndarray:
    """Quantize ``(N, 3)`` world coordinates into ``(N, 3)`` ``uint16``."""
    pts = np.asarray(pts_world_m)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.uint16)
    bmin = np.asarray(bbox_min_m, dtype=np.float64)
    bspan = np.asarray(bbox_span_m, dtype=np.float64)
    norm = (pts.astype(np.float64, copy=False) - bmin) / bspan
    clamped = np.clip(np.rint(norm * 65535.0), 0.0, 65535.0)
    return clamped.astype(np.uint16)


def pack_rgb565(rgb_u8: np.ndarray) -> np.ndarray:
    """Pack ``(N, 3)`` ``uint8`` RGB into ``(N,)`` ``uint16`` RGB565."""
    rgb = np.asarray(rgb_u8)
    if len(rgb) == 0:
        return np.empty((0,), dtype=np.uint16)
    r = (rgb[:, 0].astype(np.uint16) >> 3) & 0x1F
    g = (rgb[:, 1].astype(np.uint16) >> 2) & 0x3F
    b = (rgb[:, 2].astype(np.uint16) >> 3) & 0x1F
    return ((r << 11) | (g << 5) | b).astype(np.uint16)


@dataclass(frozen=True)
class VoxelSlotDelta:
    """Incremental geometry patch emitted by one :class:`BoundedVoxelSlotGrid` step."""

    active_slot_count: int
    upserted_records_u16: np.ndarray  # (M, 5) uint16: [slot, qx, qy, qz, rgb565]
    deleted_slots_u16: np.ndarray  # (D,) uint16


class BoundedVoxelSlotGrid:
    """Fixed-capacity 3D voxel-indexed surfel slot table with C11 delta emission."""

    def __init__(
        self,
        *,
        voxel_size_m: float = 0.008,
        max_slots: int = 62000,
        workspace_bounds_min: Sequence[float] = (0.18, -0.64, 0.96),
        workspace_bounds_max: Sequence[float] = (1.12, 0.64, 1.52),
        quant_bbox_min_m: Sequence[float] = (-0.20, -0.65, 0.90),
        quant_bbox_span_m: Sequence[float] = (1.50, 1.30, 0.80),
    ) -> None:
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be finite and positive")
        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or not (1 <= max_slots <= 65535):
            raise ValueError("max_slots must be an integer in [1, 65535]")

        self.voxel_size_m = float(voxel_size_m)
        self.max_slots = int(max_slots)
        self.ws_min = np.ascontiguousarray(workspace_bounds_min, dtype=np.float32)
        self.ws_max = np.ascontiguousarray(workspace_bounds_max, dtype=np.float32)
        self.quant_bbox_min_m = np.ascontiguousarray(quant_bbox_min_m, dtype=np.float64)
        self.quant_bbox_span_m = np.ascontiguousarray(quant_bbox_span_m, dtype=np.float64)
        if self.ws_min.shape != (3,) or self.ws_max.shape != (3,) or not np.all(self.ws_max > self.ws_min):
            raise ValueError("workspace_bounds_min/max must be 3-vectors with max > min")
        if (
            self.quant_bbox_min_m.shape != (3,)
            or self.quant_bbox_span_m.shape != (3,)
            or not np.all(self.quant_bbox_span_m > 0.0)
        ):
            raise ValueError("quant_bbox_min_m/span_m must be 3-vectors with positive span")

        spans = self.ws_max - self.ws_min
        self.nx = int(math.ceil(float(spans[0]) / self.voxel_size_m)) + 2
        self.ny = int(math.ceil(float(spans[1]) / self.voxel_size_m)) + 2
        self.nz = int(math.ceil(float(spans[2]) / self.voxel_size_m)) + 2

        self.grid_to_slot = np.full(self.nx * self.ny * self.nz, -1, dtype=np.int32)
        self.free_slots = np.arange(self.max_slots, dtype=np.int32)
        self.free_top = self.max_slots

        self.slot_alive = np.zeros(self.max_slots, dtype=np.bool_)
        self.slot_grid_idx = np.full(self.max_slots, -1, dtype=np.int32)
        self.slot_pos = np.zeros((self.max_slots, 3), dtype=np.float32)
        self.slot_rgb = np.zeros((self.max_slots, 3), dtype=np.float32)
        self.slot_log_odds = np.zeros(self.max_slots, dtype=np.int16)
        self.slot_sent_qpos = np.zeros((self.max_slots, 3), dtype=np.int32)
        self.slot_sent_rgb565 = np.zeros(self.max_slots, dtype=np.uint16)
        self.slot_never_sent = np.ones(self.max_slots, dtype=np.bool_)
        self._slot_seen_stamp = np.zeros(self.max_slots, dtype=np.int32)
        self._step_stamp = 0

    def reset(self) -> None:
        """Clear all active slots and restore full capacity."""
        self.grid_to_slot.fill(-1)
        self.free_slots[:] = np.arange(self.max_slots, dtype=np.int32)
        self.free_top = self.max_slots
        self.slot_alive.fill(False)
        self.slot_grid_idx.fill(-1)
        self.slot_log_odds.fill(0)
        self.slot_never_sent.fill(True)
        self._slot_seen_stamp.fill(0)
        self._step_stamp = 0

    def build_full_keyframe_records(self) -> np.ndarray:
        """Return ``(K, 5)`` ``uint16`` records for every currently alive slot."""
        alive_slots = np.flatnonzero(self.slot_alive).astype(np.uint16)
        if len(alive_slots) == 0:
            return np.empty((0, 5), dtype=np.uint16)
        pts = self.slot_pos[alive_slots]
        rgb = np.clip(np.rint(self.slot_rgb[alive_slots]), 0, 255).astype(np.uint8)
        qpos = quantize_points_u16(
            pts,
            bbox_min_m=self.quant_bbox_min_m,
            bbox_span_m=self.quant_bbox_span_m,
        )
        c565 = pack_rgb565(rgb)
        return np.column_stack([alive_slots, qpos, c565]).astype(np.uint16)

    def step(
        self,
        clouds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
        *,
        retire_unobserved: bool = True,
    ) -> VoxelSlotDelta:
        """Ingest one or more ``(pts_f32, rgb_u8, optional_skip_mask)`` clouds and emit delta."""
        self._step_stamp += 1
        if self._step_stamp >= 2_147_483_640:
            self._slot_seen_stamp.fill(0)
            self._step_stamp = 1

        if NATIVE is not None and hasattr(NATIVE, "voxel_slot_grid_step_fast"):
            prepared = []
            for pts_raw, rgb_raw, skip_raw in clouds:
                if len(pts_raw) == 0:
                    continue
                pts_c = (
                    pts_raw
                    if (isinstance(pts_raw, np.ndarray) and pts_raw.dtype == np.float32 and pts_raw.flags.c_contiguous)
                    else np.ascontiguousarray(pts_raw, dtype=np.float32)
                )
                rgb_c = (
                    rgb_raw
                    if (isinstance(rgb_raw, np.ndarray) and rgb_raw.dtype == np.uint8 and rgb_raw.flags.c_contiguous)
                    else np.ascontiguousarray(rgb_raw, dtype=np.uint8)
                )
                if skip_raw is None:
                    skip_c = None
                else:
                    skip_c = (
                        skip_raw
                        if (isinstance(skip_raw, np.ndarray) and skip_raw.dtype == np.bool_ and skip_raw.flags.c_contiguous)
                        else np.ascontiguousarray(skip_raw, dtype=np.bool_)
                    )
                prepared.append((pts_c, rgb_c, skip_c))

            res = NATIVE.voxel_slot_grid_step_fast(
                prepared,
                self.ws_min,
                self.ws_max,
                1.0 / self.voxel_size_m,
                self.nx,
                self.ny,
                self.nz,
                self.quant_bbox_min_m,
                self.quant_bbox_span_m,
                self.grid_to_slot,
                self.free_slots,
                int(self.free_top),
                self.slot_alive,
                self.slot_grid_idx,
                self.slot_pos,
                self.slot_rgb,
                self.slot_log_odds,
                self.slot_sent_qpos,
                self.slot_sent_rgb565,
                self.slot_never_sent,
                self._slot_seen_stamp,
                int(self._step_stamp),
                1 if retire_unobserved else 0,
            )
            if res is not NotImplemented:
                new_free_top, alive_cnt, upserted_u16, deleted_u16 = res
                self.free_top = int(new_free_top)
                return VoxelSlotDelta(
                    active_slot_count=int(alive_cnt),
                    upserted_records_u16=upserted_u16,
                    deleted_slots_u16=deleted_u16,
                )

        # Pure-NumPy deterministic fallback (first-point-wins per voxel across clouds).
        pts_parts: list[np.ndarray] = []
        rgb_parts: list[np.ndarray] = []
        for pts_raw, rgb_raw, skip_raw in clouds:
            if len(pts_raw) == 0:
                continue
            pts_f = np.asarray(pts_raw, dtype=np.float32)
            rgb_u = np.asarray(rgb_raw, dtype=np.uint8)
            if skip_raw is not None:
                keep = ~np.asarray(skip_raw, dtype=np.bool_)
                if not np.any(keep):
                    continue
                pts_f = pts_f[keep]
                rgb_u = rgb_u[keep]
            in_b = np.all((pts_f >= self.ws_min) & (pts_f <= self.ws_max), axis=1)
            if np.any(in_b):
                pts_parts.append(pts_f[in_b])
                rgb_parts.append(rgb_u[in_b])

        dirty_records: list[np.ndarray] = []
        if pts_parts:
            scene_pts = np.vstack(pts_parts) if len(pts_parts) > 1 else pts_parts[0]
            scene_rgb = np.vstack(rgb_parts) if len(rgb_parts) > 1 else rgb_parts[0]
            inv_v = 1.0 / self.voxel_size_m
            ix = np.clip(np.floor((scene_pts[:, 0] - self.ws_min[0]) * inv_v).astype(np.int32), 0, self.nx - 1)
            iy = np.clip(np.floor((scene_pts[:, 1] - self.ws_min[1]) * inv_v).astype(np.int32), 0, self.ny - 1)
            iz = np.clip(np.floor((scene_pts[:, 2] - self.ws_min[2]) * inv_v).astype(np.int32), 0, self.nz - 1)
            flat_idx = (ix * self.ny + iy) * self.nz + iz

            # Preserve exact first-seen point per voxel in cloud stream order.
            _, first_idx = np.unique(flat_idx, return_index=True)
            first_idx.sort()
            uniq_flat = flat_idx[first_idx]
            u_pts = scene_pts[first_idx]
            u_rgb_u8 = scene_rgb[first_idx]

            existing_slots = self.grid_to_slot[uniq_flat]
            is_new = existing_slots < 0
            is_old = ~is_new

            touched_list: list[np.ndarray] = []
            if np.any(is_old):
                old_s = existing_slots[is_old]
                self.slot_pos[old_s] = u_pts[is_old]
                self.slot_rgb[old_s] = u_rgb_u8[is_old].astype(np.float32)
                self.slot_log_odds[old_s] = np.minimum(7, self.slot_log_odds[old_s] + 2)
                self._slot_seen_stamp[old_s] = self._step_stamp
                touched_list.append(old_s)

            if np.any(is_new) and self.free_top > 0:
                new_flat = uniq_flat[is_new]
                new_pts = u_pts[is_new]
                new_rgb = u_rgb_u8[is_new]
                n_alloc = min(len(new_flat), self.free_top)
                # LIFO stack order matching C11 `free_slots[--free_top]`
                alloc_s = self.free_slots[self.free_top - n_alloc : self.free_top][::-1].copy()
                self.free_top -= n_alloc
                new_flat = new_flat[:n_alloc]
                self.grid_to_slot[new_flat] = alloc_s
                self.slot_alive[alloc_s] = True
                self.slot_grid_idx[alloc_s] = new_flat
                self.slot_pos[alloc_s] = new_pts[:n_alloc]
                self.slot_rgb[alloc_s] = new_rgb[:n_alloc].astype(np.float32)
                self.slot_log_odds[alloc_s] = 6
                self.slot_never_sent[alloc_s] = True
                self._slot_seen_stamp[alloc_s] = self._step_stamp
                touched_list.append(alloc_s)

            if touched_list:
                touched = np.concatenate(touched_list) if len(touched_list) > 1 else touched_list[0]
                q_cur = quantize_points_u16(
                    self.slot_pos[touched],
                    bbox_min_m=self.quant_bbox_min_m,
                    bbox_span_m=self.quant_bbox_span_m,
                ).astype(np.int32)
                rgb_cur_u8 = np.clip(np.rint(self.slot_rgb[touched]), 0, 255).astype(np.uint8)
                c565_cur = pack_rgb565(rgb_cur_u8)
                pos_diff = np.max(np.abs(q_cur - self.slot_sent_qpos[touched]), axis=1)
                changed = (
                    self.slot_never_sent[touched]
                    | (pos_diff > 0)
                    | (c565_cur != self.slot_sent_rgb565[touched])
                )
                if np.any(changed):
                    dirty = touched[changed]
                    self.slot_sent_qpos[dirty] = q_cur[changed]
                    self.slot_sent_rgb565[dirty] = c565_cur[changed]
                    self.slot_never_sent[dirty] = False
                    dirty_records.append(
                        np.column_stack(
                            [
                                dirty.astype(np.uint16),
                                q_cur[changed].astype(np.uint16),
                                c565_cur[changed],
                            ]
                        ).astype(np.uint16)
                    )

        if retire_unobserved:
            dead_slots = np.flatnonzero(
                self.slot_alive & (self._slot_seen_stamp != self._step_stamp)
            ).astype(np.int32)
            if len(dead_slots) > 0:
                self.grid_to_slot[self.slot_grid_idx[dead_slots]] = -1
                self.slot_alive[dead_slots] = False
                self.slot_grid_idx[dead_slots] = -1
                self.slot_log_odds[dead_slots] = 0
                self.slot_never_sent[dead_slots] = True
                self.free_slots[self.free_top : self.free_top + len(dead_slots)] = dead_slots
                self.free_top += len(dead_slots)
            deleted_u16 = dead_slots.astype(np.uint16)
        else:
            deleted_u16 = np.empty((0,), dtype=np.uint16)

        upserted_u16 = (
            dirty_records[0] if dirty_records else np.empty((0, 5), dtype=np.uint16)
        )
        return VoxelSlotDelta(
            active_slot_count=int(np.count_nonzero(self.slot_alive)),
            upserted_records_u16=upserted_u16,
            deleted_slots_u16=deleted_u16,
        )


__all__ = [
    "BoundedVoxelSlotGrid",
    "VoxelSlotDelta",
    "pack_rgb565",
    "quantize_points_u16",
]
