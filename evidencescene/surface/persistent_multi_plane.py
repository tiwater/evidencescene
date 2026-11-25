"""Bounded multi-plane persistent static surface reconstruction.

Composes up to ``max_planes`` bounded :class:`PersistentPlanarSurfaceReconstructor`
instances under one thread-safe coordinator with deterministic point-to-plane
ownership arbitration, unified versioned mesh snapshots, synchronized pose
corrections, and global or per-plane resets.

Typical use cases include multi-surface robotic workspaces such as a horizontal
countertop paired with a vertical backsplash, shelving tiers, or tilted fixtures,
without sacrificing single-plane memory bounds, free-space carving, or C11/SIMD
dirty-tile meshing performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import math
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from evidencescene.primitives.geometry import _backproject_masked_depth
from evidencescene.surface.measured_mesh import _validate_samples
from evidencescene.surface.persistent_static_surface import (
    PersistentPlanarSurfaceConfig,
    PersistentPlanarSurfaceReconstructor,
    PersistentSurfaceSnapshot,
    SurfaceUpdateResult,
    _freeze_copy,
    _validate_intrinsics,
    _validate_nonempty_str,
    _validate_plane_evidence,
    _validate_se3,
    _validate_timestamp_ns,
)


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class PlaneObservationSpec:
    """Measured plane evidence associated with a stable ``plane_id``."""

    plane_id: str
    plane_normal: np.ndarray
    plane_offset_m: float
    plane_support_count: int
    plane_rmse_m: float

    def __post_init__(self) -> None:
        _validate_nonempty_str(self.plane_id, "plane_id")


@dataclass(frozen=True)
class MultiPlaneSurfaceConfig:
    """Configuration for bounded multi-plane persistent surface reconstruction."""

    plane_config: PersistentPlanarSurfaceConfig = field(
        default_factory=PersistentPlanarSurfaceConfig
    )
    max_planes: int = 8
    seam_tie_tolerance_m: float = 1e-6
    enable_cross_plane_exclusion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.plane_config, PersistentPlanarSurfaceConfig):
            raise ValueError("plane_config must be a PersistentPlanarSurfaceConfig")
        if (
            isinstance(self.max_planes, bool)
            or not isinstance(self.max_planes, (int, np.integer))
            or int(self.max_planes) < 1
        ):
            raise ValueError("max_planes must be a positive integer")
        if (
            isinstance(self.seam_tie_tolerance_m, bool)
            or not math.isfinite(self.seam_tie_tolerance_m)
            or self.seam_tie_tolerance_m < 0.0
        ):
            raise ValueError("seam_tie_tolerance_m must be finite and nonnegative")
        if not isinstance(self.enable_cross_plane_exclusion, bool):
            raise ValueError("enable_cross_plane_exclusion must be a boolean")


@dataclass(frozen=True)
class MultiPlaneUpdateResult:
    """Outcome of a multi-plane integration, pose revision, or reset operation."""

    accepted: bool
    status: str
    version: int
    epoch: int
    observation_id: str | None = None
    timestamp_ns: int | None = None
    plane_results: Mapping[str, SurfaceUpdateResult] = field(default_factory=dict)
    active_plane_ids: tuple[str, ...] = ()
    total_inlier_sample_count: int = 0
    total_committed_cell_count: int = 0
    total_tentative_cell_count: int = 0
    reset_triggered: bool = False


@dataclass(frozen=True)
class MultiPlaneSurfaceSnapshot:
    """Immutable unified multi-plane mesh snapshot with per-plane provenance."""

    ready: bool
    reason: str
    version: int
    epoch: int
    coordinate_frame: str
    plane_ids: tuple[str, ...] = ()
    plane_snapshots: Mapping[str, PersistentSurfaceSnapshot] = field(default_factory=dict)
    vertices_m: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.float64))
    )
    colors_rgb_u8: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.uint8))
    )
    triangles: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.int32))
    )
    face_normals: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.float64))
    )
    vertex_plane_indices: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int32))
    )
    triangle_plane_indices: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int32))
    )
    vertex_observation_counts: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int32))
    )
    vertex_last_observed_ns: np.ndarray = field(
        default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int64))
    )
    support_count: int = 0
    surface_area_m2: float = 0.0
    maximum_edge_m: float = 0.0
    estimated_memory_bytes: int = 0
    snapshot_created_monotonic_ns: int = 0
    map_updated_monotonic_ns: int = 0
    reference_timestamp_ns: int | None = None
    latest_observation_timestamp_ns: int | None = None
    newest_contributing_timestamp_ns: int | None = None
    oldest_contributing_timestamp_ns: int | None = None

    def snapshot_age_ns(self, now_monotonic_ns: int | None = None) -> int:
        """Wall/monotonic age of this materialized snapshot object in nanoseconds."""
        now = (
            time.monotonic_ns()
            if now_monotonic_ns is None
            else _validate_timestamp_ns(now_monotonic_ns, "now_monotonic_ns")
        )
        return max(0, now - self.snapshot_created_monotonic_ns)

    def observation_freshness_ns(
        self, current_sensor_timestamp_ns: int
    ) -> dict[str, int | None]:
        """Sensor-clock lag of the latest stream frame and active multi-plane geometry."""
        now_ts = _validate_timestamp_ns(
            current_sensor_timestamp_ns, "current_sensor_timestamp_ns"
        )

        def _lag(ts: int | None) -> int | None:
            return None if ts is None else now_ts - ts

        return {
            "latest_observation_lag_ns": _lag(self.latest_observation_timestamp_ns),
            "newest_contributing_lag_ns": _lag(self.newest_contributing_timestamp_ns),
            "oldest_contributing_lag_ns": _lag(self.oldest_contributing_timestamp_ns),
        }


def _arbitrate_point_ownership(
    pts_frame: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    *,
    max_residual_m: float,
    seam_tie_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically assign each 3D sample to at most one winning plane.

    Planes must be ordered deterministically (sorted by ``plane_id``). Each point
    is assigned to the plane with minimum orthogonal distance within
    ``max_residual_m``. Ties within ``seam_tie_tolerance_m`` resolve to the
    lowest plane index (lexicographically smallest ``plane_id``).

    Returns ``(winner_idx, inlier_matrix)`` where ``winner_idx[k] == -1`` if point
    ``k`` exceeds ``max_residual_m`` for all planes, and ``inlier_matrix[k, p]``
    indicates whether point ``k`` lies within ``max_residual_m`` of plane ``p``.
    """
    n_pts = len(pts_frame)
    n_planes = len(normals)
    if n_pts == 0 or n_planes == 0:
        return np.empty(n_pts, dtype=np.int32), np.zeros((n_pts, n_planes), dtype=bool)
    residuals = np.abs(pts_frame @ normals.T + offsets[None, :])
    inlier_matrix = residuals <= max_residual_m
    any_inlier = np.any(inlier_matrix, axis=1)
    min_res = np.min(residuals, axis=1)
    within_tie = inlier_matrix & (residuals <= (min_res[:, None] + seam_tie_tolerance_m))
    winner = np.where(any_inlier, np.argmax(within_tie, axis=1), -1).astype(np.int32)
    return winner, inlier_matrix


class PersistentMultiPlaneSurfaceReconstructor:
    """Thread-safe coordinator for up to ``max_planes`` persistent planar surfaces.

    Each plane is backed by an independent :class:`PersistentPlanarSurfaceReconstructor`
    preserving its bounded cell storage, physics-weighted fusion, free-space carving,
    dirty-tile incremental meshing, and pose revision window. When multiple planes
    are present in the same frame, overlapping seam samples are deterministically
    assigned to the minimum-residual plane (lexicographic ``plane_id`` tie-break)
    so corners and counter-backsplash junctions never double-count geometry.
    """

    def __init__(
        self,
        *,
        coordinate_frame: str | None = None,
        config: MultiPlaneSurfaceConfig | None = None,
    ) -> None:
        self._config = config or MultiPlaneSurfaceConfig()
        self._lock = threading.RLock()
        self._coordinate_frame: str | None = (
            None
            if coordinate_frame is None
            else _validate_nonempty_str(coordinate_frame, "coordinate_frame")
        )
        self._planes: dict[str, PersistentPlanarSurfaceReconstructor] = {}
        self._version = 0
        self._epoch = 0
        self._map_updated_monotonic_ns = time.monotonic_ns()
        self._latest_observation_timestamp_ns: int | None = None
        self._cached_snapshot_key: tuple[int, int] | None = None
        self._cached_snapshot: MultiPlaneSurfaceSnapshot | None = None

    @property
    def config(self) -> MultiPlaneSurfaceConfig:
        return self._config

    @property
    @_locked
    def version(self) -> int:
        return self._version

    @property
    @_locked
    def epoch(self) -> int:
        return self._epoch

    @property
    @_locked
    def coordinate_frame(self) -> str | None:
        return self._coordinate_frame

    @property
    @_locked
    def plane_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._planes.keys()))

    def _invalidate_cache(self) -> None:
        self._cached_snapshot_key = None
        self._cached_snapshot = None

    def _sum_cell_counts(self) -> tuple[int, int]:
        committed = 0
        tentative = 0
        for recon in self._planes.values():
            c, t = recon._active_counts()
            committed += c
            tentative += t
        return committed, tentative

    @_locked
    def register_plane(
        self,
        plane_id: str,
        *,
        plane_normal=None,
        plane_offset_m: float | None = None,
    ) -> bool:
        """Explicitly register a plane slot before integrating observations.

        Returns ``True`` if registered (or already present with compatible config),
        or ``False`` if ``max_planes`` would be exceeded.
        """
        pid = _validate_nonempty_str(plane_id, "plane_id")
        if pid in self._planes:
            return True
        if len(self._planes) >= self._config.max_planes:
            return False
        recon = PersistentPlanarSurfaceReconstructor(
            coordinate_frame=self._coordinate_frame,
            plane_normal=plane_normal,
            plane_offset_m=plane_offset_m,
            config=self._config.plane_config,
        )
        self._planes[pid] = recon
        self._version += 1
        self._map_updated_monotonic_ns = time.monotonic_ns()
        self._invalidate_cache()
        return True

    @_locked
    def remove_plane(self, plane_id: str) -> bool:
        """Remove a plane from the registry. Returns ``True`` if removed."""
        pid = _validate_nonempty_str(plane_id, "plane_id")
        if pid not in self._planes:
            return False
        del self._planes[pid]
        self._version += 1
        self._map_updated_monotonic_ns = time.monotonic_ns()
        self._invalidate_cache()
        return True

    @_locked
    def reset(
        self,
        *,
        plane_id: str | None = None,
        reason: str = "manual_reset",
    ) -> MultiPlaneUpdateResult:
        """Reset all planes (bumping ``epoch``) or a single named plane."""
        _validate_nonempty_str(reason, "reason")
        if plane_id is not None:
            pid = _validate_nonempty_str(plane_id, "plane_id")
            recon = self._planes.get(pid)
            if recon is None:
                committed, tentative = self._sum_cell_counts()
                return MultiPlaneUpdateResult(
                    accepted=False,
                    status="unknown_plane",
                    version=self._version,
                    epoch=self._epoch,
                    active_plane_ids=tuple(sorted(self._planes.keys())),
                    total_committed_cell_count=committed,
                    total_tentative_cell_count=tentative,
                )
            sub_res = recon.reset(reason=reason)
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._invalidate_cache()
            committed, tentative = self._sum_cell_counts()
            return MultiPlaneUpdateResult(
                accepted=True,
                status="plane_reset",
                version=self._version,
                epoch=self._epoch,
                plane_results={pid: sub_res},
                active_plane_ids=tuple(sorted(self._planes.keys())),
                total_committed_cell_count=committed,
                total_tentative_cell_count=tentative,
                reset_triggered=True,
            )

        self._planes.clear()
        self._coordinate_frame = None
        self._latest_observation_timestamp_ns = None
        self._epoch += 1
        self._version += 1
        self._map_updated_monotonic_ns = time.monotonic_ns()
        self._invalidate_cache()
        return MultiPlaneUpdateResult(
            accepted=True,
            status="reset",
            version=self._version,
            epoch=self._epoch,
            active_plane_ids=(),
            reset_triggered=True,
        )

    def _validate_planes_arg(
        self, planes: Sequence[PlaneObservationSpec]
    ) -> list[PlaneObservationSpec]:
        if isinstance(planes, (str, bytes)) or not isinstance(planes, Sequence):
            raise ValueError("planes must be a sequence of PlaneObservationSpec")
        seen: set[str] = set()
        out: list[PlaneObservationSpec] = []
        for item in planes:
            if not isinstance(item, PlaneObservationSpec):
                raise ValueError("each entry in planes must be a PlaneObservationSpec")
            if item.plane_id in seen:
                raise ValueError(f"duplicate plane_id {item.plane_id!r} in planes")
            seen.add(item.plane_id)
            out.append(item)
        out.sort(key=lambda spec: spec.plane_id)
        return out

    def _handle_frame_change(
        self,
        coordinate_frame: str,
        *,
        observation_id: str,
        timestamp_ns: int,
    ) -> tuple[bool, MultiPlaneUpdateResult | None]:
        """Return ``(reset_triggered, early_reject_result)`` for frame transitions."""
        if self._coordinate_frame is not None and coordinate_frame != self._coordinate_frame:
            if self._config.plane_config.frame_change_policy == "reject":
                committed, tentative = self._sum_cell_counts()
                return (
                    False,
                    MultiPlaneUpdateResult(
                        accepted=False,
                        status="coordinate_frame_changed",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                        active_plane_ids=tuple(sorted(self._planes.keys())),
                        total_committed_cell_count=committed,
                        total_tentative_cell_count=tentative,
                    ),
                )
            self._planes.clear()
            self._coordinate_frame = coordinate_frame
            self._epoch += 1
            self._invalidate_cache()
            return True, None
        return False, None

    @_locked
    def integrate_rgbd(
        self,
        depth_m,
        color_rgb_u8,
        camera_matrix,
        *,
        frame_from_camera,
        planes: Sequence[PlaneObservationSpec],
        coordinate_frame: str,
        timestamp_ns: int,
        observation_id: str,
        camera_id: str = "default",
        include_mask=None,
        exclude_mask=None,
        pixel_step: int | None = None,
        observation_weight: float = 1.0,
    ) -> MultiPlaneUpdateResult:
        """Integrate one calibrated RGB-D observation across multiple measured planes."""
        coordinate_frame = _validate_nonempty_str(coordinate_frame, "coordinate_frame")
        observation_id = _validate_nonempty_str(observation_id, "observation_id")
        camera_id = _validate_nonempty_str(camera_id, "camera_id")
        timestamp_ns = _validate_timestamp_ns(timestamp_ns)
        if (
            isinstance(observation_weight, bool)
            or not math.isfinite(observation_weight)
            or observation_weight <= 0.0
        ):
            raise ValueError("observation_weight must be finite and positive")

        depth = np.asarray(depth_m)
        if not np.issubdtype(depth.dtype, np.floating):
            depth = depth.astype(np.float64)
        color = np.asarray(color_rgb_u8)
        K = _validate_intrinsics(camera_matrix)
        T_fc = _validate_se3(frame_from_camera, "frame_from_camera")
        if depth.ndim != 2 or color.shape != (*depth.shape, 3) or color.dtype != np.uint8:
            raise ValueError("registered depth HxW and uint8 RGB HxWx3 must match")
        h, w = depth.shape
        if h == 0 or w == 0:
            raise ValueError("depth image must not have empty dimensions")

        step = self._config.plane_config.pixel_step if pixel_step is None else pixel_step
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) <= 0:
            raise ValueError("pixel_step must be a positive integer")
        step = int(step)

        selected = np.ones((h, w), dtype=bool)
        if include_mask is not None:
            inc = np.asarray(include_mask)
            if inc.shape != (h, w) or inc.dtype != bool:
                raise ValueError("include_mask must be a boolean HxW image matching depth")
            selected &= inc
        base_exc = np.zeros((h, w), dtype=bool)
        if exclude_mask is not None:
            exc = np.asarray(exclude_mask)
            if exc.shape != (h, w) or exc.dtype != bool:
                raise ValueError("exclude_mask must be a boolean HxW image matching depth")
            base_exc |= exc
            selected &= ~exc

        ordered_specs = self._validate_planes_arg(planes)
        frame_reset, early_reject = self._handle_frame_change(
            coordinate_frame, observation_id=observation_id, timestamp_ns=timestamp_ns
        )
        if early_reject is not None:
            return early_reject

        # Validate plane evidence and enforce max_planes cap deterministically.
        valid_specs: list[tuple[PlaneObservationSpec, np.ndarray, float]] = []
        plane_results: dict[str, SurfaceUpdateResult] = {}
        new_planes_budget = self._config.max_planes - len(self._planes)
        for spec in ordered_specs:
            ev = _validate_plane_evidence(
                spec.plane_normal,
                spec.plane_offset_m,
                spec.plane_support_count,
                spec.plane_rmse_m,
                self._config.plane_config.mesh,
            )
            if ev is None:
                plane_results[spec.plane_id] = SurfaceUpdateResult(
                    accepted=False,
                    status="invalid_plane",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                )
                continue
            if spec.plane_id not in self._planes:
                if new_planes_budget <= 0:
                    plane_results[spec.plane_id] = SurfaceUpdateResult(
                        accepted=False,
                        status="max_planes_exceeded",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                    )
                    continue
                new_planes_budget -= 1
            valid_specs.append((spec, ev[0], ev[1]))

        if not valid_specs:
            if frame_reset:
                self._version += 1
                self._map_updated_monotonic_ns = time.monotonic_ns()
            committed, tentative = self._sum_cell_counts()
            status = (
                "max_planes_exceeded"
                if any(r.status == "max_planes_exceeded" for r in plane_results.values())
                else "no_valid_planes"
            )
            return MultiPlaneUpdateResult(
                accepted=False,
                status=status,
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                plane_results=plane_results,
                active_plane_ids=tuple(sorted(self._planes.keys())),
                total_committed_cell_count=committed,
                total_tentative_cell_count=tentative,
                reset_triggered=frame_reset,
            )

        # Deterministic seam arbitration across valid planes.
        per_plane_exc: list[np.ndarray | None] = [
            exclude_mask for _ in range(len(valid_specs))
        ]
        if self._config.enable_cross_plane_exclusion and len(valid_specs) >= 2:
            sampled_depth = depth[::step, ::step]
            valid_sub = (
                selected[::step, ::step]
                & np.isfinite(sampled_depth)
                & (sampled_depth >= self._config.plane_config.minimum_depth_m)
                & (sampled_depth <= self._config.plane_config.maximum_depth_m)
            )
            vv_sub, uu_sub = np.nonzero(valid_sub)
            if len(vv_sub) > 0:
                vv = vv_sub * step
                uu = uu_sub * step
                pts_cam = _backproject_masked_depth(
                    depth,
                    selected,
                    K,
                    pixel_stride=step,
                    minimum_depth_m=self._config.plane_config.minimum_depth_m,
                    maximum_depth_m=self._config.plane_config.maximum_depth_m,
                )
                pts_frame = pts_cam @ T_fc[:3, :3].T + T_fc[:3, 3][None, :]
                normals_mat = np.stack([item[1] for item in valid_specs], axis=0)
                offsets_vec = np.array([item[2] for item in valid_specs], dtype=np.float64)
                winner, inlier_mat = _arbitrate_point_ownership(
                    pts_frame,
                    normals_mat,
                    offsets_vec,
                    max_residual_m=self._config.plane_config.mesh.maximum_plane_residual_m,
                    seam_tie_tolerance_m=self._config.seam_tie_tolerance_m,
                )
                for p_idx in range(len(valid_specs)):
                    lost_seam = inlier_mat[:, p_idx] & (winner != p_idx)
                    if np.any(lost_seam):
                        exc_p = base_exc.copy()
                        exc_p[vv[lost_seam], uu[lost_seam]] = True
                        per_plane_exc[p_idx] = exc_p

        any_accepted = False
        any_sub_reset = frame_reset
        total_inliers = 0
        for p_idx, (spec, normal, offset_m) in enumerate(valid_specs):
            recon = self._planes.get(spec.plane_id)
            if recon is None:
                recon = PersistentPlanarSurfaceReconstructor(
                    coordinate_frame=self._coordinate_frame or coordinate_frame,
                    config=self._config.plane_config,
                )
                self._planes[spec.plane_id] = recon
            res = recon.integrate_rgbd(
                depth,
                color,
                K,
                frame_from_camera=T_fc,
                plane_normal=normal,
                plane_offset_m=offset_m,
                plane_support_count=spec.plane_support_count,
                plane_rmse_m=spec.plane_rmse_m,
                coordinate_frame=coordinate_frame,
                timestamp_ns=timestamp_ns,
                observation_id=observation_id,
                camera_id=camera_id,
                include_mask=include_mask,
                exclude_mask=per_plane_exc[p_idx],
                pixel_step=step,
                observation_weight=float(observation_weight),
            )
            plane_results[spec.plane_id] = res
            if res.accepted:
                any_accepted = True
                total_inliers += res.inlier_sample_count
            if res.reset_triggered:
                any_sub_reset = True

        if any_accepted or any_sub_reset:
            if self._coordinate_frame is None:
                self._coordinate_frame = coordinate_frame
            if self._latest_observation_timestamp_ns is None or timestamp_ns > self._latest_observation_timestamp_ns:
                self._latest_observation_timestamp_ns = timestamp_ns
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._invalidate_cache()

        committed, tentative = self._sum_cell_counts()
        if any_accepted:
            all_accepted = all(r.accepted for r in plane_results.values())
            status = "integrated" if all_accepted else "partial_integrated"
        elif plane_results and all(r.status == "duplicate_ignored" for r in plane_results.values()):
            status = "duplicate_ignored"
        else:
            status = next(iter(plane_results.values())).status if plane_results else "rejected"

        return MultiPlaneUpdateResult(
            accepted=any_accepted,
            status=status,
            version=self._version,
            epoch=self._epoch,
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            plane_results=plane_results,
            active_plane_ids=tuple(sorted(self._planes.keys())),
            total_inlier_sample_count=total_inliers,
            total_committed_cell_count=committed,
            total_tentative_cell_count=tentative,
            reset_triggered=any_sub_reset,
        )

    @_locked
    def integrate_points(
        self,
        points_m,
        colors_rgb_u8,
        *,
        planes: Sequence[PlaneObservationSpec],
        coordinate_frame: str,
        timestamp_ns: int,
        observation_id: str,
        camera_id: str = "default",
        frame_from_camera=None,
        include_mask=None,
        exclude_mask=None,
        observation_weight: float = 1.0,
    ) -> MultiPlaneUpdateResult:
        """Integrate pre-registered common-frame 3D samples across multiple measured planes."""
        points, colors = _validate_samples(
            points_m, colors_rgb_u8, coordinate_frame, timestamp_ns, observation_id
        )
        camera_id = _validate_nonempty_str(camera_id, "camera_id")
        timestamp_ns = int(timestamp_ns)
        if (
            isinstance(observation_weight, bool)
            or not math.isfinite(observation_weight)
            or observation_weight <= 0.0
        ):
            raise ValueError("observation_weight must be finite and positive")

        T_fc = (
            None
            if frame_from_camera is None
            else _validate_se3(frame_from_camera, "frame_from_camera")
        )
        selected = np.ones(len(points), dtype=bool)
        if include_mask is not None:
            inc = np.asarray(include_mask)
            if inc.shape != (len(points),) or inc.dtype != bool:
                raise ValueError("include_mask must be a 1-D boolean array matching points")
            selected &= inc
        if exclude_mask is not None:
            exc = np.asarray(exclude_mask)
            if exc.shape != (len(points),) or exc.dtype != bool:
                raise ValueError("exclude_mask must be a 1-D boolean array matching points")
            selected &= ~exc

        ordered_specs = self._validate_planes_arg(planes)
        frame_reset, early_reject = self._handle_frame_change(
            coordinate_frame, observation_id=observation_id, timestamp_ns=timestamp_ns
        )
        if early_reject is not None:
            return early_reject

        valid_specs: list[tuple[PlaneObservationSpec, np.ndarray, float]] = []
        plane_results: dict[str, SurfaceUpdateResult] = {}
        new_planes_budget = self._config.max_planes - len(self._planes)
        for spec in ordered_specs:
            ev = _validate_plane_evidence(
                spec.plane_normal,
                spec.plane_offset_m,
                spec.plane_support_count,
                spec.plane_rmse_m,
                self._config.plane_config.mesh,
            )
            if ev is None:
                plane_results[spec.plane_id] = SurfaceUpdateResult(
                    accepted=False,
                    status="invalid_plane",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                )
                continue
            if spec.plane_id not in self._planes:
                if new_planes_budget <= 0:
                    plane_results[spec.plane_id] = SurfaceUpdateResult(
                        accepted=False,
                        status="max_planes_exceeded",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                    )
                    continue
                new_planes_budget -= 1
            valid_specs.append((spec, ev[0], ev[1]))

        if not valid_specs:
            if frame_reset:
                self._version += 1
                self._map_updated_monotonic_ns = time.monotonic_ns()
            committed, tentative = self._sum_cell_counts()
            status = (
                "max_planes_exceeded"
                if any(r.status == "max_planes_exceeded" for r in plane_results.values())
                else "no_valid_planes"
            )
            return MultiPlaneUpdateResult(
                accepted=False,
                status=status,
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                plane_results=plane_results,
                active_plane_ids=tuple(sorted(self._planes.keys())),
                total_committed_cell_count=committed,
                total_tentative_cell_count=tentative,
                reset_triggered=frame_reset,
            )

        per_plane_inc: list[np.ndarray] = [selected for _ in range(len(valid_specs))]
        if self._config.enable_cross_plane_exclusion and len(valid_specs) >= 2 and len(points) > 0:
            finite_sel = selected & np.isfinite(points).all(axis=1)
            sel_idx = np.flatnonzero(finite_sel)
            if len(sel_idx) > 0:
                normals_mat = np.stack([item[1] for item in valid_specs], axis=0)
                offsets_vec = np.array([item[2] for item in valid_specs], dtype=np.float64)
                winner, _ = _arbitrate_point_ownership(
                    points[sel_idx],
                    normals_mat,
                    offsets_vec,
                    max_residual_m=self._config.plane_config.mesh.maximum_plane_residual_m,
                    seam_tie_tolerance_m=self._config.seam_tie_tolerance_m,
                )
                per_plane_inc = []
                for p_idx in range(len(valid_specs)):
                    mask_p = np.zeros(len(points), dtype=bool)
                    mask_p[sel_idx[winner == p_idx]] = True
                    per_plane_inc.append(mask_p)

        any_accepted = False
        any_sub_reset = frame_reset
        total_inliers = 0
        for p_idx, (spec, normal, offset_m) in enumerate(valid_specs):
            recon = self._planes.get(spec.plane_id)
            if recon is None:
                recon = PersistentPlanarSurfaceReconstructor(
                    coordinate_frame=self._coordinate_frame or coordinate_frame,
                    config=self._config.plane_config,
                )
                self._planes[spec.plane_id] = recon
            res = recon.integrate_points(
                points,
                colors,
                plane_normal=normal,
                plane_offset_m=offset_m,
                plane_support_count=spec.plane_support_count,
                plane_rmse_m=spec.plane_rmse_m,
                coordinate_frame=coordinate_frame,
                timestamp_ns=timestamp_ns,
                observation_id=observation_id,
                camera_id=camera_id,
                frame_from_camera=T_fc,
                include_mask=per_plane_inc[p_idx],
                observation_weight=float(observation_weight),
            )
            plane_results[spec.plane_id] = res
            if res.accepted:
                any_accepted = True
                total_inliers += res.inlier_sample_count
            if res.reset_triggered:
                any_sub_reset = True

        if any_accepted or any_sub_reset:
            if self._coordinate_frame is None:
                self._coordinate_frame = coordinate_frame
            if self._latest_observation_timestamp_ns is None or timestamp_ns > self._latest_observation_timestamp_ns:
                self._latest_observation_timestamp_ns = timestamp_ns
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._invalidate_cache()

        committed, tentative = self._sum_cell_counts()
        if any_accepted:
            all_accepted = all(r.accepted for r in plane_results.values())
            status = "integrated" if all_accepted else "partial_integrated"
        elif plane_results and all(r.status == "duplicate_ignored" for r in plane_results.values()):
            status = "duplicate_ignored"
        else:
            status = next(iter(plane_results.values())).status if plane_results else "rejected"

        return MultiPlaneUpdateResult(
            accepted=any_accepted,
            status=status,
            version=self._version,
            epoch=self._epoch,
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            plane_results=plane_results,
            active_plane_ids=tuple(sorted(self._planes.keys())),
            total_inlier_sample_count=total_inliers,
            total_committed_cell_count=committed,
            total_tentative_cell_count=tentative,
            reset_triggered=any_sub_reset,
        )

    @_locked
    def correct_pose(
        self,
        observation_id: str,
        new_frame_from_camera,
        *,
        timestamp_ns: int | None = None,
    ) -> MultiPlaneUpdateResult:
        """Broadcast a camera pose correction across all registered planes."""
        observation_id = _validate_nonempty_str(observation_id, "observation_id")
        T_fc = _validate_se3(new_frame_from_camera, "new_frame_from_camera")
        if timestamp_ns is not None:
            timestamp_ns = _validate_timestamp_ns(timestamp_ns)

        plane_results: dict[str, SurfaceUpdateResult] = {}
        any_accepted = False
        total_inliers = 0
        for pid in sorted(self._planes.keys()):
            res = self._planes[pid].correct_pose(
                observation_id, T_fc, timestamp_ns=timestamp_ns
            )
            plane_results[pid] = res
            if res.accepted:
                any_accepted = True
                total_inliers += res.inlier_sample_count

        if any_accepted:
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._invalidate_cache()

        committed, tentative = self._sum_cell_counts()
        if any_accepted:
            status = "pose_corrected"
        elif plane_results and any(
            r.status == "pose_correction_out_of_window" for r in plane_results.values()
        ):
            status = "pose_correction_out_of_window"
        elif plane_results and all(
            r.status == "pose_unchanged" for r in plane_results.values()
        ):
            status = "pose_unchanged"
        else:
            status = "unknown_observation_id"

        return MultiPlaneUpdateResult(
            accepted=any_accepted,
            status=status,
            version=self._version,
            epoch=self._epoch,
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            plane_results=plane_results,
            active_plane_ids=tuple(sorted(self._planes.keys())),
            total_inlier_sample_count=total_inliers,
            total_committed_cell_count=committed,
            total_tentative_cell_count=tentative,
        )

    @_locked
    def correct_poses(
        self,
        corrections: Mapping[str, object],
        *,
        timestamp_ns: int | None = None,
    ) -> dict[str, MultiPlaneUpdateResult]:
        """Apply a batch of camera pose corrections in deterministic key order."""
        if not isinstance(corrections, Mapping):
            raise ValueError("corrections must be a mapping of observation_id -> 4x4 SE(3)")
        return {
            obs_id: self.correct_pose(obs_id, pose, timestamp_ns=timestamp_ns)
            for obs_id, pose in corrections.items()
        }

    @_locked
    def estimated_memory_bytes(self) -> int:
        return sum(recon.estimated_memory_bytes() for recon in self._planes.values())

    @_locked
    def snapshot(
        self, *, reference_timestamp_ns: int | None = None
    ) -> MultiPlaneSurfaceSnapshot:
        """Return an immutable unified snapshot across all registered planes."""
        if reference_timestamp_ns is not None:
            reference_timestamp_ns = _validate_timestamp_ns(
                reference_timestamp_ns, "reference_timestamp_ns"
            )
        now_mono = time.monotonic_ns()
        cache_key = (self._epoch, self._version)
        if (
            self._cached_snapshot is not None
            and self._cached_snapshot_key == cache_key
            and self._cached_snapshot.reference_timestamp_ns == reference_timestamp_ns
        ):
            cached = self._cached_snapshot
            return MultiPlaneSurfaceSnapshot(
                ready=cached.ready,
                reason=cached.reason,
                version=cached.version,
                epoch=cached.epoch,
                coordinate_frame=cached.coordinate_frame,
                plane_ids=cached.plane_ids,
                plane_snapshots=cached.plane_snapshots,
                vertices_m=cached.vertices_m,
                colors_rgb_u8=cached.colors_rgb_u8,
                triangles=cached.triangles,
                face_normals=cached.face_normals,
                vertex_plane_indices=cached.vertex_plane_indices,
                triangle_plane_indices=cached.triangle_plane_indices,
                vertex_observation_counts=cached.vertex_observation_counts,
                vertex_last_observed_ns=cached.vertex_last_observed_ns,
                support_count=cached.support_count,
                surface_area_m2=cached.surface_area_m2,
                maximum_edge_m=cached.maximum_edge_m,
                estimated_memory_bytes=cached.estimated_memory_bytes,
                snapshot_created_monotonic_ns=now_mono,
                map_updated_monotonic_ns=cached.map_updated_monotonic_ns,
                reference_timestamp_ns=reference_timestamp_ns,
                latest_observation_timestamp_ns=cached.latest_observation_timestamp_ns,
                newest_contributing_timestamp_ns=cached.newest_contributing_timestamp_ns,
                oldest_contributing_timestamp_ns=cached.oldest_contributing_timestamp_ns,
            )

        sorted_ids = tuple(sorted(self._planes.keys()))
        if not sorted_ids:
            snap = MultiPlaneSurfaceSnapshot(
                ready=False,
                reason="no_planes",
                version=self._version,
                epoch=self._epoch,
                coordinate_frame=self._coordinate_frame or "",
                plane_ids=(),
                plane_snapshots={},
                estimated_memory_bytes=0,
                snapshot_created_monotonic_ns=now_mono,
                map_updated_monotonic_ns=self._map_updated_monotonic_ns,
                reference_timestamp_ns=reference_timestamp_ns,
                latest_observation_timestamp_ns=self._latest_observation_timestamp_ns,
            )
            self._cached_snapshot_key = cache_key
            self._cached_snapshot = snap
            return snap

        plane_snaps: dict[str, PersistentSurfaceSnapshot] = {}
        verts_parts: list[np.ndarray] = []
        colors_parts: list[np.ndarray] = []
        tris_parts: list[np.ndarray] = []
        normals_parts: list[np.ndarray] = []
        v_plane_parts: list[np.ndarray] = []
        t_plane_parts: list[np.ndarray] = []
        v_obs_parts: list[np.ndarray] = []
        v_ts_parts: list[np.ndarray] = []

        vertex_offset = 0
        total_support = 0
        total_area = 0.0
        max_edge = 0.0
        total_mem = 0
        newest_ts: int | None = None
        oldest_ts: int | None = None
        any_ready = False

        for p_idx, pid in enumerate(sorted_ids):
            ps = self._planes[pid].snapshot(reference_timestamp_ns=reference_timestamp_ns)
            plane_snaps[pid] = ps
            total_mem += ps.estimated_memory_bytes
            if ps.ready and len(ps.vertices_m) > 0:
                any_ready = True
                nv = len(ps.vertices_m)
                nf = len(ps.triangles)
                verts_parts.append(ps.vertices_m)
                colors_parts.append(ps.colors_rgb_u8)
                tris_parts.append(ps.triangles.astype(np.int32) + vertex_offset)
                normals_parts.append(ps.face_normals)
                v_plane_parts.append(np.full(nv, p_idx, dtype=np.int32))
                t_plane_parts.append(np.full(nf, p_idx, dtype=np.int32))
                v_obs_parts.append(ps.vertex_observation_counts)
                v_ts_parts.append(ps.vertex_last_observed_ns)
                vertex_offset += nv
                total_support += ps.support_count
                total_area += ps.surface_area_m2
                if ps.maximum_edge_m > max_edge:
                    max_edge = ps.maximum_edge_m
                if ps.newest_contributing_timestamp_ns is not None:
                    if newest_ts is None or ps.newest_contributing_timestamp_ns > newest_ts:
                        newest_ts = ps.newest_contributing_timestamp_ns
                if ps.oldest_contributing_timestamp_ns is not None:
                    if oldest_ts is None or ps.oldest_contributing_timestamp_ns < oldest_ts:
                        oldest_ts = ps.oldest_contributing_timestamp_ns

        if any_ready:
            vertices_m = _freeze_copy(np.concatenate(verts_parts, axis=0))
            colors_rgb_u8 = _freeze_copy(np.concatenate(colors_parts, axis=0))
            triangles = _freeze_copy(np.concatenate(tris_parts, axis=0))
            face_normals = _freeze_copy(np.concatenate(normals_parts, axis=0))
            vertex_plane_indices = _freeze_copy(np.concatenate(v_plane_parts, axis=0))
            triangle_plane_indices = _freeze_copy(np.concatenate(t_plane_parts, axis=0))
            vertex_observation_counts = _freeze_copy(np.concatenate(v_obs_parts, axis=0))
            vertex_last_observed_ns = _freeze_copy(np.concatenate(v_ts_parts, axis=0))
            reason = "ok"
        else:
            vertices_m = _freeze_copy(np.empty((0, 3), dtype=np.float64))
            colors_rgb_u8 = _freeze_copy(np.empty((0, 3), dtype=np.uint8))
            triangles = _freeze_copy(np.empty((0, 3), dtype=np.int32))
            face_normals = _freeze_copy(np.empty((0, 3), dtype=np.float64))
            vertex_plane_indices = _freeze_copy(np.empty(0, dtype=np.int32))
            triangle_plane_indices = _freeze_copy(np.empty(0, dtype=np.int32))
            vertex_observation_counts = _freeze_copy(np.empty(0, dtype=np.int32))
            vertex_last_observed_ns = _freeze_copy(np.empty(0, dtype=np.int64))
            reason = next(iter(plane_snaps.values())).reason

        snap = MultiPlaneSurfaceSnapshot(
            ready=any_ready,
            reason=reason,
            version=self._version,
            epoch=self._epoch,
            coordinate_frame=self._coordinate_frame or "",
            plane_ids=sorted_ids,
            plane_snapshots=plane_snaps,
            vertices_m=vertices_m,
            colors_rgb_u8=colors_rgb_u8,
            triangles=triangles,
            face_normals=face_normals,
            vertex_plane_indices=vertex_plane_indices,
            triangle_plane_indices=triangle_plane_indices,
            vertex_observation_counts=vertex_observation_counts,
            vertex_last_observed_ns=vertex_last_observed_ns,
            support_count=total_support,
            surface_area_m2=total_area,
            maximum_edge_m=max_edge,
            estimated_memory_bytes=total_mem,
            snapshot_created_monotonic_ns=now_mono,
            map_updated_monotonic_ns=self._map_updated_monotonic_ns,
            reference_timestamp_ns=reference_timestamp_ns,
            latest_observation_timestamp_ns=self._latest_observation_timestamp_ns,
            newest_contributing_timestamp_ns=newest_ts,
            oldest_contributing_timestamp_ns=oldest_ts,
        )
        self._cached_snapshot_key = cache_key
        self._cached_snapshot = snap
        return snap


def extract_measured_planes(
    points_m: np.ndarray,
    *,
    max_planes: int = 4,
    ransac_iterations: int = 200,
    plane_distance_threshold_m: float = 0.006,
    minimum_plane_points: int = 120,
    peel_distance_m: float | None = None,
    min_inter_plane_angle_deg: float = 0.0,
    max_sample_points: int = 3000,
    plane_id_prefix: str = "plane",
    reference_up: Sequence[float] | np.ndarray | None = (0.0, 0.0, 1.0),
) -> list[PlaneObservationSpec]:
    """Deterministically extract up to ``max_planes`` dominant measured planes.

    Sequentially peels measured planes via :func:`evidencescene.runtime.fast_fit_shared_plane`
    (accelerated by C11 ``fit_plane_ransac_fast`` when available), computes full-cloud
    support and RMSE evidence, canonicalizes normal orientation, and returns
    :class:`PlaneObservationSpec` entries ready to feed directly into
    :class:`PersistentMultiPlaneSurfaceReconstructor`.
    """
    from evidencescene.runtime.request_evidence_deduplication import fast_fit_shared_plane

    if isinstance(max_planes, bool) or not isinstance(max_planes, (int, np.integer)) or int(max_planes) < 1:
        raise ValueError("max_planes must be a positive integer")
    if isinstance(ransac_iterations, bool) or not isinstance(ransac_iterations, (int, np.integer)) or int(ransac_iterations) < 1:
        raise ValueError("ransac_iterations must be a positive integer")
    if (
        isinstance(plane_distance_threshold_m, bool)
        or not math.isfinite(plane_distance_threshold_m)
        or plane_distance_threshold_m <= 0.0
    ):
        raise ValueError("plane_distance_threshold_m must be finite and positive")
    if (
        isinstance(minimum_plane_points, bool)
        or not isinstance(minimum_plane_points, (int, np.integer))
        or int(minimum_plane_points) < 3
    ):
        raise ValueError("minimum_plane_points must be an integer >= 3")
    if (
        isinstance(min_inter_plane_angle_deg, bool)
        or not math.isfinite(min_inter_plane_angle_deg)
        or not (0.0 <= min_inter_plane_angle_deg < 90.0)
    ):
        raise ValueError("min_inter_plane_angle_deg must be in [0, 90)")
    if isinstance(max_sample_points, bool) or not isinstance(max_sample_points, (int, np.integer)) or int(max_sample_points) < 3:
        raise ValueError("max_sample_points must be an integer >= 3")
    _validate_nonempty_str(plane_id_prefix, "plane_id_prefix")

    peel_m = float(plane_distance_threshold_m * 1.5 if peel_distance_m is None else peel_distance_m)
    if not math.isfinite(peel_m) or peel_m < plane_distance_threshold_m:
        raise ValueError("peel_distance_m must be finite and >= plane_distance_threshold_m")

    ref_up: np.ndarray | None = None
    if reference_up is not None:
        up_arr = np.asarray(reference_up, dtype=np.float64)
        if up_arr.shape != (3,) or not np.isfinite(up_arr).all() or float(np.linalg.norm(up_arr)) <= 1e-12:
            raise ValueError("reference_up must be a finite non-zero 3-vector or None")
        ref_up = up_arr / float(np.linalg.norm(up_arr))

    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_m must be an Nx3 array")
    if len(pts) == 0:
        return []
    finite_mask = np.isfinite(pts).all(axis=1)
    rem = pts[finite_mask]

    cos_max_angle = (
        math.cos(math.radians(min_inter_plane_angle_deg))
        if min_inter_plane_angle_deg > 0.0
        else 1.0
    )
    accepted_normals: list[np.ndarray] = []
    specs: list[PlaneObservationSpec] = []
    max_attempts = int(max_planes) * 3

    for _ in range(max_attempts):
        if len(specs) >= int(max_planes) or len(rem) < int(minimum_plane_points):
            break
        if len(rem) > int(max_sample_points):
            s_idx = np.linspace(0, len(rem) - 1, int(max_sample_points), dtype=np.intp)
            sample = rem[s_idx]
            min_sample_pts = max(3, int(round(minimum_plane_points * len(sample) / len(rem))))
        else:
            sample = rem
            min_sample_pts = int(minimum_plane_points)

        fitted = fast_fit_shared_plane(
            sample,
            ransac_iterations=int(ransac_iterations),
            plane_distance_threshold_m=float(plane_distance_threshold_m),
            minimum_plane_points=min_sample_pts,
        )
        if fitted is None:
            break
        n_vec, offset_m, _sub_cnt, _sub_rmse = fitted

        # Canonicalize normal sign deterministically.
        if ref_up is not None and abs(float(n_vec @ ref_up)) > 1e-3:
            if float(n_vec @ ref_up) < 0.0:
                n_vec = -n_vec
                offset_m = -offset_m
        else:
            dom = int(np.argmax(np.abs(n_vec)))
            if float(n_vec[dom]) < 0.0:
                n_vec = -n_vec
                offset_m = -offset_m

        # Evaluate full remaining cloud support & peel inlier band.
        res = np.abs(rem @ n_vec + offset_m)
        strict_inliers = res <= float(plane_distance_threshold_m)
        strict_count = int(np.count_nonzero(strict_inliers))
        peel_mask = res <= peel_m
        if strict_count >= int(minimum_plane_points):
            # Refine normal and offset on full strict inliers for accuracy.
            in_pts = rem[strict_inliers]
            center = np.mean(in_pts, axis=0)
            _, _, vt = np.linalg.svd(in_pts - center, full_matrices=False)
            n_ref = vt[-1]
            n_ref = n_ref / float(np.linalg.norm(n_ref))
            if float(n_ref @ n_vec) < 0.0:
                n_ref = -n_ref
            off_ref = -float(n_ref @ center)
            rmse_ref = float(np.sqrt(np.mean(np.square(in_pts @ n_ref + off_ref))))

            angle_ok = True
            if min_inter_plane_angle_deg > 0.0:
                for prev_n in accepted_normals:
                    if abs(float(n_ref @ prev_n)) > cos_max_angle + 1e-9:
                        angle_ok = False
                        break

            if angle_ok and math.isfinite(rmse_ref) and rmse_ref <= float(plane_distance_threshold_m):
                n_frozen = _freeze_copy(n_ref)
                accepted_normals.append(n_frozen)
                specs.append(
                    PlaneObservationSpec(
                        plane_id=f"{plane_id_prefix}_{len(specs)}",
                        plane_normal=n_frozen,
                        plane_offset_m=float(off_ref),
                        plane_support_count=strict_count,
                        plane_rmse_m=rmse_ref,
                    )
                )

        if not np.any(peel_mask):
            break
        rem = rem[~peel_mask]

    return specs


__all__ = [
    "MultiPlaneSurfaceConfig",
    "MultiPlaneSurfaceSnapshot",
    "MultiPlaneUpdateResult",
    "PersistentMultiPlaneSurfaceReconstructor",
    "PlaneObservationSpec",
    "extract_measured_planes",
]
