"""Persistent measured planar static surface reconstruction across RGB-D frames.

Integrates calibrated RGB-D or pre-registered common-frame observations into a
bounded, versioned, sparse planar cell map with exact per-observation provenance,
reversible recent pose corrections, free-space carving, sensor-physics fusion
weighting, dirty-tile incremental meshing, optional C11/SIMD acceleration, and
explicit policies for duplicates, frame/calibration changes, and resets.

Reuses the existing backprojection primitive and measured mesh topology gates:
unknown regions stay open, depth discontinuities and normal steps are not
bridged, and transient foreground or invalid depth is prevented from becoming
static geometry.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import functools
import math
import threading
import time
from typing import Literal, Mapping

import numpy as np

from evidencescene._accel import NATIVE
from evidencescene.primitives.geometry import _backproject_masked_depth
from evidencescene.surface.measured_mesh import (
    SurfaceMeshConfig,
    _quads,
    _validate_samples,
)

_CELL_COORD_BIAS = 1 << 30
_CELL_COORD_LIMIT = (1 << 29) - 1
_CELL_COORD_MASK = (1 << 32) - 1
_INITIAL_CAPACITY = 1024
_TILE_SHIFT = 5
_TILE_SIZE = 1 << _TILE_SHIFT  # 32x32 cells per spatial tile
_TILE_MASK = _TILE_SIZE - 1
_HT_EMPTY = -(1 << 63)
_POLICY_VALUES = ("reject", "reset")
_WEIGHTING_MODES = ("uniform", "sensor_physics")


def _next_pow2(n: int) -> int:
    p = 2048
    while p < n:
        p <<= 1
    return p


def _pack_cell_keys(iu: np.ndarray, iv: np.ndarray) -> np.ndarray:
    return ((iu.astype(np.int64) + _CELL_COORD_BIAS) << 32) | (
        (iv.astype(np.int64) + _CELL_COORD_BIAS) & _CELL_COORD_MASK
    )


def _unpack_cell_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    iu = (keys >> 32) - _CELL_COORD_BIAS
    iv = (keys & _CELL_COORD_MASK) - _CELL_COORD_BIAS
    return iu.astype(np.int64), iv.astype(np.int64)


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    u = axis - normal * float(np.dot(axis, normal))
    u /= float(np.linalg.norm(u))
    v = np.cross(normal, u)
    return u, v


def _validate_nonempty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_timestamp_ns(value: object, name: str = "timestamp_ns") -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _validate_se3(transform: object, name: str = "frame_from_camera") -> np.ndarray:
    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4) or not np.isfinite(mat).all():
        raise ValueError(f"{name} must be a finite 4x4 SE(3) matrix")
    if not np.allclose(mat[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4, rtol=0.0) or np.linalg.det(rot) < 0.99:
        raise ValueError(f"{name} upper-left 3x3 block must be a proper rotation matrix")
    return mat.copy()


def _validate_intrinsics(camera_matrix: object) -> np.ndarray:
    K = np.asarray(camera_matrix, dtype=np.float64)
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0.0
        or K[1, 1] <= 0.0
        or not np.allclose(K[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0.0)
        or K[0, 1] != 0.0
        or K[1, 0] != 0.0
    ):
        raise ValueError("camera_matrix must contain valid rectified pinhole intrinsics")
    return K.copy()


def _validate_plane_evidence(
    plane_normal: object,
    plane_offset_m: object,
    plane_support_count: object,
    plane_rmse_m: object,
    mesh_config: SurfaceMeshConfig,
) -> tuple[np.ndarray, float] | None:
    normal = np.asarray(plane_normal, dtype=np.float64)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0.0)
        or isinstance(plane_offset_m, bool)
        or isinstance(plane_rmse_m, bool)
        or not np.isfinite([float(plane_offset_m), float(plane_rmse_m)]).all()
        or float(plane_rmse_m) < 0.0
        or isinstance(plane_support_count, bool)
        or not isinstance(plane_support_count, (int, np.integer))
        or int(plane_support_count) < mesh_config.minimum_support
        or float(plane_rmse_m) > mesh_config.maximum_plane_residual_m
    ):
        return None
    return normal / float(np.linalg.norm(normal)), float(plane_offset_m)


def _freeze_copy(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


@dataclass(frozen=True)
class PersistentPlanarSurfaceConfig:
    """Configuration and memory/freshness bounds for persistent planar surface fusion."""

    mesh: SurfaceMeshConfig = field(default_factory=SurfaceMeshConfig)
    min_observations_to_commit: int = 2
    max_cell_weight: float = 32.0
    tentative_ttl_updates: int = 8
    pixel_step: int = 4
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 3.0
    maximum_grazing_angle_deg: float = 82.0
    weighting_mode: Literal["uniform", "sensor_physics"] = "sensor_physics"
    specular_saturation_threshold: int = 250
    specular_rgb_weight_scale: float = 0.25
    enable_free_space_carving: bool = True
    free_space_carve_margin_m: float = 0.015
    free_space_conflict_threshold: int = 2
    max_plane_angle_change_deg: float = 2.0
    max_plane_offset_change_m: float = 0.015
    frame_change_policy: Literal["reject", "reset"] = "reject"
    calibration_change_policy: Literal["reject", "reset"] = "reject"
    plane_change_policy: Literal["reject", "reset"] = "reject"
    max_cells: int = 40_000
    max_retained_observations: int = 64
    max_seen_observation_ids: int = 4096
    max_provenance_observations: int = 128
    retain_camera_points: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, SurfaceMeshConfig):
            raise ValueError("mesh must be a SurfaceMeshConfig")
        if not isinstance(self.retain_camera_points, bool):
            raise ValueError("retain_camera_points must be a bool")
        for name in (
            "min_observations_to_commit",
            "tentative_ttl_updates",
            "pixel_step",
            "free_space_conflict_threshold",
            "max_cells",
            "max_retained_observations",
            "max_seen_observation_ids",
            "max_provenance_observations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_cell_weight", "minimum_depth_m", "maximum_depth_m", "free_space_carve_margin_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (self.minimum_depth_m < self.maximum_depth_m):
            raise ValueError("minimum_depth_m must be strictly less than maximum_depth_m")
        if (
            isinstance(self.maximum_grazing_angle_deg, bool)
            or not math.isfinite(self.maximum_grazing_angle_deg)
            or not (0.0 < self.maximum_grazing_angle_deg <= 90.0)
        ):
            raise ValueError("maximum_grazing_angle_deg must be in (0, 90]")
        if self.weighting_mode not in _WEIGHTING_MODES:
            raise ValueError(f"weighting_mode must be one of {_WEIGHTING_MODES}")
        if (
            isinstance(self.specular_saturation_threshold, bool)
            or not isinstance(self.specular_saturation_threshold, (int, np.integer))
            or not (1 <= int(self.specular_saturation_threshold) <= 256)
        ):
            raise ValueError("specular_saturation_threshold must be an integer in [1, 256]")
        if (
            isinstance(self.specular_rgb_weight_scale, bool)
            or not math.isfinite(self.specular_rgb_weight_scale)
            or not (0.0 < self.specular_rgb_weight_scale <= 1.0)
        ):
            raise ValueError("specular_rgb_weight_scale must be finite and in (0, 1]")
        for name in ("max_plane_angle_change_deg", "max_plane_offset_change_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("frame_change_policy", "calibration_change_policy", "plane_change_policy"):
            if getattr(self, name) not in _POLICY_VALUES:
                raise ValueError(f"{name} must be one of {_POLICY_VALUES}")


@dataclass(frozen=True)
class ObservationProvenance:
    """Provenance record for an observation that contributed to the static surface."""

    observation_id: str
    timestamp_ns: int
    camera_id: str | None
    sample_count: int
    pose_revision: int = 0


@dataclass(frozen=True)
class SurfaceUpdateResult:
    """Outcome of an integration, pose correction, or reset call."""

    accepted: bool
    status: str
    version: int
    epoch: int
    observation_id: str | None = None
    timestamp_ns: int | None = None
    inlier_sample_count: int = 0
    updated_cell_count: int = 0
    committed_cell_count: int = 0
    tentative_cell_count: int = 0
    carved_cell_count: int = 0
    reset_triggered: bool = False


@dataclass(frozen=True)
class PersistentSurfaceSnapshot:
    """Bounded, versioned mesh snapshot with distinguishable snapshot age and observation freshness.

    - Snapshot age is measured from ``snapshot_created_monotonic_ns`` (and
      ``map_updated_monotonic_ns`` for time since last state mutation) using the
      local monotonic clock via :meth:`snapshot_age_ns`.
    - Observation freshness is measured on the sensor clock via
      ``latest_observation_timestamp_ns``, ``newest_contributing_timestamp_ns``,
      ``median_contributing_timestamp_ns``, ``oldest_contributing_timestamp_ns``,
      and per-vertex ``vertex_last_observed_ns`` (or via
      :meth:`observation_freshness_ns`).
    """

    ready: bool
    reason: str
    version: int
    epoch: int
    coordinate_frame: str
    vertices_m: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.float64)))
    colors_rgb_u8: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.uint8)))
    triangles: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.int32)))
    face_normals: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty((0, 3), dtype=np.float64)))
    plane_normal: np.ndarray | None = None
    plane_offset_m: float | None = None
    support_count: int = 0
    surface_area_m2: float = 0.0
    maximum_edge_m: float = 0.0
    vertex_observation_counts: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int32)))
    vertex_last_observed_ns: np.ndarray = field(default_factory=lambda: _freeze_copy(np.empty(0, dtype=np.int64)))
    contributing_observation_ids: tuple[str, ...] = ()
    contributing_observations: tuple[ObservationProvenance, ...] = ()
    total_integrated_observations: int = 0
    compacted_observation_count: int = 0
    snapshot_created_monotonic_ns: int = 0
    map_updated_monotonic_ns: int = 0
    reference_timestamp_ns: int | None = None
    latest_observation_timestamp_ns: int | None = None
    newest_contributing_timestamp_ns: int | None = None
    median_contributing_timestamp_ns: int | None = None
    oldest_contributing_timestamp_ns: int | None = None
    newest_observation_lag_ns: int | None = None
    oldest_vertex_lag_ns: int | None = None
    tentative_cell_count: int = 0
    estimated_memory_bytes: int = 0

    def snapshot_age_ns(self, now_monotonic_ns: int | None = None) -> int:
        """Wall/monotonic age of this materialized snapshot object in nanoseconds."""
        now = time.monotonic_ns() if now_monotonic_ns is None else _validate_timestamp_ns(now_monotonic_ns, "now_monotonic_ns")
        return max(0, now - self.snapshot_created_monotonic_ns)

    def observation_freshness_ns(self, current_sensor_timestamp_ns: int) -> dict[str, int | None]:
        """Sensor-clock lag of the latest stream frame and active mesh geometry."""
        now_ts = _validate_timestamp_ns(current_sensor_timestamp_ns, "current_sensor_timestamp_ns")

        def _lag(ts: int | None) -> int | None:
            return None if ts is None else now_ts - ts

        return {
            "latest_observation_lag_ns": _lag(self.latest_observation_timestamp_ns),
            "newest_contributing_lag_ns": _lag(self.newest_contributing_timestamp_ns),
            "median_contributing_lag_ns": _lag(self.median_contributing_timestamp_ns),
            "oldest_contributing_lag_ns": _lag(self.oldest_contributing_timestamp_ns),
        }


@dataclass
class _RetainedObservation:
    observation_id: str
    timestamp_ns: int
    camera_id: str
    frame_from_camera: np.ndarray | None
    camera_points_m: np.ndarray | None  # float32 (M, 3) in camera frame for pose revision
    camera_colors_rgb_u8: np.ndarray | None  # uint8 (M, 3)
    observation_weight: float
    cell_keys: np.ndarray  # int64 (C,)
    cell_pts: np.ndarray  # float64 (C, 3)
    cell_rgb: np.ndarray  # float32 (C, 3)
    cell_w: np.ndarray  # float64 (C,)
    pose_revision: int = 0

    def memory_bytes(self) -> int:
        total = self.cell_keys.nbytes + self.cell_pts.nbytes + self.cell_rgb.nbytes + self.cell_w.nbytes
        if self.camera_points_m is not None:
            total += self.camera_points_m.nbytes
        if self.camera_colors_rgb_u8 is not None:
            total += self.camera_colors_rgb_u8.nbytes
        if self.frame_from_camera is not None:
            total += self.frame_from_camera.nbytes
        return int(total)


@dataclass
class _SeenRecord:
    frame_from_camera: np.ndarray | None
    timestamp_ns: int


@dataclass
class _TileMeshCache:
    tri_slots: np.ndarray  # int32 (F_t, 3) global slot indices
    face_normals: np.ndarray  # float64 (F_t, 3)
    area_m2: float
    max_edge_m: float


@dataclass
class _CachedMeshData:
    version: int
    ready: bool
    reason: str
    vertices_m: np.ndarray
    colors_rgb_u8: np.ndarray
    triangles: np.ndarray
    face_normals: np.ndarray
    support_count: int
    surface_area_m2: float
    maximum_edge_m: float
    vertex_observation_counts: np.ndarray
    vertex_last_observed_ns: np.ndarray
    newest_contributing_timestamp_ns: int | None
    median_contributing_timestamp_ns: int | None
    oldest_contributing_timestamp_ns: int | None


class PersistentPlanarSurfaceReconstructor:
    """Bounded persistent reconstructor for a measured planar static surface.

    Maintains a sparse tangent-grid cell store split into an archived baseline
    and a bounded sliding revision window, paired with 32x32 dirty-tile
    incremental meshing and optional C11/SIMD kernels so recent camera pose
    corrections remain exact and reversible while long-session memory and
    per-frame update/snapshot costs remain tightly bounded.
    """

    def __init__(
        self,
        *,
        coordinate_frame: str | None = None,
        plane_normal=None,
        plane_offset_m: float | None = None,
        config: PersistentPlanarSurfaceConfig | None = None,
    ) -> None:
        self._config = config or PersistentPlanarSurfaceConfig()
        if coordinate_frame is not None:
            coordinate_frame = _validate_nonempty_str(coordinate_frame, "coordinate_frame")
        if (plane_normal is None) != (plane_offset_m is None):
            raise ValueError("plane_normal and plane_offset_m must be provided together or both omitted")
        init_normal: np.ndarray | None = None
        init_offset: float | None = None
        if plane_normal is not None:
            normal = np.asarray(plane_normal, dtype=np.float64)
            if (
                normal.shape != (3,)
                or not np.isfinite(normal).all()
                or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0.0)
                or isinstance(plane_offset_m, bool)
                or not math.isfinite(float(plane_offset_m))
            ):
                raise ValueError("plane_normal must be a unit 3-vector and plane_offset_m must be finite")
            init_normal = normal / float(np.linalg.norm(normal))
            init_offset = float(plane_offset_m)

        self._lock = threading.RLock()
        self._version: int = 0
        self._epoch: int = 0
        self._empty_reason: str = "no_observations"
        self._init_state(
            coordinate_frame=coordinate_frame,
            plane_normal=init_normal,
            plane_offset_m=init_offset,
        )

    def _init_state(
        self,
        *,
        coordinate_frame: str | None,
        plane_normal: np.ndarray | None,
        plane_offset_m: float | None,
    ) -> None:
        self._coordinate_frame: str | None = coordinate_frame
        self._plane_normal: np.ndarray | None = None
        self._plane_offset_m: float | None = None
        self._basis_u: np.ndarray | None = None
        self._basis_v: np.ndarray | None = None
        self._uv_origin: np.ndarray | None = None
        self._plane_anchor_m: np.ndarray | None = None
        if plane_normal is not None and plane_offset_m is not None:
            self._latch_plane(plane_normal, plane_offset_m)

        cap = min(_INITIAL_CAPACITY, self._config.max_cells)
        self._capacity: int = cap
        self._count: int = 0
        self._free_slots: list[int] = []
        self._key_to_slot: dict[int, int] = {}

        ht_cap = _next_pow2(cap * 2)
        self._ht_keys = np.full(ht_cap, _HT_EMPTY, dtype=np.int64)
        self._ht_slots = np.full(ht_cap, -1, dtype=np.int32)
        self._ht_dirty: bool = False

        self._active = np.zeros(cap, dtype=bool)
        self._keys = np.zeros(cap, dtype=np.int64)
        self._iu = np.zeros(cap, dtype=np.int32)
        self._iv = np.zeros(cap, dtype=np.int32)
        self._base_w = np.zeros(cap, dtype=np.float64)
        self._win_w = np.zeros(cap, dtype=np.float64)
        self._base_xyz = np.zeros((cap, 3), dtype=np.float64)
        self._win_xyz = np.zeros((cap, 3), dtype=np.float64)
        self._base_rgb = np.zeros((cap, 3), dtype=np.float32)
        self._win_rgb = np.zeros((cap, 3), dtype=np.float32)
        self._base_obs_count = np.zeros(cap, dtype=np.int32)
        self._win_obs_count = np.zeros(cap, dtype=np.int32)
        self._base_last_ts = np.full(cap, -1, dtype=np.int64)
        self._last_ts = np.full(cap, -1, dtype=np.int64)
        self._last_update_seq = np.zeros(cap, dtype=np.int64)
        self._free_space_conflicts = np.zeros(cap, dtype=np.int32)

        # Incrementally maintained fused cell values & dirty-tile caches
        self._fused_xyz = np.zeros((cap, 3), dtype=np.float64)
        self._fused_rgb = np.zeros((cap, 3), dtype=np.uint8)
        self._total_obs = np.zeros(cap, dtype=np.int32)
        self._tile_to_slots: dict[tuple[int, int], set[int]] = {}
        self._dirty_tiles: set[tuple[int, int]] = set()
        self._tile_cache: dict[tuple[int, int], _TileMeshCache] = {}

        self._retained_observations: OrderedDict[str, _RetainedObservation] = OrderedDict()
        self._seen_observations: OrderedDict[str, _SeenRecord] = OrderedDict()
        self._provenance: OrderedDict[str, ObservationProvenance] = OrderedDict()
        self._camera_calibrations: dict[str, tuple[tuple[int, int], np.ndarray]] = {}
        self._total_integrated_observations: int = 0
        self._compacted_observation_count: int = 0
        self._update_seq: int = 0
        self._latest_observation_timestamp_ns: int | None = None
        self._map_updated_monotonic_ns: int = time.monotonic_ns()
        self._cached_mesh: _CachedMeshData | None = None

    def _latch_plane(self, normal: np.ndarray, offset_m: float) -> None:
        self._plane_normal = normal.copy()
        self._plane_offset_m = float(offset_m)
        self._basis_u, self._basis_v = _tangent_basis(self._plane_normal)
        self._plane_anchor_m = -self._plane_offset_m * self._plane_normal

    @property
    def config(self) -> PersistentPlanarSurfaceConfig:
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

    @_locked
    def reset(
        self,
        *,
        coordinate_frame: str | None = None,
        plane_normal=None,
        plane_offset_m: float | None = None,
        reason: str = "explicit_reset",
    ) -> SurfaceUpdateResult:
        """Clear accumulated surface state, bumping epoch and version."""
        _validate_nonempty_str(reason, "reason")
        if coordinate_frame is not None:
            coordinate_frame = _validate_nonempty_str(coordinate_frame, "coordinate_frame")
        if (plane_normal is None) != (plane_offset_m is None):
            raise ValueError("plane_normal and plane_offset_m must be provided together or both omitted")
        init_normal: np.ndarray | None = None
        init_offset: float | None = None
        if plane_normal is not None:
            normal = np.asarray(plane_normal, dtype=np.float64)
            if (
                normal.shape != (3,)
                or not np.isfinite(normal).all()
                or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0.0)
                or isinstance(plane_offset_m, bool)
                or not math.isfinite(float(plane_offset_m))
            ):
                raise ValueError("plane_normal must be a unit 3-vector and plane_offset_m must be finite")
            init_normal = normal / float(np.linalg.norm(normal))
            init_offset = float(plane_offset_m)

        self._epoch += 1
        self._version += 1
        self._empty_reason = "empty_after_reset"
        self._init_state(
            coordinate_frame=coordinate_frame,
            plane_normal=init_normal,
            plane_offset_m=init_offset,
        )
        return SurfaceUpdateResult(
            accepted=True,
            status="reset",
            version=self._version,
            epoch=self._epoch,
            reset_triggered=True,
        )

    def _grow_storage(self, min_needed: int) -> None:
        if min_needed <= self._capacity:
            return
        new_cap = min(self._config.max_cells, max(self._capacity * 2, min_needed))
        if new_cap <= self._capacity:
            return
        pad = new_cap - self._capacity
        self._active = np.concatenate((self._active, np.zeros(pad, dtype=bool)))
        self._keys = np.concatenate((self._keys, np.zeros(pad, dtype=np.int64)))
        self._iu = np.concatenate((self._iu, np.zeros(pad, dtype=np.int32)))
        self._iv = np.concatenate((self._iv, np.zeros(pad, dtype=np.int32)))
        self._base_w = np.concatenate((self._base_w, np.zeros(pad, dtype=np.float64)))
        self._win_w = np.concatenate((self._win_w, np.zeros(pad, dtype=np.float64)))
        self._base_xyz = np.concatenate((self._base_xyz, np.zeros((pad, 3), dtype=np.float64)), axis=0)
        self._win_xyz = np.concatenate((self._win_xyz, np.zeros((pad, 3), dtype=np.float64)), axis=0)
        self._base_rgb = np.concatenate((self._base_rgb, np.zeros((pad, 3), dtype=np.float32)), axis=0)
        self._win_rgb = np.concatenate((self._win_rgb, np.zeros((pad, 3), dtype=np.float32)), axis=0)
        self._base_obs_count = np.concatenate((self._base_obs_count, np.zeros(pad, dtype=np.int32)))
        self._win_obs_count = np.concatenate((self._win_obs_count, np.zeros(pad, dtype=np.int32)))
        self._base_last_ts = np.concatenate((self._base_last_ts, np.full(pad, -1, dtype=np.int64)))
        self._last_ts = np.concatenate((self._last_ts, np.full(pad, -1, dtype=np.int64)))
        self._last_update_seq = np.concatenate((self._last_update_seq, np.zeros(pad, dtype=np.int64)))
        self._free_space_conflicts = np.concatenate((self._free_space_conflicts, np.zeros(pad, dtype=np.int32)))
        self._fused_xyz = np.concatenate((self._fused_xyz, np.zeros((pad, 3), dtype=np.float64)), axis=0)
        self._fused_rgb = np.concatenate((self._fused_rgb, np.zeros((pad, 3), dtype=np.uint8)), axis=0)
        self._total_obs = np.concatenate((self._total_obs, np.zeros(pad, dtype=np.int32)))
        self._capacity = new_cap
        need_ht = _next_pow2(new_cap * 2)
        if need_ht > len(self._ht_keys):
            self._ht_keys = np.full(need_ht, _HT_EMPTY, dtype=np.int64)
            self._ht_slots = np.full(need_ht, -1, dtype=np.int32)
            self._ht_dirty = True

    def _sync_ht(self) -> None:
        if not self._ht_dirty:
            return
        need_ht = _next_pow2(max(self._capacity, len(self._key_to_slot) + 1) * 2)
        if need_ht > len(self._ht_keys):
            self._ht_keys = np.full(need_ht, _HT_EMPTY, dtype=np.int64)
            self._ht_slots = np.full(need_ht, -1, dtype=np.int32)
        if NATIVE is not None and hasattr(NATIVE, "planar_ht_rebuild_fast"):
            NATIVE.planar_ht_rebuild_fast(self._active, self._keys, self._count, self._ht_keys, self._ht_slots)
        self._ht_dirty = False

    def _mark_cell_tile_dirty(self, iu: int, iv: int) -> None:
        tu = iu >> _TILE_SHIFT
        tv = iv >> _TILE_SHIFT
        self._dirty_tiles.add((tu, tv))
        on_u0 = (iu & _TILE_MASK) == 0
        on_v0 = (iv & _TILE_MASK) == 0
        if on_u0:
            self._dirty_tiles.add((tu - 1, tv))
        if on_v0:
            self._dirty_tiles.add((tu, tv - 1))
        if on_u0 and on_v0:
            self._dirty_tiles.add((tu - 1, tv - 1))

    def _mark_slots_tiles_dirty(self, slots: np.ndarray) -> None:
        if len(slots) == 0:
            return
        if (
            NATIVE is not None
            and hasattr(NATIVE, "planar_mesh_snapshot_ht_fast")
            and not self._tile_cache
        ):
            return
        iu_s = self._iu[slots].astype(np.int64)
        iv_s = self._iv[slots].astype(np.int64)
        tu = iu_s >> _TILE_SHIFT
        tv = iv_s >> _TILE_SHIFT
        on_u0 = (iu_s & _TILE_MASK) == 0
        on_v0 = (iv_s & _TILE_MASK) == 0
        dirty = self._dirty_tiles

        def _add_tile_pairs(u_arr: np.ndarray, v_arr: np.ndarray) -> None:
            if len(u_arr) == 0:
                return
            packed = ((u_arr + _CELL_COORD_BIAS) << 32) | ((v_arr + _CELL_COORD_BIAS) & _CELL_COORD_MASK)
            uniq = np.unique(packed)
            uu_u = ((uniq >> 32) - _CELL_COORD_BIAS).tolist()
            vv_u = ((uniq & _CELL_COORD_MASK) - _CELL_COORD_BIAS).tolist()
            for ut, vt in zip(uu_u, vv_u):
                dirty.add((ut, vt))

        _add_tile_pairs(tu, tv)
        if np.any(on_u0):
            _add_tile_pairs(tu[on_u0] - 1, tv[on_u0])
        if np.any(on_v0):
            _add_tile_pairs(tu[on_v0], tv[on_v0] - 1)
        both = on_u0 & on_v0
        if np.any(both):
            _add_tile_pairs(tu[both] - 1, tv[both] - 1)

    def _refresh_fused_and_mark_dirty(self, slots: np.ndarray) -> None:
        if len(slots) == 0:
            return
        total_w = self._base_w[slots] + self._win_w[slots]
        total_obs = self._base_obs_count[slots] + self._win_obs_count[slots]
        self._total_obs[slots] = total_obs
        pos = total_w > 0.0
        if np.any(pos):
            sp = slots[pos]
            wp = total_w[pos]
            self._fused_xyz[sp] = (self._base_xyz[sp] + self._win_xyz[sp]) / wp[:, None]
            rgb_f = (
                self._base_rgb[sp].astype(np.float64) + self._win_rgb[sp].astype(np.float64)
            ) / wp[:, None]
            self._fused_rgb[sp] = np.clip(np.rint(rgb_f), 0.0, 255.0).astype(np.uint8)
        self._mark_slots_tiles_dirty(slots)

    def _free_slot(self, slot: int) -> None:
        if not self._active[slot]:
            return
        iu = int(self._iu[slot])
        iv = int(self._iv[slot])
        tile = (iu >> _TILE_SHIFT, iv >> _TILE_SHIFT)
        tset = self._tile_to_slots.get(tile)
        if tset is not None:
            tset.discard(slot)
            if not tset:
                del self._tile_to_slots[tile]
        self._mark_cell_tile_dirty(iu, iv)

        key = int(self._keys[slot])
        self._key_to_slot.pop(key, None)
        self._ht_dirty = True
        self._active[slot] = False
        self._base_w[slot] = 0.0
        self._win_w[slot] = 0.0
        self._base_xyz[slot] = 0.0
        self._win_xyz[slot] = 0.0
        self._base_rgb[slot] = 0.0
        self._win_rgb[slot] = 0.0
        self._base_obs_count[slot] = 0
        self._win_obs_count[slot] = 0
        self._total_obs[slot] = 0
        self._base_last_ts[slot] = -1
        self._last_ts[slot] = -1
        self._last_update_seq[slot] = 0
        self._free_space_conflicts[slot] = 0
        self._free_slots.append(slot)

    def _active_counts(self) -> tuple[int, int]:
        n = self._count
        if n == 0:
            return 0, 0
        active = self._active[:n]
        total_obs = self._total_obs[:n]
        committed = int(np.count_nonzero(active & (total_obs >= self._config.min_observations_to_commit)))
        tentative = int(np.count_nonzero(active & (total_obs < self._config.min_observations_to_commit)))
        return committed, tentative

    def _prune_expired_tentative(self, protected_keys: set[int] | None = None) -> int:
        n = self._count
        if n == 0:
            return 0
        active = self._active[:n]
        total_obs = self._total_obs[:n]
        expired = np.flatnonzero(
            active
            & (total_obs < self._config.min_observations_to_commit)
            & ((self._update_seq - self._last_update_seq[:n]) > self._config.tentative_ttl_updates)
        )
        pruned = 0
        for slot in expired:
            s = int(slot)
            if protected_keys and int(self._keys[s]) in protected_keys:
                continue
            self._free_slot(s)
            pruned += 1
        return pruned

    def _ensure_capacity_for(self, num_new_cells: int, protected_keys: set[int] | None = None) -> None:
        if num_new_cells <= 0:
            return
        available = len(self._free_slots) + (self._config.max_cells - self._count)
        if available < num_new_cells:
            self._prune_expired_tentative(protected_keys=protected_keys)
            available = len(self._free_slots) + (self._config.max_cells - self._count)
        if available < num_new_cells:
            deficit = num_new_cells - available
            n = self._count
            active_slots = np.flatnonzero(self._active[:n])
            if protected_keys and len(active_slots) > 0:
                keep_evictable = np.array(
                    [int(self._keys[s]) not in protected_keys for s in active_slots],
                    dtype=bool,
                )
                active_slots = active_slots[keep_evictable]
            if len(active_slots) > 0:
                total_obs = self._total_obs[active_slots]
                is_committed = (total_obs >= self._config.min_observations_to_commit).astype(np.int64)
                order = np.lexsort((
                    self._last_ts[active_slots],
                    self._last_update_seq[active_slots],
                    total_obs,
                    is_committed,
                ))
                for slot in active_slots[order[:deficit]]:
                    self._free_slot(int(slot))
        needed_total = self._count + max(0, num_new_cells - len(self._free_slots))
        self._grow_storage(needed_total)

    def _allocate_slot(self, key: int, iu: int, iv: int) -> int:
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            if self._count >= self._capacity:
                self._grow_storage(self._count + 1)
            slot = self._count
            self._count += 1
        self._active[slot] = True
        self._keys[slot] = key
        self._iu[slot] = iu
        self._iv[slot] = iv
        self._base_w[slot] = 0.0
        self._win_w[slot] = 0.0
        self._base_xyz[slot] = 0.0
        self._win_xyz[slot] = 0.0
        self._base_rgb[slot] = 0.0
        self._win_rgb[slot] = 0.0
        self._base_obs_count[slot] = 0
        self._win_obs_count[slot] = 0
        self._total_obs[slot] = 0
        self._base_last_ts[slot] = -1
        self._last_ts[slot] = -1
        self._last_update_seq[slot] = 0
        self._free_space_conflicts[slot] = 0
        self._key_to_slot[key] = slot
        self._ht_dirty = True
        tile = (iu >> _TILE_SHIFT, iv >> _TILE_SHIFT)
        tset = self._tile_to_slots.get(tile)
        if tset is None:
            self._tile_to_slots[tile] = {slot}
        else:
            tset.add(slot)
        return slot

    def _plane_compatible(self, normal: np.ndarray, offset_m: float) -> bool:
        assert self._plane_normal is not None and self._plane_offset_m is not None
        cos_ang = float(np.clip(np.dot(normal, self._plane_normal), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_ang))
        if angle_deg > self._config.max_plane_angle_change_deg + 1e-9:
            return False
        anchor = self._plane_anchor_m if self._plane_anchor_m is not None else (-self._plane_offset_m * self._plane_normal)
        shift_m = abs(float(anchor @ normal) + float(offset_m))
        return shift_m <= self._config.max_plane_offset_change_m + 1e-9

    def _record_seen(self, observation_id: str, frame_from_camera: np.ndarray | None, timestamp_ns: int) -> None:
        self._seen_observations[observation_id] = _SeenRecord(
            frame_from_camera=None if frame_from_camera is None else frame_from_camera.copy(),
            timestamp_ns=timestamp_ns,
        )
        self._seen_observations.move_to_end(observation_id)
        while len(self._seen_observations) > self._config.max_seen_observation_ids:
            self._seen_observations.popitem(last=False)

    def _record_provenance(self, prov: ObservationProvenance) -> None:
        self._provenance[prov.observation_id] = prov
        self._provenance.move_to_end(prov.observation_id)
        while len(self._provenance) > self._config.max_provenance_observations:
            self._provenance.popitem(last=False)

    def _project_and_aggregate_cells(
        self,
        points_frame_m: np.ndarray,
        colors_rgb_u8: np.ndarray,
        *,
        camera_origin_frame_m: np.ndarray | None,
        observation_weight: float,
        selected_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Filter samples onto the reference plane and aggregate into weighted cell means.

        Returns ``(cell_keys, cell_pts, cell_rgb, cell_w, inlier_count, budget_exceeded)``.
        """
        assert self._plane_normal is not None and self._plane_offset_m is not None
        assert self._basis_u is not None and self._basis_v is not None

        empty_k = np.empty(0, dtype=np.int64)
        empty_p = np.empty((0, 3), dtype=np.float64)
        empty_c = np.empty((0, 3), dtype=np.float32)
        empty_w = np.empty(0, dtype=np.float64)

        if len(points_frame_m) == 0:
            return empty_k, empty_p, empty_c, empty_w, 0, False

        if (
            self._uv_origin is not None
            and NATIVE is not None
            and hasattr(NATIVE, "planar_project_and_aggregate_fast")
        ):
            pts_c = (
                points_frame_m
                if (
                    points_frame_m.flags.c_contiguous
                    and points_frame_m.dtype in (np.float64, np.float32)
                )
                else np.ascontiguousarray(points_frame_m, dtype=np.float64)
            )
            rgb_c = (
                colors_rgb_u8
                if (colors_rgb_u8.flags.c_contiguous and colors_rgb_u8.dtype == np.uint8)
                else np.ascontiguousarray(colors_rgb_u8, dtype=np.uint8)
            )
            sel_c = (
                None
                if selected_mask is None
                else (
                    selected_mask
                    if (selected_mask.flags.c_contiguous and selected_mask.dtype == np.bool_)
                    else np.ascontiguousarray(selected_mask, dtype=np.bool_)
                )
            )
            has_cam = 1 if camera_origin_frame_m is not None else 0
            cam_c = (
                np.ascontiguousarray(camera_origin_frame_m, dtype=np.float64)
                if has_cam
                else self._plane_normal
            )
            cos_limit = (
                math.cos(math.radians(self._config.maximum_grazing_angle_deg))
                if self._config.maximum_grazing_angle_deg < 90.0
                else 0.0
            )
            res = NATIVE.planar_project_and_aggregate_fast(
                pts_c,
                rgb_c,
                np.ascontiguousarray(self._plane_normal, dtype=np.float64),
                np.ascontiguousarray(self._basis_u, dtype=np.float64),
                np.ascontiguousarray(self._basis_v, dtype=np.float64),
                np.ascontiguousarray(self._uv_origin, dtype=np.float64),
                has_cam,
                float(self._plane_offset_m),
                float(self._config.mesh.maximum_plane_residual_m),
                float(cos_limit),
                float(self._config.mesh.cell_size_m),
                float(observation_weight),
                1 if self._config.weighting_mode == "sensor_physics" else 0,
                int(self._config.specular_saturation_threshold),
                float(self._config.specular_rgb_weight_scale),
                cam_c,
                int(self._config.mesh.maximum_grid_cells),
                int(self._config.max_cells),
                sel_c,
            )
            if res is not NotImplemented:
                return res

        if selected_mask is not None:
            points_frame_m = points_frame_m[selected_mask]
            colors_rgb_u8 = colors_rgb_u8[selected_mask]
            if len(points_frame_m) == 0:
                return empty_k, empty_p, empty_c, empty_w, 0, False

        valid = np.isfinite(points_frame_m).all(axis=1)
        residual = np.abs(points_frame_m @ self._plane_normal + self._plane_offset_m)
        valid &= residual <= self._config.mesh.maximum_plane_residual_m

        idx_v = np.flatnonzero(valid)
        if len(idx_v) == 0:
            return empty_k, empty_p, empty_c, empty_w, 0, False

        cos_theta: np.ndarray | None = None
        ray_norm: np.ndarray | None = None
        if camera_origin_frame_m is not None:
            rays = camera_origin_frame_m[None, :] - points_frame_m[idx_v]
            ray_norm_all = np.linalg.norm(rays, axis=1)
            cos_all = np.abs((rays @ self._plane_normal) / np.maximum(ray_norm_all, 1e-12))
            if self._config.maximum_grazing_angle_deg < 90.0:
                cos_limit = math.cos(math.radians(self._config.maximum_grazing_angle_deg))
                good_ray = (ray_norm_all > 1e-6) & (cos_all >= cos_limit)
                inlier_idx = idx_v[good_ray]
                cos_theta = cos_all[good_ray]
                ray_norm = ray_norm_all[good_ray]
            else:
                inlier_idx = idx_v
                cos_theta = cos_all
                ray_norm = ray_norm_all
        else:
            inlier_idx = idx_v

        inlier_count = int(len(inlier_idx))
        if inlier_count == 0:
            return empty_k, empty_p, empty_c, empty_w, 0, False

        inlier_pts = points_frame_m[inlier_idx]
        inlier_rgb = colors_rgb_u8[inlier_idx]

        uv = inlier_pts @ np.column_stack((self._basis_u, self._basis_v))
        if self._uv_origin is None:
            cell_size = self._config.mesh.cell_size_m
            self._uv_origin = np.floor(np.median(uv, axis=0) / cell_size) * cell_size
            mean_pt = np.mean(inlier_pts, axis=0)
            self._plane_anchor_m = mean_pt - (float(mean_pt @ self._plane_normal) + self._plane_offset_m) * self._plane_normal

        scaled = np.floor((uv - self._uv_origin[None, :]) / self._config.mesh.cell_size_m + 1e-9)
        if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= _CELL_COORD_LIMIT):
            return empty_k, empty_p, empty_c, empty_w, inlier_count, True

        cells_i = scaled.astype(np.int64)
        span = cells_i.max(axis=0) - cells_i.min(axis=0) + 1
        if int(span[0]) * int(span[1]) > self._config.mesh.maximum_grid_cells:
            return empty_k, empty_p, empty_c, empty_w, inlier_count, True

        # Per-sample geometric and RGB fusion weights.
        if (
            self._config.weighting_mode == "sensor_physics"
            and cos_theta is not None
            and ray_norm is not None
        ):
            z_eff = np.maximum(ray_norm, 0.2)
            w_geo = np.clip(cos_theta / (z_eff * z_eff), 0.05, 8.0)
            if self._config.specular_saturation_threshold <= 255:
                is_spec = np.min(inlier_rgb, axis=1) >= self._config.specular_saturation_threshold
                w_rgb = np.where(is_spec, w_geo * self._config.specular_rgb_weight_scale, w_geo)
            else:
                w_rgb = w_geo
        else:
            w_geo = np.ones(inlier_count, dtype=np.float64)
            w_rgb = w_geo

        keys = _pack_cell_keys(cells_i[:, 0], cells_i[:, 1])

        if NATIVE is not None and hasattr(NATIVE, "planar_cell_aggregate_fast"):
            uniq_keys, cell_pts, cell_rgb, cell_w = NATIVE.planar_cell_aggregate_fast(
                np.ascontiguousarray(inlier_pts, dtype=np.float64),
                np.ascontiguousarray(inlier_rgb, dtype=np.uint8),
                np.ascontiguousarray(w_geo, dtype=np.float64),
                np.ascontiguousarray(w_rgb, dtype=np.float64),
                np.ascontiguousarray(keys, dtype=np.int64),
                float(observation_weight),
                int(self._config.max_cells),
            )
            return uniq_keys, cell_pts, cell_rgb, cell_w, inlier_count, False

        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        uniq_keys, first_idx, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

        wg_ord = w_geo[order]
        wr_ord = w_rgb[order]
        sum_wg = np.add.reduceat(wg_ord, first_idx)
        sum_wr = np.add.reduceat(wr_ord, first_idx)
        sum_pts = np.add.reduceat(inlier_pts[order] * wg_ord[:, None], first_idx, axis=0)
        sum_rgb = np.add.reduceat(inlier_rgb[order].astype(np.float64) * wr_ord[:, None], first_idx, axis=0)
        cell_pts = sum_pts / sum_wg[:, None]
        cell_rgb = (sum_rgb / sum_wr[:, None]).astype(np.float32)
        cell_w = float(observation_weight) * (sum_wg / counts.astype(np.float64))
        if len(uniq_keys) > self._config.max_cells:
            uniq_keys = uniq_keys[: self._config.max_cells]
            cell_pts = cell_pts[: self._config.max_cells]
            cell_rgb = cell_rgb[: self._config.max_cells]
            cell_w = cell_w[: self._config.max_cells]
        return uniq_keys, cell_pts, cell_rgb, cell_w, inlier_count, False

    def _add_window_contribution(
        self,
        cell_keys: np.ndarray,
        cell_pts: np.ndarray,
        cell_rgb: np.ndarray,
        cell_w: np.ndarray,
        *,
        timestamp_ns: int,
    ) -> int:
        if len(cell_keys) == 0:
            return 0
        if len(cell_keys) > self._config.max_cells:
            cell_keys = cell_keys[: self._config.max_cells]
            cell_pts = cell_pts[: self._config.max_cells]
            cell_rgb = cell_rgb[: self._config.max_cells]
            cell_w = cell_w[: self._config.max_cells]

        use_native = (
            NATIVE is not None
            and hasattr(NATIVE, "planar_ht_lookup_fast")
            and hasattr(NATIVE, "planar_window_accumulate_fast")
        )
        if use_native:
            self._sync_ht()
            keys_c = np.ascontiguousarray(cell_keys, dtype=np.int64)
            slots, missing = NATIVE.planar_ht_lookup_fast(self._ht_keys, self._ht_slots, keys_c)
            if missing > 0:
                key_list = keys_c.tolist()
                protected_keys = set(key_list)
                self._ensure_capacity_for(int(missing), protected_keys=protected_keys)
                miss_idx = np.flatnonzero(slots < 0)
                miss_keys = keys_c[miss_idx]
                iu_m, iv_m = _unpack_cell_keys(miss_keys)
                for idx_local, key_val, iu_val, iv_val in zip(
                    miss_idx.tolist(), miss_keys.tolist(), iu_m.tolist(), iv_m.tolist()
                ):
                    slots[idx_local] = self._allocate_slot(key_val, iu_val, iv_val)
            NATIVE.planar_window_accumulate_fast(
                slots,
                self._win_w,
                self._win_xyz,
                self._win_rgb,
                self._win_obs_count,
                self._last_ts,
                self._last_update_seq,
                self._free_space_conflicts,
                np.ascontiguousarray(cell_pts, dtype=np.float64),
                np.ascontiguousarray(cell_rgb, dtype=np.float32),
                np.ascontiguousarray(cell_w, dtype=np.float64),
                self._base_w,
                self._base_xyz,
                self._base_rgb,
                self._base_obs_count,
                self._fused_xyz,
                self._fused_rgb,
                self._total_obs,
                int(timestamp_ns),
                int(self._update_seq),
            )
            self._mark_slots_tiles_dirty(slots)
            return int(len(slots))

        key_list = cell_keys.tolist()
        protected_keys = set(key_list)
        new_count = sum(1 for k in key_list if k not in self._key_to_slot)
        self._ensure_capacity_for(new_count, protected_keys=protected_keys)

        iu_arr, iv_arr = _unpack_cell_keys(cell_keys)
        slots = np.empty(len(key_list), dtype=np.int32)
        for i, key in enumerate(key_list):
            slot = self._key_to_slot.get(key)
            if slot is None:
                slot = self._allocate_slot(key, int(iu_arr[i]), int(iv_arr[i]))
            slots[i] = slot

        self._win_w[slots] += cell_w
        self._win_xyz[slots] += cell_w[:, None] * cell_pts
        self._win_rgb[slots] += cell_w.astype(np.float32)[:, None] * cell_rgb
        self._win_obs_count[slots] += 1
        self._last_ts[slots] = np.maximum(self._last_ts[slots], timestamp_ns)
        self._last_update_seq[slots] = self._update_seq
        self._free_space_conflicts[slots] = 0
        self._refresh_fused_and_mark_dirty(slots)
        return int(len(slots))

    def _remove_window_contribution(self, obs: _RetainedObservation) -> None:
        if len(obs.cell_keys) == 0:
            return
        touched_ts_slots: list[int] = []
        modified_slots: list[int] = []
        for i, key in enumerate(obs.cell_keys.tolist()):
            slot = self._key_to_slot.get(key)
            if slot is None or not self._active[slot]:
                continue
            w = float(obs.cell_w[i])
            self._win_obs_count[slot] = max(0, int(self._win_obs_count[slot]) - 1)
            if self._win_obs_count[slot] == 0:
                self._win_w[slot] = 0.0
                self._win_xyz[slot] = 0.0
                self._win_rgb[slot] = 0.0
            else:
                self._win_w[slot] = max(0.0, float(self._win_w[slot]) - w)
                self._win_xyz[slot] -= w * obs.cell_pts[i]
                self._win_rgb[slot] -= np.float32(w) * obs.cell_rgb[i]
            if self._base_obs_count[slot] + self._win_obs_count[slot] <= 0:
                self._free_slot(slot)
            else:
                modified_slots.append(slot)
                if self._last_ts[slot] == obs.timestamp_ns:
                    touched_ts_slots.append(slot)

        if modified_slots:
            self._refresh_fused_and_mark_dirty(np.asarray(modified_slots, dtype=np.int32))

        if touched_ts_slots:
            touched_keys = {int(self._keys[s]): s for s in touched_ts_slots}
            for s in touched_ts_slots:
                self._last_ts[s] = self._base_last_ts[s]
            for other in self._retained_observations.values():
                if other.observation_id == obs.observation_id or len(other.cell_keys) == 0:
                    continue
                for k in other.cell_keys.tolist():
                    s = touched_keys.get(k)
                    if s is not None and other.timestamp_ns > self._last_ts[s]:
                        self._last_ts[s] = other.timestamp_ns

    def _compact_oldest_observation(self) -> None:
        _, old_obs = self._retained_observations.popitem(last=False)
        self._compacted_observation_count += 1
        if len(old_obs.cell_keys) == 0:
            return
        max_w = self._config.max_cell_weight
        if (
            NATIVE is not None
            and hasattr(NATIVE, "planar_ht_lookup_fast")
            and hasattr(NATIVE, "planar_compact_obs_fast")
        ):
            self._sync_ht()
            slots, _ = NATIVE.planar_ht_lookup_fast(
                self._ht_keys,
                self._ht_slots,
                np.ascontiguousarray(old_obs.cell_keys, dtype=np.int64),
            )
            capped = NATIVE.planar_compact_obs_fast(
                slots,
                self._active,
                np.ascontiguousarray(old_obs.cell_pts, dtype=np.float64),
                np.ascontiguousarray(old_obs.cell_rgb, dtype=np.float32),
                np.ascontiguousarray(old_obs.cell_w, dtype=np.float64),
                self._win_w,
                self._win_xyz,
                self._win_rgb,
                self._win_obs_count,
                self._base_w,
                self._base_xyz,
                self._base_rgb,
                self._base_obs_count,
                self._base_last_ts,
                self._fused_xyz,
                self._fused_rgb,
                self._total_obs,
                float(max_w),
                int(old_obs.timestamp_ns),
            )
            if len(capped) > 0:
                self._mark_slots_tiles_dirty(capped)
            return

        capped_slots: list[int] = []
        for i, key in enumerate(old_obs.cell_keys.tolist()):
            slot = self._key_to_slot.get(key)
            if slot is None or not self._active[slot]:
                continue
            w = float(old_obs.cell_w[i])
            self._win_obs_count[slot] = max(0, int(self._win_obs_count[slot]) - 1)
            if self._win_obs_count[slot] == 0:
                self._win_w[slot] = 0.0
                self._win_xyz[slot] = 0.0
                self._win_rgb[slot] = 0.0
            else:
                self._win_w[slot] = max(0.0, float(self._win_w[slot]) - w)
                self._win_xyz[slot] -= w * old_obs.cell_pts[i]
                self._win_rgb[slot] -= np.float32(w) * old_obs.cell_rgb[i]

            new_base_w = float(self._base_w[slot]) + w
            new_base_xyz = self._base_xyz[slot] + w * old_obs.cell_pts[i]
            new_base_rgb = self._base_rgb[slot].astype(np.float64) + w * old_obs.cell_rgb[i].astype(np.float64)
            if new_base_w > max_w:
                scale = max_w / new_base_w
                new_base_w = max_w
                new_base_xyz *= scale
                new_base_rgb *= scale
                capped_slots.append(slot)
            self._base_w[slot] = new_base_w
            self._base_xyz[slot] = new_base_xyz
            self._base_rgb[slot] = new_base_rgb.astype(np.float32)
            self._base_obs_count[slot] += 1
            if old_obs.timestamp_ns > self._base_last_ts[slot]:
                self._base_last_ts[slot] = old_obs.timestamp_ns

        if capped_slots:
            self._refresh_fused_and_mark_dirty(np.asarray(capped_slots, dtype=np.int32))

    def _apply_free_space_carving(
        self,
        depth_m: np.ndarray,
        selected_mask: np.ndarray | None,
        camera_matrix: np.ndarray,
        frame_from_camera: np.ndarray,
    ) -> int:
        """Carve cells clearly contradicted by free space (measured depth behind cell).

        Occlusions (measured depth in front of cell), invalid depth, and excluded
        mask regions never carve static cells.
        """
        n = self._count
        if n == 0 or not self._config.enable_free_space_carving:
            return 0
        if NATIVE is not None and hasattr(NATIVE, "planar_free_space_carve_fast"):
            d_c = (
                depth_m
                if (depth_m.flags.c_contiguous and depth_m.dtype in (np.float64, np.float32))
                else np.ascontiguousarray(depth_m, dtype=np.float64)
            )
            if selected_mask is None:
                s_c = None
            else:
                s_c = (
                    selected_mask
                    if (selected_mask.flags.c_contiguous and selected_mask.dtype == np.bool_)
                    else np.ascontiguousarray(selected_mask, dtype=np.bool_)
                )
            to_evict = NATIVE.planar_free_space_carve_fast(
                self._active,
                self._base_w,
                self._win_w,
                self._fused_xyz,
                self._total_obs,
                self._free_space_conflicts,
                int(n),
                d_c,
                s_c,
                np.ascontiguousarray(frame_from_camera[:3, :3], dtype=np.float64),
                np.ascontiguousarray(frame_from_camera[:3, 3], dtype=np.float64),
                float(camera_matrix[0, 0]),
                float(camera_matrix[1, 1]),
                float(camera_matrix[0, 2]),
                float(camera_matrix[1, 2]),
                float(self._config.minimum_depth_m),
                float(self._config.maximum_depth_m),
                float(self._config.free_space_carve_margin_m),
                int(self._config.min_observations_to_commit),
                int(self._config.free_space_conflict_threshold),
            )
            if to_evict is not NotImplemented:
                for slot in to_evict:
                    self._free_slot(int(slot))
                return int(len(to_evict))

        if selected_mask is None:
            selected_mask = np.ones(depth_m.shape, dtype=bool)

        active_slots = np.flatnonzero(self._active[:n])
        if len(active_slots) == 0:
            return 0

        total_w = self._base_w[active_slots] + self._win_w[active_slots]
        valid_w = total_w > 0.0
        if not np.any(valid_w):
            return 0
        active_slots = active_slots[valid_w]
        pts_world = self._fused_xyz[active_slots]

        rot = frame_from_camera[:3, :3]
        trans = frame_from_camera[:3, 3]
        pts_cam = (pts_world - trans[None, :]) @ rot
        z_c = pts_cam[:, 2]
        in_z = (z_c >= self._config.minimum_depth_m) & (z_c <= self._config.maximum_depth_m)
        if not np.any(in_z):
            return 0

        cand_slots = active_slots[in_z]
        pts_cam = pts_cam[in_z]
        z_c = z_c[in_z]

        h, w_img = depth_m.shape
        u = np.rint(camera_matrix[0, 0] * (pts_cam[:, 0] / z_c) + camera_matrix[0, 2]).astype(np.int64)
        v = np.rint(camera_matrix[1, 1] * (pts_cam[:, 1] / z_c) + camera_matrix[1, 2]).astype(np.int64)
        in_img = (u >= 1) & (u < w_img - 1) & (v >= 1) & (v < h - 1)
        if not np.any(in_img):
            return 0

        cand_slots = cand_slots[in_img]
        u = u[in_img]
        v = v[in_img]
        z_c = z_c[in_img]

        unmasked = selected_mask[v, u]
        if not np.any(unmasked):
            return 0
        cand_slots = cand_slots[unmasked]
        u = u[unmasked]
        v = v[unmasked]
        z_c = z_c[unmasked]

        meas_d = depth_m[v, u]
        valid_d = np.isfinite(meas_d) & (meas_d >= self._config.minimum_depth_m) & (meas_d <= self._config.maximum_depth_m)
        if not np.any(valid_d):
            return 0
        cand_slots = cand_slots[valid_d]
        u = u[valid_d]
        v = v[valid_d]
        z_c = z_c[valid_d]
        meas_d = meas_d[valid_d]

        margin = self._config.free_space_carve_margin_m
        corroborated = np.abs(meas_d - z_c) <= margin
        if np.any(corroborated):
            self._free_space_conflicts[cand_slots[corroborated]] = 0

        behind = meas_d > (z_c + margin)
        if not np.any(behind):
            return 0

        behind_slots = cand_slots[behind]
        ub = u[behind]
        vb = v[behind]
        zb = z_c[behind]

        nbr_ok = np.ones(len(behind_slots), dtype=bool)
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dv == 0 and du == 0:
                    continue
                nd = depth_m[vb + dv, ub + du]
                nbr_ok &= (
                    selected_mask[vb + dv, ub + du]
                    & np.isfinite(nd)
                    & (nd > zb + margin)
                    & (nd <= self._config.maximum_depth_m)
                )
        carve_cand = behind_slots[nbr_ok]
        if len(carve_cand) == 0:
            return 0

        self._free_space_conflicts[carve_cand] += 1
        total_obs = self._total_obs[carve_cand]
        committed_thresh = max(3, self._config.free_space_conflict_threshold + 1)
        thresholds = np.where(
            total_obs >= self._config.min_observations_to_commit,
            committed_thresh,
            self._config.free_space_conflict_threshold,
        )
        to_evict = carve_cand[self._free_space_conflicts[carve_cand] >= thresholds]
        for slot in to_evict:
            self._free_slot(int(slot))
        return int(len(to_evict))

    def _apply_pose_correction(
        self,
        obs: _RetainedObservation,
        new_frame_from_camera: np.ndarray,
        *,
        timestamp_ns: int | None = None,
    ) -> SurfaceUpdateResult:
        if obs.frame_from_camera is None or obs.camera_points_m is None or obs.camera_colors_rgb_u8 is None:
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="pose_correction_not_available",
                version=self._version,
                epoch=self._epoch,
                observation_id=obs.observation_id,
                timestamp_ns=obs.timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )

        if np.allclose(obs.frame_from_camera, new_frame_from_camera, atol=1e-7, rtol=0.0):
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="pose_unchanged",
                version=self._version,
                epoch=self._epoch,
                observation_id=obs.observation_id,
                timestamp_ns=obs.timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )

        pts_cam64 = obs.camera_points_m.astype(np.float64)
        rot = new_frame_from_camera[:3, :3]
        trans = new_frame_from_camera[:3, 3]
        pts_frame = pts_cam64 @ rot.T + trans[None, :]
        new_keys, new_pts, new_rgb, new_w, inlier_count, budget_exceeded = self._project_and_aggregate_cells(
            pts_frame,
            obs.camera_colors_rgb_u8,
            camera_origin_frame_m=trans,
            observation_weight=obs.observation_weight,
        )
        if budget_exceeded:
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="grid_budget_exceeded",
                version=self._version,
                epoch=self._epoch,
                observation_id=obs.observation_id,
                timestamp_ns=obs.timestamp_ns,
                inlier_sample_count=inlier_count,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )

        self._update_seq += 1
        self._remove_window_contribution(obs)

        eff_ts = obs.timestamp_ns if timestamp_ns is None else timestamp_ns
        obs.frame_from_camera = new_frame_from_camera.copy()
        obs.timestamp_ns = eff_ts
        obs.cell_keys = new_keys
        obs.cell_pts = new_pts
        obs.cell_rgb = new_rgb
        obs.cell_w = new_w
        obs.pose_revision += 1

        updated = self._add_window_contribution(
            new_keys,
            new_pts,
            new_rgb,
            new_w,
            timestamp_ns=eff_ts,
        )
        self._record_seen(obs.observation_id, new_frame_from_camera, eff_ts)
        self._record_provenance(
            ObservationProvenance(
                observation_id=obs.observation_id,
                timestamp_ns=eff_ts,
                camera_id=obs.camera_id,
                sample_count=len(new_keys),
                pose_revision=obs.pose_revision,
            )
        )
        if self._latest_observation_timestamp_ns is None or eff_ts > self._latest_observation_timestamp_ns:
            self._latest_observation_timestamp_ns = eff_ts
        self._version += 1
        self._map_updated_monotonic_ns = time.monotonic_ns()
        self._cached_mesh = None

        committed, tentative = self._active_counts()
        return SurfaceUpdateResult(
            accepted=True,
            status="pose_corrected",
            version=self._version,
            epoch=self._epoch,
            observation_id=obs.observation_id,
            timestamp_ns=eff_ts,
            inlier_sample_count=inlier_count,
            updated_cell_count=updated,
            committed_cell_count=committed,
            tentative_cell_count=tentative,
        )

    @_locked
    def carve_free_space(
        self,
        depth_m,
        camera_matrix,
        *,
        frame_from_camera,
        include_mask=None,
        exclude_mask=None,
    ) -> int:
        """Apply calibrated depth free-space carving directly to active persistent cells.

        Enables point-cloud integration pipelines (such as GPU-backprojected RGB-D
        streams using :meth:`integrate_points`) to invoke the C11 3x3-verified
        free-space carver without duplicating CPU depth backprojection.
        Returns the number of evicted cells.
        """
        depth = np.asarray(depth_m)
        if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
            raise ValueError("depth_m must be a non-empty 2D depth image")
        if not np.issubdtype(depth.dtype, np.floating):
            depth = depth.astype(np.float64)
        K = _validate_intrinsics(camera_matrix)
        T_fc = _validate_se3(frame_from_camera, "frame_from_camera")

        selected: np.ndarray | None = None
        if include_mask is not None or exclude_mask is not None:
            selected = np.ones(depth.shape, dtype=bool)
            if include_mask is not None:
                inc = np.asarray(include_mask)
                if inc.shape != depth.shape or inc.dtype != bool:
                    raise ValueError("include_mask must be a boolean HxW image matching depth")
                selected &= inc
            if exclude_mask is not None:
                exc = np.asarray(exclude_mask)
                if exc.shape != depth.shape or exc.dtype != bool:
                    raise ValueError("exclude_mask must be a boolean HxW image matching depth")
                selected &= ~exc

        carved = self._apply_free_space_carving(depth, selected, K, T_fc)
        if carved > 0:
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._cached_mesh = None
        return carved

    @_locked
    def correct_pose(
        self,
        observation_id: str,
        frame_from_camera,
        *,
        timestamp_ns: int | None = None,
    ) -> SurfaceUpdateResult:
        """Revise the camera pose of a previously integrated observation within the retention window."""
        observation_id = _validate_nonempty_str(observation_id, "observation_id")
        new_pose = _validate_se3(frame_from_camera, "frame_from_camera")
        eff_ts = None if timestamp_ns is None else _validate_timestamp_ns(timestamp_ns)

        obs = self._retained_observations.get(observation_id)
        if obs is not None:
            return self._apply_pose_correction(obs, new_pose, timestamp_ns=eff_ts)

        committed, tentative = self._active_counts()
        if observation_id in self._seen_observations:
            return SurfaceUpdateResult(
                accepted=False,
                status="pose_correction_out_of_window",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=eff_ts,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )
        return SurfaceUpdateResult(
            accepted=False,
            status="pose_correction_unknown_observation",
            version=self._version,
            epoch=self._epoch,
            observation_id=observation_id,
            timestamp_ns=eff_ts,
            committed_cell_count=committed,
            tentative_cell_count=tentative,
        )

    @_locked
    def correct_poses(
        self,
        poses_by_observation_id: Mapping[str, np.ndarray],
    ) -> dict[str, SurfaceUpdateResult]:
        """Batch-correct camera poses for multiple observation IDs."""
        return {
            obs_id: self.correct_pose(obs_id, pose)
            for obs_id, pose in poses_by_observation_id.items()
        }

    @_locked
    def integrate_rgbd(
        self,
        depth_m,
        color_rgb_u8,
        camera_matrix,
        *,
        frame_from_camera,
        plane_normal=None,
        plane_offset_m: float | None = None,
        plane_support_count: int | None = None,
        plane_rmse_m: float | None = None,
        coordinate_frame: str,
        timestamp_ns: int,
        observation_id: str,
        camera_id: str = "default",
        include_mask=None,
        exclude_mask=None,
        pixel_step: int | None = None,
        observation_weight: float = 1.0,
    ) -> SurfaceUpdateResult:
        """Integrate one calibrated registered RGB-D frame into the persistent planar surface."""
        coordinate_frame = _validate_nonempty_str(coordinate_frame, "coordinate_frame")
        observation_id = _validate_nonempty_str(observation_id, "observation_id")
        camera_id = _validate_nonempty_str(camera_id, "camera_id")
        timestamp_ns = _validate_timestamp_ns(timestamp_ns)
        if isinstance(observation_weight, bool) or not math.isfinite(observation_weight) or observation_weight <= 0.0:
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

        step = self._config.pixel_step if pixel_step is None else pixel_step
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) <= 0:
            raise ValueError("pixel_step must be a positive integer")
        step = int(step)

        selected = np.ones(depth.shape, dtype=bool)
        if include_mask is not None:
            inc = np.asarray(include_mask)
            if inc.shape != depth.shape or inc.dtype != bool:
                raise ValueError("include_mask must be a boolean HxW image matching depth")
            selected &= inc
        if exclude_mask is not None:
            exc = np.asarray(exclude_mask)
            if exc.shape != depth.shape or exc.dtype != bool:
                raise ValueError("exclude_mask must be a boolean HxW image matching depth")
            selected &= ~exc

        pending_reset = False
        if self._coordinate_frame is not None and coordinate_frame != self._coordinate_frame:
            if self._config.frame_change_policy == "reject":
                committed, tentative = self._active_counts()
                return SurfaceUpdateResult(
                    accepted=False,
                    status="coordinate_frame_changed",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                    committed_cell_count=committed,
                    tentative_cell_count=tentative,
                )
            pending_reset = True

        if not pending_reset:
            retained = self._retained_observations.get(observation_id)
            if retained is not None:
                if retained.frame_from_camera is not None and np.allclose(
                    retained.frame_from_camera, T_fc, atol=1e-7, rtol=0.0
                ):
                    committed, tentative = self._active_counts()
                    return SurfaceUpdateResult(
                        accepted=False,
                        status="duplicate_ignored",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                        committed_cell_count=committed,
                        tentative_cell_count=tentative,
                    )
                return self._apply_pose_correction(retained, T_fc, timestamp_ns=timestamp_ns)

            seen = self._seen_observations.get(observation_id)
            if seen is not None:
                committed, tentative = self._active_counts()
                if seen.frame_from_camera is not None and not np.allclose(
                    seen.frame_from_camera, T_fc, atol=1e-7, rtol=0.0
                ):
                    return SurfaceUpdateResult(
                        accepted=False,
                        status="pose_correction_out_of_window",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                        committed_cell_count=committed,
                        tentative_cell_count=tentative,
                    )
                return SurfaceUpdateResult(
                    accepted=False,
                    status="duplicate_ignored",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                    committed_cell_count=committed,
                    tentative_cell_count=tentative,
                )

            prev_cal = self._camera_calibrations.get(camera_id)
            if prev_cal is not None:
                prev_shape, prev_K = prev_cal
                if prev_shape != (h, w) or not np.allclose(prev_K, K, atol=1e-6, rtol=1e-6):
                    if self._config.calibration_change_policy == "reject":
                        committed, tentative = self._active_counts()
                        return SurfaceUpdateResult(
                            accepted=False,
                            status="camera_calibration_changed",
                            version=self._version,
                            epoch=self._epoch,
                            observation_id=observation_id,
                            timestamp_ns=timestamp_ns,
                            committed_cell_count=committed,
                            tentative_cell_count=tentative,
                        )
                    pending_reset = True

        plane_result = self._resolve_plane(
            plane_normal=plane_normal,
            plane_offset_m=plane_offset_m,
            plane_support_count=plane_support_count,
            plane_rmse_m=plane_rmse_m,
            pending_reset=pending_reset,
        )
        if plane_result[0] == "invalid_plane":
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="invalid_plane",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )
        if plane_result[0] == "plane_changed":
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="plane_changed",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )
        _, resolved_normal, resolved_offset, plane_triggered_reset = plane_result
        pending_reset = pending_reset or plane_triggered_reset

        if ((h - 1) // step + 1) * ((w - 1) // step + 1) > self._config.mesh.maximum_grid_cells:
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="grid_budget_exceeded",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )

        sampled_depth = depth[::step, ::step]
        valid_sub = (
            selected[::step, ::step]
            & np.isfinite(sampled_depth)
            & (sampled_depth >= self._config.minimum_depth_m)
            & (sampled_depth <= self._config.maximum_depth_m)
        )
        vv_sub, uu_sub = np.nonzero(valid_sub)
        vv = vv_sub * step
        uu = uu_sub * step

        if len(vv) > 0:
            pts_cam = _backproject_masked_depth(
                depth,
                selected,
                K,
                pixel_stride=step,
                minimum_depth_m=self._config.minimum_depth_m,
                maximum_depth_m=self._config.maximum_depth_m,
            )
            rgb_cam = color[vv, uu]
            pts_frame = pts_cam @ T_fc[:3, :3].T + T_fc[:3, 3][None, :]
        else:
            pts_cam = np.empty((0, 3), dtype=np.float64)
            rgb_cam = np.empty((0, 3), dtype=np.uint8)
            pts_frame = np.empty((0, 3), dtype=np.float64)

        return self._commit_observation(
            pts_frame=pts_frame,
            colors_rgb_u8=rgb_cam,
            pts_cam=pts_cam if self._config.retain_camera_points else None,
            frame_from_camera=T_fc,
            coordinate_frame=coordinate_frame,
            plane_normal=resolved_normal,
            plane_offset_m=resolved_offset,
            timestamp_ns=timestamp_ns,
            observation_id=observation_id,
            camera_id=camera_id,
            observation_weight=float(observation_weight),
            pending_reset=pending_reset,
            camera_calibration=((h, w), K),
            carving_inputs=(depth, selected, K, T_fc),
        )

    @_locked
    def integrate_points(
        self,
        points_m,
        colors_rgb_u8,
        *,
        plane_normal=None,
        plane_offset_m: float | None = None,
        plane_support_count: int | None = None,
        plane_rmse_m: float | None = None,
        coordinate_frame: str,
        timestamp_ns: int,
        observation_id: str,
        camera_id: str = "default",
        frame_from_camera=None,
        include_mask=None,
        exclude_mask=None,
        observation_weight: float = 1.0,
    ) -> SurfaceUpdateResult:
        """Integrate pre-registered common-frame 3D samples and RGB into the persistent surface."""
        if (
            isinstance(points_m, np.ndarray)
            and points_m.dtype in (np.float32, np.float64)
            and points_m.ndim == 2
            and points_m.shape[1] == 3
            and isinstance(colors_rgb_u8, np.ndarray)
            and colors_rgb_u8.dtype == np.uint8
            and colors_rgb_u8.shape == points_m.shape
        ):
            coordinate_frame = _validate_nonempty_str(coordinate_frame, "coordinate_frame")
            observation_id = _validate_nonempty_str(observation_id, "source_id")
            timestamp_ns = _validate_timestamp_ns(timestamp_ns)
            points, colors = points_m, colors_rgb_u8
        else:
            points, colors = _validate_samples(points_m, colors_rgb_u8, coordinate_frame, timestamp_ns, observation_id)
            timestamp_ns = int(timestamp_ns)
        camera_id = _validate_nonempty_str(camera_id, "camera_id")
        if isinstance(observation_weight, bool) or not math.isfinite(observation_weight) or observation_weight <= 0.0:
            raise ValueError("observation_weight must be finite and positive")

        T_fc = None if frame_from_camera is None else _validate_se3(frame_from_camera, "frame_from_camera")
        selected: np.ndarray | None = None
        if include_mask is not None:
            inc = np.asarray(include_mask)
            if inc.shape != (len(points),) or inc.dtype != bool:
                raise ValueError("include_mask must be a 1-D boolean array matching points")
            selected = inc
        if exclude_mask is not None:
            exc = np.asarray(exclude_mask)
            if exc.shape != (len(points),) or exc.dtype != bool:
                raise ValueError("exclude_mask must be a 1-D boolean array matching points")
            selected = (~exc) if selected is None else (selected & ~exc)

        pending_reset = False
        if self._coordinate_frame is not None and coordinate_frame != self._coordinate_frame:
            if self._config.frame_change_policy == "reject":
                committed, tentative = self._active_counts()
                return SurfaceUpdateResult(
                    accepted=False,
                    status="coordinate_frame_changed",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                    committed_cell_count=committed,
                    tentative_cell_count=tentative,
                )
            pending_reset = True

        if not pending_reset:
            retained = self._retained_observations.get(observation_id)
            if retained is not None:
                if T_fc is None or (
                    retained.frame_from_camera is not None
                    and np.allclose(retained.frame_from_camera, T_fc, atol=1e-7, rtol=0.0)
                ):
                    committed, tentative = self._active_counts()
                    return SurfaceUpdateResult(
                        accepted=False,
                        status="duplicate_ignored",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                        committed_cell_count=committed,
                        tentative_cell_count=tentative,
                    )
                return self._apply_pose_correction(retained, T_fc, timestamp_ns=timestamp_ns)

            seen = self._seen_observations.get(observation_id)
            if seen is not None:
                committed, tentative = self._active_counts()
                if (
                    T_fc is not None
                    and seen.frame_from_camera is not None
                    and not np.allclose(seen.frame_from_camera, T_fc, atol=1e-7, rtol=0.0)
                ):
                    return SurfaceUpdateResult(
                        accepted=False,
                        status="pose_correction_out_of_window",
                        version=self._version,
                        epoch=self._epoch,
                        observation_id=observation_id,
                        timestamp_ns=timestamp_ns,
                        committed_cell_count=committed,
                        tentative_cell_count=tentative,
                    )
                return SurfaceUpdateResult(
                    accepted=False,
                    status="duplicate_ignored",
                    version=self._version,
                    epoch=self._epoch,
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                    committed_cell_count=committed,
                    tentative_cell_count=tentative,
                )

        plane_result = self._resolve_plane(
            plane_normal=plane_normal,
            plane_offset_m=plane_offset_m,
            plane_support_count=plane_support_count,
            plane_rmse_m=plane_rmse_m,
            pending_reset=pending_reset,
        )
        if plane_result[0] == "invalid_plane":
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="invalid_plane",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )
        if plane_result[0] == "plane_changed":
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="plane_changed",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
            )
        _, resolved_normal, resolved_offset, plane_triggered_reset = plane_result
        pending_reset = pending_reset or plane_triggered_reset

        if self._config.retain_camera_points:
            if selected is None:
                pts_sel = points
                rgb_sel = colors
            else:
                pts_sel = points[selected]
                rgb_sel = colors[selected]
            if T_fc is not None and len(pts_sel) > 0:
                if pts_sel.dtype == np.float32:
                    pts_cam = (pts_sel - T_fc[:3, 3].astype(np.float32)[None, :]) @ T_fc[:3, :3].astype(np.float32)
                else:
                    pts_cam = (pts_sel - T_fc[:3, 3][None, :]) @ T_fc[:3, :3]
            else:
                pts_cam = None
            pass_mask = None
        else:
            pts_sel = points
            rgb_sel = colors
            pts_cam = None
            pass_mask = selected

        return self._commit_observation(
            pts_frame=pts_sel,
            colors_rgb_u8=rgb_sel,
            pts_cam=pts_cam,
            frame_from_camera=T_fc,
            coordinate_frame=coordinate_frame,
            plane_normal=resolved_normal,
            plane_offset_m=resolved_offset,
            timestamp_ns=timestamp_ns,
            observation_id=observation_id,
            camera_id=camera_id,
            observation_weight=float(observation_weight),
            pending_reset=pending_reset,
            camera_calibration=None,
            carving_inputs=None,
            selected_mask=pass_mask,
        )

    def _resolve_plane(
        self,
        *,
        plane_normal,
        plane_offset_m: float | None,
        plane_support_count: int | None,
        plane_rmse_m: float | None,
        pending_reset: bool,
    ) -> tuple[str, np.ndarray | None, float | None, bool]:
        if plane_normal is None and plane_offset_m is None:
            if pending_reset or self._plane_normal is None or self._plane_offset_m is None:
                return ("invalid_plane", None, None, False)
            return ("ok", self._plane_normal, self._plane_offset_m, False)

        if plane_normal is None or plane_offset_m is None or plane_support_count is None or plane_rmse_m is None:
            return ("invalid_plane", None, None, False)

        validated = _validate_plane_evidence(
            plane_normal, plane_offset_m, plane_support_count, plane_rmse_m, self._config.mesh
        )
        if validated is None:
            return ("invalid_plane", None, None, False)
        n_new, d_new = validated
        if not pending_reset and self._plane_normal is not None:
            if not self._plane_compatible(n_new, d_new):
                if self._config.plane_change_policy == "reject":
                    return ("plane_changed", None, None, False)
                return ("ok", n_new, d_new, True)
            return ("ok", self._plane_normal, self._plane_offset_m, False)
        return ("ok", n_new, d_new, False)

    def _commit_observation(
        self,
        *,
        pts_frame: np.ndarray,
        colors_rgb_u8: np.ndarray,
        pts_cam: np.ndarray | None,
        frame_from_camera: np.ndarray | None,
        coordinate_frame: str,
        plane_normal: np.ndarray | None,
        plane_offset_m: float | None,
        timestamp_ns: int,
        observation_id: str,
        camera_id: str,
        observation_weight: float,
        pending_reset: bool,
        camera_calibration: tuple[tuple[int, int], np.ndarray] | None,
        carving_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        selected_mask: np.ndarray | None = None,
    ) -> SurfaceUpdateResult:
        assert plane_normal is not None and plane_offset_m is not None

        if pending_reset:
            self._epoch += 1
            self._version += 1
            self._empty_reason = "no_observations"
            self._init_state(
                coordinate_frame=coordinate_frame,
                plane_normal=plane_normal,
                plane_offset_m=plane_offset_m,
            )
        else:
            if self._coordinate_frame is None:
                self._coordinate_frame = coordinate_frame
            if self._plane_normal is None:
                self._latch_plane(plane_normal, plane_offset_m)

        if camera_calibration is not None:
            self._camera_calibrations[camera_id] = camera_calibration

        cam_origin = None if frame_from_camera is None else frame_from_camera[:3, 3]
        cell_keys, cell_pts, cell_rgb, cell_w, inlier_count, budget_exceeded = self._project_and_aggregate_cells(
            pts_frame,
            colors_rgb_u8,
            camera_origin_frame_m=cam_origin,
            observation_weight=observation_weight,
            selected_mask=selected_mask,
        )
        if budget_exceeded:
            committed, tentative = self._active_counts()
            return SurfaceUpdateResult(
                accepted=False,
                status="grid_budget_exceeded",
                version=self._version,
                epoch=self._epoch,
                observation_id=observation_id,
                timestamp_ns=timestamp_ns,
                inlier_sample_count=inlier_count,
                committed_cell_count=committed,
                tentative_cell_count=tentative,
                reset_triggered=pending_reset,
            )

        self._update_seq += 1
        self._total_integrated_observations += 1
        if self._latest_observation_timestamp_ns is None or timestamp_ns > self._latest_observation_timestamp_ns:
            self._latest_observation_timestamp_ns = timestamp_ns
        self._record_seen(observation_id, frame_from_camera, timestamp_ns)

        carved = 0
        if carving_inputs is not None and not pending_reset:
            carved = self._apply_free_space_carving(*carving_inputs)

        updated = self._add_window_contribution(
            cell_keys,
            cell_pts,
            cell_rgb,
            cell_w,
            timestamp_ns=timestamp_ns,
        )
        pruned = self._prune_expired_tentative()

        stored_cam_pts = None if pts_cam is None else np.ascontiguousarray(pts_cam, dtype=np.float32)
        stored_cam_rgb = None if pts_cam is None else np.ascontiguousarray(colors_rgb_u8, dtype=np.uint8)
        obs_rec = _RetainedObservation(
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            camera_id=camera_id,
            frame_from_camera=None if frame_from_camera is None else frame_from_camera.copy(),
            camera_points_m=stored_cam_pts,
            camera_colors_rgb_u8=stored_cam_rgb,
            observation_weight=observation_weight,
            cell_keys=cell_keys,
            cell_pts=cell_pts,
            cell_rgb=cell_rgb,
            cell_w=cell_w,
            pose_revision=0,
        )
        self._retained_observations[observation_id] = obs_rec
        while len(self._retained_observations) > self._config.max_retained_observations:
            self._compact_oldest_observation()

        if len(cell_keys) > 0:
            self._record_provenance(
                ObservationProvenance(
                    observation_id=observation_id,
                    timestamp_ns=timestamp_ns,
                    camera_id=camera_id,
                    sample_count=len(cell_keys),
                    pose_revision=0,
                )
            )

        state_changed = pending_reset or updated > 0 or carved > 0 or pruned > 0
        if state_changed:
            self._version += 1
            self._map_updated_monotonic_ns = time.monotonic_ns()
            self._cached_mesh = None

        committed, tentative = self._active_counts()
        if pending_reset:
            status = "integrated_after_reset"
        elif len(cell_keys) == 0:
            status = "no_inlier_samples"
        else:
            status = "integrated"
        return SurfaceUpdateResult(
            accepted=True,
            status=status,
            version=self._version,
            epoch=self._epoch,
            observation_id=observation_id,
            timestamp_ns=timestamp_ns,
            inlier_sample_count=inlier_count,
            updated_cell_count=updated,
            committed_cell_count=committed,
            tentative_cell_count=tentative,
            carved_cell_count=carved,
            reset_triggered=pending_reset,
        )

    @_locked
    def estimated_memory_bytes(self) -> int:
        """Approximate internal memory footprint of cell buffers, tile caches, and retained observations."""
        cell_arrays = (
            self._active,
            self._keys,
            self._iu,
            self._iv,
            self._base_w,
            self._win_w,
            self._base_xyz,
            self._win_xyz,
            self._base_rgb,
            self._win_rgb,
            self._base_obs_count,
            self._win_obs_count,
            self._base_last_ts,
            self._last_ts,
            self._last_update_seq,
            self._free_space_conflicts,
            self._fused_xyz,
            self._fused_rgb,
            self._total_obs,
            self._ht_keys,
            self._ht_slots,
        )
        total = sum(int(arr.nbytes) for arr in cell_arrays)
        total += len(self._key_to_slot) * 64
        total += len(self._seen_observations) * 192
        total += len(self._provenance) * 128
        total += sum(obs.memory_bytes() for obs in self._retained_observations.values())
        total += sum(int(tc.tri_slots.nbytes + tc.face_normals.nbytes) for tc in self._tile_cache.values())
        return int(total)

    def _rebuild_dirty_tile(self, tu: int, tv: int) -> None:
        assert self._plane_normal is not None
        min_obs = self._config.min_observations_to_commit
        u0 = tu << _TILE_SHIFT
        v0 = tv << _TILE_SHIFT

        # Gather slots from tile (tu, tv) plus 1-cell border of (+1, 0), (0, +1), (+1, +1) via vectorized scatter.
        grid = np.full((_TILE_SIZE + 1, _TILE_SIZE + 1), -1, dtype=np.int32)
        populated = 0
        for dtu, dtv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            tset = self._tile_to_slots.get((tu + dtu, tv + dtv))
            if not tset:
                continue
            s_arr = np.fromiter(tset, dtype=np.int32, count=len(tset))
            ok = self._total_obs[s_arr] >= min_obs
            if not np.any(ok):
                continue
            s_ok = s_arr[ok]
            lu = self._iu[s_ok] - u0
            lv = self._iv[s_ok] - v0
            if dtu or dtv:
                in_b = (lu >= 0) & (lu <= _TILE_SIZE) & (lv >= 0) & (lv <= _TILE_SIZE)
                if not np.any(in_b):
                    continue
                lu = lu[in_b]
                lv = lv[in_b]
                s_ok = s_ok[in_b]
            grid[lu, lv] = s_ok
            populated += int(len(s_ok))

        if populated < 4:
            self._tile_cache.pop((tu, tv), None)
            return

        if NATIVE is not None and hasattr(NATIVE, "planar_tile_quads_fast"):
            tri_slots, fnorms, area_m2, max_edge = NATIVE.planar_tile_quads_fast(
                grid,
                self._fused_xyz,
                self._plane_normal,
                float(self._config.mesh.maximum_edge_m),
                float(self._config.mesh.maximum_normal_step_m),
            )
            if len(tri_slots) == 0:
                self._tile_cache.pop((tu, tv), None)
            else:
                self._tile_cache[(tu, tv)] = _TileMeshCache(
                    tri_slots=tri_slots,
                    face_normals=fnorms,
                    area_m2=float(area_m2),
                    max_edge_m=float(max_edge),
                )
            return

        faces = _quads(grid)
        if len(faces) == 0:
            self._tile_cache.pop((tu, tv), None)
            return
        tri = self._fused_xyz[faces]
        edges = tri - np.roll(tri, 1, axis=1)
        edge_lengths = np.linalg.norm(edges, axis=2)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        keep = (
            (edge_lengths.max(axis=1) <= self._config.mesh.maximum_edge_m)
            & (area2 > 1e-12)
            & (np.abs(edges @ self._plane_normal).max(axis=1) <= self._config.mesh.maximum_normal_step_m)
            & ((cross @ self._plane_normal) > 1e-12)
        )
        faces = faces[keep].astype(np.int32, copy=False)
        if len(faces) == 0:
            self._tile_cache.pop((tu, tv), None)
            return
        self._tile_cache[(tu, tv)] = _TileMeshCache(
            tri_slots=faces,
            face_normals=cross[keep] / area2[keep, None],
            area_m2=float(area2[keep].sum() * 0.5),
            max_edge_m=float(edge_lengths[keep].max()),
        )

    def _build_mesh_cache(self) -> _CachedMeshData:
        empty_v = _freeze_copy(np.empty((0, 3), dtype=np.float64))
        empty_c = _freeze_copy(np.empty((0, 3), dtype=np.uint8))
        empty_t = _freeze_copy(np.empty((0, 3), dtype=np.int32))
        empty_n = _freeze_copy(np.empty((0, 3), dtype=np.float64))
        empty_oc = _freeze_copy(np.empty(0, dtype=np.int32))
        empty_ts = _freeze_copy(np.empty(0, dtype=np.int64))

        def _empty(reason: str, support: int = 0) -> _CachedMeshData:
            return _CachedMeshData(
                version=self._version,
                ready=False,
                reason=reason,
                vertices_m=empty_v,
                colors_rgb_u8=empty_c,
                triangles=empty_t,
                face_normals=empty_n,
                support_count=support,
                surface_area_m2=0.0,
                maximum_edge_m=0.0,
                vertex_observation_counts=empty_oc,
                vertex_last_observed_ns=empty_ts,
                newest_contributing_timestamp_ns=None,
                median_contributing_timestamp_ns=None,
                oldest_contributing_timestamp_ns=None,
            )

        if self._total_integrated_observations == 0 or self._plane_normal is None:
            return _empty(self._empty_reason)

        n = self._count
        if n == 0:
            return _empty("insufficient_committed_support")

        if NATIVE is not None and hasattr(NATIVE, "planar_mesh_snapshot_ht_fast"):
            self._sync_ht()
            (
                status,
                num_committed,
                verts,
                cols,
                tris,
                fnorms,
                v_obs,
                v_ts,
                total_area,
                max_edge,
            ) = NATIVE.planar_mesh_snapshot_ht_fast(
                self._active,
                self._base_w,
                self._win_w,
                self._total_obs,
                self._iu,
                self._iv,
                self._fused_xyz,
                self._fused_rgb,
                self._last_ts,
                self._ht_keys,
                self._ht_slots,
                int(n),
                np.ascontiguousarray(self._plane_normal, dtype=np.float64),
                int(self._config.min_observations_to_commit),
                int(self._config.mesh.minimum_support),
                float(self._config.mesh.maximum_edge_m),
                float(self._config.mesh.maximum_normal_step_m),
                int(self._config.mesh.maximum_grid_cells),
                int(self._config.mesh.maximum_vertices),
            )
            if status == 1:
                return _empty("insufficient_committed_support", support=int(num_committed))
            if status == 2:
                return _empty("surface_grid_budget_exceeded", support=int(num_committed))
            if status == 3:
                return _empty("surface_vertex_budget_exceeded", support=int(num_committed))
            if status == 4:
                return _empty("no_valid_local_triangles", support=int(num_committed))
            for arr in (verts, cols, tris, fnorms, v_obs, v_ts):
                arr.setflags(write=False)
            return _CachedMeshData(
                version=self._version,
                ready=True,
                reason="persistent_measured_planar_surface",
                vertices_m=verts,
                colors_rgb_u8=cols,
                triangles=tris,
                face_normals=fnorms,
                support_count=int(num_committed),
                surface_area_m2=float(total_area),
                maximum_edge_m=float(max_edge),
                vertex_observation_counts=v_obs,
                vertex_last_observed_ns=v_ts,
                newest_contributing_timestamp_ns=int(v_ts.max()),
                median_contributing_timestamp_ns=int(np.median(v_ts)),
                oldest_contributing_timestamp_ns=int(v_ts.min()),
            )

        total_w = self._base_w[:n] + self._win_w[:n]
        committed_mask = (
            self._active[:n]
            & (self._total_obs[:n] >= self._config.min_observations_to_commit)
            & (total_w > 0.0)
        )
        num_committed = int(np.count_nonzero(committed_mask))
        if num_committed < self._config.mesh.minimum_support:
            return _empty("insufficient_committed_support", support=num_committed)
        if num_committed > self._config.mesh.maximum_vertices:
            return _empty("surface_vertex_budget_exceeded", support=num_committed)

        committed_slots = np.flatnonzero(committed_mask)
        iu_c = self._iu[committed_slots].astype(np.int64)
        iv_c = self._iv[committed_slots].astype(np.int64)
        extent_u = int(iu_c.max()) - int(iu_c.min()) + 1
        extent_v = int(iv_c.max()) - int(iv_c.min()) + 1
        if extent_u * extent_v > self._config.mesh.maximum_grid_cells:
            return _empty("surface_grid_budget_exceeded", support=num_committed)

        # Incrementally rebuild only dirty 32x32 spatial tiles.
        if not self._tile_cache and self._tile_to_slots:
            self._dirty_tiles.update(self._tile_to_slots.keys())
        if self._dirty_tiles:
            dirty_list = sorted(self._dirty_tiles)
            self._dirty_tiles.clear()
            for tu, tv in dirty_list:
                self._rebuild_dirty_tile(tu, tv)

        if not self._tile_cache:
            return _empty("no_valid_local_triangles", support=num_committed)

        ordered_tiles = [self._tile_cache[k] for k in sorted(self._tile_cache.keys())]
        if len(ordered_tiles) == 1:
            all_tri_slots = ordered_tiles[0].tri_slots
            all_fnorms = ordered_tiles[0].face_normals
            total_area = ordered_tiles[0].area_m2
            max_edge = ordered_tiles[0].max_edge_m
        else:
            all_tri_slots = np.concatenate([t.tri_slots for t in ordered_tiles], axis=0)
            all_fnorms = np.concatenate([t.face_normals for t in ordered_tiles], axis=0)
            total_area = float(sum(t.area_m2 for t in ordered_tiles))
            max_edge = float(max(t.max_edge_m for t in ordered_tiles))

        if NATIVE is not None and hasattr(NATIVE, "planar_compact_triangles_fast"):
            status, verts, cols, tris, v_obs, v_ts = NATIVE.planar_compact_triangles_fast(
                np.ascontiguousarray(all_tri_slots, dtype=np.int32),
                self._fused_xyz[:n],
                self._fused_rgb[:n],
                self._total_obs[:n],
                self._last_ts[:n],
                int(self._config.mesh.maximum_vertices),
            )
            if status == 3:
                return _empty("surface_vertex_budget_exceeded", support=num_committed)
            fnorms = np.array(all_fnorms, copy=True)
            for arr in (verts, cols, tris, fnorms, v_obs, v_ts):
                arr.setflags(write=False)
        else:
            used_slots, inverse = np.unique(all_tri_slots, return_inverse=True)
            if len(used_slots) > self._config.mesh.maximum_vertices:
                return _empty("surface_vertex_budget_exceeded", support=num_committed)
            verts = _freeze_copy(self._fused_xyz[used_slots])
            cols = _freeze_copy(self._fused_rgb[used_slots])
            tris = _freeze_copy(inverse.reshape(-1, 3).astype(np.int32))
            fnorms = _freeze_copy(all_fnorms)
            v_obs = _freeze_copy(self._total_obs[used_slots].astype(np.int32))
            v_ts = _freeze_copy(self._last_ts[used_slots].astype(np.int64))

        return _CachedMeshData(
            version=self._version,
            ready=True,
            reason="persistent_measured_planar_surface",
            vertices_m=verts,
            colors_rgb_u8=cols,
            triangles=tris,
            face_normals=fnorms,
            support_count=num_committed,
            surface_area_m2=total_area,
            maximum_edge_m=max_edge,
            vertex_observation_counts=v_obs,
            vertex_last_observed_ns=v_ts,
            newest_contributing_timestamp_ns=int(v_ts.max()),
            median_contributing_timestamp_ns=int(np.median(v_ts)),
            oldest_contributing_timestamp_ns=int(v_ts.min()),
        )

    @_locked
    def snapshot(self, *, reference_timestamp_ns: int | None = None) -> PersistentSurfaceSnapshot:
        """Materialize a bounded, immutable, versioned mesh snapshot."""
        if reference_timestamp_ns is not None:
            ref_ts: int | None = _validate_timestamp_ns(reference_timestamp_ns, "reference_timestamp_ns")
        else:
            ref_ts = self._latest_observation_timestamp_ns

        if self._cached_mesh is None or self._cached_mesh.version != self._version:
            self._cached_mesh = self._build_mesh_cache()
        c = self._cached_mesh

        _, tentative_count = self._active_counts()
        provs = tuple(p for p in self._provenance.values() if p.sample_count > 0)
        prov_ids = tuple(p.observation_id for p in provs)

        newest_lag = (
            None
            if (ref_ts is None or c.newest_contributing_timestamp_ns is None)
            else ref_ts - c.newest_contributing_timestamp_ns
        )
        oldest_lag = (
            None
            if (ref_ts is None or c.oldest_contributing_timestamp_ns is None)
            else ref_ts - c.oldest_contributing_timestamp_ns
        )
        plane_n = None if self._plane_normal is None else _freeze_copy(self._plane_normal)

        return PersistentSurfaceSnapshot(
            ready=c.ready,
            reason=c.reason,
            version=self._version,
            epoch=self._epoch,
            coordinate_frame=self._coordinate_frame or "",
            vertices_m=c.vertices_m,
            colors_rgb_u8=c.colors_rgb_u8,
            triangles=c.triangles,
            face_normals=c.face_normals,
            plane_normal=plane_n,
            plane_offset_m=self._plane_offset_m,
            support_count=c.support_count,
            surface_area_m2=c.surface_area_m2,
            maximum_edge_m=c.maximum_edge_m,
            vertex_observation_counts=c.vertex_observation_counts,
            vertex_last_observed_ns=c.vertex_last_observed_ns,
            contributing_observation_ids=prov_ids,
            contributing_observations=provs,
            total_integrated_observations=self._total_integrated_observations,
            compacted_observation_count=self._compacted_observation_count,
            snapshot_created_monotonic_ns=time.monotonic_ns(),
            map_updated_monotonic_ns=self._map_updated_monotonic_ns,
            reference_timestamp_ns=ref_ts,
            latest_observation_timestamp_ns=self._latest_observation_timestamp_ns,
            newest_contributing_timestamp_ns=c.newest_contributing_timestamp_ns,
            median_contributing_timestamp_ns=c.median_contributing_timestamp_ns,
            oldest_contributing_timestamp_ns=c.oldest_contributing_timestamp_ns,
            newest_observation_lag_ns=newest_lag,
            oldest_vertex_lag_ns=oldest_lag,
            tentative_cell_count=tentative_count,
            estimated_memory_bytes=self.estimated_memory_bytes(),
        )
