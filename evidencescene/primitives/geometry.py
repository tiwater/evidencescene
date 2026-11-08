"""Self-contained geometric, camera, and provenance primitives for EvidenceScene.

When running inside a ``ticos-edge`` environment, shared contract types bind to
their host class definitions so ``isinstance`` checks across packages remain
identical; in a standalone ``evidencescene`` installation, the self-contained
definitions below are used directly with zero external runtime dependencies.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from threading import get_ident
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np

from evidencescene.primitives.content_hash import content_sha256


DEFAULT_PIXEL_STRIDE = 3
DEFAULT_VOXEL_SIZE_M = 0.004
DEFAULT_MINIMUM_FRAME_SUPPORT = 2
DEFAULT_MAXIMUM_SURFACE_POINT_COUNT = 20_000


def _validate_rigid_transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be one finite 4x4 transform")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must have a homogeneous bottom row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be proper")
    return value


def _backproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    pixel_stride: int,
    minimum_depth_m: float = 0.15,
    maximum_depth_m: float = 3.0,
    depth_scale_m: float = 1.0,
) -> np.ndarray:
    raw_depth = np.asarray(depth_m)
    selected = np.asarray(mask, dtype=bool)
    intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    if raw_depth.ndim != 2 or selected.shape != raw_depth.shape:
        raise ValueError("depth and mask must have the same 2D shape")
    if intrinsics.shape != (3, 3):
        raise ValueError("camera matrix must be 3x3")
    if pixel_stride <= 0:
        raise ValueError("pixel stride must be positive")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth scale must be finite and positive")
    scaled_integer_depth = (
        np.issubdtype(raw_depth.dtype, np.integer) and depth_scale_m != 1.0
    )
    if pixel_stride > 1:
        sampled_depth = raw_depth[::pixel_stride, ::pixel_stride]
        sampled_mask = selected[::pixel_stride, ::pixel_stride]
        if scaled_integer_depth:
            depth_divisor = 1.0 / float(depth_scale_m)
            valid = (
                sampled_mask
                & (sampled_depth >= float(minimum_depth_m) * depth_divisor)
                & (sampled_depth <= float(maximum_depth_m) * depth_divisor)
            )
        else:
            depth = np.asarray(sampled_depth, dtype=np.float64)
            valid = (
                sampled_mask
                & np.isfinite(depth)
                & (depth >= float(minimum_depth_m))
                & (depth <= float(maximum_depth_m))
            )
        rows, columns = np.nonzero(valid)
        rows = rows * pixel_stride
        columns = columns * pixel_stride
    else:
        if scaled_integer_depth:
            depth_divisor = 1.0 / float(depth_scale_m)
            valid = (
                selected
                & (raw_depth >= float(minimum_depth_m) * depth_divisor)
                & (raw_depth <= float(maximum_depth_m) * depth_divisor)
            )
        else:
            depth = np.asarray(raw_depth, dtype=np.float64)
            valid = (
                selected
                & np.isfinite(depth)
                & (depth >= float(minimum_depth_m))
                & (depth <= float(maximum_depth_m))
            )
        rows, columns = np.nonzero(valid)
    if not len(columns):
        return np.empty((0, 3), dtype=np.float64)
    if scaled_integer_depth:
        z = raw_depth[rows, columns].astype(np.float64) / depth_divisor
    else:
        z = np.asarray(raw_depth[rows, columns], dtype=np.float64)
    x = (columns.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points must be Nx3 and transform must be 4x4")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_matrix(transform: Any) -> np.ndarray:
    q = transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm == 0:
        return np.eye(3)
    x = q.x / norm
    y = q.y / norm
    z = q.z / norm
    w = q.w / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def mask_evidence_sha256(mask_rle: object) -> str | None:
    if not isinstance(mask_rle, Mapping):
        return None
    try:
        payload = json.dumps(
            mask_rle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def registered_feature_stream_key(
    observation: Any,
    *,
    object_id: str,
) -> list[Any] | None:
    stream = getattr(observation, "perception_stream", None)
    color_info = getattr(observation, "color_camera_info", None)
    depth_info = getattr(observation, "depth_camera_info", None) or getattr(
        observation, "camera_info", None
    )
    registration = getattr(observation, "rgbd_registration", None)
    if (
        stream is None
        or not isinstance(object_id, str)
        or not object_id.strip()
        or color_info is None
        or depth_info is None
        or registration is None
        or getattr(registration, "status", None) != "validated"
        or not isinstance(getattr(registration, "calibration_id", None), str)
        or not registration.calibration_id
    ):
        return None
    return [
        stream.stream_id,
        int(stream.epoch),
        str(observation.camera_id or observation.camera_topic),
        observation.camera_topic,
        color_info.frame_id,
        depth_info.frame_id,
        object_id,
        registration.calibration_id,
    ]


_digest_scope = ContextVar("observation_digest_scope", default=None)


@contextmanager
def observation_digest_scope(*, enabled: bool = True, capacity: int = 2):
    if type(capacity) is not int or not 1 <= capacity <= 8:
        raise ValueError("digest cache capacity must be an integer from 1 to 8")
    entries: OrderedDict = OrderedDict()
    active = [True]
    token = _digest_scope.set((get_ident(), entries, active, capacity) if enabled else None)
    try:
        yield
    finally:
        active[0] = False
        entries.clear()
        _digest_scope.reset(token)


def observation_digest(observation: Any) -> str:
    scope = _digest_scope.get()
    if scope is not None and scope[0] == get_ident() and scope[2][0]:
        if hasattr(observation, "model_dump_json") and hasattr(observation, "color_image_base64"):
            entries = scope[1]
            metadata = observation.model_dump_json(
                exclude={"color_image_base64", "depth_image_base64"}
            )
            color, depth = observation.color_image_base64, observation.depth_image_base64
            old = entries.get(id(observation))
            if old is not None and old[0] is observation and old[1:4] == (metadata, color, depth):
                return old[4]
            digest = hashlib.sha256(observation.model_dump_json().encode()).hexdigest()
            entries[id(observation)] = (observation, metadata, color, depth, digest)
            entries.move_to_end(id(observation))
            while len(entries) > scope[3]:
                entries.popitem(last=False)
            return digest
    return hashlib.sha256(observation.model_dump_json().encode()).hexdigest()


@dataclass(frozen=True)
class _StandaloneCameraInfoRecord:
    frame_id: str
    width: int
    height: int
    k: tuple[float, ...]
    distortion_model: str
    d: tuple[float, ...]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _rejected_surface(
    reason: str,
    *,
    color_timestamp_ns: int | None = None,
    depth_timestamp_ns: int | None = None,
    sync_skew_ms: float | None = None,
) -> "_StandaloneRegisteredRgbdSurface":
    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_pixels = np.empty((0, 2), dtype=np.int64)
    empty_depth = np.empty(0, dtype=np.float64)
    return _StandaloneRegisteredRgbdSurface(
        ready=False,
        reason=reason,
        source="validated_color_depth_registration",
        points_depth_m=empty_points,
        depth_pixels_uv=empty_pixels,
        color_pixels_uv=empty_pixels,
        color_depth_m=empty_depth,
        color_timestamp_ns=color_timestamp_ns,
        depth_timestamp_ns=depth_timestamp_ns,
        sync_skew_ms=sync_skew_ms,
        depth_frame_id=None,
        color_frame_id=None,
        color_image_size=(0, 0),
        registration_max_reprojection_error_px=None,
    )


@dataclass(frozen=True)
class _StandaloneRegisteredRgbdSurface:
    ready: bool
    reason: str
    source: str
    points_depth_m: np.ndarray
    depth_pixels_uv: np.ndarray
    color_pixels_uv: np.ndarray
    color_depth_m: np.ndarray
    color_timestamp_ns: int | None
    depth_timestamp_ns: int | None
    sync_skew_ms: float | None
    depth_frame_id: str | None
    color_frame_id: str | None
    color_image_size: tuple[int, int]
    registration_max_reprojection_error_px: float | None

    def metadata(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "source": self.source,
            "point_count": int(len(self.points_depth_m)),
            "depth_frame_id": self.depth_frame_id,
            "color_frame_id": self.color_frame_id,
            "color_timestamp_ns": self.color_timestamp_ns,
            "depth_timestamp_ns": self.depth_timestamp_ns,
            "sync_skew_ms": self.sync_skew_ms,
            "raw_depth_backed": self.ready and len(self.points_depth_m) > 0,
            "registration_max_reprojection_error_px": self.registration_max_reprojection_error_px,
        }

    def color_depth_image(self) -> np.ndarray:
        width, height = self.color_image_size
        image = np.zeros((height, width), dtype=np.float32)
        if self.ready and len(self.color_pixels_uv):
            image[self.color_pixels_uv[:, 1], self.color_pixels_uv[:, 0]] = np.asarray(
                self.color_depth_m, dtype=np.float32
            )
        return image


@dataclass(frozen=True)
class _StandaloneRegisteredRgbdFrame:
    ready: bool
    reason: str
    points_depth_m: np.ndarray
    depth_pixels_uv: np.ndarray
    color_pixels_uv: np.ndarray
    color_depth_m: np.ndarray
    color_timestamp_ns: int | None
    depth_timestamp_ns: int | None
    sync_skew_ms: float | None
    depth_frame_id: str | None
    color_frame_id: str | None
    color_image_size: tuple[int, int]
    registration_max_reprojection_error_px: float | None
    projection_diagnostics: Any = None

    def color_depth_image(self) -> np.ndarray:
        width, height = self.color_image_size
        image = np.zeros((height, width), dtype=np.float32)
        if self.ready and len(self.color_pixels_uv):
            image[self.color_pixels_uv[:, 1], self.color_pixels_uv[:, 0]] = np.asarray(
                self.color_depth_m, dtype=np.float32
            )
        return image

    def select_surface(
        self,
        color_mask: np.ndarray,
        *,
        min_points: int = 50,
    ) -> _StandaloneRegisteredRgbdSurface:
        mask = np.asarray(color_mask, dtype=bool)
        width, height = self.color_image_size
        if not self.ready:
            return _rejected_surface(
                self.reason,
                color_timestamp_ns=self.color_timestamp_ns,
                depth_timestamp_ns=self.depth_timestamp_ns,
                sync_skew_ms=self.sync_skew_ms,
            )
        if mask.shape != (height, width):
            return _rejected_surface(
                "color_mask_geometry_mismatch",
                color_timestamp_ns=self.color_timestamp_ns,
                depth_timestamp_ns=self.depth_timestamp_ns,
                sync_skew_ms=self.sync_skew_ms,
            )
        selected = mask[self.color_pixels_uv[:, 1], self.color_pixels_uv[:, 0]]
        if int(np.count_nonzero(selected)) < int(min_points):
            return _rejected_surface(
                "insufficient_registered_mask_depth_points",
                color_timestamp_ns=self.color_timestamp_ns,
                depth_timestamp_ns=self.depth_timestamp_ns,
                sync_skew_ms=self.sync_skew_ms,
            )
        return _StandaloneRegisteredRgbdSurface(
            ready=True,
            reason="ready",
            source="validated_color_depth_registration",
            points_depth_m=self.points_depth_m[selected],
            depth_pixels_uv=self.depth_pixels_uv[selected],
            color_pixels_uv=self.color_pixels_uv[selected],
            color_depth_m=self.color_depth_m[selected],
            color_timestamp_ns=self.color_timestamp_ns,
            depth_timestamp_ns=self.depth_timestamp_ns,
            sync_skew_ms=self.sync_skew_ms,
            depth_frame_id=self.depth_frame_id,
            color_frame_id=self.color_frame_id,
            color_image_size=self.color_image_size,
            registration_max_reprojection_error_px=self.registration_max_reprojection_error_px,
        )


@dataclass(frozen=True)
class _StandaloneMaskDepthComponentSelection:
    ready: bool
    reason: str
    point_depth_m: np.ndarray | None
    point_depth_pixel_uv: np.ndarray | None
    support_point_count: int
    local_window_point_count: int
    total_point_count: int
    minimum_depth_m: float | None
    maximum_depth_m: float | None
    local_depth_tolerance_m: float | None
    mask_centroid_color_pixel_uv: np.ndarray | None = None
    anchor_distance_px: float | None = None
    anchor_distance_fraction: float | None = None
    maximum_anchor_distance_fraction: float | None = None
    local_depth_median_m: float | None = None
    local_window_radius_px: float | None = None
    depth_mad_m: float | None = None
    local_scatter_p95_m: float | None = None


def intersect_rectified_tag_rays_with_depth_plane(
    *,
    rectified_corners_px: object,
    color_k: object,
    color_from_depth: object,
    plane_normal_depth: object,
    plane_offset_m: float,
) -> np.ndarray:
    """Intersect ordered rectified color rays with a metric depth-frame plane."""
    pixels = np.asarray(rectified_corners_px, dtype=np.float64)
    if pixels.shape != (4, 2) or not np.isfinite(pixels).all():
        raise ValueError("rectified tag corners must be a finite (4, 2) array")
    intrinsic = np.asarray(color_k, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0)
        or abs(float(np.linalg.det(intrinsic))) <= 1e-12
    ):
        raise ValueError("color intrinsic matrix is invalid")
    transform = np.asarray(color_from_depth, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("color_from_depth must be a finite (4, 4) array")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("color_from_depth must be a rigid transform")
    normal = np.asarray(plane_normal_depth, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("depth plane normal must be a finite (3,) array")
    normal_norm = float(np.linalg.norm(normal))
    offset = float(plane_offset_m)
    if normal_norm <= 1e-12 or not math.isfinite(offset):
        raise ValueError("depth plane must be finite and nondegenerate")
    normal = normal / normal_norm
    offset /= normal_norm

    homogeneous = np.column_stack((pixels, np.ones(4, dtype=np.float64)))
    rays_color = (np.linalg.inv(intrinsic) @ homogeneous.T).T
    translation = transform[:3, 3]
    origin_depth = -rotation.T @ translation
    rays_depth = (rotation.T @ rays_color.T).T
    denominators = rays_depth @ normal
    if np.any(np.abs(denominators) <= 1e-10):
        raise ValueError("color ray is parallel to the depth plane")
    distances = -(float(normal @ origin_depth) + offset) / denominators
    if np.any(distances <= 0.0) or not np.isfinite(distances).all():
        raise ValueError("depth-plane intersection lies behind the color camera")
    corners = origin_depth[None, :] + distances[:, None] * rays_depth
    plane_error = np.abs(corners @ normal + offset)
    if not np.isfinite(corners).all() or float(np.max(plane_error)) > 1e-9:
        raise ValueError("depth-plane intersection is numerically invalid")
    return corners


def _frozen_f8(value: object, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype("<f8"))
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected array shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("arrays must be finite")
    result = np.ndarray(array.shape, dtype=array.dtype, buffer=array.tobytes(order="C"))
    result.flags.writeable = False
    return result


def _unit_vec3(value: object) -> np.ndarray:
    vector = _frozen_f8(value, shape=(3,))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("direction must be nonzero")
    normalized = np.asarray(vector).copy() if abs(norm - 1.0) <= 1e-12 else np.asarray(vector) / norm
    index = int(np.argmax(np.abs(normalized)))
    if normalized[index] < 0.0:
        normalized = -normalized
    normalized[normalized == 0.0] = 0.0
    return _frozen_f8(normalized, shape=(3,))


def _canonical_plane(normal_value: object, offset_value: object) -> tuple[np.ndarray, float]:
    normal = _frozen_f8(normal_value, shape=(3,))
    offset = float(offset_value)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12 or not math.isfinite(offset):
        raise ValueError("plane parameters must be finite and nondegenerate")
    if abs(norm - 1.0) <= 1e-12:
        normal = np.asarray(normal).copy()
    else:
        normal = np.asarray(normal) / norm
        offset /= norm
    index = int(np.argmax(np.abs(normal)))
    if normal[index] < 0.0:
        normal = -normal
        offset = -offset
    normal[normal == 0.0] = 0.0
    return _frozen_f8(normal, shape=(3,)), offset


@dataclass(frozen=True)
class _StandaloneDepthBoundaryCandidate:
    camera_id: str
    frame_id: str
    timestamp_ns: int
    source_sha256: str
    plane_normal_camera: np.ndarray
    plane_offset_m: float
    plane_points_camera_m: np.ndarray
    supporting_lines_camera: tuple[tuple[np.ndarray, np.ndarray], ...]
    line_evidence: tuple[Any, ...] = ()
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.camera_id or not self.frame_id or type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("camera/frame identity and positive timestamp are required")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source SHA-256 must be lowercase hexadecimal")
        normal, offset = _canonical_plane(self.plane_normal_camera, self.plane_offset_m)
        points = np.asarray(self.plane_points_camera_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("expected a nonempty Nx3 finite array")
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
        points = _frozen_f8(points[order])
        object.__setattr__(self, "plane_normal_camera", normal)
        object.__setattr__(self, "plane_offset_m", offset)
        object.__setattr__(self, "plane_points_camera_m", points)


def _sha256_str(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _frozen_array(
    value: object,
    *,
    dtype: np.dtype,
    name: str,
    shape: tuple[int, ...] | None = None,
    last_dimension: int | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise ValueError(f"{name} must be a {dtype.str} ndarray")
    if value.ndim == 0 or (not allow_empty and value.size == 0):
        raise ValueError(f"{name} must not be empty")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if last_dimension is not None and (value.ndim < 1 or value.shape[-1] != last_dimension):
        raise ValueError(f"{name} has an invalid shape")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    result = np.ndarray(value.shape, dtype=dtype, buffer=np.ascontiguousarray(value).tobytes())
    result.flags.writeable = False
    return result


def _array_evidence(value: np.ndarray) -> dict[str, object]:
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _array_sha256(value)}


@dataclass(frozen=True)
class _StandaloneBundleSDFFrameInput:
    observation_hash: str
    source_sha256: str
    g1_registered_rgbd_authorization: str
    g1_registered_rgbd_bundle_hash: str
    raw_color_rgb: np.ndarray
    registered_color_rgb: np.ndarray
    aligned_depth_m: np.ndarray
    mask: np.ndarray
    intrinsics_camera: np.ndarray
    raw_color_rgb_sha256: str
    registered_color_rgb_sha256: str
    aligned_depth_m_sha256: str
    mask_sha256: str
    intrinsics_camera_sha256: str
    timestamp_ns: int
    camera_frame_id: str
    points_camera_m: np.ndarray
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = {
            "observation hash": self.observation_hash,
            "source hash": self.source_sha256,
            "G1 registered_rgbd bundle hash": self.g1_registered_rgbd_bundle_hash,
            "raw_color_rgb hash": self.raw_color_rgb_sha256,
            "registered_color_rgb hash": self.registered_color_rgb_sha256,
            "aligned_depth_m hash": self.aligned_depth_m_sha256,
            "mask hash": self.mask_sha256,
            "intrinsics_camera hash": self.intrinsics_camera_sha256,
        }
        for name, value in hashes.items():
            _sha256_str(value, name=name)
        if self.g1_registered_rgbd_authorization != "registered_rgbd":
            raise ValueError("G1 registered_rgbd authorization is required")
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        if not isinstance(self.camera_frame_id, str) or not self.camera_frame_id:
            raise ValueError("camera_frame_id is required")
        raw = _frozen_array(self.raw_color_rgb, dtype=np.dtype("u1"), name="raw_color_rgb", last_dimension=3)
        registered = _frozen_array(
            self.registered_color_rgb, dtype=np.dtype("u1"), name="registered_color_rgb", last_dimension=3
        )
        depth = _frozen_array(self.aligned_depth_m, dtype=np.dtype("<f4"), name="aligned_depth_m")
        mask = _frozen_array(self.mask, dtype=np.dtype("?"), name="mask")
        intrinsics = _frozen_array(
            self.intrinsics_camera, dtype=np.dtype("<f8"), name="intrinsics_camera", shape=(3, 3)
        )
        if raw.shape != registered.shape or raw.shape[:2] != depth.shape or depth.shape != mask.shape:
            raise ValueError("registered RGB-D arrays must share an HxW grid without resize")
        if np.any(depth < 0.0):
            raise ValueError("aligned_depth_m must be nonnegative metric depth")
        if not np.any((depth >= 0.1) & mask):
            raise ValueError("mask must contain valid metric depth")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0 or not np.allclose(
            intrinsics[2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0.0
        ):
            raise ValueError("intrinsics_camera must be calibrated")
        for name, val, expected in (
            ("raw_color_rgb hash", raw, self.raw_color_rgb_sha256),
            ("registered_color_rgb hash", registered, self.registered_color_rgb_sha256),
            ("aligned_depth_m hash", depth, self.aligned_depth_m_sha256),
            ("mask hash", mask, self.mask_sha256),
            ("intrinsics_camera hash", intrinsics, self.intrinsics_camera_sha256),
        ):
            if _array_sha256(val) != expected:
                raise ValueError(f"{name} does not match actual bytes")
        points = _frozen_array(self.points_camera_m, dtype=np.dtype("<f8"), name="points_camera_m", last_dimension=3)
        object.__setattr__(self, "raw_color_rgb", raw)
        object.__setattr__(self, "registered_color_rgb", registered)
        object.__setattr__(self, "aligned_depth_m", depth)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "intrinsics_camera", intrinsics)
        object.__setattr__(self, "points_camera_m", points)
        object.__setattr__(
            self,
            "content_hash",
            content_sha256(
                {
                    **hashes,
                    "g1_registered_rgbd_authorization": self.g1_registered_rgbd_authorization,
                    "timestamp_ns": self.timestamp_ns,
                    "camera_frame_id": self.camera_frame_id,
                    "raw_color_rgb": _array_evidence(raw),
                    "registered_color_rgb": _array_evidence(registered),
                    "aligned_depth_m": _array_evidence(depth),
                    "mask": _array_evidence(mask),
                    "intrinsics_camera": _array_evidence(intrinsics),
                    "points_camera_m": _array_evidence(points),
                }
            ),
        )


@dataclass(frozen=True)
class _StandaloneBundleSDFProviderResult:
    status: Literal["candidate", "not_observable"]
    reason: str
    observation_hash: str
    source_sha256: str
    camera_frame_id: str
    timestamp_ns: int
    camera_from_object: np.ndarray | None
    surface_points_camera_m: np.ndarray
    confidence: float
    diagnostics: Mapping[str, Any]
    content_hash: str = field(default="", init=False)


class _StandaloneBundleSDFProvider:
    pass


@dataclass(frozen=True)
class _StandaloneCausalRuntimeFrame:
    camera_id: str
    frame_index: int
    decoded_rgb: np.ndarray


@dataclass(frozen=True)
class _StandaloneCausalRuntimeBatchResult:
    frames: tuple[Any, ...]
    refresh_completions: tuple[Any, ...]
    batch_latency_ms: float


class _StandaloneCausalMulticameraMaskRuntime:
    pass


CameraInfoRecord = _StandaloneCameraInfoRecord  # type: ignore[misc,assignment]
RegisteredRgbdSurface = _StandaloneRegisteredRgbdSurface  # type: ignore[misc,assignment]
RegisteredRgbdFrame = _StandaloneRegisteredRgbdFrame  # type: ignore[misc,assignment]
MaskDepthComponentSelection = _StandaloneMaskDepthComponentSelection  # type: ignore[misc,assignment]
DepthBoundaryCandidate = _StandaloneDepthBoundaryCandidate  # type: ignore[misc,assignment]
BundleSDFFrameInput = _StandaloneBundleSDFFrameInput  # type: ignore[misc,assignment]
BundleSDFProviderResult = _StandaloneBundleSDFProviderResult  # type: ignore[misc,assignment]
BundleSDFProvider = _StandaloneBundleSDFProvider  # type: ignore[misc,assignment]
CausalRuntimeFrame = _StandaloneCausalRuntimeFrame  # type: ignore[misc,assignment]
CausalRuntimeBatchResult = _StandaloneCausalRuntimeBatchResult  # type: ignore[misc,assignment]
CausalMulticameraMaskRuntime = _StandaloneCausalMulticameraMaskRuntime  # type: ignore[misc,assignment]

