"""Lift causal image tracks into auditable metric object observations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from evidencescene.primitives.point_tracks import CausalPointTracks


MetricObservation = tuple[int, int, np.ndarray]


def _transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return point @ transform[:3, :3].T + transform[:3, 3]


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    return result


def _intrinsics(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(result))) < 1e-12:
        raise ValueError("camera_matrix must be invertible")
    return result


def _depth_at(
    depth_m: np.ndarray,
    position_xy: np.ndarray,
    *,
    search_radius_px: int,
) -> float | None:
    if (
        not isinstance(depth_m, np.ndarray)
        or depth_m.ndim != 2
        or not np.issubdtype(depth_m.dtype, np.floating)
    ):
        raise ValueError("depth_m must be a floating-point HxW array")
    if search_radius_px < 0 or search_radius_px > 4:
        raise ValueError("depth search radius must be in [0,4]")
    column, row = (float(value) for value in position_xy)
    center_column, center_row = int(round(column)), int(round(row))
    height, width = depth_m.shape
    if not (0 <= center_row < height and 0 <= center_column < width):
        return None
    depth = float(depth_m[center_row, center_column])
    if np.isfinite(depth) and depth > 0.0:
        return depth
    y0, y1 = max(0, center_row - search_radius_px), min(
        height, center_row + search_radius_px + 1
    )
    x0, x1 = max(0, center_column - search_radius_px), min(
        width, center_column + search_radius_px + 1
    )
    candidates = np.argwhere(np.isfinite(depth_m[y0:y1, x0:x1]) & (depth_m[y0:y1, x0:x1] > 0.0))
    if not len(candidates):
        return None
    pixels = candidates + np.asarray((y0, x0), dtype=np.int64)
    distance_squared = (pixels[:, 0] - row) ** 2 + (pixels[:, 1] - column) ** 2
    depths = depth_m[pixels[:, 0], pixels[:, 1]]
    selected = int(np.lexsort((depths, distance_squared))[0])
    return float(depths[selected])


def _table_point(
    position_xy: np.ndarray,
    depth_m: float,
    inverse_camera_matrix: np.ndarray,
    table_from_camera: np.ndarray,
) -> np.ndarray:
    camera = inverse_camera_matrix @ np.asarray(
        (position_xy[0], position_xy[1], 1.0), dtype=np.float64
    ) * depth_m
    return _transform_point(camera, table_from_camera)


@dataclass(frozen=True)
class MetricPointMapReport:
    visible_track_count: int
    depth_supported_count: int
    accepted_count: int
    rejected_rigidity_count: int
    rigidity_residual_p50_m: float | None
    rigidity_residual_p95_m: float | None


def bootstrap_registered_landmarks(
    tracks: CausalPointTracks,
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    table_from_camera: np.ndarray,
    table_from_object: np.ndarray,
    landmark_id_offset: int,
    depth_search_radius_px: int = 1,
) -> tuple[Mapping[int, np.ndarray], MetricPointMapReport]:
    """Create fixed object-frame landmarks from one registered RGB-D frame."""

    if tracks.status != "candidate":
        raise ValueError("landmark bootstrap requires candidate point tracks")
    if type(landmark_id_offset) is not int or landmark_id_offset < 0:
        raise ValueError("landmark_id_offset must be a nonnegative integer")
    inverse_camera_matrix = np.linalg.inv(_intrinsics(camera_matrix))
    table_from_camera = _matrix(table_from_camera, "table_from_camera")
    object_from_table = np.linalg.inv(
        _matrix(table_from_object, "table_from_object")
    )
    landmarks: dict[int, np.ndarray] = {}
    depth_supported = 0
    for query_id, position, visible in zip(
        tracks.query_ids, tracks.positions_xy, tracks.visible, strict=True
    ):
        if not visible:
            continue
        depth = _depth_at(
            depth_m, position, search_radius_px=depth_search_radius_px
        )
        if depth is None:
            continue
        depth_supported += 1
        table_point = _table_point(
            position, depth, inverse_camera_matrix, table_from_camera
        )
        point = _transform_point(table_point, object_from_table)
        point.flags.writeable = False
        landmarks[landmark_id_offset + int(query_id)] = point
    report = MetricPointMapReport(
        visible_track_count=int(np.sum(tracks.visible)),
        depth_supported_count=depth_supported,
        accepted_count=len(landmarks),
        rejected_rigidity_count=0,
        rigidity_residual_p50_m=None,
        rigidity_residual_p95_m=None,
    )
    return MappingProxyType(landmarks), report


def registered_metric_observations(
    index: int,
    tracks: CausalPointTracks,
    landmarks: Mapping[int, np.ndarray],
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    table_from_camera: np.ndarray,
    table_from_object: np.ndarray,
    landmark_id_offset: int,
    maximum_rigidity_residual_m: float,
    depth_search_radius_px: int = 1,
) -> tuple[list[MetricObservation], MetricPointMapReport]:
    """Return current object-local XYZ rows, gated against fixed landmarks."""

    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if not np.isfinite(maximum_rigidity_residual_m) or maximum_rigidity_residual_m <= 0.0:
        raise ValueError("maximum rigidity residual must be positive")
    inverse_camera_matrix = np.linalg.inv(_intrinsics(camera_matrix))
    table_from_camera = _matrix(table_from_camera, "table_from_camera")
    object_from_table = np.linalg.inv(
        _matrix(table_from_object, "table_from_object")
    )
    rows: list[MetricObservation] = []
    residuals: list[float] = []
    depth_supported = 0
    rejected = 0
    if tracks.status == "candidate":
        for query_id, position, visible in zip(
            tracks.query_ids, tracks.positions_xy, tracks.visible, strict=True
        ):
            landmark_id = landmark_id_offset + int(query_id)
            if not visible or landmark_id not in landmarks:
                continue
            depth = _depth_at(
                depth_m, position, search_radius_px=depth_search_radius_px
            )
            if depth is None:
                continue
            depth_supported += 1
            table_point = _table_point(
                position, depth, inverse_camera_matrix, table_from_camera
            )
            observed = _transform_point(table_point, object_from_table)
            residual = float(np.linalg.norm(observed - landmarks[landmark_id]))
            residuals.append(residual)
            if residual > maximum_rigidity_residual_m:
                rejected += 1
                continue
            observed.flags.writeable = False
            rows.append((index, landmark_id, observed))
    values = np.asarray(residuals, dtype=np.float64)
    report = MetricPointMapReport(
        visible_track_count=int(np.sum(tracks.visible)),
        depth_supported_count=depth_supported,
        accepted_count=len(rows),
        rejected_rigidity_count=rejected,
        rigidity_residual_p50_m=(
            float(np.quantile(values, 0.50)) if len(values) else None
        ),
        rigidity_residual_p95_m=(
            float(np.quantile(values, 0.95)) if len(values) else None
        ),
    )
    return rows, report


def table_point_prior_residual_m(
    observed_table_m: np.ndarray,
    landmark_object_m: np.ndarray,
    prior_table_from_object: np.ndarray,
) -> float:
    """Shared admission diagnostic for already lifted metric evidence."""
    predicted = _transform_point(landmark_object_m, prior_table_from_object)
    return float(np.linalg.norm(observed_table_m - predicted))


def registered_table_observations(
    index: int,
    tracks: CausalPointTracks,
    landmarks: Mapping[int, np.ndarray],
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    table_from_camera: np.ndarray,
    prior_table_from_object: np.ndarray,
    landmark_id_offset: int,
    maximum_prior_innovation_m: float,
    depth_search_radius_px: int = 1,
) -> tuple[list[MetricObservation], MetricPointMapReport]:
    """Return current table-frame XYZ without using a current base pose.

    The previous absolute pose is used only to reject implausible associations;
    it never participates in converting depth into the table frame.
    """

    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    if not np.isfinite(maximum_prior_innovation_m) or maximum_prior_innovation_m <= 0.0:
        raise ValueError("maximum prior innovation must be positive")
    inverse_camera_matrix = np.linalg.inv(_intrinsics(camera_matrix))
    table_from_camera = _matrix(table_from_camera, "table_from_camera")
    prior = _matrix(prior_table_from_object, "prior_table_from_object")
    rows: list[MetricObservation] = []
    residuals: list[float] = []
    depth_supported = 0
    rejected = 0
    if tracks.status == "candidate":
        for query_id, position, visible in zip(
            tracks.query_ids, tracks.positions_xy, tracks.visible, strict=True
        ):
            landmark_id = landmark_id_offset + int(query_id)
            if not visible or landmark_id not in landmarks:
                continue
            depth = _depth_at(
                depth_m, position, search_radius_px=depth_search_radius_px
            )
            if depth is None:
                continue
            depth_supported += 1
            observed = _table_point(
                position, depth, inverse_camera_matrix, table_from_camera
            )
            residual = table_point_prior_residual_m(observed, landmarks[landmark_id], prior)
            residuals.append(residual)
            if residual > maximum_prior_innovation_m:
                rejected += 1
                continue
            observed.flags.writeable = False
            rows.append((index, landmark_id, observed))
    values = np.asarray(residuals, dtype=np.float64)
    return rows, MetricPointMapReport(
        visible_track_count=int(np.sum(tracks.visible)),
        depth_supported_count=depth_supported,
        accepted_count=len(rows),
        rejected_rigidity_count=rejected,
        rigidity_residual_p50_m=(
            float(np.quantile(values, 0.50)) if len(values) else None
        ),
        rigidity_residual_p95_m=(
            float(np.quantile(values, 0.95)) if len(values) else None
        ),
    )


def bearing_observations(
    index: int,
    tracks: CausalPointTracks,
    landmarks: Mapping[int, np.ndarray],
    *,
    landmark_id_offset: int,
    minimum_confidence: float = 0.0,
) -> list[MetricObservation]:
    """Return camera pixels only; RGB cameras never invent metric depth."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0,1]")
    if tracks.status != "candidate":
        return []
    rows = []
    for query_id, position, visible, confidence in zip(
        tracks.query_ids,
        tracks.positions_xy,
        tracks.visible,
        tracks.confidence,
        strict=True,
    ):
        landmark_id = landmark_id_offset + int(query_id)
        if (
            visible
            and confidence >= minimum_confidence
            and landmark_id in landmarks
        ):
            observed = position.copy()
            observed.flags.writeable = False
            rows.append((index, landmark_id, observed))
    return rows
