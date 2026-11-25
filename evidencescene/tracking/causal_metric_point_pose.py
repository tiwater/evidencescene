"""Causal shared-SE(3) pose correction from multi-camera point tracks.

Metric RGB-D rows and RGB bearing rows are deliberately separate modalities.
Every accepted row belongs to one stable physical camera, while the solver owns
exactly one pose correction.  This prevents a dense camera (or two modalities
from the same camera) from masquerading as independent multi-camera evidence.

The implementation uses a small NumPy-only robust Levenberg-Marquardt solve so
the online path does not acquire SciPy or OpenCV as production dependencies.
It is intended for current-frame factors around an already metric pose prior,
not for unconstrained global localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PinholeCameraModel:
    camera_id: str
    camera_from_table: np.ndarray
    camera_matrix: np.ndarray


@dataclass(frozen=True)
class MetricPointObservation:
    frame_index: int
    timestamp_ns: int
    camera_id: str
    landmark_id: int
    point_prior_object_frame_m: np.ndarray
    uncertainty_m: float


@dataclass(frozen=True)
class BearingPointObservation:
    frame_index: int
    timestamp_ns: int
    camera_id: str
    landmark_id: int
    pixel_xy: np.ndarray
    uncertainty_px: float


@dataclass(frozen=True)
class TableMetricPointObservation:
    """One current RGB-D landmark measured directly in the fixed table frame."""

    frame_index: int
    timestamp_ns: int
    camera_id: str
    landmark_id: int
    point_table_frame_m: np.ndarray
    uncertainty_m: float


@dataclass(frozen=True)
class SharedSe3PoseReport:
    ready: bool
    reason: str
    frame_index: int
    timestamp_ns: int
    physical_camera_ids: tuple[str, ...]
    metric_counts: Mapping[str, int]
    bearing_counts: Mapping[str, int]
    robust_cost: float | None
    iteration_count: int
    jacobian_rank: int | None
    normal_condition_number: float | None
    translation_std_m: tuple[float, float, float] | None
    rotation_std_deg: tuple[float, float, float] | None
    raw_translation_innovation_m: float | None
    raw_rotation_innovation_deg: float | None
    translation_clipped: bool
    rotation_clipped: bool
    factor_latency_ms: float
    prelimit_metric_residual_p95_m: float | None = None
    prelimit_bearing_residual_p95_px: float | None = None


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {shape} array")
    return result


def _transform(value: object, name: str) -> np.ndarray:
    result = _finite_array(value, (4, 4), name)
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{name} rotation must be orthonormal")
    return result


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = _skew(vector)
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    axis = vector / angle
    skew = _skew(axis)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        ) / 2.0
    if math.pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh((rotation + np.eye(3)) / 2.0)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        reference = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        if float(axis @ reference) < 0.0:
            axis = -axis
        return axis * angle
    scale = angle / (2.0 * math.sin(angle))
    return scale * np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )


def _matrix_from_parameters(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_from_vector(parameters[:3])
    result[:3, 3] = parameters[3:]
    return result


def _parameters_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((_rotation_vector(matrix[:3, :3]), matrix[:3, 3]))


def _project(points_table: np.ndarray, model: PinholeCameraModel) -> np.ndarray:
    camera = (
        points_table @ model.camera_from_table[:3, :3].T
        + model.camera_from_table[:3, 3]
    )
    if (camera[:, 2] <= 1e-6).any():
        raise FloatingPointError("landmark projects behind a camera")
    normalized = camera[:, :2] / camera[:, 2:3]
    return normalized @ model.camera_matrix[:2, :2].T + model.camera_matrix[:2, 2]


class CausalMetricPointPoseRuntime:
    """Advance one bounded correction using only current multi-camera rows."""

    def __init__(
        self,
        *,
        landmarks_object_frame_m: Mapping[int, np.ndarray],
        camera_models: Sequence[PinholeCameraModel],
        physical_camera_ids: Sequence[str] | None = None,
        initial_correction: np.ndarray | None = None,
        minimum_physical_camera_count: int = 2,
        minimum_correspondence_count_per_camera: int = 6,
        correction_gain: float = 0.45,
        maximum_translation_innovation_m: float = 0.01,
        maximum_rotation_innovation_deg: float = 5.0,
        maximum_normal_condition_number: float = 1e10,
        maximum_iterations: int = 20,
    ) -> None:
        if minimum_physical_camera_count < 1:
            raise ValueError("shared pose requires at least one physical camera")
        if minimum_correspondence_count_per_camera < 3:
            raise ValueError("each camera requires at least three correspondences")
        if not 0.0 < correction_gain <= 1.0:
            raise ValueError("correction gain must be in (0,1]")
        positive = (
            maximum_translation_innovation_m,
            maximum_rotation_innovation_deg,
            maximum_normal_condition_number,
        )
        if not all(math.isfinite(float(value)) and value > 0.0 for value in positive):
            raise ValueError("innovation and conditioning limits must be positive")
        if maximum_iterations < 1 or maximum_iterations > 100:
            raise ValueError("maximum_iterations must be in [1,100]")

        landmarks: dict[int, np.ndarray] = {}
        for landmark_id, point in landmarks_object_frame_m.items():
            if type(landmark_id) is not int or landmark_id < 0:
                raise ValueError("landmark IDs must be nonnegative integers")
            value = _finite_array(point, (3,), "landmark")
            value = value.copy()
            value.flags.writeable = False
            landmarks[landmark_id] = value
        if len(landmarks) < minimum_correspondence_count_per_camera:
            raise ValueError("the point map has insufficient landmarks")

        requested_camera_ids = (
            tuple(physical_camera_ids)
            if physical_camera_ids is not None
            else tuple(item.camera_id for item in camera_models)
        )
        if (
            len(requested_camera_ids) != len(set(requested_camera_ids))
            or any(not isinstance(camera, str) or not camera for camera in requested_camera_ids)
        ):
            raise ValueError("physical camera IDs must be unique nonempty strings")
        if len(requested_camera_ids) < minimum_physical_camera_count:
            raise ValueError("runtime has insufficient physical cameras")
        models: dict[str, PinholeCameraModel] = {}
        for item in camera_models:
            if not isinstance(item, PinholeCameraModel) or not item.camera_id:
                raise ValueError("camera models need a stable nonempty camera_id")
            if item.camera_id in models:
                raise ValueError("physical camera IDs must be unique")
            if item.camera_id not in requested_camera_ids:
                raise ValueError("projection model references an unregistered physical camera")
            camera_from_table = _transform(
                item.camera_from_table, f"{item.camera_id} camera_from_table"
            ).copy()
            camera_matrix = _finite_array(
                item.camera_matrix, (3, 3), f"{item.camera_id} camera_matrix"
            ).copy()
            if (
                camera_matrix[0, 0] <= 0.0
                or camera_matrix[1, 1] <= 0.0
                or abs(float(np.linalg.det(camera_matrix))) < 1e-12
            ):
                raise ValueError("camera intrinsics must be invertible with positive focal length")
            camera_from_table.flags.writeable = False
            camera_matrix.flags.writeable = False
            models[item.camera_id] = PinholeCameraModel(
                item.camera_id, camera_from_table, camera_matrix
            )
        self.landmarks = MappingProxyType(landmarks)
        self.physical_camera_ids = requested_camera_ids
        self.camera_models = MappingProxyType(models)
        self.state = _transform(
            np.eye(4) if initial_correction is None else initial_correction,
            "initial correction",
        ).copy()
        self.minimum_physical_camera_count = minimum_physical_camera_count
        self.minimum_correspondence_count_per_camera = (
            minimum_correspondence_count_per_camera
        )
        self.correction_gain = float(correction_gain)
        self.maximum_translation_innovation_m = float(
            maximum_translation_innovation_m
        )
        self.maximum_rotation_innovation_deg = float(
            maximum_rotation_innovation_deg
        )
        self.maximum_normal_condition_number = float(
            maximum_normal_condition_number
        )
        self.maximum_iterations = maximum_iterations
        self.last_frame_index: int | None = None
        self.last_timestamp_ns: int | None = None

    def _validate_rows(
        self,
        frame_index: int,
        timestamp_ns: int,
        metric: Sequence[MetricPointObservation],
        bearing: Sequence[BearingPointObservation],
    ) -> tuple[
        tuple[MetricPointObservation, ...],
        tuple[BearingPointObservation, ...],
        dict[str, int],
        dict[str, int],
    ]:
        metric_rows: list[MetricPointObservation] = []
        bearing_rows: list[BearingPointObservation] = []
        metric_counts = {camera: 0 for camera in self.physical_camera_ids}
        bearing_counts = {camera: 0 for camera in self.physical_camera_ids}
        seen: set[tuple[str, str, int]] = set()
        for modality, rows, expected_type, shape, uncertainty_name in (
            ("metric", metric, MetricPointObservation, (3,), "uncertainty_m"),
            ("bearing", bearing, BearingPointObservation, (2,), "uncertainty_px"),
        ):
            for row in rows:
                if not isinstance(row, expected_type):
                    raise TypeError(f"{modality} rows have the wrong observation type")
                if row.frame_index != frame_index or row.timestamp_ns != timestamp_ns:
                    raise ValueError("runtime accepts current-frame observations only")
                if row.camera_id not in self.physical_camera_ids:
                    raise ValueError("observation references an unknown physical camera")
                if modality == "bearing" and row.camera_id not in self.camera_models:
                    raise ValueError("bearing observation has no projection model")
                if row.landmark_id not in self.landmarks:
                    raise ValueError("observation references an unknown landmark")
                key = (row.camera_id, modality, row.landmark_id)
                if key in seen:
                    raise ValueError("duplicate camera/modality/landmark observation")
                seen.add(key)
                uncertainty = float(getattr(row, uncertainty_name))
                if not math.isfinite(uncertainty) or uncertainty <= 0.0:
                    raise ValueError("observation uncertainty must be positive and finite")
                value = (
                    row.point_prior_object_frame_m
                    if modality == "metric"
                    else row.pixel_xy
                )
                _finite_array(value, shape, f"{modality} observation")
                if modality == "metric":
                    metric_rows.append(row)  # type: ignore[arg-type]
                    metric_counts[row.camera_id] += 1
                else:
                    bearing_rows.append(row)  # type: ignore[arg-type]
                    bearing_counts[row.camera_id] += 1
        return tuple(metric_rows), tuple(bearing_rows), metric_counts, bearing_counts

    def _residual_function(
        self,
        *,
        prior_table_from_object: np.ndarray,
        metric: Sequence[MetricPointObservation],
        bearing: Sequence[BearingPointObservation],
        included: tuple[str, ...],
        metric_counts: Mapping[str, int],
        bearing_counts: Mapping[str, int],
    ):
        total_components = sum(
            3 * metric_counts[camera] + 2 * bearing_counts[camera]
            for camera in included
        )
        camera_components = {
            camera: 3 * metric_counts[camera] + 2 * bearing_counts[camera]
            for camera in included
        }
        ordered_lm_ids = tuple(self.landmarks.keys())
        lm_slot = {lid: idx for idx, lid in enumerate(ordered_lm_ids)}
        landmarks_mat = np.asarray(
            [self.landmarks[lid] for lid in ordered_lm_ids], dtype=np.float64
        )
        prior_rot_t = prior_table_from_object[:3, :3].T
        prior_trans = prior_table_from_object[:3, 3]

        camera_specs = []
        for camera in included:
            scale = math.sqrt(
                total_components / (len(included) * camera_components[camera])
            )
            m_rows = [row for row in metric if row.camera_id == camera]
            b_rows = [row for row in bearing if row.camera_id == camera]
            m_spec = None
            if m_rows:
                m_spec = (
                    np.array([lm_slot[row.landmark_id] for row in m_rows], dtype=np.intp),
                    np.asarray(
                        [row.point_prior_object_frame_m for row in m_rows],
                        dtype=np.float64,
                    ),
                    (scale / np.array([row.uncertainty_m for row in m_rows], dtype=np.float64))[:, None],
                )
            b_spec = None
            if b_rows:
                b_spec = (
                    self.camera_models[camera],
                    np.array([lm_slot[row.landmark_id] for row in b_rows], dtype=np.intp),
                    np.asarray([row.pixel_xy for row in b_rows], dtype=np.float64),
                    (scale / np.array([row.uncertainty_px for row in b_rows], dtype=np.float64))[:, None],
                )
            camera_specs.append((m_spec, b_spec))

        def residual(parameters: np.ndarray) -> np.ndarray:
            correction = _matrix_from_parameters(parameters)
            corrected_lms = landmarks_mat @ correction[:3, :3].T + correction[:3, 3]
            parts: list[np.ndarray] = []
            for m_spec, b_spec in camera_specs:
                if m_spec is not None:
                    m_idx, m_target, m_weight = m_spec
                    parts.append(((corrected_lms[m_idx] - m_target) * m_weight).ravel())
                if b_spec is not None:
                    model, b_idx, b_target, b_weight = b_spec
                    table_points = corrected_lms[b_idx] @ prior_rot_t + prior_trans
                    projected = _project(table_points, model)
                    parts.append(((projected - b_target) * b_weight).ravel())
            return np.concatenate(parts)

        return residual

    def step(
        self,
        *,
        frame_index: int,
        timestamp_ns: int,
        prior_table_from_object: np.ndarray,
        metric_observations: Sequence[MetricPointObservation],
        bearing_observations: Sequence[BearingPointObservation],
    ) -> tuple[np.ndarray, SharedSe3PoseReport]:
        started = time.perf_counter()
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a nonnegative integer")
        if self.last_frame_index is not None and frame_index <= self.last_frame_index:
            raise ValueError("frame indices must be strictly increasing")
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("timestamps must be strictly increasing")
        prior = _transform(prior_table_from_object, "prior_table_from_object")
        metric, bearing, metric_counts, bearing_counts = self._validate_rows(
            frame_index,
            timestamp_ns,
            metric_observations,
            bearing_observations,
        )
        landmark_support = {
            camera: {row.landmark_id for row in (*metric, *bearing) if row.camera_id == camera}
            for camera in self.physical_camera_ids
        }
        included = tuple(
            camera
            for camera in sorted(self.physical_camera_ids)
            if len(landmark_support[camera])
            >= self.minimum_correspondence_count_per_camera
        )
        self.last_frame_index = frame_index
        self.last_timestamp_ns = timestamp_ns

        def report(
            *,
            ready: bool,
            reason: str,
            robust_cost: float | None = None,
            iteration_count: int = 0,
            jacobian_rank: int | None = None,
            condition: float | None = None,
            covariance_diagonal: np.ndarray | None = None,
            innovation: tuple[float, float, bool, bool] | None = None,
            measurement: np.ndarray | None = None,
        ) -> SharedSe3PoseReport:
            translation_std = None
            rotation_std = None
            metric_p95 = bearing_p95 = None
            if measurement is not None:
                errors = point_residuals_by_camera(
                    correction=measurement, prior_table_from_object=prior,
                    landmarks_object_frame_m=self.landmarks,
                    camera_models={k: v for k, v in self.camera_models.items() if k in included},
                    metric_observations=[r for r in metric if r.camera_id in included],
                    bearing_observations=[r for r in bearing if r.camera_id in included],
                )
                metric_values = [v for e in errors.values() for v in e["metric_m"]]
                bearing_values = [v for e in errors.values() for v in e["bearing_px"]]
                metric_p95 = float(np.percentile(metric_values, 95)) if metric_values else None
                bearing_p95 = float(np.percentile(bearing_values, 95)) if bearing_values else None
            if covariance_diagonal is not None:
                rotation_std = tuple(
                    float(math.degrees(math.sqrt(max(0.0, value))))
                    for value in covariance_diagonal[:3]
                )
                translation_std = tuple(
                    float(math.sqrt(max(0.0, value)))
                    for value in covariance_diagonal[3:]
                )
            return SharedSe3PoseReport(
                ready=ready,
                reason=reason,
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                physical_camera_ids=included,
                metric_counts=MappingProxyType(dict(metric_counts)),
                bearing_counts=MappingProxyType(dict(bearing_counts)),
                robust_cost=robust_cost,
                iteration_count=iteration_count,
                jacobian_rank=jacobian_rank,
                normal_condition_number=condition,
                translation_std_m=translation_std,
                rotation_std_deg=rotation_std,
                raw_translation_innovation_m=(innovation[0] if innovation else None),
                raw_rotation_innovation_deg=(innovation[1] if innovation else None),
                translation_clipped=(innovation[2] if innovation else False),
                rotation_clipped=(innovation[3] if innovation else False),
                factor_latency_ms=(time.perf_counter() - started) * 1000.0,
                prelimit_metric_residual_p95_m=metric_p95,
                prelimit_bearing_residual_p95_px=bearing_p95,
            )

        if len(included) < self.minimum_physical_camera_count:
            return self.state.copy(), report(
                ready=False, reason="insufficient_physical_camera_count"
            )
        residual = self._residual_function(
            prior_table_from_object=prior,
            metric=metric,
            bearing=bearing,
            included=included,
            metric_counts=metric_counts,
            bearing_counts=bearing_counts,
        )
        parameters = _parameters_from_matrix(self.state)
        damping = 1e-3
        iterations = 0
        try:
            values = residual(parameters)
        except FloatingPointError:
            return self.state.copy(), report(
                ready=False, reason="bearing_projects_behind_camera"
            )
        if len(values) < 6 or not np.isfinite(values).all():
            return self.state.copy(), report(
                ready=False, reason="invalid_residual_vector"
            )
        cost = float(0.5 * np.sum(np.log1p(np.square(values))))
        for iterations in range(1, self.maximum_iterations + 1):
            jacobian = np.empty((len(values), 6), dtype=np.float64)
            for column in range(6):
                epsilon = 1e-6
                shifted = parameters.copy()
                shifted[column] += epsilon
                try:
                    shifted_values = residual(shifted)
                except FloatingPointError:
                    return self.state.copy(), report(
                        ready=False,
                        reason="bearing_projects_behind_camera",
                        robust_cost=cost,
                        iteration_count=iterations,
                    )
                jacobian[:, column] = (shifted_values - values) / epsilon
            weights = 1.0 / np.sqrt(1.0 + np.square(values))
            weighted_jacobian = jacobian * weights[:, None]
            weighted_values = values * weights
            normal = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_values
            diagonal = np.maximum(np.diag(normal), 1e-9)
            try:
                delta = np.linalg.solve(
                    normal + damping * np.diag(diagonal), -gradient
                )
            except np.linalg.LinAlgError:
                return self.state.copy(), report(
                    ready=False,
                    reason="singular_normal_matrix",
                    robust_cost=cost,
                    iteration_count=iterations,
                )
            if not np.isfinite(delta).all():
                return self.state.copy(), report(
                    ready=False,
                    reason="nonfinite_pose_step",
                    robust_cost=cost,
                    iteration_count=iterations,
                )
            candidate = parameters + delta
            try:
                candidate_values = residual(candidate)
            except FloatingPointError:
                damping = min(1e9, damping * 10.0)
                continue
            if not np.isfinite(candidate_values).all():
                return self.state.copy(), report(
                    ready=False,
                    reason="nonfinite_residual_vector",
                    robust_cost=cost,
                    iteration_count=iterations,
                )
            candidate_cost = float(
                0.5 * np.sum(np.log1p(np.square(candidate_values)))
            )
            if candidate_cost < cost:
                parameters = candidate
                values = candidate_values
                if abs(cost - candidate_cost) < 1e-9 or float(np.linalg.norm(delta)) < 1e-8:
                    cost = candidate_cost
                    break
                cost = candidate_cost
                damping = max(1e-9, damping / 3.0)
            else:
                damping = min(1e9, damping * 10.0)

        jacobian = np.empty((len(values), 6), dtype=np.float64)
        for column in range(6):
            shifted = parameters.copy()
            shifted[column] += 1e-6
            jacobian[:, column] = (residual(shifted) - values) / 1e-6
        weights = 1.0 / np.sqrt(1.0 + np.square(values))
        weighted_jacobian = jacobian * weights[:, None]
        normal = weighted_jacobian.T @ weighted_jacobian
        rank = int(np.linalg.matrix_rank(weighted_jacobian, tol=1e-8))
        condition = float(np.linalg.cond(normal))
        if rank < 6:
            return self.state.copy(), report(
                ready=False,
                reason="rank_deficient_shared_pose",
                robust_cost=cost,
                iteration_count=iterations,
                jacobian_rank=rank,
                condition=condition,
            )
        if not math.isfinite(condition) or condition > self.maximum_normal_condition_number:
            return self.state.copy(), report(
                ready=False,
                reason="ill_conditioned_shared_pose",
                robust_cost=cost,
                iteration_count=iterations,
                jacobian_rank=rank,
                condition=condition,
            )
        dof = max(1, len(values) - 6)
        variance_scale = float(np.sum(np.square(values * weights)) / dof)
        covariance_diagonal = np.diag(np.linalg.inv(normal)) * variance_scale

        measurement = _matrix_from_parameters(parameters)
        relative = np.linalg.inv(self.state) @ measurement
        rotation_vector = _rotation_vector(relative[:3, :3])
        translation = relative[:3, 3].copy()
        raw_rotation = float(np.linalg.norm(rotation_vector))
        raw_translation = float(np.linalg.norm(translation))
        maximum_rotation = math.radians(self.maximum_rotation_innovation_deg)
        rotation_clipped = raw_rotation > maximum_rotation
        translation_clipped = raw_translation > self.maximum_translation_innovation_m
        if rotation_clipped:
            rotation_vector *= maximum_rotation / raw_rotation
        if translation_clipped:
            translation *= self.maximum_translation_innovation_m / raw_translation
        bounded_step = _matrix_from_parameters(
            np.concatenate(
                (
                    self.correction_gain * rotation_vector,
                    self.correction_gain * translation,
                )
            )
        )
        self.state = self.state @ bounded_step
        return self.state.copy(), report(
            ready=True,
            reason="bounded_shared_se3_update",
            robust_cost=cost,
            iteration_count=iterations,
            jacobian_rank=rank,
            condition=condition,
            covariance_diagonal=covariance_diagonal,
            measurement=measurement,
            innovation=(
                raw_translation,
                math.degrees(raw_rotation),
                translation_clipped,
                rotation_clipped,
            ),
        )


class CausalAbsoluteMetricPointPoseRuntime:
    """Estimate absolute ``table_from_object`` without a current base pose input.

    The validated correction solver is reused with an identity outer prior.
    Object landmarks transformed by its internal state therefore live directly
    in table coordinates.  The public step API intentionally has no incumbent
    pose argument, so a caller cannot feed the answer this runtime replaces.
    """

    def __init__(
        self,
        *,
        landmarks_object_frame_m: Mapping[int, np.ndarray],
        camera_models: Sequence[PinholeCameraModel],
        initial_table_from_object: np.ndarray,
        minimum_physical_camera_count: int = 2,
        minimum_correspondence_count_per_camera: int = 6,
        pose_gain: float = 0.45,
        maximum_translation_innovation_m: float = 0.01,
        maximum_rotation_innovation_deg: float = 5.0,
        maximum_normal_condition_number: float = 1e10,
        maximum_iterations: int = 20,
    ) -> None:
        self._runtime = CausalMetricPointPoseRuntime(
            landmarks_object_frame_m=landmarks_object_frame_m,
            camera_models=camera_models,
            initial_correction=initial_table_from_object,
            minimum_physical_camera_count=minimum_physical_camera_count,
            minimum_correspondence_count_per_camera=(
                minimum_correspondence_count_per_camera
            ),
            correction_gain=pose_gain,
            maximum_translation_innovation_m=maximum_translation_innovation_m,
            maximum_rotation_innovation_deg=maximum_rotation_innovation_deg,
            maximum_normal_condition_number=maximum_normal_condition_number,
            maximum_iterations=maximum_iterations,
        )

    @property
    def state(self) -> np.ndarray:
        result = self._runtime.state.copy()
        result.flags.writeable = False
        return result

    def step(
        self,
        *,
        frame_index: int,
        timestamp_ns: int,
        metric_observations: Sequence[TableMetricPointObservation],
        bearing_observations: Sequence[BearingPointObservation],
    ) -> tuple[np.ndarray, SharedSe3PoseReport]:
        metric = []
        for row in metric_observations:
            if not isinstance(row, TableMetricPointObservation):
                raise TypeError(
                    "absolute metric rows must be TableMetricPointObservation"
                )
            metric.append(
                MetricPointObservation(
                    frame_index=row.frame_index,
                    timestamp_ns=row.timestamp_ns,
                    camera_id=row.camera_id,
                    landmark_id=row.landmark_id,
                    point_prior_object_frame_m=row.point_table_frame_m,
                    uncertainty_m=row.uncertainty_m,
                )
            )
        return self._runtime.step(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            prior_table_from_object=np.eye(4, dtype=np.float64),
            metric_observations=metric,
            bearing_observations=bearing_observations,
        )


def point_residuals_by_camera(
    *,
    correction: np.ndarray,
    prior_table_from_object: np.ndarray,
    landmarks_object_frame_m: Mapping[int, np.ndarray],
    camera_models: Mapping[str, PinholeCameraModel],
    metric_observations: Sequence[MetricPointObservation],
    bearing_observations: Sequence[BearingPointObservation],
) -> Mapping[str, Mapping[str, tuple[float, ...]]]:
    """Return unnormalized residual magnitudes for validation and camera LOO."""

    transform = _transform(correction, "correction")
    prior = _transform(prior_table_from_object, "prior_table_from_object")
    corrected = {
        landmark_id: _finite_array(point, (3,), "landmark") @ transform[:3, :3].T
        + transform[:3, 3]
        for landmark_id, point in landmarks_object_frame_m.items()
    }
    camera_ids = sorted(
        set(camera_models)
        | {row.camera_id for row in metric_observations}
        | {row.camera_id for row in bearing_observations}
    )
    result: dict[str, dict[str, list[float]]] = {
        camera: {"metric_m": [], "bearing_px": []} for camera in camera_ids
    }
    for row in metric_observations:
        if row.camera_id not in result or row.landmark_id not in corrected:
            raise ValueError("metric residual row references unknown evidence")
        result[row.camera_id]["metric_m"].append(
            float(
                np.linalg.norm(
                    corrected[row.landmark_id]
                    - _finite_array(
                        row.point_prior_object_frame_m,
                        (3,),
                        "metric observation",
                    )
                )
            )
        )
    for camera in camera_models:
        rows = [row for row in bearing_observations if row.camera_id == camera]
        if not rows:
            continue
        if any(row.landmark_id not in corrected for row in rows):
            raise ValueError("bearing residual row references an unknown landmark")
        object_points = np.asarray([corrected[row.landmark_id] for row in rows])
        table_points = object_points @ prior[:3, :3].T + prior[:3, 3]
        pixels = _project(table_points, camera_models[camera])
        result[camera]["bearing_px"].extend(
            float(np.linalg.norm(pixel - _finite_array(row.pixel_xy, (2,), "pixel")))
            for row, pixel in zip(rows, pixels, strict=True)
        )
    return MappingProxyType(
        {
            camera: MappingProxyType(
                {modality: tuple(values) for modality, values in modalities.items()}
            )
            for camera, modalities in result.items()
        }
    )
