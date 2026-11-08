"""Consume original registered masks and revised poses into an observed surface.

The surface persists in the backend object gauge. It is not a complete object,
semantic body frame, table/world pose, or independently validated reconstruction.
"""
from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4
import hashlib

import numpy as np

from evidencescene.primitives.object_bound_evidence import bound_witness
from evidencescene.primitives.task_profile import PhysicalBounds

from evidencescene.primitives.bundletrack_live_input import (
    PreparedFeatureSourceBinding,
    RegisteredBundleTrackInput,
)
from evidencescene.primitives.geometry import (
    DEFAULT_MAXIMUM_SURFACE_POINT_COUNT,
    DEFAULT_MINIMUM_FRAME_SUPPORT,
    DEFAULT_PIXEL_STRIDE,
    DEFAULT_VOXEL_SIZE_M,
    _backproject_masked_depth,
    _transform_points,
    _validate_rigid_transform,
)
from evidencescene.surface.object_surface_fusion import RevisableSurfelMap, measured_free_space_margin


WIRE_SURFACE_POINT_COUNT = 1500
_MEASURED_TASK_FEATURE_SCHEMA = "ticos.measured-task-feature-owner.v1"
_MEASURED_TASK_FEATURE_OWNER = "RegisteredRgbdTaskFeatureOwner"
_MEASURED_TASK_FEATURE_METHOD = "current_registered_rgbd_feature_fit.v1"
_MEASURED_TASK_FEATURE_SCOPE = "current_registered_rgbd_frame"
_FORBIDDEN_FEATURE_PROVENANCE_TERMS = (
    "prior",
    "shadow",
    "display",
    "synthetic",
    "fallback",
    "template",
)


def _same_tracker_new_sensor_epoch(old: tuple, new: tuple) -> bool:
    return (
        isinstance(old, tuple) and isinstance(new, tuple)
        and len(old) == len(new) and len(old) > 2
        and old[0] == new[0] and old[2:] == new[2:] and old[1] != new[1]
    )


def _inverse_rigid_pose(pose: np.ndarray) -> np.ndarray:
    """Invert an already validated SE(3) pose without a general 4x4 solve."""
    rotation = pose[:3, :3]
    inverse_rotation = rotation.T
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = inverse_rotation
    inverse[:3, 3] = -inverse_rotation @ pose[:3, 3]
    return inverse


def _finite_feature_vector(value, *, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(
            not isinstance(component, bool)
            and isinstance(component, (int, float))
            and np.isfinite(float(component))
            for component in value
        )
    )


def _validated_feature_geometry(
    geometry,
    *,
    coordinate_frame: str,
    camera_from_object: np.ndarray | None,
) -> tuple[dict | None, list[str]]:
    if not isinstance(geometry, Mapping):
        return None, ["task_feature_geometry_missing"]
    schema = geometry.get("schema")
    if schema == "ticos.measured-point-feature.v1":
        point = geometry.get("point_m")
        if not _finite_feature_vector(point, length=3):
            return None, ["task_feature_point_invalid"]
        center = np.asarray(point, dtype=np.float64)
        normal = None
    elif schema in {"ticos.measured-circular-aperture.v1", "ticos.measured-aperture-plane.v1"}:
        center = geometry.get("center_m")
        normal = geometry.get("normal")
        radius = geometry.get("radius_m")
        blockers = []
        if not _finite_feature_vector(center, length=3):
            blockers.append("task_feature_aperture_center_invalid")
        if not _finite_feature_vector(normal, length=3):
            blockers.append("task_feature_aperture_normal_invalid")
        expected_orientation = "unoriented" if schema == "ticos.measured-aperture-plane.v1" else "outward_verified"
        if geometry.get("normal_orientation") != expected_orientation:
            blockers.append("task_feature_aperture_normal_unverified")
        if geometry.get("rotational_symmetry") != "continuous_about_normal":
            blockers.append("task_feature_aperture_symmetry_unverified")
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(float(radius))
            or float(radius) <= 0.0
        ):
            blockers.append("task_feature_aperture_radius_invalid")
        if blockers:
            return None, blockers
        center = np.asarray(center, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        if not np.isclose(float(np.linalg.norm(normal)), 1.0, atol=1e-3, rtol=0.0):
            return None, ["task_feature_aperture_normal_not_unit_length"]
    else:
        return None, ["task_feature_geometry_schema_unsupported"]

    if coordinate_frame == "object_gauge":
        object_from_source = np.eye(4, dtype=np.float64)
    elif coordinate_frame == "camera_local":
        if camera_from_object is None:
            return None, ["task_feature_camera_pose_required"]
        object_from_source = _inverse_rigid_pose(camera_from_object)
    else:
        return None, ["task_feature_coordinate_frame_unsupported"]

    object_center = object_from_source[:3, :3] @ center + object_from_source[:3, 3]
    result = dict(geometry)
    if schema == "ticos.measured-point-feature.v1":
        result["point_m"] = object_center.tolist()
    else:
        object_normal = object_from_source[:3, :3] @ normal
        result["center_m"] = object_center.tolist()
        result["normal"] = object_normal.tolist()
    result["coordinate_frame"] = "backend_tracking_gauge_not_semantic_body_frame"
    return result, []


def _transform_feature_geometry(geometry, transform: np.ndarray) -> dict | None:
    if not isinstance(geometry, Mapping):
        return None
    schema = geometry.get("schema")
    result = dict(geometry)
    if schema == "ticos.measured-point-feature.v1":
        point = geometry.get("point_m")
        if not _finite_feature_vector(point, length=3):
            return None
        result["point_m"] = (
            transform[:3, :3] @ np.asarray(point, dtype=np.float64)
            + transform[:3, 3]
        ).tolist()
        direction = geometry.get("direction")
        if direction is not None:
            if (not _finite_feature_vector(direction, length=3)
                    or not np.isclose(
                        float(np.linalg.norm(np.asarray(direction, dtype=np.float64))),
                        1.0,
                        atol=1e-3,
                        rtol=0.0,
                    )):
                return None
            result["direction"] = (
                transform[:3, :3] @ np.asarray(direction, dtype=np.float64)
            ).tolist()
        return result
    if schema in {"ticos.measured-circular-aperture.v1", "ticos.measured-aperture-plane.v1"}:
        center = geometry.get("center_m")
        normal = geometry.get("normal")
        if (
            not _finite_feature_vector(center, length=3)
            or not _finite_feature_vector(normal, length=3)
        ):
            return None
        result["center_m"] = (
            transform[:3, :3] @ np.asarray(center, dtype=np.float64)
            + transform[:3, 3]
        ).tolist()
        result["normal"] = (
            transform[:3, :3] @ np.asarray(normal, dtype=np.float64)
        ).tolist()
        return result
    return None


class CausalObjectSurface:
    def __init__(self, *, maximum_voxels=200_000, maximum_live_bytes=128 * 1024 * 1024,
                 pixel_stride=DEFAULT_PIXEL_STRIDE,
                 shadow_maximum_raw_depth_m: float | None = None,
                 physical_bounds: PhysicalBounds | None = None,
                 prior_profile_sha256: str | None = None,
                 shadow_visibility_margin_m: float | None = None,
                 shadow_bidirectional_visibility: bool = False,
                 shadow_retired_correction: bool = False,
                 shadow_retired_mask_conflict: bool = False,
                 shadow_retired_supplemental_conflict: bool = False,
                 shadow_retired_supplemental_mask_only: bool = False,
                 shadow_retired_supplemental_exclusive: bool = False,
                 shadow_frozen_pose_history: bool = False,
                 shadow_feature_history: bool = False,
                 shadow_feature_archive: bool = False,
                 shadow_archive_continuity_gate: bool = False,
                 shadow_feature_admission_gate: bool = False,
                 shadow_feature_continuity: dict | None = None,
                 shadow_supplemental_observations: bool = False):
        # Defaults preserve 4mm voxels, stride-3 sampling and two-observation
        # support. Optional stride is used by the offline paired study; the
        # live factory does not override it. Resource limits are unchanged.
        self._map = RevisableSurfelMap(voxel_size_m=DEFAULT_VOXEL_SIZE_M,
            maximum_voxels=maximum_voxels, maximum_live_bytes=maximum_live_bytes)
        self.maximum_live_bytes = maximum_live_bytes
        # Offline opt-in only; the live factory never supplies this option.
        # This uses the existing owner/map and does not change native masks or
        # poses. A conflict means uncertain membership, not empty geometry.
        if shadow_visibility_margin_m is not None and (
                not np.isfinite(shadow_visibility_margin_m)
                or shadow_visibility_margin_m <= 0):
            raise ValueError("shadow visibility margin must be finite and positive")
        self._shadow_visibility_margin_m = shadow_visibility_margin_m
        if type(shadow_bidirectional_visibility) is not bool:
            raise ValueError('shadow bidirectional visibility must be boolean')
        if shadow_bidirectional_visibility and shadow_visibility_margin_m is None:
            raise ValueError('shadow bidirectional visibility requires a visibility margin')
        self._shadow_bidirectional_visibility = shadow_bidirectional_visibility
        if type(shadow_retired_correction) is not bool:
            raise ValueError("shadow retired correction must be boolean")
        self._shadow_retired_correction = shadow_retired_correction
        # QUARANTINED offline reproduction only: failed to remove the target
        # XY extrema while reducing other support. Never enabled by live factory.
        # docs/operations/retired-mask-conflict-shadow-decision-20261017.json
        if type(shadow_retired_mask_conflict)is not bool or (shadow_retired_mask_conflict and not shadow_retired_correction):
            raise ValueError('Explicit retired-mask shadow requires original retired correction')
        self._shadow_retired_mask_conflict=shadow_retired_mask_conflict
        if (type(shadow_retired_supplemental_conflict) is not bool or
                (shadow_retired_supplemental_conflict and
                 not (shadow_retired_correction and shadow_supplemental_observations))):
            raise ValueError('Supplemental retired conflict requires both shadow owners')
        self._shadow_retired_supplemental_conflict = shadow_retired_supplemental_conflict
        if (type(shadow_retired_supplemental_mask_only) is not bool or
                (shadow_retired_supplemental_mask_only and not shadow_retired_supplemental_conflict)):
            raise ValueError('Supplemental mask-only ablation requires supplemental conflict shadow')
        self._shadow_retired_supplemental_mask_only = shadow_retired_supplemental_mask_only
        if (type(shadow_retired_supplemental_exclusive) is not bool or
                (shadow_retired_supplemental_exclusive and not shadow_retired_supplemental_mask_only)):
            raise ValueError('Exclusive-camera ablation requires supplemental mask-only shadow')
        self._shadow_retired_supplemental_exclusive = shadow_retired_supplemental_exclusive
        # The paired incremental shadow never revises historical poses. Verify
        # that producer contract against this owner's retained records, before
        # touching the map. Leave revising backends' default behavior unchanged.
        if type(shadow_frozen_pose_history) is not bool:
            raise ValueError('shadow frozen pose history must be boolean')
        self._shadow_frozen_pose_history = shadow_frozen_pose_history
        if type(shadow_supplemental_observations) is not bool or (
                shadow_supplemental_observations and not shadow_frozen_pose_history):
            raise ValueError('supplemental shadow requires frozen pose history')
        self._shadow_supplemental_observations = shadow_supplemental_observations
        self._supplemental_keys = {}
        self._supplemental_latest = None
        if type(shadow_feature_history) is not bool or (shadow_feature_history and not shadow_frozen_pose_history):
            raise ValueError('shadow feature history requires frozen pose history')
        self._shadow_feature_history = shadow_feature_history
        if type(shadow_feature_archive) is not bool or (shadow_feature_archive and not shadow_feature_history):
            raise ValueError('shadow feature archive requires existing feature history')
        self._shadow_feature_archive = shadow_feature_archive
        if type(shadow_archive_continuity_gate) is not bool or (shadow_archive_continuity_gate and
                (not shadow_feature_archive or shadow_feature_continuity is None)):
            raise ValueError('shadow archive gate requires archive and continuity diagnostics')
        self._shadow_archive_continuity_gate = shadow_archive_continuity_gate
        if type(shadow_feature_admission_gate) is not bool or (shadow_feature_admission_gate and shadow_feature_continuity is None):
            raise ValueError('shadow feature admission requires continuity diagnostics')
        self._shadow_feature_admission_gate = shadow_feature_admission_gate
        self._archived_feature_measurements = {}
        self._shadow_feature_continuity = None
        if shadow_feature_continuity is not None:
            if not shadow_feature_history:
                raise ValueError('continuity diagnostics require existing feature history')
            transform = _validate_rigid_transform(shadow_feature_continuity['table_from_camera'], name='existing table_from_camera').copy()
            limits = deepcopy(shadow_feature_continuity['limits'])
            for key in ('speed_m_s', 'acceleration_m_s2', 'maximum_anchor_age_s'):
                if isinstance(limits[key], bool) or not np.isfinite(limits[key]) or limits[key] <= 0:
                    raise ValueError('invalid shadow feature motion limit')
            self._shadow_feature_continuity = dict(table_from_camera=transform, limits=limits)
        self._shadow_retired_conflicts = {}
        if type(pixel_stride) is not int or pixel_stride < 1:
            raise ValueError("pixel_stride must be a positive integer")
        self.pixel_stride = pixel_stride
        # Offline range-only ablation, supplied from the captured native zfar.
        # The live factory does not enable this; default sampling is unchanged.
        if shadow_maximum_raw_depth_m is not None and (
                isinstance(shadow_maximum_raw_depth_m, bool)
                or not isinstance(shadow_maximum_raw_depth_m, (int, float))
                or not np.isfinite(shadow_maximum_raw_depth_m)
                or shadow_maximum_raw_depth_m <= 0):
            raise ValueError("shadow maximum raw depth must be finite and positive")
        self._shadow_maximum_raw_depth_m = shadow_maximum_raw_depth_m
        if physical_bounds is not None and not isinstance(physical_bounds, PhysicalBounds):
            raise ValueError("physical_bounds must be a validated PhysicalBounds prior")
        if prior_profile_sha256 is not None and (
                not isinstance(prior_profile_sha256, str)
                or len(prior_profile_sha256) != 64
                or any(c not in "0123456789abcdef" for c in prior_profile_sha256)):
            raise ValueError("prior_profile_sha256 must be a lowercase SHA-256 digest")
        self._physical_bounds = physical_bounds
        self._prior_profile_sha256 = prior_profile_sha256
        # This owner persists one backend object gauge for its lifetime.
        # Name that existing gauge, rather than treating the generic frame
        # description or a reusable stream key as its identity. A re-anchor
        # constructs another owner and must never appear aligned by name.
        self._gauge_id = f"object-gauge:{uuid4().hex}"
        self._records = {}
        self._key = None
        self._last_color_ns = self._last_depth_ns = None
        self._last_index = -1
        self._last_observed_depth_ns = None
        self._snapshot_metadata = {}
        self._closed = False

    def _normalize_measured_task_features(self, frame, candidate) -> list[dict]:
        records = candidate.get("measured_task_features")
        if records is None:
            return []
        if not isinstance(records, (list, tuple)):
            records = [None]
        camera_from_object = None
        if candidate.get("status") == "candidate":
            try:
                camera_from_object = _validate_rigid_transform(
                    candidate.get("camera_from_object"),
                    name="feature camera_from_object",
                )
            except (TypeError, ValueError):
                camera_from_object = None
        normalized = []
        seen: dict[str, int] = {}
        for raw in records:
            raw = raw if isinstance(raw, Mapping) else {}
            feature_id = raw.get("feature_id")
            feature_id = feature_id.strip() if isinstance(feature_id, str) else None
            blockers = []
            raw_validation_blockers = raw.get("validation_blockers")
            if isinstance(raw_validation_blockers, (list, tuple)):
                blockers.extend(
                    item
                    for item in raw_validation_blockers
                    if isinstance(item, str) and item
                )
            if raw.get("schema") != _MEASURED_TASK_FEATURE_SCHEMA:
                blockers.append("task_feature_owner_schema_mismatch")
            if raw.get("source_owner") != _MEASURED_TASK_FEATURE_OWNER:
                blockers.append("task_feature_measurement_owner_not_registered_owner")
            if raw.get("authority") != "owner_output":
                blockers.append("task_feature_authority_below_owner_output")
            if raw.get("status") != "available_current":
                blockers.append("task_feature_not_current")
            if not feature_id:
                blockers.append("task_feature_id_invalid")
            coordinate_frame = raw.get("coordinate_frame")
            if not isinstance(coordinate_frame, str):
                blockers.append("task_feature_coordinate_frame_missing")
                coordinate_frame = ""
            source_reference = raw.get("feature_source_reference")
            if not isinstance(source_reference, str) or not source_reference.strip():
                blockers.append("task_feature_source_reference_missing")
                source_reference = None
            elif any(
                term in source_reference.lower()
                for term in _FORBIDDEN_FEATURE_PROVENANCE_TERMS
            ):
                blockers.append(
                    "task_feature_feature_source_reference_not_measurement_provenance"
                )
            measurement_owner = raw.get("measurement_owner")
            if measurement_owner != _MEASURED_TASK_FEATURE_OWNER:
                blockers.append("task_feature_measurement_owner_not_registered_owner")
                measurement_owner = None
            if raw.get("measurement_source") != "current_registered_rgbd_depth":
                blockers.append("task_feature_measurement_source_invalid")
            if raw.get("measurement_method") != _MEASURED_TASK_FEATURE_METHOD:
                blockers.append("task_feature_measurement_method_invalid")
            if raw.get("measurement_scope") != _MEASURED_TASK_FEATURE_SCOPE:
                blockers.append("task_feature_measurement_scope_invalid")
            if raw.get("independent_of_configured_prior") is not True:
                blockers.append("task_feature_prior_independence_unverified")
            measurement_support = raw.get("measurement_support")
            if not isinstance(measurement_support, Mapping):
                blockers.append("task_feature_measurement_support_missing")
            else:
                for support_key in ("pixel_count", "depth_sample_count"):
                    support_value = measurement_support.get(support_key)
                    if (
                        isinstance(support_value, bool)
                        or not isinstance(support_value, int)
                        or support_value <= 0
                    ):
                        blockers.append(
                            f"task_feature_measurement_{support_key}_invalid"
                        )
                support_hash = measurement_support.get("support_evidence_sha256")
                if (
                    not isinstance(support_hash, str)
                    or len(support_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in support_hash
                    )
                ):
                    blockers.append("task_feature_measurement_support_hash_invalid")
            expected_hash = frame.evidence_hash
            expected_mask = candidate.get("source_mask_evidence_sha256")
            expected_stream = list(frame.stream_key)
            binding = frame.feature_source_binding
            if isinstance(binding, PreparedFeatureSourceBinding) and raw.get("measurement_input_evidence_hash") == binding.observation_hash:
                if (binding.registered_stream_key != frame.stream_key
                        or binding.color_timestamp_ns != frame.color_timestamp_ns
                        or binding.depth_timestamp_ns != frame.depth_timestamp_ns
                        or binding.registered_mask_hash != hashlib.sha256(frame.mask.tobytes()).hexdigest()):
                    blockers.append("task_feature_prepared_source_binding_stale")
                else:
                    expected_hash = binding.observation_hash
                    expected_mask = binding.source_mask_hash
                    expected_stream = list(binding.feature_stream_key)
            if raw.get("measurement_input_evidence_hash") != expected_hash:
                blockers.append("task_feature_measurement_input_mismatch")
            if raw.get("measurement_depth_timestamp_ns") != str(frame.depth_timestamp_ns):
                blockers.append("task_feature_measurement_timestamp_mismatch")
            if raw.get("measurement_mask_evidence_sha256") != expected_mask:
                blockers.append("task_feature_measurement_mask_mismatch")
            if raw.get("stream_key") != expected_stream:
                blockers.append("task_feature_measurement_stream_mismatch")
            measurement_coordinate_frame = coordinate_frame or None
            geometry_input = raw.get("feature_geometry")
            frame_color_id = getattr(frame, "color_frame_id", None)
            frame_depth_id = getattr(frame, "depth_frame_id", None)
            bridge = getattr(frame, "color_from_depth", None)
            if (
                frame_color_id
                and frame_depth_id
                and coordinate_frame == frame_depth_id
                and frame_depth_id != frame_color_id
            ):
                if bridge is None:
                    blockers.append("task_feature_rgbd_color_depth_bridge_unavailable")
                    geometry_input = None
                else:
                    try:
                        bridge = _validate_rigid_transform(
                            bridge,
                            name="current color_from_depth",
                        )
                    except (TypeError, ValueError):
                        blockers.append("task_feature_rgbd_color_depth_bridge_invalid")
                        geometry_input = None
                    else:
                        geometry_input = _transform_feature_geometry(
                            geometry_input,
                            bridge,
                        )
                        if geometry_input is None:
                            blockers.append("task_feature_rgbd_color_depth_geometry_invalid")
                        measurement_coordinate_frame = frame_color_id
            elif (
                frame_color_id
                and frame_depth_id
                and coordinate_frame not in {
                    "camera_local",
                    "object_gauge",
                    frame_color_id,
                    frame_depth_id,
                }
            ):
                blockers.append("task_feature_coordinate_frame_not_bound_to_tracker")
            validation_coordinate_frame = (
                "camera_local"
                if frame_color_id and measurement_coordinate_frame == frame_color_id
                else measurement_coordinate_frame or coordinate_frame
            )
            if blockers:
                geometry = None
            else:
                geometry, geometry_blockers = _validated_feature_geometry(
                    geometry_input,
                    coordinate_frame=validation_coordinate_frame,
                    camera_from_object=camera_from_object,
                )
                blockers.extend(geometry_blockers)
            if feature_id is not None:
                if feature_id in seen:
                    blockers.append("task_feature_duplicate_id")
                    normalized[seen[feature_id]]["validation_blockers"].append(
                        "task_feature_duplicate_id"
                    )
                else:
                    seen[feature_id] = len(normalized)
            normalized.append({
                "feature_id": feature_id,
                "source_owner": "CausalObjectSurface",
                "authority": "owner_output",
                "status": "available_current" if not blockers else "blocked",
                "gauge_id": self._gauge_id,
                "stream_key": list(frame.stream_key),
                "as_of_depth_timestamp_ns": str(frame.depth_timestamp_ns),
                "input_evidence_hash": frame.evidence_hash,
                "original_measurement_identity": {
                    "input_evidence_hash": raw.get("measurement_input_evidence_hash"),
                    "stream_key": deepcopy(raw.get("stream_key")),
                    "mask_evidence_sha256": raw.get("measurement_mask_evidence_sha256"),
                },
                "source_mask_evidence_sha256": candidate.get(
                    "source_mask_evidence_sha256"
                ),
                "feature_source_reference": source_reference,
                "feature_geometry": geometry,
                "measurement_coordinate_frame": coordinate_frame or None,
                "owner_coordinate_frame": measurement_coordinate_frame,
                "measurement_frame_bridge": (
                    "validated_current_rgbd_color_from_depth"
                    if measurement_coordinate_frame == frame_color_id
                    and coordinate_frame == frame_depth_id
                    and frame_color_id != frame_depth_id
                    else None
                ),
                "measurement_owner": measurement_owner,
                "measurement_source": raw.get("measurement_source"),
                # Provenance describes the original measurement, including its
                # native sign convention; transforming a vector does not verify
                # a physical exit direction or cross-view surface identity.
                "measurement_direction_provenance": deepcopy(
                    raw.get("measurement_direction_provenance")
                ),
                "validation_blockers": sorted(set(blockers)),
            })
        return normalized

    def update(self, frame, candidate, *, supplemental_frame=None,
               supplemental_bridge=None, supplemental_maximum_skew_ns=None):
        if self._closed:
            raise RuntimeError("object surface closed")
        if not isinstance(frame, RegisteredBundleTrackInput):
            raise ValueError("registered surface input required")
        supplemental = None
        if any(value is not None for value in
               (supplemental_frame, supplemental_bridge, supplemental_maximum_skew_ns)):
            if not self._shadow_supplemental_observations:
                raise ValueError('supplemental observations require explicit shadow opt-in')
            from evidencescene.surface.supplemental_surface_observation import prepare_packet
            supplemental = prepare_packet(frame, supplemental_frame, candidate,
                supplemental_bridge, maximum_skew_ns=supplemental_maximum_skew_ns)
            if self._supplemental_latest is not None:
                stream, color_ns, depth_ns = self._supplemental_latest
                if (supplemental_frame.stream_key != stream
                        or supplemental_frame.color_timestamp_ns <= color_ns
                        or supplemental_frame.depth_timestamp_ns <= depth_ns):
                    raise ValueError('supplemental source must remain bound and advance causally')
        if self._key is not None and self._key != frame.stream_key:
            if not _same_tracker_new_sensor_epoch(self._key, frame.stream_key):
                raise ValueError("surface object, camera, source epoch or calibration changed")
        elif (self._last_color_ns is not None and (frame.color_timestamp_ns <= self._last_color_ns
                or frame.depth_timestamp_ns <= self._last_depth_ns)):
            raise ValueError("surface input must advance causally")
        if (candidate.get("stream_key") != list(frame.stream_key)
                or candidate.get("input_evidence_hash") != frame.evidence_hash
                or candidate.get("color_timestamp_ns") != str(frame.color_timestamp_ns)
                or candidate.get("depth_timestamp_ns") != str(frame.depth_timestamp_ns)):
            raise ValueError("surface requires the exact current pose input")
        index = candidate.get("output_index")
        if type(index) is not int or index != self._last_index + 1:
            raise ValueError("surface requires consecutive backend updates")
        history = candidate.get("retained_pose_snapshot")
        if (not isinstance(history, dict)
                or history.get("scope") != "currently_retained_backend_observations_not_complete_trajectory"
                or history.get("stream_key") != list(frame.stream_key)
                or history.get("as_of_color_timestamp_ns") != str(frame.color_timestamp_ns)
                or history.get("as_of_depth_timestamp_ns") != str(frame.depth_timestamp_ns)):
            raise ValueError("surface requires identity-bound current native history")
        identity = f"{index:06d}"
        # A failed native frame can be absent from the retained history: the
        # BundleTrack owner has already forgotten it and the surface will
        # discard its record below. Avoid backprojecting that doomed frame,
        # while retaining the original evidence path for candidates and
        # retained NO_BA frames that may receive a revised pose later.
        history_observations = history["observations"]
        current_retained = any(
            row.get("backend_frame_id") == identity
            for row in history_observations
        )
        backprojection_performed = candidate.get("status") == "candidate" or current_retained
        sample_stride = self.pixel_stride
        if backprojection_performed:
            points = _backproject_masked_depth(
                frame.depth_mm,
                frame.mask,
                frame.intrinsics,
                pixel_stride=self.pixel_stride,
                depth_scale_m=0.001,
            )
            # A grazing mask can hold valid depth on pixels the stride lattice
            # never visits. Keep the lattice for frames it already samples, and
            # use every valid pixel only when that sample is empty.
            if len(points) == 0 and self.pixel_stride > 1:
                dense = _backproject_masked_depth(
                    frame.depth_mm,
                    frame.mask,
                    frame.intrinsics,
                    pixel_stride=1,
                    depth_scale_m=0.001,
                )
                if len(dense):
                    points = dense
                    sample_stride = 1
        else:
            points = np.empty((0, 3), dtype=np.float64)
        if self._shadow_maximum_raw_depth_m is not None:
            # Select existing raw coordinates after the original sampling and
            # fallback decision. An empty filtered lattice must stay empty.
            points = points[points[:, 2] <= self._shadow_maximum_raw_depth_m]
        # Diagnostic only: compute from exactly the owner's existing samples.
        # Keep local until native-history validation and map update commit.
        physical_bound_evidence = None
        if self._physical_bounds is not None:
            physical_bound_evidence = dict(
                bound_witness(points, self._physical_bounds.maximum_diameter_m),
                prior=self._physical_bounds.model_dump(mode="json"),
                prior_profile_sha256=self._prior_profile_sha256,
                stream_key=list(frame.stream_key), input_evidence_hash=frame.evidence_hash,
                color_timestamp_ns=str(frame.color_timestamp_ns),
                depth_timestamp_ns=str(frame.depth_timestamp_ns),
                output_index=index, backend_frame_id=identity,
                sample_scope="existing_surface_backprojected_masked_depth_points",
                pixel_stride=sample_stride, backprojection_performed=backprojection_performed,
                unavailable_reason=(None if len(points) else
                    "no_valid_sampled_depth" if backprojection_performed else
                    "native_frame_not_retained_backprojection_skipped"),
                semantics="conditional_configured_prior_diagnostic_not_acceptance_gate")
        measured_task_features = self._normalize_measured_task_features(
            frame, candidate
        )
        current = dict(points=points, pose=None, metadata=dict(backend_frame_id=identity,
            color_timestamp_ns=str(frame.color_timestamp_ns), depth_timestamp_ns=str(frame.depth_timestamp_ns),
            input_evidence_hash=frame.evidence_hash))
        if self._shadow_feature_history:
            # Keep the exact normalized producer output with its existing
            # observation owner. Native retirement removes both together.
            current['measured_task_features'] = deepcopy(measured_task_features)
            import json
            current['feature_history_bytes'] = len(json.dumps(measured_task_features, allow_nan=False).encode('utf-8'))
        if self._shadow_feature_continuity is not None:
            from evidencescene.primitives.feature_continuity_shadow import advance_checkpoint
            previous = self._records.get(f'{index-1:06d}')
            checkpoint = previous.get('feature_continuity_checkpoint') if previous is not None else None
            camera_pose = candidate.get('camera_from_object')
            fixed_pose = (self._shadow_feature_continuity['table_from_camera'] @ np.asarray(camera_pose)).tolist() if camera_pose is not None else None
            diagnostic, checkpoint = advance_checkpoint(checkpoint,index,dict(
                inputs={'chest':dict(depth_ns=frame.depth_timestamp_ns,input_evidence_hash=frame.evidence_hash)},
                object_local_task_features=measured_task_features,table_from_initial_object=fixed_pose),
                self._shadow_feature_continuity['limits'])
            diagnostic.update(publication_authorized=False,feature_admission_authorized=False,
                semantics='shadow_diagnostic_only_no_measurement_or_geometry_substitution')
            current['feature_continuity_checkpoint'] = checkpoint
            current['feature_continuity_diagnostic'] = diagnostic
            current['feature_history_bytes'] += len(json.dumps([checkpoint,diagnostic],allow_nan=False).encode('utf-8'))
            if self._shadow_feature_admission_gate and diagnostic['status'] == 'rejected_shadow':
                # Retained raw evidence above is unchanged. Only this shadow
                # owner's current feature output is withheld; no stale geometry.
                measured_task_features = deepcopy(measured_task_features)
                for feature in measured_task_features:
                    if feature.get('feature_id') == 'pour_outlet':
                        feature.update(status='blocked',feature_geometry=None)
                        feature['validation_blockers'] = list(feature.get('validation_blockers',[])) + [
                            'task_feature_shadow_continuity_rejected']
        if self._shadow_visibility_margin_m is not None:
            current.update(depth_mm=frame.depth_mm, intrinsics=frame.intrinsics,
                           stream_key=frame.stream_key)
        available = {**self._records, identity: current}
        records, updates = {}, {}
        revised = 0
        for row in history["observations"]:
            key = row.get("backend_frame_id")
            if key not in available or key in records:
                raise ValueError("surface history has an unknown or duplicate observation")
            old = available[key]
            if any(row.get(k) != v for k, v in old["metadata"].items()):
                raise ValueError("surface historical evidence identity mismatch")
            pose = None
            if row.get("status") == "candidate":
                pose = _validate_rigid_transform(row.get("camera_from_object"), name="historical camera_from_object").copy()
            elif row.get("status") != "not_observable" or row.get("camera_from_object") is not None:
                raise ValueError("surface historical pose status invalid")
            if key == identity:
                if row.get("status") != candidate.get("status"):
                    raise ValueError("current pose status disagrees with native history")
                if candidate.get("status") == "candidate":
                    if pose is None or not np.array_equal(pose, candidate.get("camera_from_object")):
                        raise ValueError("current pose disagrees with native history")
            unchanged = (pose is None and old["pose"] is None or
                         pose is not None and old["pose"] is not None and np.array_equal(pose, old["pose"]))
            if self._shadow_frozen_pose_history and key != identity and not unchanged:
                raise ValueError('non-revising shadow owner changed retained pose history')
            if key == identity or not unchanged:
                updates[key] = (str(frame.stream_key[2]),
                    _transform_points(old["points"], _inverse_rigid_pose(pose))
                    if pose is not None else np.empty((0, 3)))
                revised += int(key != identity)
            records[key] = dict(old, pose=pose,
                               pose_version=old.get("pose_version", 0) + int(not unchanged))
        if candidate.get("status") == "candidate" and identity not in records:
            raise ValueError("current candidate is absent from native history")
        retired = history["retired_backend_frame_ids"]
        if (not isinstance(retired, list) or len(retired) != len(set(retired))
                or set(retired) != set(self._records) - set(records)):
            raise ValueError("surface native retirement identity mismatch")
        camera_bytes = sum(r["points"].nbytes for r in records.values())
        archive = self._archived_feature_measurements
        if self._shadow_feature_archive:
            archive = dict(archive)
            if candidate.get('status') == 'candidate':
                for feature in measured_task_features:
                    # The diagnostic is scoped to pour_outlet, not the object
                    # pose or other features. Preserve raw observation history.
                    if (self._shadow_archive_continuity_gate and feature.get('feature_id') == 'pour_outlet'
                            and current['feature_continuity_diagnostic']['status'] == 'rejected_shadow'):
                        continue
                    if feature.get('status') == 'available_current' and feature.get('feature_geometry') is not None:
                        evidence = dict(
                            observation=deepcopy(current['metadata']),
                            recorded_measurement=deepcopy(feature))
                        previous_evidence = archive.get(feature['feature_id'])
                        archive[feature['feature_id']] = dict(evidence,
                            initial_recorded_evidence=deepcopy(previous_evidence['initial_recorded_evidence']
                                if previous_evidence is not None else evidence))
            if len(archive) > 32:
                raise ValueError('shadow feature archive capacity exceeded')
            camera_bytes += len(json.dumps(archive, allow_nan=False).encode('utf-8'))
        if self._shadow_feature_history:
            camera_bytes += sum(r.get('feature_history_bytes', 0) for r in records.values())
        if self._shadow_visibility_margin_m is not None:
            camera_bytes += sum(r["depth_mm"].nbytes + r["intrinsics"].nbytes
                                for r in records.values())
        if camera_bytes > self.maximum_live_bytes:
            raise ValueError("surface camera evidence memory budget exceeded")
        visibility = None
        if self._shadow_visibility_margin_m is not None:
            # Check dependencies for every retained contribution. Cache each
            # pair only while both native pose versions remain unchanged.
            # New dictionaries keep cache preparation transactional with map.
            visibility = dict(mode="offline_shadow", publication_authorized=False,
                retained_camera_evidence_bytes=camera_bytes, point_reference_comparisons=0,
                reused_point_reference_comparisons=0,
                excluded_point_count=0, margin_m=self._shadow_visibility_margin_m,
                reference_scope=('all_already_arrived_retained_native_observations_same_stream'
                    if self._shadow_bidirectional_visibility else
                    'older_currently_retained_native_observations_same_stream'),
                retired_reference_policy="removed; retired_surface_contributions_keep_last_state")
            for key, record in records.items():
                if record["pose"] is None:
                    record.pop("visibility_votes", None)
                    record.pop("visibility_excluded", None)
                    record.pop("visibility_pose_version", None)
                    updates[key] = (str(frame.stream_key[2]), np.empty((0, 3)))
                    continue
                object_points = _transform_points(record["points"], _inverse_rigid_pose(record["pose"]))
                excluded = np.zeros(len(object_points), dtype=bool)
                old_votes = (record.get("visibility_votes", {})
                             if record.get("visibility_pose_version") == record["pose_version"] else {})
                votes = {}
                for reference_key, reference in records.items():
                    if (reference_key == key or reference["pose"] is None
                            or reference["stream_key"] != record["stream_key"]
                            or (not self._shadow_bidirectional_visibility and int(reference["metadata"]["depth_timestamp_ns"])
                            >= int(record["metadata"]["depth_timestamp_ns"]))):
                        continue
                    cached = old_votes.get(reference_key)
                    if cached is not None and cached[0] == reference["pose_version"]:
                        conflict = cached[1]
                        visibility["reused_point_reference_comparisons"] += len(object_points)
                    else:
                        known, margin = measured_free_space_margin(
                            object_points, reference["depth_mm"],
                            reference["intrinsics"], reference["pose"],
                            minimum_depth_m=.15, maximum_depth_m=6., depth_scale_m=.001)
                        conflict = known & (margin > self._shadow_visibility_margin_m)
                        visibility["point_reference_comparisons"] += len(object_points)
                    votes[reference_key] = (reference["pose_version"], conflict)
                    excluded |= conflict
                visibility["excluded_point_count"] += int(excluded.sum())
                old_excluded = record.get("visibility_excluded")
                if (key in updates or old_excluded is None
                        or not np.array_equal(excluded, old_excluded)):
                    updates[key] = (str(frame.stream_key[2]), object_points[~excluded])
                record.update(visibility_votes=votes, visibility_excluded=excluded,
                              visibility_pose_version=record["pose_version"])
            cache_bytes = sum(
                r.get("visibility_excluded", np.empty(0, dtype=bool)).nbytes
                + sum(v[1].nbytes for v in r.get("visibility_votes", {}).values())
                for r in records.values())
            if camera_bytes + cache_bytes > self.maximum_live_bytes:
                raise ValueError("surface visibility evidence memory budget exceeded")
            visibility["visibility_cache_bytes"] = cache_bytes
            visibility["retained_camera_evidence_bytes"] = camera_bytes + cache_bytes
        supplemental_keys = {key: value for key, value in self._supplemental_keys.items()
                             if key not in retired}
        supplemental_retired = [self._supplemental_keys[key] for key in retired
                                if key in self._supplemental_keys]
        if supplemental is not None:
            key = ('supplemental', identity, tuple(supplemental['source_stream_key']))
            updates[key] = (str(supplemental['source_stream_key'][2]),
                            supplemental['points_object_m'])
            supplemental_keys[identity] = key
        self._map.update(updates, retire=[*retired, *supplemental_retired])
        self._supplemental_keys = supplemental_keys
        if supplemental is not None:
            self._supplemental_latest = (supplemental_frame.stream_key,
                supplemental_frame.color_timestamp_ns, supplemental_frame.depth_timestamp_ns)
        retired_correction = None
        if self._shadow_retired_correction:
            current_pose = records.get(identity, {}).get("pose")
            votes = {}
            mask_conflict_count=0
            supplemental_conflict_count=0
            if candidate.get("status") == "candidate" and current_pose is not None:
                existing_points, _, _ = self._map.snapshot(
                    minimum_observation_support=DEFAULT_MINIMUM_FRAME_SUPPORT,
                    maximum_point_count=self._map.maximum_voxels)
                evidence = measured_free_space_margin(existing_points, frame.depth_mm,
                    frame.intrinsics, current_pose, depth_scale_m=.001,
                    minimum_depth_m=.15, maximum_depth_m=6.,
                    **({'return_projection_details':True} if self._shadow_retired_mask_conflict else {}))
                known,margin=evidence[:2]
                conflict=known & (margin > .03)
                if self._shadow_retired_mask_conflict:
                    from evidencescene.surface.object_surface_fusion import conditional_mask_depth_conflict
                    mask_conflict=conditional_mask_depth_conflict(evidence[2],frame.mask)
                    mask_conflict_count=int(mask_conflict.sum())
                    conflict|=mask_conflict
                if self._shadow_retired_supplemental_conflict and supplemental is not None:
                    from evidencescene.surface.object_surface_fusion import conditional_mask_depth_conflict
                    secondary_from_object = np.linalg.inv(np.asarray(
                        supplemental_bridge['primary_from_secondary'], dtype=np.float64)) @ current_pose
                    sknown, smargin, details = measured_free_space_margin(
                        existing_points, supplemental_frame.depth_mm,
                        supplemental_frame.intrinsics, secondary_from_object,
                        depth_scale_m=.001, minimum_depth_m=.15, maximum_depth_m=6.,
                        return_projection_details=True)
                    secondary_conflict = conditional_mask_depth_conflict(details, supplemental_frame.mask)
                    if not self._shadow_retired_supplemental_mask_only:
                        secondary_conflict |= sknown & (smargin > .03)
                    if self._shadow_retired_supplemental_exclusive:
                        supplemental_camera = supplemental_frame.stream_key[2]
                        voxel_keys = np.floor(existing_points / DEFAULT_VOXEL_SIZE_M).astype(np.int64)
                        exclusive = np.array([set(self._map._totals[tuple(key)][2]) == {supplemental_camera}
                                              for key in voxel_keys], dtype=bool)
                        secondary_conflict &= exclusive
                    supplemental_conflict_count = int(secondary_conflict.sum())
                    conflict |= secondary_conflict
                keys = np.floor(existing_points / DEFAULT_VOXEL_SIZE_M).astype(np.int64)
                for key in keys[conflict]:
                    key = tuple(key)
                    votes[key] = min(3, self._shadow_retired_conflicts.get(key, 0) + 1)
            retired_correction = self._map.shadow_discard_retired_contributions(
                [key for key, count in votes.items() if count >= 3])
            retired_correction.update(consecutive_conflicts_required=3, margin_m=.03,
                conflict_voxels_this_update=len(votes),
                unknown_or_nonconflicting_policy="reset_consecutive_count")
            self._shadow_retired_conflicts = votes
            if self._shadow_retired_supplemental_conflict:
                retired_correction.update(supplemental_conflict_shadow=True,
                    supplemental_mask_only=self._shadow_retired_supplemental_mask_only,
                    supplemental_exclusive=self._shadow_retired_supplemental_exclusive,
                    supplemental_conflicts_this_update=supplemental_conflict_count,
                    conflict_is_material_truth=False)
            if self._shadow_retired_mask_conflict:
                retired_correction.update(conditional_mask_conflict_shadow=True,
                    conditional_mask_conflicts_this_update=mask_conflict_count,
                    conditional_mask_conflict_is_material_truth=False)
        self._records = records
        self._archived_feature_measurements = archive
        self._key = frame.stream_key
        self._last_color_ns, self._last_depth_ns = frame.color_timestamp_ns, frame.depth_timestamp_ns
        self._last_index = index
        integrated = (identity in records and records[identity]["pose"] is not None
                      and len(updates[identity][1]) > 0)
        if integrated:
            self._last_observed_depth_ns = frame.depth_timestamp_ns
        self._snapshot_metadata = dict(
            coordinate_frame="backend_tracking_gauge_not_semantic_body_frame",
            stream_key=list(frame.stream_key), as_of_depth_timestamp_ns=str(frame.depth_timestamp_ns),
            input_evidence_hash=frame.evidence_hash,
            latest_surface_observation_depth_timestamp_ns=(str(self._last_observed_depth_ns)
                if self._last_observed_depth_ns is not None else None),
            voxel_size_m=DEFAULT_VOXEL_SIZE_M, input_update_count=index + 1,
            pixel_stride=sample_stride,
            revised_observation_count=revised, retained_camera_observation_count=len(records),
            native_retirement_policy="retain_last_supported_surface_contribution_not_current_measurement",
            measured_task_features=deepcopy(measured_task_features),
            unseen_geometry_verified=False, world_pose_authorized=False, publication_authorized=False)
        source_mask_hash = candidate.get("source_mask_evidence_sha256")
        if self._shadow_feature_continuity is not None:
            self._snapshot_metadata['feature_continuity_diagnostic'] = deepcopy(current['feature_continuity_diagnostic'])
        if isinstance(source_mask_hash, str) and source_mask_hash:
            self._snapshot_metadata["source_mask_evidence_sha256"] = source_mask_hash
        if physical_bound_evidence is not None:
            self._snapshot_metadata["physical_bound_evidence"] = physical_bound_evidence
        if visibility is not None:
            self._snapshot_metadata["visibility_evidence"] = visibility
        if retired_correction is not None:
            self._snapshot_metadata["shadow_retired_correction"] = retired_correction
        if self._shadow_supplemental_observations:
            self._snapshot_metadata['supplemental_observation'] = (
                dict({key: value for key, value in supplemental.items() if key != 'points_object_m'},
                     scope='shadow_surface_contribution_not_authorized')
                if supplemental is not None else None)
            self._snapshot_metadata['supplemental_retained_observation_count'] = len(self._supplemental_keys)
        return dict(
            self.snapshot(maximum_point_count=WIRE_SURFACE_POINT_COUNT),
            current_observation_integrated=integrated,
            read_mode="advanced",
        )

    def snapshot(self, *, maximum_point_count=DEFAULT_MAXIMUM_SURFACE_POINT_COUNT):
        """Read existing geometry without advancing any observation timestamp."""
        if self._closed:
            raise RuntimeError("object surface closed")
        xyz, support, cameras = self._map.snapshot(minimum_observation_support=DEFAULT_MINIMUM_FRAME_SUPPORT,
            maximum_point_count=maximum_point_count)
        return self._snapshot_from_arrays(xyz, support, cameras)

    def shadow_selected_snapshot(self, *, maximum_point_count=WIRE_SURFACE_POINT_COUNT):
        """Read-only existing-point selection; never registered on a live wire.

        The selector and its thresholds are the existing offline development
        policy. Return its diagnostic separately from the original payload.
        No full-map Python point lists are created before selecting indices.
        """
        if self._closed:
            raise RuntimeError("object surface closed")
        from evidencescene.surface.shadow_surface_snapshot_selection import select_existing_points_shadow
        xyz, support, cameras, keys = self._map.snapshot(
            minimum_observation_support=DEFAULT_MINIMUM_FRAME_SUPPORT,
            maximum_point_count=self._map.maximum_voxels, include_voxel_keys=True)
        selection = select_existing_points_shadow(
            xyz, support, cameras, keys, budget=maximum_point_count)
        if selection['wire_authorized'] is not False:
            raise RuntimeError('shadow selection cannot authorize publication')
        indices = selection['indices']
        # All supported totals fit the map's own voxel budget. Reuse that
        # population count rather than walking the same totals a second time.
        payload = self._snapshot_from_arrays(
            xyz[indices], support[indices], cameras[indices], point_count=len(xyz))
        return payload, selection

    def _snapshot_from_arrays(self, xyz, support, cameras, *, point_count=None):
        """One original serializer for both default and shadow array readers."""
        from copy import deepcopy
        if point_count is None:
            point_count = self._map.supported_point_count(
                minimum_observation_support=DEFAULT_MINIMUM_FRAME_SUPPORT)
        return dict(deepcopy(self._snapshot_metadata),
            gauge_id=self._gauge_id,
            status="observed_surface_candidate" if len(xyz) else "insufficient_surface_support",
            points_object_m=xyz.tolist(), observation_support=support.tolist(), camera_support=cameras.tolist(),
            point_count=point_count,
            current_observation_integrated=False, read_mode="read_only",
            world_pose_authorized=False, publication_authorized=False)

    def shadow_feature_history(self):
        """Read exact retained owner evidence; never interpret it as current."""
        if self._closed:
            raise RuntimeError('object surface closed')
        if not self._shadow_feature_history:
            raise RuntimeError('shadow feature history disabled')
        return dict(publication_authorized=False,world_pose_authorized=False,
            gauge_id=self._gauge_id,read_mode='read_only',
            scope='retained_native_observations_not_complete_trajectory_or_current_evidence',
            observations=[dict(deepcopy(record['metadata']),
                camera_from_object=record['pose'].tolist() if record['pose'] is not None else None,
                **({key:deepcopy(record[key]) for key in ('feature_continuity_checkpoint','feature_continuity_diagnostic')}
                   if 'feature_continuity_checkpoint' in record else {}),
                measured_task_features=deepcopy(record.get('measured_task_features', [])))
                for record in self._records.values()])

    def shadow_feature_archive(self):
        """Historical source evidence only, never a current or trusted landmark."""
        if self._closed:
            raise RuntimeError('object surface closed')
        if not self._shadow_feature_archive:
            raise RuntimeError('shadow feature archive disabled')
        return dict(publication_authorized=False, world_pose_authorized=False,
            current_measurement=False, landmark_identity_verified=False,
            status='historical_measurement_evidence', read_mode='read_only',
            gauge_id=self._gauge_id,
            observations=[dict(deepcopy(record),
                age_s=(self._last_depth_ns-int(record['observation']['depth_timestamp_ns']))/1e9)
                for record in self._archived_feature_measurements.values()])

    def shadow_archived_point_projection(self, feature_id, *, maximum_age_s):
        """Read-only historical point transformed by this owner's latest pose.

        Not a current measurement or validated landmark. Never included in the
        normal snapshot, task facts or world wire; no external pose is accepted.
        """
        archive = self.shadow_feature_archive()  # enforces opt-in and lifecycle
        if isinstance(maximum_age_s, bool) or not isinstance(maximum_age_s, (int, float)) or not np.isfinite(maximum_age_s) or maximum_age_s <= 0:
            raise ValueError('positive finite explicit archive age limit required')
        result = dict(publication_authorized=False, world_pose_authorized=False,
            current_measurement=False, landmark_identity_verified=False,
            read_mode='read_only', status='unavailable', feature_id=feature_id,
            gauge_id=self._gauge_id, projection_depth_timestamp_ns=None if self._last_depth_ns is None else str(self._last_depth_ns),
            maximum_age_s=float(maximum_age_s), projected_point_camera_m=None)
        evidence = next((entry for entry in archive['observations']
                         if entry['recorded_measurement']['feature_id'] == feature_id), None)
        if evidence is None:
            return dict(result, reason='historical_feature_unavailable')
        feature = evidence['recorded_measurement']
        result.update(original_measurement=deepcopy(feature), age_s=evidence['age_s'])
        if evidence['age_s'] < 0 or evidence['age_s'] > maximum_age_s:
            return dict(result, reason='historical_feature_expired')
        if feature.get('gauge_id') != self._gauge_id or feature.get('stream_key') != list(self._key):
            return dict(result, reason='historical_feature_identity_mismatch')
        current = [record for record in self._records.values()
                   if int(record['metadata']['depth_timestamp_ns']) == self._last_depth_ns]
        if len(current) != 1 or current[0]['pose'] is None:
            return dict(result, reason='current_owner_pose_unavailable')
        geometry = feature.get('feature_geometry') or {}
        if geometry.get('schema') != 'ticos.measured-point-feature.v1':
            return dict(result, reason='historical_point_geometry_required')
        transformed = _transform_feature_geometry(geometry, current[0]['pose'])
        if transformed is None:
            return dict(result, reason='historical_point_geometry_invalid')
        return dict(result, status='historical_point_projection', reason=None,
            projected_point_camera_m=transformed['point_m'],
            projection_pose_observation=deepcopy(current[0]['metadata']),
            coordinate_scope='existing_surface_input_camera_not_world')

    def close(self):
        self._closed = True
        self._shadow_retired_conflicts.clear()
        self._records.clear()
        self._supplemental_keys.clear()
        self._supplemental_latest = None
        self._archived_feature_measurements.clear()
        self._map.close()
        self._snapshot_metadata.clear()
