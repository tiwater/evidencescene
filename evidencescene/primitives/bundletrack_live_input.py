"""Bind current registered RGB-D and a selected dense mask to tracker input.

No detection, association, pose estimation or world promotion happens here.
The caller must supply the mask selected for this exact observation and object.
"""
from dataclasses import dataclass, field
import base64
import hashlib
from io import BytesIO
import json
import math
import threading
from threading import Thread

import cv2
import numpy as np
from PIL import Image

from typing import Any

from evidencescene.primitives.registered_rgbd import rectify_registered_rgbd
from evidencescene.primitives.geometry import (
    _rotation_matrix,
    RegisteredRgbdFrame,
)
CameraObservation = Any


@dataclass(frozen=True)
class PreparedFeatureSourceBinding:
    observation_hash: str
    feature_stream_key: tuple
    source_mask_hash: str
    registered_mask_hash: str
    registered_stream_key: tuple
    color_timestamp_ns: int
    depth_timestamp_ns: int


@dataclass(frozen=True)
class RegisteredBundleTrackInput:
    stream_key: tuple[object, ...]
    color_timestamp_ns: int
    depth_timestamp_ns: int
    color_rgb: np.ndarray
    depth_mm: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray
    source_evidence_hash: str
    _evidence_hash: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_hash_thread: Thread | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_hash_error: BaseException | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evidence_hash_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )
    _take_ownership: bool = field(default=False, repr=False, compare=False)
    # Sensor-grid color and anchor. SAM tracks these when the color camera
    # has distortion; BundleTrack still receives the rectified arrays above.
    # Omitted from the evidence digest so the native input hash stays the
    # rectified RGB-D contract.
    sam_color_rgb: np.ndarray | None = field(default=None, compare=False, repr=False)
    sam_anchor_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    sensor_calibration: tuple | None = field(default=None, compare=False, repr=False)
    color_frame_id: str | None = field(default=None, compare=False, repr=False)
    depth_frame_id: str | None = field(default=None, compare=False, repr=False)
    color_from_depth: np.ndarray | None = field(default=None, compare=False, repr=False)
    feature_source_binding: PreparedFeatureSourceBinding | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        for name in ("color_rgb", "depth_mm", "mask", "intrinsics"):
            value = getattr(self, name)
            # Rectification allocates these arrays specifically for this
            # immutable input. Freeze such owned contiguous buffers in place
            # instead of copying every pixel into a second bytes-backed array.
            # Public/direct constructors keep the historical defensive copy.
            if (
                self._take_ownership
                and isinstance(value, np.ndarray)
                and value.flags.owndata
                and value.flags.c_contiguous
            ):
                value.setflags(write=False)
                array = value
            else:
                array = _immutable(value)
            object.__setattr__(self, name, array)
        if self.sam_color_rgb is not None:
            object.__setattr__(
                self,
                "sam_color_rgb",
                np.array(self.sam_color_rgb, dtype=np.uint8, copy=True, order="C"),
            )
        if self.sam_anchor_mask is not None:
            object.__setattr__(
                self,
                "sam_anchor_mask",
                np.array(self.sam_anchor_mask, dtype=bool, copy=True, order="C"),
            )
        if self.color_from_depth is not None:
            transform = np.asarray(self.color_from_depth, dtype=np.float64)
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0)
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3],
                    np.eye(3),
                    atol=1e-5,
                    rtol=0.0,
                )
                or not np.isclose(float(np.linalg.det(transform[:3, :3])), 1.0, atol=1e-5, rtol=0.0)
            ):
                raise ValueError("color_from_depth must be a finite rigid transform")
            transform = np.array(transform, dtype=np.float64, copy=True, order="C")
            transform.setflags(write=False)
            object.__setattr__(self, "color_from_depth", transform)

    def _compute_evidence_hash(self) -> str:
        """Compute the unchanged immutable-input digest.

        The digest is serialization/provenance only.  Keeping it out of the
        constructor allows the owner to overlap this CPU scan with the
        existing SAM2 or native BundleTrack call; the value and byte order are
        intentionally identical to the historical eager implementation.
        """
        digest = hashlib.sha256(
            json.dumps(
                dict(
                    source=self.source_evidence_hash,
                    stream_key=self.stream_key,
                    color_timestamp_ns=self.color_timestamp_ns,
                    depth_timestamp_ns=self.depth_timestamp_ns,
                ),
                sort_keys=True,
            ).encode()
        )
        for name in ("color_rgb", "depth_mm", "mask", "intrinsics"):
            array = getattr(self, name)
            digest.update(json.dumps((array.dtype.str, array.shape)).encode())
            _update_array_digest(digest, array)
        return digest.hexdigest()

    def _run_evidence_hash(self) -> None:
        try:
            value = self._compute_evidence_hash()
        except BaseException as error:  # pragma: no cover - surfaced by property
            with self._evidence_hash_lock:
                object.__setattr__(self, "_evidence_hash_error", error)
            return
        with self._evidence_hash_lock:
            object.__setattr__(self, "_evidence_hash", value)

    def prefetch_evidence_hash(self) -> None:
        """Start the immutable digest without blocking the estimator owner."""
        with self._evidence_hash_lock:
            if (
                self._evidence_hash is not None
                or self._evidence_hash_thread is not None
            ):
                return
            worker = Thread(
                target=self._run_evidence_hash,
                name="ticos-edge-evidence-hash",
                daemon=True,
            )
            object.__setattr__(self, "_evidence_hash_thread", worker)
        worker.start()

    @property
    def evidence_hash(self) -> str:
        """Return the same digest as before, joining only if still running."""
        with self._evidence_hash_lock:
            worker = self._evidence_hash_thread
            value = self._evidence_hash
            error = self._evidence_hash_error
        if worker is None and value is None and error is None:
            self._run_evidence_hash()
        elif worker is not None:
            worker.join()
        with self._evidence_hash_lock:
            value = self._evidence_hash
            error = self._evidence_hash_error
        if error is not None:
            raise RuntimeError("registered input evidence hash failed") from error
        if value is None:
            raise RuntimeError("registered input evidence hash did not complete")
        return value


def _immutable(value: np.ndarray) -> np.ndarray:
    return np.ndarray(value.shape, dtype=value.dtype,
        buffer=np.ascontiguousarray(value).tobytes())


def _update_array_digest(digest, value: np.ndarray) -> None:
    """Hash an owned immutable array without a second bytes allocation."""
    digest.update(memoryview(value).cast("B"))


def prepare_bundletrack_input(observation: CameraObservation, *,
                             registered: RegisteredRgbdFrame,
                             selected_mask: np.ndarray,
                             object_id: str,
                             color_rgb: np.ndarray | None = None,
                             max_sync_skew_ms: float = 20.0) -> RegisteredBundleTrackInput:
    """Use the original rectification, without a sequence manifest or file I/O.

    The hash binds supplied evidence, not proof of object association. Both
    that association and registration authorization remain upstream contracts.
    """
    return _prepare_registered_input(observation, registered=registered,
        selected_mask=selected_mask, object_id=object_id, color_rgb=color_rgb,
        require_mask_support=True,
        max_sync_skew_ms=max_sync_skew_ms)


def prepare_visual_anchor(observation: CameraObservation, *, registered: RegisteredRgbdFrame,
                          selected_mask: np.ndarray, object_id: str,
                          color_rgb: np.ndarray | None = None,
                          max_sync_skew_ms: float = 20.0) -> RegisteredBundleTrackInput:
    """Bind an explicit sensor mask to SAM memory without authorizing metric pose."""
    if not isinstance(selected_mask, np.ndarray) or not np.any(selected_mask):
        raise ValueError("nonempty explicit visual anchor required")
    return _prepare_registered_input(observation, registered=registered,
        selected_mask=selected_mask, object_id=object_id, color_rgb=color_rgb,
        require_mask_support=False, max_sync_skew_ms=max_sync_skew_ms)


def prepare_tracking_observation(observation: CameraObservation, *,
                                 registered: RegisteredRgbdFrame,
                                 object_id: str,
                                 color_rgb: np.ndarray | None = None,
                                 max_sync_skew_ms: float = 20.0) -> RegisteredBundleTrackInput:
    """Rectify a new observation before SAM produces its current mask.

    The empty mask is explicit, never a full-frame/bbox fallback. This input
    cannot pass the tracking session's final mask-depth gate by itself.
    """
    return _prepare_registered_input(observation, registered=registered,
        selected_mask=np.zeros((observation.color_height, observation.color_width), dtype=bool),
        color_rgb=color_rgb,
        object_id=object_id, require_mask_support=False,
        max_sync_skew_ms=max_sync_skew_ms)


def _prepare_registered_input(observation, *, registered, selected_mask, object_id,
                              color_rgb, require_mask_support, max_sync_skew_ms):
    stream = observation.perception_stream
    color_info = observation.color_camera_info
    depth_info = observation.depth_camera_info
    registration = observation.rgbd_registration
    if stream is None or not observation.camera_id or not isinstance(object_id, str) or not object_id.strip():
        raise ValueError("explicit stream, camera and selected object identity required")
    if (registration.status != "validated" or registration.color_from_depth is None
            or color_info is None or depth_info is None or not registered.ready):
        raise ValueError("current validated RGB-D registration required")
    color_ns, depth_ns = observation.color_timestamp_ns, observation.depth_timestamp_ns
    try:
        maximum_skew_ns = int(float(max_sync_skew_ms) * 1_000_000.0)
    except (TypeError, ValueError, OverflowError):
        maximum_skew_ns = 0
    if (not math.isfinite(float(max_sync_skew_ms)) or float(max_sync_skew_ms) <= 0.0
            or maximum_skew_ns <= 0):
        raise ValueError("max_sync_skew_ms must be finite and positive")
    if (type(color_ns) is not int or type(depth_ns) is not int
            or min(color_ns, depth_ns) <= 0 or abs(color_ns - depth_ns) > maximum_skew_ns):
        raise ValueError(
            f"tracking requires positive sensor timestamps within {float(max_sync_skew_ms):g} ms"
        )
    if (registered.color_timestamp_ns != color_ns or registered.depth_timestamp_ns != depth_ns
            or registered.color_frame_id != color_info.frame_id
            or registered.depth_frame_id != depth_info.frame_id):
        raise ValueError("registered frame identity does not match observation")
    color = _decode_color_rgb(observation.color_image_base64) if color_rgb is None else np.asarray(color_rgb)
    shape = (observation.color_height, observation.color_width)
    if (color.shape[:2] != shape or shape != (color_info.height, color_info.width)
            or registered.color_image_size != (shape[1], shape[0])
            or not isinstance(selected_mask, np.ndarray) or selected_mask.dtype != np.bool_
            or selected_mask.shape != shape):
        raise ValueError("current dense mask and color camera grid must match")
    rgb, depth, mask, intrinsics, diagnostics = rectify_registered_rgbd(
        color=color, color_mask=selected_mask, color_info=color_info, registered=registered)
    # Preserve the existing exporter support gate, not an arbitrary relaxed
    # live threshold. The tracker may subsequently reject the candidate too.
    if require_mask_support and diagnostics["mask_depth_supported_pixel_count"] < 40:
        raise ValueError("insufficient registered mask depth support")
    calibration = dict(
        registration=registration.model_dump(mode="json"),
        color_camera_info=color_info.model_dump(mode="json"),
        depth_camera_info=depth_info.model_dump(mode="json"))
    calibration_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    key = (stream.stream_id, stream.epoch, observation.camera_id,
           observation.camera_topic, color_info.frame_id, depth_info.frame_id, object_id, calibration_hash)
    metadata = dict(stream_key=key, color_timestamp_ns=color_ns, depth_timestamp_ns=depth_ns,
                    calibration=calibration)
    # The immutable input object hashes the rectified RGB-D/mask/intrinsics
    # arrays again as its final evidence hash. Keep this source digest focused
    # on the original transport pixels and selected mask so each array is
    # scanned once while the final hash still binds every input to native code.
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for array in (color, selected_mask):
        _update_array_digest(digest, np.ascontiguousarray(array))
    distortion = tuple(float(value) for value in (getattr(color_info, "d", ()) or ()))
    sam_color = None
    sam_anchor = None
    sensor_calibration = None
    # Undistortion is a few pixels, but it is enough to make SAM2's object
    # score collapse on the exposure step (paired 307) while the same pixels
    # in the original JPEG keep the carton. Track the sensor image, then warp
    # the mask onto this rectified grid for depth and BundleTrack.
    if any(abs(value) > 1e-12 for value in distortion):
        sam_color = color
        if np.any(selected_mask):
            sam_anchor = selected_mask
        sensor_calibration = (
            int(color_info.width),
            int(color_info.height),
            float(color_info.fx),
            float(color_info.fy),
            float(color_info.cx),
            float(color_info.cy),
            distortion,
        )
    from evidencescene.primitives.image_mask import encode_mask_rle
    from evidencescene.primitives.geometry import (
        mask_evidence_sha256,
        observation_digest,
        registered_feature_stream_key,
    )
    feature_key = registered_feature_stream_key(observation, object_id=object_id)
    feature_binding = PreparedFeatureSourceBinding(
        observation_digest(observation), tuple(feature_key),
        mask_evidence_sha256(encode_mask_rle(selected_mask)),
        hashlib.sha256(mask.tobytes()).hexdigest(), key, color_ns, depth_ns,
    ) if feature_key is not None else None
    return RegisteredBundleTrackInput(key, color_ns, depth_ns,
        rgb, depth, mask, intrinsics, digest.hexdigest(), _take_ownership=True,
        sam_color_rgb=sam_color, sam_anchor_mask=sam_anchor,
        sensor_calibration=sensor_calibration,
        color_frame_id=color_info.frame_id,
        depth_frame_id=depth_info.frame_id,
        color_from_depth=_registration_transform_matrix(registration.color_from_depth),
        feature_source_binding=feature_binding)


def _registration_transform_matrix(transform: object) -> np.ndarray:
    rotation = _rotation_matrix(transform)
    translation = transform.translation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (
        float(translation.x),
        float(translation.y),
        float(translation.z),
    )
    return matrix


def _decode_color_rgb(payload: str) -> np.ndarray:
    """Decode the transport image without changing RGB pixel evidence.

    OpenCV's JPEG/PNG decoder is materially faster than constructing a PIL
    image for the 1280x720 live frames.  Keep a PIL fallback for compressed
    formats OpenCV does not recognize and preserve the existing RGB contract.
    """
    raw = base64.b64decode(payload, validate=True)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    # OpenCV 4.13 can decode directly into RGB, avoiding a second full-frame
    # channel-swap copy. Keep the older BGR path for deployment images whose
    # OpenCV does not expose IMREAD_COLOR_RGB.
    rgb_flag = getattr(cv2, "IMREAD_COLOR_RGB", None)
    decoded = cv2.imdecode(
        encoded,
        rgb_flag if rgb_flag is not None else cv2.IMREAD_COLOR,
    )
    if decoded is not None:
        if rgb_flag is not None:
            return decoded
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

