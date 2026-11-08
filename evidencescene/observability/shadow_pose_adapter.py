"""Fail-closed relative-gauge shadow pose candidate adapter for offline scoring.

State owner: OriginalPersistentWorldRegistry and its existing pose owner remain
authoritative. This module neither runs a tracker nor publishes to world models,
realtime sinks, or teleop-assist wire state.

Relative-gauge trackers return camera-current-from-camera-initial transforms.
The caller must supply an already measured camera-initial-from-object gauge for
the same source and object; the adapter does not estimate or validate its
provenance. Its result alone is not a registered-input receipt, timing evidence,
or a passing shadow-pose-owner scorecard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _rigid(value: object) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        return None
    return matrix


def adapt_shadow_pose(
    *,
    camera_current_from_camera_initial: object,
    camera_initial_from_object: object,
    lost: object,
) -> dict[str, Any]:
    """Return a shadow-only candidate or a blocked result with no pose.

    ``lost`` must be an actual boolean from the corresponding object and frame;
    absent, ambiguous, or non-boolean status is blocked. An upstream matrix is
    never exposed when lost, even if it remains a valid homogeneous transform.
    """
    result: dict[str, Any] = {
        "status": "not_observable",
        "reason": None,
        "camera_from_object": None,
        "wire_authorized": False,
    }
    if type(lost) is not bool:
        result["reason"] = "shadow_lost_status_missing_or_invalid"
        return result
    if lost:
        result["reason"] = "shadow_tracker_lost"
        return result
    relative = _rigid(camera_current_from_camera_initial)
    initial = _rigid(camera_initial_from_object)
    if relative is None or initial is None:
        result["reason"] = "shadow_pose_or_initial_gauge_invalid"
        return result
    pose = _rigid(relative @ initial)
    if pose is None:
        result["reason"] = "shadow_composed_pose_invalid"
        return result
    result.update(status="candidate", reason=None, camera_from_object=pose.tolist())
    return result


def adapt_shadow_poses(
    *,
    object_keys: Sequence[str | int],
    camera_current_from_camera_initial: object,
    camera_initial_from_object: Mapping[str | int, object],
    lost: Sequence[object],
) -> list[dict[str, Any]]:
    """Adapt explicitly ordered object slots without identity inference.

    Exact key/gauge correspondence is required. This function cannot validate
    that the caller's keys are real object identities or that the evidence is
    current; it does not authorize wire output.
    """
    if not isinstance(object_keys, (list, tuple)) or not object_keys:
        raise ValueError("object keys must be a nonempty ordered list")
    if any(not ((type(key) is str and bool(key)) or type(key) is int) for key in object_keys):
        raise ValueError("object keys must be nonempty strings or integers")
    if len(set(object_keys)) != len(object_keys):
        raise ValueError("object keys must be unique")
    if (
        not isinstance(camera_initial_from_object, Mapping)
        or {(type(key), key) for key in camera_initial_from_object}
        != {(type(key), key) for key in object_keys}
    ):
        raise ValueError("initial gauges must exactly match object keys and types")
    if (
        not isinstance(camera_current_from_camera_initial, (list, tuple, np.ndarray))
        or (
            isinstance(camera_current_from_camera_initial, np.ndarray)
            and camera_current_from_camera_initial.ndim == 0
        )
        or len(camera_current_from_camera_initial) != len(object_keys)
    ):
        raise ValueError("pose slots must match object keys")
    if not isinstance(lost, (list, tuple)) or len(lost) != len(object_keys):
        raise ValueError("loss slots must match object keys")
    return [
        {
            "object_key": key,
            **adapt_shadow_pose(
                camera_current_from_camera_initial=pose,
                camera_initial_from_object=camera_initial_from_object[key],
                lost=flag,
            ),
        }
        for key, pose, flag in zip(object_keys, camera_current_from_camera_initial, lost)
    ]
