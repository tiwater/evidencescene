"""Core geometric, RLE mask, content-hashing, and registered RGB-D primitives."""

from __future__ import annotations

from evidencescene.primitives import (
    bundletrack_live_input,
    content_hash,
    feature_continuity_shadow,
    geometry,
    image_mask,
    mask_depth_pca_pose,
    object_bound_evidence,
    registered_rgbd,
    sam2_observation_session,
    shadow_pose_owner,
    task_profile,
)
from evidencescene.primitives.bundletrack_live_input import (
    PreparedFeatureSourceBinding,
    RegisteredBundleTrackInput,
    prepare_tracking_observation,
)
from evidencescene.primitives.content_hash import (
    canonical_json_bytes,
    content_sha256,
    finite_json_value,
)
from evidencescene.primitives.geometry import (
    CameraInfoRecord,
    MaskDepthComponentSelection,
    RegisteredRgbdFrame,
    RegisteredRgbdSurface,
)
from evidencescene.primitives.image_mask import (
    decode_mask_rle,
    encode_mask_rle,
    rle_area,
    rle_difference_runs,
    rle_intersection_area,
    rle_intersection_runs,
    rle_iou,
    rle_union_runs,
)
from evidencescene.primitives.mask_depth_pca_pose import (
    backproject_masked_depth_points,
    estimate_mask_depth_centroid_camera,
    recover_camera_from_object_with_prior,
    recover_camera_from_object_with_surface_prior,
    tick_is_mask_available,
)
from evidencescene.primitives.object_bound_evidence import bound_witness
from evidencescene.primitives.registered_rgbd import (
    rectify_registered_rgbd,
)
from evidencescene.primitives.sam2_observation_session import SAM2ObservationSession
from evidencescene.primitives.shadow_pose_owner import (
    ShadowPoseOwner,
    ShadowPoseTick,
    score_controlled_reanchor_policy,
    should_request_controlled_reanchor,
)

__all__ = [
    "CameraInfoRecord",
    "MaskDepthComponentSelection",
    "PreparedFeatureSourceBinding",
    "RegisteredBundleTrackInput",
    "RegisteredRgbdFrame",
    "RegisteredRgbdSurface",
    "SAM2ObservationSession",
    "ShadowPoseOwner",
    "ShadowPoseTick",
    "backproject_masked_depth_points",
    "bound_witness",
    "bundletrack_live_input",
    "canonical_json_bytes",
    "content_hash",
    "content_sha256",
    "decode_mask_rle",
    "encode_mask_rle",
    "estimate_mask_depth_centroid_camera",
    "feature_continuity_shadow",
    "finite_json_value",
    "geometry",
    "image_mask",
    "mask_depth_pca_pose",
    "object_bound_evidence",
    "prepare_tracking_observation",
    "recover_camera_from_object_with_prior",
    "recover_camera_from_object_with_surface_prior",
    "rectify_registered_rgbd",
    "registered_rgbd",
    "rle_area",
    "rle_difference_runs",
    "rle_intersection_area",
    "rle_intersection_runs",
    "rle_iou",
    "rle_union_runs",
    "sam2_observation_session",
    "score_controlled_reanchor_policy",
    "shadow_pose_owner",
    "should_request_controlled_reanchor",
    "task_profile",
    "tick_is_mask_available",
]
