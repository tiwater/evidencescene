"""Causal object surface reconstruction and RevisableSurfelMap selectors."""

from __future__ import annotations

from evidencescene.surface import (
    causal_metric_point_map,
    causal_object_surface,
    causal_surface_track_memory,
    live_surface_fusion,
    object_surface_fusion,
    realtime_surface_projection,
    shadow_surface_snapshot_selection,
    shape_prior,
    supplemental_surface_observation,
    surface_corroboration,
)
from evidencescene.surface.causal_object_surface import CausalObjectSurface
from evidencescene.surface.live_surface_fusion import (
    ProjectionTarget,
    SurfaceEvidence,
    TableAlignedSurfaceProjection,
    project_table_aligned_surfaces,
)
from evidencescene.surface.object_surface_fusion import (
    RevisableSurfelMap,
    measured_free_space_margin,
    observation_voxel_centroids,
)
from evidencescene.surface.realtime_surface_projection import (
    FusedSurfaceProjection,
    NativeDepthProjectionShadow,
    ProjectedSurfaceSamples,
    SurfaceProjection,
    fuse_surface_projections,
    project_surface_samples_to_camera,
    project_surface_to_camera,
    project_surface_via_native_depth_shadow,
    surface_row_spans,
    validate_projected_surface_depth,
)
from evidencescene.surface.shadow_surface_snapshot_selection import (
    select_existing_points_shadow,
)
from evidencescene.surface.shape_prior import (
    PreparedSceneObjectPrior,
    Sim3AnchorResult,
    anchor_shape_prior_to_depth,
    build_prepared_scene_object_prior,
    fuse_prior_with_measured_surface,
    transfer_measured_rgb_to_prior,
)
from evidencescene.surface.supplemental_surface_observation import (
    prepare_packet,
)

__all__ = [
    "CausalObjectSurface",
    "FusedSurfaceProjection",
    "NativeDepthProjectionShadow",
    "PreparedSceneObjectPrior",
    "ProjectedSurfaceSamples",
    "ProjectionTarget",
    "RevisableSurfelMap",
    "Sim3AnchorResult",
    "SurfaceEvidence",
    "SurfaceProjection",
    "TableAlignedSurfaceProjection",
    "anchor_shape_prior_to_depth",
    "build_prepared_scene_object_prior",
    "causal_metric_point_map",
    "causal_object_surface",
    "causal_surface_track_memory",
    "fuse_prior_with_measured_surface",
    "fuse_surface_projections",
    "live_surface_fusion",
    "measured_free_space_margin",
    "object_surface_fusion",
    "observation_voxel_centroids",
    "prepare_packet",
    "project_surface_samples_to_camera",
    "project_surface_to_camera",
    "project_surface_via_native_depth_shadow",
    "project_table_aligned_surfaces",
    "realtime_surface_projection",
    "select_existing_points_shadow",
    "shadow_surface_snapshot_selection",
    "shape_prior",
    "supplemental_surface_observation",
    "surface_corroboration",
    "surface_row_spans",
    "transfer_measured_rgb_to_prior",
    "validate_projected_surface_depth",
]
