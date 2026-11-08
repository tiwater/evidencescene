"""Standard evaluation metrics for 6D object pose tracking and 3D surface reconstruction."""

from __future__ import annotations

from evidencescene.metrics.pose_metrics import (
    PoseSequenceScorecard,
    TrajectoryDeterminismSpread,
    align_initial_gauge,
    compute_add,
    compute_adds,
    compute_failure_inclusive_recall_auc,
    compute_geodesic_rotation_error_deg,
    compute_trajectory_determinism_spread,
    compute_translation_error_m,
    evaluate_pose_sequence,
)
from evidencescene.metrics.surface_metrics import (
    SurfaceReconstructionScorecard,
    evaluate_surface_reconstruction,
    quantize_surface_grid,
)

__all__ = [
    "PoseSequenceScorecard",
    "SurfaceReconstructionScorecard",
    "TrajectoryDeterminismSpread",
    "align_initial_gauge",
    "compute_add",
    "compute_adds",
    "compute_failure_inclusive_recall_auc",
    "compute_geodesic_rotation_error_deg",
    "compute_trajectory_determinism_spread",
    "compute_translation_error_m",
    "evaluate_pose_sequence",
    "evaluate_surface_reconstruction",
    "pose_metrics",
    "quantize_surface_grid",
    "surface_metrics",
]
