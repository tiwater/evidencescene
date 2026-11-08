"""Causal factor-graph observability (F, G, R, M) and shadow pose-owner evaluation."""

from __future__ import annotations

from evidencescene.observability import (
    factor_graph_observability,
    native_history_trace,
    shadow_missing_mask_tick,
    shadow_pose_adapter,
    shadow_pose_owner_ab,
    shadow_pose_owner_ab_registered_sequence,
    shadow_registered_input,
    shadow_score_tick,
    shadow_visual_rejected_tick,
)
from evidencescene.observability.factor_graph_observability import (
    BirthFactorCommit,
    BirthObservationRecord,
    CausalAttributionVerdict,
    FeatureBirthLedger,
    PriorConnectivityReport,
    SchurObservabilityReport,
    build_se3_landmark_jacobians,
    evaluate_prior_connected_admission,
    evaluate_schur_observability_gate,
    verify_causal_pose_attribution,
)
from evidencescene.observability.shadow_missing_mask_tick import (
    make_missing_mask_shadow_tick,
)
from evidencescene.observability.shadow_pose_adapter import (
    adapt_shadow_pose,
    adapt_shadow_poses,
)
from evidencescene.observability.shadow_registered_input import (
    prepare_shadow_registered_inputs,
)
from evidencescene.observability.shadow_score_tick import (
    make_shadow_score_tick,
)
from evidencescene.observability.shadow_visual_rejected_tick import (
    make_visual_rejected_shadow_tick,
)

__all__ = [
    "BirthFactorCommit",
    "BirthObservationRecord",
    "CausalAttributionVerdict",
    "FeatureBirthLedger",
    "PriorConnectivityReport",
    "SchurObservabilityReport",
    "adapt_shadow_pose",
    "adapt_shadow_poses",
    "build_se3_landmark_jacobians",
    "evaluate_prior_connected_admission",
    "evaluate_schur_observability_gate",
    "factor_graph_observability",
    "make_missing_mask_shadow_tick",
    "make_shadow_score_tick",
    "make_visual_rejected_shadow_tick",
    "native_history_trace",
    "prepare_shadow_registered_inputs",
    "shadow_missing_mask_tick",
    "shadow_pose_adapter",
    "shadow_pose_owner_ab",
    "shadow_pose_owner_ab_registered_sequence",
    "shadow_registered_input",
    "shadow_score_tick",
    "shadow_visual_rejected_tick",
    "verify_causal_pose_attribution",
]
