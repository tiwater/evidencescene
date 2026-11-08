"""Zero-loss GPU/CPU overlap, O(runs) integer RLE mask arithmetic, and vectorized AprilTag RANSAC."""

from __future__ import annotations

from evidencescene.runtime import (
    coplanar_rgbd_tag_observations,
    request_evidence_deduplication,
    sam2_cpu_preparation,
    sam2_gpu_window_overlap,
)
from evidencescene.runtime.request_evidence_deduplication import (
    RequestDeduplicationStats,
    fast_fit_shared_plane,
    install_request_evidence_deduplication,
    request_evidence_deduplication_scope,
    rle_mask_box_and_centroid,
)
from evidencescene.runtime.sam2_cpu_preparation import (
    cpu_preparation_threads,
    validate_cpu_prepare_threads,
)
from evidencescene.runtime.sam2_gpu_window_overlap import (
    PendingObservationCpuPreparation,
    validate_and_create_pending_preparation,
)

__all__ = [
    "PendingObservationCpuPreparation",
    "RequestDeduplicationStats",
    "coplanar_rgbd_tag_observations",
    "cpu_preparation_threads",
    "fast_fit_shared_plane",
    "install_request_evidence_deduplication",
    "request_evidence_deduplication",
    "request_evidence_deduplication_scope",
    "rle_mask_box_and_centroid",
    "sam2_cpu_preparation",
    "sam2_gpu_window_overlap",
    "validate_and_create_pending_preparation",
    "validate_cpu_prepare_threads",
]
