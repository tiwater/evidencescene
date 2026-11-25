# EvidenceScene v2.4.0

## Summary

EvidenceScene `v2.4.0` extends persistent static surface reconstruction to multi-plane scenes, adds subspace-projected Schur observability regularization for partially observable and rotationally symmetric objects, and consolidates the reproducible public competitor benchmark suite onto `main`.

## Highlights

### 1. Multi-Plane Persistent Static Surface Reconstruction (`evidencescene.surface.persistent_multi_plane`)
- Added `PersistentMultiPlaneSurfaceReconstructor`, `PlaneObservationSpec`, `MultiPlaneSurfaceConfig`, `MultiPlaneSurfaceSnapshot`, and `MultiPlaneUpdateResult`.
- Coordinates up to `max_planes` (default `8`) bounded `PersistentPlanarSurfaceReconstructor` instances under a single `threading.RLock`, inheriting C11/SIMD dirty-tile incremental meshing, `sensor_physics` weighting, free-space carving, and bounded memory budgets.
- **Deterministic seam arbitration**: Overlapping corner/seam samples (e.g., horizontal countertop + vertical backsplash) are deterministically assigned to the minimum-residual plane with lexicographic `plane_id` tie-breaking (`seam_tie_tolerance_m`), preventing double-layer mesh overlap at junctions without interfering with free-space carving.
- **Unified versioned snapshots & synchronized lifecycle**: `snapshot()` returns an immutable merged mesh with `vertex_plane_indices`, `triangle_plane_indices`, and per-plane `plane_snapshots`. Supports synchronized `correct_pose` / `correct_poses` broadcasts as well as per-plane and global `reset()`.

### 2. Subspace Schur Observability Regularization (`evidencescene.observability.factor_graph_observability`)
- Added `project_schur_observable_subspace(...)` and `SchurSubspaceRegularizationReport` alongside the unchanged binary `evaluate_schur_observability_gate(...)`.
- Performs eigendecomposition of the symmetric 6×6 marginal pose information block `pose_marginal_information_6x6` and projects the right-trivialized relative $\mathfrak{se}(3)$ tangent $\xi = \log(T_{\text{prior}}^{-1} T_{\text{cand}})$ onto the observable subspace before mapping back via exact $\mathrm{SE}(3)$ exponential map.
- Supports both `"tikhonov"` (closed-form MAP shrinkage $\lambda_k^+ / (\lambda_k^+ + \lambda_{\text{prior}})$ with null-space clamping) and `"hard_subspace"` (exact orthogonal projection with bit-identical full-rank passthrough), clamping rank-deficient directions (e.g., cylindrical spin around a symmetry axis) strictly to `prior_pose` while retaining observable DoFs, and failing back cleanly to `prior_pose` when rank is 0.

### 3. Reproducible Public Competitor Suite Consolidation
- Consolidated `benchmarks/competitors/2026-11-21-public-suite/` (Laval 3-sequence + YCBInEOAT 9-sequence unified 2×2 runners, scoring scripts, and JSON scorecards) cleanly onto `main` with public-hygiene test coverage in `tests/test_performance_leaderboard.py`.

## Verification

- Full standalone test suite passes with both native C11 acceleration enabled and pure-NumPy fallback (`EVIDENCESCENE_DISABLE_NATIVE=1`).
- New unit tests in `tests/test_persistent_multi_plane.py` and `tests/test_observability_metrics_and_cli.py::TestSchurSubspaceRegularization` verify L-shaped counter+backsplash fusion, seam arbitration tie-breaking, occlusion retention, pose correction broadcasts, resets, rank-6/rank-5/rank-0 subspace projection, and SE(3) validity.
