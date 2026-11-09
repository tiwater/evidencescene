# EvidenceScene Preprint (Main Text + Appendices)

`main.tex` is the unified arXiv preprint:

> **EvidenceScene: Observability-Conditioned Pose Tracking and Revisable
> 3D Scene Reconstruction for Robot Manipulation**

It is a mechanism-focused study combining the core method and evaluation
(Sections I–VI) with the full 31-framework audit, bilateral kinematic loop
closure, and bimanual pouring telemetry in Appendices A–C. It separates:

1. the pinned upstream BundleTrack and Point2Pose implementations;
2. the EvidenceScene repairs and their controlled ablations; and
3. the live-authority boundary: the native BundleTrack branch owns the live
   pose/surface path, while the Point2Pose branch is an offline shadow.

## Scope and evidence status

All quantitative comparisons are development results on already-opened
recordings. Pose scores use reference masks and reference-assisted
initialization; surface scores use a CAD or sampled-model reference. They are
not held-out, blind, or physical-ground-truth results. No result in this
manuscript authorizes a live pose-estimator cutover.

The claims in this preprint are supported by the figures and tables
vendored here and by the benchmark evidence directory under `benchmarks/`.

## Build

From this directory:

```sh
tectonic main.tex
cp main.pdf evidencescene.pdf
```

The build is self-contained: all figures and table fragments used by
`main.tex` are checked in under this directory or `../figures/`. The checked-in
PDF is `evidencescene.pdf`.

Build verification: 8 pages, no overfull boxes, no undefined references.
`\sloppy` is enabled because several code identifiers are long; underfull-box
warnings are typographic only.

## Assets

| Asset | Role |
| --- | --- |
| `pose-table.tex` | YCBInEOAT reference-mask development comparison |
| `fork-reference-table.tex` | Laval Vicon B/P/F comparison |
| `anchor-admission-table.tex` | G/R/M graph interventions |
| `surface-transaction-table.tex` | G/M + S factorial |
| `budget-transfer-table.tex` | 1,500-point selector transfer |
| `../figures/fig_cookiejar_paradox.pdf` | availability--surface paradox |
| `../figures/fig_graph_ablation.pdf` | global-graph switch ablation |
| `../figures/fig_surface_interaction.pdf` | M x S interaction |
| `../figures/fig_surface_cdf.pdf` | reference-pose surface counterfactual |
| `../figures/fig_budget_transfer.pdf` | selector coverage transfer |

The five figure PDFs and `budget-transfer-table.tex` are vendored under
`papers/` so that the manuscript builds without external experiment directories
at compile time.

## Evidence map

The benchmark receipts and replay scripts are organized under `benchmarks/` in
the EvidenceScene repository. The most relevant study bundles are:

- `laval-fork-development-20261030/` — F/G/R/M/S scores, graph audits, surface
  transactions, and the original figure scripts.
- `reference-pose-surface-counterfactual-20261101/` — native vs. reference-pose
  surface attribution.
- `existing-point-budget-selection-transfer-20261101/` — selector transfer and
  latency.
- `composed-native-retention-public-20261101/` — native BundleTrack repair
  comparison (see the native evidence receipt referenced by the paper).
- `sam-gpu-window-cpu-overlap-20261102/` and
  `request-evidence-deduplication-20261102/` — bit-exact runtime ablations.
- `performance-claim-audit-20261101/` — claim-level verification entry point.

Those directories contain the per-frame scores, input receipts, source patches,
and replay scripts. They are large (the full tree is multi-gigabyte), so this
repository vendors only the derived assets needed to build the manuscript.

## Older material

`detailed-evidence.tex` and `detailed-evidence.pdf` preserve the earlier,
maximally cautious development report. They are not the current submission and
are retained for provenance. Some table fragments in this directory
(e.g., `bundletrack-count-values-table.tex`) are unused by `main.tex` and are
kept as evidence artifacts.

## Pinned upstream versions

- BundleTrack: `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`
- Point2Pose: `51856226610df75e5c06e8de545bd27f7c4ba99c`
- Project page: <https://tiwater.github.io/evidencescene/>

## Regenerating from evidence

The checked-in tables are the review copy. They can be regenerated offline from
the vendored receipts:

```sh
cd ../..
make tables
```

`benchmarks/receipts/` contains the ten score files needed by
`generate_table.py` and `generate_generated_mask_table.py`; both generators
verify the SHA-256 digest of every receipt before rendering. Raw multi-gigabyte
bag recordings are excluded from git; only the derived assets and JSON receipts
are checked in. Do not regenerate a table
from a changed receipt without recording the new source identity in the paper.
