# Full 9/9 Official YCBInEOAT & 3/3 Laval Real-Video Public Benchmark Suite

This snapshot upgrades the public benchmark evaluation of **EvidenceScene** along four dimensions:

1. **Full 9/9 Official YCBInEOAT Coverage (`7,449` total / `7,440` post-initial frames):** Stages and evaluates all nine official YCBInEOAT sequences under **both** Oracle Reference Masks and Shared Autonomous SAM 2 Video Masks (Point2Pose `sam2.1_hiera_large` segmentation fed identically to both trackers).
2. **Unified 3D Surface Reconstruction Evaluation (`P10`, `C10`, `F1@10`, `F95`, `R95`):** Reconstructs causal Frame-0-aligned surfaces in `RevisableSurfelMap` at `4 mm` resolution (strided keyframes, `support >= 2`, `support10/coarse12mm` `1,500`-point budget, **measured sensor surfels only**, `is_prior = False`) and scores them against CAD surfaces without ICP drift masking.
3. **Moving-Block Bootstrap 95% Confidence Intervals (`b = 20, B = 2,000`):** Quantifies finite-horizon statistical uncertainty for temporally correlated video frames via circular moving-block bootstrap CIs (`evidencescene.metrics`) on per-sequence AUCs, pooled frame-weighted AUCs, and paired differences (`EvidenceScene - Point2Pose`).
4. **CAD-Assisted FoundationPose Reference + Real-Video Laval Unified `2x2` Factorial & C11 Schur Benchmark:** Evaluates CVPR 2024 `FoundationPose` (`NVlabs a1b694b`) across all `9/9` YCBInEOAT sequences and all `3/3` textured 1080p Laval clips (`704` total / `701` post-initial frames), runs the frozen-threshold unified-branch `2x2` factorial (`Gate G+R x Birth-Indexed RevisableSurfelMap S`) directly on real 1080p Laval RGB-D video, and benchmarks C11 `evaluate_schur_observability_gate` against pure-NumPy SVD across all `18/18` real connected GTSAM updates.

---

## Pinned Code & Dataset Provenance

| Entity | Pinned Revision / Digest | Role |
| --- | --- | --- |
| **EvidenceScene** Composed Native | `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1` + deterministic patches + `NO_BA` retirement | Model-free primary pose & `RevisableSurfelMap` owner |
| **Point2Pose** (Upstream) | `51856226610df75e5c06e8de545bd27f7c4ba99c` (`BootsTAPIR` + `SAM 2.1`) | Model-free baseline & shared SAM 2 mask generator |
| **FoundationPose** (CVPR 2024) | `a1b694b83e633c2cb6115b9063d940a687759392` (`NVlabs/FoundationPose`) | **CAD-assisted reference** (receives exact object mesh) |

### Newly Staged Official Rutgers PRACSYS YCBInEOAT Archives

All six previously local sequences (`cracker_box_reorient`, `mustard0`, `bleach0`, `bleach_hard_00_03_chaitanya`, `sugar_box_yalehand0`, `tomato_soup_can_yalehand0`) are combined with the three official Rutgers archives below:

| Sequence | Frames (`n_post_init`) | Official Archive SHA-256 |
| --- | ---: | --- |
| `mustard_easy_00_02` | `689` (`688`) | `825ade51d2f49d062d03364c79b9a31b648e04ee7499709c2bc9950b90c4645a` |
| `sugar_box1` | `907` (`906`) | `85f0be38402109af59fa3f85234f8f76eeb3d44283703079382f633eee865032` |
| `cracker_box_yalehand0` | `1,327` (`1,326`) | `60da341cc4cc4a642cef9ccc6a4f945c6753cd9da3378004ca353b8bf6b78bd7` |

**Hardware:** NVIDIA RTX 5090 32 GB, AMD Ryzen 7 9800X3D (Linux). All GPU runs executed strictly sequentially (`OMP_NUM_THREADS=1`).

---

## Results Summary

### 1. Full 9/9 Official YCBInEOAT (`7,440` Post-Initial Frames)

All three systems are scored symmetrically under `evidencescene.metrics.evaluate_pose_sequence` using canonical PoseCNN/BOP `est -> ref` nearest-vertex ADD-S (`min_{q in M_ref} ||p_est - q||_2`) and corresponding-vertex ADD over `0-100 mm` (failures scored zero). Note: frozen October 2026 receipts (`benchmarks/receipts/*/score.json`, Table I in the paper) used the reverse `ref -> est` query direction (`min_{p in M_est} ||q_ref - p||_2`); `score_public_suite.py` verifies exact parity on all six overlapping historical `Point2Pose` receipts in that direction while reporting unified canonical `est -> ref` scores here.

| Track / Metric (`9/9`, `n = 7,440`) | Point2Pose (Model-Free) | **EvidenceScene (Model-Free, Ours)** | FoundationPose (CAD-Assisted*) |
| --- | ---: | ---: | ---: |
| **Ref-Mask Pooled ADD-S AUC** `[95% CI]` | `0.8813` `[0.8612, 0.8993]` | **`0.9239`** `[0.9140, 0.9325]` | `0.9551` `[0.9496, 0.9595]` |
| **Ref-Mask Pooled ADD AUC** `[95% CI]` | `0.7942` `[0.7709, 0.8162]` | **`0.8447`** `[0.8312, 0.8576]` | `0.9165` `[0.9092, 0.9228]` |
| **Ref-Mask `ADD-S <= 10 mm`** | `5,529 / 7,440` | **`5,611 / 7,440`** | `7,257 / 7,440` |
| **Ref-Mask Paired `Delta(ES - P2P)` ADD-S / ADD** | — | **`+0.0426`** `[+0.0279, +0.0593]` / **`+0.0504`** `[+0.0358, +0.0663]` | — |
| **SAM 2 Auto Pooled ADD-S AUC** `[95% CI]` | `0.8828` `[0.8630, 0.9018]` | **`0.9313`** `[0.9216, 0.9393]` | — |
| **SAM 2 Auto Pooled ADD AUC** `[95% CI]` | `0.7968` `[0.7710, 0.8230]` | **`0.8640`** `[0.8524, 0.8749]` | — |
| **SAM 2 Auto Paired `Delta(ES - P2P)` ADD-S / ADD** | — | **`+0.0485`** `[+0.0300, +0.0670]` / **`+0.0672`** `[+0.0461, +0.0892]` | — |
| **Ref 3D Surface `P10 / C10` (`F95`)** | `85.49% / 48.90%` (`18.05 mm`) | **`86.22% / 49.52%` (`14.59 mm`)** | `92.65% / 54.92%` (`8.25 mm`) |
| **Auto 3D Surface `P10 / C10` (`F95`)** | `83.76% / 51.51%` (`23.25 mm`) | **`82.70% / 49.87%` (`14.17 mm`)** | — |
| **Full-Audit Tracking Latency (`p50`)** | `121.13 ms` (new Ref) / `206.59 ms` (new Gen) | **`49.95 ms` (Ref)** / **`56.60 ms` (Auto)** | `11.84 ms` (CAD*) |

\* **Model-Free vs. CAD-Assisted Scope:** `FoundationPose` receives the exact 3D object CAD mesh (`textured_simple.obj` on YCBInEOAT, `geometry.ply` + `.bmp` texture on Laval); `EvidenceScene` and `Point2Pose` reconstruct geometry online without CAD priors. On visually symmetric `sugar_box_yalehand0`, FoundationPose's initial CAD `register()` locks $180^\circ$ rotated relative to the annotation frame (`no_align` ADD AUC `0.2871`, ADD-S `0.9523`), which Frame-0 gauge alignment maps to ADD `0.8694` (`0.9228` ADD-S).

Full machine-readable per-sequence metrics and bootstrap CIs are archived in [`results/ycbineoat_9seq_full_scorecard.json`](results/ycbineoat_9seq_full_scorecard.json).

---

### 2. Real-Video Public Unified `2x2` Factorial & 1080p Laval Comparison (`701` Post-Initial Frames)

| Track / Arm (`Cookiejar` 295 + `Clock-hard` 202 + `Clock-full` 204) | Comm. Avail. | Ret. Avail. | Mean Comm. ADD AUC | Mean Ret. ADD AUC | Mean `P10` | Mean `C10` | Mean `F1@10` | Mean `F95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unified `Gate OFF, S OFF` | `450 / 701` | `701 / 701` | `0.5088` | `0.6606` | `80.44%` | `39.31%` | `51.84%` | `13.95 mm` |
| Unified `Gate OFF, S ON` | `450 / 701` | `701 / 701` | `0.5088` | `0.6606` | `82.38%` | `37.35%` | `51.18%` | `14.16 mm` |
| Unified `Gate ON, S OFF` | `407 / 701` | `701 / 701` | `0.4658` | `0.6405` | **`98.64%`** | **`41.62%`** | **`58.48%`** | **`7.34 mm`** |
| **Unified `Gate ON, S ON`** | `407 / 701` | `701 / 701` | `0.4658` | `0.6405` | **`98.62%`** | `41.40%` | `58.24%` | **`7.40 mm`** |

- **Cookiejar Highlights:** Enabling `Gate ON` blocks `33` disconnected/rank-deficient GTSAM updates (`30` disconnected, `3` rank/Schur-deficient out of `35` updates), lifting `1,500`-pt surface precision `P10` from **`45.20%` (`S OFF`) / `51.33%` (`S ON`) to `100.00%`**, raising `C10` from `36.31%` to `45.63%` (`F1@10` `40.27% -> 62.67%`), and cutting `F95` by **`4.59x`** (`25.19 -> 5.49 mm`).
- **C11 Schur Gate Benchmark across All `18` Connected Real Laval GTSAM Updates:** `18/18` updates executed on C11 fast path (`SCHUR_MAX_COLS = 192`), **`0` rank/admission parity mismatches** vs. pure-NumPy SVD, median **`585.98 us` vs. `1,122.61 us`** (`1.92x` speedup; mean `3,167.16 us`).
- **Cross-Method Laval Reference:** On `Cookiejar` / `Clock-hard` / `Clock-full`, Model-Free **EvidenceScene Composed Native** achieves ADD AUCs `0.8133` / `0.8958` / `0.9133` (ADD-S `0.9422` / `0.9562` / `0.9447`, $P_{10}$ `84.40%` / `96.20%` / `99.73%`, $p_{50}$ `122.5` / `117.5` / `119.5 ms`), compared to Scanned-Mesh **FoundationPose** (CAD*) ADD AUCs `0.9150` / `0.9371` / `0.9465` (ADD-S `0.9594` / `0.9700` / `0.9743`, $p_{50}$ `12.7` / `12.5` / `12.9 ms`).

Full machine-readable Laval receipts are archived in [`results/laval_3seq_competitors_and_unified_2x2.json`](results/laval_3seq_competitors_and_unified_2x2.json).

---

## Reproducing

Point environment overrides to your local dataset and external research checkouts (kept cleanly outside the `evidencescene` repository):

```bash
export EVIDENCESCENE_YCB_ROOT=/path/to/YCBInEOAT_all_9_sequences
export EVIDENCESCENE_LAVAL_CACHE_ROOT=/path/to/laval_registered_actions
export EVIDENCESCENE_PRIOR_YCB_ROOT=/path/to/prior_ycb_results
export EVIDENCESCENE_FOUNDATIONPOSE_ROOT=/path/to/FoundationPose-a1b694b
export EVIDENCESCENE_POINT2POSE_ROOT=/path/to/point-to-pose
export EVIDENCESCENE_BUNDLESDF_SOURCE=/path/to/BundleSDF-runtime-pinned
export EVIDENCESCENE_BUNDLESDF_BUILD=/path/to/composed-native-build
export EVIDENCESCENE_BUNDLESDF_PREFIX=/path/to/bundlesdf-deps-prefix
export EVIDENCESCENE_LOFTR_CHECKPOINT=/path/to/BundleTrack/LoFTR/weights/outdoor_ds.ckpt
export EVIDENCESCENE_RUN_CACHE_ROOT=/path/to/run_cache

# 1. CAD-assisted FoundationPose (YCBInEOAT + textured Laval meshes)
/path/to/fp-venv/bin/python benchmarks/competitors/2026-11-21-public-suite/run_foundationpose_suite.py --include-bleach0-timing

# 2. Point2Pose missing YCBInEOAT reference & SAM 2 auto-mask runs
/path/to/bt-venv/bin/python benchmarks/competitors/2026-11-21-public-suite/run_point2pose_suite.py

# 3. EvidenceScene Composed Native across all 9/9 YCBInEOAT (Ref + Gen) & 3/3 Laval
/path/to/bt-venv/bin/python benchmarks/competitors/2026-11-21-public-suite/run_bundletrack_suite.py --include-generated --include-laval

# 4. Real-video Laval Unified 2x2 Factorial + C11 Schur Gate Benchmark
/path/to/bt-venv/bin/python benchmarks/competitors/2026-11-21-public-suite/run_laval_unified_2x2.py

# 5. Unified Scoring + 95% Block-Bootstrap CIs + LaTeX Table Generation
/path/to/bt-venv/bin/python benchmarks/competitors/2026-11-21-public-suite/score_public_suite.py
```

## Caveats

- Video frames within each continuous manipulation clip remain temporally correlated; circular moving-block bootstrap CIs (`b = 20`) account for short-range autocorrelation within sequences, not cross-clip domain shift across nine/three clips.
- All benchmark scores remain offline development evaluations conditioned on depth intrinsics, initial masks, and offline CAD/Vicon reference poses; physical live-system site acceptance remains `0/6`.
