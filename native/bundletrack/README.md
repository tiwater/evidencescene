# Deterministic Native C++/CUDA `BundleTrack` (`native/bundletrack/`)

This directory contains the deterministic C++/CUDA source tree and patch series for **EvidenceScene**'s native 6-DoF keyframe pose-graph tracker (built upon upstream `NVlabs/BundleSDF` revision `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1`).

## 1. Directory Layout

```text
native/bundletrack/
├── CMakeLists.txt           # CUDA CMake build configuration (supports SM 80/86/89/90/120)
├── manifest.json            # Pinned upstream revision and SHA-256 verified default patch manifest
├── prepare_source.py        # Automated checkout & patch verifier against upstream BundleSDF
├── patches/                 # Modular git patches (0001–0004) and ablation status notes
│   ├── 0001-cmake-sm120-compatibility.patch
│   ├── 0002-support-disabled-pose-optimization.patch
│   ├── 0003-release-ransac-confidence-buffer.patch
│   └── 0004-select-dense-direction-by-count.patch
└── src/                     # Composed deterministic C++/CUDA BundleTrack sources
    ├── Bundler.cpp / .h     # Guarded contiguous-NO_BA frame retirement & keyframe window
    ├── FeatureManager.cpp   # Deterministic RANSAC winner selection & correspondence scoring
    ├── Frame.cpp / .h       # Keyframe state, CUDA cache binding, and orthonormal SE(3) pose inverse
    ├── Utils.cpp / .h       # Pinhole geometry & Eigen/CUDA matrix utilities
    └── cuda/                # Lexicographic CUDA hypothesis reduction & storeFrame kernels
```

## 2. Key Algorithmic Repairs in `src/`

1. **Deterministic CUDA RANSAC Winner Selection (`src/cuda/` & `src/FeatureManager.cpp`):**
   Replaces non-associative parallel atomic tie-breaking with a deterministic lexicographic reduction over `(inlier_count, -residual_L1, hypothesis_id)` and selects dense correspondence directions by valid depth count (`0004-select-dense-direction-by-count.patch`). Across 100 repeated native RANSAC + Procrustes calls on identical 13-correspondence inputs, run-to-run trajectory variance drops from `18.4 mm / 4.72°` to **`0.00 mm / 0.00°`**.
2. **Exact $\mathrm{SE}(3)$ Inverse-Pose Transfer in `CUDACache::storeFrame` (`src/cuda/` & `src/Frame.cpp`):**
   Replaces single-precision $4\times 4$ LU inversion and lossy float truncation with analytical group inversion $\mathbf{T}^{-1} = [\mathbf{R}^\top, -\mathbf{R}^\top\mathbf{t}; \mathbf{0}^\top, 1]$ and element-wise `float4x4` coefficient transfer, bounding backprojection error to $\le 1.30 \times 10^{-7}\,\mathrm{m}$.
3. **Guarded Contiguous-`NO_BA` Non-Keyframe Retirement (`src/Bundler.cpp`):**
   Invokes `Bundler::forgetFrame` on non-keyframes beyond the `window_size + 1` sliding budget when the most recent `window_size + 1` frames are contiguous `NO_BA` frames, while protecting the initial anchor frame, active keyframes, and pose-cached neighbors. Reclaims **`928.4–1,994.0 MiB` RSS** under continuous occlusion with **100% bit-exact trajectory and surface parity**.

## 3. Preparing and Building

To prepare a verified upstream `BundleSDF` checkout with the default manifest patches:

```bash
python3 native/bundletrack/prepare_source.py /path/to/BundleSDF-checkout
```

To build the composed deterministic `EvidenceScene` native libraries against an existing `BundleSDF` dependency prefix:

```bash
# Copy the composed deterministic C++/CUDA sources into your BundleSDF checkout
cp -r native/bundletrack/CMakeLists.txt native/bundletrack/src/* /path/to/BundleSDF-checkout/BundleTrack/

# Configure and build with CMake
cmake -S /path/to/BundleSDF-checkout/BundleTrack \
      -B /path/to/BundleSDF-build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/BundleSDF-build -j"$(nproc)"
```

## 4. Patch Manifest Governance

- **`0001-cmake-sm120-compatibility.patch`** *(Default Manifest)*: Adds modern NVIDIA Blackwell / Ada / Hopper (`sm_89`, `sm_90`, `sm_120`) CUDA architecture compatibility.
- **`0002-support-disabled-pose-optimization.patch`** *(Default Manifest)*: Exposes controlled `NO_BA` status handling when pose optimization is bypassed on under-constrained frames.
- **`0003-release-ransac-confidence-buffer.patch`** *(Opt-in Ablation)*: Releases the temporary CUDA RANSAC confidence buffer prior to normal return; documented in [`0003-release-ransac-confidence-buffer.status.md`](patches/0003-release-ransac-confidence-buffer.status.md).
- **`0004-select-dense-direction-by-count.patch`** *(Opt-in Ablation)*: Dereferences valid-depth count pointers during bidirectional dense correspondence selection; documented in [`0004-select-dense-direction-by-count.status.md`](patches/0004-select-dense-direction-by-count.status.md).
