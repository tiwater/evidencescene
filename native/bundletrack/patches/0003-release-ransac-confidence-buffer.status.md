# Quarantined: RANSAC confidence-buffer release

`0003-release-ransac-confidence-buffer.patch` fixes an actual normal-return
allocation leak but is intentionally absent from the default manifest.

The fixed recorded-pair test removes 8 MiB of visible device-memory growth after
4,096 calls with exact inlier-output parity. However, eight complete development
replays show repeated Cookiejar rotation/ADD regression. Cause is unresolved;
changed allocation reuse and existing address-sensitive code are hypotheses,
not demonstrated explanations. Do not promote solely from micro-test parity.

Producer/owner remains native SiftManager/GluNet and the existing Frame state.
No replacement estimator or world publication is authorized by this patch.

Evidence: see `papers/evidencescene/bundletrack-repeat-table.tex`,
`papers/evidencescene/bundletrack-order-cross-table.tex`, and
`papers/evidencescene-monograph/main.tex`. Existing default manifest binaries are unchanged.
