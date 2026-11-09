# Experimental: dense direction by actual valid-depth count

Producer: CUDACache::storeFrameCalibrated / countNumValidDepth.
Consumer/owner: existing SolverBundling dense source/target selection; existing
Bundler and persistent surface retain output ownership. This patch dereferences
both valid-depth counts and retains the existing index tie break. It is absent
from the default manifest and does not authorize a pose-owner cutover.

The old pointer predicate was experimentally shown to change optimized poses on
identical cache inputs (up to 2.674 degrees versus 0.001581-degree repeat drift).
With this count rule common to both arms, eight complete development replays
compare the quarantined 0003 normal-return free against no free. The previous
large Cookiejar rotation/ADD regression does not recur; cross-arm trajectory
rotation disagreement is 1.288–3.933 degrees, compared with 35.919–42.667 degrees
in the earlier pointer-rule experiment. These are separate campaigns, not a
randomized factorial test. Surface coverage decreases slightly in all four new
pairs, and quality changes remain mixed. This is not proof of physical
noninferiority, superiority, full resource stability, or live latency compliance.

0003 remains quarantined. Do not enable either patch through the default
manifest based solely on these development recordings. The source correction
and its interaction evidence are kept reviewable for a future validated branch.

Evidence:
- `papers/evidencescene/bundletrack-count-values-table.tex`
- `papers/evidencescene/bundletrack-order-table.tex`
- `papers/evidencescene/bundletrack-winner-table.tex`
