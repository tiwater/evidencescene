#!/usr/bin/env python3
"""End-to-end closed-loop demonstration of EvidenceScene.

Runs a 24-frame synthetic RGB-D manipulation sequence through:
1. O(runs) RLE mask encoding, bounding-box/centroid extraction, and pairwise IoU
2. Factor-graph observability gating (F + G + R + M) with collinear degeneracy rejection
3. Lineage-indexed RevisableSurfelMap (M + S) with atomic pose-revision rebuild
4. Causal SE(3) trajectory prediction with C1 occlusion re-acquisition blending
5. L2 ballistic pouring safety verification and standard ADD/ADD-S/P_10/C_10 scoring
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidencescene.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
