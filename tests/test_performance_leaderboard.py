"""Contract checks for the homepage performance dashboard."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_dashboard_keeps_own_and_external_data_separate() -> None:
    data = json.loads((ROOT / "docs" / "performance_leaderboard.json").read_text())
    assert data["schema"] == "evidencescene.performance-dashboard.v1"
    kernel = data["kernel"]
    assert kernel["protocol"] == "kernel-v1"
    assert len(kernel["rows"]) == 7
    for row in kernel["rows"]:
        assert row["baseline_ms"] > 0 and row["candidate_ms"] > 0
        assert row["speedup"] > 0

    snapshot = data["external_snapshot"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot["date"])
    controlled = snapshot["controlled"]
    assert controlled["protocol"] == "ycbineoat_bleach0_refmask_40f_v1"
    assert re.fullmatch(r"[0-9a-f]{64}", controlled["input_sha256"])
    assert len(controlled["rows"]) >= 2
    for row in controlled["rows"]:
        assert row["p50_ms"] > 0 and row["p95_ms"] >= row["p50_ms"]
    assert len(snapshot["reference"]["rows"]) >= 4
    assert "not run by CI" in snapshot["note"]


def test_homepage_renders_own_and_dated_external_tables() -> None:
    page = (ROOT / "docs" / "index.html").read_text()
    assert page.index('id="overview"') < page.index('id="abstract"') < page.index('id="papers"')
    assert "Project Overview" in page
    block = page.split("PERFORMANCE_LEADERBOARD_START", 1)[1].split(
        "PERFORMANCE_LEADERBOARD_END", 1
    )[0]
    assert "EvidenceScene kernel performance" in block
    assert "External competitor snapshot" in block
    assert "Controlled same-input subset" in block
    assert "Reference context" in block
    assert "ycbineoat_bleach0_refmask_40f_v1" in block
    assert "BundleSDF" in block
    assert "FoundationPose" in block
    assert "MonoGS" in block
    assert "MASt3R" in block
    assert "nvblox" in block

    demo_page = (ROOT / "docs" / "demo" / "index.html").read_text()
    assert 'id="demo-scene3d"' in demo_page
    assert "initScene3DViewer" in demo_page
    assert "cameras_4view_sync.mp4" in demo_page
    assert "See the scene" not in demo_page

    bench_page = (ROOT / "docs" / "benchmark" / "index.html").read_text()
    assert 'id="public-pose"' in bench_page
    assert 'id="surface-ablation"' in bench_page
    assert 'id="runtime-overlap"' in bench_page
    assert 'id="calibration-pouring"' in bench_page
    assert 'id="framework-audit"' in bench_page
    assert 'id="kernel-competitor"' in bench_page
    assert "e3eb54684e0d0383737faa77fc04420aead32826dbf305d0ff75dead54513948" in bench_page
    assert "ycbineoat_bleach0_refmask_40f_v1" in bench_page
    assert 'href="benchmark/"' in page


def test_ablation_benchmark_suite_invariants() -> None:
    from benchmarks.performance.bench_ablation import run_ablation_suite

    report = run_ablation_suite()
    assert report["schema"] == "evidencescene.ablation-benchmark.v1"
    abl = report["ablations"]
    assert abl["1_observability_gate_vs_ungated_pose"]["add_error_reduction_x"] > 3.0
    assert abl["1_observability_gate_vs_ungated_pose"]["run_to_run_spread_mm"] == 0.0
    assert abl["2_revisable_surfel_vs_static_voxel"]["f95_error_reduction_x"] > 1.8
    assert abl["3_rle_algebra_vs_dense_1080p"]["exact_parity"] is True
    assert abl["3_rle_algebra_vs_dense_1080p"]["speedup_x"] > 3.0
    assert abl["4_c1_causal_se3_vs_unblended_reacquisition"]["evidencescene_c1_position_jump_mm"] == 0.0
    assert abl["4_c1_causal_se3_vs_unblended_reacquisition"]["monotonic_horizon_uncertainty"] is True


def test_public_pages_and_docs_never_mention_ticos() -> None:
    """Public pages, dashboard JSON, README, and papers must not leak internal 'ticos' references."""
    public_files = [
        ROOT / "README.md",
        ROOT / "docs" / "index.html",
        ROOT / "docs" / "demo" / "index.html",
        ROOT / "docs" / "benchmark" / "index.html",
        ROOT / "docs" / "performance_leaderboard.json",
        ROOT / "docs" / "assets" / "js" / "scene3d-viewer.js",
        ROOT / "papers" / "evidencescene-short" / "main.tex",
        ROOT / "papers" / "evidencescene-short" / "detailed-evidence.tex",
        ROOT / "papers" / "evidencescene-short" / "README.md",
        ROOT / "benchmarks" / "competitors" / "2026-11-06-ycb" / "README.md",
        ROOT / "benchmarks" / "sam_gpu_window_cpu_overlap_20261102" / "README.md",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert "ticos" not in text.lower(), f"Found forbidden 'ticos' mention in {path}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "img.shields.io" not in readme
    assert "camo.githubusercontent.com" not in readme
    for img_src in re.findall(r'<img[^>]+src="([^"]+)"', readme):
        assert not img_src.startswith(("http://", "https://")), f"External image in README: {img_src}"
        assert (ROOT / img_src).is_file(), f"Missing README image: {img_src}"
