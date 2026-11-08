"""Render EvidenceScene's homepage performance dashboard from checked-in JSON.

EvidenceScene's own kernel regression is the live dashboard. Competitor rows
are a dated, manually produced snapshot under ``benchmarks/competitors/`` and
are not run by CI or packaged as dependencies.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "performance_leaderboard.json"
PAGE = ROOT / "docs" / "index.html"
DEMO_PAGE = ROOT / "docs" / "demo" / "index.html"
START = "<!-- PERFORMANCE_LEADERBOARD_START -->"
END = "<!-- PERFORMANCE_LEADERBOARD_END -->"
DEMO_START = "<!-- DEMO_PERFORMANCE_LEADERBOARD_START -->"
DEMO_END = "<!-- DEMO_PERFORMANCE_LEADERBOARD_END -->"


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _link(text: str, url: str) -> str:
    safe_text = html.escape(text)
    safe_url = html.escape(url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_text}</a>' if url else safe_text


def _kernel_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['case'])}</code></td>"
            f"<td>{_fmt(row['baseline_ms'], 3)}</td>"
            f"<td><strong>{_fmt(row['candidate_ms'], 3)}</strong></td>"
            f"<td>{_fmt(row['speedup'], 2)}x</td>"
            "</tr>"
        )
    return f"""
      <h3>EvidenceScene kernel performance (<code>{html.escape(section['protocol'])}</code>)</h3>
      <p class="caption" style="margin-top:0.35rem;">
        Baseline <code>{html.escape(section['baseline'])}</code> → candidate <code>{html.escape(section['candidate_label'])}</code>
        (<code>{html.escape(section['candidate'])}</code>). Median latency in milliseconds; higher speed-up is better.
      </p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>Workload</th><th>Baseline (ms)</th><th>Current (ms)</th><th>Speed-up</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      <p class="caption">
        Geometric-mean speed-up: <strong>{_fmt(section['geomean_speedup'], 2)}x</strong>;
        output digests: <strong>{html.escape(section['output_parity'])}</strong>.
        {html.escape(section['notes'])}
      </p>
"""


def _controlled_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        css = ' class="highlight-row"' if row.get("highlight") else ""
        rows.append(
            f"<tr{css}>"
            f"<td><strong>{html.escape(row['system'])}</strong><br><code>{html.escape(row.get('variant', ''))}</code></td>"
            f"<td>{html.escape(row['scope'])}</td>"
            f"<td><strong>{_fmt(row.get('p50_ms'))}</strong></td>"
            f"<td>{_fmt(row.get('p95_ms'))}</td>"
            f"<td>{_fmt(row.get('throughput_fps'), 1)}</td>"
            f"<td>{_fmt(row.get('peak_rss_mb'), 1)}</td>"
            f"<td>{html.escape(row.get('frames', ''))}</td>"
            "</tr>"
        )
    return f"""
      <h4 style="margin-top:1.25rem;">Controlled same-input subset</h4>
      <p class="caption" style="margin-top:0.25rem;">
        Protocol <code>{html.escape(section['protocol'])}</code> · input SHA-256
        <code>{html.escape(section['input_sha256'][:16])}…</code><br>
        {html.escape(section['hardware'])} · {html.escape(section['software'])}
      </p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>System</th><th>Scope</th><th>p50 (ms)</th><th>p95 (ms)</th><th>Throughput (fps)</th><th>Peak RSS (MB)</th><th>Frames</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
"""


def _reference_table(section: dict) -> str:
    rows = []
    for row in section["rows"]:
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(row['system'])}</strong></td>"
            f"<td>{html.escape(row['scope'])}</td>"
            f"<td>{html.escape(row.get('latency_text', 'n/a'))}</td>"
            f"<td>{html.escape(row.get('hardware', ''))}</td>"
            f"<td>{_link(row.get('source', 'source'), row.get('source_url', ''))}</td>"
            "</tr>"
        )
    return f"""
      <h4 style="margin-top:1.5rem;">Reference context (not directly comparable)</h4>
      <p class="caption" style="margin-top:0.25rem;">{html.escape(section['note'])}</p>
      <div class="table-wrapper">
        <table class="benchmark-table">
          <thead><tr>
            <th>System</th><th>Scope / protocol</th><th>Reported latency</th><th>Hardware</th><th>Source</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
"""


def _external_snapshot(section: dict) -> str:
    return f"""
      <h3 style="margin-top:2rem;">External competitor snapshot ({html.escape(section['date'])})</h3>
      <p class="caption" style="margin-top:0.35rem;">
        {_link('Snapshot protocol and commands', section['url'])}. {html.escape(section['note'])}
      </p>
      {_controlled_table(section['controlled'])}
      {_reference_table(section['reference'])}
"""


def render() -> str:
    data = json.loads(DATA.read_text())
    return (
        '<div class="claim-callout">'
        "<strong>Scope:</strong> the kernel table tracks EvidenceScene's own regression. "
        "The competitor table is a dated, manually produced snapshot — not a CI job, package dependency or maintained leaderboard. "
        f'Last updated: {html.escape(data["updated"])}.'
        "</div>\n"
        + _kernel_table(data["kernel"])
        + _external_snapshot(data["external_snapshot"])
    )


def render_demo_summary() -> str:
    data = json.loads(DATA.read_text())
    kernel = data["kernel"]
    controlled = data["external_snapshot"]["controlled"]
    rows = []
    for row in kernel["rows"]:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['case'])}</code></td>"
            f"<td>{_fmt(row['baseline_ms'], 3)} ms</td>"
            f"<td><strong>{_fmt(row['candidate_ms'], 3)} ms</strong></td>"
            f"<td><strong>{_fmt(row['speedup'], 2)}x</strong></td>"
            "</tr>"
        )
    es_row = controlled["rows"][0]
    comp_row = controlled["rows"][1]
    return f"""        <p><strong>1. Kernel Benchmark Suite (<code>{html.escape(kernel['protocol'])}</code>):</strong> Geometric-mean speed-up of <strong>{_fmt(kernel['geomean_speedup'], 2)}x</strong> over baseline <code>{html.escape(kernel['baseline'])}</code> with <strong>{html.escape(kernel['output_parity'])}</strong> output digests (updated {html.escape(data['updated'])}).</p>
        <div class="table-wrapper" style="overflow-x:auto;margin:0.8rem 0;">
          <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
            <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.18);text-align:left;">
              <th style="padding:0.4rem;">Workload</th><th style="padding:0.4rem;">Baseline</th><th style="padding:0.4rem;">Current</th><th style="padding:0.4rem;">Speed-up</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <p><strong>2. Controlled Same-Input System Snapshot (<code>{html.escape(controlled['protocol'])}</code>):</strong> <strong>{html.escape(es_row['system'])}</strong> achieves <strong>{_fmt(es_row['p50_ms'], 2)} ms p50 / {_fmt(es_row['throughput_fps'], 1)} FPS</strong> vs. {html.escape(comp_row['system'])} at {_fmt(comp_row['p50_ms'], 2)} ms p50 / {_fmt(comp_row['throughput_fps'], 1)} FPS on identical locked input (<code>{html.escape(controlled['input_sha256'][:12])}…</code>).</p>
"""


def main() -> int:
    page = PAGE.read_text()
    block = START + "\n" + render() + END
    if START not in page or END not in page:
        raise SystemExit("performance leaderboard markers are missing")
    before, remainder = page.split(START, 1)
    _, after = remainder.split(END, 1)
    PAGE.write_text(before + block + after)

    if DEMO_PAGE.exists():
        demo_text = DEMO_PAGE.read_text()
        if DEMO_START in demo_text and DEMO_END in demo_text:
            d_before, d_rem = demo_text.split(DEMO_START, 1)
            _, d_after = d_rem.split(DEMO_END, 1)
            d_block = DEMO_START + "\n" + render_demo_summary() + "        " + DEMO_END
            DEMO_PAGE.write_text(d_before + d_block + d_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
