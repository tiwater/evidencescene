#!/usr/bin/env python3
"""Static AST architecture & DAG dependency checker for EvidenceScene (`make check-arch`).

Enforces:
  1. Zero circular dependencies across subpackages (`primitives`, `observability`,
     `surface`, `tracking`, `runtime`, `metrics`, `teleop`).
  2. Strict lower-layer purity: `primitives` never imports higher subpackages.
  3. Zero imports from `benchmarks` inside the `evidencescene` runtime package.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "evidencescene"

LOWER_PRIMITIVES = "primitives"
HIGHER_SUBPACKAGES = frozenset({"observability", "surface", "tracking", "runtime", "metrics", "teleop"})


def _find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    visited: set[str] = set()
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[list[str]] = []

    def dfs(node: str) -> None:
        visited.add(node)
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt not in visited:
                dfs(nxt)
            elif nxt in on_stack:
                idx = stack.index(nxt)
                cycles.append(stack[idx:] + [nxt])
        stack.pop()
        on_stack.remove(node)

    for n in sorted(graph.keys()):
        if n not in visited:
            dfs(n)
    return cycles


def main() -> int:
    errors: list[str] = []
    edges: dict[str, set[str]] = {}

    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(PKG_ROOT)
        src_pkg = rel.parts[0] if len(rel.parts) > 1 else "_root"
        edges.setdefault(src_pkg, set())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            mod = ""
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("benchmarks"):
                        errors.append(f"[pkg-bench] {rel}:{lineno} imports `{alias.name}`")
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod.startswith("benchmarks"):
                    errors.append(f"[pkg-bench] {rel}:{lineno} imports `{mod}`")
                if mod.startswith("evidencescene."):
                    parts = mod.split(".")
                    tgt_pkg = parts[1] if len(parts) > 1 else "_root"
                    if tgt_pkg != src_pkg and tgt_pkg != "_accel":
                        edges[src_pkg].add(tgt_pkg)
                        if src_pkg == LOWER_PRIMITIVES and tgt_pkg in HIGHER_SUBPACKAGES:
                            errors.append(
                                f"[layer] {rel}:{lineno} `primitives` must not import higher layer `{tgt_pkg}`"
                            )

    for cyc in _find_cycles(edges):
        errors.append(f"[cycle] circular subpackage dependency: {' -> '.join(cyc)}")

    if errors:
        print("EvidenceScene architecture check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    edge_summary = ", ".join(f"{k}->{sorted(v)}" for k, v in sorted(edges.items()) if v)
    print(f"EvidenceScene architecture check OK (strict DAG, zero cycles): {edge_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
