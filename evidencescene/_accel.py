"""Optional C11 native kernel loader with content-addressed JIT compile fallback.

Loads a prebuilt ``evidencescene._native`` extension if installed (for example
via ``pip install -e .`` or wheel build), or lazily compiles ``_native.c`` into a
content-addressed per-user cache directory (``~/.cache/evidencescene/native``)
so raw git worktrees and development checkouts also run at full native speed.
Falls back cleanly to pure-NumPy implementations whenever compilation is
unavailable or ``EVIDENCESCENE_DISABLE_NATIVE=1`` is set.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from types import ModuleType
import uuid

_CACHE_PREFIX = "evidencescene_native"


def _ext_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if isinstance(suffix, str) and suffix:
        return suffix
    return ".pyd" if os.name == "nt" else ".so"


def _cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    candidate = base / "evidencescene" / "native"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".write_test_{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / f"{_CACHE_PREFIX}_{os.getuid() if hasattr(os, 'getuid') else 0}"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _load_extension_from_path(path: Path) -> ModuleType | None:
    if not path.is_file():
        return None
    try:
        loader = importlib.machinery.ExtensionFileLoader("evidencescene._native", str(path))
        spec = importlib.util.spec_from_loader("evidencescene._native", loader)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - safe fallback to pure Python
        return None


def _compile_and_load_native() -> ModuleType | None:
    src_path = Path(__file__).with_name("_native.c")
    if not src_path.is_file():
        return None

    cc = (
        os.environ.get("CC")
        or shutil.which("cc")
        or shutil.which("gcc")
        or shutil.which("clang")
    )
    if not cc:
        return None
    inc_dir = sysconfig.get_path("include")
    if not inc_dir or not (Path(inc_dir) / "Python.h").is_file():
        return None

    src_bytes = src_path.read_bytes()
    tag = f"{sys.version_info.major}.{sys.version_info.minor}:{platform.machine()}:{sys.platform}"
    digest = hashlib.sha256(src_bytes + b"|" + tag.encode("utf-8")).hexdigest()[:20]

    suffix = _ext_suffix()
    out_dir = _cache_dir()
    target = out_dir / f"_native_{digest}{suffix}"
    cached = _load_extension_from_path(target)
    if cached is not None:
        return cached

    tmp_out = out_dir / f"._native_{digest}.{os.getpid()}.{uuid.uuid4().hex[:8]}{suffix}"
    link_flags = (
        ["-bundle", "-undefined", "dynamic_lookup"]
        if sys.platform == "darwin"
        else ["-shared", "-fPIC", "-pthread"]
    )
    base_cmd = [
        cc,
        *link_flags,
        "-O3",
        "-std=c11",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I",
        inc_dir,
        str(src_path),
        "-lm",
        "-o",
        str(tmp_out),
    ]
    # Try -march=native first for local AVX2/AVX-512 codegen, then generic.
    arch_flag = "-march=native" if platform.machine().lower() in {"x86_64", "amd64", "aarch64", "arm64"} else None
    attempts = ([base_cmd[0], arch_flag, *base_cmd[1:]], base_cmd) if arch_flag else (base_cmd,)
    compiled = False
    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True)
            if proc.returncode == 0 and tmp_out.is_file():
                compiled = True
                break
        except OSError:
            continue
    if not compiled:
        try:
            tmp_out.unlink()
        except OSError:
            pass
        return None

    try:
        os.replace(tmp_out, target)
    except OSError:
        pass
    loaded = _load_extension_from_path(target)
    if loaded is not None:
        return loaded
    loaded = _load_extension_from_path(tmp_out)
    return loaded


def _load_native() -> ModuleType | None:
    if os.environ.get("EVIDENCESCENE_DISABLE_NATIVE") == "1":
        return None
    try:
        from evidencescene import _native  # type: ignore[attr-defined]

        return _native
    except ImportError:
        pass
    return _compile_and_load_native()


NATIVE: ModuleType | None = _load_native()


def native_enabled() -> bool:
    """Return True when the native C11/SIMD acceleration module is active."""
    return NATIVE is not None


__all__ = ["NATIVE", "native_enabled"]
