"""Build configuration with optional C11/SIMD acceleration extension."""

from __future__ import annotations

import os
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allow pure-Python wheel installation if a local C compiler is unavailable."""

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception:  # noqa: BLE001 - optional=True fallback
            if not getattr(ext, "optional", False):
                raise


if os.name == "nt":
    extra_compile_args = ["/O2", "/fp:precise"]
else:
    extra_compile_args = ["-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off"]

ext_modules = [
    Extension(
        "evidencescene._native",
        sources=["evidencescene/_native.c"],
        optional=True,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
)
