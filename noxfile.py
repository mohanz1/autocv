"""Nox entry point."""

from __future__ import annotations

import pathlib
import runpy
import sys

sys.path.append(".")

ci_path = pathlib.Path("pipelines")
for f in ci_path.glob("*.nox.py"):
    runpy.run_path(str(f))
