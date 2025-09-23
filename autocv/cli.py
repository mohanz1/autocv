"""Command-line interface for reporting package metadata."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from . import _about


def _format_lines() -> list[str]:
    lines: list[str] = []
    path = Path(_about.__file__).resolve().parent
    sha1 = _about.__git_sha1__[:8]
    version = _about.__version__
    uname = " ".join(part.strip() for part in platform.uname() if part and part.strip())
    lines.append(f"autocv {version} [{sha1}]")
    lines.append(f"located at {path}")
    lines.append(f"{platform.python_implementation()} {platform.python_version()} {platform.python_compiler()}")
    if uname:
        lines.append(uname)
    return lines


def main() -> None:
    """Emit package metadata to standard output."""
    sys.stdout.write("\n".join(_format_lines()) + "\n")
