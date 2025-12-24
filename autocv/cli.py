"""Command-line entry point for AutoCV.

The CLI prints the installed version, location, and interpreter/runtime details
to stderr, then exits.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from . import _about  # pyright: ignore[reportPrivateUsage]

__all__ = ("main",)


def main() -> None:
    """Print package info to stderr and return."""
    about_file = _about.__file__
    path = Path(about_file).resolve().parent if about_file else Path.cwd()
    sha1 = _about.__git_sha1__[:8]
    version = _about.__version__
    python_summary = f"{platform.python_implementation()} {platform.python_version()} {platform.python_compiler()}"
    uname_summary = " ".join(part.strip() for part in platform.uname() if part and part.strip())

    sys.stderr.write(f"autocv ({version}) [{sha1}]\n")
    sys.stderr.write(f"located at {path}\n")
    sys.stderr.write(f"{python_summary}\n")
    sys.stderr.write(f"{uname_summary}\n")
