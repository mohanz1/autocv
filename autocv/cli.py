"""Provides the `python -m autocv` and `autocv` commands to the shell."""

from __future__ import annotations

import pathlib
import platform
import sys

from autocv import _about


def main() -> None:
    """Print package info and exit."""
    path = str(pathlib.Path(_about.__file__).resolve().parent)
    sha1 = _about.__git_sha1__[:8]
    version = _about.__version__
    py_impl = platform.python_implementation()
    py_ver = platform.python_version()
    py_compiler = platform.python_compiler()

    sys.stderr.write(f"autocv ({version}) [{sha1}]\n")
    sys.stderr.write(f"located at {path}\n")
    sys.stderr.write(f"{py_impl} {py_ver} {py_compiler}\n")
    sys.stderr.write(" ".join(frag.strip() for frag in platform.uname() if frag and frag.strip()) + "\n")
