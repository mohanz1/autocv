"""Additional utilities for Nox."""

from __future__ import annotations

import os
import shutil

from pipelines import nox

DIRECTORIES_TO_DELETE = [
    ".nox",
    "build",
    "dist",
    "hikari.egg-info",
    "public",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
]

FILES_TO_DELETE = [".coverage", "package-lock.json"]

TO_DELETE = [(shutil.rmtree, DIRECTORIES_TO_DELETE), (os.remove, FILES_TO_DELETE)]


@nox.session(venv_backend="none")
def purge(session: nox.Session) -> None:
    """Delete any nox-generated files."""
    for func, trash_list in TO_DELETE:
        for trash in trash_list:
            try:
                func(trash)
            except Exception as exc:  # noqa: BLE001, PERF203
                session.warn(f"[ FAIL ] Failed to remove {trash!r}: {exc!s}")
            else:
                session.log(f"[  OK  ] Removed {trash!r}")
