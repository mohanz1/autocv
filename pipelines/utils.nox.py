"""Additional utilities for Nox."""

from __future__ import annotations

import os
import shutil

from pipelines import nox

DIRECTORIES_TO_DELETE = [
    ".nox",
    "build",
    "dist",
    "autocv.egg-info",
    "public",
    ".pytest",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
]

FILES_TO_DELETE = [".coverage", "package-lock.json"]

TO_DELETE = [(shutil.rmtree, DIRECTORIES_TO_DELETE), (os.remove, FILES_TO_DELETE)]


def _try_delete(session: nox.Session, func: object, /, path: str) -> None:
    try:
        func(path)
    except Exception as exc:  # noqa: BLE001
        session.warn(f"[ FAIL ] Failed to remove {path!r}: {exc!s}")
    else:
        session.log(f"[  OK  ] Removed {path!r}")


@nox.session(venv_backend="none")
def purge(session: nox.Session) -> None:
    """Delete any nox-generated files."""
    for func, trash_list in TO_DELETE:
        for trash in trash_list:
            _try_delete(session, func, trash)
