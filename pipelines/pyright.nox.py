"""Pyright integrations."""

from __future__ import annotations

from pipelines import nox


@nox.session()
def pyright(session: nox.Session) -> None:
    """Perform static type analysis on Python source code using pyright.

    This session is currently optional; enable it in CI if/when you want
    an additional type-checking pass alongside mypy.
    """
    nox.sync(session, self=True, groups=["pyright"])
    session.run("pyright")
