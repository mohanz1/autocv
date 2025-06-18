"""Dependency scanning."""

from __future__ import annotations

from pipelines import nox


@nox.session()
def audit(session: nox.Session) -> None:
    """Perform dependency scanning."""
    nox.sync(session, groups=["audit"])
    session.run("uv-secure", "--forbid-yanked", "--desc", "--aliases")
