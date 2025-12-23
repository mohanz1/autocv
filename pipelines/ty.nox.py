"""Ty integrations."""

from __future__ import annotations

from pipelines import nox


@nox.session()
def ty(session: nox.Session) -> None:
    """Perform static type analysis on Python source code using ty."""
    nox.sync(session, self=True, groups=["ty"])

    session.run("ty", "check", *session.posargs)
