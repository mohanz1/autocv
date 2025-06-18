from __future__ import annotations

from pipelines import nox


@nox.session()
def ruff(session: nox.Session) -> None:
    """Run code linting using ruff."""
    nox.sync(session, self=True, groups=["ruff"])

    session.run("ruff", "check", *session.posargs)
