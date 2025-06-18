"""Pyright integrations."""

from __future__ import annotations

from pipelines import nox


@nox.session()
def pyright(session: nox.Session) -> None:
    """Perform static type analysis on Python source code using pyright.

    At the time of writing this, this pipeline will not run successfully,
    as hikari does not have 100% compatibility with pyright just yet. This
    exists to make it easier to test and eventually reach that 100% compatibility.
    """
    nox.sync(session, self=True, groups=["pyright"])
    session.run("pyright")
