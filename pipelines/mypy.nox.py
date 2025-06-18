from __future__ import annotations

from pipelines import config
from pipelines import nox


@nox.session()
def mypy(session: nox.Session) -> None:
    """Perform static type analysis on Python source code using mypy."""
    nox.sync(session, self=True, groups=["mypy"])

    session.run("mypy", "-p", config.MAIN_PACKAGE, "--config", config.PYPROJECT_TOML)
