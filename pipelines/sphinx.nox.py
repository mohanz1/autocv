"""Documentation pages generation."""

from __future__ import annotations

from pipelines import config, nox


@nox.session()
def sphinx(session: nox.Session) -> None:
    """Generate docs using sphinx."""
    nox.sync(session, self=True, groups=["sphinx", "ruff"])

    session.run("sphinx-build", "-W", config.DOCUMENTATION_DIRECTORY, config.DOCUMENTATION_OUTPUT_PATH)
