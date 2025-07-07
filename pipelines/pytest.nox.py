"""Pytest integration."""

from __future__ import annotations

import typing

from pipelines import config
from pipelines import nox

RUN_FLAGS = ["-c", config.PYPROJECT_TOML, "--showlocals"]
COVERAGE_FLAGS = [
    "--cov",
    config.MAIN_PACKAGE,
    "--cov-config",
    config.PYPROJECT_TOML,
    "--cov-report",
    "term",
    "--cov-report",
    f"html:{config.COVERAGE_HTML_PATH}",
    "--cov-report",
    "xml",
]


@nox.session()
def pytest(session: nox.Session) -> None:
    """Run unit tests and measure code coverage.

    Coverage can be disabled with the `--skip-coverage` flag.
    """
    _pytest(session)


def _pytest(
    session: nox.Session, *, extras_install: typing.Sequence[str] = (), python_flags: typing.Sequence[str] = ()
) -> None:
    nox.sync(session, self=True, extras=extras_install, groups=["pytest"])

    flags = RUN_FLAGS

    if "--coverage" in session.posargs:
        session.posargs.remove("--coverage")
        flags.extend(COVERAGE_FLAGS)

    session.run("python", *python_flags, "-m", "pytest", *flags, *session.posargs, config.TEST_PACKAGE)
