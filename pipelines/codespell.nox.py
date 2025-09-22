from __future__ import annotations

from pipelines import config, nox

IGNORED_WORDS = ["amin", "arange"]


@nox.session()
def codespell(session: nox.Session) -> None:
    """Run codespell to check for spelling mistakes."""
    nox.sync(session, groups=["codespell"])
    session.run(
        "codespell",
        "--builtin",
        "clear,rare,code",
        "--ignore-words-list",
        ",".join(IGNORED_WORDS),
        *config.FULL_REFORMATTING_PATHS,
    )
