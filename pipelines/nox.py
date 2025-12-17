from __future__ import annotations

import typing

import nox

NoxCallbackSigT = typing.Callable[[nox.Session], None]

# Default sessions should be defined here
nox.options.sessions = ["reformat-code", "codespell", "pytest", "ruff", "mypy"]
nox.options.default_venv_backend = "uv"


def session(**kwargs: typing.Any) -> typing.Callable[[NoxCallbackSigT], NoxCallbackSigT]:  # noqa: ANN401
    """Session wrapper to give default job kwargs."""

    def decorator(func: NoxCallbackSigT) -> NoxCallbackSigT:
        name = func.__name__.replace("_", "-")
        reuse_venv = kwargs.pop("reuse_venv", True)
        return nox.session(name=name, reuse_venv=reuse_venv, **kwargs)(func)

    return decorator


def sync(
    session: nox.Session, /, *, self: bool = False, extras: typing.Sequence[str] = (), groups: typing.Sequence[str] = ()
) -> None:
    """Install session packages using `uv sync`."""
    if extras and not self:
        msg = "When specifying extras, set `self=True`."
        raise RuntimeError(msg)

    args: list[str] = []
    for extra in extras:
        args.extend(("--extra", extra))

    group_flag = "--group" if self else "--only-group"
    for group in groups:
        args.extend((group_flag, group))

    if not self and groups:
        args.append("--no-install-project")

    session.run_install(
        "uv", "sync", "--locked", *args, silent=True, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    )
