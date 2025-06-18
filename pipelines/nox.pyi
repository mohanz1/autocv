import typing as _typing

import nox as _nox

session = _nox.session
Session = _nox.Session

def sync(
    session: _nox.Session,
    /,
    *,
    self: bool = False,
    extras: _typing.Sequence[str] = (),
    groups: _typing.Sequence[str] = (),
) -> None: ...
