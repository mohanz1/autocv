"""Validation decorators for window handles and image buffers.

These helpers guard methods that require a valid Win32 window handle or a
populated OpenCV image buffer, raising domain-specific exceptions early when
preconditions are not met.
"""

from __future__ import annotations

__all__ = (
    "check_valid_hwnd",
    "check_valid_image",
)

import functools
from typing import TYPE_CHECKING, Concatenate, Final, ParamSpec, Protocol, TypeVar, cast

from autocv.models import InvalidHandleError, InvalidImageError

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt


class _HasHandle(Protocol):
    """Protocol for objects exposing a Win32 window handle."""

    hwnd: int


class _HasImageBuffer(Protocol):
    """Protocol for objects exposing an OpenCV-compatible image buffer."""

    opencv_image: npt.NDArray[np.uint8]


P = ParamSpec("P")
R = TypeVar("R")
HandleOwner = TypeVar("HandleOwner", bound=_HasHandle)
ImageOwner = TypeVar("ImageOwner", bound=_HasImageBuffer)

_UNSET_HANDLE: Final[int] = -1


def _validate_hwnd(instance: _HasHandle) -> int:
    """Return a valid window handle or raise ``InvalidHandleError``.

    Args:
        instance: Object that exposes a ``hwnd`` attribute or ``_ensure_hwnd`` helper.

    Returns:
        int: Validated window handle.

    Raises:
        InvalidHandleError: If the handle is unset or cannot be coerced to an ``int``.
    """
    ensure_hwnd_candidate = getattr(instance, "_ensure_hwnd", None)
    if callable(ensure_hwnd_candidate):
        ensure_hwnd = cast("Callable[[], int]", ensure_hwnd_candidate)
        return int(ensure_hwnd())

    hwnd = getattr(instance, "hwnd", _UNSET_HANDLE)
    if hwnd == _UNSET_HANDLE:
        raise InvalidHandleError(hwnd)
    try:
        return int(hwnd)
    except (TypeError, ValueError) as exc:
        raise InvalidHandleError(_UNSET_HANDLE) from exc


def _validate_image(instance: _HasImageBuffer) -> npt.NDArray[np.uint8]:
    """Return the ``opencv_image`` buffer or raise ``InvalidImageError`` when empty.

    Args:
        instance: Object that exposes an ``opencv_image`` NumPy buffer.

    Returns:
        npt.NDArray[np.uint8]: Non-empty image buffer.

    Raises:
        InvalidImageError: If the buffer is missing or empty.
    """
    opencv_image = getattr(instance, "opencv_image", None)
    if opencv_image is None or getattr(opencv_image, "size", 0) == 0:
        raise InvalidImageError
    return cast("npt.NDArray[np.uint8]", opencv_image)


def check_valid_hwnd(
    func: Callable[Concatenate[HandleOwner, P], R],
) -> Callable[Concatenate[HandleOwner, P], R]:
    """Ensure the bound instance exposes a valid ``hwnd`` before calling.

    Args:
        func: Method that expects a valid window handle on ``self``.

    Returns:
        Callable[..., R]: Wrapped callable that validates ``hwnd`` first.
    """

    @functools.wraps(func)
    def wrapper(self: HandleOwner, *args: P.args, **kwargs: P.kwargs) -> R:
        _validate_hwnd(self)
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[HandleOwner, P], R]", wrapper)


def check_valid_image(
    func: Callable[Concatenate[ImageOwner, P], R],
) -> Callable[Concatenate[ImageOwner, P], R]:
    """Ensure the bound instance contains a populated image buffer before calling.

    Args:
        func: Method that expects a non-empty ``opencv_image`` on ``self``.

    Returns:
        Callable[..., R]: Wrapped callable that validates ``opencv_image`` first.
    """

    @functools.wraps(func)
    def wrapper(self: ImageOwner, *args: P.args, **kwargs: P.kwargs) -> R:
        _validate_image(self)
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[ImageOwner, P], R]", wrapper)
