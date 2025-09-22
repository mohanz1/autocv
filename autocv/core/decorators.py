"""Validation decorators for window handles and image buffers."""

from __future__ import annotations

__all__ = (
    "check_valid_hwnd",
    "check_valid_image",
)

import functools
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

from autocv.models import InvalidHandleError, InvalidImageError

if TYPE_CHECKING:
    from collections.abc import Callable

    from autocv.core import Vision, WindowCapture

P = ParamSpec("P")
R = TypeVar("R")
WindowCaptureT = TypeVar("WindowCaptureT", bound="WindowCapture")
VisionT = TypeVar("VisionT", bound="Vision")


def check_valid_hwnd(func: Callable[Concatenate[WindowCaptureT, P], R]) -> Callable[Concatenate[WindowCaptureT, P], R]:
    """Ensure the bound instance exposes a valid ``hwnd``.

    Args:
        func (Callable[Concatenate[WindowCaptureT, P], R]): Method being wrapped.

    Returns:
        Callable[Concatenate[WindowCaptureT, P], R]: Wrapped callable that raises
            :class:`InvalidHandleError` when ``self.hwnd`` equals ``-1``.

    Raises:
        InvalidHandleError: If ``self.hwnd`` has not been set on the instance.
    """

    @functools.wraps(func)
    def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
        if getattr(self, "hwnd", -1) == -1:
            raise InvalidHandleError(self.hwnd)
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[WindowCaptureT, P], R]", wrapper)


def check_valid_image(func: Callable[Concatenate[VisionT, P], R]) -> Callable[Concatenate[VisionT, P], R]:
    """Ensure the bound instance exposes a populated ``opencv_image`` buffer.

    Args:
        func (Callable[Concatenate[VisionT, P], R]): Method being wrapped.

    Returns:
        Callable[Concatenate[VisionT, P], R]: Wrapped callable that raises
            :class:`InvalidImageError` when ``self.opencv_image`` is empty.

    Raises:
        InvalidImageError: If the instance lacks an image or the array size is zero.
    """

    @functools.wraps(func)
    def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
        opencv_image = getattr(self, "opencv_image", None)
        if opencv_image is None or opencv_image.size == 0:
            raise InvalidImageError
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[VisionT, P], R]", wrapper)
