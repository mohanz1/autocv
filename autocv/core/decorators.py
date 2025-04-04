"""This module provides decorators that validate certain attributes on a class instance.

Decorators:
  - check_valid_hwnd: Ensures an object has a valid (non-negative) `hwnd` attribute.
  - check_valid_image: Ensures an object has a non-empty `opencv_image` attribute.
"""

from __future__ import annotations

__all__ = ("check_valid_hwnd", "check_valid_image")

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
    """Decorator to ensure the instance has a valid window handle.

    This decorator checks that the `hwnd` attribute exists on the instance and is not -1.

    Raises:
        InvalidHandleError: If `self.hwnd` is missing or equals -1.

    Returns:
        The wrapped function if the handle is valid.
    """

    @functools.wraps(func)
    def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
        if getattr(self, "hwnd", -1) == -1:
            raise InvalidHandleError(self.hwnd)
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[WindowCaptureT, P], R]", wrapper)


def check_valid_image(func: Callable[Concatenate[VisionT, P], R]) -> Callable[Concatenate[VisionT, P], R]:
    """Decorator to ensure the instance has a valid image.

    This decorator checks that the `opencv_image` attribute exists on the instance and is non-empty.

    Raises:
        InvalidImageError: If `self.opencv_image` is missing or has zero size.

    Returns:
        The wrapped function if the image is valid.
    """

    @functools.wraps(func)
    def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
        opencv_image = getattr(self, "opencv_image", None)
        if opencv_image is None or opencv_image.size == 0:
            raise InvalidImageError
        return func(self, *args, **kwargs)

    return cast("Callable[Concatenate[VisionT, P], R]", wrapper)
