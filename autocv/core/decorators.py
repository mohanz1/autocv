"""This module provides decorators that validate certain attributes on a class instance.

Decorators:
    - check_valid_hwnd: Ensures an object has a non-negative `hwnd` attribute.
    - check_valid_image: Ensures an object has a non-empty `opencv_image` attribute.
"""

from __future__ import annotations

__all__ = ("check_valid_hwnd", "check_valid_image")

import functools
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

from autocv.core import Vision, WindowCapture
from autocv.models import InvalidHandleError, InvalidImageError

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")
WindowCaptureT = TypeVar("WindowCaptureT", bound=WindowCapture)
VisionT = TypeVar("VisionT", bound=Vision)


def check_valid_hwnd(func: Callable[Concatenate[WindowCaptureT, P], R]) -> Callable[Concatenate[WindowCaptureT, P], R]:
    """Decorator that checks if `self.hwnd` exists and is not -1.

    Raises:
        InvalidHandleError: If `self.hwnd` is not found or equals -1.
    """

    @functools.wraps(func)
    def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
        # Just check if 'hwnd' exists and is not -1
        if not hasattr(self, "hwnd") or self.hwnd == -1:
            raise InvalidHandleError(self.hwnd)
        return func(self, *args, **kwargs)

    wrapped = functools.wraps(func)(wrapper)
    # Cast the wrapped function to the correct type so Mypy stops complaining
    return cast("Callable[Concatenate[WindowCaptureT, P], R]", wrapped)


def check_valid_image(func: Callable[Concatenate[VisionT, P], R]) -> Callable[Concatenate[VisionT, P], R]:
    """Decorator that checks if `self.opencv_image` exists and is non-empty.

    Raises:
       InvalidImageError: If `self.opencv_image` is not found or has size 0.
    """

    @functools.wraps(func)
    def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
        # Just check if 'opencv_image' exists and is non-empty
        if not hasattr(self, "opencv_image") or self.opencv_image.size == 0:
            raise InvalidImageError
        return func(self, *args, **kwargs)

    wrapped = functools.wraps(func)(wrapper)
    # Cast the wrapped function to the correct type so Mypy stops complaining
    return cast("Callable[Concatenate[VisionT, P], R]", wrapped)
