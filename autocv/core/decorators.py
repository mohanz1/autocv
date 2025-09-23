"""Validation decorators for window handles and image buffers."""

from __future__ import annotations

__all__ = (
    "check_valid_hwnd",
    "check_valid_image",
)

import functools
import typing
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

from autocv.models import InvalidHandleError, InvalidImageError

if TYPE_CHECKING:
    from autocv.core import Vision, WindowCapture

P = ParamSpec("P")
R = TypeVar("R")
WindowCaptureT = TypeVar("WindowCaptureT", bound="WindowCapture")
VisionT = TypeVar("VisionT", bound="Vision")


if TYPE_CHECKING:

    def check_valid_hwnd(
        func: typing.Callable[Concatenate[WindowCaptureT, P], R],
    ) -> typing.Callable[Concatenate[WindowCaptureT, P], R]:
        """Identity decorator hint for static type checkers."""
        return func

    def check_valid_image(
        func: typing.Callable[Concatenate[VisionT, P], R],
    ) -> typing.Callable[Concatenate[VisionT, P], R]:
        """Identity decorator hint for static type checkers."""
        return func
else:

    def check_valid_hwnd(
        func: typing.Callable[Concatenate[WindowCaptureT, P], R],
    ) -> typing.Callable[Concatenate[WindowCaptureT, P], R]:
        """Ensure the bound instance exposes a valid ``hwnd``."""

        @functools.wraps(func)
        def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
            if getattr(self, "hwnd", -1) == -1:
                raise InvalidHandleError(self.hwnd)
            return func(self, *args, **kwargs)

        return cast("typing.Callable[Concatenate[WindowCaptureT, P], R]", wrapper)

    def check_valid_image(
        func: typing.Callable[Concatenate[VisionT, P], R],
    ) -> typing.Callable[Concatenate[VisionT, P], R]:
        """Ensure the bound instance exposes a populated ``opencv_image`` buffer."""

        @functools.wraps(func)
        def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
            opencv_image = getattr(self, "opencv_image", None)
            if opencv_image is None or opencv_image.size == 0:
                raise InvalidImageError
            return func(self, *args, **kwargs)

        return cast("typing.Callable[Concatenate[VisionT, P], R]", wrapper)
