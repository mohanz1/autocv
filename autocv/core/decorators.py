"""Validation decorators for window handles and image buffers."""

from __future__ import annotations

__all__ = (
    "check_valid_hwnd",
    "check_valid_image",
)

import functools
import typing as t

from autocv.models import InvalidHandleError, InvalidImageError

if t.TYPE_CHECKING:
    from autocv.core import Vision, WindowCapture

P = t.ParamSpec("P")
R = t.TypeVar("R")
WindowCaptureT = t.TypeVar("WindowCaptureT", bound="WindowCapture")
VisionT = t.TypeVar("VisionT", bound="Vision")

WindowMethod = t.Callable[t.Concatenate[WindowCaptureT, P], R]
VisionMethod = t.Callable[t.Concatenate[VisionT, P], R]


if t.TYPE_CHECKING:

    def check_valid_hwnd(func: WindowMethod) -> WindowMethod:
        """Identity decorator hint for static type checkers."""
        return func

    def check_valid_image(func: VisionMethod) -> VisionMethod:
        """Identity decorator hint for static type checkers."""
        return func
else:

    def check_valid_hwnd(func: WindowMethod) -> WindowMethod:
        """Ensure the bound instance exposes a valid ``hwnd``."""

        @functools.wraps(func)
        def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
            if getattr(self, "hwnd", -1) == -1:
                raise InvalidHandleError(self.hwnd)
            return func(self, *args, **kwargs)

        return t.cast("WindowMethod", wrapper)

    def check_valid_image(func: VisionMethod) -> VisionMethod:
        """Ensure the bound instance exposes a populated ``opencv_image`` buffer."""

        @functools.wraps(func)
        def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
            opencv_image = getattr(self, "opencv_image", None)
            if opencv_image is None or opencv_image.size == 0:
                raise InvalidImageError
            return func(self, *args, **kwargs)

        return t.cast("VisionMethod", wrapper)
