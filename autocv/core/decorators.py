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


if t.TYPE_CHECKING:

    def check_valid_hwnd(
        func: t.Callable[t.Concatenate[WindowCaptureT, P], R],
    ) -> t.Callable[t.Concatenate[WindowCaptureT, P], R]:
        """Identity decorator hint for static type checkers."""
        return func

    def check_valid_image(
        func: t.Callable[t.Concatenate[VisionT, P], R],
    ) -> t.Callable[t.Concatenate[VisionT, P], R]:
        """Identity decorator hint for static type checkers."""
        return func
else:

    def check_valid_hwnd(
        func: t.Callable[t.Concatenate[WindowCaptureT, P], R],
    ) -> t.Callable[t.Concatenate[WindowCaptureT, P], R]:
        """Ensure the bound instance exposes a valid ``hwnd``."""

        @functools.wraps(func)
        def wrapper(self: WindowCaptureT, *args: P.args, **kwargs: P.kwargs) -> R:
            if getattr(self, "hwnd", -1) == -1:
                raise InvalidHandleError(self.hwnd)
            return func(self, *args, **kwargs)

        return t.cast(
            "t.Callable[t.Concatenate[WindowCaptureT, P], R]",
            wrapper,
        )

    def check_valid_image(
        func: t.Callable[t.Concatenate[VisionT, P], R],
    ) -> t.Callable[t.Concatenate[VisionT, P], R]:
        """Ensure the bound instance exposes a populated ``opencv_image`` buffer."""

        @functools.wraps(func)
        def wrapper(self: VisionT, *args: P.args, **kwargs: P.kwargs) -> R:
            opencv_image = getattr(self, "opencv_image", None)
            if opencv_image is None or opencv_image.size == 0:
                raise InvalidImageError
            return func(self, *args, **kwargs)

        return t.cast(
            "t.Callable[t.Concatenate[VisionT, P], R]",
            wrapper,
        )
