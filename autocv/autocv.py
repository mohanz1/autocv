"""High-level automation facade.

The :class:`~autocv.autocv.AutoCV` class composes lower-level primitives from
:mod:`autocv.core` into a single entry point for window capture, interactive
tools, and input automation.
"""

from __future__ import annotations

__all__ = ("AutoCV",)

import logging
import sys
from pathlib import Path
from tkinter import Tk
from typing import TYPE_CHECKING, Protocol, cast

import cv2 as cv
import win32gui
from typing_extensions import Self, override

from .color_picker import ColorPicker
from .core import Input, check_valid_hwnd, check_valid_image
from .image_filter import ImageFilter
from .image_picker import ImagePicker

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .models import FilterSettings


class _AntiGcpModule(Protocol):
    """Protocol describing the prebuilt ``antigcp`` extension module."""

    def antigcp(self, hwnd: int) -> bool:
        """Patch ``GetCursorPos`` for anti-GCP behavior."""
        ...


class AutoCV(Input):
    """Coordinate window capture, live inspection tools, and automation hooks."""

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initialize the automation facade.

        Args:
            hwnd: Window handle that receives capture and input. Defaults to ``-1``.

        Raises:
            FileNotFoundError: If a matching prebuilt extension directory is missing for the interpreter version.
            ImportError: If the ``antigcp`` prebuilt cannot be imported.
        """
        super().__init__(hwnd)

        version = sys.version_info
        pyd_dir = Path(__file__).parent / "prebuilt" / f"python{version.major}{version.minor}"

        if not pyd_dir.exists():
            msg = f"Missing prebuilt extension directory: {pyd_dir}"
            raise FileNotFoundError(msg)

        pyd_dir_str = str(pyd_dir)
        if not sys.path or sys.path[0] != pyd_dir_str:
            try:
                sys.path.remove(pyd_dir_str)
            except ValueError:
                pass
            sys.path.insert(0, pyd_dir_str)
        import antigcp  # noqa: PLC0415

        self._antigcp: _AntiGcpModule = cast("_AntiGcpModule", antigcp)
        self._instance_logger: logging.Logger = (
            logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))
        )

    @check_valid_hwnd
    def get_hwnd(self: Self) -> int:
        """Return the current target window handle."""
        return self.hwnd

    @override
    @check_valid_hwnd
    def get_window_size(self: Self, *, use_cache: bool = False) -> tuple[int, int]:
        """Return the client area dimensions for the active window."""
        if use_cache:
            return super().get_window_size(use_cache=use_cache)
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        return right - left, bottom - top

    def antigcp(self: Self) -> bool:
        """Install the GetCursorPos patch shipped with the prebuilt extension.

        Returns:
            bool: ``True`` when the patch succeeds, otherwise ``False``.
        """
        return self._antigcp.antigcp(self._get_topmost_hwnd())

    @check_valid_hwnd
    def image_picker(
        self: Self,
    ) -> tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
        """Launch the ROI picker overlay.

        Returns:
            tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
                Captured region and bounding rectangle, or ``(None, None)`` when cancelled.
        """
        self._instance_logger.debug("Setting up image picker.")
        root = Tk()
        try:
            app = ImagePicker(self.hwnd, root)
            root.mainloop()
            return app.result, app.rect
        finally:
            root.destroy()

    @check_valid_hwnd
    def color_picker(self: Self) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
        """Launch the pixel colour picker.

        Returns:
            tuple[tuple[int, int, int], tuple[int, int]] | None:
                Selected RGB colour with screen coordinates, or ``None``.
        """
        self._instance_logger.debug("Setting up color picker.")
        root = Tk()
        try:
            app = ColorPicker(self.hwnd, root)
            root.mainloop()
            return app.result
        finally:
            root.destroy()

    @check_valid_image
    def image_filter(self: Self) -> FilterSettings:
        """Return interactive filter settings derived from the current backbuffer."""
        return ImageFilter(self.opencv_image).filter_settings

    @check_valid_image
    def show_backbuffer(self: Self, *, live: bool = False) -> None:
        """Display the active backbuffer in an OpenCV window.

        Args:
            live: When ``True``, wait briefly (1ms) and then return.

                When ``False``, block until a key press (OpenCV ``waitKey(0)``).
        """
        self._instance_logger.debug("Showing backbuffer.")
        cv.imshow("AutoCV Backbuffer", self.opencv_image)
        cv.waitKey(1 if live else 0)
