"""High-level automation facade.

The :class:`~autocv.autocv.AutoCV` class composes lower-level primitives from
:mod:`autocv.core` into a single entry point for window capture, interactive
tools, and input automation.
"""

from __future__ import annotations

__all__ = ("AutoCV",)

import logging
import sys
from importlib import import_module
from pathlib import Path
from tkinter import Tk
from typing import TYPE_CHECKING, Protocol, Self, cast

import numpy as np
import win32gui
from typing_extensions import override

import cv2 as cv

from .color_picker import ColorPicker
from .core import Input, check_valid_hwnd, check_valid_image
from .image_filter import ImageFilter
from .image_picker import ImagePicker, ImagePickerCapture
from .models import InvalidHandleError

if TYPE_CHECKING:
    import numpy.typing as npt

    from .models import FilterSettings


class _AntiGcpModule(Protocol):
    """Protocol describing the prebuilt ``antigcp`` extension module."""

    def antigcp(self, hwnd: int) -> bool:
        """Patch ``GetCursorPos`` for anti-GCP behavior."""
        ...


def _prebuilt_dir_for_runtime() -> Path:
    """Return the prebuilt extension directory for the active Python runtime."""
    version = sys.version_info
    return Path(__file__).parent / "prebuilt" / f"python{version.major}{version.minor}"


def _prioritize_sys_path(path: Path) -> None:
    """Move ``path`` to the front of ``sys.path`` while removing duplicates."""
    path_str = str(path)
    sys.path = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _load_antigcp_module(prebuilt_dir: Path) -> _AntiGcpModule:
    """Load the bundled ``antigcp`` module from ``prebuilt_dir``."""
    _prioritize_sys_path(prebuilt_dir)
    expected_parent = prebuilt_dir.resolve()
    cached_module = sys.modules.get("antigcp")
    if cached_module is not None:
        cached_file = getattr(cached_module, "__file__", None)
        if cached_file is not None and Path(cached_file).resolve().parent == expected_parent:
            return cast("_AntiGcpModule", cached_module)
        sys.modules.pop("antigcp", None)

    return cast("_AntiGcpModule", import_module("antigcp"))


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

        pyd_dir = _prebuilt_dir_for_runtime()
        if not pyd_dir.exists():
            msg = f"Missing prebuilt extension directory: {pyd_dir}"
            raise FileNotFoundError(msg)

        self._antigcp = _load_antigcp_module(pyd_dir)
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
        """Return the current window dimensions."""
        if use_cache:
            return super().get_window_size(use_cache=use_cache)
        return self._bounds_to_size(self._fetch_window_bounds(self.hwnd))

    @staticmethod
    def _fetch_window_bounds(hwnd: int) -> tuple[int, int, int, int]:
        """Fetch absolute window bounds from Win32 for the active attachment."""
        try:
            return win32gui.GetWindowRect(hwnd)
        except win32gui.error as exc:
            raise InvalidHandleError(hwnd) from exc

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
                Captured region and full target-window bounds, or ``(None, None)`` when cancelled.

        Note:
            This method preserves the historical ``(image, rect)`` contract, where ``rect`` is the
            full window bounds. Use :meth:`image_picker_capture` for explicit ROI bounds.
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
    def image_picker_capture(self: Self) -> ImagePickerCapture:
        """Launch the ROI picker overlay and return structured capture metadata."""
        self._instance_logger.debug("Setting up structured image picker.")
        root = Tk()
        try:
            app = ImagePicker(self.hwnd, root)
            root.mainloop()
            return app.capture
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
        image = self.opencv_image
        if isinstance(image, np.ndarray):
            image = self._require_color_image(image, caller="image_filter")
        return ImageFilter(image).filter_settings

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
