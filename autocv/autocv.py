"""High-level AutoCV facade that wires capture, inspection interfaces, and automation helpers."""

from __future__ import annotations

__all__ = ("AutoCV",)

import ctypes
import importlib
import logging
import platform
import sys
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from tkinter import Tk
from typing import TYPE_CHECKING

import cv2 as cv
import win32gui

from .color_picker import ColorPicker
from .core import Input, check_valid_hwnd, check_valid_image
from .image_filter import ImageFilter
from .image_picker import ImagePicker

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .models import FilterSettings


LOGGER = logging.getLogger(__name__)
_SUPPORTED_SHCORE_RELEASES = {"10", "11"}
_PREBUILT_DIR_FMT = "python{major}{minor}"


def _configure_dpi_awareness() -> None:
    """Apply the best DPI awareness available on the current platform."""
    if platform.system() != "Windows":
        LOGGER.debug("Skipping DPI awareness setup on %s", platform.system())
        return

    release = platform.release()
    try:
        if release in _SUPPORTED_SHCORE_RELEASES:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        else:
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError) as exc:
        LOGGER.warning("Unable to configure DPI awareness for release %s: %s", release, exc)


@contextmanager
def _tk_root(title: str) -> AbstractContextManager[Tk]:
    """Yield a Tk root window and ensure it is torn down afterwards."""
    root = Tk()
    root.title(title)
    try:
        yield root
    finally:
        root.destroy()


def _load_antigcp_module() -> object:
    """Load the platform-specific ``antigcp`` extension."""
    version = sys.version_info
    prebuilt_dir = (
        Path(__file__).parent / "prebuilt" / _PREBUILT_DIR_FMT.format(major=version.major, minor=version.minor)
    )
    if not prebuilt_dir.exists():
        msg = f"No antigcp prebuilt found for CPython {version.major}.{version.minor}."
        raise FileNotFoundError(msg)

    path_str = str(prebuilt_dir)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        module = importlib.import_module("antigcp")
    except ImportError as exc:
        msg = "Failed to import antigcp prebuilt extension"
        raise ImportError(msg) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass
    return module


class AutoCV(Input):
    """Coordinate window capture, live inspection tools, and automation hooks."""

    def __init__(self, hwnd: int = -1) -> None:
        """Initialise the automation facade for ``hwnd``."""
        super().__init__(hwnd)
        self._logger = LOGGER.getChild(self.__class__.__name__).getChild(str(id(self)))
        self._antigcp = _load_antigcp_module()

    @check_valid_hwnd
    def get_hwnd(self) -> int:
        """Return the active window handle."""
        return self.hwnd

    @check_valid_hwnd
    def get_window_size(self) -> tuple[int, int]:
        """Return the client area dimensions for the active window."""
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        return right - left, bottom - top

    def antigcp(self) -> bool:
        """Install the GetCursorPos hook supplied by the prebuilt extension."""
        return bool(self._antigcp.antigcp(self._get_topmost_hwnd()))

    @check_valid_hwnd
    def image_picker(
        self,
    ) -> tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
        """Launch a region-of-interest picker and return the captured data."""
        with _tk_root("AutoCV Image Picker") as root:
            app = ImagePicker(self.hwnd, root)
            root.mainloop()
            return app.result, app.rect

    @check_valid_hwnd
    def color_picker(self) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
        """Launch the pixel colour picker overlay."""
        with _tk_root("AutoCV Color Picker") as root:
            app = ColorPicker(self.hwnd, root)
            root.mainloop()
            return app.result

    @check_valid_image
    def image_filter(self) -> FilterSettings:
        """Return interactive filter settings derived from the backbuffer."""
        return ImageFilter(self.opencv_image).filter_settings

    @check_valid_image
    def show_backbuffer(self, *, live: bool = False) -> None:
        """Display the active backbuffer in an OpenCV window."""
        self._logger.debug("Displaying backbuffer (live=%s)", live)
        cv.imshow("AutoCV Backbuffer", self.opencv_image)
        cv.waitKey(int(live))


_configure_dpi_awareness()
