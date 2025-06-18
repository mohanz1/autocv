"""The `autocv` module provides a comprehensive interface for automation and computer vision tasks.

This module contains the AutoCV class, which extends the functionality of the Input class from the core module.
It provides high-level methods for screen capture, image processing, color picking, and window manipulation,
making it an all-encompassing tool for building computer vision and GUI automation applications.

Classes:
    AutoCV: Inherits from Input, providing methods for window and image manipulation, OCR, and user input simulation.
"""

from __future__ import annotations

__all__ = ("AutoCV",)

import logging
import sys
import typing
from pathlib import Path
from tkinter import Tk
from typing import TYPE_CHECKING

import cv2 as cv
import win32gui
from typing_extensions import Self

from .color_picker import ColorPicker
from .core import Input
from .core import check_valid_hwnd
from .core import check_valid_image
from .image_filter import ImageFilter
from .image_picker import ImagePicker

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .models import FilterSettings


class AutoCV(Input):
    """Provides an interface for interacting with windows and images on a computer screen.

    AutoCV uses OpenCV and Tesseract OCR to capture and process images, perform color picking,
    and simulate user input. It offers methods for window manipulation, screen capture,
    and image analysis.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes an AutoCV instance with the specified window handle.

        Args:
            hwnd (int): The window handle to use for the AutoCV instance. Defaults to -1.
        """
        super().__init__(hwnd)

        # Determine the appropriate directory based on the current Python version.
        version = sys.version_info
        pyd_dir = Path(__file__).parent / "prebuilt" / f"python{version.major}{version.minor}"

        # Append the directory to sys.path if it exists, and attempt to import antigcp.
        if pyd_dir.exists():
            sys.path.insert(0, str(pyd_dir))
            try:
                import antigcp  # type: ignore[import-not-found] # noqa: PLC0415

                self._antigcp = antigcp
            except ImportError as e:
                raise ImportError from e
        else:
            raise FileNotFoundError

        # Create a dedicated logger for this instance.
        self._instance_logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))

    @check_valid_hwnd
    def get_hwnd(self: Self) -> int:
        """Returns the current window handle.

        Returns:
            int: The window handle.
        """
        return self.hwnd

    @check_valid_hwnd
    def get_window_size(self: Self) -> tuple[int, int]:
        """Retrieves the width and height of the current window.

        Returns:
            tuple[int, int]: A tuple (width, height) in pixels.
        """
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        return right - left, bottom - top

    def antigcp(self: Self) -> bool:
        """Patches the target process by replacing its GetCursorPos function.

        Returns:
            bool: True if successfully patched; otherwise, False.
        """
        return typing.cast("bool", self._antigcp.antigcp(self._get_topmost_hwnd()))

    @check_valid_hwnd
    def image_picker(self: Self) -> tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
        """Launches the image picker interface for region selection.

        Returns:
            tuple[npt.NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
                A tuple containing the selected image as a NumPy array and its rectangle (x, y, width, height),
                or (None, None) if no image was selected.
        """
        self._instance_logger.debug("Setting up image picker.")
        root = Tk()
        app = ImagePicker(self.hwnd, root)
        root.mainloop()
        image = app.result
        rect = app.rect
        root.destroy()
        return image, rect

    @check_valid_hwnd
    def color_picker(self: Self) -> tuple[tuple[int, int, int], tuple[int, int]] | None:
        """Launches the color picker interface for pixel color selection.

        Returns:
            tuple[tuple[int, int, int], tuple[int, int]] | None:
                A tuple containing the selected color (R, G, B) and its screen coordinates,
                or None if no color was selected.
        """
        self._instance_logger.debug("Setting up color picker.")
        root = Tk()
        app = ColorPicker(self.hwnd, root)
        root.mainloop()
        color_with_point = app.result
        root.destroy()
        return color_with_point

    @check_valid_image
    def image_filter(self: Self) -> FilterSettings:
        """Applies image filtering operations to the current backbuffer and returns the filter settings.

        Returns:
            FilterSettings: The current filter settings after applying the image filter.
        """
        # Instantiate ImageFilter using the current backbuffer image.
        return ImageFilter(self.opencv_image).filter_settings

    @check_valid_image
    def show_backbuffer(self: Self, *, live: bool = False) -> None:
        """Displays the current backbuffer image in a window.

        Args:
            live (bool, optional): If True, shows a live refreshing view. Defaults to False.
        """
        self._instance_logger.debug("Showing backbuffer.")
        cv.imshow("AutoCV Backbuffer", self.opencv_image)
        cv.waitKey(int(live))
