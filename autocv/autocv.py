"""The `autocv` module provides a comprehensive interface for automation and computer vision tasks.

This module contains the AutoCV class which extends the functionality of the `Input` class from the `core` module. It
provides high-level methods for screen capture, image processing, color picking, and window manipulation. The module is
designed to be an all-encompassing tool for building computer vision and GUI automation applications.

Classes:
- AutoCV: Inherits from Input, providing methods for finding and manipulating windows, capturing and processing images,
performing OCR, and simulating user input.
"""

from __future__ import annotations

__all__ = ("AutoCV",)

import logging
from tkinter import Tk
from typing import TYPE_CHECKING

import cv2 as cv
import win32api
import win32con
import win32gui
import win32process
from typing_extensions import Self

from .color_picker import ColorPicker
from .core import Input
from .core.vision import check_valid_hwnd, check_valid_image
from .image_filter import ImageFilter
from .image_picker import ImagePicker

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .models import ColorWithPoint, FilterSettings, Rectangle


class AutoCV(Input):
    """AutoCV is a class that provides an interface for interacting with windows and images on a computer screen.

    It uses OpenCV and Tesseract OCR to manipulate the images and extract information. This class provides methods for
    finding windows by title, setting the current window and inner window by title, getting all visible windows, getting
    all child windows of the current window, setting an image as the current image buffer, refreshing the image of the
    current window, and extracting text and color information from the current image buffer.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes an instance of the AutoCV class with a specified window handle (hwnd).

        Args:
            hwnd (Optional[int]): The window handle to use for the AutoCV instance. Defaults to None.
        """
        super().__init__(hwnd)
        self._instance_logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))

    @check_valid_hwnd
    def get_hwnd(self: Self) -> int:
        """Gets the handle for the specified window.

        Returns:
            int: The handle for the specified window.
        """
        return self.hwnd

    @check_valid_hwnd
    def get_window_size(self: Self) -> tuple[int, int]:
        """Retrieves the size of the window.

        Returns:
            A tuple representing the width and height of the window in pixels.
        """
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top
        return width, height

    def antigcp(self: Self) -> bool:
        """Replaces the `GetCursorPos` function in the `user32.dll` module of the target process.

        Returns:
            True if the function was successfully patched, False otherwise.
        """
        # Get the process ID of the target process.
        process_id = win32process.GetWindowThreadProcessId(self._get_topmost_hwnd())[1]

        # Open a handle to the target process.
        process_handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, bInherit=False, pid=process_id)  # type: ignore[call-arg]

        module_handle = win32api.GetModuleHandle("user32.dll")
        function_address = win32api.GetProcAddress(module_handle, "GetCursorPos")  # type: ignore[arg-type]

        # Define the bytes to write to the process memory to replace the `GetCursorPos` function.
        new_bytes = b"\x33\xc0\xc3"  # xor eax, eax; ret;

        # Write the new function bytes to the process memory.
        bytes_written = win32process.WriteProcessMemory(process_handle, function_address, new_bytes)  # type: ignore[no-untyped-call]

        # Read the process memory to verify that the new function was written correctly.
        bytes_read = win32process.ReadProcessMemory(process_handle, function_address, len(new_bytes))  # type: ignore[no-untyped-call]
        buffer = memoryview(bytes_read)

        # Close the handle to the target process.
        win32api.CloseHandle(process_handle)

        # Check if the new function was written correctly.
        return bytes_written == len(new_bytes) and buffer.tobytes() == new_bytes

    @check_valid_hwnd
    def image_picker(self: Self) -> tuple[npt.NDArray[np.uint8] | None, Rectangle | None]:
        """Sets up an image picker interface and returns the selected image as a NumPy array.

        Returns:
            Optional[np.ndarray]: The selected image as a NumPy array, or None if no image was selected.
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
    def color_picker(self: Self) -> ColorWithPoint | None:
        """Sets up a color picker interface and returns the selected color and its location.

        Returns:
            ColorWithPoint: The selected color and its location.
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
        """A class for applying an HSV filter, Canny edge detection, erode, and dilate operations to an image."""
        return ImageFilter(self.opencv_image).filter_settings

    @check_valid_image
    def show_backbuffer(self: Self, *, live: bool = False) -> None:
        """Displays the backbuffer image of the window.

        Args:
            live (bool): Whether to show a live refreshing view. Defaults to False.
        """
        self._instance_logger.debug("Showing backbuffer.")
        cv.imshow("AutoCV Backbuffer", self.opencv_image)
        cv.waitKey(int(live))
