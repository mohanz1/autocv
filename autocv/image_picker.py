"""This module defines the ImagePicker class which allows users to select a region of a window and take a screenshot.

It uses Tkinter for the GUI and Win32 APIs to interact with the window and capture the screenshot.
"""

from __future__ import annotations

__all__ = ("ImagePicker",)

from tkinter import BOTH
from tkinter import YES
from tkinter import Canvas
from tkinter import Event
from tkinter import Frame
from tkinter import Tk
from tkinter import Toplevel
from typing import TYPE_CHECKING
from typing import Any

import win32con
import win32gui
from typing_extensions import Self

from .core import Vision

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class ImagePicker:
    """Allows the user to select a region of a window and capture a screenshot of that region.

    This class creates an interactive overlay window using Tkinter, where the user can click and drag
    to select a region. Win32 APIs are used to bring the target window to the foreground and capture its screenshot.
    """

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Initializes the ImagePicker instance and sets up the GUI overlay for region selection.

        Args:
            hwnd (int): The window handle of the window to capture.
            master (Tk): The parent Tkinter window.
        """
        self.hwnd: int = hwnd
        self.master: Tk = master
        self.snip_surface: Canvas  # Initialized in create_screen_canvas
        self.start_x: int = -1
        self.start_y: int = -1
        self.current_x: int = -1
        self.current_y: int = -1
        self.result: npt.NDArray[np.uint8] | None = None
        self.rect: tuple[int, int, int, int] | None = None

        self.master.title("AutoCV Image Picker")
        self.master_screen = Toplevel(self.master)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", "maroon3")
        self.picture_frame = Frame(self.master_screen, background="maroon3")
        self.picture_frame.pack(fill=BOTH, expand=YES)

        self.create_screen_canvas()

    def create_screen_canvas(self: Self) -> None:
        """Creates the canvas and transparent overlay window for region selection.

        The method brings the target window to the foreground, creates a full-screen overlay
        with a semi-transparent canvas, and binds mouse events for region selection.
        """
        # Bring the target window to the foreground
        win32gui.ShowWindow(self.hwnd, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.hwnd)
        self.master_screen.deiconify()
        self.master.withdraw()

        # Create the canvas for drawing the selection rectangle
        self.snip_surface = Canvas(self.picture_frame, cursor="cross", bg="grey11")
        self.snip_surface.pack(fill=BOTH, expand=YES)

        # Bind mouse events
        self.snip_surface.bind("<ButtonPress-1>", self.on_button_press)
        self.snip_surface.bind("<B1-Motion>", self.on_snip_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self.on_button_release)

        # Position the overlay window over the target window
        x1, y1, x2, y2 = win32gui.GetWindowRect(self.hwnd)
        width = x2 - x1
        height = y2 - y1
        self.master_screen.geometry(f"{width + 2}x{height + 2}+{x1}+{y1}")
        self.master_screen.attributes("-alpha", 0.3)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)
        self.master_screen.overrideredirect(boolean=True)
        self.master_screen.focus_set()

    def on_button_press(self: Self, event: Event[Any]) -> None:
        """Handles the left mouse button press to record the starting coordinates of the selection.

        Args:
            event (Event): The Tkinter event containing the x and y coordinates.
        """
        self.start_x = event.x
        self.start_y = event.y
        # Create an initial rectangle (ID 1) with minimal size
        self.snip_surface.create_rectangle(0, 0, 1, 1, outline="red", width=3, fill="maroon3")

    def on_snip_drag(self: Self, event: Event[Any]) -> None:
        """Updates the selection rectangle as the mouse is dragged.

        Args:
            event (Event): The Tkinter event containing the current mouse coordinates.
        """
        self.current_x, self.current_y = event.x, event.y
        self.snip_surface.coords(1, self.start_x, self.start_y, self.current_x, self.current_y)

    def take_bounded_screenshot(self: Self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Captures a screenshot of the selected region of the window.

        A Vision instance is created to capture the current image of the window, and then the image
        is cropped to the specified region.

        Args:
            x1 (float): The starting x-coordinate of the region.
            y1 (float): The starting y-coordinate of the region.
            x2 (float): The ending x-coordinate of the region.
            y2 (float): The ending y-coordinate of the region.
        """
        vision = Vision(self.hwnd)
        vision.refresh()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self.result = vision.opencv_image[y1:y2, x1:x2]
        # Store full window dimensions as a fallback or for additional context.
        win_rect = win32gui.GetWindowRect(self.hwnd)
        self.rect = (win_rect[0], win_rect[1], win_rect[2] - win_rect[0], win_rect[3] - win_rect[1])

    def on_button_release(self: Self, _: Event[Canvas]) -> None:
        """Finalizes the region selection and captures the screenshot.

        Once the left mouse button is released, the selected region is determined from the recorded
        coordinates, a screenshot is taken, and the overlay window is closed.
        """
        x1 = min(self.start_x, self.current_x)
        y1 = min(self.start_y, self.current_y)
        x2 = max(self.start_x, self.current_x)
        y2 = max(self.start_y, self.current_y)
        self.take_bounded_screenshot(x1, y1, x2, y2)
        self.master_screen.destroy()
        self.master_screen.quit()
