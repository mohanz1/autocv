"""This module defines the ImagePicker class which allows users to select a region of a window and take a screenshot.

It uses Tkinter for the GUI and Win32 APIs to interact with the window and capture the screenshot.
"""

from __future__ import annotations

__all__ = ("ImagePicker",)

from tkinter import BOTH, YES, Canvas, Event, Frame, Tk, Toplevel
from typing import TYPE_CHECKING, Any

import win32con
import win32gui
from typing_extensions import Self

from .core import Vision

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class ImagePicker:
    """A class for selecting a region of a window and taking a screenshot of that region.

    It provides an interactive overlay window for users to select the desired area. The class uses Tkinter to create the
    GUI and the Win32 API to manage window operations and capture screenshots.
    """

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """A class for taking a screenshot of a selected region of a window.

        Args:
            hwnd (int): The window handle of the window to take the screenshot of.
            master (Tk): The parent tkinter window of the ImagePicker.
        """
        self.hwnd: int = hwnd
        self.master: Tk = master
        self.snip_surface: Canvas
        self.start_x = -1
        self.start_y = -1
        self.current_x = -1
        self.current_y = -1
        self.result: npt.NDArray[np.uint8] | None = None
        self.rect: Rectangle | None = None

        self.master.title("AutoCV Image Picker")
        self.master_screen = Toplevel(self.master)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", "maroon3")
        self.picture_frame = Frame(self.master_screen, background="maroon3")
        self.picture_frame.pack(fill=BOTH, expand=YES)

        self.create_screen_canvas()

    def create_screen_canvas(self: Self) -> None:
        """Creates the tkinter canvas and the transparent overlay window."""
        win32gui.ShowWindow(self.hwnd, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.hwnd)
        self.master_screen.deiconify()
        self.master.withdraw()

        self.snip_surface = Canvas(self.picture_frame, cursor="cross", bg="grey11")
        self.snip_surface.pack(fill=BOTH, expand=YES)

        self.snip_surface.bind("<ButtonPress-1>", self.on_button_press)
        self.snip_surface.bind("<B1-Motion>", self.on_snip_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self.on_button_release)

        coords = Rectangle.from_coordinates(win32gui.GetWindowRect(self.hwnd))
        self.master_screen.geometry(f"{coords.width + 2}x{coords.height + 2}+{coords.left}+{coords.top}")
        self.master_screen.attributes("-alpha", 0.3)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)  # noqa: FBT003
        self.master_screen.overrideredirect(boolean=True)
        self.master_screen.focus_set()

    def on_button_press(self: Self, event: Event) -> None:  # type: ignore[type-arg]
        """Records the starting position of the mouse when the left mouse button is pressed down."""
        self.start_x = event.x
        self.start_y = event.y
        self.snip_surface.create_rectangle(0, 0, 1, 1, outline="red", width=3, fill="maroon3")

    def on_snip_drag(self: Self, event: Event) -> None:  # type: ignore[type-arg]
        """Updates the position of the rectangle as the mouse is dragged."""
        self.current_x, self.current_y = (event.x, event.y)
        self.snip_surface.coords(1, self.start_x, self.start_y, self.current_x, self.current_y)

    def take_bounded_screenshot(self: Self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Takes a screenshot of the selected region of the window.

        Args:
            x1 (float): The starting x position of the selected region.
            y1 (float): The starting y position of the selected region.
            x2 (float): The ending x position of the selected region.
            y2 (float): The ending y position of the selected region.
        """
        # Capture screenshot and crop to selected region
        vision = Vision(self.hwnd)
        vision.refresh()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self.result = vision.opencv_image[y1:y2, x1:x2]
        self.rect = Rectangle.from_coordinates((x1, y1, x2, y2))

    def on_button_release(self: Self, _: Any) -> None:
        """Handles the release of the screenshot button by taking a screenshot of the selected region."""
        # Exit screenshot mode and destroy the screenshot window

        # Take a screenshot of the selected region
        self.take_bounded_screenshot(
            min(self.start_x, self.current_x),
            min(self.start_y, self.current_y),
            max(self.start_x, self.current_x),
            max(self.start_y, self.current_y),
        )
        self.master_screen.destroy()
        self.master_screen.quit()
