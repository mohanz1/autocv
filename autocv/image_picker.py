"""Tkinter overlay for selecting and capturing AutoCV window regions."""

from __future__ import annotations

__all__ = ("ImagePicker",)

import tkinter as tk
from tkinter import BOTH, YES, Canvas, Event, Frame, Tk, Toplevel
from typing import TYPE_CHECKING

import win32con
import win32gui
from typing_extensions import Self

from .core import Vision

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

OVERLAY_BACKGROUND = "maroon3"
OVERLAY_ALPHA = 0.3
SELECTION_OUTLINE = "red"
SELECTION_WIDTH = 3


class ImagePicker:
    """Interactive overlay for selecting and capturing a region of a window."""

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Create the overlay and prepare for region selection on ``hwnd``."""
        self.hwnd = hwnd
        self.master = master
        self.master_screen = Toplevel(master)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", OVERLAY_BACKGROUND)
        self.picture_frame = Frame(self.master_screen, background=OVERLAY_BACKGROUND)
        self.picture_frame.pack(fill=BOTH, expand=YES)

        self.snip_surface: Canvas
        self.start_x = -1
        self.start_y = -1
        self.current_x = -1
        self.current_y = -1
        self._rect_id: int | None = None
        self.result: npt.NDArray[np.uint8] | None = None
        self.rect: tuple[int, int, int, int] | None = None

        self.master.title("AutoCV Image Picker")
        self.master.protocol("WM_DELETE_WINDOW", self._close)
        self.master_screen.protocol("WM_DELETE_WINDOW", self._close)

        self.create_screen_canvas()

    def create_screen_canvas(self: Self) -> None:
        """Bring the target window forward and display the transparent overlay."""
        win32gui.ShowWindow(self.hwnd, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.hwnd)

        self.master_screen.deiconify()
        self.master.withdraw()

        self.snip_surface = Canvas(self.picture_frame, cursor="cross", bg="grey11", highlightthickness=0)
        self.snip_surface.pack(fill=BOTH, expand=YES)

        self.snip_surface.bind("<ButtonPress-1>", self.on_button_press)
        self.snip_surface.bind("<B1-Motion>", self.on_snip_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self.on_button_release)

        x1, y1, x2, y2 = win32gui.GetWindowRect(self.hwnd)
        width = x2 - x1
        height = y2 - y1
        self.master_screen.geometry(f"{width + 2}x{height + 2}+{x1}+{y1}")
        self.master_screen.attributes("-alpha", OVERLAY_ALPHA)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)
        self.master_screen.overrideredirect(True)
        self.master_screen.focus_set()

    def on_button_press(self: Self, event: Event[Canvas]) -> None:
        """Record the starting coordinates and initialise the selection rectangle."""
        self.start_x = event.x
        self.start_y = event.y
        if self._rect_id is not None:
            self.snip_surface.delete(self._rect_id)
        rect_id = self.snip_surface.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x + 1,
            self.start_y + 1,
            outline=SELECTION_OUTLINE,
            width=SELECTION_WIDTH,
            fill=OVERLAY_BACKGROUND,
        )
        self._rect_id = rect_id if isinstance(rect_id, int) else 1

    def on_snip_drag(self: Self, event: Event[Canvas]) -> None:
        """Update the selection rectangle as the cursor moves."""
        self.current_x = event.x
        self.current_y = event.y
        rect_id = self._rect_id or 1
        self.snip_surface.coords(rect_id, self.start_x, self.start_y, self.current_x, self.current_y)

    def on_button_release(self: Self, _event: Event[Canvas]) -> None:
        """Capture the selected region and close the overlay."""
        left = min(self.start_x, self.current_x)
        top = min(self.start_y, self.current_y)
        right = max(self.start_x, self.current_x)
        bottom = max(self.start_y, self.current_y)
        if right - left <= 0 or bottom - top <= 0:
            return

        self.take_bounded_screenshot(left, top, right, bottom)
        self._close()

    def take_bounded_screenshot(self: Self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Capture the user-selected region into ``result`` and record window bounds."""
        vision = Vision(self.hwnd)
        try:
            vision.refresh()
        except OSError:
            self.result = None
            self.rect = None
            return

        frame = vision.opencv_image
        if frame.size == 0:
            self.result = None
            self.rect = None
            return

        height, width = frame.shape[:2]
        x_start = max(0, min(int(min(x1, x2)), width - 1))
        y_start = max(0, min(int(min(y1, y2)), height - 1))
        x_end = max(0, min(int(max(x1, x2)), width))
        y_end = max(0, min(int(max(y1, y2)), height))
        if x_start >= x_end or y_start >= y_end:
            self.result = None
            self.rect = None
            return

        self.result = frame[y_start:y_end, x_start:x_end]
        win_rect = win32gui.GetWindowRect(self.hwnd)
        self.rect = (win_rect[0], win_rect[1], win_rect[2] - win_rect[0], win_rect[3] - win_rect[1])

    def _close(self: Self) -> None:
        """Destroy the overlay and unblock the Tk root loop."""
        quit_cb = getattr(self.master_screen, "quit", None)
        if callable(quit_cb):
            try:
                quit_cb()
            except tk.TclError:
                pass
        destroy_cb = getattr(self.master_screen, "destroy", None)
        exists_cb = getattr(self.master_screen, "winfo_exists", None)
        if callable(destroy_cb):
            try:
                if not callable(exists_cb) or exists_cb():
                    destroy_cb()
            except tk.TclError:
                pass
        master_quit = getattr(self.master, "quit", None)
        master_exists = getattr(self.master, "winfo_exists", None)
        if callable(master_quit):
            try:
                if not callable(master_exists) or master_exists():
                    master_quit()
            except tk.TclError:
                pass
