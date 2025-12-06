"""Tkinter overlay for selecting and capturing AutoCV window regions."""

from __future__ import annotations

__all__ = ("ImagePicker",)

import logging
from tkinter import BOTH, YES, Canvas, Event, Frame, Tk, Toplevel
from typing import TYPE_CHECKING, Final, Protocol, Self

import win32con
import win32gui

from autocv.models import InvalidHandleError

from .core import Vision

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

_OVERLAY_ALPHA: Final[float] = 0.3
_OVERLAY_COLOR = "maroon3"
_CURSOR_STYLE = "cross"
_RECT_OUTLINE = "red"
_RECT_WIDTH: Final[int] = 3
_DEFAULT_GEOM_PADDING: Final[int] = 2

logger = logging.getLogger(__name__)


class ImagePickerController:
    """Encapsulate window attachment and capture for region selection."""

    def __init__(self, hwnd: int, vision_factory: Callable[[int], Vision] | None = None) -> None:
        self.hwnd = hwnd
        self._vision_factory = vision_factory
        self.vision: Vision | None = None

    def bring_to_foreground(self) -> None:
        """Show and foreground the target window."""
        win32gui.ShowWindow(self.hwnd, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.hwnd)

    def window_rect(self) -> tuple[int, int, int, int]:
        """Return the window rectangle (x1, y1, x2, y2)."""
        return win32gui.GetWindowRect(self.hwnd)

    def capture_region(self, x1: int, y1: int, x2: int, y2: int) -> npt.NDArray[np.uint8]:
        """Capture the region defined in client coordinates."""
        if self.vision is None:
            factory = self._vision_factory or Vision
            self.vision = factory(self.hwnd)
        try:
            self.vision.refresh()
        except InvalidHandleError as exc:
            logger.debug("Skipping refresh due to invalid window handle: %s", exc)
        return self.vision.opencv_image[y1:y2, x1:x2]

    def full_rect_as_bounds(self) -> tuple[int, int, int, int]:
        """Return the full window bounds as (x, y, width, height)."""
        x1, y1, x2, y2 = self.window_rect()
        return x1, y1, x2 - x1, y2 - y1


class CanvasEvent(Protocol):
    """Protocol for tkinter canvas events carrying x/y coordinates."""

    x: int
    y: int


class ImagePicker:
    """Interactive overlay for selecting and capturing a region of a window."""

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Initializes the ImagePicker instance and sets up the GUI overlay for region selection.

        Args:
            hwnd: Window handle of the window to capture.
            master: Parent Tkinter window.
        """
        self.hwnd: int = hwnd
        self.master: Tk = master
        self.controller = ImagePickerController(hwnd)
        self.snip_surface: Canvas  # Initialized in create_screen_canvas
        self.start_x: int = -1
        self.start_y: int = -1
        self.current_x: int = -1
        self.current_y: int = -1
        self.result: npt.NDArray[np.uint8] | None = None
        self.rect: tuple[int, int, int, int] | None = None
        self._rect_id: int | None = None

        self.master.title("AutoCV Image Picker")
        self.master_screen = Toplevel(self.master)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", _OVERLAY_COLOR)
        self.picture_frame = Frame(self.master_screen, background=_OVERLAY_COLOR)
        self.picture_frame.pack(fill=BOTH, expand=YES)

        self.create_screen_canvas()

    def create_screen_canvas(self: Self) -> None:
        """Creates the canvas and transparent overlay window for region selection.

        The method brings the target window to the foreground, creates a full-screen overlay
        with a semi-transparent canvas, and binds mouse events for region selection.
        """
        # Bring the target window to the foreground
        self.controller.bring_to_foreground()
        self.master_screen.deiconify()
        self.master.withdraw()

        # Create the canvas for drawing the selection rectangle
        self.snip_surface = Canvas(self.picture_frame, cursor=_CURSOR_STYLE, bg="grey11")
        self.snip_surface.pack(fill=BOTH, expand=YES)

        # Bind mouse events
        self.snip_surface.bind("<ButtonPress-1>", self.on_button_press)
        self.snip_surface.bind("<B1-Motion>", self.on_snip_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self.on_button_release)

        # Position the overlay window over the target window
        x1, y1, x2, y2 = self.controller.window_rect()
        width = x2 - x1
        height = y2 - y1
        self.master_screen.geometry(f"{width + _DEFAULT_GEOM_PADDING}x{height + _DEFAULT_GEOM_PADDING}+{x1}+{y1}")
        self.master_screen.attributes("-alpha", _OVERLAY_ALPHA)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)
        self.master_screen.overrideredirect(boolean=True)
        self.master_screen.focus_set()

    def on_button_press(self: Self, event: CanvasEvent) -> None:
        """Handle the left mouse button press to record the starting coordinates of the selection.

        Args:
            event: Tkinter event containing the x and y coordinates.
        """
        self.start_x = event.x
        self.start_y = event.y
        # Create an initial rectangle with minimal size
        self._rect_id = self.snip_surface.create_rectangle(
            0, 0, 1, 1, outline=_RECT_OUTLINE, width=_RECT_WIDTH, fill=_OVERLAY_COLOR
        )

    def on_snip_drag(self: Self, event: CanvasEvent) -> None:
        """Update the selection rectangle as the mouse is dragged.

        Args:
            event: Tkinter event containing the current mouse coordinates.
        """
        self.current_x, self.current_y = event.x, event.y
        if self._rect_id is None:
            self._rect_id = 1
        self.snip_surface.coords(self._rect_id, self.start_x, self.start_y, self.current_x, self.current_y)

    def take_bounded_screenshot(self: Self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Capture a screenshot of the selected region of the window.

        A Vision instance is created to capture the current image of the window, and then the image
        is cropped to the specified region.

        Args:
            x1: Starting x-coordinate of the region.
            y1: Starting y-coordinate of the region.
            x2: Ending x-coordinate of the region.
            y2: Ending y-coordinate of the region.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self.result = self.controller.capture_region(x1, y1, x2, y2)
        # Store full window dimensions as a fallback or for additional context.
        self.rect = self.controller.full_rect_as_bounds()

    def on_button_release(self: Self, _: Event[Canvas]) -> None:
        """Finalize the region selection and capture the screenshot.

        Once the left mouse button is released, the selected region is determined from the recorded
        coordinates, a screenshot is taken, and the overlay window is closed.
        """
        x1 = min(self.start_x, self.current_x)
        y1 = min(self.start_y, self.current_y)
        x2 = max(self.start_x, self.current_x)
        y2 = max(self.start_y, self.current_y)
        if x1 == x2 or y1 == y2:
            self.master_screen.destroy()
            self.master_screen.quit()
            return
        self.take_bounded_screenshot(x1, y1, x2, y2)
        self.master_screen.destroy()
        self.master_screen.quit()
