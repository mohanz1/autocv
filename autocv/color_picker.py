"""The `color_picker` module provides functionality for picking colors from a specified window.

It features a ColorPicker class that allows users to pick a color from any point on the screen.
This module is useful in applications that require accurate color selection or recognition.
The ColorPicker class uses a Tkinter interface to display a magnified area around the mouse cursor,
and it integrates with the Vision class from the core module for screen capture and color analysis.
"""

from __future__ import annotations

__all__ = ("ColorPicker",)

from pathlib import Path
from tkinter import NW, Canvas, Tk, Toplevel
from typing import Protocol, Self

import cv2 as cv
import numpy as np
import numpy.typing as npt
import win32api
import win32con
import win32gui
from PIL import Image, ImageDraw, ImageFont, ImageTk
from PIL.ImageTk import PhotoImage

from .core import Vision

# Constants for sampling and magnification
PIXELS = 1
ZOOM = 40
REFRESH_DELAY_MS = 5
_DEFAULT_CANVAS_COLOR: int = 217
_FONT_PATH = Path(__file__).parent / "data" / "Helvetica.ttf"


class ColorPickerController:
    """Encapsulate Vision interactions for sampling cursor-colour patches."""

    def __init__(self, hwnd: int, default_canvas: npt.NDArray[np.uint8]) -> None:
        self.hwnd = hwnd
        self.vision = Vision(hwnd)
        self.default_canvas = default_canvas

    def capture_cursor_patch(self) -> tuple[npt.NDArray[np.uint8], int, int, tuple[int, int]]:
        """Capture a small patch around the current cursor; fall back to default canvas when out of bounds."""
        cursor_pos = win32gui.GetCursorPos()
        x, y = win32gui.ScreenToClient(self.hwnd, cursor_pos)

        self.vision.refresh()
        frame = self.vision.opencv_image
        if frame.size == 0:
            return self.default_canvas, x, y, cursor_pos

        cropped = frame[y - PIXELS : y + PIXELS + 1, x - PIXELS : x + PIXELS + 1, ::-1]
        if x < 0 or y < 0 or cropped.size == 0:
            return self.default_canvas, x, y, cursor_pos
        return cropped, x, y, cursor_pos

    @property
    def frame(self) -> npt.NDArray[np.uint8]:
        """Latest captured frame."""
        return self.vision.opencv_image


class MouseEvent(Protocol):
    """Protocol for tkinter mouse events carrying x/y coordinates."""

    x: int
    y: int


class ColorPicker:
    """Interactive magnifier that returns pixel samples from a target window.

    Attributes:
        hwnd: Window handle that supplies frames for sampling.
        master: Root Tk instance responsible for lifecycle management.
        size: Side length (in pixels) of the magnifier canvas.
        snip_surface: Canvas displaying the magnified cursor region.
        result: Latest sampled RGB colour and screen coordinate.
        prev_state: Cached state of the left mouse button for edge detection.
        controller: Capture controller used to retrieve frames.
    """

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Create a ColorPicker instance and set up the interactive overlay.

        Args:
            hwnd: Handle of the window to pick colors from.
            master: Parent Tkinter window.
        """
        self.hwnd = hwnd
        self.master = master
        self.size = (PIXELS * 2 + 1) * ZOOM
        self.snip_surface: Canvas  # Initialized in create_screen_canvas()
        self.start_x: int = -1
        self.start_y: int = -1
        self.current_x: int = -1
        self.current_y: int = -1
        self.result: tuple[tuple[int, int, int], tuple[int, int]] | None = None

        # Get the initial state of the left mouse button.
        self.prev_state = win32api.GetKeyState(win32con.VK_LBUTTON)

        # Create a default canvas image (a neutral gray background) for when the cursor is outside bounds.
        self.default_canvas = (
            np.ones((PIXELS * 2 + 1, PIXELS * 2 + 1, 3), dtype=np.uint8) * _DEFAULT_CANVAS_COLOR
        ).astype(np.uint8)

        # Controller encapsulating capture logic.
        self.controller = ColorPickerController(hwnd, self.default_canvas)

        self.master.title("AutoCV Color Picker")
        self.master_screen = Toplevel(master)
        self.picture_frame = None  # Placeholder; will be set in create_screen_canvas()
        self.create_screen_canvas()

    def set_geometry(self: Self, cursor_pos: tuple[int, int] | None = None) -> None:
        """Set the position and size of the color picker window.

        Args:
            cursor_pos: Current screen coordinates of the cursor. Defaults to the current cursor position.
        """
        if not self.master_screen or not self.master_screen.winfo_exists():
            return

        x, y = cursor_pos or win32gui.GetCursorPos()
        self.master_screen.geometry(f"{self.size}x{self.size}+{x}+{y}")

    def create_screen_canvas(self: Self) -> None:
        """Create the Tkinter canvas and overlay window for the color picker."""
        self.master_screen.withdraw()
        self.master.withdraw()

        # Create a PhotoImage from the default canvas.
        img = Image.fromarray(self.default_canvas)
        photo_img = ImageTk.PhotoImage(img)

        self.snip_surface = Canvas(self.master_screen, width=self.size, height=self.size, highlightthickness=0)
        self.snip_surface.pack()

        self.snip_surface.create_image(0, 0, image=photo_img, anchor=NW)
        self.snip_surface.img = photo_img  # type: ignore[attr-defined]

        self.set_geometry()
        self.master_screen.deiconify()
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)
        self.master_screen.overrideredirect(boolean=True)
        self.master_screen.focus_set()
        self.master.after(0, self.on_tick)

    def on_tick(self: Self) -> None:
        """Update the color picker canvas based on the current cursor position."""
        if not self.master_screen:
            return

        cropped_image, x, y, cursor_pos = self.controller.capture_cursor_patch()
        self.set_geometry(cursor_pos)

        # Resize the cropped image to create a magnified view.
        resized_img = cv.resize(cropped_image, None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
        img = Image.fromarray(resized_img)
        img = self.draw_cursor_coordinates(img, x, y)
        photo_image = PhotoImage(img)

        # Update the canvas with the new image.
        self.snip_surface.delete("center")
        self.snip_surface.create_image(0, 0, image=photo_image, anchor=NW)
        self.snip_surface.img = photo_image  # type: ignore[attr-defined]
        self.draw_center_rectangle(cropped_image)

        # Check if the left mouse button state has changed (indicating a click).
        curr_state = win32api.GetKeyState(win32con.VK_LBUTTON)
        if self.prev_state != curr_state:
            if curr_state >= 0:
                self.handle_button_press(x, y)
            self.prev_state = curr_state

        # Schedule the next update.
        if self.master:
            self.master.after(REFRESH_DELAY_MS, self.on_tick)

    def draw_cursor_coordinates(self: Self, img: Image.Image, x: int, y: int) -> Image.Image:
        """Draw the current cursor coordinates onto the magnified image.

        Args:
            img: Magnified image.
            x: X-coordinate of the cursor.
            y: Y-coordinate of the cursor.

        Returns:
            Image.Image: Updated image with cursor coordinates drawn.
        """
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(str(_FONT_PATH), 8)
        try:
            color = self.controller.frame[y + PIXELS, x, ::-1]
        except IndexError:
            color = np.zeros(3, dtype=np.uint8)
        inverse_color = 255 - color
        hex_color = f"#{inverse_color[0]:02x}{inverse_color[1]:02x}{inverse_color[2]:02x}"
        draw.text(
            (img.width // 2, img.height - ZOOM // 2),
            f"{x},{y}",
            fill=hex_color,
            font=font,
            anchor="mm",
        )
        return img

    def draw_center_rectangle(self: Self, cropped_image: npt.NDArray[np.uint8]) -> None:
        """Draw a center rectangle on the color picker canvas as a visual marker.

        The rectangle's outline color is determined by the inverse of the average color of the cropped region.

        Args:
            cropped_image: Cropped image around the cursor.
        """
        average_color_row = np.nanmean(cropped_image, axis=0)
        average_color = np.round(np.average(average_color_row, axis=0)).astype(int)
        if np.isnan(average_color).any():
            average_color = np.zeros(3, dtype=np.uint8)
        inverse_color = 255 - average_color
        hex_color = f"#{inverse_color[0]:02x}{inverse_color[1]:02x}{inverse_color[2]:02x}"
        rect_pos = PIXELS * ZOOM
        self.snip_surface.create_rectangle(
            rect_pos,
            rect_pos,
            rect_pos + ZOOM,
            rect_pos + ZOOM,
            dash=(3, 5),
            tags="center",
            outline=hex_color,
        )

    def handle_button_press(self: Self, x: int, y: int) -> None:
        """Handle a mouse click to pick a color from the screen.

        The method retrieves the color of the pixel at the given coordinates. If the coordinates are
        outside the bounds of the captured image, a default error value is stored.

        Args:
            x: X-coordinate of the picked pixel.
            y: Y-coordinate of the picked pixel.
        """
        if x < 0 or x >= self.vision.opencv_image.shape[1] or y < 0 or y >= self.vision.opencv_image.shape[0]:
            self.result = ((-1, -1, -1), (-1, -1))
        else:
            color = np.flip(self.vision.opencv_image[y, x])
            color_list = color.tolist()
            self.result = ((int(color_list[0]), int(color_list[1]), int(color_list[2])), (x, y))

        self.master_screen.destroy()
        self.master_screen.quit()

    @property
    def vision(self) -> Vision:
        """Expose the underlying Vision instance for compatibility."""
        return self.controller.vision
