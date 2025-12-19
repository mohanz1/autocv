"""Pixel color picker overlay.

The :class:`~autocv.color_picker.ColorPicker` widget displays a magnified view
of the cursor location within the target window. Clicking selects the pixel
color and stores it in :attr:`~autocv.color_picker.ColorPicker.result`.
"""

from __future__ import annotations

__all__ = ("ColorPicker",)

from pathlib import Path
from tkinter import NW, Canvas, Tk, Toplevel
from typing import Final, TypeAlias

import cv2 as cv
import numpy as np
import numpy.typing as npt
import win32api
import win32con
import win32gui
from PIL import Image, ImageDraw, ImageFont, ImageTk
from PIL.ImageTk import PhotoImage
from typing_extensions import Self

from . import constants
from .core import Vision

ImageArray = npt.NDArray[np.uint8]
Point = tuple[int, int]
Color = tuple[int, int, int]

# Pyright can struggle to infer NumPy-derived scalar types in strict mode; this
# helper keeps the mean calculation in plain Python to preserve a concrete
# `tuple[int, int, int]` result.
ColorTotals: TypeAlias = tuple[int, int, int]

# Constants for sampling and magnification.
PIXELS: Final[int] = constants.PIXEL_RADIUS
ZOOM: Final[int] = constants.PIXEL_ZOOM
REFRESH_DELAY_MS: Final[int] = constants.REFRESH_DELAY_MS
_DEFAULT_CANVAS_COLOR: Final[int] = constants.FALLBACK_COLOR
RGB_CHANNELS: Final[int] = 3
_FONT_PATH: Final[Path] = Path(__file__).parent / "data" / "Helvetica.ttf"
_FONT_SIZE: Final[int] = 8


class ColorPickerController:
    """Encapsulate Vision interactions for sampling cursor-colour patches."""

    def __init__(self, hwnd: int, default_canvas: ImageArray) -> None:
        self.hwnd = hwnd
        self.vision = Vision(hwnd)
        self.default_canvas = default_canvas

    def capture_cursor_patch(self) -> tuple[ImageArray, int, int, Point]:
        """Capture a small patch around the current cursor; fall back to default canvas when out of bounds."""
        cursor_pos = win32gui.GetCursorPos()
        x, y = win32gui.ScreenToClient(self.hwnd, cursor_pos)

        self.vision.refresh()
        frame = self.vision.opencv_image
        if frame.size == 0:
            return self.default_canvas, x, y, cursor_pos

        if x < 0 or y < 0:
            return self.default_canvas, x, y, cursor_pos

        cropped = frame[y - PIXELS : y + PIXELS + 1, x - PIXELS : x + PIXELS + 1, ::-1]
        if cropped.size == 0:
            return self.default_canvas, x, y, cursor_pos
        return cropped, x, y, cursor_pos

    @property
    def frame(self) -> ImageArray:
        """Latest captured frame."""
        return self.vision.opencv_image


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
        self.result: tuple[Color, Point] | None = None

        # Get the initial state of the left mouse button.
        self.prev_state = win32api.GetKeyState(win32con.VK_LBUTTON)

        # Create a default canvas image (a neutral gray background) for when the cursor is outside bounds.
        self.default_canvas: ImageArray = np.full(
            (PIXELS * 2 + 1, PIXELS * 2 + 1, 3),
            _DEFAULT_CANVAS_COLOR,
            dtype=np.uint8,
        )
        self._font = ImageFont.truetype(str(_FONT_PATH), _FONT_SIZE)

        # Controller encapsulating capture logic.
        self.controller = ColorPickerController(hwnd, self.default_canvas)

        self.master.title("AutoCV Color Picker")
        self.master_screen = Toplevel(master)
        self.create_screen_canvas()

    def set_geometry(self: Self, cursor_pos: Point | None = None) -> None:
        """Set the position and size of the color picker window.

        Args:
            cursor_pos: Current screen coordinates of the cursor. Defaults to the current cursor position.
        """
        if not self.master_screen.winfo_exists():
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
        if not self.master_screen.winfo_exists():
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
        if self.master_screen.winfo_exists():
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
            font=self._font,
            anchor="mm",
        )
        return img

    def draw_center_rectangle(self: Self, cropped_image: ImageArray) -> None:
        """Draw a center rectangle on the color picker canvas as a visual marker.

        The rectangle's outline color is determined by the inverse of the average color of the cropped region.

        Args:
            cropped_image: Cropped image around the cursor.
        """
        if cropped_image.size == 0:
            mean_r, mean_g, mean_b = (0, 0, 0)
        else:
            height, width, channels = cropped_image.shape
            if channels != RGB_CHANNELS:
                msg = f"Expected an RGB patch with {RGB_CHANNELS} channels."
                raise ValueError(msg)
            total_r = 0
            total_g = 0
            total_b = 0
            for row in range(height):
                for col in range(width):
                    r_i, g_i, b_i = (int(v) for v in cropped_image[row, col])
                    total_r += r_i
                    total_g += g_i
                    total_b += b_i

            count = height * width
            mean_r, mean_g, mean_b = (total_r // count, total_g // count, total_b // count)

        inv_r, inv_g, inv_b = (255 - mean_r, 255 - mean_g, 255 - mean_b)
        hex_color = f"#{inv_r:02x}{inv_g:02x}{inv_b:02x}"
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
        frame = self.vision.opencv_image
        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            self.result = ((-1, -1, -1), (-1, -1))
        else:
            r, g, b = (int(v) for v in frame[y, x, ::-1])
            self.result = ((r, g, b), (x, y))

        self.master_screen.destroy()
        self.master_screen.quit()

    @property
    def vision(self) -> Vision:
        """Expose the underlying Vision instance for compatibility."""
        return self.controller.vision
