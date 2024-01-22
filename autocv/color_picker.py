"""The `color_picker` module provides functionality for picking colors from a specified window.

It features a ColorPicker class that allows users to pick a color from any point on the screen. This module is
particularly useful in applications that require color selection or recognition, such as graphic design tools or
automation scripts.

The ColorPicker class uses a Tkinter interface to display a magnified area around the mouse cursor, allowing users to
accurately pick colors from pixels. It integrates with the Vision class from the `core` module for screen capture and
color analysis.
"""

from __future__ import annotations

__all__ = ("ColorPicker",)

from pathlib import Path
from tkinter import NW, Canvas, Tk, Toplevel

import cv2 as cv
import numpy as np
import numpy.typing as npt
import win32api
import win32con
import win32gui
from PIL import Image, ImageDraw, ImageFont, ImageTk
from PIL.ImageTk import PhotoImage
from typing_extensions import Self

from .core import Vision
from .models import Color, ColorWithPoint, Point

PIXELS = 1
ZOOM = 40


class ColorPicker:
    """The ColorPicker class provides an interactive color picking tool using a Tkinter window.

    It captures a portion of the screen around the mouse cursor, allowing users to pick a pixel's color from within a
    specified window.

    Attributes:
        hwnd (int): Handle of the window from which colors will be picked.
        master (Tk): The root Tkinter window for the color picker interface.
        size (int): The size of the color picker magnifier.
        snip_surface (Canvas|None): The Tkinter canvas on which the magnified screen portion is drawn.
        result (ColorWithPoint|None): The most recently picked color along with its screen coordinates.
        prev_state (int): The previous state of the left mouse button to detect clicks.
        vision (Vision): An instance of Vision used for capturing and processing the screen image.

    The class creates a magnified view of the screen around the cursor's current position, allowing the user
    to pick a color by clicking. The picked color, along with its screen coordinates, is stored in `result` attribute.
    """

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Creates a color picker instance.

        Args:
        ----
            hwnd (int): The handle of the window to pick colors from.
            master (Tk): The Tk instance to create the color picker on.
        """
        self.hwnd = hwnd
        self.master = master
        self.size = (PIXELS * 2 + 1) * ZOOM
        self.snip_surface: Canvas
        self.result: ColorWithPoint
        self.prev_state = win32api.GetKeyState(win32con.VK_LBUTTON)  # type: ignore[no-untyped-call]
        self.vision = Vision(self.hwnd)

        # Create a default canvas for when the mouse is outside the window
        self.default_canvas = (np.ones((PIXELS * 2 + 1, PIXELS * 2 + 1, 3), dtype=np.uint8) * 217).astype(np.uint8)

        self.master.title("AutoCV Image Picker")
        self.master_screen = Toplevel(master)
        self.picture_frame = None
        self.create_screen_canvas()

    def set_geometry(self: Self, cursor_pos: tuple[int, int] | None = None) -> None:
        """Sets the position and size of the color picker window.

        Args:
        ----
            cursor_pos (Tuple[int, int], optional): The current position of the cursor, if available. Defaults to None.
        """
        if not self.master_screen or not self.master_screen.winfo_exists():
            return

        x, y = cursor_pos or win32gui.GetCursorPos()
        self.master_screen.geometry(f"{self.size}x{self.size}+{x}+{y}")

    def create_screen_canvas(self: Self) -> None:
        """Creates the canvas for the color picker and sets up the window."""
        self.master_screen.withdraw()
        self.master.withdraw()

        img = Image.fromarray(self.default_canvas)
        img = ImageTk.PhotoImage(img)

        self.snip_surface = Canvas(self.master_screen, width=self.size, height=self.size, highlightthickness=0)
        self.snip_surface.pack()

        self.snip_surface.create_image(0, 0, image=img, anchor=NW)
        self.snip_surface.img = img  # type: ignore[attr-defined]

        self.set_geometry()
        self.master_screen.deiconify()
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)  # noqa: FBT003
        self.master_screen.overrideredirect(boolean=True)
        self.master_screen.focus_set()  # Restricted access main menu
        self.master.after(0, self.on_tick)

    def on_tick(self: Self) -> None:
        """Handles the mouse movement and updates the color picker canvas accordingly."""
        if not self.master_screen:
            return

        # Refresh the vision and get the cursor position
        self.vision.refresh()
        cursor_pos = win32gui.GetCursorPos()
        x, y = win32gui.ScreenToClient(self.hwnd, cursor_pos)

        # Set the geometry of the color picker window and get the cropped image
        self.set_geometry(cursor_pos)
        cropped_image = self.vision.opencv_image[y - PIXELS : y + PIXELS + 1, x - PIXELS : x + PIXELS + 1, ::-1]

        # If the cursor is outside the window or the cropped image is empty, use the default canvas
        if x < 0 or y < 0 or not cropped_image.any():
            cropped_image = self.default_canvas

        # Resize the image and add the cursor coordinates
        img = cv.resize(cropped_image, None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
        img = Image.fromarray(img)
        img = self.draw_cursor_coordinates(img, x, y)
        img = PhotoImage(img)

        # Update the canvas with the new image and draw the center rectangle
        self.snip_surface.delete("center")
        self.snip_surface.create_image(0, 0, image=img, anchor=NW)
        # This seems repetitive but it doesn't work without this line
        self.snip_surface.img = img  # type: ignore[attr-defined]
        self.draw_center_rectangle(cropped_image)

        # Check if the left mouse button was pressed
        curr_state = win32api.GetKeyState(win32con.VK_LBUTTON)  # type: ignore[no-untyped-call]
        if self.prev_state != curr_state:
            if curr_state >= 0:
                self.handle_button_press(x, y)
            self.prev_state = curr_state

        # Call this function again after 5 milliseconds
        if self.master:
            self.master.after(5, self.on_tick)

    def draw_cursor_coordinates(self: Self, img: Image.Image, x: int, y: int) -> Image.Image:
        """Draws the cursor coordinates onto the given image.

        Args:
        ----
            img (Image.Image): The image to draw on.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.

        Returns:
        -------
            Image.Image: The updated image.
        """
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(str(Path(__file__).parent / "data" / "Helvetica.ttf"), 8)
        try:
            color = self.vision.opencv_image[y + PIXELS, x, ::-1]
        except IndexError:
            color = np.zeros(3, dtype=np.uint8)

        inverse_average_color = Color.to_hex((255 - color).tolist())
        draw.text(
            (img.width // 2, img.height - ZOOM // 2),
            f"{x},{y}",
            inverse_average_color,
            font,
            "mm",
        )
        return img

    def draw_center_rectangle(self: Self, cropped_image: npt.NDArray[np.uint8]) -> None:
        """Draw the center rectangle on the color picker canvas.

        Args:
        ----
            cropped_image: The cropped image around the current mouse position.
        """
        average_color_row = np.nanmean(cropped_image, axis=0)
        average_color = np.round(np.average(average_color_row, axis=0)).astype(int)

        if np.isnan(average_color).any():
            average_color = np.zeros(3, dtype=np.uint8)

        inverse_average_color = 255 - average_color
        hex_color = Color.to_hex(inverse_average_color)
        size = PIXELS * ZOOM
        self.snip_surface.create_rectangle(
            size,
            size,
            size + ZOOM,
            size + ZOOM,
            dash=(3, 5),
            tags="center",
            outline=hex_color,
        )

    def handle_button_press(self: Self, x: int, y: int) -> None:
        """Get the color of the pixel at the given coordinates (x,y).

        Args:
        ----
            x (int): The x coordinate of the pixel.
            y (int): The y coordinate of the pixel.
        """
        # Check if the coordinates are within the bounds of the image
        if x < 0 or x > self.vision.opencv_image.shape[1] or y < 0 or y > self.vision.opencv_image.shape[0]:
            self.result = ColorWithPoint(Color(-1, -1, -1), Point(-1, -1))
        else:
            # Get the color of the pixel at (x,y) and create a ColorWithPoint instance
            color = np.flip(self.vision.opencv_image[y, x])
            self.result = ColorWithPoint.from_ndarray_sequence(color, (x, y))
        self.master_screen.destroy()
        self.master_screen.quit()
