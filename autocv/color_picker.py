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

# Constants for sampling and magnification
PIXELS = 1
ZOOM = 40


class ColorPicker:
    """Provides an interactive color picking tool using a Tkinter window.

    Captures a magnified view of a small region around the mouse cursor,
    allowing the user to accurately select a pixel's color from a target window.

    Attributes:
        hwnd (int): Handle of the window from which colors will be picked.
        master (Tk): The root Tkinter window for the color picker interface.
        size (int): The size (in pixels) of the magnifier window.
        snip_surface (Canvas): The Tkinter canvas on which the magnified view is drawn.
        result (tuple[tuple[int, int, int], tuple[int, int]] | None): The most recently picked color and its screen
            coordinates.
        prev_state (int): The previous state of the left mouse button to detect clicks.
        vision (Vision): A Vision instance used for screen capture and color analysis.
    """

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Creates a ColorPicker instance and sets up the interactive overlay.

        Args:
            hwnd (int): The handle of the window to pick colors from.
            master (Tk): The parent Tkinter window.
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

        # Create a Vision instance for screen capture.
        self.vision = Vision(self.hwnd)

        # Create a default canvas image (a neutral gray background) for when the cursor is outside bounds.
        self.default_canvas = (np.ones((PIXELS * 2 + 1, PIXELS * 2 + 1, 3), dtype=np.uint8) * 217).astype(np.uint8)

        self.master.title("AutoCV Color Picker")
        self.master_screen = Toplevel(master)
        self.picture_frame = None  # Placeholder; will be set in create_screen_canvas()
        self.create_screen_canvas()

    def set_geometry(self: Self, cursor_pos: tuple[int, int] | None = None) -> None:
        """Sets the position and size of the color picker window.

        Args:
            cursor_pos (tuple[int, int], optional): The current screen coordinates of the cursor.
                If not provided, the current cursor position is obtained from Win32 APIs.
        """
        if not self.master_screen or not self.master_screen.winfo_exists():
            return

        x, y = cursor_pos or win32gui.GetCursorPos()
        self.master_screen.geometry(f"{self.size}x{self.size}+{x}+{y}")

    def create_screen_canvas(self: Self) -> None:
        """Creates the Tkinter canvas and overlay window for the color picker.

        This method sets up a transparent overlay window, places a canvas on it, and binds the necessary mouse events.
        """
        self.master_screen.withdraw()
        self.master.withdraw()

        # Create a PhotoImage from the default canvas.
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
        self.master_screen.focus_set()
        self.master.after(0, self.on_tick)

    def on_tick(self: Self) -> None:
        """Periodically updates the color picker canvas based on the current cursor position.

        Captures a small region around the cursor, magnifies it, draws cursor coordinates and a center marker,
        and checks for mouse clicks to pick a color.
        """
        if not self.master_screen:
            return

        # Refresh the screen capture.
        self.vision.refresh()
        cursor_pos = win32gui.GetCursorPos()
        x, y = win32gui.ScreenToClient(self.hwnd, cursor_pos)

        self.set_geometry(cursor_pos)
        # Crop a small region around the cursor.
        cropped_image = self.vision.opencv_image[y - PIXELS : y + PIXELS + 1, x - PIXELS : x + PIXELS + 1, ::-1]
        # Use default canvas if the cropped image is empty or the cursor is out of bounds.
        if x < 0 or y < 0 or not cropped_image.any():
            cropped_image = self.default_canvas

        # Resize the cropped image to create a magnified view.
        img = cv.resize(cropped_image, None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
        img = Image.fromarray(img)
        img = self.draw_cursor_coordinates(img, x, y)
        img = PhotoImage(img)

        # Update the canvas with the new image.
        self.snip_surface.delete("center")
        self.snip_surface.create_image(0, 0, image=img, anchor=NW)
        self.snip_surface.img = img  # type: ignore[attr-defined]
        self.draw_center_rectangle(cropped_image)

        # Check if the left mouse button state has changed (indicating a click).
        curr_state = win32api.GetKeyState(win32con.VK_LBUTTON)
        if self.prev_state != curr_state:
            if curr_state >= 0:
                self.handle_button_press(x, y)
            self.prev_state = curr_state

        # Schedule the next update.
        if self.master:
            self.master.after(5, self.on_tick)

    def draw_cursor_coordinates(self: Self, img: Image.Image, x: int, y: int) -> Image.Image:
        """Draws the current cursor coordinates onto the magnified image.

        Args:
            img (Image.Image): The magnified image.
            x (int): The x-coordinate of the cursor.
            y (int): The y-coordinate of the cursor.

        Returns:
            Image.Image: The updated image with cursor coordinates drawn.
        """
        draw = ImageDraw.Draw(img)
        font_path = Path(__file__).parent / "data" / "Helvetica.ttf"
        font = ImageFont.truetype(str(font_path), 8)
        try:
            color = self.vision.opencv_image[y + PIXELS, x, ::-1]
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
        """Draws a center rectangle on the color picker canvas as a visual marker.

        The rectangle's outline color is determined by the inverse of the average color of the cropped region.

        Args:
            cropped_image (npt.NDArray[np.uint8]): The cropped image around the cursor.
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
        """Handles a mouse click to pick a color from the screen.

        The method retrieves the color of the pixel at the given coordinates. If the coordinates are
        outside the bounds of the captured image, a default error value is stored.

        Args:
            x (int): The x-coordinate of the picked pixel.
            y (int): The y-coordinate of the picked pixel.
        """
        if x < 0 or x >= self.vision.opencv_image.shape[1] or y < 0 or y >= self.vision.opencv_image.shape[0]:
            self.result = ((-1, -1, -1), (-1, -1))
        else:
            color = np.flip(self.vision.opencv_image[y, x])
            color_list = color.tolist()
            self.result = ((int(color_list[0]), int(color_list[1]), int(color_list[2])), (x, y))

        self.master_screen.destroy()
        self.master_screen.quit()
