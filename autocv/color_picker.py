"""Colour picker overlay backed by AutoCV screen captures."""

from __future__ import annotations

__all__ = ("ColorPicker",)

import tkinter as tk
from pathlib import Path
from tkinter import NW, Canvas, Tk, Toplevel

import cv2 as cv
import numpy as np
import numpy.typing as npt
import win32api
import win32con
import win32gui
from PIL import Image, ImageDraw, ImageFont, ImageTk
from typing_extensions import Self

from .core import Vision

PIXELS = 1
ZOOM = 40
DEFAULT_GREY = 217
UPDATE_INTERVAL_MS = 5
FONT_SIZE = 8


class ColorPicker:
    """Interactive magnifier that returns pixel samples from a target window."""

    def __init__(self: Self, hwnd: int, master: Tk) -> None:
        """Initialise the picker overlay attached to ``hwnd``."""
        self.hwnd = hwnd
        self.master = master
        self.master_screen = Toplevel(master)
        self.master_screen.withdraw()

        self.snip_surface: Canvas
        self.result: tuple[tuple[int, int, int], tuple[int, int]] | None = None

        self.prev_state = win32api.GetKeyState(win32con.VK_LBUTTON)
        self.vision = Vision(self.hwnd)
        self.default_canvas: npt.NDArray[np.uint8] = np.full(
            (PIXELS * 2 + 1, PIXELS * 2 + 1, 3),
            DEFAULT_GREY,
            dtype=np.uint8,
        )
        self._font = self._load_font()
        self._tick_job: str | None = None
        self._photo_image: ImageTk.PhotoImage | None = None

        self.master.title("AutoCV Color Picker")
        self.master.protocol("WM_DELETE_WINDOW", self.close)
        self.master_screen.protocol("WM_DELETE_WINDOW", self.close)

        self.create_screen_canvas()

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_path = Path(__file__).parent / "data" / "Helvetica.ttf"
        try:
            return ImageFont.truetype(str(font_path), FONT_SIZE)
        except OSError:
            return ImageFont.load_default()

    def _master_exists(self) -> bool:
        exists = getattr(self.master, "winfo_exists", None)
        if callable(exists):
            try:
                return bool(exists())
            except tk.TclError:
                return False
        return True

    def _screen_exists(self) -> bool:
        exists = getattr(self.master_screen, "winfo_exists", None)
        if callable(exists):
            try:
                return bool(exists())
            except tk.TclError:
                return False
        return True

    def set_geometry(self: Self, cursor_pos: tuple[int, int] | None = None) -> None:
        """Position the overlay around the provided cursor location."""
        if not self._screen_exists():
            return
        x, y = cursor_pos or win32gui.GetCursorPos()
        size = (PIXELS * 2 + 1) * ZOOM
        self.master_screen.geometry(f"{size}x{size}+{x}+{y}")

    def create_screen_canvas(self: Self) -> None:
        """Instantiate the transparent overlay and start the capture loop."""
        self.master.withdraw()
        self.master_screen.withdraw()

        img = Image.fromarray(self.default_canvas)
        self._photo_image = ImageTk.PhotoImage(img)

        self.snip_surface = Canvas(
            self.master_screen,
            width=self._photo_image.width(),
            height=self._photo_image.height(),
            highlightthickness=0,
        )
        self.snip_surface.pack()
        self.snip_surface.create_image(0, 0, image=self._photo_image, anchor=NW, tags="canvas")
        self.snip_surface.img = self._photo_image  # type: ignore[attr-defined]

        self.snip_surface.bind("<ButtonPress-1>", self._on_button_press)
        self.snip_surface.bind("<B1-Motion>", self._on_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self._on_button_release)

        self.master_screen.deiconify()
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)
        self.master_screen.overrideredirect(True)
        try:
            self.master_screen.focus_set()
        except tk.TclError:
            pass
        after_cb = getattr(self.master, "after", None)
        if callable(after_cb):
            job = after_cb(0, self.on_tick)
            if job is not None:
                self._tick_job = str(job)

    def on_tick(self: Self) -> None:
        """Refresh the magnified preview around the cursor."""
        if not self._screen_exists():
            return

        try:
            self.vision.refresh()
            frame = self.vision.opencv_image
        except OSError:
            frame = np.zeros((0, 0, 3), dtype=np.uint8)

        cursor_pos = win32gui.GetCursorPos()
        client_x, client_y = win32gui.ScreenToClient(self.hwnd, cursor_pos)
        self.set_geometry(cursor_pos)

        if (
            frame.size == 0
            or client_x - PIXELS < 0
            or client_y - PIXELS < 0
            or client_x + PIXELS >= frame.shape[1]
            or client_y + PIXELS >= frame.shape[0]
        ):
            cropped = self.default_canvas
        else:
            cropped = frame[
                client_y - PIXELS : client_y + PIXELS + 1,
                client_x - PIXELS : client_x + PIXELS + 1,
                ::-1,
            ]

        resized_img = cv.resize(cropped, None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
        img = Image.fromarray(resized_img)
        img = self.draw_cursor_coordinates(img, client_x, client_y)
        self._photo_image = ImageTk.PhotoImage(img)

        self.snip_surface.delete("canvas")
        self.snip_surface.create_image(0, 0, image=self._photo_image, anchor=NW, tags="canvas")
        self.snip_surface.img = self._photo_image  # type: ignore[attr-defined]
        self.draw_center_rectangle(cropped)

        curr_state = win32api.GetKeyState(win32con.VK_LBUTTON)
        if self.prev_state != curr_state:
            if curr_state >= 0:
                self.handle_button_press(client_x, client_y)
            self.prev_state = curr_state

        if self._master_exists():
            after_cb = getattr(self.master, "after", None)
            if callable(after_cb):
                job = after_cb(UPDATE_INTERVAL_MS, self.on_tick)
                if job is not None:
                    self._tick_job = str(job)

    def draw_cursor_coordinates(self: Self, img: Image.Image, x: int, y: int) -> Image.Image:
        """Overlay the current cursor coordinates onto the magnified image."""
        draw = ImageDraw.Draw(img)
        frame = self.vision.opencv_image
        try:
            colour = frame[y + PIXELS, x, ::-1]
        except (IndexError, ValueError):
            colour = np.zeros(3, dtype=np.uint8)
        inverse = 255 - colour
        draw.text(
            (img.width // 2, img.height - ZOOM // 2),
            f"{x},{y}",
            fill=f"#{int(inverse[0]):02x}{int(inverse[1]):02x}{int(inverse[2]):02x}",
            font=self._font,
            anchor="mm",
        )
        return img

    def draw_center_rectangle(self: Self, cropped_image: npt.NDArray[np.uint8]) -> None:
        """Highlight the central pixel within the magnified preview."""
        self.snip_surface.delete("center")
        average_colour = np.nanmean(cropped_image, axis=(0, 1))
        if np.isnan(average_colour).any():
            average_colour = np.zeros(3, dtype=float)
        inverse = 255 - average_colour
        hex_colour = f"#{int(inverse[0]):02x}{int(inverse[1]):02x}{int(inverse[2]):02x}"
        rect_pos = PIXELS * ZOOM
        self.snip_surface.create_rectangle(
            rect_pos,
            rect_pos,
            rect_pos + ZOOM,
            rect_pos + ZOOM,
            dash=(3, 5),
            tags="center",
            outline=hex_colour,
        )

    def handle_button_press(self: Self, x: int, y: int) -> None:
        """Sample the colour beneath the cursor and close the picker."""
        frame = self.vision.opencv_image
        if frame.size == 0 or x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            self.result = ((-1, -1, -1), (-1, -1))
        else:
            colour = frame[y, x, ::-1]
            self.result = (
                (int(colour[0]), int(colour[1]), int(colour[2])),
                (x, y),
            )
        self.close()

    def close(self: Self) -> None:
        """Tear down the overlay and cancel any pending updates."""
        after_cancel = getattr(self.master, "after_cancel", None)
        if self._tick_job and callable(after_cancel) and self._master_exists():
            try:
                after_cancel(self._tick_job)
            except tk.TclError:
                pass
            self._tick_job = None
        if self._screen_exists():
            try:
                self.master_screen.destroy()
            except tk.TclError:
                pass
        quit_cb = getattr(self.master_screen, "quit", None)
        if callable(quit_cb):
            try:
                quit_cb()
            except tk.TclError:
                pass
        master_quit = getattr(self.master, "quit", None)
        if callable(master_quit) and self._master_exists():
            try:
                master_quit()
            except tk.TclError:
                pass

    def _on_button_press(self: Self, _event: tk.Event[Canvas]) -> None:
        """Tk binding placeholder - sampling is driven by the polling loop."""

    def _on_drag(self: Self, _event: tk.Event[Canvas]) -> None:
        """Tk binding placeholder - sampling is driven by the polling loop."""

    def _on_button_release(self: Self, _event: tk.Event[Canvas]) -> None:
        """Tk binding placeholder - sampling is driven by the polling loop."""
