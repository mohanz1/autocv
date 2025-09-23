"""Interactive tool for sampling colours from AutoCV window captures."""

from __future__ import annotations

__all__ = ("AutoColorAid",)

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import cv2
import numpy as np
import sv_ttk
from PIL import Image, ImageDraw, ImageTk
from typing_extensions import Self

from autocv import AutoCV

UPDATE_INTERVAL_MS = 5
PIXEL_RADIUS = 1
ZOOM_SCALE = 40
DEFAULT_REGION_COLOUR = 217
INFO_TEMPLATE = "best color: {}\nbest tolerance: {}"


@dataclass(slots=True)
class PixelSample:
    """Represents a sampled pixel with its image coordinates and RGB colour."""

    row: int
    col: int
    r: int
    g: int
    b: int

    @property
    def colour(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def coordinates(self) -> tuple[int, int]:
        return (self.row, self.col)


class AutoColorAid(tk.Tk):
    """Tkinter application for sampling colours from AutoCV-captured frames."""

    def __init__(self) -> None:
        """Initialise the Auto Color Aid UI and capture helpers."""
        super().__init__()
        self.title("Auto Color Aid")

        self.autocv = AutoCV()
        self._photo_image: ImageTk.PhotoImage | None = None
        self._pixel_region_photoimage: ImageTk.PhotoImage | None = None
        self._last_mouse_pos: tuple[int, int] = (-1, -1)
        self._has_valid_main = False
        self._has_valid_child = False
        self._best_color: tuple[int, int, int] = (0, 0, 0)
        self._best_tolerance = -1
        self._samples: dict[str, PixelSample] = {}
        self._refresh_after_id: str | None = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_widgets()
        self._bind_events()
        self._apply_theme()

    # -- public API -----------------------------------------------------

    def run(self) -> None:
        """Start the UI update loop and block until the window closes."""
        self._update_image()
        self.mainloop()

    # -- widget construction --------------------------------------------

    def _create_widgets(self) -> None:
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=0)
        self.main_frame.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self.main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        main_label = ttk.Label(top_bar, text="Main Window:")
        main_label.pack(side="left", padx=(0, 5))
        self.main_window_picker = ttk.Combobox(top_bar, values=(), state="readonly", width=30)
        self.main_window_picker.set("Select a main window")
        self.main_window_picker.pack(side="left", padx=(0, 5))

        refresh_button = ttk.Button(top_bar, text="Refresh", command=self._refresh_main_windows)
        refresh_button.pack(side="left", padx=(0, 10))

        child_label = ttk.Label(top_bar, text="Child Window:")
        child_label.pack(side="left", padx=(0, 5))
        self.child_window_picker = ttk.Combobox(top_bar, values=(), state="readonly", width=30)
        self.child_window_picker.set("Select a child window")
        self.child_window_picker.pack(side="left", padx=(0, 5))

        self.mark_best_color_var = tk.BooleanVar(value=False)
        mark_toggle = ttk.Checkbutton(top_bar, text="Mark Best Color Only", variable=self.mark_best_color_var)
        mark_toggle.pack(side="left", padx=(10, 5))

        image_frame = ttk.Frame(self.main_frame)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.image_label = ttk.Label(image_frame, text="Image goes here", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Motion>", self._on_mouse_move)
        self.image_label.bind("<Button-1>", self._on_mouse_click)

        right_frame = ttk.Frame(self.main_frame, width=180)
        right_frame.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=5)
        self.pixel_region_label = ttk.Label(right_frame)
        self.pixel_region_label.pack(pady=(5, 5))

        colours_label = ttk.Label(right_frame, text="Selected Colours", font=("Arial", 13))
        colours_label.pack(pady=(5, 0))

        self.colors_tree = ttk.Treeview(
            right_frame,
            columns=("coords", "rgb"),
            show="headings",
            selectmode="extended",
            height=10,
        )
        self.colors_tree.heading("coords", text="(row, col)")
        self.colors_tree.heading("rgb", text="(R, G, B)")
        self.colors_tree.column("coords", anchor="center", width=110)
        self.colors_tree.column("rgb", anchor="center", width=110)
        self.colors_tree.pack(fill="both", expand=True, padx=5, pady=5)

        delete_button = ttk.Button(right_frame, text="Delete Selected", command=self._on_delete_selected)
        delete_button.pack(pady=(0, 10))

        self.info_label = ttk.Label(right_frame, text=INFO_TEMPLATE.format("???", "???"))
        self.info_label.pack()

        self._refresh_main_windows()

    def _bind_events(self) -> None:
        self.main_window_picker.bind("<<ComboboxSelected>>", self._on_main_window_selected)
        self.child_window_picker.bind("<<ComboboxSelected>>", self._on_child_window_selected)

    # -- theme & shutdown ------------------------------------------------

    def _apply_theme(self) -> None:
        try:
            sv_ttk.set_theme("dark")
        except tk.TclError:
            pass

    def _on_close(self) -> None:
        if self._refresh_after_id is not None:
            self.after_cancel(self._refresh_after_id)
            self._refresh_after_id = None
        self.destroy()

    # -- window selection ------------------------------------------------

    def _refresh_main_windows(self) -> None:
        windows_with_hwnds = self.autocv.get_windows_with_hwnds()
        window_names = [name for _, name in windows_with_hwnds]
        self.main_window_picker["values"] = window_names
        if window_names:
            self.main_window_picker.set("Select a main window")
        else:
            self.main_window_picker.set("No windows detected")
        self.child_window_picker["values"] = []
        self.child_window_picker.set("Select a child window")
        self._has_valid_main = False
        self._has_valid_child = False
        self._clear_samples()

    def _on_main_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        selected_main_window = self.main_window_picker.get().strip()
        if not selected_main_window or selected_main_window.startswith("No "):
            return
        if not self.autocv.set_hwnd_by_title(selected_main_window):
            self._has_valid_main = False
            self._has_valid_child = False
            self.child_window_picker["values"] = []
            self.child_window_picker.set("Window not found")
            self._clear_samples()
            return

        self._has_valid_main = True
        self._has_valid_child = False
        self._clear_samples()

        child_windows = self.autocv.get_child_windows()
        child_names = [name for _, name in child_windows]
        self.child_window_picker["values"] = child_names
        if child_names:
            self.child_window_picker.set("Select a child window")
        else:
            self.child_window_picker.set("No child windows found")

    def _on_child_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        selected_child_window = self.child_window_picker.get().strip()
        if not selected_child_window or selected_child_window.startswith("No "):
            return

        changed = False
        while self.autocv.set_inner_hwnd_by_title(selected_child_window):
            changed = True
        self._has_valid_child = self._has_valid_main and (changed or self.autocv.hwnd != -1)
        if self._has_valid_child:
            self._clear_samples()

    # -- sample management -----------------------------------------------

    def _clear_samples(self) -> None:
        for item in self.colors_tree.get_children(""):
            self.colors_tree.delete(item)
        self._samples.clear()
        self._best_color = (0, 0, 0)
        self._best_tolerance = -1
        self.info_label.config(text=INFO_TEMPLATE.format("???", "???"))

    def _on_delete_selected(self) -> None:
        selection = self.colors_tree.selection()
        if not selection:
            return
        for item in selection:
            self.colors_tree.delete(item)
            self._samples.pop(item, None)
        self._update_best_color()

    def _add_sample(self, row: int, col: int, r_: int, g_: int, b_: int) -> None:
        sample = PixelSample(row=row, col=col, r=r_, g=g_, b=b_)
        item = self.colors_tree.insert(
            "",
            tk.END,
            values=(f"({row}, {col})", f"({r_}, {g_}, {b_})"),
        )
        self._samples[item] = sample
        self._update_best_color()

    def _update_best_color(self) -> None:
        if not self._samples:
            self._best_color = (0, 0, 0)
            self._best_tolerance = -1
            self.info_label.config(text=INFO_TEMPLATE.format("???", "???"))
            return

        colors = np.array([sample.colour for sample in self._samples.values()], dtype=np.uint16)
        max_color = np.amax(colors, axis=0)
        min_color = np.amin(colors, axis=0)
        best_color = ((max_color + min_color) // 2).astype(np.uint8)
        self._best_tolerance = int((max_color - min_color).max() + 1)
        self._best_color = (
            int(best_color[0]),
            int(best_color[1]),
            int(best_color[2]),
        )
        self.info_label.config(text=INFO_TEMPLATE.format(self._best_color, self._best_tolerance))

    # -- rendering -------------------------------------------------------

    def _update_image(self) -> None:
        frame_available = False
        if self._has_valid_main and self._has_valid_child:
            try:
                self.autocv.refresh()
            except OSError:
                self.image_label.config(text="Unable to capture image", image="")
            else:
                frame_available = self.autocv.opencv_image.size > 0

        if frame_available:
            self._draw_color_markers()
            image_rgb = Image.fromarray(self.autocv.opencv_image[..., ::-1])
            self._photo_image = ImageTk.PhotoImage(image_rgb)
            self.image_label.config(image=self._photo_image, text="")
        else:
            self._photo_image = None
            self.image_label.config(text="Image goes here", image="")

        self._show_3x3_region()
        self._refresh_after_id = self.after(UPDATE_INTERVAL_MS, self._update_image)

    def _draw_color_markers(self: Self) -> None:
        if not self._samples:
            return

        if self.mark_best_color_var.get():
            if self._best_tolerance >= 0:
                points = self.autocv.find_color(self._best_color, tolerance=self._best_tolerance)
                if points:
                    self.autocv.draw_points(points)
            return

        for sample in self._samples.values():
            points = self.autocv.find_color(sample.colour)
            if points:
                self.autocv.draw_points(points)

    def _get_mouse_coords_in_frame(self, event: tk.Event[ttk.Label]) -> tuple[int | None, int | None]:
        if self._photo_image is None or self.autocv.opencv_image.size == 0:
            return None, None

        frame = self.autocv.opencv_image
        frame_height, frame_width, _ = frame.shape
        disp_w = self._photo_image.width()
        disp_h = self._photo_image.height()
        if disp_w == 0 or disp_h == 0:
            return None, None

        mouse_x = min(max(event.x, 0), disp_w - 1)
        mouse_y = min(max(event.y, 0), disp_h - 1)

        ratio_x = frame_width / disp_w
        ratio_y = frame_height / disp_h

        col = int(mouse_x * ratio_x)
        row = int(mouse_y * ratio_y)
        if 0 <= row < frame_height and 0 <= col < frame_width:
            return row, col
        return None, None

    def _show_3x3_region(self) -> None:
        frame = self.autocv.opencv_image
        if frame.size == 0:
            self._pixel_region_photoimage = None
            self.pixel_region_label.config(image="", text="No region")
            return

        y, x = self._last_mouse_pos
        h, w, _ = frame.shape

        if x - PIXEL_RADIUS < 0 or y - PIXEL_RADIUS < 0 or x + PIXEL_RADIUS >= w or y + PIXEL_RADIUS >= h:
            cropped_image = np.full((3, 3, 3), DEFAULT_REGION_COLOUR, dtype=np.uint8)
        else:
            cropped_image = np.array(
                frame[y - PIXEL_RADIUS : y + PIXEL_RADIUS + 1, x - PIXEL_RADIUS : x + PIXEL_RADIUS + 1, ::-1],
                dtype=np.uint8,
            )

        resized_img = cv2.resize(cropped_image, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(resized_img)
        draw = ImageDraw.Draw(img)
        center_offset = PIXEL_RADIUS * ZOOM_SCALE
        rect = (
            center_offset,
            center_offset,
            center_offset + ZOOM_SCALE,
            center_offset + ZOOM_SCALE,
        )
        try:
            b, g, r_ = frame[y, x]
            pixel_color = (int(r_), int(g), int(b))
        except (IndexError, ValueError):
            pixel_color = (0, 0, 0)
        inverse_color: tuple[int, int, int] = (
            255 - pixel_color[0],
            255 - pixel_color[1],
            255 - pixel_color[2],
        )
        draw.rectangle(rect, outline=inverse_color, width=1)
        self._pixel_region_photoimage = ImageTk.PhotoImage(img)
        self.pixel_region_label.config(image=self._pixel_region_photoimage, text="")

    # -- event handlers --------------------------------------------------

    def _on_mouse_move(self, event: tk.Event[ttk.Label]) -> None:
        coords = self._get_mouse_coords_in_frame(event)
        if coords == (None, None):
            return
        row, col = coords
        if row is None or col is None:
            return
        self._last_mouse_pos = (row, col)
        self._show_3x3_region()

    def _on_mouse_click(self, event: tk.Event[ttk.Label]) -> None:
        coords = self._get_mouse_coords_in_frame(event)
        if coords == (None, None):
            return
        row, col = coords
        if row is None or col is None:
            return
        frame = self.autocv.opencv_image
        b, g, r_ = frame[row, col]
        self._add_sample(row, col, int(r_), int(g), int(b))


if __name__ == "__main__":
    AutoColorAid().run()
