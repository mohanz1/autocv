"""Color sampling assistant for AutoCV.

The :class:`~autocv.auto_color_aid.AutoColorAid` application streams frames from
an external window via :class:`~autocv.autocv.AutoCV` and allows users sample pixel
colors. Sampled colors are summarized as a "best color" with an associated
tolerance to help tune automation scripts.
"""

from __future__ import annotations

__all__ = ("AutoColorAid",)

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt
import sv_ttk
from PIL import Image, ImageDraw, ImageTk

from . import constants

try:
    from .autocv import AutoCV
except ImportError:  # pragma: no cover
    # Allow running this module directly in editable environments.
    from autocv.autocv import AutoCV

if TYPE_CHECKING:
    from collections.abc import Iterable


Color: TypeAlias = tuple[int, int, int]  # RGB
PixelSample: TypeAlias = tuple[int, int, int, int, int]  # row, col, R, G, B
ImageArray: TypeAlias = npt.NDArray[np.uint8]
Point: TypeAlias = tuple[int, int]

_REFRESH_DELAY_MS: Final[int] = constants.REFRESH_DELAY_MS
_ZOOM: Final[int] = constants.PIXEL_ZOOM
_PIXELS: Final[int] = constants.PIXEL_RADIUS
_FALLBACK_COLOR: Final[int] = constants.FALLBACK_COLOR

_LISTBOX_BEST_COLOR_TEXT: Final[str] = "best color: ???"
_LISTBOX_BEST_TOLERANCE_TEXT: Final[str] = "best tolerance: ???"
_PLACEHOLDER_IMAGE_TEXT: Final[str] = "Image goes here"
_NO_CAPTURE_TEXT: Final[str] = "No image captured"


def _new_pixel_samples() -> list[PixelSample]:
    return []


class _FrameShape(NamedTuple):
    height: int
    width: int
    channels: int


class MouseEvent(Protocol):
    """Protocol for tkinter mouse events carrying x/y coordinates."""

    x: int
    y: int


@dataclass(slots=True)
class ColorSelectionState:
    """Track selected pixels and compute aggregate colour statistics."""

    pixels: list[PixelSample] = field(default_factory=_new_pixel_samples)
    best_color: Color = (0, 0, 0)
    best_tolerance: int = -1

    def add(self, sample: PixelSample) -> None:
        """Append a pixel sample and recompute best color and tolerance."""
        self.pixels.append(sample)
        self._recompute()

    def remove_indices(self, indices: set[int]) -> None:
        """Remove samples whose indices are in ``indices`` and recompute state."""
        if not indices:
            return
        self.pixels = [sample for idx, sample in enumerate(self.pixels) if idx not in indices]
        self._recompute()

    def _recompute(self) -> None:
        """Recalculate best color and tolerance from the current pixel set."""
        if not self.pixels:
            self.best_color = (0, 0, 0)
            self.best_tolerance = -1
            return

        colors = np.array([sample[2:] for sample in self.pixels], dtype=np.uint16)
        max_color = np.amax(colors, axis=0)
        min_color = np.amin(colors, axis=0)

        best_color = ((max_color + min_color) // 2).astype(np.uint8)
        self.best_tolerance = int((max_color - min_color).max() + 1)
        self.best_color = (int(best_color[0]), int(best_color[1]), int(best_color[2]))


class AutoColorAidController:
    """Encapsulate AutoCV interactions and colour selection state."""

    def __init__(self) -> None:
        self.autocv = AutoCV()
        self.state = ColorSelectionState()

    def refresh_frame(self) -> None:
        """Refresh the underlying AutoCV backbuffer."""
        self.autocv.refresh()

    @property
    def frame(self) -> ImageArray:
        """Return the current OpenCV frame."""
        return self.autocv.opencv_image

    def window_titles(self) -> list[str]:
        """Return available top-level window titles."""
        return [name for _, name in self.autocv.get_windows_with_hwnds()]

    def child_titles(self) -> list[str]:
        """Return titles of child windows for the current main window."""
        return [name for _, name in self.autocv.get_child_windows()]

    def attach_main(self, title: str) -> bool:
        """Attach to a main window by title."""
        return self.autocv.set_hwnd_by_title(title)

    def attach_child(self, class_name: str) -> bool:
        """Attach to a child window by class title."""
        return self.autocv.set_inner_hwnd_by_title(class_name)

    def add_sample(self, row: int, col: int, bgr: Iterable[int]) -> None:
        """Record a pixel sample in RGB order."""
        b, g, r = (int(v) for v in bgr)
        self.state.add((row, col, r, g, b))

    def remove_samples(self, indices: set[int]) -> None:
        """Remove pixel samples by index."""
        self.state.remove_indices(indices)

    @property
    def best_color(self) -> Color:
        """Best computed colour in RGB."""
        return self.state.best_color

    @property
    def best_tolerance(self) -> int:
        """Best computed tolerance."""
        return self.state.best_tolerance

    def draw_markers(self, *, mark_best_only: bool) -> None:
        """Draw either best colour matches or all sampled colours."""
        if not self.state.pixels:
            return

        if mark_best_only:
            points = self.autocv.find_color(self.best_color, tolerance=self.best_tolerance)
            if points:
                self.autocv.draw_points(points)
            return

        for _, _, r_, g_, b_ in self.state.pixels:
            points = self.autocv.find_color((r_, g_, b_))
            if points:
                self.autocv.draw_points(points)


@dataclass(slots=True)
class _AutoColorAidWidgets:
    main_frame: ttk.Frame
    main_window_picker: ttk.Combobox
    child_window_picker: ttk.Combobox
    mark_best_color_var: tk.BooleanVar
    image_label: ttk.Label
    pixel_region_label: ttk.Label
    colors_listbox: ttk.Treeview
    info_label: ttk.Label


class AutoColorAid(tk.Tk):
    """Tkinter application for sampling colours from AutoCV-captured frames.

    Note:
        Instantiating this class starts the Tk main loop and blocks until the UI
        is closed.
    """

    def __init__(self) -> None:
        """Initialize the app, build widgets, and start the refresh loop."""
        super().__init__()
        self.title("Auto Color Aid")

        self.controller = AutoColorAidController()
        self._color_state = self.controller.state
        self._best_color: Color = self._color_state.best_color
        self._best_tolerance: int = self._color_state.best_tolerance

        self._photo_image: ImageTk.PhotoImage | None = None
        self._pixel_region_photoimage: ImageTk.PhotoImage | None = None
        self._last_mouse_pos: Point = (-1, -1)

        self._has_valid_main = False
        self._has_valid_child = False

        widgets = self._create_widgets()
        self.main_frame = widgets.main_frame
        self.main_window_picker = widgets.main_window_picker
        self.child_window_picker = widgets.child_window_picker
        self.mark_best_color_var = widgets.mark_best_color_var
        self.image_label = widgets.image_label
        self.pixel_region_label = widgets.pixel_region_label
        self.colors_listbox = widgets.colors_listbox
        self.info_label = widgets.info_label

        self._bind_events()

        sv_ttk.set_theme("dark")

        self._update_image()
        self.mainloop()

    def _create_widgets(self) -> _AutoColorAidWidgets:
        """Create and lay out the UI widgets."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        main_label = ttk.Label(top_bar, text="Main Window:")
        main_label.pack(side="left", padx=(0, 5))
        windows = self.controller.window_titles()
        main_window_picker = ttk.Combobox(top_bar, values=windows, state="readonly", width=25)
        main_window_picker.set("Select a main window")
        main_window_picker.pack(side="left", padx=(0, 5))

        refresh_button = ttk.Button(top_bar, text="Refresh", command=self._refresh_main_windows)
        refresh_button.pack(side="left", padx=(0, 10))

        child_label = ttk.Label(top_bar, text="Child Window:")
        child_label.pack(side="left", padx=(0, 5))
        child_window_picker = ttk.Combobox(top_bar, values=[], state="readonly", width=25)
        child_window_picker.set("Select a child window")
        child_window_picker.pack(side="left", padx=(0, 5))

        mark_best_color_var = tk.BooleanVar(value=False)
        mark_toggle = ttk.Checkbutton(top_bar, text="Mark Best Color Only", variable=mark_best_color_var)
        mark_toggle.pack(side="left", padx=(10, 5))

        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        image_label = ttk.Label(image_frame, text=_PLACEHOLDER_IMAGE_TEXT, anchor="center")
        image_label.pack(fill="both", expand=True)
        image_label.bind("<Motion>", self._on_mouse_move)
        image_label.bind("<Button-1>", self._on_mouse_click)

        right_frame = ttk.Frame(main_frame, width=150)
        right_frame.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=5)

        pixel_region_label = ttk.Label(right_frame)
        pixel_region_label.pack(pady=(5, 5))

        colors_label = ttk.Label(right_frame, text="Selected Colors", font=("Arial", 14))
        colors_label.pack(pady=(5, 0))

        colors_listbox = ttk.Treeview(right_frame, show="tree")
        colors_listbox.pack(fill="both", expand=True, padx=5, pady=5)

        delete_button = ttk.Button(right_frame, text="Delete Selected", command=self._on_delete_selected)
        delete_button.pack(pady=(0, 10))

        info_label = ttk.Label(right_frame, text=f"{_LISTBOX_BEST_COLOR_TEXT}\n{_LISTBOX_BEST_TOLERANCE_TEXT}")
        info_label.pack()

        return _AutoColorAidWidgets(
            main_frame=main_frame,
            main_window_picker=main_window_picker,
            child_window_picker=child_window_picker,
            mark_best_color_var=mark_best_color_var,
            image_label=image_label,
            pixel_region_label=pixel_region_label,
            colors_listbox=colors_listbox,
            info_label=info_label,
        )

    def _bind_events(self) -> None:
        """Bind widget events to their handlers."""
        self.main_window_picker.bind("<<ComboboxSelected>>", self._on_main_window_selected)
        self.child_window_picker.bind("<<ComboboxSelected>>", self._on_child_window_selected)

    def _refresh_main_windows(self) -> None:
        """Refresh the list of top-level windows and populate the picker."""
        self.main_window_picker["values"] = self.controller.window_titles()
        self.main_window_picker.set("Select a main window")

    def _on_main_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        """Handle selection events from the main-window picker."""
        selected_main_window = self.main_window_picker.get()
        if not selected_main_window:
            return

        self.controller.attach_main(selected_main_window)
        self._has_valid_main = True

        child_titles = self.controller.child_titles()
        if child_titles:
            self.child_window_picker["values"] = child_titles
            self.child_window_picker.set("Select a child window")
        else:
            self.child_window_picker["values"] = []
            self.child_window_picker.set("No child windows found")
            self._has_valid_child = False

    def _on_child_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        """Handle selection events from the child-window picker."""
        selected_child_window = self.child_window_picker.get()
        if not selected_child_window:
            return

        while self.controller.attach_child(selected_child_window):
            pass
        self._has_valid_child = True

    def _on_delete_selected(self) -> None:
        """Delete selected items from the colors listbox and refresh best-color computation."""
        selected_ids = set(self.colors_listbox.selection())
        if not selected_ids:
            return

        children = list(self.colors_listbox.get_children(""))
        removed_indices = {idx for idx, child in enumerate(children) if child in selected_ids}

        self._color_state.remove_indices(removed_indices)

        self.colors_listbox.delete(*children)
        for row, col, r_, g_, b_ in self._color_state.pixels:
            self._insert_listbox_item(row, col, r_, g_, b_, add_to_state=False)
        self._update_best_color()

    def _update_best_color(self) -> None:
        """Recompute and display the best color derived from selected pixels."""
        self._best_color = self._color_state.best_color
        self._best_tolerance = self._color_state.best_tolerance
        self._refresh_best_color_display()

    def _refresh_best_color_display(self) -> None:
        """Update the best-colour label from cached ``_best_*`` state."""
        if self._best_tolerance == -1:
            text = f"{_LISTBOX_BEST_COLOR_TEXT}\n{_LISTBOX_BEST_TOLERANCE_TEXT}"
        else:
            text = f"best color: {self._best_color}\nbest tolerance: {self._best_tolerance}"
        self.info_label.config(text=text)

    def _insert_listbox_item(self, row: int, col: int, r_: int, g_: int, b_: int, *, add_to_state: bool = True) -> None:
        """Insert a color and coordinate entry into the listbox and record it."""
        r_int, g_int, b_int = int(r_), int(g_), int(b_)
        text = f"({r_int}, {g_int}, {b_int}) @ ({row}, {col})"
        self.colors_listbox.insert("", tk.END, text=text)
        if add_to_state:
            self._color_state.add((row, col, r_int, g_int, b_int))

    def _draw_color_markers(self) -> None:
        """Overlay markers for either the best colour or every sampled colour."""
        if not self._color_state.pixels:
            return

        if self.mark_best_color_var.get():
            points = self.controller.autocv.find_color(self._best_color, tolerance=self._best_tolerance)
            if points:
                self.controller.autocv.draw_points(points)
            return

        for _, _, r_, g_, b_ in self._color_state.pixels:
            points = self.controller.autocv.find_color((r_, g_, b_))
            if points:
                self.controller.autocv.draw_points(points)

    def _update_image(self) -> None:
        """Refresh the displayed image and schedule the next update."""
        if self._has_valid_main and self._has_valid_child:
            self.controller.refresh_frame()
            frame = self.controller.frame
            if frame.size > 0:
                self._draw_color_markers()
                image_rgb = Image.fromarray(frame[..., ::-1])
                self._photo_image = ImageTk.PhotoImage(image_rgb)
                self.image_label.config(image=self._photo_image, text="")
            else:
                self.image_label.config(text=_NO_CAPTURE_TEXT, image="")
        else:
            self.image_label.config(text=_PLACEHOLDER_IMAGE_TEXT, image="")

        self._show_3x3_region()
        self.after(_REFRESH_DELAY_MS, self._update_image)

    def _get_mouse_coords_in_frame(self, event: MouseEvent) -> tuple[int | None, int | None]:
        """Convert mouse coordinates from the image label to frame coordinates."""
        if not (self._has_valid_main and self._has_valid_child):
            return None, None
        if self._photo_image is None or self.controller.frame.size == 0:
            return None, None

        frame = self.controller.frame
        frame_height, frame_width, _ = _FrameShape(*frame.shape)
        disp_w = self._photo_image.width()
        disp_h = self._photo_image.height()

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
        """Render a magnified 3x3 pixel preview around the last mouse position."""
        frame = self.controller.frame
        if frame.size == 0:
            self._pixel_region_photoimage = None
            self.pixel_region_label.config(image="", text="No region")
            return

        zoom = _ZOOM
        pixels = _PIXELS
        y, x = self._last_mouse_pos
        h, w, _ = frame.shape

        if x - pixels < 0 or y - pixels < 0 or x + pixels >= w or y + pixels >= h:
            cropped_image: ImageArray = np.full((3, 3, 3), _FALLBACK_COLOR, dtype=np.uint8)
        else:
            cropped_image = np.array(
                frame[y - pixels : y + pixels + 1, x - pixels : x + pixels + 1, ::-1],
                dtype=np.uint8,
            )

        resized_img = cv2.resize(cropped_image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(resized_img)
        draw = ImageDraw.Draw(img)

        center_top_left = (pixels * zoom, pixels * zoom)
        center_bottom_right = (center_top_left[0] + zoom, center_top_left[1] + zoom)

        try:
            b, g, r_ = frame[y, x]
            pixel_color: Color = (int(r_), int(g), int(b))
        except IndexError:
            pixel_color = (0, 0, 0)

        inverse_color: Color = (255 - pixel_color[0], 255 - pixel_color[1], 255 - pixel_color[2])
        draw.rectangle(
            (center_top_left[0], center_top_left[1], center_bottom_right[0], center_bottom_right[1]),
            outline=inverse_color,
            width=1,
        )
        self._pixel_region_photoimage = ImageTk.PhotoImage(img)
        self.pixel_region_label.config(image=self._pixel_region_photoimage, text="")

    def _on_mouse_move(self, event: MouseEvent) -> None:
        """Handle pointer motion to refresh the magnified pixel preview."""
        row, col = self._get_mouse_coords_in_frame(event)
        if row is not None and col is not None:
            self._last_mouse_pos = (row, col)
            self._show_3x3_region()

    def _on_mouse_click(self, event: MouseEvent) -> None:
        """Persist the colour under the cursor when the preview is clicked."""
        row, col = self._get_mouse_coords_in_frame(event)
        if row is None or col is None:
            return

        frame = self.controller.frame
        b, g, r_ = frame[row, col]
        self._insert_listbox_item(row, col, int(r_), int(g), int(b))
        self._update_best_color()


if __name__ == "__main__":
    AutoColorAid()
