"""A module providing a tkinter-based application that allows color selection and analysis from external windows.

This application captures frames from external windows via AutoCV, displays a live view,
and let's the user inspect and select pixel colors. Selected colors are stored and analyzed
to compute a "best color" based on the selected pixels.
"""

from __future__ import annotations

__all__ = ("AutoColorAid",)

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

import cv2
import numpy as np
import sv_ttk
from PIL import Image, ImageDraw, ImageTk
from typing_extensions import Self

from autocv import AutoCV

if TYPE_CHECKING:
    from numpy.typing import NDArray

Color = tuple[int, int, int]
PixelSample = tuple[int, int, int, int, int]  # row, col, R, G, B
_REFRESH_DELAY_MS: Final[int] = 5
_ZOOM: Final[int] = 40
_PIXELS: Final[int] = 1
_FALLBACK_COLOR: Final[int] = 217
_LISTBOX_BEST_COLOR_TEXT = "best color: ???"
_LISTBOX_BEST_TOLERANCE_TEXT = "best tolerance: ???"
_PLACEHOLDER_IMAGE_TEXT = "Image goes here"
_NO_CAPTURE_TEXT = "No image captured"


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

    pixels: list[PixelSample] = field(default_factory=list)
    best_color: Color = (0, 0, 0)
    best_tolerance: int = -1

    def add(self, sample: PixelSample) -> None:
        """Append a pixel sample to the collection."""
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

        colors = np.array([(r, g, b) for (_, _, r, g, b) in self.pixels], dtype=np.uint16)
        max_color = np.amax(colors, axis=0)
        min_color = np.amin(colors, axis=0)
        best_color = ((max_color + min_color) // 2).astype(np.uint8)
        self.best_tolerance = int((max_color - min_color).max() + 1)
        self.best_color = tuple(best_color.tolist())


class AutoColorAidController:
    """Encapsulate AutoCV interactions and colour selection state."""

    def __init__(self) -> None:
        self.autocv = AutoCV()
        self.state = ColorSelectionState()

    def refresh_frame(self) -> None:
        """Refresh the underlying AutoCV backbuffer."""
        self.autocv.refresh()

    @property
    def frame(self) -> NDArray[np.uint8]:
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

    def add_sample(self, row: int, col: int, bgr: tuple[int, int, int]) -> None:
        """Record a pixel sample in RGB order."""
        b, g, r = bgr
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


class AutoColorAid(tk.Tk):
    """Tkinter application for sampling colours from AutoCV-captured frames.

    Streams window frames via AutoCV, offers pixel inspection tools,
    and computes helper statistics for automation tuning.
    """

    def __init__(self) -> None:
        """Initialise the Tk window, AutoCV capture, and recurring refresh loop."""
        super().__init__()
        self.title("Auto Color Aid")

        # Controller encapsulates AutoCV and colour state.
        self.controller = AutoColorAidController()

        # Internal attributes to hold PhotoImage references (to prevent garbage collection).
        self._photo_image: ImageTk.PhotoImage | None = None
        self._pixel_region_photoimage: ImageTk.PhotoImage | None = None
        self._last_mouse_pos: tuple[int, int] = (-1, -1)

        # Flags to track if main and child windows are set.
        self._has_valid_main = False
        self._has_valid_child = False

        # Store selected pixel data as a list of tuples: (row, col, R, G, B).
        self._color_state = self.controller.state

        # Track best color and tolerance computed from selected pixels.
        self._best_color: Color = self._color_state.best_color
        self._best_tolerance: int = self._color_state.best_tolerance

        # Build UI widgets and bind events.
        self._create_widgets()
        self._bind_events()

        # Apply the dark theme.
        sv_ttk.set_theme("dark")

        # Start the periodic image update loop.
        self._update_image()

        # Start the main application loop.
        self.mainloop()

    def _create_widgets(self) -> None:
        """Create and lay out all UI widgets using grid geometry.

        The UI consists of a top bar (with window pickers and a refresh button),
        a main image display area, and a right panel for pixel preview and a list of selected colors.
        """
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Configure grid columns/rows.
        self.main_frame.columnconfigure(0, weight=1)  # Main image column.
        self.main_frame.columnconfigure(1, weight=0)  # Color list column.
        self.main_frame.rowconfigure(1, weight=1)

        # Top bar with window selection and controls.
        top_bar = ttk.Frame(self.main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        # Main window picker.
        main_label = ttk.Label(top_bar, text="Main Window:")
        main_label.pack(side="left", padx=(0, 5))
        windows = self.controller.window_titles()
        self.main_window_picker = ttk.Combobox(top_bar, values=windows, state="readonly", width=25)
        self.main_window_picker.set("Select a main window")
        self.main_window_picker.pack(side="left", padx=(0, 5))

        # Refresh button.
        refresh_button = ttk.Button(top_bar, text="Refresh", command=self._refresh_main_windows)
        refresh_button.pack(side="left", padx=(0, 10))

        # Child window picker.
        child_label = ttk.Label(top_bar, text="Child Window:")
        child_label.pack(side="left", padx=(0, 5))
        self.child_window_picker = ttk.Combobox(top_bar, values=[], state="readonly", width=25)
        self.child_window_picker.set("Select a child window")
        self.child_window_picker.pack(side="left", padx=(0, 5))

        # Toggle for best-color marking.
        self.mark_best_color_var = tk.BooleanVar(value=False)
        mark_toggle = ttk.Checkbutton(top_bar, text="Mark Best Color Only", variable=self.mark_best_color_var)
        mark_toggle.pack(side="left", padx=(10, 5))

        # Main image display area.
        image_frame = ttk.Frame(self.main_frame)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.image_label = ttk.Label(image_frame, text="Image goes here", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.image_label.bind("<Motion>", self._on_mouse_move)
        self.image_label.bind("<Button-1>", self._on_mouse_click)

        # Right panel for pixel preview and selected colors.
        right_frame = ttk.Frame(self.main_frame, width=150)
        right_frame.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=5)
        self.pixel_region_label = ttk.Label(right_frame)
        self.pixel_region_label.pack(pady=(5, 5))
        colors_label = ttk.Label(right_frame, text="Selected Colors", font=("Arial", 14))
        colors_label.pack(pady=(5, 0))
        self.colors_listbox = ttk.Treeview(right_frame, show="tree")
        self.colors_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        delete_button = ttk.Button(right_frame, text="Delete Selected", command=self._on_delete_selected)
        delete_button.pack(pady=(0, 10))
        self.info_label = ttk.Label(right_frame, text=f"{_LISTBOX_BEST_COLOR_TEXT}\n{_LISTBOX_BEST_TOLERANCE_TEXT}")
        self.info_label.pack()

    def _bind_events(self) -> None:
        """Bind events for widget interactions to their handlers."""
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

        # Update child window picker based on the selected main window.
        child_windows_names = self.controller.child_titles()
        if child_windows_names:
            self.child_window_picker["values"] = child_windows_names
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
        """Compute the best color from selected pixels and update the info display."""
        self._best_color = self._color_state.best_color
        self._best_tolerance = self._color_state.best_tolerance
        self._refresh_best_color_display()

    def _refresh_best_color_display(self) -> None:
        """Update the best-colour label from the cached state."""
        if self._best_tolerance == -1:
            text = f"{_LISTBOX_BEST_COLOR_TEXT}\n{_LISTBOX_BEST_TOLERANCE_TEXT}"
        else:
            text = f"best color: {self._best_color}\nbest tolerance: {self._best_tolerance}"
        self.info_label.config(text=text)

    def _insert_listbox_item(self, row: int, col: int, r_: int, g_: int, b_: int, *, add_to_state: bool = True) -> None:
        """Insert a color and coordinate entry into the colors listbox and record it."""
        text = f"({r_}, {g_}, {b_}) @ ({row}, {col})"
        self.colors_listbox.insert("", tk.END, text=text)
        if add_to_state:
            self._color_state.add((row, col, r_, g_, b_))

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

    def _draw_color_markers(self: Self) -> None:
        """Overlay markers for either the best colour or every sampled colour."""
        if not self._color_state.pixels:
            return

        if self.mark_best_color_var.get():
            points = self.controller.autocv.find_color(self._best_color, tolerance=self._best_tolerance)
            if points:
                self.controller.autocv.draw_points(points)
        else:
            for _, _, r_, g_, b_ in self._color_state.pixels:
                points = self.controller.autocv.find_color((r_, g_, b_))
                if points:
                    self.controller.autocv.draw_points(points)

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

        cropped_image: NDArray[Any]
        if x - pixels < 0 or y - pixels < 0 or x + pixels >= w or y + pixels >= h:
            cropped_image = np.ones((3, 3, 3), dtype=np.uint8) * _FALLBACK_COLOR
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
            pixel_color = (r_, g, b)
        except IndexError:
            pixel_color = (0, 0, 0)
        inverse_color = tuple(255 - c for c in pixel_color)
        draw.rectangle(
            (center_top_left[0], center_top_left[1], center_bottom_right[0], center_bottom_right[1]),
            outline=inverse_color,
            width=1,
        )
        self._pixel_region_photoimage = ImageTk.PhotoImage(img)
        self.pixel_region_label.config(image=self._pixel_region_photoimage, text="")

    def _on_mouse_move(self, event: MouseEvent) -> None:
        """Handle pointer motion to refresh the magnified pixel preview.

        Args:
            event: Mouse move event from the preview label.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is not None and col is not None:
            self._last_mouse_pos = (row, col)
            self._show_3x3_region()

    def _on_mouse_click(self, event: MouseEvent) -> None:
        """Persist the colour under the cursor when the preview is clicked.

        Args:
            event: Mouse click event from the preview label.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is None or col is None:
            return
        frame = self.controller.frame
        b, g, r_ = frame[row, col]
        self._insert_listbox_item(row, col, r_, g, b)
        self._update_best_color()


if __name__ == "__main__":
    app = AutoColorAid()
