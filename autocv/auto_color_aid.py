"""A module providing a tkinter-based application that allows color selection and analysis from external windows.

This application captures frames from external windows via AutoCV, displays a live view,
and let's the user inspect and select pixel colors. Selected colors are stored and analyzed
to compute a "best color" based on the selected pixels.
"""

from __future__ import annotations

__all__ = ("AutoColorAid",)

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
import sv_ttk
from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk

from autocv import AutoCV

if TYPE_CHECKING:
    from PIL.ImageTk import PhotoImage


class AutoColorAid(tk.Tk):
    """GUI application for color selection and analysis from captured windows.

    This application uses AutoCV to capture images from external windows, displays
    the captured frame, and provides pixel-level inspection (including a zoomed 3x3 region)
    along with a list of selected colors and computed "best color" information.
    """

    def __init__(self) -> None:
        """Initializes the main application window.

        Sets up the tkinter root window, initializes an AutoCV object for capturing frames,
        creates the UI widgets, binds event handlers, and starts the periodic image update loop.
        """
        super().__init__()
        self.title("Auto Color Aid")

        # Create an instance of AutoCV for capturing frames.
        self.autocv = AutoCV()

        # Internal attributes to hold PhotoImage references (to prevent garbage collection).
        self._photo_image: PhotoImage
        self._pixel_region_photoimage: ImageTk.PhotoImage | None = None
        self._last_mouse_pos = (-1, -1)

        # Flags to track if main and child windows are set.
        self._has_valid_main = False
        self._has_valid_child = False

        # Store selected pixel data as a list of tuples: (row, col, R, G, B).
        self.selected_pixels: list[tuple[int, int, int, int, int]] = []

        # Track best color and tolerance computed from selected pixels.
        self._best_color = (0, 0, 0)
        self._best_tolerance = -1

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
        """Creates and lays out all UI widgets using grid geometry.

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
        windows_with_hwnds = self.autocv.get_windows_with_hwnds()
        windows = [name for (_, name) in windows_with_hwnds]
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
        self.info_label = ttk.Label(right_frame, text="best color: ???\nbest tolerance: ???")
        self.info_label.pack()

    def _bind_events(self) -> None:
        """Binds events for widget interactions to their handlers."""
        self.main_window_picker.bind("<<ComboboxSelected>>", self._on_main_window_selected)
        self.child_window_picker.bind("<<ComboboxSelected>>", self._on_child_window_selected)

    def _refresh_main_windows(self) -> None:
        """Refreshes the list of main windows and updates the main window picker."""
        windows_with_hwnds = self.autocv.get_windows_with_hwnds()
        windows = [name for (_, name) in windows_with_hwnds]
        self.main_window_picker["values"] = windows
        self.main_window_picker.set("Select a main window")

    def _on_main_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        """Handles selection of a main window from the combo box.

        Args:
            _ (tk.Event): The event object (unused).
        """
        selected_main_window = self.main_window_picker.get()
        if not selected_main_window:
            return

        self.autocv.set_hwnd_by_title(selected_main_window)
        self._has_valid_main = True

        # Update child window picker based on the selected main window.
        child_windows = self.autocv.get_child_windows()
        child_windows_names = [name for (_, name) in child_windows]
        if child_windows_names:
            self.child_window_picker["values"] = child_windows_names
            self.child_window_picker.set("Select a child window")
        else:
            self.child_window_picker["values"] = []
            self.child_window_picker.set("No child windows found")
            self._has_valid_child = False

    def _on_child_window_selected(self, _: tk.Event[ttk.Combobox]) -> None:
        """Handles selection of a child window from the combo box.

        Args:
            _ (tk.Event): The event object (unused).
        """
        selected_child_window = self.child_window_picker.get()
        if not selected_child_window:
            return

        while self.autocv.set_inner_hwnd_by_title(selected_child_window):
            pass
        self._has_valid_child = True

    def _on_delete_selected(self) -> None:
        """Deletes selected items from the colors listbox and updates the selected pixels list.

        Rebuilds the list of selected pixels by excluding deleted items, then refreshes the best-color computation.
        """
        selected_items = self.colors_listbox.selection()
        remaining = []
        all_items = self.colors_listbox.get_children("")

        for child in all_items:
            if child not in selected_items:
                text = self.colors_listbox.item(child, "text")
                color_part, coord_part = text.split("@")
                color_str = color_part.strip().strip("()")
                coords_str = coord_part.strip().strip("()")
                r_, g_, b_ = [int(x) for x in color_str.split(",")]
                row, col = [int(x) for x in coords_str.split(",")]
                remaining.append((row, col, r_, g_, b_))

        self.colors_listbox.delete(*all_items)
        self.selected_pixels = []
        for row, col, r_, g_, b_ in remaining:
            self._insert_listbox_item(row, col, r_, g_, b_)
        self._update_best_color()

    def _update_best_color(self) -> None:
        """Computes the best color from selected pixels and updates the info display.

        The best color is computed as the midpoint (per channel) of the minimum and maximum RGB values,
        and the best tolerance is the maximum difference among any channel plus one.
        """
        if not self.selected_pixels:
            self._best_color = (0, 0, 0)
            self._best_tolerance = -1
            self.info_label.config(text="best color: ???\nbest tolerance: ???")
            return

        colors = np.array([(r, g, b) for (_, _, r, g, b) in self.selected_pixels], dtype=np.uint16)
        max_color = np.amax(colors, axis=0)
        min_color = np.amin(colors, axis=0)
        best_color = ((max_color + min_color) // 2).astype(np.uint8)
        self._best_tolerance = int((max_color - min_color).max() + 1)
        self._best_color = tuple(best_color.tolist())
        self.info_label.config(text=f"best color: {self._best_color}\nbest tolerance: {self._best_tolerance}")

    def _insert_listbox_item(self, row: int, col: int, r_: int, g_: int, b_: int) -> None:
        """Inserts a color and coordinate entry into the colors listbox and records it.

        Args:
            row (int): The y-coordinate of the pixel.
            col (int): The x-coordinate of the pixel.
            r_ (int): Red channel value.
            g_ (int): Green channel value.
            b_ (int): Blue channel value.
        """
        text = f"({r_}, {g_}, {b_}) @ ({row}, {col})"
        self.colors_listbox.insert("", tk.END, text=text)
        self.selected_pixels.append((row, col, r_, g_, b_))

    def _update_image(self) -> None:
        """Refreshes the displayed image and schedules the next update.

        Updates the main image display with the latest frame captured via AutoCV,
        draws markers for selected pixels, updates the zoomed 3x3 region preview,
        and reschedules the update.
        """
        if self._has_valid_main and self._has_valid_child:
            self.autocv.refresh()
            if self.autocv.opencv_image.size > 0:
                self._draw_color_markers()
                image_rgb = Image.fromarray(self.autocv.opencv_image[..., ::-1])
                self._photo_image = ImageTk.PhotoImage(image_rgb)
                self.image_label.config(image=self._photo_image, text="")
            else:
                self.image_label.config(text="No image captured", image="")
        else:
            self.image_label.config(text="Image goes here", image="")

        self._show_3x3_region()
        self.after(5, self._update_image)

    def _draw_color_markers(self) -> None:
        """Draws markers on the image for each selected pixel.

        If the "Mark Best Color Only" toggle is enabled, markers are drawn using the computed best color.
        Otherwise, each pixel is marked using its own color.
        """
        if not self.selected_pixels:
            return

        if self.mark_best_color_var.get():
            points = self.autocv.find_color(self._best_color, tolerance=self._best_tolerance)
            if points:
                self.autocv.draw_points(points)
        else:
            for _, _, r_, g_, b_ in self.selected_pixels:
                points = self.autocv.find_color((r_, g_, b_))
                if points:
                    self.autocv.draw_points(points)

    def _get_mouse_coords_in_frame(self, event: tk.Event[ttk.Label]) -> tuple[int | None, int | None]:
        """Converts mouse coordinates from the image label to frame coordinates in the captured image.

        Args:
            event (tk.Event): The mouse event containing x and y positions on the label.

        Returns:
            tuple[int | None, int | None]: The (row, col) coordinates in the frame, or (None, None) if out of bounds.
        """
        if not (self._has_valid_main and self._has_valid_child):
            return None, None
        if not hasattr(self, "_photo_image") or self.autocv.opencv_image.size == 0:
            return None, None

        frame = self.autocv.opencv_image
        frame_height, frame_width, _ = frame.shape
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
        """Extracts a 3x3 pixel region around the last mouse position and displays it in the right panel.

        If the region is out of bounds, a default gray image is displayed.
        """
        if self.autocv.opencv_image.size == 0:
            self._pixel_region_photoimage = None
            self.pixel_region_label.config(image="", text="No region")
            return

        zoom = 40
        pixels = 1
        y, x = self._last_mouse_pos
        frame = self.autocv.opencv_image
        h, w, _ = frame.shape

        if x - pixels < 0 or y - pixels < 0 or x + pixels >= w or y + pixels >= h:
            cropped_image = np.ones((3, 3, 3), dtype=np.uint8) * 217
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

    def _on_mouse_move(self, event: tk.Event[ttk.Label]) -> None:
        """Handles mouse movement over the main image label to update pixel preview.

        Args:
            event (tk.Event): The mouse event containing position data.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is not None and col is not None:
            self._last_mouse_pos = (row, col)
            self._show_3x3_region()

    def _on_mouse_click(self, event: tk.Event[ttk.Label]) -> None:
        """Handles mouse clicks on the main image label to record the selected color.

        Retrieves the color at the clicked position, inserts it into the listbox,
        and updates the best color computation.

        Args:
            event (tk.Event): The mouse event containing the click position.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is None or col is None:
            return
        frame = self.autocv.opencv_image
        b, g, r_ = frame[row, col]
        self._insert_listbox_item(row, col, r_, g, b)
        self._update_best_color()


if __name__ == "__main__":
    app = AutoColorAid()
