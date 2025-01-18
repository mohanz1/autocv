"""A module providing a tkinter-based application that allows color selection and analysis from external windows."""

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import sv_ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk

from autocv import AutoCV


class AutoColorAid(tk.Tk):
    """A GUI application for color selection and analysis from captured windows."""

    def __init__(self) -> None:
        """Initialize the main application window.

        Sets up the tkinter root window, initializes the AutoCV object for capturing
        frames from external windows, creates all UI widgets, binds event handlers,
        and starts the periodic image update loop.
        """
        super().__init__()
        self.title("Auto Color Aid")

        # Create an instance of AutoCV for retrieving HWNDs, etc.
        self.autocv = AutoCV()

        # Internal attributes for tracking current PhotoImage (prevent GC).
        self._photo_image: ImageTk.PhotoImage
        self._pixel_region_photoimage = None
        self._last_mouse_pos = (-1, -1)

        # Track whether main and child windows are set.
        self._has_valid_main = False
        self._has_valid_child = False

        # Store each clicked color + coordinate as a list of tuples: (row, col, R, G, B).
        self.selected_pixels: list[tuple[int, int, int, int, int]] = []

        # Track the best color (computed by _update_best_color).
        self._best_color = (0, 0, 0)
        self._best_tolerance = -1

        # Build the UI.
        self._create_widgets()
        self._bind_events()

        # Apply the dark theme from sv_ttk.
        sv_ttk.set_theme("dark")

        # Start the image refresh loop.
        self._update_image()

        # Start the main application loop.
        self.mainloop()

    def _create_widgets(self) -> None:
        """Create and lay out all widgets in the application using grid geometry.

        This includes the top bar (combo boxes for main/child windows, refresh button,
        and the best-color toggle), the main image display area, and the right
        panel for pixel preview and selected colors.
        """
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Configure grid columns/rows.
        self.main_frame.columnconfigure(0, weight=1)  # for image
        self.main_frame.columnconfigure(1, weight=0)  # for color list
        self.main_frame.rowconfigure(1, weight=1)

        # Top bar (combo boxes + refresh button).
        top_bar = ttk.Frame(self.main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        # Main Window Label.
        main_label = ttk.Label(top_bar, text="Main Window:")
        main_label.pack(side="left", padx=(0, 5))

        # Main Window Picker.
        windows_with_hwnds = self.autocv.get_windows_with_hwnds()
        windows = [name for (_, name) in windows_with_hwnds]
        self.main_window_picker = ttk.Combobox(top_bar, values=windows, state="readonly", width=25)
        self.main_window_picker.set("Select a main window")
        self.main_window_picker.pack(side="left", padx=(0, 5))

        # Refresh Button.
        refresh_button = ttk.Button(top_bar, text="Refresh", command=self._refresh_main_windows)
        refresh_button.pack(side="left", padx=(0, 10))

        # Child Window Label.
        child_label = ttk.Label(top_bar, text="Child Window:")
        child_label.pack(side="left", padx=(0, 5))

        # Child Window Picker.
        self.child_window_picker = ttk.Combobox(top_bar, values=[], state="readonly", width=25)
        self.child_window_picker.set("Select a child window")
        self.child_window_picker.pack(side="left", padx=(0, 5))

        # Checkbutton for toggling best color vs. individual colors.
        self.mark_best_color_var = tk.BooleanVar(value=False)
        mark_toggle = ttk.Checkbutton(top_bar, text="Mark Best Color Only", variable=self.mark_best_color_var)
        mark_toggle.pack(side="left", padx=(10, 5))

        # Image area.
        image_frame = ttk.Frame(self.main_frame)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.image_label = ttk.Label(image_frame, text="Image goes here", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        # Bind events for pixel inspection/click.
        self.image_label.bind("<Motion>", self._on_mouse_move)
        self.image_label.bind("<Button-1>", self._on_mouse_click)

        # Right frame for pixel preview & selected colors.
        right_frame = ttk.Frame(self.main_frame, width=150)
        right_frame.grid(row=1, column=1, sticky="ns", padx=(0, 5), pady=5)

        # Zoomed pixel region preview.
        self.pixel_region_label = ttk.Label(right_frame)
        self.pixel_region_label.pack(pady=(5, 5))

        # Selected Colors list.
        colors_label = ttk.Label(right_frame, text="Selected Colors", font=("Arial", 14))
        colors_label.pack(pady=(5, 0))

        self.colors_listbox = ttk.Treeview(right_frame, show="tree")
        self.colors_listbox.pack(fill="both", expand=True, padx=5, pady=5)

        delete_button = ttk.Button(right_frame, text="Delete Selected", command=self._on_delete_selected)
        delete_button.pack(pady=(0, 10))

        self.info_label = ttk.Label(right_frame, text="best color: ???\nbest tolerance: ???")
        self.info_label.pack()

    def _bind_events(self) -> None:
        """Bind widget events (combobox selection) to their respective handlers."""
        self.main_window_picker.bind("<<ComboboxSelected>>", self._on_main_window_selected)
        self.child_window_picker.bind("<<ComboboxSelected>>", self._on_child_window_selected)

    def _refresh_main_windows(self) -> None:
        """Fetch the current main windows and update the main window combo box.

        Clears the combo box and repopulates it with any top-level windows found
        by the AutoCV instance.
        """
        windows_with_hwnds = self.autocv.get_windows_with_hwnds()
        windows = [name for (_, name) in windows_with_hwnds]
        self.main_window_picker["values"] = windows
        self.main_window_picker.set("Select a main window")

    def _on_main_window_selected(self, _: "tk.Event[ttk.Combobox]") -> None:
        """Event handler for when the user picks a main window.

        Args:
            _: The event object (unused).
        """
        selected_main_window = self.main_window_picker.get()
        if not selected_main_window:
            return

        self.autocv.set_hwnd_by_title(selected_main_window)
        self._has_valid_main = True

        # Get all child windows for that main HWND.
        child_windows = self.autocv.get_child_windows()
        child_windows_names = [name for (_, name) in child_windows]

        if child_windows_names:
            self.child_window_picker["values"] = child_windows_names
            self.child_window_picker.set("Select a child window")
        else:
            self.child_window_picker["values"] = []
            self.child_window_picker.set("No child windows found")
            self._has_valid_child = False

    def _on_child_window_selected(self, _: "tk.Event[ttk.Combobox]") -> None:
        """Event handler for when the user picks a child window.

        Args:
            _: The event object (unused).
        """
        selected_child_window = self.child_window_picker.get()
        if not selected_child_window:
            return

        # If multiple child windows have the same title, set them all.
        while self.autocv.set_inner_hwnd_by_title(selected_child_window):
            pass
        self._has_valid_child = True

    def _on_delete_selected(self) -> None:
        """Delete all selected items from the colors_listbox and from self.selected_pixels.

        Rebuilds self.selected_pixels by excluding any listbox items that are selected,
        then clears and repopulates the listbox with the remaining items.
        """
        selected_items = self.colors_listbox.selection()
        remaining = []
        all_items = self.colors_listbox.get_children("")

        # Rebuild self.selected_pixels from items that are not deleted.
        for child in all_items:
            if child not in selected_items:
                text = self.colors_listbox.item(child, "text")
                # We stored them as "({r}, {g}, {b}) @ ({row}, {col})".
                color_part, coord_part = text.split("@")
                color_str = color_part.strip().strip("()")
                coords_str = coord_part.strip().strip("()")
                r_, g_, b_ = [int(x) for x in color_str.split(",")]
                row, col = [int(x) for x in coords_str.split(",")]
                remaining.append((row, col, r_, g_, b_))

        # Clear and reinsert the non-deleted items.
        self.colors_listbox.delete(*all_items)
        self.selected_pixels = []
        for row, col, r_, g, b in remaining:
            self._insert_listbox_item(row, col, r_, g, b)

        # Recompute best color.
        self._update_best_color()

    def _update_best_color(self) -> None:
        """Compute the "best color" from the selected pixels.

        The best color is chosen as the midpoint (per channel) of the min and max RGB
        values from the selected pixels. The best tolerance is the maximum difference
        among any channel + 1.

        Updates self._best_color, self._best_tolerance, and self.info_label text.
        """
        if not self.selected_pixels:
            self._best_color = (0, 0, 0)
            self._best_tolerance = -1
            self.info_label.config(text="best color: ???\nbest tolerance: ???")
            return

        # Convert to an array of (R,G,B).
        colors = np.array([(r, g, b) for (_, _, r, g, b) in self.selected_pixels], dtype=np.uint16)
        max_color = np.amax(colors, axis=0)
        min_color = np.amin(colors, axis=0)
        best_color = ((max_color + min_color) // 2).astype(np.uint8)
        self._best_tolerance = (max_color - min_color).max() + 1

        self._best_color = tuple(best_color.tolist())  # (R, G, B).
        self.info_label.config(text=f"best color: {self._best_color}\nbest tolerance: {self._best_tolerance}")

    def _insert_listbox_item(self, row: int, col: int, r_: int, g_: int, b_: int) -> None:
        """Insert a color + coordinate item into the colors_listbox and store it in memory.

        Args:
            row (int): The row (y-coordinate) of the pixel.
            col (int): The column (x-coordinate) of the pixel.
            r_ (int): The red channel value.
            g_ (int): The green channel value.
            b_ (int): The blue channel value.
        """
        text = f"({r_}, {g_}, {b_}) @ ({row}, {col})"
        self.colors_listbox.insert("", tk.END, text=text)
        self.selected_pixels.append((row, col, r_, g_, b_))

    def _update_image(self) -> None:
        """Refresh the displayed image (if valid) and schedule the next update.

        Draws color markers on the captured frame (if any are selected), converts the
        frame from BGR to RGB, and displays it on image_label. Also updates the 3x3
        zoomed pixel region and then calls itself again after a short delay.
        """
        if self._has_valid_main and self._has_valid_child:
            self.autocv.refresh()

            if self.autocv.opencv_image.size > 0:
                # Draw circles for the selected colors.
                self._draw_color_markers()

                # Convert BGR->RGB and display.
                image_rgb = Image.fromarray(self.autocv.opencv_image[..., ::-1])
                self._photo_image = ImageTk.PhotoImage(image_rgb)
                self.image_label.config(image=self._photo_image, text="")
            else:
                self.image_label.config(text="No image captured", image="")
        else:
            self.image_label.config(text="Image goes here", image="")

        # Update the 3x3 region in the right panel.
        self._show_3x3_region()

        # Schedule the next frame update (~every 5 ms -> up to ~200 FPS if resources allow).
        self.after(5, self._update_image)

    def _draw_color_markers(self) -> None:
        """Draw small circles for each selected pixel.

        Either draws them in each pixel's own color, or in the "best color" if the
        Mark Best Color Only toggle is checked. The circles are drawn directly
        onto self.autocv.opencv_image using AutoCV's drawing methods.
        """
        if not self.selected_pixels:
            return

        use_best_color = self.mark_best_color_var.get()

        if use_best_color:
            # Use the best color for all.
            points = self.autocv.find_color(self._best_color, tolerance=self._best_tolerance)
            if points:
                self.autocv.draw_points(points)
        else:
            # Use each pixel's own color individually.
            for _, _, r_, g_, b_ in self.selected_pixels:
                points = self.autocv.find_color((r_, g_, b_))
                if points:
                    self.autocv.draw_points(points)

    def _get_mouse_coords_in_frame(self, event: "tk.Event[ttk.Label]") -> tuple[int | None, int | None]:
        """Convert the mouse position (event.x, event.y) on image_label into (row, col) in self.autocv.opencv_image.

        If invalid or out of bounds, returns (None, None).

        Args:
            event ("tk.Event[tk.Tk]"): The Tkinter event containing mouse position data.

        Returns:
            tuple[int | None, int | None]: (row, col) in the image, or (None, None)
            if out of bounds.
        """
        if not (self._has_valid_main and self._has_valid_child):
            return None, None
        if self._photo_image is None or self.autocv.opencv_image.size == 0:
            return None, None

        frame = self.autocv.opencv_image
        frame_height, frame_width, _ = frame.shape

        # Dimensions of the displayed image.
        disp_w = self._photo_image.width()
        disp_h = self._photo_image.height()

        # Mouse coords in label.
        mouse_x = event.x
        mouse_y = event.y

        # Clamp to display bounds.
        mouse_x = min(max(mouse_x, 0), disp_w - 1)
        mouse_y = min(max(mouse_y, 0), disp_h - 1)

        # For 1:1 display scale (assuming displayed image is the same size as frame).
        ratio_x = frame_width / disp_w
        ratio_y = frame_height / disp_h

        col = int(mouse_x * ratio_x)
        row = int(mouse_y * ratio_y)

        if 0 <= row < frame_height and 0 <= col < frame_width:
            return row, col

        return None, None

    def _show_3x3_region(self) -> None:
        """Extract the 3x3 region around the last mouse position from self.autocv.opencv_image, then display it.

        If out of bounds or no frame is available, shows a fallback gray region.
        """
        if self.autocv.opencv_image.size == 0:
            self._pixel_region_photoimage = None
            self.pixel_region_label.config(image="", text="No region")
            return

        zoom = 40  # Scale factor for each pixel.
        pixels = 1  # We want a 3x3 region (center ± 1 in each direction).
        y, x = self._last_mouse_pos

        frame = self.autocv.opencv_image
        h, w, _ = frame.shape

        # Check boundaries for a valid 3x3 region.
        if x - pixels < 0 or y - pixels < 0 or x + pixels >= w or y + pixels >= h:
            # Out of bounds => create a gray fallback image.
            cropped_image = np.ones((3, 3, 3), dtype=np.uint8) * 217
        else:
            # BGR->RGB by slicing [::-1]
            cropped_image = frame[y - pixels : y + pixels + 1, x - pixels : x + pixels + 1, ::-1]  # type: ignore[assignment]

        # Scale up via OpenCV using nearest-neighbor.
        img = cv2.resize(cropped_image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

        # Convert to PIL Image.
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # Draw a rectangle around the center pixel.
        center_top_left = (pixels * zoom, pixels * zoom)  # e.g. (40, 40) if zoom=40.
        center_bottom_right = (center_top_left[0] + zoom, center_top_left[1] + zoom)

        try:
            b, g, r_ = frame[y, x]
            pixel_color = (r_, g, b)
        except IndexError:
            pixel_color = (0, 0, 0)

        # Invert the pixel color for the outline color.
        inverse_color = tuple(255 - c for c in pixel_color)
        draw.rectangle(
            (center_top_left[0], center_top_left[1], center_bottom_right[0], center_bottom_right[1]),
            outline=inverse_color,
            width=1,
        )

        # Attempt to load a TTF font; fallback to default if not available.
        font_path = Path(__file__).parent / "data" / "Helvetica.ttf"
        font = ImageFont.truetype(str(font_path), 8) if font_path.exists() else ImageFont.load_default()

        # Place coordinate text near the bottom of the zoomed region.
        text_x = img.width // 2
        text_y = img.height - zoom // 2
        draw.text((text_x, text_y), f"{x},{y}", fill=inverse_color, font=font, anchor="mm")

        # Convert to PhotoImage and update the label.
        self._pixel_region_photoimage = ImageTk.PhotoImage(img)  # type: ignore[assignment]
        self.pixel_region_label.config(image=self._pixel_region_photoimage, text="")  # type: ignore[call-overload]

    def _on_mouse_move(self, event: "tk.Event[ttk.Label]") -> None:
        """Event handler for mouse movement over the main image label.

        Updates the _last_mouse_pos with the correct (row, col) from the frame,
        then calls _show_3x3_region to refresh the zoomed region.

        Args:
            event ("tk.Event[tk.Tk]"): The Tkinter mouse event object.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is not None and col is not None:
            self._last_mouse_pos = (row, col)
            self._show_3x3_region()

    def _on_mouse_click(self, event: "tk.Event[ttk.Label]") -> None:
        """Event handler for mouse clicks on the main image label.

        Determines the clicked pixel's color (in BGR from the frame), inserts it
        into the listbox, and updates the best color computation.

        Args:
            event ("tk.Event[tk.Tk]"): The Tkinter mouse event object.
        """
        row, col = self._get_mouse_coords_in_frame(event)
        if row is None or col is None:
            return

        frame = self.autocv.opencv_image
        b, g, r_ = frame[row, col]  # BGR format.

        # Insert item into the listbox and update best color.
        self._insert_listbox_item(row, col, r_, g, b)
        self._update_best_color()


if __name__ == "__main__":
    app = AutoColorAid()
