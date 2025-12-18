from __future__ import annotations

import numpy as np
import pytest
from mock import MagicMock, patch

from autocv.auto_color_aid import (
    AutoColorAid,
    ColorSelectionState,
    _LISTBOX_BEST_COLOR_TEXT,
    _LISTBOX_BEST_TOLERANCE_TEXT,
    _NO_CAPTURE_TEXT,
    _PLACEHOLDER_IMAGE_TEXT,
    _REFRESH_DELAY_MS,
)


@pytest.fixture
def app():
    with (
        patch("autocv.auto_color_aid.AutoCV") as mock_autocv_cls,
        patch("autocv.auto_color_aid.tk.Tk.__init__", return_value=None),
        patch("autocv.auto_color_aid.tk.Tk.title"),
        patch("autocv.auto_color_aid.tk.BooleanVar"),
        patch("autocv.auto_color_aid.ttk.Frame"),
        patch("autocv.auto_color_aid.ttk.Label"),
        patch("autocv.auto_color_aid.ttk.Combobox"),
        patch("autocv.auto_color_aid.ttk.Button"),
        patch("autocv.auto_color_aid.ttk.Checkbutton"),
        patch("autocv.auto_color_aid.ttk.Treeview"),
        patch("autocv.auto_color_aid.sv_ttk.set_theme"),
        patch("autocv.auto_color_aid.ImageTk.PhotoImage") as mock_photo_image_cls,
        patch("autocv.auto_color_aid.AutoColorAid.mainloop"),
        patch("autocv.auto_color_aid.AutoColorAid.after", return_value=None),
    ):
        photo_image = MagicMock()
        photo_image.width.return_value = 100
        photo_image.height.return_value = 100
        mock_photo_image_cls.return_value = photo_image

        mock_autocv = mock_autocv_cls.return_value
        mock_autocv.get_windows_with_hwnds.return_value = []
        mock_autocv.get_child_windows.return_value = []
        mock_autocv.opencv_image = np.zeros((0, 0, 3), dtype=np.uint8)

        instance = AutoColorAid()
        yield instance


def test_insert_listbox_item_updates_state_and_widget(app):
    app.colors_listbox = MagicMock()
    app._color_state = ColorSelectionState()

    app._insert_listbox_item(1, 2, 3, 4, 5, add_to_state=True)
    assert app._color_state.pixels == [(1, 2, 3, 4, 5)]
    app.colors_listbox.insert.assert_called()

    app._insert_listbox_item(9, 9, 9, 9, 9, add_to_state=False)
    assert app._color_state.pixels == [(1, 2, 3, 4, 5)]


def test_refresh_best_color_display_formats_placeholder_and_value(app):
    app.info_label = MagicMock()

    app._best_tolerance = -1
    app._refresh_best_color_display()
    app.info_label.config.assert_called_with(text=f"{_LISTBOX_BEST_COLOR_TEXT}\n{_LISTBOX_BEST_TOLERANCE_TEXT}")

    app._best_color = (1, 2, 3)
    app._best_tolerance = 10
    app._refresh_best_color_display()
    app.info_label.config.assert_called_with(text="best color: (1, 2, 3)\nbest tolerance: 10")


def test_on_main_window_selected_populates_child_picker_and_flags(app):
    app.main_window_picker = MagicMock()
    app.child_window_picker = MagicMock()
    app.main_window_picker.get.return_value = "Main Window"
    app.controller.attach_main = MagicMock(return_value=True)
    app.controller.child_titles = MagicMock(return_value=["Child 1"])

    app._on_main_window_selected(MagicMock())
    assert app._has_valid_main is True
    app.child_window_picker.__setitem__.assert_called_with("values", ["Child 1"])
    app.child_window_picker.set.assert_called_with("Select a child window")


def test_on_main_window_selected_handles_no_children(app):
    app.main_window_picker = MagicMock()
    app.child_window_picker = MagicMock()
    app.main_window_picker.get.return_value = "Main Window"
    app.controller.attach_main = MagicMock(return_value=True)
    app.controller.child_titles = MagicMock(return_value=[])

    app._on_main_window_selected(MagicMock())
    assert app._has_valid_main is True
    assert app._has_valid_child is False
    app.child_window_picker.__setitem__.assert_called_with("values", [])
    app.child_window_picker.set.assert_called_with("No child windows found")


def test_on_child_window_selected_loops_until_false(app):
    app.child_window_picker = MagicMock()
    app.child_window_picker.get.return_value = "Child"
    app.controller.attach_child = MagicMock(side_effect=[True, False])

    app._on_child_window_selected(MagicMock())
    assert app.controller.attach_child.call_count == 2
    assert app._has_valid_child is True


def test_get_mouse_coords_in_frame_requires_ready_state(app):
    event = MagicMock(x=5, y=6)

    app._has_valid_main = False
    app._has_valid_child = False
    row, col = app._get_mouse_coords_in_frame(event)
    assert row is None and col is None

    app._has_valid_main = True
    app._has_valid_child = True
    app._photo_image = None
    row, col = app._get_mouse_coords_in_frame(event)
    assert row is None and col is None


def test_get_mouse_coords_in_frame_maps_and_clamps_coordinates(app):
    app._has_valid_main = True
    app._has_valid_child = True
    app.controller = MagicMock()
    app.controller.frame = np.zeros((20, 30, 3), dtype=np.uint8)
    app._photo_image = MagicMock()
    app._photo_image.width.return_value = 10
    app._photo_image.height.return_value = 10

    event = MagicMock(x=9, y=9)
    assert app._get_mouse_coords_in_frame(event) == (18, 27)

    clamped = MagicMock(x=-100, y=1000)
    assert app._get_mouse_coords_in_frame(clamped) == (18, 0)


def test_show_3x3_region_handles_empty_and_out_of_bounds(app):
    app.pixel_region_label = MagicMock()
    app.controller = MagicMock()
    app.controller.frame = np.zeros((0, 0, 3), dtype=np.uint8)

    app._show_3x3_region()
    app.pixel_region_label.config.assert_called_with(image="", text="No region")

    app.controller.frame = np.zeros((5, 5, 3), dtype=np.uint8)
    app._last_mouse_pos = (0, 0)
    app._show_3x3_region()
    assert app.pixel_region_label.config.call_count >= 2

    app._last_mouse_pos = (999, 999)
    app._show_3x3_region()


def test_draw_color_markers_marks_best_and_all_samples(app):
    app.controller = MagicMock()
    app.controller.autocv = MagicMock()
    app._color_state = ColorSelectionState(pixels=[(0, 0, 10, 20, 30), (1, 1, 11, 22, 33)])
    app._best_color = (10, 20, 30)
    app._best_tolerance = 5

    app.mark_best_color_var = MagicMock()
    app.mark_best_color_var.get.return_value = True
    app.controller.autocv.find_color.return_value = [(1, 2)]
    app._draw_color_markers()
    app.controller.autocv.draw_points.assert_called_with([(1, 2)])

    app.controller.autocv.find_color.reset_mock()
    app.controller.autocv.draw_points.reset_mock()
    app.mark_best_color_var.get.return_value = False
    app.controller.autocv.find_color.side_effect = [[(1, 2)], []]
    app._draw_color_markers()
    assert app.controller.autocv.find_color.call_count == 2
    app.controller.autocv.draw_points.assert_called_once_with([(1, 2)])


def test_update_image_covers_placeholder_and_capture_branches(app):
    app.after = MagicMock()
    app._show_3x3_region = MagicMock()
    app._draw_color_markers = MagicMock()
    app.image_label = MagicMock()
    app.controller = MagicMock()
    app.controller.refresh_frame = MagicMock()

    app._has_valid_main = False
    app._has_valid_child = False
    app.controller.frame = np.zeros((0, 0, 3), dtype=np.uint8)
    app._update_image()
    app.image_label.config.assert_called_with(text=_PLACEHOLDER_IMAGE_TEXT, image="")

    app._has_valid_main = True
    app._has_valid_child = True
    app.controller.frame = np.zeros((0, 0, 3), dtype=np.uint8)
    app._update_image()
    app.image_label.config.assert_called_with(text=_NO_CAPTURE_TEXT, image="")

    app.controller.frame = np.ones((2, 2, 3), dtype=np.uint8) * 255
    app._update_image()
    assert app.controller.refresh_frame.call_count >= 1
    app._draw_color_markers.assert_called()
    app.after.assert_called_with(_REFRESH_DELAY_MS, app._update_image)


def test_mouse_event_handlers_delegate_and_update_state(app):
    app._get_mouse_coords_in_frame = MagicMock(return_value=(1, 2))
    app._show_3x3_region = MagicMock()
    app._on_mouse_move(MagicMock(x=0, y=0))
    assert app._last_mouse_pos == (1, 2)
    app._show_3x3_region.assert_called_once()

    app._insert_listbox_item = MagicMock()
    app._update_best_color = MagicMock()
    app.controller = MagicMock()
    app.controller.frame = np.zeros((3, 3, 3), dtype=np.uint8)
    app._on_mouse_click(MagicMock(x=0, y=0))
    app._insert_listbox_item.assert_called_once()
    app._update_best_color.assert_called_once()
