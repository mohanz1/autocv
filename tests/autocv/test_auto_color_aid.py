import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from autocv.color_picker import ColorPicker


@pytest.fixture
def dummy_tk():
    return MagicMock()


@pytest.fixture
def dummy_image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def dummy_color_picker(dummy_tk, dummy_image):
    with (
        patch("autocv.color_picker.Vision") as MockVision,
        patch("autocv.color_picker.Toplevel") as MockToplevel,
        patch("autocv.color_picker.ImageTk.PhotoImage", return_value=MagicMock()),
    ):
        mock_vision = MockVision.return_value
        mock_vision.opencv_image = dummy_image
        mock_toplevel = MagicMock()
        MockToplevel.return_value = mock_toplevel
        return ColorPicker(hwnd=12345, master=dummy_tk)


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(10, 10))
def test_set_geometry_updates_position(mock_cursor, dummy_color_picker):
    dummy_color_picker.set_geometry()
    dummy_color_picker.master_screen.geometry.assert_called_with("120x120+10+10")


@patch("autocv.color_picker.ImageTk.PhotoImage")
def test_create_screen_canvas_sets_photo_image(mock_photo, dummy_color_picker):
    dummy_color_picker.create_screen_canvas()
    assert hasattr(dummy_color_picker.snip_surface, "img")


@patch("autocv.color_picker.ImageTk.PhotoImage", return_value=MagicMock())
@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(10, 10))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(10, 10))
def test_on_tick_handles_valid_image(mock_stc, mock_cursor, mock_photo, dummy_color_picker):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()  # Prevent GUI window from showing

    dummy_color_picker.snip_surface = MagicMock()
    dummy_color_picker.snip_surface.create_image = MagicMock()
    dummy_color_picker.snip_surface.delete = MagicMock()
    dummy_color_picker.draw_center_rectangle = MagicMock()
    dummy_color_picker.draw_cursor_coordinates = lambda img, x, y: img
    dummy_color_picker.master.after = lambda delay, cb: None
    dummy_color_picker.master_screen = MagicMock()
    dummy_color_picker.master_screen.winfo_exists.return_value = True

    with patch("autocv.color_picker.cv.resize", return_value=np.zeros((40, 40, 3), dtype=np.uint8)):
        dummy_color_picker.on_tick()

    root.destroy()


@patch("autocv.color_picker.Path.exists", return_value=False)
def test_draw_cursor_coordinates_adds_text(_, dummy_color_picker):
    img = Image.new("RGB", (40, 40))
    result = dummy_color_picker.draw_cursor_coordinates(img, 5, 5)
    assert isinstance(result, Image.Image)


def test_draw_center_rectangle_draws_rect(dummy_color_picker):
    dummy_color_picker.snip_surface = MagicMock()
    dummy_color_picker.draw_center_rectangle(np.ones((3, 3, 3), dtype=np.uint8))


def test_handle_button_press_sets_result_valid(dummy_color_picker):
    dummy_color_picker.master_screen = MagicMock()
    dummy_color_picker.handle_button_press(5, 5)
    assert dummy_color_picker.result[1] == (5, 5)


def test_handle_button_press_sets_result_invalid(dummy_color_picker):
    dummy_color_picker.master_screen = MagicMock()
    dummy_color_picker.handle_button_press(-1, -1)
    assert dummy_color_picker.result == ((-1, -1, -1), (-1, -1))
