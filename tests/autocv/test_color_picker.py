from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import win32gui
from PIL import Image

from autocv.color_picker import REFRESH_DELAY_MS, ColorPicker, ColorPickerController, _mean_patch_rgb
from autocv.models import InvalidHandleError


@pytest.fixture
def dummy_tk():
    return MagicMock()


@pytest.fixture
def dummy_image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def dummy_color_picker(dummy_tk, dummy_image):
    with (
        patch("autocv.color_picker.Vision") as mock_vision_cls,
        patch("autocv.color_picker.Toplevel") as mock_toplevel_cls,
        patch("autocv.color_picker.Canvas") as mock_canvas_cls,
        patch("autocv.color_picker.ImageTk.PhotoImage", return_value=MagicMock()),
        patch("autocv.color_picker.win32api.GetKeyState", return_value=0),
    ):
        mock_vision = mock_vision_cls.return_value
        mock_vision.opencv_image = dummy_image
        mock_toplevel = MagicMock()
        mock_toplevel.winfo_exists.return_value = True
        mock_toplevel_cls.return_value = mock_toplevel
        mock_canvas = MagicMock()
        mock_canvas_cls.return_value = mock_canvas

        return ColorPicker(hwnd=12345, master=dummy_tk)


def test_mean_patch_rgb_returns_black_for_empty_patch():
    assert _mean_patch_rgb(np.zeros((0, 0, 3), dtype=np.uint8)) == (0, 0, 0)


def test_mean_patch_rgb_rejects_non_rgb_patch():
    with pytest.raises(ValueError, match="Expected an RGB patch"):
        _mean_patch_rgb(np.zeros((2, 2), dtype=np.uint8))


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(10, 10))
def test_set_geometry_updates_position(mock_cursor, dummy_color_picker):
    dummy_color_picker.set_geometry()

    dummy_color_picker.master_screen.geometry.assert_called_with("120x120+10+10")


def test_set_geometry_skips_destroyed_window(dummy_color_picker):
    dummy_color_picker.master_screen.winfo_exists.return_value = False
    dummy_color_picker.master_screen.geometry.reset_mock()

    dummy_color_picker.set_geometry((10, 20))

    dummy_color_picker.master_screen.geometry.assert_not_called()


@patch("autocv.color_picker.ImageTk.PhotoImage", return_value=MagicMock())
def test_create_screen_canvas_sets_photo_image(mock_photo, dummy_color_picker):
    dummy_color_picker.create_screen_canvas()

    assert hasattr(dummy_color_picker.snip_surface, "img")
    dummy_color_picker.master.after.assert_called_with(0, dummy_color_picker.on_tick)


@patch("autocv.color_picker.PhotoImage", return_value=MagicMock())
@patch("autocv.color_picker.cv.resize", return_value=np.zeros((40, 40, 3), dtype=np.uint8))
@patch("autocv.color_picker.win32api.GetKeyState", return_value=0)
@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(10, 10))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(10, 10))
def test_on_tick_handles_valid_image(
    mock_screen_to_client,
    mock_cursor,
    mock_get_key_state,
    mock_resize,
    mock_photo_image,
    dummy_color_picker,
):
    dummy_color_picker.snip_surface = MagicMock()
    dummy_color_picker.draw_center_rectangle = MagicMock()
    dummy_color_picker.draw_cursor_coordinates = MagicMock(side_effect=lambda img, x, y: img)
    dummy_color_picker.master_screen.winfo_exists.return_value = True
    dummy_color_picker.prev_state = 0
    dummy_color_picker.master.after.reset_mock()

    dummy_color_picker.on_tick()

    dummy_color_picker.snip_surface.delete.assert_called_once_with("center")
    dummy_color_picker.snip_surface.create_image.assert_called_once()
    dummy_color_picker.draw_center_rectangle.assert_called_once()
    dummy_color_picker.master.after.assert_called_with(REFRESH_DELAY_MS, dummy_color_picker.on_tick)


def test_on_tick_returns_when_window_is_destroyed(dummy_color_picker):
    dummy_color_picker.master_screen.winfo_exists.return_value = False
    dummy_color_picker.snip_surface = MagicMock()

    dummy_color_picker.on_tick()

    dummy_color_picker.snip_surface.delete.assert_not_called()


@patch("autocv.color_picker.PhotoImage", return_value=MagicMock())
@patch("autocv.color_picker.cv.resize", return_value=np.zeros((40, 40, 3), dtype=np.uint8))
@patch("autocv.color_picker.win32api.GetKeyState", return_value=0)
def test_on_tick_handles_button_release_click(mock_get_key_state, mock_resize, mock_photo_image, dummy_color_picker):
    dummy_color_picker.snip_surface = MagicMock()
    dummy_color_picker.draw_center_rectangle = MagicMock()
    dummy_color_picker.draw_cursor_coordinates = MagicMock(side_effect=lambda img, x, y: img)
    dummy_color_picker.handle_button_press = MagicMock()
    dummy_color_picker.master_screen.winfo_exists.return_value = True
    dummy_color_picker.controller.capture_cursor_patch = MagicMock(
        return_value=(np.zeros((3, 3, 3), dtype=np.uint8), 4, 5, (1, 2))
    )
    dummy_color_picker.prev_state = -1
    dummy_color_picker.master.after.reset_mock()

    dummy_color_picker.on_tick()

    dummy_color_picker.handle_button_press.assert_called_once_with(4, 5)
    assert dummy_color_picker.prev_state == 0


@patch("autocv.color_picker.PhotoImage", return_value=MagicMock())
@patch("autocv.color_picker.cv.resize", return_value=np.zeros((40, 40, 3), dtype=np.uint8))
@patch("autocv.color_picker.win32api.GetKeyState", return_value=-1)
def test_on_tick_updates_prev_state_without_click_on_button_down(
    mock_get_key_state,
    mock_resize,
    mock_photo_image,
    dummy_color_picker,
):
    dummy_color_picker.snip_surface = MagicMock()
    dummy_color_picker.draw_center_rectangle = MagicMock()
    dummy_color_picker.draw_cursor_coordinates = MagicMock(side_effect=lambda img, x, y: img)
    dummy_color_picker.handle_button_press = MagicMock()
    dummy_color_picker.master_screen.winfo_exists.side_effect = [True, True, False]
    dummy_color_picker.controller.capture_cursor_patch = MagicMock(
        return_value=(np.zeros((3, 3, 3), dtype=np.uint8), 4, 5, (1, 2))
    )
    dummy_color_picker.prev_state = 0

    dummy_color_picker.on_tick()

    dummy_color_picker.handle_button_press.assert_not_called()
    assert dummy_color_picker.prev_state == -1


def test_draw_cursor_coordinates_adds_text(dummy_color_picker):
    img = Image.new("RGB", (40, 40))

    result = dummy_color_picker.draw_cursor_coordinates(img, 5, 5)

    assert isinstance(result, Image.Image)


def test_draw_cursor_coordinates_uses_fallback_for_out_of_bounds(dummy_color_picker):
    dummy_color_picker.controller.vision.opencv_image = np.zeros((2, 2, 3), dtype=np.uint8)
    img = Image.new("RGB", (40, 40))

    result = dummy_color_picker.draw_cursor_coordinates(img, 50, 50)

    assert isinstance(result, Image.Image)


def test_draw_center_rectangle_draws_rect(dummy_color_picker):
    dummy_color_picker.snip_surface = MagicMock()

    dummy_color_picker.draw_center_rectangle(np.ones((3, 3, 3), dtype=np.uint8))

    dummy_color_picker.snip_surface.create_rectangle.assert_called_once()


def test_handle_button_press_sets_result_valid(dummy_color_picker):
    dummy_color_picker.master_screen = MagicMock()

    dummy_color_picker.handle_button_press(5, 5)

    assert dummy_color_picker.result == ((255, 255, 255), (5, 5))


def test_handle_button_press_sets_result_invalid(dummy_color_picker):
    dummy_color_picker.master_screen = MagicMock()

    dummy_color_picker.handle_button_press(-1, -1)

    assert dummy_color_picker.result == ((-1, -1, -1), (-1, -1))


def test_handle_button_press_sets_result_invalid_for_non_rgb_frame(dummy_color_picker):
    dummy_color_picker.master_screen = MagicMock()
    dummy_color_picker.controller.vision.opencv_image = np.zeros((10, 10), dtype=np.uint8)

    dummy_color_picker.handle_button_press(1, 1)

    assert dummy_color_picker.result == ((-1, -1, -1), (-1, -1))


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(0, 0))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(0, 0))
def test_capture_cursor_patch_returns_default_canvas_for_partial_edges(mock_screen_to_client, mock_cursor):
    default_canvas = np.full((3, 3, 3), 7, dtype=np.uint8)
    with patch("autocv.color_picker.Vision") as mock_vision_cls:
        mock_vision = mock_vision_cls.return_value
        mock_vision.opencv_image = np.full((5, 5, 3), 255, dtype=np.uint8)
        controller = ColorPickerController(hwnd=12345, default_canvas=default_canvas)

        cropped, x, y, cursor_pos = controller.capture_cursor_patch()

    assert np.array_equal(cropped, default_canvas)
    assert (x, y) == (0, 0)
    assert cursor_pos == (0, 0)
    mock_vision.refresh.assert_called_once_with()


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(5, 6))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(20, 20))
def test_capture_cursor_patch_returns_default_canvas_for_positive_out_of_bounds(mock_screen_to_client, mock_cursor):
    default_canvas = np.full((3, 3, 3), 7, dtype=np.uint8)
    with patch("autocv.color_picker.Vision") as mock_vision_cls:
        mock_vision = mock_vision_cls.return_value
        mock_vision.opencv_image = np.zeros((5, 5, 3), dtype=np.uint8)
        controller = ColorPickerController(hwnd=12345, default_canvas=default_canvas)

        cropped, x, y, cursor_pos = controller.capture_cursor_patch()

    assert np.array_equal(cropped, default_canvas)
    assert (x, y) == (20, 20)
    assert cursor_pos == (5, 6)


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(5, 6))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(1, 2))
def test_capture_cursor_patch_returns_default_canvas_for_non_rgb_frame(mock_screen_to_client, mock_cursor):
    default_canvas = np.full((3, 3, 3), 7, dtype=np.uint8)
    with patch("autocv.color_picker.Vision") as mock_vision_cls:
        mock_vision = mock_vision_cls.return_value
        mock_vision.opencv_image = np.zeros((5, 5), dtype=np.uint8)
        controller = ColorPickerController(hwnd=12345, default_canvas=default_canvas)

        cropped, x, y, cursor_pos = controller.capture_cursor_patch()

    assert np.array_equal(cropped, default_canvas)
    assert (x, y) == (1, 2)
    assert cursor_pos == (5, 6)


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(5, 6))
@patch("autocv.color_picker.win32gui.ScreenToClient", side_effect=win32gui.error(1400, "ScreenToClient", "Invalid"))
def test_capture_cursor_patch_returns_default_canvas_for_invalid_client_conversion(
    mock_screen_to_client,
    mock_cursor,
):
    default_canvas = np.full((3, 3, 3), 7, dtype=np.uint8)
    with patch("autocv.color_picker.Vision") as mock_vision_cls:
        controller = ColorPickerController(hwnd=12345, default_canvas=default_canvas)

        cropped, x, y, cursor_pos = controller.capture_cursor_patch()

    assert np.array_equal(cropped, default_canvas)
    assert (x, y) == (-1, -1)
    assert cursor_pos == (5, 6)
    mock_vision_cls.return_value.refresh.assert_not_called()


@patch("autocv.color_picker.win32gui.GetCursorPos", return_value=(5, 6))
@patch("autocv.color_picker.win32gui.ScreenToClient", return_value=(1, 2))
def test_capture_cursor_patch_returns_default_canvas_for_invalid_refresh(mock_screen_to_client, mock_cursor):
    default_canvas = np.full((3, 3, 3), 7, dtype=np.uint8)
    with patch("autocv.color_picker.Vision") as mock_vision_cls:
        mock_vision = mock_vision_cls.return_value
        mock_vision.refresh.side_effect = InvalidHandleError(12345)
        controller = ColorPickerController(hwnd=12345, default_canvas=default_canvas)

        cropped, x, y, cursor_pos = controller.capture_cursor_patch()

    assert np.array_equal(cropped, default_canvas)
    assert (x, y) == (-1, -1)
    assert cursor_pos == (5, 6)


def test_color_picker_falls_back_to_default_font(dummy_tk, dummy_image):
    with (
        patch("autocv.color_picker.Vision") as mock_vision_cls,
        patch("autocv.color_picker.Toplevel"),
        patch("autocv.color_picker.Canvas", return_value=MagicMock()),
        patch("autocv.color_picker.ImageTk.PhotoImage", return_value=MagicMock()),
        patch("autocv.color_picker.ImageFont.truetype", side_effect=OSError("missing")),
        patch("autocv.color_picker.ImageFont.load_default", return_value=MagicMock()) as mock_load_default,
        patch("autocv.color_picker.win32api.GetKeyState", return_value=0),
    ):
        mock_vision_cls.return_value.opencv_image = dummy_image

        picker = ColorPicker(hwnd=12345, master=dummy_tk)

    assert picker._font is mock_load_default.return_value


def test_vision_property_returns_controller_vision(dummy_color_picker):
    assert dummy_color_picker.vision is dummy_color_picker.controller.vision
