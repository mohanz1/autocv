from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import win32gui

from autocv.image_picker import ImagePicker, ImagePickerCapture, ImagePickerController
from autocv.models import InvalidHandleError


@pytest.fixture
def image_picker():
    with (
        patch("autocv.image_picker.Toplevel") as mock_toplevel_cls,
        patch("autocv.image_picker.Frame") as mock_frame_cls,
        patch("autocv.image_picker.Canvas") as mock_canvas_cls,
        patch("autocv.image_picker.win32gui.GetWindowRect", return_value=(0, 0, 100, 100)),
        patch("autocv.image_picker.win32gui.ShowWindow"),
        patch("autocv.image_picker.win32gui.SetForegroundWindow"),
    ):
        master = MagicMock()
        master_screen = MagicMock()
        master_screen.winfo_exists.return_value = True
        mock_toplevel_cls.return_value = master_screen
        canvas = MagicMock()
        mock_canvas_cls.return_value = canvas

        picker = ImagePicker(hwnd=1234, master=master)
        picker.snip_surface = canvas
        yield picker


def test_window_rect_translates_invalid_handle():
    controller = ImagePickerController(hwnd=1234)

    with patch(
        "autocv.image_picker.win32gui.GetWindowRect",
        side_effect=win32gui.error(1400, "GetWindowRect", "Invalid window handle."),
    ):
        with pytest.raises(InvalidHandleError) as exc_info:
            controller.window_rect()

    assert exc_info.value.hwnd == 1234


def test_capture_region_creates_vision_once_and_crops():
    vision = MagicMock()
    vision.opencv_image = np.arange(10000).reshape((100, 100))
    factory = MagicMock(return_value=vision)
    controller = ImagePickerController(hwnd=1234, vision_factory=factory)

    result = controller.capture_region(10, 10, 20, 20)

    assert result.shape == (10, 10)
    factory.assert_called_once_with(1234)
    vision.refresh.assert_called_once_with()


def test_capture_region_skips_invalid_refresh_and_reuses_vision():
    vision = MagicMock()
    vision.opencv_image = np.arange(10000).reshape((100, 100))
    vision.refresh.side_effect = InvalidHandleError(1234)
    factory = MagicMock(return_value=vision)
    controller = ImagePickerController(hwnd=1234, vision_factory=factory)

    result = controller.capture_region(10, 10, 12, 12)
    second = controller.capture_region(0, 0, 1, 1)

    assert result.shape == (2, 2)
    assert second.shape == (1, 1)
    factory.assert_called_once_with(1234)
    assert vision.refresh.call_count == 2


@patch("autocv.image_picker.win32gui.GetWindowRect", return_value=(5, 6, 15, 26))
def test_full_rect_as_bounds(mock_get_window_rect):
    controller = ImagePickerController(hwnd=1234)

    assert controller.full_rect_as_bounds() == (5, 6, 10, 20)


def test_on_button_press_creates_rectangle(image_picker):
    mock_event = MagicMock()
    mock_event.x = 10
    mock_event.y = 20

    image_picker.on_button_press(mock_event)

    assert image_picker.start_x == 10
    assert image_picker.start_y == 20
    assert image_picker.current_x == 10
    assert image_picker.current_y == 20
    image_picker.snip_surface.create_rectangle.assert_called_once_with(
        10,
        20,
        10,
        20,
        outline="red",
        width=3,
        fill="maroon3",
    )


def test_on_snip_drag_updates_coords(image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 20
    image_picker._rect_id = 7
    mock_event = MagicMock()
    mock_event.x = 50
    mock_event.y = 60

    image_picker.on_snip_drag(mock_event)

    image_picker.snip_surface.coords.assert_called_with(7, 10, 20, 50, 60)


def test_on_snip_drag_creates_rectangle_when_missing(image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 20
    image_picker.snip_surface.create_rectangle.return_value = 11
    mock_event = MagicMock()
    mock_event.x = 50
    mock_event.y = 60

    image_picker.on_snip_drag(mock_event)

    assert image_picker._rect_id == 11
    image_picker.snip_surface.create_rectangle.assert_called_with(
        10,
        20,
        50,
        60,
        outline="red",
        width=3,
        fill="maroon3",
    )
    image_picker.snip_surface.coords.assert_called_once_with(11, 10, 20, 50, 60)


def test_take_bounded_screenshot_captures_and_stores_bounds(image_picker):
    image_picker.controller = MagicMock()
    image_picker.controller.capture_region.return_value = np.zeros((10, 10), dtype=np.uint8)
    image_picker.controller.full_rect_as_bounds.return_value = (0, 0, 100, 100)

    image_picker.take_bounded_screenshot(10, 10, 20, 20)

    assert image_picker.result.shape == (10, 10)
    assert image_picker.selection_rect == (10, 10, 10, 10)
    assert image_picker.rect == (0, 0, 100, 100)
    image_picker.controller.capture_region.assert_called_once_with(10, 10, 20, 20)


def test_capture_property_returns_structured_result(image_picker):
    image_picker.result = np.zeros((1, 1), dtype=np.uint8)
    image_picker.selection_rect = (1, 2, 3, 4)
    image_picker.rect = (10, 20, 30, 40)

    capture = image_picker.capture

    assert capture == ImagePickerCapture(
        image=image_picker.result,
        selection_rect=(1, 2, 3, 4),
        window_rect=(10, 20, 30, 40),
    )


@patch.object(ImagePicker, "take_bounded_screenshot")
def test_on_button_release_calls_take_screenshot_and_closes(mock_take, image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 10
    image_picker.current_x = 20
    image_picker.current_y = 30
    image_picker.master_screen.destroy = MagicMock()
    image_picker.master_screen.quit = MagicMock()
    mock_event = MagicMock()
    mock_event.x = 25
    mock_event.y = 35

    image_picker.on_button_release(mock_event)

    mock_take.assert_called_once_with(10, 10, 25, 35)
    image_picker.master_screen.destroy.assert_called_once()
    image_picker.master_screen.quit.assert_called_once()


@patch.object(ImagePicker, "take_bounded_screenshot")
def test_on_button_release_falls_back_to_tracked_coords_when_event_lacks_position(mock_take, image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 10
    image_picker.current_x = 20
    image_picker.current_y = 30
    image_picker.master_screen.destroy = MagicMock()
    image_picker.master_screen.quit = MagicMock()

    image_picker.on_button_release(MagicMock())

    mock_take.assert_called_once_with(10, 10, 20, 30)
    image_picker.master_screen.destroy.assert_called_once()
    image_picker.master_screen.quit.assert_called_once()


@patch.object(ImagePicker, "take_bounded_screenshot")
def test_on_button_release_skips_zero_size_selection(mock_take, image_picker):
    image_picker.master_screen.destroy = MagicMock()
    image_picker.master_screen.quit = MagicMock()
    mock_event = MagicMock()
    mock_event.x = 10
    mock_event.y = 20

    image_picker.on_button_press(mock_event)
    image_picker.on_button_release(mock_event)

    mock_take.assert_not_called()
    image_picker.master_screen.destroy.assert_called_once()
    image_picker.master_screen.quit.assert_called_once()
