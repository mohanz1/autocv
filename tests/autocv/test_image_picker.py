import pytest
import numpy as np
from mock import MagicMock, patch
from tkinter import Tk, TclError
from autocv.image_picker import ImagePicker


@pytest.fixture
def image_picker():
    with (
        patch("autocv.image_picker.Toplevel") as mock_toplevel,
        patch("autocv.image_picker.Frame") as mock_frame,
        patch("autocv.image_picker.Canvas") as mock_canvas,
        patch("autocv.image_picker.win32gui.GetWindowRect", return_value=(0, 0, 100, 100)),
        patch("autocv.image_picker.win32gui.ShowWindow"),
        patch("autocv.image_picker.win32gui.SetForegroundWindow"),
    ):
        mock_canvas.return_value.bind = MagicMock()
        try:
            master = Tk()
        except TclError:
            pytest.skip("Tk not available on this system")
        picker = ImagePicker(hwnd=1234, master=master)
        picker.snip_surface = mock_canvas.return_value
        yield picker
        master.destroy()


def test_on_button_press_creates_rectangle(image_picker):
    mock_event = MagicMock()
    mock_event.x = 10
    mock_event.y = 20
    image_picker.on_button_press(mock_event)
    assert image_picker.start_x == 10
    assert image_picker.start_y == 20
    image_picker.snip_surface.create_rectangle.assert_called_once()


def test_on_snip_drag_updates_coords(image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 20
    mock_event = MagicMock()
    mock_event.x = 50
    mock_event.y = 60
    image_picker.on_snip_drag(mock_event)
    image_picker.snip_surface.coords.assert_called_with(1, 10, 20, 50, 60)


@patch("autocv.image_picker.Vision")
@patch("autocv.image_picker.win32gui.GetWindowRect", return_value=(0, 0, 100, 100))
def test_take_bounded_screenshot_captures_and_crops(mock_rect, mock_vision, image_picker):
    mock_instance = mock_vision.return_value
    mock_instance.opencv_image = np.arange(10000).reshape((100, 100))
    image_picker.take_bounded_screenshot(10, 10, 20, 20)
    assert image_picker.result.shape == (10, 10)
    assert image_picker.rect == (0, 0, 100, 100)


@patch.object(ImagePicker, "take_bounded_screenshot")
def test_on_button_release_calls_take_screenshot_and_closes(mock_take, image_picker):
    image_picker.start_x = 10
    image_picker.start_y = 10
    image_picker.current_x = 20
    image_picker.current_y = 30
    image_picker.master_screen.destroy = MagicMock()
    image_picker.master_screen.quit = MagicMock()
    mock_event = MagicMock()

    image_picker.on_button_release(mock_event)

    mock_take.assert_called_once_with(10, 10, 20, 30)
    image_picker.master_screen.destroy.assert_called_once()
    image_picker.master_screen.quit.assert_called_once()
