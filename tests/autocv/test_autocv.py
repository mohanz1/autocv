# test_autocv.py

import pytest
from mock import patch, MagicMock

from autocv import AutoCV


@pytest.fixture
def autocv():
    # Bypass antigcp and patch logger
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        return AutoCV(hwnd=1111)


class TestAutoCV:
    @patch("autocv.autocv.win32gui.GetWindowRect", return_value=(0, 0, 800, 600))
    def test_get_window_size(self, mock_get_rect, autocv):
        size = autocv.get_window_size()
        assert size == (800, 600)

    def test_get_hwnd_valid(self, autocv):
        assert autocv.get_hwnd() == 1111

    @patch("autocv.autocv.Tk")
    @patch("autocv.autocv.ImagePicker")
    def test_image_picker_returns_result(self, mock_picker, mock_tk, autocv):
        picker_instance = MagicMock()
        picker_instance.result = "image"
        picker_instance.rect = (10, 20, 30, 40)
        mock_picker.return_value = picker_instance

        result = autocv.image_picker()
        assert result == ("image", (10, 20, 30, 40))

    @patch("autocv.autocv.Tk")
    @patch("autocv.autocv.ColorPicker")
    def test_color_picker_returns_result(self, mock_picker, mock_tk, autocv):
        picker_instance = MagicMock()
        picker_instance.result = ((255, 255, 255), (100, 100))
        mock_picker.return_value = picker_instance

        result = autocv.color_picker()
        assert result == ((255, 255, 255), (100, 100))

    @patch("autocv.autocv.ImageFilter")
    def test_image_filter_returns_settings(self, mock_filter, autocv):
        mock_settings = MagicMock()
        mock_filter.return_value.filter_settings = mock_settings
        autocv.opencv_image = MagicMock()
        settings = autocv.image_filter()
        assert settings == mock_settings

    @patch("autocv.autocv.cv.imshow")
    @patch("autocv.autocv.cv.waitKey")
    def test_show_backbuffer_calls_imshow(self, mock_waitkey, mock_imshow, autocv):
        autocv.opencv_image = MagicMock()
        autocv.show_backbuffer(live=True)
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once()
