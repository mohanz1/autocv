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
    @patch("autocv.autocv.Path.exists", return_value=False)
    def test_init_raises_when_prebuilt_missing(self, mock_exists):
        with pytest.raises(FileNotFoundError, match="Missing prebuilt extension directory"):
            AutoCV(hwnd=1111)

    @patch("autocv.autocv.win32gui.GetWindowRect", return_value=(0, 0, 800, 600))
    def test_get_window_size(self, mock_get_rect, autocv):
        size = autocv.get_window_size()
        assert size == (800, 600)

    @patch("autocv.autocv.win32gui.GetWindowRect", return_value=(0, 0, 800, 600))
    def test_get_window_size_uses_cache_path(self, mock_get_rect, autocv):
        size = autocv.get_window_size(use_cache=True)
        assert size == (800, 600)

    def test_get_hwnd_valid(self, autocv):
        assert autocv.get_hwnd() == 1111

    def test_antigcp_delegates_to_prebuilt_extension(self, autocv):
        autocv._antigcp = MagicMock()
        autocv._antigcp.antigcp.return_value = True
        autocv._get_topmost_hwnd = MagicMock(return_value=1234)

        assert autocv.antigcp() is True
        autocv._antigcp.antigcp.assert_called_once_with(1234)

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
