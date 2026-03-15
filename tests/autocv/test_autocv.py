import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from autocv import AutoCV
from autocv.autocv import _load_antigcp_module, _prioritize_sys_path
from autocv.image_picker import ImagePickerCapture
from autocv.models import InvalidHandleError


@pytest.fixture
def autocv():
    fake_prebuilt_dir = Path(r"C:\fake\prebuilt\python312")
    with (
        patch("autocv.autocv._prebuilt_dir_for_runtime", return_value=fake_prebuilt_dir),
        patch("autocv.autocv.Path.exists", return_value=True),
        patch("autocv.autocv._load_antigcp_module", return_value=MagicMock()),
        patch("autocv.autocv.logging.getLogger"),
    ):
        return AutoCV(hwnd=1111)


class TestAutoCV:
    @patch("autocv.autocv._prebuilt_dir_for_runtime", return_value=Path(r"C:\fake\missing"))
    @patch("autocv.autocv.Path.exists", return_value=False)
    def test_init_raises_when_prebuilt_missing(self, mock_exists, mock_prebuilt_dir):
        with pytest.raises(FileNotFoundError, match="Missing prebuilt extension directory"):
            AutoCV(hwnd=1111)

    @patch.object(AutoCV, "_fetch_window_bounds", return_value=(0, 0, 800, 600))
    def test_get_window_size(self, mock_fetch_bounds, autocv):
        size = autocv.get_window_size()
        assert size == (800, 600)

    @patch("autocv.core.window_capture.win32gui.GetWindowRect", return_value=(0, 0, 800, 600))
    def test_get_window_size_uses_cache_path(self, mock_get_rect, autocv):
        size = autocv.get_window_size(use_cache=True)
        assert size == (800, 600)

    @patch.object(AutoCV, "_fetch_window_bounds", side_effect=InvalidHandleError(1111))
    def test_get_window_size_propagates_invalid_handle(self, mock_fetch_bounds, autocv):
        with pytest.raises(InvalidHandleError):
            autocv.get_window_size()

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
    @patch("autocv.autocv.ImagePicker")
    def test_image_picker_capture_returns_structured_result(self, mock_picker, mock_tk, autocv):
        picker_instance = MagicMock()
        picker_instance.capture = ImagePickerCapture(
            image="image",
            selection_rect=(1, 2, 3, 4),
            window_rect=(10, 20, 30, 40),
        )
        mock_picker.return_value = picker_instance

        result = autocv.image_picker_capture()

        assert result == picker_instance.capture

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
        autocv.opencv_image = np.zeros((1, 1, 3), dtype=np.uint8)
        settings = autocv.image_filter()
        assert settings == mock_settings

    def test_image_filter_rejects_non_color_image(self, autocv):
        autocv.opencv_image = np.zeros((1, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="image_filter requires a 3-channel BGR image"):
            autocv.image_filter()

    @patch("autocv.autocv.cv.imshow")
    @patch("autocv.autocv.cv.waitKey")
    def test_show_backbuffer_calls_imshow(self, mock_waitkey, mock_imshow, autocv):
        autocv.opencv_image = MagicMock()
        autocv.show_backbuffer(live=True)
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once()


def test_prioritize_sys_path_moves_path_to_front_without_duplicates():
    original_path = list(sys.path)
    target = r"C:\fake\prebuilt\python312"
    try:
        sys.path = [r"C:\one", target, r"C:\two", target]
        _prioritize_sys_path(Path(target))
        assert sys.path[0] == target
        assert sys.path.count(target) == 1
    finally:
        sys.path = original_path


def test_load_antigcp_module_reuses_cached_module_from_same_prebuilt_dir():
    module = SimpleNamespace(__file__=r"C:\fake\prebuilt\python312\antigcp.pyd", antigcp=MagicMock())
    with patch.object(sys, "path", [r"C:\start"]), patch.dict("sys.modules", {"antigcp": module}, clear=False):
        loaded = _load_antigcp_module(Path(r"C:\fake\prebuilt\python312"))

    assert loaded is module


@patch("autocv.autocv.import_module")
def test_load_antigcp_module_reloads_cached_module_from_other_dir(mock_import_module):
    cached_module = SimpleNamespace(__file__=r"C:\other\antigcp.pyd", antigcp=MagicMock())
    imported_module = SimpleNamespace(__file__=r"C:\fake\prebuilt\python312\antigcp.pyd", antigcp=MagicMock())
    mock_import_module.return_value = imported_module

    with patch.object(sys, "path", [r"C:\start"]), patch.dict("sys.modules", {"antigcp": cached_module}, clear=False):
        loaded = _load_antigcp_module(Path(r"C:\fake\prebuilt\python312"))

    assert loaded is imported_module
    mock_import_module.assert_called_once_with("antigcp")
