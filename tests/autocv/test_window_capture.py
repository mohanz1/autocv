import numpy as np
import pytest
from mock import patch, MagicMock
from autocv import AutoCV
from autocv.utils import filtering


@pytest.fixture
def autocv():
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        obj = AutoCV(hwnd=1234)
        obj.opencv_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return obj


class TestAutoCVWindowCapture:
    @patch("win32gui.EnumWindows")
    @patch("win32gui.GetWindowText", return_value="TestWindow")
    @patch("win32gui.IsWindowVisible", return_value=True)
    def test_get_windows_with_hwnds(self, visible, get_text, enum, autocv):
        enum.side_effect = lambda func, arg: func(42, arg)
        result = autocv.get_windows_with_hwnds()
        assert result == [(42, "TestWindow")]

    @patch.object(AutoCV, "get_windows_with_hwnds", return_value=[(1001, "App Name")])
    @patch("autocv.utils.filtering.find_first", return_value=(1001, "App Name"))
    def test_set_hwnd_by_title_success(self, mock_find, mock_windows, autocv):
        success = autocv.set_hwnd_by_title("App", case_insensitive=True)
        assert success is True
        assert autocv.hwnd == 1001

    @patch.object(AutoCV, "get_windows_with_hwnds", return_value=[])
    @patch("autocv.utils.filtering.find_first", return_value=None)
    def test_set_hwnd_by_title_fail(self, mock_find, mock_windows, autocv):
        success = autocv.set_hwnd_by_title("Missing")
        assert success is False

    @patch("win32gui.EnumChildWindows")
    @patch("win32gui.GetClassName", return_value="Child")
    def test_get_child_windows(self, mock_class, mock_enum, autocv):
        mock_enum.side_effect = lambda hwnd, func, arg: func(1234, arg)
        result = autocv.get_child_windows()
        assert result == [(1234, "Child")]

    @patch.object(AutoCV, "get_child_windows", return_value=[(5678, "ChildWindow")])
    @patch("autocv.utils.filtering.find_first", return_value=(5678, "ChildWindow"))
    def test_set_inner_hwnd_by_title(self, mock_find, mock_get, autocv):
        success = autocv.set_inner_hwnd_by_title("Child")
        assert success is True
        assert autocv.hwnd == 5678

    @patch.object(AutoCV, "get_windows_with_hwnds", return_value=[(111, "FooBar"), (222, "Baz")])
    def test_get_hwnd_by_title_variants(self, mock_windows, autocv):
        assert autocv.get_hwnd_by_title("Foo") == 111
        assert autocv.get_hwnd_by_title("foo") is None
        assert autocv.get_hwnd_by_title("foo", case_insensitive=True) == 111

    @patch.object(AutoCV, "get_child_windows", return_value=[(1, ""), (2, "ChildPanel")])
    def test_find_child_by_class_skips_empty(self, mock_children, autocv):
        with patch("autocv.utils.filtering.find_first", wraps=filtering.find_first) as wrapped:
            result = autocv._find_child_by_class(lambda name: "Child" in name)
        assert result == 2
        assert wrapped.called

    @patch.object(AutoCV, "get_child_windows", return_value=[])
    @patch("autocv.utils.filtering.find_first", return_value=None)
    def test_set_inner_hwnd_by_title_failure(self, mock_find, mock_children, autocv):
        assert autocv.set_inner_hwnd_by_title("Missing") is False
