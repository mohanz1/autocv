from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from autocv.core.window_capture import WindowCapture
from autocv.models import InvalidHandleError


def test_is_attached_and_detach_clears_cache():
    wc = WindowCapture()
    assert wc.is_attached is False
    assert wc.last_frame is None

    wc.attach(123)
    assert wc.is_attached is True

    wc._cached_bounds = (0, 0, 1, 1)
    wc._last_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    wc.detach()

    assert wc.is_attached is False
    assert wc._cached_bounds is None
    assert wc.last_frame is None


def test_get_window_bounds_uses_cache():
    wc = WindowCapture(hwnd=111)
    with patch.object(WindowCapture, "_fetch_window_bounds", return_value=(1, 2, 3, 4)) as mock_fetch:
        assert wc.get_window_bounds(use_cache=True) == (1, 2, 3, 4)
        assert wc.get_window_bounds(use_cache=True) == (1, 2, 3, 4)
        assert mock_fetch.call_count == 1


def test_get_window_size_and_bounds_to_rect():
    bounds = (10, 20, 30, 50)
    assert WindowCapture.bounds_to_rect(bounds) == (0, 0, 20, 30)

    wc = WindowCapture(hwnd=1)
    with patch.object(WindowCapture, "_fetch_window_bounds", return_value=bounds):
        assert wc.get_window_size() == (20, 30)


def test_find_by_title_casefold_and_exact():
    windows = [(1, "Foo Bar"), (2, "Baz")]
    assert WindowCapture._find_by_title("foo", windows, case_insensitive=True) == 1
    assert WindowCapture._find_by_title("Foo", windows, case_insensitive=False) == 1
    assert WindowCapture._find_by_title("nope", windows, case_insensitive=True) is None


def test_set_hwnd_by_title_attaches_when_found():
    wc = WindowCapture()
    with patch.object(WindowCapture, "get_windows_with_hwnds", return_value=[(5, "Hello")]):
        assert wc.set_hwnd_by_title("Hell") is True
        assert wc.hwnd == 5

    wc2 = WindowCapture()
    with patch.object(WindowCapture, "get_windows_with_hwnds", return_value=[(6, "Other")]):
        assert wc2.set_hwnd_by_title("Missing") is False
        assert wc2.hwnd == -1


def test_get_hwnd_by_title_delegates_to_find():
    wc = WindowCapture()
    with patch.object(WindowCapture, "get_windows_with_hwnds", return_value=[(1, "App Name")]):
        assert wc.get_hwnd_by_title("App") == 1


@patch("autocv.core.window_capture.win32gui.EnumChildWindows")
@patch("autocv.core.window_capture.win32gui.GetClassName", return_value="Child")
def test_get_child_windows_enumerates(mock_get_class, mock_enum):
    wc = WindowCapture(hwnd=1)
    mock_enum.side_effect = lambda hwnd, func, arg: func(1234, arg)
    assert wc.get_child_windows() == [(1234, "Child")]


@patch("autocv.core.window_capture.win32gui.IsWindowVisible", return_value=False)
def test_window_enumeration_handler_skips_invisible(mock_visible):
    windows: list[tuple[int, str]] = []
    WindowCapture._window_enumeration_handler(1, windows)
    assert windows == []


@patch("autocv.core.window_capture.win32gui.IsWindowVisible", return_value=True)
@patch("autocv.core.window_capture.win32gui.GetWindowText", return_value="")
def test_window_enumeration_handler_skips_empty_title(mock_get, mock_visible):
    windows: list[tuple[int, str]] = []
    WindowCapture._window_enumeration_handler(1, windows)
    assert windows == []


@patch("autocv.core.window_capture.win32gui.IsWindowVisible", return_value=True)
@patch("autocv.core.window_capture.win32gui.GetWindowText", return_value="Title")
def test_window_enumeration_handler_collects_visible_title(mock_get, mock_visible):
    windows: list[tuple[int, str]] = []
    WindowCapture._window_enumeration_handler(1, windows)
    assert windows == [(1, "Title")]


@patch("autocv.core.window_capture.win32gui.GetClassName", return_value="")
def test_child_window_enumeration_handler_skips_empty_class_name(mock_get):
    children: list[tuple[int, str]] = []
    WindowCapture._child_window_enumeration_handler(1, children)
    assert children == []


@patch("autocv.core.window_capture.win32gui.GetClassName", return_value="Class")
def test_child_window_enumeration_handler_collects_class_name(mock_get):
    children: list[tuple[int, str]] = []
    WindowCapture._child_window_enumeration_handler(1, children)
    assert children == [(1, "Class")]


def test_normalize_region_defaults_to_full_window():
    assert WindowCapture._normalize_region(None, (0, 0, 10, 20)) == (0, 0, 10, 20)


@pytest.mark.parametrize("region", [(0, 0, 0, 1), (0, 0, 1, 0), (0, 0, -1, 1), (0, 0, 1, -1)])
def test_normalize_region_rejects_non_positive_dimensions(region):
    with pytest.raises(ValueError, match="positive width and height"):
        WindowCapture._normalize_region(region, (0, 0, 10, 10))


def test_normalize_region_rejects_outside_bounds():
    with pytest.raises(ValueError, match="outside the window bounds"):
        WindowCapture._normalize_region((20, 20, 5, 5), (0, 0, 10, 10))


def test_normalize_region_clamps_to_bounds():
    assert WindowCapture._normalize_region((-5, -5, 10, 10), (0, 0, 10, 10)) == (0, 0, 5, 5)
    assert WindowCapture._normalize_region((5, 5, 10, 10), (0, 0, 10, 10)) == (5, 5, 5, 5)


def test_capture_frame_persists_last_frame_when_requested():
    wc = WindowCapture(hwnd=1)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    with (
        patch.object(WindowCapture, "_get_window_bounds_cached", return_value=(0, 0, 1, 1)),
        patch.object(WindowCapture, "_normalize_region", return_value=(0, 0, 1, 1)),
        patch.object(WindowCapture, "_bitblt_to_array", return_value=frame),
    ):
        result = wc.capture_frame(persist=True)
    assert result is frame
    assert wc.last_frame is frame


def test_capture_frame_does_not_persist_when_disabled():
    wc = WindowCapture(hwnd=1)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    with (
        patch.object(WindowCapture, "_get_window_bounds_cached", return_value=(0, 0, 1, 1)),
        patch.object(WindowCapture, "_normalize_region", return_value=(0, 0, 1, 1)),
        patch.object(WindowCapture, "_bitblt_to_array", return_value=frame),
    ):
        wc.capture_frame(persist=False)
    assert wc.last_frame is None


@patch("autocv.core.window_capture.win32gui.GetWindowDC", return_value=0)
def test_bitblt_to_array_raises_invalid_handle(mock_get_dc):
    with pytest.raises(InvalidHandleError):
        WindowCapture._bitblt_to_array(123, (0, 0, 1, 1))


def test_bitblt_to_array_returns_bgr_and_cleans_up():
    rect = (0, 0, 2, 3)
    mem_dc = MagicMock()
    bmp_dc = MagicMock()
    bitmap = MagicMock()
    mem_dc.CreateCompatibleDC.return_value = bmp_dc
    bitmap.GetBitmapBits.return_value = bytes(2 * 3 * 4)
    bitmap.GetHandle.return_value = 999

    with (
        patch("autocv.core.window_capture.win32gui.GetWindowDC", return_value=123),
        patch("autocv.core.window_capture.win32gui.ReleaseDC") as mock_release,
        patch("autocv.core.window_capture.win32gui.DeleteObject") as mock_delete,
        patch("autocv.core.window_capture.win32ui.CreateDCFromHandle", return_value=mem_dc),
        patch("autocv.core.window_capture.win32ui.CreateBitmap", return_value=bitmap),
    ):
        frame = WindowCapture._bitblt_to_array(1, rect)

    assert frame.shape == (3, 2, 3)
    assert frame.flags.c_contiguous
    bmp_dc.DeleteDC.assert_called_once()
    mem_dc.DeleteDC.assert_called_once()
    mock_release.assert_called_once_with(1, 123)
    mock_delete.assert_called_once_with(999)


def test_bitblt_to_array_cleans_up_when_compatible_dc_creation_fails():
    mem_dc = MagicMock()
    mem_dc.CreateCompatibleDC.side_effect = RuntimeError("boom")

    with (
        patch("autocv.core.window_capture.win32gui.GetWindowDC", return_value=123),
        patch("autocv.core.window_capture.win32gui.ReleaseDC") as mock_release,
        patch("autocv.core.window_capture.win32gui.DeleteObject") as mock_delete,
        patch("autocv.core.window_capture.win32ui.CreateDCFromHandle", return_value=mem_dc),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            WindowCapture._bitblt_to_array(1, (0, 0, 1, 1))

    mem_dc.DeleteDC.assert_called_once()
    mock_release.assert_called_once_with(1, 123)
    mock_delete.assert_not_called()


def test_bitblt_to_array_releases_dc_when_initial_dc_creation_fails():
    with (
        patch("autocv.core.window_capture.win32gui.GetWindowDC", return_value=123),
        patch("autocv.core.window_capture.win32gui.ReleaseDC") as mock_release,
        patch("autocv.core.window_capture.win32ui.CreateDCFromHandle", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            WindowCapture._bitblt_to_array(1, (0, 0, 1, 1))

    mock_release.assert_called_once_with(1, 123)
