import numpy as np
import pytest
import win32con
from unittest.mock import MagicMock, call, patch

from autocv import AutoCV
from autocv.core.input import wind_mouse


@pytest.fixture
def autocv():
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        return AutoCV(hwnd=1234)


class TestAutoCVInput:
    @patch("win32gui.ClientToScreen", return_value=(500, 500))
    @patch("win32gui.SendMessage")
    @patch("win32api.SendMessage")
    @patch("win32api.PostMessage")
    def test__move_mouse_ghost(self, post, send_api, send_gui, to_screen, autocv):
        send_gui.return_value = 7
        autocv._move_mouse(100, 100, ghost_mouse=True)
        send_gui.assert_called_once_with(1234, win32con.WM_NCHITTEST, 0, autocv._make_lparam(500, 500))
        send_api.assert_called_once_with(
            1234,
            win32con.WM_SETCURSOR,
            1234,
            autocv._make_lparam(7, win32con.WM_MOUSEMOVE),
        )
        post.assert_called_once_with(1234, win32con.WM_MOUSEMOVE, 0, autocv._make_lparam(100, 100))
        assert autocv.get_last_moved_point() == (100, 100)

    @patch("autocv.core.Input._move_mouse")
    def test_move_mouse_not_human_like(self, move, autocv):
        autocv.move_mouse(50, 50, human_like=False)
        move.assert_called_once_with(50, 50, ghost_mouse=True)

    @patch("win32gui.ClientToScreen", return_value=(200, 200))
    @patch("win32gui.SendMessage")
    @patch("win32gui.PostMessage")
    @patch("win32api.SendMessage")
    @patch("win32gui.GetParent", return_value=1234)
    @patch("win32gui.GetAncestor", return_value=1234)
    @patch.object(AutoCV, "_sleep_ms")
    def test_click_mouse_ghost(self, mock_sleep, ga, gp, send_api, post, send, to_screen, autocv):
        autocv._last_moved_point = (10, 10)
        autocv.click_mouse(button=1, send_message=False)
        client_lparam = autocv._make_lparam(10, 10)
        post.assert_any_call(1234, win32con.WM_LBUTTONDOWN, 1, client_lparam)
        post.assert_any_call(1234, win32con.WM_LBUTTONUP, 0, client_lparam)

    @patch("win32api.MAKELONG", return_value=123)
    @patch("win32gui.ClientToScreen", return_value=(200, 200))
    @patch("win32gui.SendMessage")
    @patch("win32api.SendMessage")
    @patch("win32gui.GetParent", return_value=1234)
    @patch("win32gui.GetAncestor", return_value=1234)
    @patch.object(AutoCV, "_sleep_ms")
    def test_click_mouse_send_message(self, mock_sleep, ga, gp, send_api, send, to_screen, make_long, autocv):
        autocv._last_moved_point = (10, 10)
        autocv.click_mouse(button=1, send_message=True)
        send.assert_any_call(1234, win32con.WM_LBUTTONDOWN, 1, 123)
        send.assert_any_call(1234, win32con.WM_LBUTTONUP, 0, 123)

    @patch("win32api.SendMessage")
    @patch("win32api.MapVirtualKey", return_value=42)
    def test_press_vk_key(self, mvk, send, autocv):
        autocv.press_vk_key(65)
        assert send.call_count >= 3

    @patch("win32api.MapVirtualKey", return_value=42)
    def test_make_vk_lparam_keyup_sets_flag(self, mvk, autocv):
        down = autocv._make_vk_lparam(65)
        up = autocv._make_vk_lparam(65, keyup=True)
        assert up != down

    def test_make_lparam_packs_words(self, autocv):
        assert autocv._make_lparam(1, 2) == (2 << 16) | 1
        assert autocv._make_lparam(-1, -2) == 0xFFFEFFFF

    @patch("autocv.core.input.time.sleep")
    def test_sleep_ms_uses_rng_range(self, mock_sleep, autocv):
        autocv._sleep_ms(10, 11)
        mock_sleep.assert_called_once()

    @patch("win32api.SendMessage")
    @patch("win32api.MapVirtualKey", return_value=42)
    def test_release_vk_key(self, mvk, send, autocv):
        autocv.release_vk_key(65)
        assert send.call_count >= 3

    @patch.object(AutoCV, "press_vk_key")
    @patch.object(AutoCV, "release_vk_key")
    def test_send_vk_key(self, release, press, autocv):
        autocv.send_vk_key(65)
        press.assert_called_once()
        release.assert_called_once()

    @patch("win32api.GetAsyncKeyState", return_value=1)
    def test_get_async_key_state_true(self, mock_key_state):
        assert AutoCV.get_async_key_state(65) is True

    @patch("win32api.SendMessage")
    @patch("win32api.MapVirtualKey", return_value=42)
    @patch("win32api.VkKeyScan", return_value=65)
    @patch.object(AutoCV, "_sleep_ms")
    def test_send_keys(self, mock_sleep, scan, map_vk, send, autocv):
        autocv.send_keys("a")
        assert send.call_count > 0

    @patch("win32api.SendMessage")
    @patch("win32api.MapVirtualKey", return_value=42)
    @patch("win32api.VkKeyScan", return_value=0x0141)
    @patch.object(AutoCV, "_sleep_ms")
    @patch.object(AutoCV, "_set_window_active")
    def test_send_keys_holds_shift_for_shifted_char(self, mock_active, mock_sleep, scan, map_vk, send, autocv):
        autocv.send_keys("A")

        key_lparam = autocv._make_vk_lparam(65)
        shift_down = autocv._make_vk_lparam(win32con.VK_SHIFT)
        shift_up = autocv._make_vk_lparam(win32con.VK_SHIFT, keyup=True)
        assert send.call_args_list == [
            call(1234, win32con.WM_KEYDOWN, win32con.VK_SHIFT, shift_down),
            call(1234, win32con.WM_KEYDOWN, 65, key_lparam),
            call(1234, win32con.WM_CHAR, ord("A"), key_lparam),
            call(1234, win32con.WM_KEYUP, 65, key_lparam | 0xC0000000),
            call(1234, win32con.WM_KEYUP, win32con.VK_SHIFT, shift_up),
        ]
        assert mock_active.call_args_list == [call(active=True), call(active=False)]

    @patch("win32api.VkKeyScan", return_value=-1)
    @patch.object(AutoCV, "_set_window_active")
    def test_send_keys_releases_active_state_on_unmappable_character(self, mock_active, scan, autocv):
        with pytest.raises(ValueError, match="Cannot map character"):
            autocv.send_keys("\u2603")

        assert mock_active.call_args_list == [call(active=True), call(active=False)]

    @patch("win32gui.ClientToScreen", return_value=(200, 200))
    @patch("win32api.SetCursorPos")
    def test__move_mouse_non_ghost_calls_set_cursor_pos(self, mock_set_cursor, mock_to_screen, autocv):
        autocv._move_mouse(10, 20, ghost_mouse=False)
        mock_set_cursor.assert_called_once_with((200, 200))
        assert autocv.get_last_moved_point() == (10, 20)

    @patch("autocv.core.input.wind_mouse")
    def test_move_mouse_human_like_invokes_wind_mouse(self, mock_wind_mouse, autocv):
        def fake_wind_mouse(*, mover, end, **_kwargs):
            mover(int(end[0]), int(end[1]))

        mock_wind_mouse.side_effect = fake_wind_mouse
        autocv._move_mouse = MagicMock()

        autocv.move_mouse(5, 6, human_like=True, ghost_mouse=False)

        autocv._move_mouse.assert_called_once_with(5, 6, ghost_mouse=False)
        assert autocv.get_last_moved_point() == (5, 6)

    @patch("win32api.VkKeyScan", return_value=0x0341)
    def test_resolve_keypress_extracts_virtual_key_and_modifiers(self, scan, autocv):
        vk_code, modifiers = autocv._resolve_keypress("A")
        assert vk_code == 0x41
        assert modifiers == (win32con.VK_SHIFT, win32con.VK_CONTROL)


@patch("autocv.core.input.time.sleep")
@patch("autocv.core.input.time.perf_counter", return_value=0.0)
def test_wind_mouse_moves_to_target(mock_perf_counter, mock_sleep):
    rng = np.random.default_rng(0)
    points: list[tuple[int, int]] = []

    wind_mouse(
        rng=rng,
        mover=lambda x, y: points.append((x, y)),
        start=(0.0, 0.0),
        end=(50.0, 0.0),
        gravity=9.0,
        wind=3.0,
        min_wait=0.0,
        max_wait=0.0,
        max_step=10.0,
        target_area=8.0,
        timeout_seconds=1.0,
    )

    assert len(points) > 1
    assert points[-1] == (50, 0)


@patch("autocv.core.input.time.perf_counter", side_effect=[0.0, 10.0])
def test_wind_mouse_raises_timeout(mock_perf_counter):
    rng = np.random.default_rng(0)

    with pytest.raises(TimeoutError):
        wind_mouse(
            rng=rng,
            mover=lambda _x, _y: None,
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            gravity=9.0,
            wind=3.0,
            min_wait=0.0,
            max_wait=0.0,
            max_step=10.0,
            target_area=8.0,
            timeout_seconds=1.0,
        )
