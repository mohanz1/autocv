import numpy as np
import pytest
from mock import patch, MagicMock
from autocv import AutoCV


@pytest.fixture
def autocv():
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        return AutoCV(hwnd=1234)


class TestAutoCVInput:
    @patch("win32gui.ClientToScreen", return_value=(500, 500))
    @patch("win32gui.SendMessage")
    @patch("win32api.PostMessage")
    def test__move_mouse_ghost(self, post, send, to_screen, autocv):
        autocv._move_mouse(100, 100, ghost_mouse=True)
        post.assert_called()
        send.assert_called()
        assert autocv.get_last_moved_point() == (100, 100)

    @patch("autocv.core.Input._move_mouse")
    def test_move_mouse_not_human_like(self, move, autocv):
        autocv.move_mouse(50, 50, human_like=False)
        move.assert_called_once_with(50, 50, ghost_mouse=True)

    @patch("win32gui.ClientToScreen", return_value=(200, 200))
    @patch("win32gui.SendMessage")
    @patch("win32gui.PostMessage")
    @patch("win32gui.GetParent", return_value=1234)
    @patch("win32gui.GetAncestor", return_value=1234)
    def test_click_mouse_ghost(self, ga, gp, post, send, to_screen, autocv):
        autocv._last_moved_point = (10, 10)
        autocv.click_mouse(button=1, send_message=False)
        post.assert_called()

    @patch("win32api.SendMessage")
    @patch("win32api.MapVirtualKey", return_value=42)
    def test_press_vk_key(self, mvk, send, autocv):
        autocv.press_vk_key(65)
        assert send.call_count >= 3

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
    @patch("win32api.VkKeyScan", return_value=65)
    @patch("win32api.MapVirtualKey", return_value=1)
    def test_send_keys(self, mvk, scan, send, autocv):
        autocv.send_keys("a")
        assert send.call_count > 0
