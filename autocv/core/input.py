"""This module defines the Input class, which includes methods for simulating mouse and keyboard input.

It provides functions for moving the mouse cursor (with human-like motion),
clicking, sending keystrokes, and more, allowing for interaction with an application
in a way that mimics human input.
"""

from __future__ import annotations

__all__ = ("Input",)

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self

import numpy as np
import win32api
import win32con
import win32gui

from autocv import constants

from . import check_valid_hwnd
from .vision import Vision

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


THREE = 3
HALF = 0.5


@dataclass(frozen=True, slots=True)
class _MotionConfig:
    """Tunable parameters for human-like mouse motion."""

    speed_base: int = constants.MOTION_SPEED_BASE
    gravity_min: float = constants.MOTION_GRAVITY_MIN
    gravity_jitter: float = constants.MOTION_GRAVITY_JITTER
    wind_min: float = constants.MOTION_WIND_MIN
    wind_jitter: float = constants.MOTION_WIND_JITTER
    timeout_seconds: float = constants.MOTION_TIMEOUT_SECONDS


def wind_mouse(
    rng: np.random.Generator,
    mover: Callable[[int, int], None],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    gravity: float,
    wind: float,
    min_wait: float,
    max_wait: float,
    max_step: float,
    target_area: float,
    timeout_seconds: float,
) -> None:
    """Move a point from start to end using wind/gravity-inspired motion.

    Args:
        rng: Random generator for jitter and timing.
        mover: Callable that moves the cursor to integer coordinates.
        start: Starting coordinates.
        end: Destination coordinates.
        gravity: Pull toward destination.
        wind: Randomness factor.
        min_wait: Minimum wait between steps (ms).
        max_wait: Maximum wait between steps (ms).
        max_step: Maximum step length per iteration.
        target_area: Threshold under which the cursor eases into the target.
        timeout_seconds: Hard timeout for the movement.

    Raises:
        TimeoutError: If movement exceeds ``timeout_seconds``.
    """
    timeout = time.perf_counter() + timeout_seconds
    sqrt_3 = math.sqrt(3)
    sqrt_5 = math.sqrt(5)

    xs, ys = start
    xe, ye = end
    x, y = xs, ys
    velo_x = 0.0
    velo_y = 0.0
    wind_x = 0.0
    wind_y = 0.0

    while True:
        if time.perf_counter() > timeout:
            raise TimeoutError

        if rng.random() < HALF:
            wind_x = wind_x / sqrt_3 + (rng.random() * (round(wind) * 2 + 1) - wind) / sqrt_5
            wind_y = wind_y / sqrt_3 + (rng.random() * (round(wind) * 2 + 1) - wind) / sqrt_5

        traveled_distance = math.hypot(x - xs, y - ys)
        remaining_distance = math.hypot(x - xe, y - ye)

        if remaining_distance <= 1:
            break

        if remaining_distance < target_area:
            step = (remaining_distance / 2) + (rng.random() * 6 - 3)
        elif traveled_distance < target_area:
            if traveled_distance < THREE:
                traveled_distance = 10 * rng.random()
            step = traveled_distance * (1 + rng.random() * 3)
        else:
            step = max_step

        step = min(step, max_step)
        if step < THREE:
            step = 3 + (rng.random() * 3)

        velo_x += wind_x + gravity * (xe - x) / remaining_distance
        velo_y += wind_y + gravity * (ye - y) / remaining_distance

        velo_mag = math.hypot(velo_x, velo_y)
        if velo_mag > step:
            random_dist = step / 3.0 + (step / 2 * rng.random())
            velo_x = (velo_x / velo_mag) * random_dist
            velo_y = (velo_y / velo_mag) * random_dist

        idle = (max_wait - min_wait) * (math.hypot(velo_x, velo_y) / max_step) + min_wait

        x += velo_x
        y += velo_y

        mover(round(x), round(y))
        time.sleep(idle / 100)

    mover(round(xe), round(ye))


class Input(Vision):
    """Extends the Vision class with functionalities for simulating user input.

    This class supports human-like mouse movements, clicks, and keyboard key presses.
    Randomness and delays are incorporated to mimic natural interaction.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes an Input object.

        Args:
            hwnd (int): Window handle that will receive simulated input. Defaults to -1.
        """
        super().__init__(hwnd)
        self._last_moved_point: tuple[int, int] = (0, 0)
        self._rng = np.random.default_rng()

        self._motion_config = _MotionConfig()
        self.__speed: Final[int] = self._motion_config.speed_base
        self.__gravity: Final[float] = (
            self._motion_config.gravity_min + self._rng.random() * self._motion_config.gravity_jitter
        )
        self.__wind: Final[float] = self._motion_config.wind_min + self._rng.random() * self._motion_config.wind_jitter

    def get_last_moved_point(self: Self) -> tuple[int, int]:
        """Returns the last point where the mouse cursor was moved.

        Returns:
            tuple[int, int]: Last cursor position that was targeted.
        """
        return self._last_moved_point

    @check_valid_hwnd
    def _get_topmost_hwnd(self: Self) -> int:
        """Retrieves the topmost (root) window handle for the target window.

        Returns:
            int: Handle for the topmost ancestor window.
        """
        parent_hwnd: int = win32gui.GetAncestor(self.hwnd, win32con.GA_ROOT)
        logger.debug("Found parent handle: %s", parent_hwnd)
        return parent_hwnd

    @staticmethod
    def _make_lparam(x: int, y: int) -> int:
        """Construct a packed LPARAM value for Win32 messages.

        Args:
            x (int): X-coordinate to encode in the message.
            y (int): Y-coordinate to encode in the message.

        Returns:
            int: Packed LPARAM value suitable for Win32 messages.
        """
        return (y << 16) | (x & 0xFFFF)

    @check_valid_hwnd
    def move_mouse(self: Self, x: int, y: int, *, human_like: bool = True, ghost_mouse: bool = True) -> None:
        """Moves the mouse cursor to the specified (x, y) coordinates.

        The movement is relative to the client area of the target window. If `human_like` is True,
        the movement will simulate natural motion.

        Args:
            x (int): Target x-coordinate inside the client area.
            y (int): Target y-coordinate inside the client area.
            human_like (bool): When ``True``, simulate human-like mouse motion.
            ghost_mouse (bool): When ``True``, rely on ghost mouse behaviour; otherwise emit OS cursor updates.
        """
        if not human_like:
            self._move_mouse(x, y, ghost_mouse=ghost_mouse)
            return

        # Determine a random speed factor.
        speed = (self._rng.random() * 15 + 30) / 10
        wind_mouse(
            rng=self._rng,
            mover=lambda xi, yi: self._move_mouse(xi, yi, ghost_mouse=ghost_mouse),
            start=(*self._last_moved_point,),
            end=(x, y),
            gravity=9,
            wind=3,
            min_wait=5 / speed,
            max_wait=10 / speed,
            max_step=10 * speed,
            target_area=8 * speed,
            timeout_seconds=self._motion_config.timeout_seconds,
        )

        self._last_moved_point = (x, y)

    @check_valid_hwnd
    def _move_mouse(self: Self, x: int, y: int, *, ghost_mouse: bool = True) -> None:
        """Moves the mouse cursor to the specified (x, y) coordinates.

        The movement is performed relative to the client area of the target window.
        If `ghost_mouse` is True, the movement is simulated via message passing.

        Args:
            x (int): Target x-coordinate inside the client area.
            y (int): Target y-coordinate inside the client area.
            ghost_mouse (bool): When ``True``, simulate the movement using ghost mouse techniques.
        """
        # Convert client coordinates to screen coordinates.
        screen_point = win32gui.ClientToScreen(self.hwnd, (x, y))
        if ghost_mouse:
            result = win32gui.SendMessage(self.hwnd, win32con.WM_NCHITTEST, 0, self._make_lparam(x, y))
            win32api.SendMessage(
                self.hwnd,
                win32con.WM_SETCURSOR,
                self.hwnd,  # type: ignore[arg-type]
                self._make_lparam(result, win32con.WM_MOUSEMOVE),  # type: ignore[arg-type]
            )
            win32api.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, self._make_lparam(*screen_point))
        else:
            win32api.SetCursorPos(screen_point)
        self._last_moved_point = (x, y)

    @check_valid_hwnd
    def click_mouse(self: Self, button: int = 1, *, send_message: bool = False) -> None:
        """Simulates a mouse click at the last moved position.

        Args:
            button (int): Mouse button identifier (1=left, 2=right, 3=middle).
            send_message (bool): When ``True``, dispatches ``SendMessage`` instead of ``PostMessage``.

        """
        screen_point = win32gui.ClientToScreen(self.hwnd, self._last_moved_point)
        button_messages = {
            1: win32con.WM_LBUTTONDOWN,
            2: win32con.WM_RBUTTONDOWN,
            3: win32con.WM_MBUTTONDOWN,
        }
        button_to_press = button_messages.get(button, win32con.WM_LBUTTONDOWN)
        screen_lparam = self._make_lparam(*screen_point)
        result = win32gui.SendMessage(self.hwnd, win32con.WM_NCHITTEST, 0, screen_lparam)
        client_lparam = self._make_lparam(*self._last_moved_point)

        # Notify the parent window.
        win32gui.SendMessage(
            win32gui.GetParent(self.hwnd),
            win32con.WM_PARENTNOTIFY,
            button_to_press,
            client_lparam,
        )

        top_hwnd = self._get_topmost_hwnd()
        lparam_button = self._make_lparam(result, button_to_press)
        win32gui.SendMessage(top_hwnd, win32con.WM_MOUSEACTIVATE, top_hwnd, lparam_button)
        win32api.SendMessage(self.hwnd, win32con.WM_SETCURSOR, self.hwnd, lparam_button)  # type: ignore[arg-type]

        if send_message:
            last_moved_point_lparam = win32api.MAKELONG(*self._last_moved_point)  # type: ignore[no-untyped-call]
            win32gui.SendMessage(self.hwnd, button_to_press, button, last_moved_point_lparam)
            time.sleep(self._rng.integers(10, 50).astype(int) / 1_000)
            win32gui.SendMessage(self.hwnd, button_to_press + 1, 0, last_moved_point_lparam)
        else:
            win32gui.PostMessage(self.hwnd, button_to_press, button, screen_lparam)
            time.sleep(self._rng.integers(10, 50).astype(int) / 1_000)
            win32gui.PostMessage(self.hwnd, button_to_press + 1, 0, screen_lparam)

    @check_valid_hwnd
    def press_vk_key(self: Self, vk_code: int) -> None:
        """Simulates pressing a virtual key.

        Args:
            vk_code (int): Virtual-key code to press.

        """
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]
        l_param = (scan_code << 16) | 1
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_CHAR, chr(vk_code), l_param)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def release_vk_key(self: Self, vk_code: int) -> None:
        """Simulates releasing a virtual key.

        Args:
            vk_code (int): Virtual-key code to release.

        """
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]
        l_param = (scan_code << 16) | 1
        l_param |= 0xC0000000
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def send_vk_key(self: Self, vk_code: int) -> None:
        """Sends a virtual key by simulating a press and release.

        Args:
            vk_code (int): Virtual-key code to send.

        """
        self.press_vk_key(vk_code)
        time.sleep(self._rng.integers(3, 5).astype(int) / 1_000)
        self.release_vk_key(vk_code)

    @staticmethod
    def get_async_key_state(vk_code: int) -> bool:
        """Retrieves the asynchronous state of a specified virtual key.

        Args:
            vk_code (int): Virtual-key code tested via ``GetAsyncKeyState``.

        Returns:
            bool: ``True`` if the key was pressed since the previous poll, otherwise ``False``.
        """
        return bool(win32api.GetAsyncKeyState(vk_code))

    @check_valid_hwnd
    def send_keys(self: Self, characters: str) -> None:
        """Sends a sequence of keystrokes to the active window.

        Args:
            characters (str): Characters to emit sequentially.

        """
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]

        for c in characters:
            vk = win32api.VkKeyScan(c)
            scan_code = win32api.MapVirtualKey(ord(c.upper()), 0)  # type: ignore[no-untyped-call]
            l_param = (scan_code << 16) | 1
            win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk, l_param)
            win32api.SendMessage(self.hwnd, win32con.WM_CHAR, ord(c), l_param)  # type: ignore[arg-type]
            time.sleep(self._rng.integers(3, 5).astype(int) / 1_000)
            l_param |= 0xC0000000
            win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk, l_param)
            time.sleep(self._rng.integers(20, 60).astype(int) / 1_000)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]
