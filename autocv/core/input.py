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

import numpy as np
import win32api
import win32con
import win32gui
from typing_extensions import Self

from .decorators import check_valid_hwnd
from .vision import Vision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MouseMotionConfig:
    speed: int
    gravity: float
    wind: float


BUTTON_DOWN_MESSAGES: dict[int, int] = {
    1: win32con.WM_LBUTTONDOWN,
    2: win32con.WM_RBUTTONDOWN,
    3: win32con.WM_MBUTTONDOWN,
}
MOUSE_UP_OFFSET = 1

THREE = 3
HALF = 0.5


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
        self._mouse_motion = MouseMotionConfig(
            speed=16,
            gravity=8 + self._rng.random() / 2,
            wind=4 + self._rng.random() / 2,
        )

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

    def _sleep_ms(self: Self, low: int, high: int) -> None:
        """Pause execution for a random duration in milliseconds."""
        if low > high:
            low, high = high, low
        delay_ms = int(self._rng.integers(low, high + 1))
        time.sleep(delay_ms / 1_000)

    @check_valid_hwnd
    def move_mouse(self: Self, x: int, y: int, *, human_like: bool = True, ghost_mouse: bool = True) -> None:
        """Move the cursor to client coordinates, optionally simulating human motion."""
        if not human_like:
            self._move_mouse(x, y, ghost_mouse=ghost_mouse)
            return

        # Determine a random speed factor.
        speed = (self._rng.random() * 15 + 30) / 10
        config = self._mouse_motion
        self._wind_mouse(
            *self._last_moved_point,
            x,
            y,
            gravity=config.gravity,
            wind=config.wind,
            min_wait=5 / speed,
            max_wait=10 / speed,
            max_step=10 * speed,
            target_area=8 * speed,
            ghost_mouse=ghost_mouse,
        )

    def _wind_mouse(
        self: Self,
        xs: float,
        ys: float,
        xe: float,
        ye: float,
        gravity: float,
        wind: float,
        min_wait: float,
        max_wait: float,
        max_step: float,
        target_area: float,
        *,
        ghost_mouse: bool = True,
    ) -> None:
        """Moves the mouse using a wind-based, human-like trajectory.

        The movement is influenced by gravity and wind, producing a curved path.

        Args:
            xs (float): Starting x-coordinate of the cursor.
            ys (float): Starting y-coordinate of the cursor.
            xe (float): Destination x-coordinate.
            ye (float): Destination y-coordinate.
            gravity (float): Strength of the pull toward the destination.
            wind (float): Randomness factor to jitter the trajectory.
            min_wait (float): Minimum wait time between steps, in milliseconds.
            max_wait (float): Maximum wait time between steps, in milliseconds.
            max_step (float): Maximum distance covered per iteration.
            target_area (float): Threshold under which the cursor eases into the target.
            ghost_mouse (bool): When ``True``, rely on ghost mouse simulation instead of OS cursor updates.

        Raises:
            TimeoutError: If the movement does not complete within 15 seconds.
        """
        # Set a timeout for the entire movement.
        timeout = time.perf_counter() + 15

        sqrt_3 = math.sqrt(3)
        sqrt_5 = math.sqrt(5)

        x, y = xs, ys
        velo_x = 0.0
        velo_y = 0.0
        wind_x = 0.0
        wind_y = 0.0

        while True:
            if time.perf_counter() > timeout:
                raise TimeoutError

            # Randomly adjust wind.
            if self._rng.random() < HALF:
                wind_x = wind_x / sqrt_3 + (self._rng.random() * (round(wind) * 2 + 1) - wind) / sqrt_5
                wind_y = wind_y / sqrt_3 + (self._rng.random() * (round(wind) * 2 + 1) - wind) / sqrt_5

            traveled_distance = math.hypot(x - xs, y - ys)
            remaining_distance = math.hypot(x - xe, y - ye)

            if remaining_distance <= 1:
                break

            if remaining_distance < target_area:
                step = (remaining_distance / 2) + (self._rng.random() * 6 - 3)
            elif traveled_distance < target_area:
                if traveled_distance < THREE:
                    traveled_distance = 10 * self._rng.random()
                step = traveled_distance * (1 + self._rng.random() * 3)
            else:
                step = max_step

            step = min(step, max_step)
            if step < THREE:
                step = 3 + (self._rng.random() * 3)

            velo_x += wind_x + gravity * (xe - x) / remaining_distance
            velo_y += wind_y + gravity * (ye - y) / remaining_distance

            if math.hypot(velo_x, velo_y) > step:
                random_dist = step / 3.0 + (step / 2 * self._rng.random())
                velo_mag = math.hypot(velo_x, velo_y)
                velo_x = (velo_x / velo_mag) * random_dist
                velo_y = (velo_y / velo_mag) * random_dist

            idle = (max_wait - min_wait) * (math.hypot(velo_x, velo_y) / max_step) + min_wait

            x += velo_x
            y += velo_y

            self._move_mouse(round(x), round(y), ghost_mouse=ghost_mouse)
            time.sleep(idle / 100)

        self._move_mouse(round(xe), round(ye), ghost_mouse=ghost_mouse)

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
        """Simulate a mouse click at the last recorded position."""
        screen_point = win32gui.ClientToScreen(self.hwnd, self._last_moved_point)
        button_to_press = BUTTON_DOWN_MESSAGES.get(button, win32con.WM_LBUTTONDOWN)
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
            self._sleep_ms(10, 49)
            win32gui.SendMessage(self.hwnd, button_to_press + MOUSE_UP_OFFSET, 0, last_moved_point_lparam)
        else:
            win32gui.PostMessage(self.hwnd, button_to_press, button, screen_lparam)
            self._sleep_ms(10, 49)
            win32gui.PostMessage(self.hwnd, button_to_press + MOUSE_UP_OFFSET, 0, screen_lparam)

    @check_valid_hwnd
    def press_vk_key(self: Self, vk_code: int) -> None:
        """Press a virtual key by sending the corresponding window messages."""
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]
        l_param = (scan_code << 16) | 1
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_CHAR, chr(vk_code), l_param)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def release_vk_key(self: Self, vk_code: int) -> None:
        """Release a virtual key that was previously pressed."""
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]
        l_param = (scan_code << 16) | 1
        l_param |= 0xC0000000
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def send_vk_key(self: Self, vk_code: int) -> None:
        """Press and release a virtual key with a short delay."""
        self.press_vk_key(vk_code)
        self._sleep_ms(3, 4)
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
        """Emit each character in sequence to the active window."""
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]

        for c in characters:
            vk = win32api.VkKeyScan(c)
            scan_code = win32api.MapVirtualKey(ord(c.upper()), 0)  # type: ignore[no-untyped-call]
            l_param = (scan_code << 16) | 1
            win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk, l_param)
            win32api.SendMessage(self.hwnd, win32con.WM_CHAR, ord(c), l_param)  # type: ignore[arg-type]
            self._sleep_ms(3, 4)
            l_param |= 0xC0000000
            win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk, l_param)
            self._sleep_ms(20, 59)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]
