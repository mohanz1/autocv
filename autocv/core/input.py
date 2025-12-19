"""Win32 input simulation helpers.

This module defines :class:`~autocv.core.input.Input`, an extension of
:class:`~autocv.core.vision.Vision` that can simulate mouse and keyboard input
for a target window handle.

Most APIs accept client-area coordinates relative to the target window.
"""

from __future__ import annotations

__all__ = ("Input",)

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import numpy as np
import win32api
import win32con
import win32gui
from typing_extensions import Self

from autocv import constants

from .decorators import check_valid_hwnd
from .vision import Vision

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

ClientPoint: TypeAlias = tuple[int, int]
FloatPoint: TypeAlias = tuple[float, float]


_MIN_STEP_PIXELS: Final[int] = 3
_HALF_PROBABILITY: Final[float] = 0.5
_SLEEP_DIVISOR: Final[float] = 100.0

_CLICK_DELAY_MS_RANGE: Final[tuple[int, int]] = (10, 50)
_KEY_PRESS_DELAY_MS_RANGE: Final[tuple[int, int]] = (3, 5)
_BETWEEN_KEYS_DELAY_MS_RANGE: Final[tuple[int, int]] = (20, 60)

_MOUSE_BUTTON_DOWN_MESSAGES: Final[dict[int, int]] = {
    1: win32con.WM_LBUTTONDOWN,
    2: win32con.WM_RBUTTONDOWN,
    3: win32con.WM_MBUTTONDOWN,
}

_KEYUP_LPARAM_FLAG: Final[int] = 0xC0000000

# Backwards compatibility for legacy module-level names.
THREE: Final[int] = _MIN_STEP_PIXELS
HALF: Final[float] = _HALF_PROBABILITY


@dataclass(frozen=True, slots=True)
class _MotionConfig:
    """Tunable parameters for human-like mouse motion."""

    timeout_seconds: float = constants.MOTION_TIMEOUT_SECONDS
    gravity: float = 9.0
    wind: float = 3.0
    speed_random_range: float = 15.0
    speed_random_offset: float = 30.0
    speed_random_divisor: float = 10.0
    min_wait_base: float = 5.0
    max_wait_base: float = 10.0
    max_step_base: float = 10.0
    target_area_base: float = 8.0


def wind_mouse(
    rng: np.random.Generator,
    mover: Callable[[int, int], None],
    start: FloatPoint,
    end: FloatPoint,
    *,
    gravity: float,
    wind: float,
    min_wait: float,
    max_wait: float,
    max_step: float,
    target_area: float,
    timeout_seconds: float,
) -> None:
    """Move a point from ``start`` to ``end`` using wind/gravity-inspired motion.

    The algorithm adds random "wind" to introduce jitter while applying "gravity"
    toward the destination. The supplied ``mover`` callback is invoked with
    rounded integer coordinates for each intermediate step.

    Args:
        rng: Random generator for jitter and timing.
        mover: Callable that moves the cursor to integer coordinates.
        start: Starting coordinates.
        end: Destination coordinates.
        gravity: Pull toward destination.
        wind: Randomness factor.
        min_wait: Minimum wait between steps, scaled via ``time.sleep(min_wait / 100)``.
        max_wait: Maximum wait between steps, scaled via ``time.sleep(max_wait / 100)``.
        max_step: Maximum step length per iteration.
        target_area: Threshold under which the cursor eases into the target.
        timeout_seconds: Hard timeout for the movement.

    Raises:
        TimeoutError: If movement exceeds ``timeout_seconds``.
    """
    deadline = time.perf_counter() + timeout_seconds
    sqrt_3 = math.sqrt(3.0)
    sqrt_5 = math.sqrt(5.0)
    wind_range = round(wind) * 2 + 1

    start_x, start_y = start
    end_x, end_y = end
    x, y = start_x, start_y
    velocity_x = 0.0
    velocity_y = 0.0
    wind_x = 0.0
    wind_y = 0.0

    while True:
        if time.perf_counter() > deadline:
            raise TimeoutError

        if rng.random() < HALF:
            wind_x = wind_x / sqrt_3 + (rng.random() * wind_range - wind) / sqrt_5
            wind_y = wind_y / sqrt_3 + (rng.random() * wind_range - wind) / sqrt_5

        traveled_distance = math.hypot(x - start_x, y - start_y)
        remaining_distance = math.hypot(x - end_x, y - end_y)

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

        velocity_x += wind_x + gravity * (end_x - x) / remaining_distance
        velocity_y += wind_y + gravity * (end_y - y) / remaining_distance

        velocity_mag = math.hypot(velocity_x, velocity_y)
        if velocity_mag > step:
            random_dist = step / 3.0 + (step / 2 * rng.random())
            velocity_x = (velocity_x / velocity_mag) * random_dist
            velocity_y = (velocity_y / velocity_mag) * random_dist

        idle = (max_wait - min_wait) * (math.hypot(velocity_x, velocity_y) / max_step) + min_wait

        x += velocity_x
        y += velocity_y

        mover(round(x), round(y))
        time.sleep(idle / _SLEEP_DIVISOR)

    mover(round(end_x), round(end_y))


class Input(Vision):
    """Simulate mouse and keyboard input for a target window handle.

    The class extends :class:`~autocv.core.vision.Vision` with helpers that post
    and send Win32 messages for mouse movement, clicks, and keystrokes.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initialise the input helper.

        Args:
            hwnd: Window handle that will receive simulated input. Defaults to ``-1``.
        """
        super().__init__(hwnd)
        self._last_moved_point: ClientPoint = (0, 0)
        self._rng: np.random.Generator = np.random.default_rng()
        self._motion_config: _MotionConfig = _MotionConfig()
        # Compatibility: historical private attributes retained for callers that inspect them
        # and to preserve the original RNG advancement order.
        self.__speed: Final[int] = constants.MOTION_SPEED_BASE
        self.__gravity: Final[float] = (
            constants.MOTION_GRAVITY_MIN + self._rng.random() * constants.MOTION_GRAVITY_JITTER
        )
        self.__wind: Final[float] = constants.MOTION_WIND_MIN + self._rng.random() * constants.MOTION_WIND_JITTER

    def get_last_moved_point(self: Self) -> ClientPoint:
        """Return the last client-area point targeted by :meth:`move_mouse`.

        Returns:
            The most recent ``(x, y)`` point in client coordinates.
        """
        return self._last_moved_point

    def _sleep_ms(self: Self, low: int, high: int) -> None:
        """Sleep for a random duration in milliseconds drawn from ``[low, high)``."""
        time.sleep(int(self._rng.integers(low, high)) / 1_000)

    def _set_window_active(self: Self, *, active: bool) -> None:
        """Send a ``WM_ACTIVATE`` message to the target window."""
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, int(active), self.hwnd)  # type: ignore[arg-type]

    @staticmethod
    def _make_vk_lparam(vk_code: int, *, keyup: bool = False) -> int:
        """Construct an LPARAM value for WM_KEYDOWN/WM_KEYUP messages."""
        scan_code = int(win32api.MapVirtualKey(vk_code, 0))  # type: ignore[no-untyped-call]
        l_param = (scan_code << 16) | 1
        if keyup:
            l_param |= _KEYUP_LPARAM_FLAG
        return l_param

    @check_valid_hwnd
    def _get_topmost_hwnd(self: Self) -> int:
        """Return the topmost (root) window handle for the target window."""
        parent_hwnd: int = win32gui.GetAncestor(self.hwnd, win32con.GA_ROOT)
        logger.debug("Found parent handle: %s", parent_hwnd)
        return parent_hwnd

    @staticmethod
    def _make_lparam(x: int, y: int) -> int:
        """Construct a packed LPARAM value for Win32 messages.

        Args:
            x: X-coordinate (low word).
            y: Y-coordinate (high word).

        Returns:
            Packed LPARAM value suitable for Win32 message APIs.
        """
        return (y << 16) | (x & 0xFFFF)

    @check_valid_hwnd
    def move_mouse(self: Self, x: int, y: int, *, human_like: bool = True, ghost_mouse: bool = True) -> None:
        """Move the mouse cursor to ``(x, y)`` in client coordinates.

        Args:
            x: Target X coordinate inside the window client area.
            y: Target Y coordinate inside the window client area.
            human_like: When ``True``, simulate human-like mouse motion.
            ghost_mouse: When ``True``, simulate motion using Win32 messages; when ``False``, move the OS cursor.
        """
        if not human_like:
            self._move_mouse(x, y, ghost_mouse=ghost_mouse)
            return

        motion = self._motion_config
        speed = (
            self._rng.random() * motion.speed_random_range + motion.speed_random_offset
        ) / motion.speed_random_divisor
        start: FloatPoint = (float(self._last_moved_point[0]), float(self._last_moved_point[1]))
        end: FloatPoint = (float(x), float(y))
        wind_mouse(
            rng=self._rng,
            mover=lambda xi, yi: self._move_mouse(xi, yi, ghost_mouse=ghost_mouse),
            start=start,
            end=end,
            gravity=motion.gravity,
            wind=motion.wind,
            min_wait=motion.min_wait_base / speed,
            max_wait=motion.max_wait_base / speed,
            max_step=motion.max_step_base * speed,
            target_area=motion.target_area_base * speed,
            timeout_seconds=motion.timeout_seconds,
        )

        self._last_moved_point = (x, y)

    @check_valid_hwnd
    def _move_mouse(self: Self, x: int, y: int, *, ghost_mouse: bool = True) -> None:
        """Move the mouse cursor to ``(x, y)`` in client coordinates."""
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
        """Click a mouse button at the last moved position.

        Args:
            button: Mouse button identifier (1=left, 2=right, 3=middle).
            send_message: When ``True``, use ``SendMessage`` instead of ``PostMessage`` for the click.
        """
        screen_point = win32gui.ClientToScreen(self.hwnd, self._last_moved_point)
        button_to_press = _MOUSE_BUTTON_DOWN_MESSAGES.get(button, win32con.WM_LBUTTONDOWN)
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

        dispatch = win32gui.SendMessage if send_message else win32gui.PostMessage
        click_lparam: int = int(win32api.MAKELONG(*self._last_moved_point)) if send_message else screen_lparam  # type: ignore[no-untyped-call]
        dispatch(self.hwnd, button_to_press, button, click_lparam)
        self._sleep_ms(*_CLICK_DELAY_MS_RANGE)
        dispatch(self.hwnd, button_to_press + 1, 0, click_lparam)

    @check_valid_hwnd
    def press_vk_key(self: Self, vk_code: int) -> None:
        """Press a virtual key using Win32 keyboard messages.

        Args:
            vk_code: Virtual-key code (``VK_*``) to press.

        Notes:
            This method sends ``WM_CHAR`` with ``chr(vk_code)`` to preserve historical behaviour.
        """
        l_param = self._make_vk_lparam(vk_code)
        self._set_window_active(active=True)
        win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_CHAR, chr(vk_code), l_param)  # type: ignore[arg-type]
        self._set_window_active(active=False)

    @check_valid_hwnd
    def release_vk_key(self: Self, vk_code: int) -> None:
        """Release a virtual key using Win32 keyboard messages.

        Args:
            vk_code: Virtual-key code (``VK_*``) to release.
        """
        l_param = self._make_vk_lparam(vk_code, keyup=True)
        self._set_window_active(active=True)
        win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk_code, l_param)  # type: ignore[arg-type]
        self._set_window_active(active=False)

    @check_valid_hwnd
    def send_vk_key(self: Self, vk_code: int) -> None:
        """Send a virtual key by pressing and releasing it.

        Args:
            vk_code: Virtual-key code (``VK_*``) to send.
        """
        self.press_vk_key(vk_code)
        self._sleep_ms(*_KEY_PRESS_DELAY_MS_RANGE)
        self.release_vk_key(vk_code)

    @staticmethod
    def get_async_key_state(vk_code: int) -> bool:
        """Return the asynchronous state of a specified virtual key.

        Args:
            vk_code: Virtual-key code tested via ``GetAsyncKeyState``.

        Returns:
            ``True`` if the key was pressed since the previous poll, otherwise ``False``.
        """
        return bool(win32api.GetAsyncKeyState(vk_code))

    @check_valid_hwnd
    def send_keys(self: Self, characters: str) -> None:
        """Send a sequence of characters to the target window.

        Args:
            characters: Characters to emit sequentially.
        """
        self._set_window_active(active=True)

        send_message = cast("Any", win32api.SendMessage)
        vk_key_scan = cast("Any", win32api.VkKeyScan)
        map_virtual_key = cast("Any", win32api.MapVirtualKey)

        for character in characters:
            vk: int = int(vk_key_scan(character))
            scan_code: int = int(map_virtual_key(ord(character.upper()), 0))
            l_param: int = (scan_code << 16) | 1
            send_message(self.hwnd, win32con.WM_KEYDOWN, vk, l_param)
            send_message(self.hwnd, win32con.WM_CHAR, ord(character), l_param)
            self._sleep_ms(*_KEY_PRESS_DELAY_MS_RANGE)
            send_message(self.hwnd, win32con.WM_KEYUP, vk, l_param | _KEYUP_LPARAM_FLAG)
            self._sleep_ms(*_BETWEEN_KEYS_DELAY_MS_RANGE)

        self._set_window_active(active=False)
