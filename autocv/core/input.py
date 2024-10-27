"""This module defines the Input class, which include methods for simulating mouse and keyboard input.

It includes methods for moving the mouse cursor, clicking, sending keystrokes, and more, allowing for interaction with
the application under control is a way that mimics human input.
"""

from __future__ import annotations

__all__ = ("Input",)

import logging
import math
import time
from random import randint, random
from typing import Final

import win32api
import win32con
import win32gui
from typing_extensions import Self

from .vision import Vision, check_valid_hwnd

logger = logging.getLogger(__name__)


class Input(Vision):
    """The Input class extends the Vision class by adding functionalities for simulating user input.

    This includes mouse movements, clicks, and keyboard key presses. It aims to provide a more human-like interaction
    with the application by incorporating randomness and delays in its actions.
    """

    SPEED: Final[int] = 16
    GRAVITY: Final[float] = 8 + random() / 2
    WIND: Final[float] = 4 + random() / 2

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes an Input object by setting up the internal state and inheriting from the Vision class.

        Args:
            hwnd (int): The window handle to which the input should be directed. If None, input will need to be
                directed manually later.
        """
        super().__init__(hwnd)
        self._last_moved_point: tuple[int, int] = 0, 0

    def get_last_moved_point(self: Self) -> tuple[int, int]:
        """Returns the last point to which the mouse cursor was moved using the move_mouse() method.

        Returns:
            tuple[int, int]: A tuple containing the (x, y) coordinates of the last point to which the mouse cursor was
                moved.
        """
        return self._last_moved_point

    @check_valid_hwnd
    def _get_topmost_hwnd(self: Self) -> int:
        """Gets topmost handle for the specified window.

        Returns:
            int: The handle for the specified window.
        """
        parent_hwnd: int = win32gui.GetAncestor(self.hwnd, win32con.GA_ROOT)
        logger.debug("Found parent handle: %s", parent_hwnd)
        return parent_hwnd

    @staticmethod
    def _make_lparam(x: int, y: int) -> int:
        """Makes a valid `lParam` value for `WM_NCHITTEST` and `WM_MOUSEMOVE` messages.

        Args:
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.

        Returns:
            int: The `lParam` value.
        """
        return (y << 16) | (x & 0xFFFF)

    @check_valid_hwnd
    def move_mouse(self: Self, x: int, y: int, *, human_like: bool = True, ghost_mouse: bool = True) -> None:
        """Moves the mouse cursor to the given (x, y) coordinates.

        This is done relative to the top-left corner of the client area of the target window. Determines which part of
        the target window the mouse is over, sets the mouse cursor to the appropriate shape, and then moves the mouse
        cursor to the target point.

        Args:
            x (int): The x-coordinate of the target point.
            y (int): The y-coordinate of the target point.
            human_like (bool): Whether to teleport the mouse or move it more human-like. Defaults to human-like.
            ghost_mouse (bool): Whether to move the ghost mouse or physical mouse. Defaults to ghost mouse.

        Returns:
            None
        """
        if not human_like:
            return self._move_mouse(x, y, ghost_mouse=ghost_mouse)

        speed = (random() * 15 + 30) / 10
        self._wind_mouse(
            *self._last_moved_point,
            x,
            y,
            9,
            3,
            5 / speed,
            10 / speed,
            10 * speed,
            8 * speed,
            ghost_mouse=ghost_mouse,
        )
        return None

    def _wind_mouse(  # noqa: PLR0913
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
        sqrt_3 = math.sqrt(3)
        sqrt_5 = math.sqrt(5)

        x = xs
        y = ys
        velo_x = 0.0
        velo_y = 0.0
        wind_x = 0.0
        wind_y = 0.0

        while True:
            countdown = time.perf_counter() + 15

            if time.perf_counter() > countdown:
                raise TimeoutError

            if random() < 0.5:
                wind_x = wind_x / sqrt_3 + (random() * (round(wind) * 2 + 1) - wind) / sqrt_5
                wind_y = wind_y / sqrt_3 + (random() * (round(wind) * 2 + 1) - wind) / sqrt_5

            traveled_distance = math.hypot(x - xs, y - ys)
            remaining_distance = math.hypot(x - xe, y - ye)

            if remaining_distance <= 1:
                break

            if remaining_distance < target_area:
                step = (remaining_distance / 2) + (random() * 6 - 3)
            elif traveled_distance < target_area:
                if traveled_distance < 3:
                    traveled_distance = 10 * random()
                step = traveled_distance * (1 + random() * 3)
            else:
                step = max_step

            step = min(step, max_step)
            if step < 3:
                step = 3 + (random() * 3)

            velo_x += wind_x
            velo_y += wind_y
            velo_x += gravity * (xe - x) / remaining_distance
            velo_y += gravity * (ye - y) / remaining_distance

            if math.hypot(velo_x, velo_y) > step:
                random_dist = step / 3.0 + (step / 2 * random())
                velo_mag = math.sqrt(velo_x * velo_x + velo_y * velo_y)
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
        """Moves the mouse cursor to the given (x, y) coordinates.

        This is done relative to the top-left corner of the client area of the target window. Determines which part of
        the target window the mouse is over, sets the mouse cursor to the appropriate shape, and then moves the mouse
        cursor to the target point.

        Args:
            x (int): An integer representing the x-coordinate of the target point.
            y (int): An integer representing the y-coordinate of the target point.
            ghost_mouse (bool): Whether to move the ghost mouse or physical mouse. Defaults to ghost mouse.

        Returns:
            None
        """
        # Convert the target point from client coordinates to screen coordinates.
        screen_point = win32gui.ClientToScreen(self.hwnd, (x, y))

        if ghost_mouse:
            # Determine which part of the target window the mouse is over.
            result = win32gui.SendMessage(self.hwnd, win32con.WM_NCHITTEST, 0, self._make_lparam(x, y))

            # Set the mouse cursor to the appropriate shape.
            win32api.SendMessage(
                self.hwnd,
                win32con.WM_SETCURSOR,
                self.hwnd,  # type: ignore[arg-type]
                self._make_lparam(result, win32con.WM_MOUSEMOVE),  # type: ignore[arg-type]
            )

            # Move the mouse cursor to the target point.
            win32api.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, self._make_lparam(*screen_point))
        else:
            win32api.SetCursorPos(screen_point)

        self._last_moved_point = x, y

    @check_valid_hwnd
    def click_mouse(self: Self, button: int = 1, *, send_message: bool = False) -> None:
        """Clicks the mouse button at the last moved point.

        Args:
            button (int): An integer representing the mouse button to press. 1 for left button, 2 for right button, 3
                for middle button. Default is 1.
            send_message (bool): by default, click_mouse will use Windows' PostMessage function to click. In some apps,
                SendMessage is needed instead. In these cases you set send_message=True.

        Returns:
            None
        """
        # Convert the last moved point from client coordinates to screen coordinates.
        screen_point = win32gui.ClientToScreen(self.hwnd, self._last_moved_point)

        # Determine which button to press.
        button_messages = {
            1: win32con.WM_LBUTTONDOWN,
            2: win32con.WM_RBUTTONDOWN,
            3: win32con.WM_MBUTTONDOWN,
        }
        button_to_press = button_messages.get(button, win32con.WM_LBUTTONDOWN)

        # Determine which part of the target window the mouse is over.
        screen_lparam = self._make_lparam(*screen_point)
        result = win32gui.SendMessage(self.hwnd, win32con.WM_NCHITTEST, 0, screen_lparam)

        # Notify the parent window of the button press.
        client_lparam = self._make_lparam(*self._last_moved_point)
        win32gui.SendMessage(
            win32gui.GetParent(self.hwnd),
            win32con.WM_PARENTNOTIFY,
            button_to_press,
            client_lparam,
        )

        # Set the mouse cursor to the appropriate shape.
        top_hwnd = self._get_topmost_hwnd()
        lparam_button = self._make_lparam(result, button_to_press)
        win32gui.SendMessage(top_hwnd, win32con.WM_MOUSEACTIVATE, top_hwnd, lparam_button)
        win32api.SendMessage(self.hwnd, win32con.WM_SETCURSOR, self.hwnd, lparam_button)  # type: ignore[arg-type]

        if send_message:
            last_moved_point_lparam = win32api.MAKELONG(*self._last_moved_point)  # type: ignore[no-untyped-call]
            # Simulate the button press at the target point.
            win32gui.SendMessage(self.hwnd, button_to_press, button, last_moved_point_lparam)
            time.sleep(randint(10, 50) / 1_000)

            # Release the button at the target point.
            win32gui.SendMessage(self.hwnd, button_to_press + 1, 0, last_moved_point_lparam)
        else:
            # Simulate the button press at the target point.
            win32gui.PostMessage(self.hwnd, button_to_press, button, screen_lparam)
            time.sleep(randint(10, 50) / 1_000)

            # Release the button at the target point.
            win32gui.PostMessage(self.hwnd, button_to_press + 1, 0, screen_lparam)

    @check_valid_hwnd
    def press_vk_key(self: Self, vk_code: int) -> None:
        """Simulates the pressing of a virtual key code in the active window.

        Args:
            vk_code (int): The virtual key code to simulate the press of. This value can be obtained from the Microsoft
                website.

        Returns:
            None
        """
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]
        # Create the lparam value for the key down message
        l_param = (scan_code << 16) | 1

        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_CHAR, chr(vk_code), l_param)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def release_vk_key(self: Self, vk_code: int) -> None:
        """Simulates the releasing of a virtual key code in the active window.

        Args:
            vk_code (int): The virtual key code to simulate the release of. This value can be obtained from the
                Microsoft website.

        Returns:
            None
        """
        # Set the transition state bit in the lparam value for the key up message
        scan_code = win32api.MapVirtualKey(vk_code, 0)  # type: ignore[no-untyped-call]

        # Create the lparam value for the key down message
        l_param = (scan_code << 16) | 1
        l_param |= 0xC0000000

        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk_code, l_param)  # type: ignore[arg-type]
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]

    @check_valid_hwnd
    def send_vk_key(self: Self, vk_code: int) -> None:
        """Sends a virtual key code to the active window.

        Args:
            vk_code (int): The virtual key code to send. This value can be obtained from the Microsoft website.

        Returns:
            None
        """
        self.press_vk_key(vk_code)
        time.sleep(randint(3, 5) / 1_000)
        self.release_vk_key(vk_code)

    @staticmethod
    def get_async_key_state(vk_code: int) -> bool:
        """Retrieves the status of the specified key.

        Args:
            vk_code (int): Specifies one of 256 possible virtual-key codes.

        Returns:
            bool: Specifies whether the key was pressed since the last call to GetAsyncKeyState.
        """
        return bool(win32api.GetAsyncKeyState(vk_code))  # type: ignore[no-untyped-call]

    @check_valid_hwnd
    def send_keys(self: Self, characters: str) -> None:
        """Sends a series of keyboard input to the active window.

        Args:
            characters (str): The characters to send.

        Returns:
            None
        """
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 1, self.hwnd)  # type: ignore[arg-type]
        for c in characters:
            vk = win32api.VkKeyScan(c)
            scan_code = win32api.MapVirtualKey(ord(c.upper()), 0)  # type: ignore[no-untyped-call]

            # Create the lparam value for the key down message
            l_param = (scan_code << 16) | 1

            win32api.SendMessage(self.hwnd, win32con.WM_KEYDOWN, vk, l_param)
            win32api.SendMessage(self.hwnd, win32con.WM_CHAR, ord(c), l_param)  # type: ignore[arg-type]
            time.sleep(randint(3, 5) / 1_000)

            # Set the transition state bit in the lparam value for the key up message
            l_param |= 0xC0000000
            win32api.SendMessage(self.hwnd, win32con.WM_KEYUP, vk, l_param)
            time.sleep(randint(20, 60) / 1_000)
        win32api.SendMessage(self.hwnd, win32con.WM_ACTIVATE, 0, self.hwnd)  # type: ignore[arg-type]
