"""This module defines the WindowCapture class which is used for capturing images from specified windows on the desktop.

It includes methods for obtaining window handles and titles, capturing specific windows, and listing child windows.
The class interacts with the Windows API to manage window handles and capture screens.
"""

from __future__ import annotations

__all__ = ("WindowCapture",)


from typing import TYPE_CHECKING

import win32gui
from typing_extensions import Self

from autocv.utils import filtering

if TYPE_CHECKING:
    from collections.abc import Sequence


class WindowCapture:
    """The WindowCapture class is used to capture images of windows on the desktop.

    It provides methods for getting information about windows, setting the current window to a new window, and getting a
    list of child windows for the current window. The class can be initialized with a handle to a specific window to
    capture or left unspecified to capture the entire screen.
    """

    def __init__(self: Self, hwnd: int | None = None) -> None:
        """Initialize the WindowCapture object.

        Args:
        ----
            hwnd: The handle to the window to capture. If None, it will need to be set later.
        """
        self.hwnd = hwnd or -1

    @staticmethod
    def _window_enumeration_handler(hwnd: int, top_windows: list[tuple[int, str]]) -> None:
        """Adds window title and ID to the array.

        Args:
        ----
            hwnd (int): The window handle.
            top_windows (List[Tuple[int, str]]): The list of top windows with window handle and title.
        """
        title = win32gui.GetWindowText(hwnd)
        if win32gui.IsWindowVisible(hwnd) and title:
            top_windows.append((hwnd, title))

    @staticmethod
    def _child_window_enumeration_handler(hwnd: int, child_windows: list[tuple[int, str]]) -> None:
        """Adds window class name and ID to the array.

        Args:
        ----
            hwnd (int): The window handle.
            child_windows (List[Tuple[int, str]]): The list of child windows with window handle and class name.
        """
        class_name = win32gui.GetClassName(hwnd)
        if class_name:
            child_windows.append((hwnd, class_name))

    def get_windows_with_hwnds(self: Self) -> Sequence[tuple[int, str]]:
        """Returns a list of all visible windows with their corresponding IDs.

        Returns:
        -------
            List[Tuple[int, str]]: A list of tuples containing the window handle (HWND) and the window title (str).
        """
        top_windows: list[tuple[int, str]] = []
        win32gui.EnumWindows(self._window_enumeration_handler, top_windows)
        return top_windows

    def get_hwnd_by_title(self: Self, title: str) -> int | None:
        """Returns the window ID of the first visible window whose title contains the specified string.

        Args:
        ----
            title (str): A string to search for in the window titles.

        Returns:
        -------
            Optional[int]: The window ID of the first visible window whose title contains the specified string,
                         or None if no matching window is found.
        """
        top_windows = self.get_windows_with_hwnds()
        title = title.casefold()
        first = filtering.find_first(lambda x: title in x[1].casefold(), top_windows)
        return first[0] if first else None

    def get_child_windows(self: Self) -> Sequence[tuple[int, str]]:
        """Returns a list of all child windows for the current window.

        Returns:
        -------
            List[Tuple[int, str]]: A list of tuples, each containing a child window's ID and class name.
        """
        child_windows: list[tuple[int, str]] = []
        win32gui.EnumChildWindows(self.hwnd, self._child_window_enumeration_handler, child_windows)
        return child_windows

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Sets the current window to the first visible window whose title contains the specified string.

        Args:
        ----
            title (str): A string to search for in the window titles.
            case_insensitive (bool): Whether to use case sensitive when searching.

        Returns:
        -------
            bool: True if a matching window is found and the current window is set to it, False otherwise.
        """
        top_windows = self.get_windows_with_hwnds()

        if case_insensitive:
            title = title.casefold()

        first = filtering.find_first(
            lambda x: title in x[1].casefold() if case_insensitive else title in x[1],
            top_windows,
        )
        if first:
            self.hwnd = first[0]
            return True
        return False

    def set_inner_hwnd_by_title(self: Self, class_name: str) -> bool:
        """Sets the current window to the first child window whose class name contains the specified string.

        Args:
        ----
            class_name (str): A string representing the class name to search for in child windows.

        Returns:
        -------
            bool: True if a child window was found and the current window was set to it. False otherwise.
        """
        child_windows: list[tuple[int, str]] = []
        win32gui.EnumChildWindows(self.hwnd, self._child_window_enumeration_handler, child_windows)

        class_name = class_name.casefold()
        first = filtering.find_first(lambda x: class_name in x[1].casefold(), child_windows)
        if first:
            self.hwnd = first[0]
            return True
        return False
