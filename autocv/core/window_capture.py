"""This module defines the WindowCapture class for capturing images from windows on the desktop.

It provides methods for enumerating windows, capturing screenshots, and managing child windows.
"""

from __future__ import annotations

__all__ = ("WindowCapture",)

import win32gui
from typing_extensions import Self

from autocv.utils import filtering


class WindowCapture:
    """A class for capturing images of windows on the desktop.

    This class provides methods to obtain information about windows, capture images from specific windows,
    and list child windows for the current window. It interacts with the Windows API to manage window handles
    and capture screens.

    Attributes:
        hwnd (int): The handle of the current window. Defaults to -1 if not set.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes the WindowCapture instance.

        Args:
            hwnd: The handle to the window to capture. If not provided, the instance will be initialized with -1.
        """
        self.hwnd = hwnd

    @staticmethod
    def _window_enumeration_handler(hwnd: int, top_windows: list[tuple[int, str]]) -> None:
        """Appends the window handle and title of a visible window to the provided list.

        Args:
            hwnd: The window handle.
            top_windows: A list to which the tuple (hwnd, title) is appended for each visible window.
        """
        title = win32gui.GetWindowText(hwnd)
        if win32gui.IsWindowVisible(hwnd) and title:
            top_windows.append((hwnd, title))

    @staticmethod
    def _child_window_enumeration_handler(hwnd: int, child_windows: list[tuple[int, str]]) -> None:
        """Appends the child window handle and class name to the provided list.

        Args:
            hwnd: The window handle.
            child_windows: A list to which the tuple (hwnd, class_name) is appended for each child window.
        """
        class_name = win32gui.GetClassName(hwnd)
        if class_name:
            child_windows.append((hwnd, class_name))

    def get_windows_with_hwnds(self: Self) -> list[tuple[int, str]]:
        """Retrieves all visible windows with their handles and titles.

        Returns:
            A list of tuples, where each tuple contains a window handle and its corresponding title.
        """
        top_windows: list[tuple[int, str]] = []
        win32gui.EnumWindows(self._window_enumeration_handler, top_windows)
        return top_windows

    def get_hwnd_by_title(self: Self, title: str) -> int | None:
        """Finds the first visible window whose title contains the specified string.

        Args:
            title: A substring to search for within window titles.

        Returns:
            The window handle of the first matching window, or None if no match is found.
        """
        top_windows = self.get_windows_with_hwnds()
        title_lower = title.casefold()
        found = filtering.find_first(lambda x: title_lower in x[1].casefold(), top_windows)
        return found[0] if found else None

    def get_child_windows(self: Self) -> list[tuple[int, str]]:
        """Retrieves all child windows for the current window.

        Returns:
            A list of tuples, where each tuple contains a child window handle and its class name.
        """
        child_windows: list[tuple[int, str]] = []
        win32gui.EnumChildWindows(self.hwnd, self._child_window_enumeration_handler, child_windows)
        return child_windows

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Sets the current window handle based on a window title substring.

        Args:
            title: A substring to search for within window titles.
            case_insensitive: If True, the search will be case-insensitive. Defaults to False.

        Returns:
            True if a matching window was found and the current window handle was updated; False otherwise.
        """
        top_windows = self.get_windows_with_hwnds()
        if case_insensitive:
            title_lower = title.casefold()
            found = filtering.find_first(lambda x: title_lower in x[1].casefold(), top_windows)
        else:
            found = filtering.find_first(lambda x: title in x[1], top_windows)
        if found:
            self.hwnd = found[0]
            return True
        return False

    def set_inner_hwnd_by_title(self: Self, class_name: str) -> bool:
        """Sets the current window handle based on a child window's class name substring.

        Args:
            class_name: A substring to search for within child window class names.

        Returns:
            True if a matching child window was found and the current window handle was updated; False otherwise.
        """
        child_windows = self.get_child_windows()
        class_name_lower = class_name.casefold()
        found = filtering.find_first(lambda x: class_name_lower in x[1].casefold(), child_windows)
        if found:
            self.hwnd = found[0]
            return True
        return False
