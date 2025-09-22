"""Window enumeration and handle management helpers.

Exposes the :class:`WindowCapture` base that other AutoCV components use to
locate windows, capture their surfaces, and walk child handles.
"""

from __future__ import annotations

__all__ = ("WindowCapture",)

import win32gui
from typing_extensions import Self

from autocv.utils import filtering


class WindowCapture:
    """Capture metadata and child handles for a Windows GUI window.

    Attributes:
        hwnd (int): Handle for the window currently targeted by the instance.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initialise the :class:`WindowCapture` wrapper.

        Args:
            hwnd (int): Window handle to operate on. Defaults to ``-1`` until
                a valid handle is set.
        """
        self.hwnd = hwnd

    @staticmethod
    def _window_enumeration_handler(hwnd: int, top_windows: list[tuple[int, str]]) -> None:
        """Collect visible top-level windows during enumeration.

        Args:
            hwnd (int): Handle of the window yielded by ``EnumWindows``.
            top_windows (list[tuple[int, str]]): Accumulator receiving the
                ``(handle, title)`` tuples for visible windows.
        """
        title = win32gui.GetWindowText(hwnd)
        if win32gui.IsWindowVisible(hwnd) and title:
            top_windows.append((hwnd, title))

    @staticmethod
    def _child_window_enumeration_handler(hwnd: int, child_windows: list[tuple[int, str]]) -> None:
        """Collect visible child windows during enumeration.

        Args:
            hwnd (int): Handle of the child window yielded by ``EnumChildWindows``.
            child_windows (list[tuple[int, str]]): Accumulator receiving
                ``(handle, class_name)`` tuples for child windows.
        """
        class_name = win32gui.GetClassName(hwnd)
        if class_name:
            child_windows.append((hwnd, class_name))

    def get_windows_with_hwnds(self: Self) -> list[tuple[int, str]]:
        """Return all visible windows exposed by ``EnumWindows``.

        Returns:
            list[tuple[int, str]]: ``(handle, title)`` pairs for visible windows.
        """
        top_windows: list[tuple[int, str]] = []
        win32gui.EnumWindows(self._window_enumeration_handler, top_windows)
        return top_windows

    def get_hwnd_by_title(self: Self, title: str) -> int | None:
        """Find the first window whose title contains ``title``.

        Args:
            title (str): Substring to match against window titles.

        Returns:
            int | None: Matching window handle or ``None`` if no window matches.
        """
        top_windows = self.get_windows_with_hwnds()
        title_lower = title.casefold()
        found = filtering.find_first(lambda x: title_lower in x[1].casefold(), top_windows)
        return found[0] if found else None

    def get_child_windows(self: Self) -> list[tuple[int, str]]:
        """Return child windows for the current ``hwnd``.

        Returns:
            list[tuple[int, str]]: ``(handle, class_name)`` pairs for child windows.
        """
        child_windows: list[tuple[int, str]] = []
        win32gui.EnumChildWindows(self.hwnd, self._child_window_enumeration_handler, child_windows)
        return child_windows

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Update ``hwnd`` when a matching window title is discovered.

        Args:
            title (str): Substring searched for within window titles.
            case_insensitive (bool): If ``True``, perform a case-insensitive
                comparison. Defaults to ``False``.

        Returns:
            bool: ``True`` when a matching window is found, otherwise ``False``.
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
        """Update ``hwnd`` to the first child window whose class matches.

        Args:
            class_name (str): Substring searched for within child window class names.

        Returns:
            bool: ``True`` when a matching child window is found, otherwise ``False``.
        """
        child_windows = self.get_child_windows()
        class_name_lower = class_name.casefold()
        found = filtering.find_first(lambda x: class_name_lower in x[1].casefold(), child_windows)
        if found:
            self.hwnd = found[0]
            return True
        return False
