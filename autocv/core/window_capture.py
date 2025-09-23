"""Window enumeration and handle management helpers.

Exposes the :class:`WindowCapture` base that other AutoCV components use to
locate windows, capture their surfaces, and walk child handles.
"""

from __future__ import annotations

__all__ = ("WindowCapture",)

from collections.abc import Callable

import win32gui
from typing_extensions import Self

from autocv.utils import filtering

WindowCollector = Callable[[int, list[tuple[int, str]]], bool]


def _enum_windows(collector: WindowCollector) -> list[tuple[int, str]]:
    """Run a Win32 enumeration callback and collect results."""
    windows: list[tuple[int, str]] = []

    def handler(hwnd: int, _extra: object) -> bool:
        return collector(hwnd, windows)

    win32gui.EnumWindows(handler, None)
    return windows


def _enum_children(parent: int, collector: WindowCollector) -> list[tuple[int, str]]:
    """Enumerate child windows for ``parent``."""
    children: list[tuple[int, str]] = []

    def handler(hwnd: int, _extra: object) -> bool:
        return collector(hwnd, children)

    win32gui.EnumChildWindows(parent, handler, None)
    return children


class WindowCapture:
    """Capture metadata and child handles for a Windows GUI window."""

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initialise the wrapper with an optional window handle."""
        self.hwnd = hwnd

    @staticmethod
    def get_windows_with_hwnds() -> list[tuple[int, str]]:
        """Return all visible windows exposed by ``EnumWindows``."""

        def collect(hwnd: int, acc: list[tuple[int, str]]) -> bool:
            title = win32gui.GetWindowText(hwnd)
            if win32gui.IsWindowVisible(hwnd) and title:
                acc.append((hwnd, title))
            return True

        return _enum_windows(collect)

    def _find_window(self: Self, matcher: Callable[[str], bool]) -> int | None:
        """Return the first window handle whose title satisfies ``matcher``."""
        windows = self.get_windows_with_hwnds()
        found = filtering.find_first(lambda item: matcher(item[1]), windows)
        return found[0] if found else None

    def get_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> int | None:
        """Find the first window whose title contains ``title``."""
        if case_insensitive:
            lowered = title.casefold()
            return self._find_window(lambda candidate: lowered in candidate.casefold())
        return self._find_window(lambda candidate: title in candidate)

    def get_child_windows(self: Self) -> list[tuple[int, str]]:
        """Return child windows for the current ``hwnd``."""

        def collect(hwnd: int, acc: list[tuple[int, str]]) -> bool:
            class_name = win32gui.GetClassName(hwnd)
            if class_name:
                acc.append((hwnd, class_name))
            return True

        return _enum_children(self.hwnd, collect)

    def _find_child_by_class(self: Self, matcher: Callable[[str], bool]) -> int | None:
        """Return the first child window handle whose class satisfies ``matcher``."""
        children = self.get_child_windows()
        found = filtering.find_first(lambda item: matcher(item[1]), children)
        return found[0] if found else None

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Update ``hwnd`` when a matching window title is discovered."""
        found = self.get_hwnd_by_title(title, case_insensitive=case_insensitive)
        if found is not None:
            self.hwnd = found
            return True
        return False

    def set_inner_hwnd_by_title(self: Self, class_name: str) -> bool:
        """Update ``hwnd`` to the first child window whose class matches."""
        lowered = class_name.casefold()
        found = self._find_child_by_class(lambda candidate: lowered in candidate.casefold())
        if found is not None:
            self.hwnd = found
            return True
        return False
