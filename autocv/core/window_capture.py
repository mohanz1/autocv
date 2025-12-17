"""Window enumeration and capture helpers.

The :class:`WindowCapture` base locates windows, walks child handles, and
captures window contents into NumPy arrays for downstream vision routines.
"""

from __future__ import annotations

__all__ = ("WindowCapture",)

from typing import TYPE_CHECKING, Final, TypeAlias

import numpy as np
import numpy.typing as npt
import win32con
import win32gui
import win32ui
from typing_extensions import Self

from autocv.models import InvalidHandleError
from autocv.utils import filtering

if TYPE_CHECKING:
    from collections.abc import Callable

WindowHandle: TypeAlias = int
WindowEntry: TypeAlias = tuple[WindowHandle, str]
ChildWindowEntry: TypeAlias = WindowEntry
Bounds: TypeAlias = tuple[int, int, int, int]  # left, top, right, bottom
Rect: TypeAlias = tuple[int, int, int, int]  # x, y, width, height
Size: TypeAlias = tuple[int, int]
NDArrayUint8: TypeAlias = npt.NDArray[np.uint8]

_UNSET_HANDLE: Final[int] = -1


class WindowCapture:
    """Base class for discovering Win32 windows and capturing their contents.

    Responsibilities:
        - manage window handles and cached window bounds,
        - enumerate top-level or child windows,
        - provide rectangle helpers for window-relative coordinates,
        - capture window regions into numpy arrays for downstream processing.
    """

    __slots__ = ("_cached_bounds", "_last_frame", "hwnd")

    def __init__(self: Self, hwnd: WindowHandle = _UNSET_HANDLE) -> None:
        """Initialise the window capture base.

        Args:
            hwnd: Window handle to operate on. Defaults to ``-1`` until a
                valid handle is set.
        """
        self.hwnd: WindowHandle = hwnd
        self._cached_bounds: Bounds | None = None
        self._last_frame: NDArrayUint8 | None = None

    #
    # Handle and bounds helpers
    #
    @property
    def is_attached(self: Self) -> bool:
        """Return whether a handle has been configured."""
        return self.hwnd != _UNSET_HANDLE

    @property
    def last_frame(self: Self) -> NDArrayUint8 | None:
        """Return the latest captured frame cached by ``capture_frame`` when ``persist`` is True."""
        return self._last_frame

    def attach(self: Self, hwnd: WindowHandle) -> None:
        """Attach to a new window handle and clear cached state."""
        self.hwnd = hwnd
        self.invalidate_cache()

    def detach(self: Self) -> None:
        """Detach from any window handle and clear cached state."""
        self.hwnd = _UNSET_HANDLE
        self.invalidate_cache()

    def invalidate_cache(self: Self) -> None:
        """Reset cached bounds and the last captured frame."""
        self._cached_bounds = None
        self._last_frame = None

    def get_window_bounds(self: Self, *, use_cache: bool = False) -> Bounds:
        """Return the window bounds.

        Args:
            use_cache: When True, reuse cached bounds if available.

        Returns:
            Bounds: Tuple of ``(left, top, right, bottom)``.
        """
        hwnd = self._ensure_hwnd()
        if not use_cache or self._cached_bounds is None:
            self._cached_bounds = self._fetch_window_bounds(hwnd)
        return self._cached_bounds

    def get_window_size(self: Self, *, use_cache: bool = False) -> Size:
        """Return the window size.

        Args:
            use_cache: When True, reuse cached bounds if available.

        Returns:
            Size: Tuple of ``(width, height)``.
        """
        return self._bounds_to_size(self.get_window_bounds(use_cache=use_cache))

    @staticmethod
    def bounds_to_rect(bounds: Bounds) -> Rect:
        """Convert absolute bounds into a window-relative rectangle.

        Args:
            bounds: Absolute screen-space bounds.

        Returns:
            Rect: Rectangle expressed as ``(x, y, width, height)``.
        """
        width, height = WindowCapture._bounds_to_size(bounds)
        return 0, 0, width, height

    @staticmethod
    def _bounds_to_size(bounds: Bounds) -> Size:
        """Convert absolute bounds into a ``(width, height)`` tuple."""
        left, top, right, bottom = bounds
        return right - left, bottom - top

    #
    # Window enumeration
    #
    def get_windows_with_hwnds(self: Self) -> list[WindowEntry]:
        """Return all visible windows exposed by ``EnumWindows``."""
        return self._run_enum_windows(self._window_enumeration_handler)

    def get_hwnd_by_title(self: Self, title: str) -> WindowHandle | None:
        """Find the first window whose title contains the provided text.

        Args:
            title: Title substring to search for (case-insensitive).

        Returns:
            WindowHandle | None: Matching handle or ``None``.
        """
        return self._find_by_title(title, self.get_windows_with_hwnds(), case_insensitive=True)

    def get_child_windows(self: Self, hwnd: WindowHandle | None = None) -> list[ChildWindowEntry]:
        """Return child windows for the provided or current handle.

        Args:
            hwnd: Optional target handle; defaults to the current attachment.

        Returns:
            list[ChildWindowEntry]: Child handles and class names.
        """
        target_hwnd = hwnd if hwnd is not None else self._ensure_hwnd()
        child_windows: list[ChildWindowEntry] = []
        win32gui.EnumChildWindows(target_hwnd, self._child_window_enumeration_handler, child_windows)
        return child_windows

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Update ``hwnd`` when a matching window title is discovered.

        Args:
            title: Title substring to search for.
            case_insensitive: When True, perform case-insensitive matching.

        Returns:
            bool: True when the handle was updated.
        """
        windows = self.get_windows_with_hwnds()
        found = self._find_by_title(title, windows, case_insensitive=case_insensitive)
        if found is not None:
            self.attach(found)
            return True
        return False

    def set_inner_hwnd_by_title(self: Self, class_name: str) -> bool:
        """Update ``hwnd`` to the first child window whose class matches.

        Args:
            class_name: Substring of the child window class to match.

        Returns:
            bool: True when the handle was updated.
        """
        child_windows = self.get_child_windows()
        class_name_lower = class_name.casefold()
        found = filtering.find_first(lambda x: class_name_lower in x[1].casefold(), child_windows)
        if found:
            self.attach(found[0])
            return True
        return False

    #
    # Capture pipeline
    #
    def capture_frame(self: Self, region: Rect | None = None, *, persist: bool = True) -> NDArrayUint8:
        """Capture the current window region into a contiguous BGR image.

        Args:
            region (tuple[int, int, int, int] | None): Optional capture region specified
                as ``(x, y, width, height)`` in window coordinates. Defaults to the full
                window when omitted.
            persist (bool): When ``True``, store the frame in ``last_frame`` for reuse.

        Returns:
            NDArrayUint8: Captured frame in BGR channel order.

        Raises:
            ValueError: If ``region`` falls completely outside the window bounds.
        """
        hwnd = self._ensure_hwnd()
        bounds = self.get_window_bounds()
        capture_rect = self._normalize_region(region, bounds)

        frame = self._bitblt_to_array(hwnd, capture_rect)
        if persist:
            self._last_frame = frame
        return frame

    #
    # Internal helpers
    #
    def _ensure_hwnd(self: Self) -> WindowHandle:
        """Return the configured handle or raise ``InvalidHandleError``."""
        if self.hwnd == _UNSET_HANDLE:
            raise InvalidHandleError(self.hwnd)
        return self.hwnd

    @staticmethod
    def _fetch_window_bounds(hwnd: WindowHandle) -> Bounds:
        """Fetch absolute window bounds from the OS."""
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return left, top, right, bottom

    @staticmethod
    def _find_by_title(title: str, windows: list[WindowEntry], *, case_insensitive: bool) -> WindowHandle | None:
        """Return a window handle when the title matches the supplied substring."""
        matcher: Callable[[WindowEntry], bool]
        if case_insensitive:
            lowered_title = title.casefold()

            def matcher(entry: WindowEntry) -> bool:
                return lowered_title in entry[1].casefold()
        else:

            def matcher(entry: WindowEntry) -> bool:
                return title in entry[1]

        found = filtering.find_first(matcher, windows)
        return found[0] if found else None

    @staticmethod
    def _run_enum_windows(handler: Callable[[WindowHandle, list[WindowEntry]], None]) -> list[WindowEntry]:
        """Run a Win32 enumeration callback and collect results."""
        windows: list[WindowEntry] = []
        win32gui.EnumWindows(handler, windows)
        return windows

    @staticmethod
    def _normalize_region(region: Rect | None, bounds: Bounds) -> Rect:
        """Clamp the requested region to the window bounds.

        Args:
            region: Desired capture rectangle relative to the window.
            bounds: Absolute window bounds.

        Returns:
            Rect: Region clamped to the window.

        Raises:
            ValueError: If the region lies completely outside the window.
        """
        if region is None:
            width, height = WindowCapture._bounds_to_size(bounds)
            return 0, 0, width, height

        x, y, width, height = region
        window_width, window_height = WindowCapture._bounds_to_size(bounds)

        max_width = max(window_width - x, 0)
        max_height = max(window_height - y, 0)
        clamped_width = min(width, max_width)
        clamped_height = min(height, max_height)

        if clamped_width <= 0 or clamped_height <= 0:
            msg = "Capture region lies outside the window bounds."
            raise ValueError(msg)

        return x, y, clamped_width, clamped_height

    @staticmethod
    def _bitblt_to_array(hwnd: WindowHandle, rect: Rect) -> NDArrayUint8:
        """Perform a BitBlt capture of the given window region into a numpy array.

        The capture is performed directly into a compatible bitmap using only the
        requested region to avoid extra cropping or copies.

        Args:
            hwnd: Window handle to capture from.
            rect: Region within the window specified as ``(x, y, width, height)``.

        Raises:
            InvalidHandleError: If the window device context cannot be acquired.
        """
        x, y, width, height = rect

        window_dc = win32gui.GetWindowDC(hwnd)
        if window_dc == 0:
            raise InvalidHandleError(hwnd)
        mem_dc = win32ui.CreateDCFromHandle(window_dc)
        bmp_dc = mem_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()

        try:
            bitmap.CreateCompatibleBitmap(mem_dc, width, height)
            bmp_dc.SelectObject(bitmap)
            bmp_dc.BitBlt((0, 0), (width, height), mem_dc, (x, y), win32con.SRCCOPY)

            # Convert raw bitmap data into a contiguous BGR array.
            signed_ints_array = bitmap.GetBitmapBits(True)
            frame = np.frombuffer(signed_ints_array, dtype=np.uint8).reshape((height, width, 4))
            return np.ascontiguousarray(frame[..., :3])
        finally:
            # Ensure GDI resources are always released.
            bmp_dc.DeleteDC()
            mem_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, window_dc)
            win32gui.DeleteObject(bitmap.GetHandle())

    @staticmethod
    def _window_enumeration_handler(hwnd: WindowHandle, top_windows: list[WindowEntry]) -> None:
        """Collect visible top-level windows during enumeration."""
        title = win32gui.GetWindowText(hwnd)
        if win32gui.IsWindowVisible(hwnd) and title:
            top_windows.append((hwnd, title))

    @staticmethod
    def _child_window_enumeration_handler(hwnd: WindowHandle, child_windows: list[ChildWindowEntry]) -> None:
        """Collect child windows encountered during enumeration."""
        class_name = win32gui.GetClassName(hwnd)
        if class_name:
            child_windows.append((hwnd, class_name))
