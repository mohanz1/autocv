"""Win32 window enumeration and capture helpers.

This module exposes :class:`~autocv.core.window_capture.WindowCapture`, a small building block for the AutoCV runtime
that can:

- enumerate visible top-level windows and their titles,
- walk child windows for a given handle and return their class names,
- capture a full window or window-relative sub-rectangle as a contiguous NumPy array.

All captured frames are returned in BGR channel order (OpenCV compatible) with ``dtype=uint8``.
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

from .decorators import check_valid_hwnd

if TYPE_CHECKING:
    from collections.abc import Sequence

WindowHandle: TypeAlias = int
WindowEntry: TypeAlias = tuple[WindowHandle, str]
ChildWindowEntry: TypeAlias = WindowEntry
Bounds: TypeAlias = tuple[int, int, int, int]  # left, top, right, bottom
Rect: TypeAlias = tuple[int, int, int, int]  # x, y, width, height
Size: TypeAlias = tuple[int, int]
NDArrayUint8: TypeAlias = npt.NDArray[np.uint8]

_UNSET_HANDLE: Final[WindowHandle] = -1
_BGRA_CHANNELS: Final[int] = 4
_BGR_CHANNELS: Final[int] = 3


class WindowCapture:
    """Base class for discovering Win32 windows and capturing their contents.

    This class stores a target Win32 handle (``hwnd``) and offers helper methods for:

    - enumerating visible windows (top-level) and child windows,
    - resolving handles by title/class substring,
    - capturing window pixels via a GDI BitBlt into a BGR ``numpy.ndarray``.
    """

    __slots__ = ("_cached_bounds", "_last_frame", "hwnd")

    def __init__(self: Self, hwnd: WindowHandle = _UNSET_HANDLE) -> None:
        """Initialise the window capture helper.

        Args:
            hwnd: Window handle to operate on. Defaults to ``-1`` until a valid handle is set.
        """
        self.hwnd: WindowHandle = hwnd
        self._cached_bounds: Bounds | None = None
        self._last_frame: NDArrayUint8 | None = None

    #
    # Handle and bounds helpers
    #
    @property
    def is_attached(self: Self) -> bool:
        """Return whether a handle value has been configured.

        This property only checks the sentinel ``-1`` value; it does not validate the handle with the OS.
        """
        return self.hwnd != _UNSET_HANDLE

    @property
    def last_frame(self: Self) -> NDArrayUint8 | None:
        """Return the latest frame cached by ``capture_frame`` when ``persist`` is ``True``."""
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

    def _get_window_bounds_cached(self: Self, *, use_cache: bool) -> Bounds:
        """Return window bounds, optionally reusing cached values.

        This helper assumes ``hwnd`` has already been validated (e.g. via
        :func:`~autocv.core.decorators.check_valid_hwnd`) and exists to keep the public API thin while avoiding
        duplicated caching logic.
        """
        if use_cache and self._cached_bounds is not None:
            return self._cached_bounds

        self._cached_bounds = self._fetch_window_bounds(self.hwnd)
        return self._cached_bounds

    @check_valid_hwnd
    def get_window_bounds(self: Self, *, use_cache: bool = False) -> Bounds:
        """Return the window bounds.

        Args:
            use_cache: When ``True``, reuse cached bounds if available.

        Returns:
            Bounds: Tuple of ``(left, top, right, bottom)``.

        Raises:
            InvalidHandleError: If no window handle is attached.
        """
        return self._get_window_bounds_cached(use_cache=use_cache)

    def get_window_size(self: Self, *, use_cache: bool = False) -> Size:
        """Return the window size.

        Args:
            use_cache: When ``True``, reuse cached bounds if available.

        Returns:
            Size: Tuple of ``(width, height)``.

        Raises:
            InvalidHandleError: If no window handle is attached.
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
        """Return all visible top-level windows with non-empty titles."""
        windows: list[WindowEntry] = []
        win32gui.EnumWindows(self._window_enumeration_handler, windows)
        return windows

    def get_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = True) -> WindowHandle | None:
        """Find the first window whose title contains the provided text.

        Args:
            title: Title substring to search for.
            case_insensitive: When ``True``, perform case-insensitive matching.

        Returns:
            WindowHandle | None: Matching handle or ``None``.
        """
        return self._find_by_title(title, self.get_windows_with_hwnds(), case_insensitive=case_insensitive)

    @check_valid_hwnd
    def get_child_windows(self: Self, hwnd: WindowHandle | None = None) -> list[ChildWindowEntry]:
        """Return child windows for the provided or current handle.

        Args:
            hwnd: Optional target handle; defaults to the current attachment.

        Returns:
            list[ChildWindowEntry]: Child handles and class names.

        Raises:
            InvalidHandleError: If the instance is not attached to a valid handle (validated by the decorator).
        """
        target_hwnd = hwnd or self.hwnd
        child_windows: list[ChildWindowEntry] = []
        win32gui.EnumChildWindows(target_hwnd, self._child_window_enumeration_handler, child_windows)
        return child_windows

    def set_hwnd_by_title(self: Self, title: str, *, case_insensitive: bool = False) -> bool:
        """Update ``hwnd`` when a matching window title is discovered.

        Args:
            title: Title substring to search for.
            case_insensitive: When ``True``, perform case-insensitive matching.

        Returns:
            bool: True when the handle was updated.
        """
        windows = self.get_windows_with_hwnds()
        found = self._find_by_title(title, windows, case_insensitive=case_insensitive)
        return self._attach_if_found(found)

    def set_inner_hwnd_by_title(self: Self, class_name: str) -> bool:
        """Update ``hwnd`` to the first child window whose class matches.

        Args:
            class_name: Substring of the child window class to match (case-insensitive).

        Returns:
            bool: True when the handle was updated.

        Raises:
            InvalidHandleError: If no window handle is attached when enumerating child windows.
        """
        child_windows = self.get_child_windows()
        found = self._find_by_title(class_name, child_windows, case_insensitive=True)
        return self._attach_if_found(found)

    #
    # Capture pipeline
    #
    @check_valid_hwnd
    def capture_frame(
        self: Self,
        region: Rect | None = None,
        *,
        persist: bool = True,
        use_cache: bool = False,
    ) -> NDArrayUint8:
        """Capture the current window region into a contiguous BGR image.

        Args:
            region: Optional capture region specified as ``(x, y, width, height)`` in window coordinates.
                Defaults to the full window when omitted.
            persist: When ``True``, store the frame in ``last_frame`` for reuse.
            use_cache: When ``True``, reuse cached window bounds when clamping the capture region.

        Returns:
            NDArrayUint8: Captured frame in BGR channel order.

        Raises:
            InvalidHandleError: If no window handle is attached.
            ValueError: If ``region`` falls completely outside the window bounds or has non-positive dimensions.
        """
        bounds = self._get_window_bounds_cached(use_cache=use_cache)
        capture_rect = self._normalize_region(region, bounds)
        frame = self._bitblt_to_array(self.hwnd, capture_rect)
        if persist:
            self._last_frame = frame
        return frame

    def _attach_if_found(self: Self, hwnd: WindowHandle | None) -> bool:
        """Attach to ``hwnd`` when not ``None`` and return whether the handle was updated."""
        if hwnd is None:
            return False
        self.attach(hwnd)
        return True

    @staticmethod
    def _fetch_window_bounds(hwnd: WindowHandle) -> Bounds:
        """Fetch absolute window bounds from the OS."""
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        return left, top, right, bottom

    @staticmethod
    def _find_by_title(title: str, windows: Sequence[WindowEntry], *, case_insensitive: bool) -> WindowHandle | None:
        """Return a window handle whose title/class contains ``title``.

        The scan preserves the ordering produced by the underlying Win32 enumeration call.
        """
        if case_insensitive:
            needle = title.casefold()
            for hwnd, window_title in windows:
                if needle in window_title.casefold():
                    return hwnd
            return None

        for hwnd, window_title in windows:
            if title in window_title:
                return hwnd
        return None

    @staticmethod
    def _normalize_region(region: Rect | None, bounds: Bounds) -> Rect:
        """Clamp the requested region to the window bounds.

        Args:
            region: Desired capture rectangle relative to the window.
            bounds: Absolute window bounds.

        Returns:
            Rect: Region clamped to the window.

        Raises:
            ValueError: If the region lies completely outside the window or has non-positive dimensions.
        """
        if region is None:
            width, height = WindowCapture._bounds_to_size(bounds)
            return 0, 0, width, height

        x, y, width, height = region
        window_width, window_height = WindowCapture._bounds_to_size(bounds)

        if width <= 0 or height <= 0:
            msg = "Capture region must have positive width and height."
            raise ValueError(msg)

        left = max(x, 0)
        top = max(y, 0)
        right = min(x + width, window_width)
        bottom = min(y + height, window_height)

        clamped_width = right - left
        clamped_height = bottom - top

        if clamped_width <= 0 or clamped_height <= 0:
            msg = "Capture region lies outside the window bounds."
            raise ValueError(msg)

        return left, top, clamped_width, clamped_height

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
        mem_dc = None
        bmp_dc = None
        bitmap = None

        try:
            mem_dc = win32ui.CreateDCFromHandle(window_dc)
            bmp_dc = mem_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mem_dc, width, height)
            bmp_dc.SelectObject(bitmap)
            bmp_dc.BitBlt((0, 0), (width, height), mem_dc, (x, y), win32con.SRCCOPY)

            bitmap_bits = bitmap.GetBitmapBits(True)
            bgra_frame = np.frombuffer(bitmap_bits, dtype=np.uint8).reshape((height, width, _BGRA_CHANNELS))
            return np.ascontiguousarray(bgra_frame[..., :_BGR_CHANNELS])
        finally:
            if bmp_dc is not None:
                bmp_dc.DeleteDC()
            if mem_dc is not None:
                mem_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, window_dc)
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())

    @staticmethod
    def _window_enumeration_handler(hwnd: WindowHandle, top_windows: list[WindowEntry]) -> None:
        """Collect visible top-level windows during enumeration."""
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd)
        if title:
            top_windows.append((hwnd, title))

    @staticmethod
    def _child_window_enumeration_handler(hwnd: WindowHandle, child_windows: list[ChildWindowEntry]) -> None:
        """Collect child windows encountered during enumeration."""
        class_name = win32gui.GetClassName(hwnd)
        if class_name:
            child_windows.append((hwnd, class_name))
