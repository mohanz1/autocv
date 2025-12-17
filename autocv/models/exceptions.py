"""Exception classes used across AutoCV.

These lightweight subclasses communicate invalid state encountered while
validating window handles, image buffers, or sequence lengths.
"""

from __future__ import annotations

__all__ = (
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)

from typing_extensions import Self


class InvalidHandleError(Exception):
    """Raised when a window handle is missing or otherwise unusable.

    Attributes:
        hwnd: Handle value that failed validation.
    """

    def __init__(self: Self, hwnd: int) -> None:
        """Initialise the error with the offending handle.

        Args:
            hwnd: Handle value that triggered the error.
        """
        self.hwnd: int = hwnd
        super().__init__(f"Invalid handle: {hwnd}. Please set handle before calling this method.")


class InvalidImageError(Exception):
    """Raised when an OpenCV image buffer is empty or unset."""

    def __init__(self: Self) -> None:
        """Initialise the error for a missing or empty image buffer."""
        super().__init__("Invalid image. Please call refresh() before calling this method.")


class InvalidLengthError(Exception):
    """Raised when iterable length mismatches the expected size.

    Attributes:
        expected: Target number of items required by the caller.
        received: Number of items supplied by the caller.
    """

    def __init__(self: Self, expected: int, received: int) -> None:
        """Initialise the error with the expected versus received lengths.

        Args:
            expected: Number of items the routine requires.
            received: Number of items provided.
        """
        self.expected: int = expected
        self.received: int = received
        super().__init__(f"Expected length {expected}. Instead received length {received}.")
