"""Module to define custom exceptions for the AutoCV project.

These exceptions are used to more clearly indicate specific errors that can occur in the application, such as invalid
handles, images, or lengths.
"""

from __future__ import annotations

__all__ = (
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)

from typing import Self


class InvalidHandleError(Exception):
    """Exception raised for errors in the input handle."""

    def __init__(self: Self, hwnd: int) -> None:
        """Initialize the exception with the invalid handle.

        Args:
            hwnd (int): The invalid handle that caused the exception.
        """
        super().__init__(f"Invalid handle: {hwnd}. Please set handle before calling this method.")


class InvalidImageError(Exception):
    """Exception raised for errors in the input image."""

    def __init__(self: Self) -> None:
        """Initialize the exception indicating an issue with the image."""
        super().__init__("Invalid image. Please call refresh() before calling this method.")


class InvalidLengthError(Exception):
    """Exception raised for errors in the input length."""

    def __init__(self: Self, expected: int, received: int) -> None:
        """Initialize the exception with expected and received lengths.

        Args:
            expected (int): The expected length.
            received (int): The length that was received.
        """
        super().__init__(f"Expected length {expected}. Instead received length {received}.")
