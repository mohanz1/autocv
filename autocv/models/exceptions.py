"""Module to define custom exceptions for the AutoCV project.

This module provides custom exception classes to indicate specific errors that can occur in the application,
such as invalid handles, images, or lengths.
"""

from __future__ import annotations

__all__ = (
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)

from typing_extensions import Self


class InvalidHandleError(Exception):
    """Exception raised when an invalid window handle is encountered.

    Attributes:
        hwnd (int): The invalid handle that caused the exception.
    """

    def __init__(self: Self, hwnd: int) -> None:
        """Initialize the exception with the invalid handle.

        Args:
            hwnd (int): The invalid handle value.
        """
        super().__init__(f"Invalid handle: {hwnd}. Please set handle before calling this method.")


class InvalidImageError(Exception):
    """Exception raised when an invalid image is encountered.

    This exception typically indicates that an image has not been captured or refreshed properly.
    """

    def __init__(self: Self) -> None:
        """Initialize the exception for an invalid image."""
        super().__init__("Invalid image. Please call refresh() before calling this method.")


class InvalidLengthError(Exception):
    """Exception raised when the length of an input does not meet the expected value.

    Attributes:
        expected (int): The expected length.
        received (int): The length that was received.
    """

    def __init__(self: Self, expected: int, received: int) -> None:
        """Initialize the exception with the expected and received lengths.

        Args:
            expected (int): The expected length.
            received (int): The actual length received.
        """
        super().__init__(f"Expected length {expected}. Instead received length {received}.")
