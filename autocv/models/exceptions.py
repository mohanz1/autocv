"""Domain-specific exception types used across AutoCV."""

from __future__ import annotations

__all__ = (
    "InvalidHandleError",
    "InvalidImageError",
    "InvalidLengthError",
)


class InvalidHandleError(ValueError):
    """Raised when a window handle is missing or invalid."""

    __slots__ = ("hwnd",)

    def __init__(self, hwnd: int) -> None:
        """Store the offending handle for later inspection."""
        self.hwnd = hwnd
        super().__init__(f"Invalid window handle: {hwnd}.")


class InvalidImageError(RuntimeError):
    """Raised when an OpenCV image buffer has not been initialised."""

    __slots__ = ()

    def __init__(self) -> None:
        """Explain that callers must refresh the buffer before use."""
        super().__init__("Invalid image buffer; call refresh() before accessing image data.")


class InvalidLengthError(ValueError):
    """Raised when an iterable does not contain the expected number of items."""

    __slots__ = ("expected", "received")

    def __init__(self, expected: int, received: int) -> None:
        """Record expected versus received item counts."""
        self.expected = expected
        self.received = received
        super().__init__(f"Expected {expected} items but received {received}.")
