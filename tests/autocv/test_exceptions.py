import pytest

from autocv.models import InvalidHandleError, InvalidImageError, InvalidLengthError


def test_invalid_handle_error_exposes_handle_and_message():
    err = InvalidHandleError(123)
    assert err.hwnd == 123
    assert "Invalid handle" in str(err)
    assert "123" in str(err)


def test_invalid_image_error_has_message():
    err = InvalidImageError()
    assert "Invalid image" in str(err)


def test_invalid_length_error_exposes_lengths_and_message():
    err = InvalidLengthError(expected=3, received=5)
    assert err.expected == 3
    assert err.received == 5
    assert "Expected length 3" in str(err)
    assert "received length 5" in str(err)
