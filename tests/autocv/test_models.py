import pytest

from autocv.models import (
    FilterSettings,
    InvalidHandleError,
    InvalidImageError,
    InvalidLengthError,
)


def test_filter_settings_defaults():
    settings = FilterSettings()
    assert settings.h_max == 179
    assert settings.s_max == 255
    assert settings.v_max == 255
    assert not hasattr(settings, "__dict__")


def test_filter_settings_custom_values():
    settings = FilterSettings(h_min=10, h_max=20, s_add=5)
    assert settings.h_min == 10
    assert settings.h_max == 20
    assert settings.s_add == 5


def test_invalid_handle_error_exposes_value():
    error = InvalidHandleError(123)
    assert error.hwnd == 123
    assert "123" in str(error)


def test_invalid_image_error_message():
    error = InvalidImageError()
    assert "refresh" in str(error).lower()


def test_invalid_length_error_fields():
    error = InvalidLengthError(expected=2, received=1)
    assert error.expected == 2
    assert error.received == 1
    assert "2" in str(error) and "1" in str(error)


def test_models_all_exports():
    from autocv import models

    assert set(models.__all__) == {
        "FilterSettings",
        "InvalidHandleError",
        "InvalidImageError",
        "InvalidLengthError",
    }


def test_invalid_handle_error_is_value_error():
    assert isinstance(InvalidHandleError(5), ValueError)


def test_invalid_image_error_is_runtime_error():
    assert isinstance(InvalidImageError(), RuntimeError)


def test_invalid_length_error_is_value_error():
    assert isinstance(InvalidLengthError(1, 0), ValueError)
