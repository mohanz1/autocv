import pytest
import numpy as np
from autocv.models import InvalidHandleError, InvalidImageError
from autocv.core import decorators


# Dummy classes for decorator tests
class DummyWindowCapture:
    def __init__(self, hwnd):
        self.hwnd = hwnd

    @decorators.check_valid_hwnd
    def validated_method(self):
        return True


class DummyEnsureHandle:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.ensure_calls = 0

    def _ensure_hwnd(self):
        self.ensure_calls += 1
        return self.hwnd

    @decorators.check_valid_hwnd
    def validated_method(self):
        return True


class DummyVision:
    def __init__(self, image):
        self.opencv_image = image

    @decorators.check_valid_image
    def validated_method(self):
        return True


def test_check_valid_hwnd_valid():
    assert DummyWindowCapture(1).validated_method() is True


def test_check_valid_hwnd_invalid():
    with pytest.raises(InvalidHandleError):
        DummyWindowCapture(-1).validated_method()


def test_check_valid_hwnd_uses_ensure_hwnd():
    dummy = DummyEnsureHandle(123)
    assert dummy.validated_method() is True
    assert dummy.ensure_calls == 1


def test_check_valid_hwnd_rejects_non_int_handle():
    with pytest.raises(InvalidHandleError):
        DummyWindowCapture("not-an-int").validated_method()


def test_check_valid_image_valid():
    img = np.ones((5, 5, 3), dtype=np.uint8)
    assert DummyVision(img).validated_method() is True


def test_check_valid_image_invalid():
    with pytest.raises(InvalidImageError):
        DummyVision(np.array([], dtype=np.uint8)).validated_method()
