from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autocv.image_filter import FilterEngine, ImageFilter, _apply_channel_offset
from autocv.models import FilterSettings


@pytest.fixture
def dummy_image():
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def test_apply_channel_offset_saturates_high_and_low():
    channel = np.array([[250, 5]], dtype=np.uint8)

    raised = _apply_channel_offset(channel, 10)
    lowered = _apply_channel_offset(channel, -10)

    assert np.array_equal(raised, np.array([[255, 15]], dtype=np.uint8))
    assert np.array_equal(lowered, np.array([[240, 0]], dtype=np.uint8))


@patch("autocv.image_filter.cv.getTrackbarPos", return_value=0)
@patch("autocv.image_filter.cv.namedWindow")
@patch("autocv.image_filter.cv.createTrackbar")
@patch("autocv.image_filter.cv.imshow")
@patch("autocv.image_filter.cv.getWindowProperty", return_value=-1)
@patch("autocv.image_filter.cv.waitKey", return_value=27)
@patch("autocv.image_filter.cv.destroyAllWindows")
def test_init_runs_without_error(
    mock_destroy, mock_waitkey, mock_prop, mock_imshow, mock_trackbar, mock_named, mock_gettrack, dummy_image
):
    ImageFilter(dummy_image)  # should complete immediately


@patch("autocv.image_filter.ImageFilter.get_filtered_image", return_value=np.zeros((1, 1, 3), dtype=np.uint8))
@patch("autocv.image_filter.cv.getTrackbarPos", return_value=0)
@patch("autocv.image_filter.cv.namedWindow")
@patch("autocv.image_filter.cv.createTrackbar")
@patch("autocv.image_filter.cv.imshow")
@patch("autocv.image_filter.cv.getWindowProperty", side_effect=[0, 0])
@patch("autocv.image_filter.cv.waitKey", side_effect=[0, 27])
@patch("autocv.image_filter.cv.destroyAllWindows")
def test_init_event_loop_updates_until_escape(
    mock_destroy,
    mock_waitkey,
    mock_prop,
    mock_imshow,
    mock_trackbar,
    mock_named,
    mock_gettrack,
    mock_get_filtered,
    dummy_image,
):
    ImageFilter(dummy_image)
    assert mock_waitkey.call_count == 2
    assert mock_imshow.call_count >= 2


@patch("autocv.image_filter.cv.getTrackbarPos", return_value=5)
def test_update_filter_settings_sets_all_fields(mock_get, dummy_image):
    f = ImageFilter.__new__(ImageFilter)
    f.filter_settings = FilterSettings()
    f.image = dummy_image
    f.hsv_image = np.zeros_like(dummy_image)

    f.update_filter_settings()

    for attr in [
        "h_min",
        "h_max",
        "s_min",
        "s_max",
        "v_min",
        "v_max",
        "s_add",
        "s_subtract",
        "v_add",
        "v_subtract",
        "canny_threshold1",
        "canny_threshold2",
        "erode_kernel_size",
        "dilate_kernel_size",
    ]:
        assert getattr(f.filter_settings, attr) == 5


@patch("autocv.image_filter.ImageFilter.update_filter_settings")
def test_get_filtered_image_returns_array(mock_update, dummy_image):
    f = ImageFilter.__new__(ImageFilter)
    f.image = dummy_image
    f.hsv_image = np.full_like(dummy_image, (60, 255, 255))  # green HSV
    f.filter_settings = FilterSettings(
        h_min=0,
        h_max=179,
        s_min=0,
        s_max=255,
        v_min=0,
        v_max=255,
        s_add=10,
        s_subtract=5,
        v_add=10,
        v_subtract=5,
        canny_threshold1=50,
        canny_threshold2=150,
        erode_kernel_size=3,
        dilate_kernel_size=3,
    )

    result = f.get_filtered_image()
    assert isinstance(result, np.ndarray)
    assert result.ndim in (2, 3)


def test_ensure_engine_raises_when_missing_image():
    f = ImageFilter.__new__(ImageFilter)
    with pytest.raises(ValueError, match="requires an image"):
        f._ensure_engine()


def test_filter_engine_rejects_non_color_image():
    with pytest.raises(ValueError, match="requires a BGR image"):
        FilterEngine(np.zeros((10, 10), dtype=np.uint8))
