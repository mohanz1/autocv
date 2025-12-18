import numpy as np
import pytest

from autocv.core.image_processing import filter_colors


def test_filter_colors_returns_mask_for_single_color():
    # 2x2 BGR image: black, red, green, blue
    image = np.array(
        [
            [[0, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )

    mask = filter_colors(image, (255, 0, 0))
    assert mask.shape == (2, 2)
    assert mask.dtype == np.uint8
    assert mask[0, 1] == 255
    assert mask[0, 0] == 0
    assert mask[1, 0] == 0
    assert mask[1, 1] == 0


def test_filter_colors_supports_multiple_colors_and_keep_original_colors():
    # 2x2 BGR image: black, red, green, blue
    image = np.array(
        [
            [[0, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )

    filtered = filter_colors(image, [(255, 0, 0), (0, 0, 255)], keep_original_colors=True)
    assert filtered.shape == image.shape
    assert np.array_equal(filtered[0, 1], image[0, 1])  # red kept
    assert np.array_equal(filtered[1, 1], image[1, 1])  # blue kept
    assert np.array_equal(filtered[0, 0], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(filtered[1, 0], np.zeros(3, dtype=np.uint8))


def test_filter_colors_rejects_negative_tolerance():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        filter_colors(image, (0, 0, 0), tolerance=-1)


def test_filter_colors_rejects_empty_color_sequence():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        filter_colors(image, [])


def test_filter_colors_rejects_invalid_color_shape():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        filter_colors(image, [(1, 2)])
