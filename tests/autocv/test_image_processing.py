import numpy as np
import pytest

from autocv.core.image_processing import filter_colors


def test_filter_colors_single_colour_mask():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[1, 1] = (10, 20, 30)

    mask = filter_colors(image, (30, 20, 10), tolerance=0)

    assert mask.dtype == np.uint8
    assert mask[1, 1] == 255
    assert mask.sum() == 255


def test_filter_colors_multiple_colours_returns_filtered_image():
    image = np.array(
        [
            [[0, 0, 255], [0, 255, 0]],
            [[255, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    rgb_targets = [(255, 0, 0), (0, 0, 255)]

    filtered = filter_colors(image, rgb_targets, tolerance=10, keep_original_colors=True)

    assert np.array_equal(filtered[0, 0], image[0, 0])
    assert np.array_equal(filtered[1, 0], image[1, 0])
    assert filtered[0, 1].sum() == 0
    assert filtered[1, 1].sum() == 0


def test_filter_colors_negative_tolerance_raises():
    with pytest.raises(ValueError):
        filter_colors(np.zeros((1, 1, 3), dtype=np.uint8), (0, 0, 0), tolerance=-1)


def test_filter_colors_invalid_colour_shape():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        filter_colors(image, [(255, 0)])
