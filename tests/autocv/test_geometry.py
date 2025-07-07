import numpy as np
import pytest
from autocv.utils import geometry


def test_get_center_rect():
    assert geometry.get_center((0, 0, 10, 20)) == (5, 10)


def test_get_center_contour():
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])
    assert geometry.get_center(contour) == (1, 1)


def test_get_center_invalid_contour():
    contour = np.array([[[0, 0]], [[1, 1]]])  # m00 == 0
    with pytest.raises(ValueError):
        geometry.get_center(contour)


def test_get_random_point_rect():
    point = geometry.get_random_point((10, 10, 5, 5))
    assert 10 <= point[0] < 15
    assert 10 <= point[1] < 15


def test_sort_shapes_top_bottom():
    shapes = [(0, 10, 1, 1), (0, 5, 1, 1), (0, 15, 1, 1)]
    sorted_shapes = geometry.sort_shapes((100, 100), shapes, "top_bottom")
    assert sorted_shapes == [(0, 5, 1, 1), (0, 10, 1, 1), (0, 15, 1, 1)]
