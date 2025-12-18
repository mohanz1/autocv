import numpy as np
import pytest
from autocv.utils import geometry


def test_get_center_rect():
    assert geometry.get_center((0, 0, 10, 20)) == (5, 10)


def test_get_center_contour():
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])
    assert geometry.get_center(contour) == (1, 1)


def test_get_center_contour_flat_points():
    contour = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert geometry.get_center(contour) == (1, 1)


def test_get_center_contour_nested_points():
    contour = np.array([[[0, 0], [2, 0]], [[2, 2], [0, 2]]])
    assert geometry.get_center(contour) == (1, 1)


def test_get_center_invalid_contour():
    contour = np.array([[[0, 0]], [[1, 1]]])  # m00 == 0
    with pytest.raises(ValueError):
        geometry.get_center(contour)


def test_get_center_empty_contour():
    with pytest.raises(ValueError):
        geometry.get_center(np.array([], dtype=np.int32))


def test_get_center_rejects_invalid_contour_shape():
    contour = np.zeros((1, 1, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        geometry.get_center(contour)


def test_get_random_point_rect():
    point = geometry.get_random_point((10, 10, 5, 5))
    assert 10 <= point[0] < 15
    assert 10 <= point[1] < 15


def test_get_random_point_rect_invalid():
    with pytest.raises(ValueError):
        geometry.get_random_point((0, 0, 0, 10))


def test_get_random_point_contour_accepts_boundary(monkeypatch):
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])

    monkeypatch.setattr(geometry, "_MAX_RANDOM_POINT_ATTEMPTS", 1)
    monkeypatch.setattr(geometry, "_RNG", np.random.default_rng(0))
    monkeypatch.setattr(geometry.cv2, "pointPolygonTest", lambda *args, **kwargs: 0)

    point = geometry.get_random_point(contour)
    assert isinstance(point, tuple)
    assert len(point) == 2


def test_get_random_point_contour_raises_when_never_inside(monkeypatch):
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])

    monkeypatch.setattr(geometry, "_MAX_RANDOM_POINT_ATTEMPTS", 1)
    monkeypatch.setattr(geometry, "_RNG", np.random.default_rng(0))
    monkeypatch.setattr(geometry.cv2, "pointPolygonTest", lambda *args, **kwargs: -1)

    with pytest.raises(ValueError):
        geometry.get_random_point(contour)


def test_get_random_point_contour_rejects_zero_area_bounding_rect(monkeypatch):
    contour = np.array([[[0, 0]], [[2, 0]], [[2, 2]], [[0, 2]]])
    monkeypatch.setattr(geometry.cv2, "boundingRect", lambda *_: (0, 0, 0, 0))

    with pytest.raises(ValueError):
        geometry.get_random_point(contour)


def test_sort_shapes_top_bottom():
    shapes = [(0, 10, 1, 1), (0, 5, 1, 1), (0, 15, 1, 1)]
    sorted_shapes = geometry.sort_shapes((100, 100), shapes, "top_bottom")
    assert sorted_shapes == [(0, 5, 1, 1), (0, 10, 1, 1), (0, 15, 1, 1)]


def test_sort_shapes_bottom_top():
    shapes = [(0, 10, 1, 1), (0, 5, 1, 1), (0, 15, 1, 1)]
    sorted_shapes = geometry.sort_shapes((100, 100), shapes, "bottom_top")
    assert sorted_shapes == [(0, 15, 1, 1), (0, 10, 1, 1), (0, 5, 1, 1)]


def test_sort_shapes_left_right_and_right_left():
    shapes = [(10, 0, 1, 1), (0, 0, 1, 1), (20, 0, 1, 1)]
    assert geometry.sort_shapes((100, 100), shapes, "left_right") == [(0, 0, 1, 1), (10, 0, 1, 1), (20, 0, 1, 1)]
    assert geometry.sort_shapes((100, 100), shapes, "right_left") == [(20, 0, 1, 1), (10, 0, 1, 1), (0, 0, 1, 1)]


def test_sort_shapes_inner_outer_and_outer_inner():
    shapes = [(0, 0, 1, 1), (40, 50, 1, 1), (90, 90, 1, 1)]
    assert geometry.sort_shapes((100, 100), shapes, "inner_outer") == [(40, 50, 1, 1), (90, 90, 1, 1), (0, 0, 1, 1)]
    assert geometry.sort_shapes((100, 100), shapes, "outer_inner") == [(0, 0, 1, 1), (90, 90, 1, 1), (40, 50, 1, 1)]


def test_sort_shapes_unknown_returns_original_list():
    shapes = [(0, 0, 1, 1)]
    result = geometry.sort_shapes((100, 100), shapes, "unknown")
    assert result is shapes
