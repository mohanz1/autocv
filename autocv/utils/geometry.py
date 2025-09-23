"""Geometry helpers for bounding boxes and contours."""

from __future__ import annotations

__all__ = ("get_center", "get_random_point", "sort_shapes")

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

RNG = np.random.default_rng()
_ATTEMPT_LIMIT = 1_024
_RECT_SIZE = 4
_CONTOUR_DIMS = 3
_CONTOUR_SINGLE = 1

Shape = tuple[int, int, int, int] | npt.NDArray[np.uintp]


def _as_contour(shape: npt.NDArray[np.uintp]) -> npt.NDArray[np.int32]:
    """Return an ``(N, 2)`` contour array."""
    if shape.ndim == _CONTOUR_DIMS and shape.shape[1] == _CONTOUR_SINGLE:
        shape = shape.reshape(-1, 2)
    return shape.astype(np.int32, copy=False)


def get_center(shape: Shape) -> tuple[int, int]:
    """Return the centre of a rectangle or contour."""
    if isinstance(shape, np.ndarray):
        contour = _as_contour(shape)
        moments = cv2.moments(contour)
        if not moments["m00"]:
            msg = "Contour area is zero; cannot compute centroid."
            raise ValueError(msg)
        return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

    if len(shape) != _RECT_SIZE:
        msg = "Rectangle must be a (x, y, w, h) tuple."
        raise ValueError(msg)
    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: Shape) -> tuple[int, int]:
    """Return a random point inside ``shape``."""
    if isinstance(shape, np.ndarray):
        contour = _as_contour(shape)
        x, y, w, h = cv2.boundingRect(contour)
        for _ in range(_ATTEMPT_LIMIT):
            rx = int(RNG.integers(x, x + w))
            ry = int(RNG.integers(y, y + h))
            if cv2.pointPolygonTest(contour, (rx, ry), measureDist=False) >= 0:
                return rx, ry
        msg = "Failed to sample a point inside the contour."
        raise ValueError(msg)

    x, y, w, h = shape
    return int(RNG.integers(x, x + w)), int(RNG.integers(y, y + h))


def sort_shapes(
    window_size: tuple[int, int],
    shapes: list[Shape],
    sort_by: Literal[
        "inner_outer",
        "outer_inner",
        "left_right",
        "right_left",
        "top_bottom",
        "bottom_top",
    ],
) -> list[Shape]:
    """Return shapes ordered according to ``sort_by``."""

    def window_distance(shape: Shape) -> float:
        cx, cy = get_center(shape)
        wx, wy = window_size[0] // 2, window_size[1] // 2
        return float((cx - wx) ** 2 + (cy - wy) ** 2)

    key_map: dict[str, tuple[Callable[[Shape], float], bool]] = {
        "inner_outer": (window_distance, False),
        "outer_inner": (window_distance, True),
        "left_right": (lambda s: get_center(s)[0], False),
        "right_left": (lambda s: get_center(s)[0], True),
        "top_bottom": (lambda s: get_center(s)[1], False),
        "bottom_top": (lambda s: get_center(s)[1], True),
    }

    key, reverse = key_map.get(sort_by, (None, False))
    if key is None:
        return shapes
    return sorted(shapes, key=key, reverse=reverse)
