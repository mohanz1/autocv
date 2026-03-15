"""Geometry helpers for rectangles and OpenCV contours.

The public functions in this module accept either rectangle tuples in
``(x, y, w, h)`` form or OpenCV contour arrays and provide convenient operations
such as computing centers, sampling random points, and sorting shapes.
"""

from __future__ import annotations

__all__ = ("get_center", "get_random_point", "sort_shapes")

from typing import TYPE_CHECKING, Final, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt

Point: TypeAlias = tuple[int, int]
Rect: TypeAlias = tuple[int, int, int, int]  # (x, y, w, h)
Contour: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = Rect | npt.NDArray[np.generic]
SortBy: TypeAlias = Literal[
    "inner_outer",
    "outer_inner",
    "left_right",
    "right_left",
    "top_bottom",
    "bottom_top",
]

_RNG: Final[np.random.Generator] = np.random.default_rng()
_MAX_RANDOM_POINT_ATTEMPTS: Final[int] = 10_000
_CONTOUR_NDIM_NESTED: Final[int] = 3
_CONTOUR_NDIM_FLAT: Final[int] = 2
_CONTOUR_SINGLETON_DIM: Final[int] = 1
_CONTOUR_COORD_DIM: Final[int] = 2


def _normalize_contour(contour: npt.NDArray[np.generic]) -> Contour:
    """Return a contiguous ``int32`` contour in the OpenCV ``(N, 1, 2)`` layout.

    Args:
        contour: Contour array, typically shaped ``(N, 1, 2)`` or ``(N, 2)``.

    Returns:
        Normalised contour array shaped ``(N, 1, 2)`` with dtype ``int32``.

    Raises:
        ValueError: If the input contour is empty or has an unsupported shape.
    """
    if contour.size == 0:
        msg = "Contour is empty."
        raise ValueError(msg)

    contour_i32 = np.asarray(contour, dtype=np.int32)
    if (
        contour_i32.ndim == _CONTOUR_NDIM_NESTED
        and contour_i32.shape[1] == _CONTOUR_SINGLETON_DIM
        and contour_i32.shape[2] == _CONTOUR_COORD_DIM
    ):
        return np.ascontiguousarray(contour_i32)
    if contour_i32.ndim == _CONTOUR_NDIM_FLAT and contour_i32.shape[1] == _CONTOUR_COORD_DIM:
        return np.ascontiguousarray(contour_i32.reshape(-1, 1, _CONTOUR_COORD_DIM))
    if contour_i32.ndim == _CONTOUR_NDIM_NESTED and contour_i32.shape[-1] == _CONTOUR_COORD_DIM:
        return np.ascontiguousarray(contour_i32.reshape(-1, 1, _CONTOUR_COORD_DIM))
    msg = "Contour must have shape (N, 2) or (N, 1, 2)."
    raise ValueError(msg)


def get_center(shape: Shape) -> Point:
    """Return the center (x, y) of a rectangle or contour.

    Args:
        shape: Rectangle as ``(x, y, w, h)`` or contour array shaped ``(N, 2)`` or ``(N, 1, 2)``.

    Returns:
        Center coordinates.

    Raises:
        ValueError: If the contour area is zero.
    """
    if isinstance(shape, np.ndarray):
        contour = _normalize_contour(shape)
        moments = cv2.moments(contour)
        m00 = float(moments["m00"])
        if m00 == 0.0:
            msg = "Contour area is zero; cannot compute center."
            raise ValueError(msg)
        cx = int(float(moments["m10"]) / m00)
        cy = int(float(moments["m01"]) / m00)
        return cx, cy

    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: Shape) -> Point:
    """Return a random point (x, y) within a rectangle or contour.

    Args:
        shape: Rectangle as ``(x, y, w, h)`` or contour array shaped ``(N, 2)`` or ``(N, 1, 2)``.

    Returns:
        Random point inside the shape.

    Raises:
        ValueError: If the contour is empty or invalid, or the rectangle is malformed.
    """
    if isinstance(shape, np.ndarray):
        contour = _normalize_contour(shape)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            msg = "Contour bounding rectangle has zero area."
            raise ValueError(msg)

        for include_boundary in (False, True):
            for _ in range(_MAX_RANDOM_POINT_ATTEMPTS):
                rx = int(_RNG.integers(x, x + w))
                ry = int(_RNG.integers(y, y + h))
                test_value = cv2.pointPolygonTest(contour, (rx, ry), measureDist=False)
                if test_value > 0 or (include_boundary and test_value == 0):
                    return rx, ry

        msg = "Failed to sample a point inside the contour."
        raise ValueError(msg)

    x, y, w, h = shape
    if w <= 0 or h <= 0:
        msg = "Rectangle width and height must be positive."
        raise ValueError(msg)
    rand_x = int(_RNG.integers(x, x + w))
    rand_y = int(_RNG.integers(y, y + h))
    return rand_x, rand_y


def sort_shapes(
    window_size: tuple[int, int],
    shapes: list[Shape],
    sort_by: SortBy | str,
) -> list[Shape]:
    """Sort shapes based on the specified criterion.

    Shapes can be rectangles or contours and are ordered using their centers.

    Args:
        window_size: Window size as ``(width, height)`` for computing the window center.
        shapes: List of rectangles ``(x, y, w, h)`` or contour arrays.
        sort_by: Sorting criterion: ``inner_outer``, ``outer_inner``, ``left_right``,
            ``right_left``, ``top_bottom``, or ``bottom_top``.

    Returns:
        Sorted shapes. When ``sort_by`` is unrecognised, returns ``shapes`` unchanged.

    Raises:
        ValueError: If any shape is unrecognized or has zero area when calculating the center.
    """
    window_width, window_height = window_size
    window_center_x = window_width // 2
    window_center_y = window_height // 2

    def distance_from_window_center(s: Shape) -> float:
        """Compute squared distance from a shape's center to the window center."""
        center_x, center_y = get_center(s)
        dx = center_x - window_center_x
        dy = center_y - window_center_y
        return float(dx * dx + dy * dy)

    def center_x(shape: Shape) -> float:
        """Return the shape center x-coordinate as a float."""
        return float(get_center(shape)[0])

    def center_y(shape: Shape) -> float:
        """Return the shape center y-coordinate as a float."""
        return float(get_center(shape)[1])

    key_func: Callable[[Shape], float]
    reverse_sort: bool
    match sort_by:
        case "inner_outer":
            key_func = distance_from_window_center
            reverse_sort = False
        case "outer_inner":
            key_func = distance_from_window_center
            reverse_sort = True
        case "left_right":
            key_func = center_x
            reverse_sort = False
        case "right_left":
            key_func = center_x
            reverse_sort = True
        case "top_bottom":
            key_func = center_y
            reverse_sort = False
        case "bottom_top":
            key_func = center_y
            reverse_sort = True
        case _:
            return shapes

    return sorted(shapes, key=key_func, reverse=reverse_sort)
