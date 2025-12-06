"""Geometry helpers for rectangles and contours."""

from __future__ import annotations

__all__ = ("get_center", "get_random_point", "sort_shapes")

from typing import TYPE_CHECKING, Literal, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

THREE = 3
Rect: TypeAlias = tuple[int, int, int, int]  # x, y, w, h
Contour: TypeAlias = npt.NDArray[np.uintp]
Shape: TypeAlias = Rect | Contour

_rng = np.random.default_rng()


def get_center(shape: Shape) -> tuple[int, int]:
    """Return the center (x, y) of a rectangle or contour.

    Args:
        shape: Rectangle as ``(x, y, w, h)`` or contour array shaped ``(N, 2)`` or ``(N, 1, 2)``.

    Returns:
        tuple[int, int]: Center coordinates.

    Raises:
        ValueError: If the contour area is zero.
    """
    if isinstance(shape, np.ndarray):
        # Reshape contour to (N, 2) if necessary (e.g., when shape is (N, 1, 2))
        contour = shape.reshape(-1, 2) if shape.ndim == THREE and shape.shape[1] == 1 else shape

        # Calculate centroid using image moments
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            raise ValueError
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: Shape) -> tuple[int, int]:
    """Return a random point (x, y) within a rectangle or contour.

    Args:
        shape: Rectangle as ``(x, y, w, h)`` or contour array shaped ``(N, 2)`` or ``(N, 1, 2)``.

    Returns:
        tuple[int, int]: Random point inside the shape.

    Raises:
        ValueError: If the contour is empty or invalid, or the rectangle is malformed.
    """
    if isinstance(shape, np.ndarray):
        # Convert contour to int32 and get its bounding rectangle
        int_shape = shape.astype(np.int32)
        x, y, w, h = cv2.boundingRect(int_shape)

        rng = np.random.default_rng()
        while True:
            # Generate a random candidate point within the bounding rectangle
            rx = rng.integers(x, x + w).astype(int)
            ry = rng.integers(y, y + h).astype(int)

            # Check if the point is inside the contour
            if cv2.pointPolygonTest(int_shape, (rx, ry), measureDist=False) > 0:
                return rx, ry

    x, y, w, h = shape
    rand_x = _rng.integers(x, x + w).astype(int)
    rand_y = _rng.integers(y, y + h).astype(int)
    return rand_x, rand_y


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
    """Sort shapes based on the specified criterion.

    Shapes can be rectangles or contours and are ordered using their centers.

    Args:
        window_size: Window size as ``(width, height)`` for computing the window center.
        shapes: List of rectangles ``(x, y, w, h)`` or contour arrays.
        sort_by: Sorting criterion: ``inner_outer``, ``outer_inner``, ``left_right``,
            ``right_left``, ``top_bottom``, or ``bottom_top``.

    Returns:
        list[tuple[int, int, int, int] | npt.NDArray[np.uintp]]: Sorted shapes.

    Raises:
        ValueError: If any shape is unrecognized or has zero area when calculating the center.
    """
    window_center_x = window_size[0] // 2
    window_center_y = window_size[1] // 2

    def distance_from_window_center(s: Shape) -> float:
        """Compute squared distance from a shape's center to the window center."""
        center_x, center_y = get_center(s)
        return (center_x - window_center_x) ** 2 + (center_y - window_center_y) ** 2

    strategies: dict[str, tuple[Callable[[Shape], float], bool]] = {
        "inner_outer": (distance_from_window_center, False),
        "outer_inner": (distance_from_window_center, True),
        "left_right": (lambda s: float(get_center(s)[0]), False),
        "right_left": (lambda s: float(get_center(s)[0]), True),
        "top_bottom": (lambda s: float(get_center(s)[1]), False),
        "bottom_top": (lambda s: float(get_center(s)[1]), True),
    }

    if sort_by not in strategies:
        return shapes

    key_func, reverse_sort = strategies[sort_by]
    return sorted(shapes, key=key_func, reverse=reverse_sort)
